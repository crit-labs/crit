"""The smoke tests: small checks of one quality component each, run against a served engine in seconds.

Each test compares two ways the engine computes the same thing (a prefix against the whole prompt, decode against
prefill, a reply against its score, a request against itself after another one), or checks a property any working
model has (finite, ranked log-probabilities; a low loss on plain prose; copying a repeated text). None needs a
reference model, so they run on any engine crit serves, and each failure names the component that broke.
"""
import math
import threading
import time

from .client import CONCURRENT, LOGPROBS, PLAIN, SCORE, TOKENIZE, Unsupported
from .compare import DECODE, TOLERANCES, compare, is_choice, nll, same_tokens
from .texts import PROMPTS, PROSE, recall_words

TESTS = []
# Prompt lengths for the prefix test, at the edges of the tiles, chunks and buckets engines often use.
PREFIXES = (17, 63, 64, 65, 127, 128, 129, 255, 256, 257, 511, 512, 513)
# The same past 1k tokens, where engines switch to sparse or windowed attention, other chunking or another bucket:
# one length just past each power of two and one between, so a few prefills reach every range up to `--long`.
LONG_PREFIXES = (1025, 2049, 3001, 4097, 6001, 8193)
# A long prefix is compared on its last rows: a fault past a threshold changes the rows there, and a mask that lets
# a row see later tokens shows only in a prefix's last rows, which the rows before them would dilute.
WINDOW = 64


def test(*needs):
    """Register a test that needs these capabilities."""
    def register(function):
        TESTS.append((function.__name__, needs, function))
        return function
    return register


class Context:
    def __init__(self, client, profile="loose", long=4096):
        self.client, self.profile = client, profile
        self.tolerance, self.decode_tolerance = TOLERANCES[profile], DECODE[profile]
        self.capabilities = client.capabilities()
        context = client.context() or long + 64
        self.long = max(256, min(long, context - 64))
        self._cache = {}

    def once(self, key, make):
        if key not in self._cache:
            self._cache[key] = make()
        return self._cache[key]

    def prose(self):
        return self.once("prose", lambda: self.client.tokenize(PROSE))

    def prose_rows(self):
        return self.once("prose_rows", lambda: self.client.score(self.prose())[0])

    def recall_ids(self, length):
        """(ids, offset) of `length` tokens: the special tokens, then a random word sequence written twice; offset is
        the row that predicts the second copy's first token."""
        special = self.once("special", lambda: self.client.tokenize("", True))
        half = (length - len(special)) // 2
        words = self.client.tokenize(recall_words(half), False)[:half]
        return special + words + words, len(special) + half - 1

    def long_ids(self):
        """`long` tokens: the prose, then random common words. Unlike recall's text no span repeats, so every row
        is uncertain enough to move when a path computes it wrong (a copied span reads the same on any path)."""
        def make():
            ids = self.prose()
            words = self.client.tokenize(recall_words(self.long, seed=11), False)
            return (ids + words)[:self.long]
        return self.once("long", make)

    def long_rows(self):
        return self.once("long_rows", lambda: self.client.score(self.long_ids())[0])


def check_rows(rows, label):
    """Every row finite and in range, its alternatives distinct and ranked, its token no likelier than its top-1,
    and the alternatives' probabilities summing to at most 1. Returns the first problem, or None."""
    for index, (token, logprob, alternatives) in enumerate(rows):
        values = [logprob] + [p for _, p in alternatives]
        if not all(isinstance(v, (int, float)) and math.isfinite(v) for v in values):
            return f"{label} row {index}: a log-probability that is not a finite number ({values[:3]})"
        if any(v > 1e-4 or v < -9999 for v in values):
            return f"{label} row {index}: a log-probability outside [-9999, 0] ({max(values)})"
        if not alternatives:
            continue
        if any(a[1] < b[1] for a, b in zip(alternatives, alternatives[1:])):
            return f"{label} row {index}: alternatives not ranked"
        if len({t for t, _ in alternatives}) != len(alternatives):
            return f"{label} row {index}: an alternative listed twice"
        if logprob > alternatives[0][1] + 1e-3:
            return f"{label} row {index}: the token ({logprob:.3f}) is likelier than the top-1 ({alternatives[0][1]:.3f})"
        listed = dict(alternatives)
        if token in listed and abs(listed[token] - logprob) > 1e-3:
            return f"{label} row {index}: the token's log-probability differs from its alternative's"
        total = sum(math.exp(p) for p in listed.values())
        if total > 1.001:
            return f"{label} row {index}: the alternatives' probabilities sum to {total:.4f}"
    return None


@test(SCORE, TOKENIZE)
def rows_valid(ctx):
    """Prefill rows are well formed: one per position after the first, finite, ranked, normalised."""
    ids, rows = ctx.prose(), ctx.prose_rows()
    if len(rows) != len(ids) - 1:
        return "failed", f"{len(rows)} rows for a prompt of {len(ids)} tokens, expected {len(ids) - 1}"
    problem = check_rows(rows, "prompt")
    return ("failed", problem) if problem else ("passed", f"{len(rows)} rows finite, ranked and normalised")


@test(SCORE, TOKENIZE)
def fluent(ctx):
    """Plain prose reads at a low perplexity: broken weights, dequantisation or norms read it at hundreds."""
    loss = nll(ctx.prose_rows())
    limit = 40.0
    state = "passed" if math.exp(loss) <= limit else "failed"
    return state, f"perplexity {math.exp(loss):.2f} on {len(ctx.prose())} tokens of prose (limit {limit:g})"


@test(SCORE, TOKENIZE)
def causal(ctx):
    """A prefix's rows equal the whole prompt's first rows: no position sees a later token, and prompt lengths at
    tile and chunk edges compute the same rows."""
    ids, full = ctx.prose(), ctx.prose_rows()
    lengths = [n for n in PREFIXES if n < len(ids)]
    worst = None
    for n in lengths:
        rows = ctx.client.score(ids[:n])[0]
        ok, summary = compare(rows, full[:n - 1], ctx.tolerance)
        if not ok:
            return "failed", f"prefix of {n} tokens against the whole prompt of {len(ids)}: {summary}"
        worst = worst or summary
    return "passed", f"{len(lengths)} prefixes ({lengths[0]} to {lengths[-1]} tokens) match the whole prompt"


@test(SCORE, TOKENIZE)
def decode_matches_prefill(ctx):
    """Decoding a text after a prefill scores it as prefilling all of it does: KV cache writes and reads, positions
    and the decode kernels agree with the prefill path."""
    ids, full = ctx.prose(), ctx.prose_rows()
    steps, details = 48, []
    for split in (16, 200):
        if split + steps > len(ids):
            continue
        _, rows = ctx.client.score(ids[:split], ids[split:split + steps], prompt_rows=False)
        problem = check_rows(rows, "continuation")
        if problem:
            return "failed", problem
        ok, summary = compare(rows, full[split - 1:split - 1 + steps], ctx.decode_tolerance)
        if not ok:
            return "failed", f"{steps} tokens decoded after {split}: {summary}"
        details.append(f"after {split}: {summary}")
    return "passed", "; ".join(details)


@test(SCORE, TOKENIZE)
def causal_long(ctx):
    """causal past 1k tokens, up to `long`: prefixes at power-of-two edges and between them against the whole long
    prompt, each on its last rows. Catches sparse or windowed attention, chunked prefill and buckets that only start
    past a length."""
    ids, full = ctx.long_ids(), ctx.long_rows()
    lengths = [n for n in LONG_PREFIXES if n < len(ids)]
    if not lengths:
        return "skipped", f"the longest prompt is {len(ids)} tokens (--long, the context)"
    for n in lengths:
        rows = ctx.client.score(ids[:n])[0]
        if len(rows) != n - 1:
            return "failed", f"{len(rows)} rows for a prompt of {n} tokens"
        start = max(0, n - 1 - WINDOW)
        ok, summary = compare(rows[start:], full[start:n - 1], ctx.tolerance)
        if not ok:
            return "failed", f"prefix of {n} tokens against the whole prompt of {len(ids)}, rows {start} to {n - 2}: " \
                             + summary
    return "passed", f"{len(lengths)} prefixes ({lengths[0]} to {lengths[-1]} tokens) match the whole prompt of " \
                     f"{len(ids)} on their last {WINDOW} rows"


@test(SCORE, TOKENIZE)
def decode_long(ctx):
    """decode_matches_prefill after long prompts: 48 tokens decoded after prefills past 1k tokens against
    prefilling the whole long prompt."""
    ids, full = ctx.long_ids(), ctx.long_rows()
    steps = 48
    splits = sorted({n for n in (1100, 2100, len(ids) - steps) if 1024 < n <= len(ids) - steps})
    if not splits:
        return "skipped", f"the longest prompt is {len(ids)} tokens (--long, the context)"
    details = []
    for split in splits:
        _, rows = ctx.client.score(ids[:split], ids[split:split + steps], prompt_rows=False)
        problem = check_rows(rows, "continuation")
        if problem:
            return "failed", f"after {split}: {problem}"
        ok, summary = compare(rows, full[split - 1:split - 1 + steps], ctx.decode_tolerance)
        if not ok:
            return "failed", f"{steps} tokens decoded after {split}: {summary}"
        details.append(f"after {split}: mean delta {summary.split('mean delta ')[1].split(',')[0]}")
    return "passed", "; ".join(details)


@test(LOGPROBS, TOKENIZE)
def greedy_is_top(ctx):
    """A greedy reply takes its top-1 token at every step, and scoring the reply as fixed text gives the same
    rows: sampling, draft acceptance and the reply's log-probabilities agree."""
    messages = [{"role": "user", "content": PROMPTS[0]}]
    tokens, rows, _ = ctx.client.chat(messages, 32)
    if not tokens:
        return "failed", "the greedy reply has no tokens"
    problem = check_rows(rows, "reply")
    if problem:
        return "failed", problem
    for index, (token, row) in enumerate(zip(tokens, rows)):
        if not is_choice(row, token, ctx.tolerance["tie"]):
            return "failed", f"token {index} of the reply is not its row's top-1 (it has {row[1]:.3f}, " \
                             f"the top-1 {row[2][0][1]:.3f})"
    if SCORE not in ctx.capabilities:
        return "passed", f"{len(tokens)} tokens, each its row's top-1 (no chat-score-v1 to score the reply)"
    _, scored = ctx.client.score(ctx.client.render(messages), tokens, prompt_rows=False)
    ok, summary = compare(rows, scored, ctx.tolerance)
    return ("passed" if ok else "failed"), f"{len(tokens)} tokens, each its row's top-1; reply against its score: " \
                                           f"{summary}"


def _reply(ctx, prompt, plain=False):
    return ctx.client.chat([{"role": "user", "content": prompt}], 32, plain=plain)


def _tie(ctx):
    """Repeated requests may part only at a near-tie when the engine's arithmetic may depend on what it served
    before (loose); strict asks for equal tokens."""
    return ctx.tolerance["tie"] if ctx.profile == "loose" else 0


@test(LOGPROBS)
def repeatable(ctx):
    """The same request twice gives the same reply."""
    first, rows, _ = _reply(ctx, PROMPTS[1])
    second, _, _ = _reply(ctx, PROMPTS[1])
    ok, summary = same_tokens(first, rows, second, _tie(ctx))
    return ("passed" if ok else "failed"), summary


@test(LOGPROBS)
def isolated(ctx):
    """A request answers the same after a long, different request: no state leaks from one request to the next
    (KV, recurrent state, prompt cache, expert residency)."""
    first, rows, _ = _reply(ctx, PROMPTS[2])
    ctx.client.chat([{"role": "user", "content": PROSE + "\n\nSummarise this text in one sentence."}], 24)
    again, _, _ = _reply(ctx, PROMPTS[2])
    ok, summary = same_tokens(first, rows, again, _tie(ctx))
    return ("passed" if ok else "failed"), "after a long request: " + summary


@test(LOGPROBS, PLAIN)
def plain_equal(ctx):
    """A speculative engine's reply equals its plain decode, one token per target pass."""
    if not ctx.client.facts().get("speculative"):
        return "skipped", "the engine reports no speculative decoding"
    plain, rows, _ = _reply(ctx, PROMPTS[0], plain=True)
    drafted, _, _ = _reply(ctx, PROMPTS[0])
    ok, summary = same_tokens(plain, rows, drafted, ctx.tolerance["tie"])
    return ("passed" if ok else "failed"), "speculative against plain: " + summary


@test(LOGPROBS, CONCURRENT)
def concurrent(ctx):
    """Requests running at the same time answer as each does alone: no request reads another's state."""
    width = min(ctx.client.concurrency(), len(PROMPTS))
    if width < 2:
        return "skipped", "the engine runs one request at a time"
    alone = [_reply(ctx, p) for p in PROMPTS[:width]]
    together, errors = [None] * width, []

    def run(index):
        try:
            together[index] = _reply(ctx, PROMPTS[index])
        except Exception as error:
            errors.append(error)
    threads = [threading.Thread(target=run, args=(i,)) for i in range(width)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    if errors:
        return "failed", f"a concurrent request failed: {errors[0]}"
    for index, ((tokens, rows, _), (other, _, _)) in enumerate(zip(alone, together)):
        ok, summary = same_tokens(tokens, rows, other, ctx.tolerance["tie"])
        if not ok:
            return "failed", f"request {index} of {width} at once: {summary}"
    return "passed", f"{width} requests at once answered as each alone"


@test(SCORE, TOKENIZE)
def recall(ctx):
    """A random text written twice reads the second time at a very low loss, in prefill and in decode, at a short
    and at a long position: attention, positions and the KV cache work across the whole context."""
    details = []
    for length in sorted({min(1024, ctx.long), ctx.long}):
        ids, offset = ctx.recall_ids(length)
        rows = ctx.client.score(ids)[0]
        special = len(ctx.once("special", lambda: ctx.client.tokenize("", True)))
        first, second = nll(rows[special:offset]), nll(rows[offset + 8:])
        quarter = max(1, (len(rows) - offset - 8) // 4)
        quarters = [nll(rows[offset + 8 + i * quarter:offset + 8 + (i + 1) * quarter]) for i in range(4)]
        if not second <= 1.0 or not second < first / 3:
            return "failed", (f"{len(ids)} tokens: the repeat reads at {second:.3f} nats a token against "
                              f"{first:.3f} the first time; by quarter {', '.join(f'{q:.2f}' for q in quarters)}")
        steps = 32
        split = len(ids) - steps
        _, decoded = ctx.client.score(ids[:split], ids[split:], prompt_rows=False)
        loss = nll(decoded)
        if not loss <= 1.0:
            return "failed", f"{len(ids)} tokens: decoding the repeat's last {steps} tokens reads at {loss:.3f} " \
                             f"nats a token, its prefill at {nll(rows[split - 1:]):.3f}"
        details.append(f"{len(ids)} tokens: repeat {second:.3f} vs first {first:.3f} nats, decode {loss:.3f}")
    return "passed", "; ".join(details)


def run(client, only=None, profile="loose", long=4096, log=print, budget=300):
    """Run the smoke tests against `client`'s engine. Returns items {name, state, detail, seconds}, the shape of
    crit run serve --check's report: state passed, failed or skipped. A suite still running after `budget` seconds
    stops with a failed `budget` item: a working engine answers these requests in a minute or two, so one that
    cannot is too slow, or slows down as it serves, and the rest would only hold the machine."""
    ctx = Context(client, profile, long)
    items, begun = [], time.perf_counter()
    for name, needs, function in TESTS:
        if only and name not in only:
            continue
        spent = time.perf_counter() - begun
        if budget is not None and spent > budget:
            left = [n for n, _, _ in TESTS[[t[0] for t in TESTS].index(name):] if not only or n in only]
            item = {"name": "budget", "state": "failed", "seconds": 0.0,
                    "detail": f"the suite passed its budget of {budget:g} s after {spent:.0f} s; not run: "
                              + ", ".join(left)}
            items.append(item)
            log(f"{'failed':>7}  {'budget':<24} {0:6.2f}s  {item['detail']}")
            break
        missing = [n for n in needs if n not in ctx.capabilities]
        started = time.perf_counter()
        if missing:
            state, detail = "skipped", "the engine does not declare " + ", ".join(missing)
        else:
            try:
                state, detail = function(ctx)
            except Unsupported as error:
                state, detail = "skipped", f"not offered: {error}"
            except Exception as error:
                state, detail = "failed", f"{type(error).__name__}: {error}"[:2000]
        item = {"name": name, "state": state, "detail": detail, "seconds": round(time.perf_counter() - started, 3)}
        items.append(item)
        log(f"{state:>7}  {name:<24} {item['seconds']:6.2f}s  {detail}")
    return items
