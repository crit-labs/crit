"""Drive a toy Python step model through crit_core.native.StepEngine and Core's concurrent runtime (crit/step_engine.h,
the crit_core_steps library): requests decode the same tokens batched as one at a time, steps hold several requests
and mix prompt chunks with decode rows, cancellation and retirement reach the model and release every session,
page-reserving admission defers requests, log-probabilities, replay and stop tokens pass through, scores run as Core's
barrier, and a failing step fails the engine. Core's loader and its adapters over a model plugin are
tests/test_core_engine.py's."""
import ctypes
import sys
import threading
import time
import types
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "core" / "python"))

from crit_core.batch import DECODE, PREFILL, PageReservations, StepLayout, step_item  # noqa: E402
from crit_core.native import (NativeError, NativeRounds, Scores, StepEngine,  # noqa: E402
                              StepResult, _LAUNCH, _WAIT)

VOCAB, STOP = 1000, 7


def expect(condition, message):
    if not condition:
        raise AssertionError(message)


def successor(seen):
    """The toy model: the next token is a hash of every token a row sees, so wrong bookkeeping changes it."""
    h = 1469598103934665603
    for position, token in enumerate(seen):
        h = ((h ^ ((token + 1) * (position + 1))) * 1099511628211) % 2 ** 64
    return h % VOCAB


def oracle(prompt, output):
    rows, out = list(prompt), []
    for _ in range(output):
        out.append(successor(rows))
        rows.append(out[-1])
    return out


class Session:
    def __init__(self, request):
        self.request, self.prompt = request, list(request.prompt)
        self.rows, self.outputs = [], []        # materialized rows; published outputs
        self.staged, self.produced, self.pending, self.pages = None, [], None, 0


class ToyStepModel:
    """A batched model in Python: one step lays out every scheduled row with crit_core.batch (decode rows first, then
    prompt chunks) and computes each logits row from the row's own history only."""

    def __init__(self, concurrency=4, pages=None, fail_on=None, slow=None):
        # A step holding the `slow` prompt's request takes 20 ms, so a client can cancel that request mid-decode.
        self.concurrency, self.fail_on, self.slow = concurrency, fail_on, slow
        self.reservations = PageReservations(pages, 4, 64) if pages else None
        self.steps, self.opened, self.retired, self.cancelled, self.closed = [], [], [], 0, False
        self.lock = threading.Lock()   # the callbacks run on Core's one worker; this only guards the test's reads

    def info(self):
        return {"implementation": "toy step model", "vocabulary": VOCAB, "context": 64, "prefill_chunk": 8,
                "max_concurrency": self.concurrency, "max_pending": 16, "max_batch_tokens": 24, "stop_tokens": [STOP],
                "sampling": {"temperature": True}, "token_logprobs": True, "max_top_logprobs": 2, "scoring": True,
                "score_paths": ["prefill"], "replay": True, "speculative": False, "native_libraries": ["toy.so"]}

    def admission(self, request):
        return self.reservations.admission(len(request.prompt), request.output_tokens) if self.reservations else "ready"

    def open(self, request):
        session = Session(request)
        if self.reservations:
            session.pages = self.reservations.reserve(len(request.prompt), request.output_tokens)
        self.opened.append(request.key)
        return session

    def step(self, rows):
        layout = StepLayout(step_item(row, len(row.session.prompt), len(row.session.rows)) for row in rows)
        if self.slow is not None and any(row.session.prompt == self.slow for row in rows):
            time.sleep(0.02)
        with self.lock:
            self.steps.append((len(rows), layout.decode_items))
        results = [None] * len(rows)
        for placement in layout.placements:
            s = rows[placement.item].session
            start = len(s.rows)
            inputs = [s.pending] if placement.phase == DECODE else s.prompt[start:start + placement.rows]
            if self.fail_on is not None and self.fail_on in inputs:
                raise RuntimeError("the toy model refuses token %d" % self.fail_on)
            s.staged, s.produced = inputs, []
            if placement.logits:
                replay = s.request.replay
                s.produced = [replay[len(s.outputs)] if replay else successor(s.rows + inputs)]
            logprobs = None
            if s.request.logprobs is not None and s.produced:
                token = s.produced[0]
                logprobs = [(-0.5, [(token, -0.5), ((token + 1) % VOCAB, -0.75)])]
            results[placement.item] = StepResult(placement.rows if placement.phase == PREFILL else 0, s.produced,
                                                 placement.logits > 0, 0, placement.rows, logprobs)
        return results

    def commit(self, decisions):
        for d in decisions:
            s = d.session
            if s.staged is None:
                continue
            if d.cancelled or d.discard:
                s.staged = None
                self.cancelled += d.cancelled
                continue
            s.rows += s.staged
            s.staged = None
            if d.accepted:
                s.pending = s.produced[0]
                s.outputs += s.produced[:d.accepted]

    def retire(self, session, finish):
        with self.lock:
            self.retired.append(finish)
        if self.reservations:
            self.reservations.release(session.pages)

    def score(self, ids, first_target, path="", queries=None, top_k=0):
        targets = range(first_target, len(ids))
        return Scores([-0.5] * len(targets), [[-1.0] * len(row) for row in queries] if queries else None,
                      [successor(ids[:t]) for t in targets],
                      [[(ids[t], -0.5), ((ids[t] + 1) % VOCAB, -0.75)][:top_k] for t in targets] if top_k else None)

    def close(self):
        self.closed = True


def prompts():
    return [[(r * 37 + i * 11) % VOCAB for i in range(1 + r * 5)] for r in range(9)]


def run_all(engine, output):
    with ThreadPoolExecutor(9) as pool:
        return list(pool.map(lambda p: engine.generate(p, output, stop_tokens=[])[0], prompts()))


def batched_equals_serial(library):
    for concurrency in (1, 4):
        model = ToyStepModel(concurrency)
        engine = StepEngine(model, library)
        expect(engine.info()["max_concurrent_requests"] == concurrency + 16, engine.info())
        got = run_all(engine, 12)
        expect(got == [oracle(p, 12) for p in prompts()], ("tokens differ from the oracle", concurrency))
        largest = max(n for n, _ in model.steps)
        if concurrency > 1:
            expect(largest > 1 and any(0 < d < n for n, d in model.steps), ("steps hold several requests and mix", model.steps))
        else:
            expect(largest == 1, model.steps)
        engine.close()
        expect(model.closed and not engine._sessions, "close releases every session and closes the model")
        expect(len(model.retired) == len(model.opened) == 9 and set(model.retired) == {"length"}, model.retired)


def cancellation_and_retirement(library):
    model = ToyStepModel(4, slow=[1, 2, 3])
    engine = StepEngine(model, library)
    seen = []
    with ThreadPoolExecutor(3) as pool:
        long = pool.submit(engine.generate, [1, 2, 3], 40, stop_tokens=[], on_tokens=seen.extend,
                           cancelled=lambda: len(seen) >= 3)
        peers = [pool.submit(engine.generate, p, 10, stop_tokens=[]) for p in prompts()[:2]]
        expect(long.result(timeout=30) == (None, {"cancelled": True}), "the long request is cancelled")
        expect([f.result(timeout=30)[0] for f in peers] == [oracle(p, 10) for p in prompts()[:2]], "peers are unaffected")
    expect(3 <= len(seen) < 40, seen)
    expect("cancelled" in model.retired, ("the model retires the cancelled request", model.retired))
    tokens, result = engine.generate([5, 6], 4, stop_tokens=[])
    expect(tokens == oracle([5, 6], 4) and result["finish"] == "length", "the engine serves after a cancellation")
    expect(engine.generate([1], 4, cancelled=lambda: True) == (None, {"cancelled": True}), "early cancellation")
    engine.close()
    expect(not engine._sessions and len(model.retired) == len(model.opened), (model.retired, model.opened))


def admission(library):
    # 20 pages of 4 rows: these requests need 9 to 14 pages, so at most two run at once; the rest wait their turn.
    model = ToyStepModel(6, pages=20)
    engine = StepEngine(model, library)
    later = prompts()[4:]
    with ThreadPoolExecutor(len(later)) as pool:
        got = list(pool.map(lambda p: engine.generate(p, 12, stop_tokens=[])[0], later))
    expect(got == [oracle(p, 12) for p in later], "deferred requests decode the oracle's tokens")
    expect(max(n for n, _ in model.steps) <= 2 and model.reservations.reserved == 0, model.steps)
    try:
        engine.generate([1] * 62, 4)
    except NativeError as error:
        expect("prompt plus output exceeds context" in str(error), error)
    else:
        raise AssertionError("a request beyond the context must fail")
    engine.close()


def options(library):
    model = ToyStepModel(4)
    engine = StepEngine(model, library)
    rows = []
    tokens, result = engine.generate([1, 2, 3], 5, stop_tokens=[], logprobs=2, on_tokens=lambda ts, lps: rows.extend(zip(ts, lps)))
    expect(tokens == oracle([1, 2, 3], 5) and [t for t, _ in rows] == tokens, (tokens, rows))
    expect(all(lp == -0.5 and top == [(t, -0.5), ((t + 1) % VOCAB, -0.75)] for t, (lp, top) in rows), rows)
    # Replay decodes the continuation; Core ends it at the model's stop token unless stops are disabled.
    tokens, result = engine.generate([1, 2, 3], 6, replay=[12, 13, STOP, 14, 15, 16])
    expect(tokens == [12, 13, STOP] and result["finish"] == "stop", (tokens, result))
    tokens, result = engine.generate([1, 2, 3], 6, stop_tokens=[], replay=[12, 13, STOP, 14, 15, 16])
    expect(tokens == [12, 13, STOP, 14, 15, 16] and result["finish"] == "length", (tokens, result))
    expect(model.opened and model.retired.count("stop") == 1, model.retired)
    # Scores run on Core's worker as an exclusive barrier, with the engine's own top-k on a named path.
    scored = engine.score([1, 2, 3, 4], 2, "prefill", top_k=2)
    expect(scored.target == [-0.5, -0.5] and scored.argmax == [successor([1, 2]), successor([1, 2, 3])]
           and scored.top == [[(3, -0.5), (4, -0.75)], [(4, -0.5), (5, -0.75)]], scored)
    queried = engine.score([1, 2, 3, 4], 2, "prefill", [[7, 8], [9, 10]])
    expect(queried.query == [[-1.0, -1.0], [-1.0, -1.0]], queried)
    try:
        engine.generate([1, 2], 3, sampling={"top_p": 0.5})
    except NativeError as error:
        expect("does not support a requested sampling option" in str(error), error)
    else:
        raise AssertionError("Core refuses sampling fields the model does not apply")
    engine.close()


def failure(library):
    model = ToyStepModel(4, fail_on=666)
    engine = StepEngine(model, library)
    expect(engine.generate([1, 2], 3, stop_tokens=[])[0] == oracle([1, 2], 3), "a healthy request")
    try:
        engine.generate([5, 666, 6], 3)
    except NativeError as error:
        expect("step engine step failed: RuntimeError: the toy model refuses token 666" in str(error), error)
    else:
        raise AssertionError("a failing step must fail the request")
    try:
        engine.generate([1, 2], 3)
        raise AssertionError("a failed step engine refuses later requests")
    except NativeError:
        pass
    engine.close()
    try:
        StepEngine(types.SimpleNamespace(info=lambda: {"context": 64}, open=None, step=lambda rows: []), library)
    except NativeError as error:
        expect("vocabulary" in str(error), error)
    else:
        raise AssertionError("a step model without a vocabulary must not open")


def round_logprob(token):
    """RoundModel's log-probability of an emitted token and its two ranked alternatives."""
    value = -((token % 5) + 1) / 4.0
    return value, [(token, value), ((token + 1) % VOCAB, value - 1.0)]


class RoundModel:
    """Native decode rounds (NativeRounds) without a GPU: the "device" round is a C callback (launch) that decodes up
    to width + 1 tokens for each of slots 0 .. rows - 1 into a host block, as a captured graph would. The model's own
    step() only prefills (and decodes the row counts given no width); Core runs every other decode step itself. With
    `top` alternatives its rounds have log-probability buffers, which only a round launched in its log-probability form
    writes (round_logprob of each emitted token; the plain form leaves NaN and a wrong id there) and in which a replay
    row emits its continuation."""

    COUNTERS = ("expert_reads", "expert_hits")

    def __init__(self, width=2, widths=None, rows=4, telemetry=False, policy=None, top=0):
        self.width, self.rows, self.telemetry, self.top = width, rows, telemetry, top
        self.widths = widths or [width] * rows
        # The engine's widths array, which Core reads before each round; `policy` (rows, width) -> the next widths,
        # written as a captured round's own copy would, inside the round.
        self.width_array = (ctypes.c_int32 * rows)(*[-1 if w is None else int(w) for w in self.widths])
        self.policy, self.round_widths, self.forms = policy, [], []
        self.slots, self.history, self.allowance = [], [None] * rows, [0] * rows
        self.continuation = [None] * rows   # a replay row's tokens still to emit
        # Each slot's row: tokens emitted, drafts matched, frontier, the tokens, and with telemetry its words: its
        # accepted drafts, then the engine's two counters (round_counters).
        self.stride = 3 + width + 1 + (3 if telemetry else 0)
        self.block = (ctypes.c_int64 * (rows * self.stride))()
        cols = width + 1
        self.logprobs = (ctypes.c_float * (rows * cols))()
        self.top_ids = (ctypes.c_int64 * (rows * cols * max(top, 1)))()
        self.top_logprobs = (ctypes.c_float * (rows * cols * max(top, 1)))()
        self.decode_steps, self.native_rounds_run, self.commits, self.native = 0, 0, [], None
        self._launch = _LAUNCH(self.launch)
        self._wait = _WAIT(lambda _user: 0)

    def info(self):
        return {"implementation": "toy native rounds", "vocabulary": VOCAB, "context": 64, "prefill_chunk": 64,
                "max_concurrency": self.rows, "max_pending": 16, "max_batch_tokens": 0,
                "max_row_tokens": self.width + 1, "decode_token_work": self.width + 1,
                **({"round_counters": list(self.COUNTERS)} if self.telemetry else {}),
                **({"token_logprobs": True, "max_top_logprobs": self.top, "replay": True} if self.top else {})}

    def native_rounds(self, _library):
        ranked = dict(logprobs=ctypes.addressof(self.logprobs), top_ids=ctypes.addressof(self.top_ids),
                      top_logprobs=ctypes.addressof(self.top_logprobs), top=self.top) if self.top else {}
        self.native = NativeRounds(self._launch, self._wait, None, ctypes.addressof(self.block), self.stride,
                                   8, self.rows, ctypes.addressof(self.width_array), self.width,
                                   telemetry=3 + self.width + 1 if self.telemetry else 0,
                                   counters=len(self.COUNTERS) if self.telemetry else 0, **ranked)
        return self.native

    def launch(self, _user, rows, width, logprobs):
        self.native_rounds_run += 1
        self.round_widths.append(width)
        self.forms.append(bool(logprobs))
        if self.policy is not None:
            for r in range(self.rows):
                if self.width_array[r] >= 0:
                    self.width_array[r] = self.policy(rows, width)
        stride, cols = self.stride, self.width + 1
        for slot in range(rows):
            seen, count = self.history[slot], min(width + 1, self.allowance[slot])
            produced = []
            for _ in range(count):
                # A replay row follows its continuation only in the log-probability form, as a captured round does.
                follows = logprobs and self.continuation[slot]
                produced.append(self.continuation[slot].pop(0) if follows else successor(seen))
                seen.append(produced[-1])
            self.allowance[slot] -= count
            self.block[slot * stride] = count
            self.block[slot * stride + 1] = max(0, count - 1)
            self.block[slot * stride + 2] = len(seen)
            for t, token in enumerate(produced):
                self.block[slot * stride + 3 + t] = token
                value, top = round_logprob(token) if logprobs else (float("nan"), [(VOCAB - 1, float("nan"))] * 2)
                self.logprobs[slot * cols + t] = value
                for a in range(self.top):
                    self.top_ids[(slot * cols + t) * self.top + a], self.top_logprobs[(slot * cols + t) * self.top + a] \
                        = top[a]
            if self.telemetry:   # accepted drafts, then the counters: ten expert reads and one hit per token
                words = slot * stride + 3 + self.width + 1
                self.block[words], self.block[words + 1], self.block[words + 2] = max(0, count - 1), 10 * count, count
        return 0

    def open(self, request):
        return Session(request)

    def step(self, rows):
        out = {}
        decode = [r for r in rows if r.prefill_complete]
        for r in rows:
            if r.prefill_complete:
                continue
            s = r.session
            s.slot = len(self.slots)
            self.slots.append(s)
            self.history[s.slot] = list(s.prompt)
            replay = list(s.request.replay or [])
            first = replay.pop(0) if replay else successor(self.history[s.slot])
            self.continuation[s.slot] = replay
            self.history[s.slot].append(first)
            self.allowance[s.slot] = s.request.output_tokens - 1
            self.native.bind(s.slot, s)
            scored = [round_logprob(first)] if s.request.logprobs is not None else None
            out[id(s)] = StepResult(len(s.prompt), [first], True, 0, 0, scored)
        if decode:   # a row count without a native width: the model decodes it one token a row
            self.decode_steps += 1
            for r in decode:
                seen = self.history[r.session.slot]
                token = successor(seen) if self.allowance[r.session.slot] > 0 else None
                if token is not None:
                    seen.append(token)
                    self.allowance[r.session.slot] -= 1
                out[id(r.session)] = StepResult(0, [token] if token is not None else [], True)
        return [out[id(r.session)] for r in rows]

    def commit(self, decisions):
        self.commits.append((self.native.last, [(d.accepted, d.terminal) for d in decisions]))

    def retire(self, session, finish):
        slot, last = session.slot, len(self.slots) - 1
        if slot != last:
            moved = self.slots[last]
            self.history[slot], self.allowance[slot] = self.history[last], self.allowance[last]
            self.continuation[slot] = self.continuation[last]
            moved.slot = slot
            self.slots[slot] = moved
            self.native.bind(slot, moved)
        self.slots.pop()
        self.native.unbind(last)


def native_rounds(library):
    model = RoundModel()
    engine = StepEngine(model, library)
    got = run_all(engine, 12)
    expect(got == [oracle(p, 12) for p in prompts()], "native rounds decode the oracle's tokens")
    expect(model.native_rounds_run > 0 and model.decode_steps == 0, ("Core ran the decode steps", model.decode_steps))
    native_commits = [c for c in model.commits if c[0][2]]
    expect(native_commits and all(any(t for _, t in d) for _, d in native_commits),
           ("after a native round, commit only for a decision that ends a row", model.commits))
    engine.close()
    # A row count the model gives no width decodes through step(), the others natively.
    model = RoundModel(widths=[2, None, 2, 2])
    engine = StepEngine(model, library)
    got = run_all(engine, 12)
    expect(got == [oracle(p, 12) for p in prompts()], "mixed native and step decoding keeps the tokens")
    expect(model.native_rounds_run > 0 and model.decode_steps > 0, (model.native_rounds_run, model.decode_steps))
    core = engine.telemetry()["core"]
    expect(core["native_rounds"] == model.native_rounds_run and core["steps"] > core["native_rounds"], core)
    engine.close()
    # Widths the engine rewrites between steps (a draft-width policy; here cycling 2, 0, 1): Core runs each round at
    # the width its row count holds then, and the tokens stay the oracle's.
    model = RoundModel(policy=lambda rows, width: (width + 1) % 3)
    engine = StepEngine(model, library)
    got = run_all(engine, 12)
    expect(got == [oracle(p, 12) for p in prompts()], "rewritten widths keep the tokens")
    expect(model.decode_steps == 0 and set(model.round_widths) == {0, 1, 2}, ("every width ran", model.round_widths))
    engine.close()
    # A width past the widest round is refused when the engine opens and when a round would run it.
    model = RoundModel(widths=[3, 2, 2, 2])
    try:
        StepEngine(model, library).close()
    except NativeError as error:
        expect("wider" in str(error), error)
    else:
        raise AssertionError("a width past the widest must not open")
    model = RoundModel(policy=lambda rows, width: 3)
    engine = StepEngine(model, library)
    try:
        run_all(engine, 12)
    except NativeError as error:
        expect("wider" in str(error), error)
    else:
        raise AssertionError("a round past the widest must not run")
    finally:
        engine.close()
    # Native rounds an engine's settings cannot hold are refused when it opens, not when a round exceeds a grant.
    for info, why in (({"max_row_tokens": 2}, "max_row_tokens"), ({"decode_token_work": 1}, "decode_token_work"),
                      ({"max_batch_tokens": 6}, "max_batch_tokens"), ({"round_counters": ["reads"]}, "round_counters")):
        model = RoundModel()
        model.info = lambda base=model.info, info=info: {**base(), **info}
        try:
            StepEngine(model, library).close()
        except NativeError as error:
            expect(why in str(error), (why, error))
        else:
            raise AssertionError(f"native rounds past {why} must not open")


def native_round_forms(library):
    """Core launches a native round in its log-probability form only when one of its rows' requests asks for
    log-probabilities or replays a continuation, else in its plain form: a plain request's rounds run plain, a
    request asking for log-probabilities gets the log-probability rounds' values, a replay follows its continuation,
    and a plain request beside them decodes the tokens it decodes alone."""
    model = RoundModel(width=2, rows=3, top=2)
    engine = StepEngine(model, library)
    forms = model.forms
    try:
        tokens, _ = engine.generate(prompts()[1], 12, stop_tokens=[])
        expect(tokens == oracle(prompts()[1], 12), "a plain request decodes the oracle's tokens")
        expect(forms and not any(forms), ("a plain request's rounds run the plain form", forms))
        expect(engine.telemetry()["core"]["native_logprob_rounds"] == 0, engine.telemetry()["core"])
        start, rows = len(forms), []
        tokens, _ = engine.generate(prompts()[2], 12, stop_tokens=[], logprobs=2,
                                    on_tokens=lambda ts, lps: rows.extend(zip(ts, lps)))
        expect(tokens == oracle(prompts()[2], 12) and [t for t, _ in rows] == tokens, (tokens, rows))
        expect(forms[start:] and all(forms[start:]), ("its rounds run the log-probability form", forms))
        expect(all((lp, top) == round_logprob(t) for t, (lp, top) in rows), ("the rounds' log-probabilities", rows))
        start, text = len(forms), [12, 13, 14, 15, 16, 17, 18]
        tokens, _ = engine.generate([1, 2, 3], len(text), stop_tokens=[], replay=text)
        expect(tokens == text, ("a replay follows its continuation", tokens))
        expect(forms[start:] and all(forms[start:]), ("a replay's rounds run the log-probability form", forms))
        # A plain request beside one asking for log-probabilities and a replay: the rounds holding either run the
        # log-probability form, the plain request's later rounds the plain one, and its tokens are those it decodes
        # alone.
        start = len(forms)
        requests = [(prompts()[3], 30, {}), (prompts()[4], 6, {"logprobs": 2}), ([4, 5], 4, {"replay": text[:4]})]
        with ThreadPoolExecutor(len(requests)) as pool:
            found = list(pool.map(lambda r: engine.generate(r[0], r[1], stop_tokens=[], **r[2])[0], requests))
        expect(found == [oracle(prompts()[3], 30), oracle(prompts()[4], 6), text[:4]], found)
        expect(True in forms[start:] and False in forms[start:], ("both forms ran", forms[start:]))
        core = engine.telemetry()["core"]
        expect(core["native_logprob_rounds"] == forms.count(True) and core["native_rounds"] == len(forms),
               (core, forms))
    finally:
        engine.close()


class ChunkedModel(ToyStepModel):
    """ToyStepModel recording where each prompt chunk starts, with prompt work granted in whole chunks
    (prefill_work_quantum, the prefill chunk) when `quantum`."""

    def __init__(self, quantum):
        super().__init__(concurrency=4)
        self.quantum, self.starts = quantum, []

    def info(self):
        return {**super().info(), **({"prefill_work_quantum": 8} if self.quantum else {})}

    def step(self, rows):
        for row in rows:
            if not row.prefill_complete:
                self.starts.append(len(row.session.rows))
        return super().step(rows)


def prefill_quantum(library):
    """A step model that declares its prefill chunk as the prefill work quantum gets prompt chunks that start at
    multiples of it, however the requests share their steps, and decodes the oracle's tokens."""
    model = ChunkedModel(quantum=True)
    engine = StepEngine(model, library)
    try:
        expect(run_all(engine, 6) == [oracle(p, 6) for p in prompts()], "the tokens")
        expect(model.starts and all(start % 8 == 0 for start in model.starts), ("chunks at multiples of 8", model.starts))
    finally:
        engine.close()


def request_facts(library):
    """Every generation's result carries its rounds' facts: Core counts each decode row as a round, takes accepted
    drafts, the width and the engine's named counters from a native round's words, and merges the model's
    request_facts(session) over them round by round (its counts of a declared counter add to the native rounds'); a
    listener gets each round's tokens in one call; facts larger than Core's first buffer are asked of the model once;
    a cancelled request's facts are never asked for."""
    # One request at a time: every decode row natively, or (no width for one row) every one through step().
    for widths in (None, [None, 2, 2, 2]):
        model = RoundModel(telemetry=True, widths=widths)
        model.request_facts = lambda session: {"verify_ms": 2.0, "expert_reads": 5, "finish": "never a fact"}
        engine = StepEngine(model, library)
        try:
            for prompt in prompts():
                deliveries = []
                tokens, result = engine.generate(prompt, 12, stop_tokens=[], on_tokens=deliveries.append)
                expect(tokens == oracle(prompt, 12), "the tokens")
                expect(result["finish"] == "length" and result["verify_ms"] == 2.0, result)
                if widths is None:
                    expect(result["rounds"] == len(result["accepted"]) == len(result["drafted"]), result)
                    expect(sum(result["accepted"]) + result["rounds"] == len(tokens) - 1, ("a round a decode row", result))
                    expect(result["drafted"] == [2] * result["rounds"], result)
                    expect((result["expert_reads"], result["expert_hits"]) == (5 + 10 * (len(tokens) - 1),
                                                                              len(tokens) - 1),
                           ("the model's reads add to the native rounds'", result))
                    expect(max(len(d) for d in deliveries) > 1, ("a round's tokens in one call", deliveries))
                else:
                    # step() rows of one token each: rounds, no drafts to report, the model's reads alone.
                    expect(result["rounds"] == len(tokens) - 1 and "accepted" not in result, result)
                    expect(result["expert_reads"] == 5 and "expert_hits" not in result, result)
        finally:
            engine.close()
    # Without request_facts, Core's facts alone; without drafts, only the round count.
    model = ToyStepModel()
    engine = StepEngine(model, library)
    try:
        tokens, result = engine.generate(prompts()[0], 6, stop_tokens=[])
        expect(result["rounds"] == len(tokens) - 1 and "accepted" not in result and "expert_reads" not in result, result)
    finally:
        engine.close()
    # Facts past Core's first 64 KiB: the model is asked once, and its facts reach the result whole.
    asked = []
    model = ToyStepModel()
    model.request_facts = lambda session: asked.append(session) or {"verify_ms": 1.0, "pad": "x" * 70000}
    engine = StepEngine(model, library)
    try:
        _, result = engine.generate(prompts()[1], 4, stop_tokens=[])
        expect(len(asked) == 1 and len(result.get("pad", "")) == 70000, (len(asked), sorted(result)))
        # A cancelled request: its facts are never asked for (nobody would read them).
        asked.clear()
        model.slow, seen = prompts()[2], []
        found = engine.generate(prompts()[2], 40, stop_tokens=[], on_tokens=seen.extend, cancelled=lambda: bool(seen))
        expect(found == (None, {"cancelled": True}) and seen and not asked, (found, seen, len(asked)))
    finally:
        engine.close()
    # A model's step() rounds take their places among Core's, round by round.
    from crit_core.native import merge_facts
    core = {"rounds": 4, "accepted": [2, 0, 1, 0], "drafted": [2, -1, 2, -1], "_native": [1, 0, 1, 0],
            "expert_reads": 30}
    merged = merge_facts({"accepted": [1, 2], "drafted": [3, 3], "expert_reads": 4, "verify_ms": 1.0}, core,
                         ("expert_reads",))
    expect(merged == {"rounds": 4, "accepted": [2, 1, 1, 2], "drafted": [2, 3, 2, 3], "expert_reads": 34,
                      "verify_ms": 1.0}, merged)
    kept = merge_facts({"accepted": [1]}, dict(core), ())
    expect(kept["accepted"] == [2, 0, 1, 0] and "drafted" not in kept and "_native" not in kept, kept)


def main(library):
    batched_equals_serial(library)
    native_rounds(library)
    native_round_forms(library)
    request_facts(library)
    prefill_quantum(library)
    cancellation_and_retirement(library)
    admission(library)
    options(library)
    failure(library)
    print("step engine checks passed")


if __name__ == "__main__":
    main(sys.argv[1])
