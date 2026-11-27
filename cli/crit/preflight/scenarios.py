"""The scenarios of a load: a candidate served as crit run serve --check serves it, the scenarios that need no load of
their own run on it inside (in_load, from serve's ready hook), and every item joins one report.

What this toolkit's bugs showed up in during engine builds: short, 32k and long prompts, a ladder of prompt lengths
whose prefill rate must rise with the length (where the experts are off the device), a cancel in the middle of a
prefill, reopening at a lower context (a closed engine's elastic ranges must not outlive it), drafting from the draft
layer and through the drafter seam, crit smoke, crit run serve --check (with --data where it is meaningful) with
crit-eval's gate pre-check, and each prompt arithmetic's cost with the CPU tier computing decode misses. A device
memory pressure is made by a VRAM ballast (crit_plugins.memory's CRIT_VRAM_BALLAST: leave:BYTES free), so
expert-offload places experts off the device. Gates that judge quality against the full model's text are left out on
a cut (CUT_MODEL_GATES: a cut model's text is nonsense); self-consistency gates stay. Every size a scenario uses is the
caller's (Sizes): the preflight takes them from the run, the stress test from its fixtures."""
import contextlib
from dataclasses import dataclass
import http.client
import json
import os
import socket
import time
import traceback

PRECHECKS = 3              # with spread: crit-eval's gate pre-check, run this often in one load
CANCEL_AFTER_S = 1.5       # a cancel this far into a long prompt's prefill (or half the prefill, if shorter)
LADDER_SLACK = 0.9         # a rung's prefill rate against the rung's before it: under this share of it fails
# Gates a cut model cannot pass: they judge its text against a full model's.
CUT_MODEL_GATES = {
    "smoke_fluent": "judges perplexity against a working model's text; a cut model's text is nonsense",
    "smoke_recall": "asks the model to read a repeated passage at under 1 nat a token, which only a working model does",
    "tool_call": "parses the arguments of the model's own tool call, which a cut model does not close within its tokens"}
FILLER = ("The river bends around the old mill, and the path follows it past the orchard and the stone wall. ")
FILLER_TOKENS = 22         # FILLER's prompt tokens a copy for a 150k-vocabulary BPE tokenizer, where none is measured


def log(message):
    print(time.strftime("%H:%M:%S ") + str(message), flush=True)


@dataclass(frozen=True)
class Sizes:
    """The prompt and context sizes of a load's scenarios, in tokens. context: the run's (the engine opens at it);
    long: the 32k prompt; lower: the context reopened at; cancel: the prompt cancelled inside its prefill; ladder: the
    prompt ladder's rungs; followup: the first turn of the prompt-cache follow-up; calibrate: size prompts by the
    tokens the engine counts for FILLER (two requests at the load's start) rather than by FILLER_TOKENS."""
    context: int
    long: int = 32768
    lower: int = 65536
    cancel: int = 65536
    ladder: tuple = (24, 96, 192, 400, 1600, 6400)
    followup: int = 4096
    calibrate: bool = False


# Of the first turn's prompt tokens, the follow-up's prompt cache must serve at least this share: the turns share the
# first user message, whatever the chat template renders after it.
FOLLOWUP_SHARE = 0.5
OVER_BUDGET = "left out: the preflight's time budget has no room for it"


def words(tokens, per_copy=FILLER_TOKENS):
    """A user message of about `tokens` prompt tokens, FILLER repeated (per_copy: its tokens a copy)."""
    return FILLER * max(1, tokens // per_copy)


def content(data):
    return data["choices"][0]["message"]["content"]


# -- requests ------------------------------------------------------------------------------------------------------------
class Chat:
    """A small OpenAI chat client for the served engine at `port`."""

    def __init__(self, port, timeout=900):
        self.port, self.timeout = port, timeout
        self.model = None

    def _connection(self):
        return http.client.HTTPConnection("127.0.0.1", self.port, timeout=self.timeout)

    def models(self):
        c = self._connection()
        c.request("GET", "/v1/models")
        found = json.loads(c.getresponse().read())
        c.close()
        return [m["id"] for m in found.get("data", [])]

    def chat(self, content, max_tokens=16, **extra):
        return self.messages([{"role": "user", "content": content}], max_tokens, **extra)

    def per_copy(self):
        """FILLER's prompt tokens a copy, as the engine counts them (one copy and two, one token each): any
        tokenizer, no tokenize call."""
        one = self.chat("Calibrate: " + FILLER, 1)[0]["usage"]["prompt_tokens"]
        two = self.chat("Calibrate: " + FILLER * 2, 1)[0]["usage"]["prompt_tokens"]
        return max(1, two - one)

    def messages(self, messages, max_tokens=16, **extra):
        self.model = self.model or self.models()[0]
        body = {"model": self.model, "messages": messages, "max_tokens": max_tokens, "temperature": 0, **extra}
        c = self._connection()
        began = time.perf_counter()
        c.request("POST", "/v1/chat/completions", json.dumps(body), {"Content-Type": "application/json"})
        response = c.getresponse()
        data = json.loads(response.read() or b"{}")
        c.close()
        if response.status != 200:
            raise RuntimeError(f"HTTP {response.status}: {json.dumps(data)[:400]}")
        return data, time.perf_counter() - began

    def cancelled_stream(self, content, after):
        """A streamed request cancelled `after` seconds in (its connection closed), as a client that goes away."""
        self.model = self.model or self.models()[0]
        body = {"model": self.model, "messages": [{"role": "user", "content": content}], "max_tokens": 64,
                "stream": True, "temperature": 0}
        sock = socket.create_connection(("127.0.0.1", self.port), timeout=self.timeout)
        payload = json.dumps(body).encode()
        sock.sendall(b"POST /v1/chat/completions HTTP/1.1\r\nHost: 127.0.0.1\r\nContent-Type: application/json\r\n"
                     b"Content-Length: " + str(len(payload)).encode() + b"\r\n\r\n" + payload)
        time.sleep(after)
        sock.shutdown(socket.SHUT_RDWR)
        sock.close()


# -- the report ----------------------------------------------------------------------------------------------------------
class Report:
    def __init__(self, log=log):
        self.items, self.log = [], log

    def add(self, name, state, seconds=0.0, detail=""):
        item = {"scenario": name, "state": state, "seconds": round(float(seconds), 1), "detail": str(detail)[:1500]}
        self.items.append(item)
        self.log(f"{state:8s} {name} ({item['seconds']} s) {item['detail'][:200]}")
        return item

    @contextlib.contextmanager
    def scenario(self, name):
        began = time.perf_counter()
        found = {"state": "passed", "detail": ""}
        try:
            yield found
        except Exception as error:   # the scenario's failure, named in the report; the sweep goes on
            found = {"state": "failed", "detail": f"{type(error).__name__}: {error}"}
            traceback.print_exc()
        self.add(name, found["state"], time.perf_counter() - began, found["detail"])


# -- the scenarios inside a load -----------------------------------------------------------------------------------------
def in_load(report, prefix, served, port, extras, sizes, cut=True, budget=None, companion=None):
    """The scenarios that run inside a load, on the engine crit run serve serves (its ready hook). cut: the model is
    a layer cut, whose text is nonsense (CUT_MODEL_GATES). budget: a crit.preflight.plan.Budget whose time a
    scenario must fit in to run (None: every planned scenario runs). companion: the folder of the run's companion
    drafter that the model plugin's layout reads as its draft block (drafting_companion drafts through the model's
    draft head from it), None where the run names none it reads."""
    from crit_plugins.steps import GRACE as grace   # Core's cancel grace, crit serve's (crit_api.residency)
    chat = Chat(port)
    per = chat.per_copy() if sizes.calibrate else FILLER_TOKENS

    def w(tokens):
        return words(tokens, per)

    def due(name):
        """Whether scenario `name` runs: planned, and within the budget (else it is reported skipped)."""
        if budget is None or budget.fits(name):
            return True
        report.add(prefix + name, "skipped", 0.0, f"{OVER_BUDGET} ({budget.describe(name)})")
        return False
    with report.scenario(prefix + "placement") as found:
        engine = served.current[0] if served.current else None
        telemetry = engine.telemetry() if engine is not None else {}
        offload = (telemetry or {}).get("expert_offload") or {}
        cache = offload.get("expert_cache") or {}
        found["state"] = "info"
        found["detail"] = json.dumps({**{k: offload[k] for k in ("tier", "misses", "slots") if k in offload},
                                      **{k: cache[k] for k in ("slots", "ring_slots", "experts", "hits", "reads")
                                         if k in cache},
                                      "reasons": {k: str(v)[:200] for k, v in (offload.get("reasons") or {}).items()},
                                      "telemetry_keys": sorted(telemetry or {})[:30]})
    if "short" in extras:
        with report.scenario(prefix + "short_request") as found:
            data, seconds = chat.chat("Count from one to five.", 16)
            found["detail"] = f"{data['usage']['completion_tokens']} tokens in {seconds:.2f} s"
    if "ladder" in extras and due("prompt_ladder"):
        # Prompts of every length, each a text no earlier request began with (no prompt cache serves it). A request's
        # fixed costs weigh less the longer its prompt, so the rate rises from each rung to the next; a rung slower
        # than the one before it pays something that prompt's length does not ask for (a sweep of every off-device
        # expert for a prompt that routes a few of them), which fails.
        with report.scenario(prefix + "prompt_ladder") as found:
            rungs, slow = [], []
            for tokens in sizes.ladder:
                data, seconds = chat.chat(f"Text {tokens}: " + w(tokens), 1)
                prompt = data["usage"]["prompt_tokens"]
                if rungs and prompt / seconds < LADDER_SLACK * rungs[-1][0] / rungs[-1][1]:
                    slow.append(f"{prompt:,} tokens are prefilled slower than {rungs[-1][0]:,}")
                rungs.append((prompt, seconds))
            found["detail"] = "; ".join(f"{p:,} tokens {s:.2f} s ({p / s:,.0f} tok/s)" for p, s in rungs)
            if slow:
                found["state"], found["detail"] = "failed", "; ".join(slow) + ": " + found["detail"]
    if "32k" in extras and due("prompt_32k"):
        with report.scenario(prefix + "prompt_32k") as found:
            data, seconds = chat.chat(w(sizes.long - 512), 8)
            prompt = data["usage"]["prompt_tokens"]
            found["detail"] = f"{prompt:,} prompt tokens in {seconds:.1f} s ({prompt / seconds:,.0f} tok/s)"
    if "cancel" in extras and due("cancel_in_prefill"):
        with report.scenario(prefix + "cancel_in_prefill") as found:
            _, alone = chat.chat("Say yes.", 4)
            # Two texts no earlier request began with: the cancelled one is not the uncancelled one, which the prompt
            # cache keeps, so its prefill is computed and the cancel lands inside it.
            _, whole = chat.chat("Whole: " + w(sizes.cancel), 1)   # the long prompt's prefill, uncancelled
            chat.cancelled_stream("Cancelled: " + w(sizes.cancel), min(CANCEL_AFTER_S, whole / 2))
            _, after = chat.chat("Say yes.", 4)
            settle = after - alone   # what the request after the cancel waited for it to settle
            found["detail"] = (f"cancelled {min(CANCEL_AFTER_S, whole / 2):.1f} s into a {whole:.1f} s prefill; the "
                               f"next request waited {settle:.2f} s for it to settle (grace {grace:.0f} s; alone it "
                               f"takes {alone:.2f} s)")
            if settle > grace:
                found["state"] = "failed"
    if "smoke" in extras:
        from crit_smoke import Client, run
        began = time.perf_counter()
        for item in run(Client(f"http://127.0.0.1:{port}", timeout=120), log=lambda line: None):
            name = "smoke_" + item["name"]
            state, detail = item["state"], item.get("detail", "")
            if cut and name in CUT_MODEL_GATES and state == "failed":
                state, detail = "skipped", "left out: " + CUT_MODEL_GATES[name] + " (" + detail[:300] + ")"
            report.add(prefix + name, state, item.get("seconds", 0.0), detail)
        report.log(f"smoke at the run's context: {time.perf_counter() - began:.0f} s")
    if "drafting" in extras and due("drafting_draft_layer"):
        with report.scenario(prefix + "drafting_draft_layer") as found:
            prompt = "Write the numbers from one to twenty, separated by commas."
            drafted, _ = chat.chat(prompt, 48)
            facts = (served.current[0].spec().get("facts") or {}) if served.current else {}
            found["detail"] = f"greedy reply {drafted['choices'][0]['message']['content'][:60]!r}; drafter facts " \
                              f"{json.dumps(facts.get('drafter') or facts.get('mtp') or {})[:300]}"
        with report.scenario(prefix + "drafting_drafter_seam") as found:
            variant = json.loads(json.dumps(served_configuration(served)))
            variant["plugins"]["use"]["drafter"] = "mtp-decode"
            variant["plugins"]["options"]["mtp-decode"] = {"override": 1}
            _, _, port2 = served.open(configuration=variant)
            seam = Chat(port2)
            found_reply, _ = seam.chat(prompt, 48)
            _, _, port = served.open()   # back to the candidate's own drafter
            chat.port = port
            same = found_reply["choices"][0]["message"]["content"] == drafted["choices"][0]["message"]["content"]
            found["detail"] = ("mtp-decode (override 1) over the draft layer gives the greedy reply of "
                               "greedy-head-drafter" if same else
                               f"mtp-decode's greedy reply differs: {found_reply['choices'][0]['message']['content'][:80]!r}")
            if not same:
                found["state"] = "failed"
    if "arithmetic" in extras:
        # Where the CPU tier computes decode misses (forced here), decode rows take its declared order, and prompt
        # rows too at the default level (exact); an engine that asks row-exact prompt rows gets the faster tensor
        # arithmetic for them: each way's decode-against-prefill drift and its prefill rate.
        from crit_smoke import Client, run
        base = served_configuration(served)
        for arithmetic in ("tensor", "declared"):
            with report.scenario(prefix + f"prompt_arithmetic_{arithmetic}") as found:
                variant = json.loads(json.dumps(base))
                variant["plugins"]["options"].setdefault("expert-offload", {}).update(
                    misses="cpu", prompt_arithmetic=arithmetic)   # the CPU tier computes decode misses
                if arithmetic == "tensor":   # its prompt rows are not the decode rows' bits: the level that allows it
                    variant["exactness"] = {**(variant.get("exactness") or {}), "prompt": "row-exact"}
                _, _, port2 = served.open(configuration=variant)
                seam = Chat(port2)
                seam.chat("warm " + w(sizes.long - 512), 1)   # another text: no prompt cache serves the next
                data, seconds = seam.chat(f"{arithmetic} " + w(sizes.long - 512), 1)
                rate = data["usage"]["prompt_tokens"] / seconds
                items = {i["name"]: i for i in run(Client(f"http://127.0.0.1:{port2}", timeout=120),
                                                   only=["decode_matches_prefill", "decode_long"],
                                                   log=lambda line: None)}
                engine = served.current[0] if served.current else None
                offload = ((engine.telemetry() if engine else {}) or {}).get("expert_offload") or {}
                found["detail"] = (f"misses {offload.get('misses')}; 32k prefill {rate:,.0f} tok/s; " + "; ".join(
                    f"{n} {i['state']} ({i.get('detail', '')[:140]})" for n, i in items.items()))
                if any(i["state"] == "failed" for i in items.values()):
                    found["state"] = "failed"
        _, _, port = served.open()
        chat.port = port
    if "lower" in extras and due(f"reopen_at_{sizes.lower // 1024}k"):
        with report.scenario(prefix + f"reopen_at_{sizes.lower // 1024}k") as found:
            before = reserved_ranges(served)
            closes = len(served.closes)
            _, _, port2 = served.open(sizes.lower)
            data, seconds = Chat(port2).chat(w(min(sizes.long, sizes.lower // 2) - 512), 8)
            _, _, port = served.open()
            chat.port = port
            after = reserved_ranges(served)
            found["detail"] = (f"reopened at {sizes.lower:,} tokens, a {data['usage']['prompt_tokens']:,}-token "
                               f"prompt in {seconds:.1f} s; the device's elastic ranges "
                               f"{sum(before.values()) >> 20:,} MiB before, {sum(after.values()) >> 20:,} MiB after "
                               "reopening at the first context")
            # The engine reopened as it was holds the ranges it held: more means a closed engine's ranges outlived it
            # (an engine build's pool leak: its plugins never closed their pools, the device's registry kept them).
            if not before:   # the engine reports no elastic range: nothing to compare, not a pass
                found["state"] = "failed"
                found["detail"] += "; the engine's telemetry names no elastic range (memory.elastic)"
            elif sum(after.values()) > sum(before.values()):
                found["state"] = "failed"
                found["detail"] += "; left by the closed engines: " + ", ".join(sorted(set(after) - set(before)))
            judge_closes(found, served.closes[closes:])
    if "same" in extras and due("reopen_same_context"):
        # The engine closed and opened again at the context it serves, in the same host: what a reload or an
        # ablation's variant does. Core measures the close (crit_plugins.lifetime) and refuses the reload beside a
        # close that kept memory; a reload that fits only because the context is lower cannot hide one here.
        with report.scenario(prefix + "reopen_same_context") as found:
            closes = len(served.closes)
            tokens = served.opened_at
            began_at = time.perf_counter()
            _, _, port = served.open(tokens)
            chat.port = port
            data, seconds = chat.chat(w(512), 8)
            found["detail"] = (f"closed and reopened at {tokens:,} tokens in {time.perf_counter() - began_at:.1f} s, "
                               f"a {data['usage']['prompt_tokens']:,}-token prompt in {seconds:.1f} s")
            judge_closes(found, served.closes[closes:])
    if "concurrency" in extras and due("concurrency"):
        # The engine's lanes at once: each request decodes what it decodes alone (Core's batching contract).
        with report.scenario(prefix + "concurrency") as found:
            lanes = lanes_of(served)
            prompts = [f"Request {i}: list the numbers from {i} to {i + 9}, separated by commas." for i in range(lanes)]
            alone = [content(chat.chat(text, 24)[0]) for text in prompts]
            from concurrent.futures import ThreadPoolExecutor
            with ThreadPoolExecutor(lanes) as pool:
                together = list(pool.map(lambda text: content(Chat(chat.port).chat(text, 24)[0]), prompts))
            differ = [i for i, (a, b) in enumerate(zip(alone, together)) if a != b]
            found["detail"] = (f"{lanes} requests at once (the engine's lanes) decode what each decodes alone"
                               if not differ else
                               f"{len(differ)} of {lanes} requests at once decode otherwise than alone: request "
                               f"{differ[0]} {together[differ[0]][:60]!r} against {alone[differ[0]][:60]!r}")
            if lanes < 2:
                found["state"], found["detail"] = "skipped", "the engine runs one request at a time"
            elif differ:
                found["state"] = "failed"
    if "followup" in extras and due("prompt_cache_followup"):
        # A conversation's next turn begins as the ended one did: the prompt cache serves what they share (a pool that
        # keeps only a sliding window's ring of the ended sequence serves none of it).
        with report.scenario(prefix + "prompt_cache_followup") as found:
            first = [{"role": "user", "content": "Follow-up: " + w(sizes.followup)}]
            data, seconds = chat.messages(first, 8)
            turn = first + [{"role": "assistant", "content": content(data)},
                            {"role": "user", "content": "And once more, briefly."}]
            again, after = chat.messages(turn, 8)
            prompt = data["usage"]["prompt_tokens"]
            cached = (again["usage"].get("prompt_tokens_details") or {}).get("cached_tokens")
            found["detail"] = (f"first turn {prompt:,} prompt tokens in {seconds:.2f} s; the follow-up's "
                               f"{again['usage']['prompt_tokens']:,} in {after:.2f} s, "
                               f"{cached if cached is not None else 'none'} of them cached")
            if cached is None or cached < FOLLOWUP_SHARE * prompt:
                found["state"] = "failed"
                found["detail"] += (f": the prompt cache served under {FOLLOWUP_SHARE:.0%} of the turn the follow-up "
                                    "begins with")
    if "companion" in extras and due("drafting_companion"):
        # The run's companion drafter (EAGLE-3, DFlash, DSpark) through the library's drafter of it: drafting only
        # proposes, so the greedy reply is the one without it.
        with report.scenario(prefix + "drafting_companion") as found:
            from .candidate import COMPANION_DRAFTER, DEVICE_DEFAULTS
            prompt = "Write the numbers from one to twenty, separated by commas."
            plain = content(chat.chat(prompt, 48)[0])
            variant = json.loads(json.dumps(served_configuration(served)))
            drafter = COMPANION_DRAFTER
            if companion is not None:   # the model plugin's own draft block, from the companion's folder
                drafter = DEVICE_DEFAULTS["drafter"]
                own = variant["plugins"].setdefault("options", {}).setdefault(variant["plugins"]["use"]["model"], {})
                own.update({"mtp": True, "draft_folder": str(companion)})
            variant["plugins"]["use"]["drafter"] = drafter
            try:
                _, _, port2 = served.open(configuration=variant)
                drafted = content(Chat(port2).chat(prompt, 48)[0])
            finally:
                _, _, port = served.open()   # back to the candidate as built
                chat.port = port
            found["detail"] = (f"{drafter} gives the greedy reply without it" if drafted == plain else
                               f"{drafter}'s greedy reply differs: {drafted[:80]!r} against {plain[:80]!r}")
            if companion is not None:
                found["detail"] += f" (the model's draft head from {companion})"
            if drafted != plain:
                found["state"] = "failed"
    return chat.port


def lanes_of(served):
    """The requests Core's runtime runs at once on the served engine (its facts' max_concurrent_requests), 1 where it
    states none."""
    engine = served.current[0] if served.current else None
    facts = ((engine.spec() if engine is not None else {}) or {}).get("facts") or {}
    return max(1, int(facts.get("max_concurrent_requests") or 1))


def judge_closes(found, closes):
    """A reopen scenario's word on the engine closes it made (crit serve's last_close, Served.closes): failed where a
    close did not return its memory, its report in the detail; failed too where none was judged, since a reopen
    scenario exists to see that."""
    kept = [close for close in closes if close.get("returned") is False]
    if kept:
        found["state"] = "failed"
        found["detail"] += "; the closed engine did not return its memory: " + str(kept[0].get("message"))
    elif any(close.get("returned") is True for close in closes):
        found["detail"] += f"; {len(closes)} close(s) returned their memory (crit_plugins.lifetime)"
    else:
        found["state"] = "failed"
        found["detail"] += ("; no close was judged: " + str(closes[-1].get("message")) if closes else
                            "; the host reported no engine close")


def reserved_ranges(served):
    """The served engine's elastic ranges: owner/name -> its reserved bytes, from its telemetry (memory.elastic, its
    resident host's crit_plugins.memory report). Not this process's: the engine runs in crit serve's host."""
    engine = served.current[0] if served.current else None
    elastic = (((engine.telemetry() if engine is not None else {}) or {}).get("memory") or {}).get("elastic") or {}
    return {key: int(value.get("reserved") or 0) for key, value in elastic.items()}


def served_configuration(served):
    from ..run.records import read_json
    folder = served.store.directory(served.run_id) / "candidates" / served.candidate_id
    return read_json(folder / "candidate.json")["configuration"]


def precheck_spread(report, prefix, served, store, run_id, data, port, count):
    """crit-eval's gate pre-check `count` times on one load: each run's items, and whether they agree."""
    from ..run.records import load_run
    from ..run.served import _precheck
    run = load_run(store.directory(run_id))
    runs = []
    for number in range(count):
        began = time.perf_counter()
        items = _precheck(served, run, store.directory(run_id), data, port, log=lambda line: None)
        runs.append(items)
        for item in items:
            report.add(f"{prefix}precheck#{number + 1}/{item['name']}", item["state"], time.perf_counter() - began,
                       item.get("detail", ""))
        _, _, port = served.open()
    return port, runs


def spread_item(report, prefix, runs):
    """Whether the pre-check's runs on one load agree, item by item (crit run serve --check's own run last)."""
    names = sorted({item["name"] for items in runs for item in items})
    spread = []
    for name in names:
        found = [next((i for i in items if i["name"] == name), {"state": "missing"}) for items in runs]
        if len({i["state"] for i in found}) > 1:
            spread.append(f"{name}: " + ", ".join(i["state"] + (f" ({i.get('detail', '')[:80]})"
                                                                if i["state"] == "failed" else "") for i in found))
    report.add(prefix + "precheck_spread", "failed" if spread else "passed", 0.0,
               "; ".join(spread) if spread else f"{len(runs)} runs agree on every item")


@contextlib.contextmanager
def environment(**values):
    before = {k: os.environ.get(k) for k in values}
    os.environ.update({k: v for k, v in values.items() if v is not None})
    for k, v in values.items():
        if v is None:
            os.environ.pop(k, None)
    try:
        yield
    finally:
        for k, v in before.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


# -- one load ------------------------------------------------------------------------------------------------------------
def load(report, store, run_id, candidate_id, prefix, extras, sizes, data=None, plugin_tests=False, cut=True,
         leave=None, plugin_cache=None, refusal=None, budget=None, smoke=None, companion=None):
    """One load: crit run serve --check of the run's candidate, the in-load scenarios in its ready hook. Its serving
    checks join the report, named PREFIX/ITEM. leave: the device bytes a VRAM ballast leaves free (None: no ballast);
    plugin_cache: the plugin cache's folder ('' for none, None for the run's own); refusal: a text a refused load
    names where refusing it is the right outcome (a pressure too tight for any cache). budget: in_load's; smoke:
    whether serve --check runs crit smoke (default: unless the load runs it itself, "smoke" in extras); companion:
    in_load's. Returns the checks' result."""
    from ..run import external
    from ..run.served import serve
    captured = {}
    wrapped = external.serving

    def serving(*args, **kwargs):
        captured["served"] = found = wrapped(*args, **kwargs)
        return found

    def ready(port):
        served = captured["served"]
        report.add(prefix + "ready", "info", time.perf_counter() - began,
                   "the engine answered this long after crit run serve --check began (its plugin tests, then the load)")
        port = in_load(report, prefix, served, port, extras, sizes, cut, budget, companion)
        if "spread" in extras and data is not None:
            captured["prechecks"] = precheck_spread(report, prefix, served, store, run_id, data, port,
                                                    PRECHECKS - 1)[1]

    began = time.perf_counter()
    external.serving = serving
    try:
        with environment(CRIT_VRAM_BALLAST=f"leave:{leave}" if leave else None, CRIT_PLUGIN_CACHE=plugin_cache):
            result = serve(store, run_id, candidate_id, ready=ready, check=True,
                           data=str(data) if data is not None and "precheck" in extras else None,
                           smoke=("smoke" not in extras) if smoke is None else smoke, plugin_tests=plugin_tests)
    except Exception as error:
        refused = refusal is not None and refusal in str(error)
        # Where no cache can stage a prompt the right outcome is a refusal at load naming what it needs.
        report.add(prefix + "load", "passed" if refused else "failed", time.perf_counter() - began,
                   ("refused at load, as it should be: " if refused else "") + f"{type(error).__name__}: {error}")
        if not refused:
            traceback.print_exc()
        return None
    finally:
        external.serving = wrapped
    seconds = time.perf_counter() - began
    report.add(prefix + "load", "passed" if result["attempt"].get("outcome") == "completed" else "failed", seconds,
               f"crit run serve --check: attempt {result['attempt'].get('outcome')}")
    checks = result.get("checks") or {}
    folder = store.directory(run_id) / "candidates" / candidate_id / "logs"
    reports = sorted(folder.glob("serve-check-*.json"), key=lambda p: p.stat().st_mtime)
    items = json.loads(reports[-1].read_text(encoding="utf-8")).get("checks", []) if reports else []
    own = []   # crit run serve --check's own pre-check items: the spread's last run
    for item in items:
        state = item.get("state", "unknown")
        if cut and item["name"].split("@")[0] in CUT_MODEL_GATES and state == "failed":
            state, item = "skipped", {**item, "detail": "left out: " + CUT_MODEL_GATES[item["name"].split("@")[0]]
                                      + " (" + item.get("detail", "")[:300] + ")"}
        if item["name"].startswith("precheck"):
            own.append(item)
            number = len(captured.get("prechecks") or []) + 1
            item = {**item, "name": f"precheck#{number}/" + item["name"] if "prechecks" in captured
                    else "precheck/" + item["name"]}
        report.add(prefix + item["name"], state, item.get("seconds", 0.0), item.get("detail", ""))
    if "prechecks" in captured:
        spread_item(report, prefix, captured["prechecks"] + ([own] if own else []))
    return checks


@contextlib.contextmanager
def gpu(shared, label, log=log):
    """The GPU for one load: the machine's shared GPU lock (`shared`, an flock file other work on the GPU takes too;
    empty for none) and crit's engine lock (crit_core.machine), taken before the load and given back after it, so
    work waiting for the GPU runs between loads."""
    from crit_core.machine import engine_lock
    handle = None
    if shared and os.name == "nt":
        log(f"the GPU lock {shared} is an flock file, which Windows does not take: none taken")
    elif shared:
        import fcntl
        handle = open(shared, "a+")
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            log(f"waiting for the GPU lock {shared}")
            fcntl.flock(handle, fcntl.LOCK_EX)
    try:
        with engine_lock(label, log=log):
            yield
    finally:
        if handle is not None:
            handle.close()   # closing the file releases the flock


def prepare_data(model, data):
    """crit-eval's prepared data for `model` in folder `data`, made once (crit eval prepare: the suite's prompts as the
    model's token ids, no weights read)."""
    from pathlib import Path
    from crit_eval.quality.prepare import PREPARED
    data = Path(data)
    if not (data / PREPARED).is_file():
        data.mkdir(parents=True, exist_ok=True)
        from crit_eval.cli import main as eval_main
        code = eval_main(["prepare", "--model", str(model), "--data", str(data)])
        if code:
            raise RuntimeError(f"crit eval prepare exited {code}")
    return data
