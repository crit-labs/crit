"""Check records crit writes from its own evidence: the gates of a retained Eval result (crit run check --from-eval)
and the standard serving checks against the packaged candidate (crit run serve --check).

From Eval, each of the run's fixed gates is judged by the Eval gates its criteria name: arithmetic by quality and
sampling; functional by coverage, repeatable, request_history, prompt_cache, retrieval, lower_context and concurrency
(and exactness when the run requires it); resources by capacity and the device memory left free at the required
context against the run's VRAM headroom. One check per case and gate cites the retained result.
"""
import http.client
import json
import socket
import threading
import time

from crit_core.serving import (LOGPROBS_CAPABILITY, TEMPLATE_KWARGS_CAPABILITY, THINKING_BUDGET_CAPABILITY,
                                TOOL_CHOICE_CAPABILITY, TOOLS_CAPABILITY)

from .records import digest, file_sha, ident, load_run, now, quality_mode, quality_terms, read_json, safe_path

GATES = {"arithmetic": ("quality", "sampling"),
         "functional": ("coverage", "repeatable", "request_history", "prompt_cache", "retrieval", "lower_context",
                        "concurrency")}


def eval_results(folder):
    """The candidate's Eval results whose attempt completed: an attempt that stopped keeps what it measured beside
    its journal, but its result judges no gate (crit run ... eval --resume finishes it)."""
    found = []
    for path in sorted((folder / "eval").glob("attempt-*.json")):
        outcome = folder / "attempts" / path.stem / "result.json"
        if outcome.is_file() and read_json(outcome).get("outcome") == "completed":
            found.append(path)
    return found


def from_eval(store, run_id, candidate_id, attempt=None):
    """Record the arithmetic, functional and resources checks of every case from the candidate's Eval result
    (`attempt`, default its newest). Returns the recorded checks."""
    from .store import amendments
    root = store.directory(run_id)
    run = load_run(root)
    folder = safe_path(root, f"candidates/{ident(candidate_id, 'candidate')}")
    manifest = folder / "package" / "manifest.json"
    if not manifest.is_file():
        raise ValueError("Package the candidate and run crit run ... eval before recording checks from it")
    results = eval_results(folder)
    if attempt is not None:
        results = [p for p in results if p.stem == ident(attempt, "attempt")]
    if not results:
        raise ValueError(f"{candidate_id} has no retained Eval result" + (f" for {attempt}" if attempt else "")
                         + ": run crit run ... eval first")
    path = results[-1]
    reservation = read_json(folder / "attempts" / path.stem / "reservation.json")
    package_sha = digest(read_json(manifest))
    if reservation.get("package_sha256") != package_sha:
        raise ValueError(f"{path.relative_to(root).as_posix()} measured another package")
    result = read_json(path)
    # The terms in force now, every resume's amendment applied: a result from before an amendment is stale.
    latest = max(p.stem for p in (root / "segments").glob("segment-*.json"))
    budget, exactness = quality_terms(run["spec"]["correctness"], amendments(root, latest))
    mode = quality_mode(run["spec"]["correctness"])
    required = result.get("requirements") or {}
    # The mode too: two modes may share a budget, but not their forks, honesty bound and prefill limits. A result
    # that states no mode is refused with the rest.
    if (required.get("quality_budget") != budget or required.get("exactness") != exactness
            or required.get("quality_mode") != mode):
        raise ValueError(f"{path.relative_to(root).as_posix()} ran under quality terms {required}, the run's are "
                         f"{{'quality_budget': {budget}, 'exactness': '{exactness}', 'quality_mode': '{mode}'}}: "
                         "evaluate the candidate again")
    gates = {g["name"]: g for g in (result.get("status") or {}).get("gates", [])}
    verdicts = {}
    for gate, names in GATES.items():
        names = names + (("exactness",) if gate == "functional" and exactness == "required" else ())
        verdicts[gate] = _judge([gates.get(n) or {"name": n, "state": "unmeasured", "detail": "not in the result"}
                                 for n in names])
    free = ((result.get("context") or {}).get("device_memory") or {}).get("free_bytes")
    headroom = run["spec"]["hardware_constraints"]["vram_headroom_bytes"]
    memory = ({"name": "device_memory", "state": "unmeasured", "detail": "the result holds no device memory sample"}
              if free is None else
              {"name": "device_memory", "state": "pass" if free >= headroom else "fail",
               "detail": f"{free / 2**30:.2f} GiB free at the required context, the run's VRAM headroom is "
                         f"{headroom / 2**30:.2f} GiB (host memory: crit run measure)"})
    verdicts["resources"] = _judge([gates.get("capacity") or {"name": "capacity", "state": "unmeasured",
                                                              "detail": "not in the result"}, memory])
    evidence = [{"path": path.relative_to(root).as_posix(), "sha256": file_sha(path)}]
    recorded = []
    for case in run["spec"]["cases"]:
        for gate, (status, rationale) in verdicts.items():
            recorded.append(store.append(run_id, candidate_id, "check", {
                "case_id": case["case_id"], "package_sha256": package_sha, "gate": gate,
                "criteria_sha256": digest(run["spec"]["correctness"]), "status": status,
                "rationale": f"crit Eval {path.stem}: {rationale}"[:8000], "evidence": evidence,
                "evidence_level": "artifact_verified", "recorded_at": now(),
                "extensions": {"recorded_by": "crit run check --from-eval"}}))
    return recorded


def _judge(items):
    """(status, rationale) of gate items in Eval's states: failed if any failed, unverified if any unmeasured."""
    states = {i["state"] for i in items}
    status = "failed" if "fail" in states else "passed" if states == {"pass"} else "unverified"
    return status, "; ".join(f"{i['name']} {i['state']}: {i.get('detail', '')}".rstrip(": ") for i in items)


# ------------------------------------------------------------------------------------------- serving checks

QUESTION = [{"role": "user", "content": "Name three rivers in Europe and give one fact about each."}]
LONG = [{"role": "user", "content": "Write a long, detailed essay about the history of European rivers."}]
TOOLS = [{"type": "function", "function": {"name": "get_weather", "description": "The current weather in a city",
                                           "parameters": {"type": "object", "properties": {"city": {"type": "string"}},
                                                          "required": ["city"]}}}]
# Checks a served engine must pass; the others depend on what the model chooses or declares.
REQUIRED = ("consecutive", "system_message", "streaming", "stop_string", "cancel", "stop_engine")
# The reasoning a check's chat request may write on a package that ends reasoning at a budget, so its reply is written.
CHECK_THINKING_TOKENS = 16


class Client:
    def __init__(self, port, timeout, defaults=None):
        self.port, self.timeout = port, timeout
        self.connection = None   # chat's connection while it waits, for abort
        self.defaults = {"temperature": 0, "max_tokens": 48, **(defaults or {})}   # every chat request's

    def chat(self, body):
        connection = http.client.HTTPConnection("127.0.0.1", self.port, timeout=self.timeout)
        self.connection = connection
        try:
            connection.request("POST", "/v1/chat/completions", json.dumps(self._body(body)),
                               {"Content-Type": "application/json"})
            response = connection.getresponse()
            data = json.loads(response.read() or b"{}")
        finally:
            connection.close()
        if response.status != 200:
            raise ValueError(f"crit serve answered {response.status}: {(data.get('error') or {}).get('message', data)}")
        return data

    def _body(self, body):
        """The request: the defaults under `body`; no reasoning budget with logprobs, a response_format or a forced call, which crit serve
        refuses one with."""
        found = {**self.defaults, **body}
        if found.get("logprobs") or found.get("response_format") or found.get("tool_choice") not in (None, "auto", "none"):
            found.pop("thinking_budget_tokens", None)
        return found

    def abort(self):
        """Close the connection chat waits on (from another thread): crit serve cancels a request whose client
        disconnects, and chat raises."""
        connection = self.connection
        if connection is not None and connection.sock is not None:
            try:
                connection.sock.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass

    def post(self, path, body):
        """POST a JSON body to crit serve; its JSON answer, ValueError for any status but 200."""
        connection = http.client.HTTPConnection("127.0.0.1", self.port, timeout=self.timeout)
        try:
            connection.request("POST", path, json.dumps(body), {"Content-Type": "application/json"})
            response = connection.getresponse()
            data = json.loads(response.read() or b"{}")
        finally:
            connection.close()
        if response.status != 200:
            raise ValueError(f"crit serve answered {response.status}: {(data.get('error') or {}).get('message', data)}")
        return data

    def models(self):
        connection = http.client.HTTPConnection("127.0.0.1", self.port, timeout=self.timeout)
        try:
            connection.request("GET", "/v1/models")
            response = connection.getresponse()
            data = json.loads(response.read() or b"{}")
        finally:
            connection.close()
        return data.get("data") or []

    def stream(self, body, cancel_after_first=False):
        """The streamed content and finish reason; with cancel_after_first, the client disconnects after the first
        content piece."""
        connection = http.client.HTTPConnection("127.0.0.1", self.port, timeout=self.timeout)
        content, finish = [], None
        try:
            connection.request("POST", "/v1/chat/completions",
                               json.dumps({**self._body(body), "stream": True}),
                               {"Content-Type": "application/json"})
            response = connection.getresponse()
            if response.status != 200:
                raise ValueError(f"crit serve answered {response.status}: {response.read()[:500]!r}")
            for line in response:
                if not line.startswith(b"data: "):
                    continue
                data = line[6:].strip()
                if data == b"[DONE]":
                    break
                event = json.loads(data)
                if "error" in event:
                    raise ValueError("crit serve failed the request: " + str(event["error"].get("message")))
                for choice in event.get("choices") or []:
                    piece = (choice.get("delta") or {}).get("content")
                    content.append(piece or "")
                    finish = choice.get("finish_reason") or finish
                    if cancel_after_first and piece:
                        response.close()  # the client goes away mid-reply
                        return "".join(content), "cancelled"
        finally:
            connection.close()
        return "".join(content), finish


def _reply(data):
    choice = data["choices"][0]
    return choice["message"].get("content") or "", choice.get("finish_reason")


# Words the full-context prompt repeats: ordinary prose, so every prompt chunk is ordinary work.
FILLER = ("The river runs past the old mill and under the stone bridge, where the town keeps its market on "
          "Saturdays. Farmers bring apples, cheese and bread, and the children watch the boats go by. ")
# Tokens the full-context pass decodes after its prompt.
FULL_CONTEXT_DECODE = 16
# Seconds a serving check waits for its request; the full-context request, for the engine's next step.
CHECK_SECONDS = 600
# Seconds between the full-context wait's reads of the engine's step count.
PROGRESS_POLL_SECONDS = 5.0
# Seconds of prompt steps the full-context wait times before it projects the prefill, and between its progress lines.
PREFILL_SETTLE_SECONDS = 30.0
PREFILL_REPORT_SECONDS = 60.0


class PrefillTooSlow(Exception):
    """At the rate its first prompt steps ran, the full-context prompt cannot finish before the check's deadline."""


def full_context_checks(port, engine, kv_tokens, headroom, grace, timeout, log, left=None):
    """One chat request at the full context: a user message of repeated FILLER, as many copies as fit in kv_tokens
    less FULL_CONTEXT_DECODE tokens (sized from the prompt tokens the engine counts for one copy and two), then up to
    FULL_CONTEXT_DECODE decoded tokens, on an engine whose Core steps are probed (crit_core.native.STEP_PROBE).
    That request takes as long as the engine's prefill of the whole context, which on a slow link or a small device
    can be many times `timeout`: it waits while the engine keeps finishing steps (_progressing) and fails when
    none finishes for `timeout` seconds, crit serve's own bound on it being the attempt's deadline. With `left`
    (() -> the seconds the attempt has left) it also fails as soon as the prefill cannot end in them at the rate the
    engine's first prompt steps ran (_prefill_watch: "prefill too slow, about N tok/s", after PREFILL_SETTLE_SECONDS
    of prompt steps instead of at the deadline), and logs the rate while it waits.
    From Core's probe in the engine's telemetry, of this request's steps alone (_since): full_context_prompt_step
    passes when the slowest prompt step finishes within `grace` (a cancel settles only between steps, and the host is
    retired with its weights when a cancel does not settle within crit serve's grace), full_context_device_memory
    when the least free device memory at any probed step's peak, prompt or decode, is at least `headroom` (the run's
    vram_headroom_bytes), each step's reading taken as if the memory outside the engine had stayed at the least it
    held during the request (_device_memory). An engine that reports no probe (its steps not on Core's StepEngine, or no CUDA for
    memory) leaves the item unverified."""
    client = Client(port, timeout)
    names = ("full_context_prompt_step", "full_context_device_memory")
    started = time.perf_counter()
    try:
        model = (client.models() or [{}])[0].get("id")

        def complete(copies, tokens):
            return client.chat({"model": model, "messages": [{"role": "user", "content": FILLER * copies}],
                                "max_tokens": tokens})["usage"]["prompt_tokens"]
        # The filler's tokens per copy, from what the engine counts for one copy and two: any tokenizer, no
        # tokenize call.
        one = complete(1, 8)
        per = complete(2, 8) - one
        if per <= 0:
            raise ValueError(f"two copies of the filler counted {one + per} prompt tokens, one {one}")
        copies = 1 + (kv_tokens - FULL_CONTEXT_DECODE - one) // per
        expected = one + (copies - 1) * per
        log(f"serve check full context: a prompt of about {expected:,} tokens and up to "
            f"{FULL_CONTEXT_DECODE} decoded tokens")
        core = (engine.telemetry() or {}).get("core") or {}   # before the request: its probe and prompt tokens
        before = core.get("probe")
        waiting = Client(port, None)   # bounded by the engine's steps instead (_progressing)
        body = {"model": model, "messages": [{"role": "user", "content": FILLER * copies}],
                "max_tokens": FULL_CONTEXT_DECODE}
        watch = _prefill_watch(engine, core.get("prompt_tokens") or 0, expected, left, log)
        prompted = _progressing(lambda: waiting.chat(body), waiting, engine, timeout,
                                watch=watch)["usage"]["prompt_tokens"]
        probe = _since(before, _probe(engine))
    except Exception as exc:
        detail = f"{type(exc).__name__}: {exc}"[:2000]
        items = [{"name": name, "state": "failed", "detail": detail} for name in names]
    else:
        items = [_prompt_step(probe, grace, prompted), _device_memory(probe, headroom)]
    seconds = round(time.perf_counter() - started, 3)
    for item in items:
        item["seconds"] = seconds
        log(f"serve check {item['name']}: {item['state']} ({item['detail']})")
    return items


def _prefill_watch(engine, start, expected, left, log, settle=None, report=None, clock=time.perf_counter):
    """The full-context wait's look at the prefill, called at every poll: from Core's count of prefilled prompt tokens
    (its telemetry's core.prompt_tokens, `start` before the request) it times the prompt steps after the first one it
    sees finish, and once they span `settle` seconds projects the rest of the `expected`-token prompt at that rate.
    The projection is a floor (prefill slows as the context grows), so when it already passes the seconds `left()`
    gives, the prompt cannot finish: PrefillTooSlow, with the rate. It logs the rate every `report` seconds. An engine
    that reports no count is not judged."""
    settle = PREFILL_SETTLE_SECONDS if settle is None else settle
    report = PREFILL_REPORT_SECONDS if report is None else report
    first = {}

    def watch():
        counted = _prompt_tokens(engine)
        if counted is None or counted <= start:
            return
        now, done = clock(), counted - start
        if not first:
            first.update(at=now, done=done, said=now)
            return
        if done <= first["done"] or now - first["at"] < settle:
            return
        rate = (done - first["done"]) / (now - first["at"])
        seconds = max(0, expected - done) / rate
        if left is not None and seconds > left():
            raise PrefillTooSlow(
                f"prefill too slow, about {rate:,.0f} tok/s: {done:,} of about {expected:,} prompt tokens are in, "
                f"the rest needs {_minutes(seconds)} or more at that rate (prefill slows as the context grows) and "
                f"the check's reservation has {_minutes(max(0.0, left()))} left. Make prefill faster (larger prompt "
                "steps within the cancel grace, fewer copies a step) before checking again")
        if now - first["said"] >= report:
            first["said"] = now
            log(f"serve check full context: {done:,} of about {expected:,} prompt tokens, about {rate:,.0f} tok/s, "
                f"{_minutes(seconds)} or more to go")
    return watch


def _minutes(seconds):
    return f"{seconds / 60:.0f} min" if seconds >= 90 else f"{seconds:.0f} s"


def _prompt_tokens(engine):
    """The prompt tokens Core's steps prefilled so far (telemetry's core.prompt_tokens), None when the engine
    reports none or its report fails."""
    try:
        return ((engine.telemetry() or {}).get("core") or {}).get("prompt_tokens")
    except Exception:
        return None


def _progressing(call, client, engine, stall, poll=None, watch=None):
    """call()'s result, run on a thread and waited for while the engine keeps stepping: Core's step count (its
    telemetry's core.steps, prompt and decode steps alike) read every `poll` seconds. When it has not moved for `stall`
    seconds, or for an engine that reports none `stall` seconds have passed, the client's connection is closed (crit
    serve cancels the request) and TimeoutError raised; so it is when `watch()`, called at every poll, raises, and
    that error is raised. call()'s own error is raised as it is."""
    poll = PROGRESS_POLL_SECONDS if poll is None else poll
    found = {}

    def run():
        try:
            found["value"] = call()
        except BaseException as exc:   # re-raised on the caller's thread
            found["error"] = exc
    thread = threading.Thread(target=run, name="crit full-context request", daemon=True)
    thread.start()
    steps, moved = None, time.perf_counter()
    while True:
        thread.join(poll)
        if not thread.is_alive():
            break
        if watch is not None:
            try:
                watch()
            except Exception:
                client.abort()
                thread.join(poll)
                raise
        counted = _steps(engine)
        if counted is not None and counted != steps:
            steps, moved = counted, time.perf_counter()
        elif time.perf_counter() - moved >= stall:
            client.abort()
            thread.join(poll)
            raise TimeoutError(f"no engine step finished in {stall:g} s" if steps is not None else
                               f"timed out after {stall:g} s (the engine reports no step count)")
    if "error" in found:
        raise found["error"]
    return found["value"]


def _steps(engine):
    """Core's steps so far (telemetry's core.steps), None when the engine reports none or its report fails."""
    try:
        return ((engine.telemetry() or {}).get("core") or {}).get("steps")
    except Exception:
        return None


def _probe(engine):
    return ((engine.telemetry() or {}).get("core") or {}).get("probe")


def _since(before, after):
    """Core's probe of the steps between two telemetry reads, per kind: their count, the slowest's seconds, the
    least free device memory and each step's (free, outside) memory, from the samples Core keeps (its recent ones;
    outside None from a Core that does not sample it). None when the engine reports no probe."""
    if after is None:
        return None
    found = {}
    for kind, value in after.items():
        steps = value["steps"] - ((before or {}).get(kind) or {}).get("steps", 0)
        samples = value.get("recent", [])[-steps:] if steps > 0 else []
        if not samples:
            continue
        found[kind] = {"steps": steps, "slowest_seconds": max(sample[0] for sample in samples)}
        memory = [(sample[1], sample[2] if len(sample) > 2 else None) for sample in samples if sample[1] is not None]
        if memory:
            found[kind]["min_free_device_bytes"] = min(free for free, _ in memory)
            found[kind]["memory"] = memory
    return found


def _prompt_step(probe, grace, prompted):
    prompt = (probe or {}).get("prompt")
    if not prompt:
        return {"name": "full_context_prompt_step", "state": "unverified",
                "detail": "the engine reports no probed prompt steps (Core's StepEngine probes them)"}
    slowest = prompt["slowest_seconds"]
    return {"name": "full_context_prompt_step", "state": "passed" if slowest < grace else "failed",
            "detail": f"the slowest of {prompt['steps']} prompt steps, up to a {prompted:,}-token prompt, took "
                      f"{slowest:.3f} s; "
                      f"a cancel settles between steps and crit serve retires the host after {grace:g} s"
                      + ("" if slowest < grace else ": prefill in shorter steps (smaller chunks; where a step's "
                         "time is bringing weights to the device, experts kept in host memory, a model plugin "
                         "that runs its prompt spans layer by layer over several steps: model-template's passes)")}


def engine_free(pairs, baseline=()):
    """The engine's own free device memory at each of `pairs` [(free bytes the device read, memory outside the engine
    then or None)]: each reading as if the memory outside the engine (the CUDA driver, other programs on the card:
    crit_plugins.memory's other()) had stayed at the least it held over the readings and `baseline` (more readings of
    it), so another program's allocations do not count against the engine. Returns ([(judged, free, outside)], the
    least outside or None, the most or None); a reading without its outside memory is judged as the device read it."""
    outside = [other for _, other in pairs if other is not None] + [other for other in baseline if other is not None]
    floor = min(outside) if outside else None
    judged = [(free if other is None or floor is None else free + other - floor, free, other) for free, other in pairs]
    return judged, floor, (max(outside) if outside else None)


def _device_memory(probe, headroom):
    """The least free device memory at a probed step's peak against `headroom`, judged by the engine alone: each
    step's free memory as if the memory outside the engine (the CUDA driver, other programs on the card: Core's
    outside reading) had stayed at the least it held during the request, so another program's allocations during a
    step do not count against the engine and the headroom stays the room the engine leaves on top of what others
    hold. A Core that does not sample the outside memory is judged by the device's reading."""
    readings = {kind: value.get("memory") or [(value["min_free_device_bytes"], None)]
                for kind, value in (probe or {}).items() if "min_free_device_bytes" in value}
    if not readings:
        return {"name": "full_context_device_memory", "state": "unverified",
                "detail": "the engine reports no probed device memory (Core's StepEngine on CUDA probes it)"}
    every = [(kind, pair) for kind, pairs in readings.items() for pair in pairs]
    judged, floor, most = engine_free([pair for _, pair in every])
    found = {}
    for (kind, _), reading in zip(every, judged):
        if kind not in found or reading[0] < found[kind][0]:
            found[kind] = reading
    least, free, other = min(found.values(), key=lambda found: found[0])
    shown = ", ".join(f"{kind} {value[0] / 2**30:.2f} GiB" for kind, value in sorted(found.items()))
    netted = ("" if floor is None else
              f"; memory outside the engine (the CUDA driver, other programs) held {floor / 2**20:,.0f} to "
              f"{most / 2**20:,.0f} MiB during the request, and the least-free step's "
              f"{((other or floor) - floor) / 2**20:,.0f} MiB above its least was subtracted from what the step used "
              f"(the device read {free / 2**30:.2f} GiB free there)")
    return {"name": "full_context_device_memory", "state": "passed" if least >= headroom else "failed",
            "detail": f"least free device memory at a step's peak up to the full context: {shown}; the run's VRAM "
                      f"headroom is {headroom / 2**30:.2f} GiB" + netted
                      + ("" if least >= headroom else
                         ": a step took device memory that no plugin of crit_plugins.memory's ledger made room "
                         "for (an elastic cache grows into whatever a smaller KV cache or shorter chunks free, so "
                         "those do not move this; `crit plugins memory PACKAGE --model MODEL` lists what holds "
                         "the device)")}


# The decode rounds a drafting engine's check must have served before its drafts are judged: a working drafter keeps
# some drafts within a few greedy replies to real prompts, and fewer rounds may hold none by chance.
DRAFT_ROUNDS = 32


def core_counts(engine):
    """What Core counted of an engine load's requests, from its telemetry: the kernels compiled once its load was
    over (core.compiles_after_load), its drafting (core.drafts) and its steps' memory (memory.steps: segments
    refused, past the arena, past the headroom, since it served); {} where the engine reports none or its report
    fails."""
    try:
        telemetry = engine.telemetry() or {}
    except Exception:   # an engine that cannot report counts nothing
        return {}
    core = telemetry.get("core") or {}
    found = {key: core[key] for key in ("compiles_after_load", "drafts") if isinstance(core.get(key), dict)}
    steps = (telemetry.get("memory") or {}).get("steps")
    if isinstance(steps, dict):
        found["steps"] = steps
    return found


def core_items(counts, log):
    """The check's items from Core's counters over every load it served (core_counts, one a load):
    compiles_after_load fails where a request compiled a kernel or built an extension (every plugin compiles at
    load: Core's warm requests serve each kind of request first, and a request that compiles waits for it), and
    drafts_accepted fails a drafting engine that kept none of its drafts over at least DRAFT_ROUNDS decode rounds
    (its drafter proposes nothing the model keeps: one that never asks the draft head the model loaded, a head that
    reads the wrong layer). An engine that reports neither leaves them unverified."""
    items = []
    seen = [c["compiles_after_load"] for c in counts if c.get("compiles_after_load")]
    if not seen:
        state, detail = "unverified", "the engine reports no compile watch (Core's loader keeps one)"
    else:
        total = sum(int(c.get("count") or 0) for c in seen)
        named = [what for c in seen for what in c.get("first") or ()]
        state = "passed" if not total else "failed"
        detail = ("no kernel compiled and no extension was built after load, over every request of the check"
                  if not total else
                  f"{total} kernel compiles or extension builds after load, during the check's requests: "
                  + "; ".join(named[:4]) + ". A plugin compiles every kernel its requests launch at load (its "
                  "warm-up, or Core's warm requests: crit_core.engine.warm_requests), never on a request")
    items.append({"name": "compiles_after_load", "state": state, "detail": detail})
    drafts = [c["drafts"] for c in counts if c.get("drafts")]
    rounds = sum(int(d.get("rounds") or 0) for d in drafts)
    accepted = sum(int(d.get("accepted") or 0) for d in drafts)
    known = drafts and all(d.get("drafted") is not None for d in drafts)
    proposed = sum(int(d.get("drafted") or 0) for d in drafts) if known else None
    if not drafts:
        state, detail = "passed", "no request of the check drafted"
    elif rounds < DRAFT_ROUNDS:
        state, detail = "unverified", f"{rounds} drafting decode rounds, fewer than the {DRAFT_ROUNDS} judged"
    elif not accepted:
        state = "failed"
        detail = (f"none of the drafts of {rounds} decode rounds was accepted"
                  + (f" ({proposed} proposed)" if proposed is not None else "") +
                  ": the drafter proposes nothing the model keeps (one that never asks the draft head the model "
                  "loaded, such as mtp-decode's copy drafts alone, or a draft head reading the wrong layer); drafting "
                  "then costs every round and gains none")
    else:
        state = "passed"
        detail = (f"{accepted} drafts accepted over {rounds} decode rounds"
                  + (f" ({accepted / proposed:.0%} of {proposed} proposed)" if proposed else ""))
    items.append({"name": "drafts_accepted", "state": state, "detail": detail})
    items.append(step_memory_item(counts))
    for item in items:
        log(f"serve check {item['name']}: {item['state']} ({item['detail']})")
    return items


def step_memory_item(counts):
    """step_memory, from Core's allocator over every load the check served (core_counts' steps, counted since each
    load began serving): failed where a step's segment was refused (the device had not the bytes: a request failed
    or waited on torch's retry), passed the step arena (Core's load measured no step that large: a step kind its warm
    requests do not reach, or a plugin's temporary that grows with the request) or left less free than the run's
    VRAM headroom, judged by the engine's own use: each segment's free memory as if the memory outside the engine had
    stayed at the least it held over the load (engine_free, from Core's segment_memory), so other programs on the
    card do not move the verdict. A load whose Core reports no such readings (or more segments than it keeps) is
    judged by the device's reading. An engine without Core's allocator (no CUDA, no compiler) leaves it
    unverified."""
    seen = [c["steps"] for c in counts if c.get("steps")]
    if not seen or any(s.get("unaccounted") for s in seen):
        why = next((s["unaccounted"] for s in seen if s.get("unaccounted")), "the engine reports no step memory")
        return {"name": "step_memory", "state": "unverified", "detail": why}
    total = {key: sum(int(s.get(key) or 0) for s in seen) for key in ("steps", "refused", "overflows", "breaches")}
    least, outside = [], []
    raw = total["breaches"]
    for s in seen:
        pairs = [tuple(pair) for pair in s.get("segment_memory") or ()]
        if not pairs or s.get("segment_memory_cut") or "headroom" not in s:
            if s.get("least_free") is not None:
                least.append(s["least_free"])
            continue
        judged, floor, most = engine_free(pairs, [s.get("outside_at_serve")])
        total["breaches"] += sum(1 for value, _, _ in judged if value < int(s["headroom"])) - int(s.get("breaches") or 0)
        least.append(min(value for value, _, _ in judged))
        if floor is not None:
            outside.append((floor, most))
    arena = max(int(s.get("arena") or 0) for s in seen)
    said = (f"{total['steps']} steps, arena {arena / 2**20:.0f} MiB"
            + (f", least free {min(least) / 2**20:.0f} MiB" if least else "")
            + (f"; memory outside the engine held {min(f for f, _ in outside) / 2**20:,.0f} to "
               f"{max(m for _, m in outside) / 2**20:,.0f} MiB, its growth set aside"
               + (f" ({raw} segments left less than the headroom by the device's reading)"
                  if raw != total["breaches"] else "") if outside else ""))
    wrong = [f"{total[key]} {what}" for key, what in (
        ("refused", "segments refused (the device lacked the bytes)"),
        ("overflows", "segments past the step arena (a step larger than any the load measured)"),
        ("breaches", "segments that left less free than the VRAM headroom")) if total[key]]
    if total["overflows"]:   # which kind of step outgrew the arena: the most each kind took, largest first
        most = {}
        for s in seen:
            for kind, took in (s.get("most") or {}).items():
                most[kind] = max(most.get(kind, 0), int(took or 0))
        if most:
            said += "; the most a step took, by kind: " + ", ".join(
                f"{kind} {took / 2**20:,.0f} MiB" for kind, took in sorted(most.items(), key=lambda kv: -kv[1]))
    if wrong:
        return {"name": "step_memory", "state": "failed",
                "detail": "; ".join(wrong) + f" over the check's requests ({said}). Core's arena is what the load's "
                          "largest steps took (crit_plugins.memory size_arena): a step past it is a kind no warm "
                          "request reaches, or a temporary that grows with the request"}
    return {"name": "step_memory", "state": "passed",
            "detail": f"every step's segments within the arena and the headroom ({said})"}


def plan_kv(planned, facts):
    """(state, detail) of plan_kv: whether the engine's KV pools take the bytes a token its plan budgeted (the
    plan's kv_bytes_per_token against the engine facts' own, within 1%)."""
    served = facts.get("kv_bytes_per_token")
    if not served:
        return "failed", (f"the candidate's plan budgets {planned:,} KV bytes a token and the engine's facts declare "
                          "no kv_bytes_per_token: its kv_cache plugin states what its pools take")
    if abs(int(served) - int(planned)) > 0.01 * int(planned):
        return "failed", (f"the engine's KV pools take {int(served):,} bytes a token, the candidate's plan budgeted "
                          f"{planned:,} ({int(served) / int(planned):.2f}x): the plan's device budget (its expert "
                          "cache, its context) rests on the other figure. Plan from the engine's facts (the kv_cache "
                          "plugin's resolution of its KV options: an auto precision, indexer keys, a draft layer's "
                          "pool), or set the KV options that serve what the plan budgets")
    return "passed", f"the engine's KV pools take the {int(served):,} bytes a token its plan budgeted"


def serve_checks(port, capabilities, variables, timeout, log, planned_activations=None, planned_kv=None):
    """Run the standard serving checks against crit serve at `port`; returns their items (name, state, detail)
    without the final stop_engine item, which the caller adds once the engine has stopped. planned_activations: the
    prompt activations the candidate's plan declared (its prompt_activations), which the engine's facts must match
    (plan_activations): a candidate that serves other activations than it planned is caught at its first check, not
    at the next plan after a whole Eval. planned_kv: the plan's kv_bytes_per_token, which the engine's facts must
    give (plan_kv): a plan that budgeted the device for another KV (the protocol's FP8 price where the kv_cache
    plugin resolved BF16 and added indexer keys) is caught there too."""
    client, items = Client(port, timeout, {"thinking_budget_tokens": CHECK_THINKING_TOKENS}
                                if THINKING_BUDGET_CAPABILITY in capabilities else None), []

    def check(name, run):
        started = time.perf_counter()
        try:
            state, detail = run()
        except Exception as exc:
            state, detail = "failed", f"{type(exc).__name__}: {exc}"[:2000]
        items.append({"name": name, "state": state, "detail": detail,
                      "seconds": round(time.perf_counter() - started, 3)})
        log(f"serve check {name}: {state} ({detail})")

    first = {}

    def consecutive():
        first["reply"], _ = _reply(client.chat({"messages": QUESTION}))
        again, _ = _reply(client.chat({"messages": QUESTION}))
        if again != first["reply"]:
            return "failed", f"two greedy requests answered differently: {first['reply'][:80]!r} vs {again[:80]!r}"
        return "passed", f"two greedy requests answered alike ({len(first['reply'])} characters)"
    check("consecutive", consecutive)
    baseline = first.get("reply")

    def system_message():
        reply, _ = _reply(client.chat({"messages": [{"role": "system", "content": "Answer briefly."}] + QUESTION}))
        return "passed", f"a leading system message was accepted ({len(reply)} characters)"
    check("system_message", system_message)

    def streaming():
        content, finish = client.stream({"messages": QUESTION})
        if baseline is not None and content != baseline:
            return "failed", f"the streamed reply {content[:80]!r} differs from the plain one {baseline[:80]!r}"
        return "passed", f"streamed pieces join to the plain reply (finish {finish})"
    check("streaming", streaming)

    def stop_string():
        if not baseline or len(baseline) < 8:
            return "unverified", "the greedy reply is too short to cut at a stop string"
        stop = baseline[len(baseline) // 2:len(baseline) // 2 + 4]
        expected = baseline[:baseline.find(stop)]
        reply, finish = _reply(client.chat({"messages": QUESTION, "stop": [stop]}))
        if reply != expected or finish != "stop":
            return "failed", f"stop {stop!r}: got {reply[-40:]!r} (finish {finish}), expected the reply cut before it"
        return "passed", f"stop {stop!r} cut the reply at {len(expected)} characters with finish stop"
    check("stop_string", stop_string)

    def cancel():
        client.stream({"messages": LONG, "max_tokens": 2048}, cancel_after_first=True)
        reply, _ = _reply(client.chat({"messages": QUESTION}))
        if baseline is not None and reply != baseline:
            return "failed", "after a client disconnected mid-reply, the next greedy reply differs"
        return "passed", "a client disconnecting mid-reply cancelled it; the next request answered as before"
    check("cancel", cancel)

    if TOOLS_CAPABILITY in capabilities:
        forced = TOOL_CHOICE_CAPABILITY in capabilities

        def tool_call():
            question = [{"role": "user", "content": "What is the weather in Lisbon right now? Use the tool."}]
            data = client.chat({"messages": question, "tools": TOOLS, "max_tokens": 256,
                                "tool_choice": "required" if forced else "auto"})
            calls = data["choices"][0]["message"].get("tool_calls") or []
            if not calls:
                return ("failed" if forced else "unverified"), "the model answered without calling the tool"
            call = calls[0]
            arguments = json.loads(call["function"]["arguments"])
            if call["function"]["name"] != "get_weather" or not isinstance(arguments, dict):
                return "failed", f"unexpected call {call['function']}"
            follow = question + [{"role": "assistant", "content": None, "tool_calls": calls},
                                 {"role": "tool", "tool_call_id": call["id"], "content": '{"temperature_c": 18}'}]
            reply, _ = _reply(client.chat({"messages": follow, "tools": TOOLS, "max_tokens": 64}))
            return "passed", f"called get_weather({arguments}); answered the result with {len(reply)} characters"
        check("tool_call", tool_call)

    if LOGPROBS_CAPABILITY in capabilities:
        def logprobs():
            data = client.chat({"messages": QUESTION, "logprobs": True, "top_logprobs": 2})
            rows = (data["choices"][0].get("logprobs") or {}).get("content") or []
            if len(rows) != data["usage"]["completion_tokens"] or any(len(r.get("top_logprobs", [])) != 2 for r in rows):
                return "failed", f"{len(rows)} log-probability rows for {data['usage']['completion_tokens']} tokens"
            return "passed", f"{len(rows)} tokens with their log-probability and 2 alternatives"
        check("logprobs", logprobs)

    seen = {}

    def draft_bytes():
        facts = seen["facts"] = next(iter(client.models()), {}).get("engine_facts") or {}
        if not facts.get("speculative"):
            return "passed", "the engine drafts nothing"
        if not facts.get("draft_bytes_per_step"):
            return "failed", ("the engine drafts but its facts declare no draft_bytes_per_step (the bytes one draft "
                              "step reads: the draft layers and the head they read): crit-eval cannot price the round's "
                              "floor, nor compare the draft step with the checkpoint's draft layers for the closed track")
        return "passed", f"a draft step reads {facts['draft_bytes_per_step']} bytes"
    check("draft_bytes", draft_bytes)
    # The served engine's facts beside the check, for what reads them later (crit kernels sweep --matrices).
    if seen.get("facts"):
        items[-1]["engine_facts"] = seen["facts"]

    if planned_activations is not None:
        def plan_activations():
            from .plans import activations_served
            served = (seen.get("facts") or {}).get("prompt_activations")
            if served is None:
                return "passed", "the engine declares no prompt activations: no quantized format's prompt GEMMs"
            if not activations_served(planned_activations, served):
                return "failed", (f"the engine serves prompt activations {json.dumps(served)}, the candidate's plan "
                                  f"declared {json.dumps(planned_activations)}: set the projections plugin's options "
                                  "to what the plan declares (its facts' prompt_activations say what each option "
                                  "gave), or plan what is served")
            return "passed", f"the engine serves the prompt activations its plan declared ({json.dumps(served)})"
        check("plan_activations", plan_activations)

    if planned_kv is not None:
        check("plan_kv", lambda: plan_kv(planned_kv, seen.get("facts") or {}))

    switches = {k: not v for k, v in variables.items() if isinstance(v, bool)}
    if switches and TEMPLATE_KWARGS_CAPABILITY in capabilities:
        def template_switch():
            client.chat({"messages": QUESTION, "chat_template_kwargs": switches, "max_tokens": 16})
            reply, _ = _reply(client.chat({"messages": QUESTION}))
            if baseline is not None and reply != baseline:
                return "failed", f"after a request with {switches}, the default request answers differently"
            return "passed", f"a request with {switches} served; the next default request answered as before"
        check("template_switch", template_switch)
    return items


def verdict(items):
    """(status, rationale) of the serving checks: failed if any failed, unverified if a required one is. An item
    marked `hint` (crit-eval's gate pre-check: its verdicts are hints, never a gate) is in the rationale, as a hint,
    and leaves the status alone."""
    states = {i["name"]: i["state"] for i in items if not i.get("hint")}
    if "failed" in states.values():
        status = "failed"
    elif all(states.get(name) == "passed" for name in REQUIRED):
        status = "passed"
    else:
        status = "unverified"
    return status, "crit run serve --check: " + "; ".join(
        f"{i['name']} {i['state']}" + (" (hint)" if i.get("hint") else "") for i in items)
