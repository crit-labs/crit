"""Check records crit writes from its own evidence: the gates of a retained Eval result (crit run check --from-eval)
and the standard serving checks against the packaged candidate (crit run serve --check).

From Eval, each of the run's fixed gates is judged by the Eval gates its criteria name: arithmetic by quality and
sampling; functional by coverage, repeatable, request_history, prompt_cache, retrieval, lower_context and concurrency
(and exactness when the run requires it); resources by capacity and the device memory left free at the required
context against the run's VRAM headroom. One check per case and gate cites the retained result.
"""
import http.client
import json
import time

from crit_core.serving import LOGPROBS_CAPABILITY, TEMPLATE_KWARGS_CAPABILITY, TOOL_CHOICE_CAPABILITY, TOOLS_CAPABILITY

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


class Client:
    def __init__(self, port, timeout):
        self.port, self.timeout = port, timeout

    def chat(self, body):
        connection = http.client.HTTPConnection("127.0.0.1", self.port, timeout=self.timeout)
        try:
            connection.request("POST", "/v1/chat/completions", json.dumps({"temperature": 0, "max_tokens": 48, **body}),
                               {"Content-Type": "application/json"})
            response = connection.getresponse()
            data = json.loads(response.read() or b"{}")
        finally:
            connection.close()
        if response.status != 200:
            raise ValueError(f"crit serve answered {response.status}: {(data.get('error') or {}).get('message', data)}")
        return data

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
                               json.dumps({"temperature": 0, "max_tokens": 48, **body, "stream": True}),
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


def full_context_checks(port, engine, kv_tokens, headroom, grace, timeout, log):
    """One chat request at the full context: a user message of repeated FILLER, as many copies as fit in kv_tokens
    less FULL_CONTEXT_DECODE tokens (sized from the prompt tokens the engine counts for one copy and two), then up to
    FULL_CONTEXT_DECODE decoded tokens, on an engine whose Core steps are probed (crit_core.native.STEP_PROBE).
    From Core's probe in the engine's telemetry, of this request's steps alone (_since): full_context_prompt_step
    passes when the slowest prompt step finishes within `grace` (a cancel settles only between steps, and the host is
    retired with its weights when a cancel does not settle within crit serve's grace), full_context_device_memory
    when the least free device memory at any probed step's peak, prompt or decode, is at least `headroom` (the run's
    vram_headroom_bytes). An engine that reports no probe (its steps not on Core's StepEngine, or no CUDA for
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
        log(f"serve check full context: a prompt of about {one + (copies - 1) * per:,} tokens and up to "
            f"{FULL_CONTEXT_DECODE} decoded tokens")
        before = _probe(engine)
        prompted = complete(copies, FULL_CONTEXT_DECODE)
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


def _probe(engine):
    return ((engine.telemetry() or {}).get("core") or {}).get("probe")


def _since(before, after):
    """Core's probe of the steps between two telemetry reads, per kind: their count, the slowest's seconds and the
    least free device memory, from the samples Core keeps (its recent ones). None when the engine reports no probe."""
    if after is None:
        return None
    found = {}
    for kind, value in after.items():
        steps = value["steps"] - ((before or {}).get(kind) or {}).get("steps", 0)
        samples = value.get("recent", [])[-steps:] if steps > 0 else []
        if not samples:
            continue
        found[kind] = {"steps": steps, "slowest_seconds": max(s for s, _ in samples)}
        free = [f for _, f in samples if f is not None]
        if free:
            found[kind]["min_free_device_bytes"] = min(free)
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
                      + ("" if slowest < grace else ": prefill in smaller chunks")}


def _device_memory(probe, headroom):
    found = {kind: value["min_free_device_bytes"] for kind, value in (probe or {}).items()
             if "min_free_device_bytes" in value}
    if not found:
        return {"name": "full_context_device_memory", "state": "unverified",
                "detail": "the engine reports no probed device memory (Core's StepEngine on CUDA probes it)"}
    least = min(found.values())
    shown = ", ".join(f"{kind} {value / 2**30:.2f} GiB" for kind, value in sorted(found.items()))
    return {"name": "full_context_device_memory", "state": "passed" if least >= headroom else "failed",
            "detail": f"least free device memory at a step's peak up to the full context: {shown}; the run's VRAM "
                      f"headroom is {headroom / 2**30:.2f} GiB"}


def serve_checks(port, capabilities, variables, timeout, log):
    """Run the standard serving checks against crit serve at `port`; returns their items (name, state, detail)
    without the final stop_engine item, which the caller adds once the engine has stopped."""
    client, items = Client(port, timeout), []

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
    """(status, rationale) of the serving checks: failed if any failed, unverified if a required one is."""
    states = {i["name"]: i["state"] for i in items}
    if "failed" in states.values():
        status = "failed"
    elif all(states.get(name) == "passed" for name in REQUIRED):
        status = "passed"
    else:
        status = "unverified"
    return status, "crit run serve --check: " + "; ".join(f"{i['name']} {i['state']}" for i in items)
