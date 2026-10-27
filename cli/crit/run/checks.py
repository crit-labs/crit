"""Check records crit writes from its own evidence: the gates of a retained Eval result (crit run check --from-eval)
and the standard serving checks against the packaged candidate (crit run serve --check).

From Eval, each of the run's fixed gates is judged by the Eval gates its criteria name: arithmetic by quality and
sampling; functional by coverage, repeatable, request_history, prompt_cache, retrieval and lower_context (and
exactness when the run requires it); resources by capacity and the device memory left free at the required
context against the run's VRAM headroom. One check per case and gate cites the retained result.
"""
import http.client
import json
import time

from .records import digest, file_sha, ident, load_run, now, quality_terms, read_json, safe_path

GATES = {"arithmetic": ("quality", "sampling"),
         "functional": ("coverage", "repeatable", "request_history", "prompt_cache", "retrieval", "lower_context")}


def eval_results(folder):
    return sorted((folder / "eval").glob("attempt-*.json"))


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
    required = result.get("requirements") or {}
    if required.get("quality_budget") != budget or required.get("exactness") != exactness:
        raise ValueError(f"{path.relative_to(root).as_posix()} ran under quality terms {required}, the run's are "
                         f"{{'quality_budget': {budget}, 'exactness': '{exactness}'}}: evaluate the candidate again")
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

    if "chat-tools-v1" in capabilities:
        forced = "chat-tool-choice-v1" in capabilities

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

    if "chat-logprobs-v1" in capabilities:
        def logprobs():
            data = client.chat({"messages": QUESTION, "logprobs": True, "top_logprobs": 2})
            rows = (data["choices"][0].get("logprobs") or {}).get("content") or []
            if len(rows) != data["usage"]["completion_tokens"] or any(len(r.get("top_logprobs", [])) != 2 for r in rows):
                return "failed", f"{len(rows)} log-probability rows for {data['usage']['completion_tokens']} tokens"
            return "passed", f"{len(rows)} tokens with their log-probability and 2 alternatives"
        check("logprobs", logprobs)

    switches = {k: not v for k, v in variables.items() if isinstance(v, bool)}
    if switches and "chat-template-kwargs-v1" in capabilities:
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
