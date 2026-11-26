"""crit run ... switch-check: the check AGENTS.md asks for after sealing. crit serve serves the selected candidate, a
request switches it to another sealed candidate of the run and a third back, each a greedy reply to the same prompt;
the selected candidate's two replies must be equal. crit serve is the user's way to serve the run (crit_api's
ServedEngine behind its chat API, here in this process on a free loopback port), so the check runs the path the
user's switch runs, not a measurement's. The record is work/switch-check.json in the run folder."""
import http.client
import json
import threading
import time

from .records import write_json

PROMPT = "Name three rivers in Europe and one fact about each."
MAX_TOKENS = 64
TIMEOUT = 1800   # a request that loads another candidate waits for its load: an expert store can take minutes


def sealed_pair(view, selected=None, other=None):
    """(selected, other): the run's selected candidate unless given, and the newest other sealed candidate unless
    given. Both must be sealed: crit serve serves only sealed candidates."""
    sealed = [c["candidate_id"] for c in view["candidates"] if c.get("sealed")]
    selected = selected or view["state"].get("selected_candidate_id")
    if not selected:
        raise ValueError("The run has no selected candidate: select one (crit run ... select) or give --candidate")
    if selected not in sealed:
        raise ValueError(f"{selected} is not sealed: crit serve serves sealed candidates")
    if other is None:
        others = [c for c in sealed if c != selected]
        if not others:
            raise ValueError("The run has no other sealed candidate to switch to")
        other = others[-1]
    elif other not in sealed or other == selected:
        raise ValueError(f"--other {other}: give another sealed candidate of the run")
    return selected, other


def _chat(address, model, prompt, max_tokens):
    connection = http.client.HTTPConnection(*address, timeout=TIMEOUT)
    try:
        body = {"model": model, "temperature": 0, "max_tokens": max_tokens,
                "messages": [{"role": "user", "content": prompt}]}
        started = time.monotonic()
        connection.request("POST", "/v1/chat/completions", json.dumps(body), {"Content-Type": "application/json"})
        response = connection.getresponse()
        data = response.read().decode("utf-8")
        if response.status != 200:
            raise ValueError(f"crit serve answered {response.status} for {model}: {data[:500]}")
        reply = json.loads(data)
        return reply["choices"][0]["message"].get("content"), reply.get("model"), time.monotonic() - started
    finally:
        connection.close()


def switch_check(store, run_id, selected=None, other=None, prompt=PROMPT, max_tokens=MAX_TOKENS, project=None,
                 log=print):
    """Serve `selected` (default the run's selected candidate) through crit serve, switch to `other` (default the
    newest other sealed candidate) and back, a greedy reply each; write work/switch-check.json and return it. Raises
    ValueError when the selected candidate's replies before and after the switch differ."""
    from crit_api.engine import ServedEngine
    from crit_api.server import Api, Server
    from crit_core.machine import engine_lock
    selected, other = sealed_pair(store.get(run_id), selected, other)
    root = store.directory(run_id)
    steps = []
    with engine_lock(f"crit run switch-check {run_id} {selected}"):
        engine = ServedEngine(store.root, run_id, candidate_id=selected,
                              **({"project": project} if project is not None else {}))
        try:
            log(f"crit run switch-check: loading {selected}")
            engine.open()
            server = Server(("127.0.0.1", 0), Api(engine))
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            try:
                for candidate in (selected, other, selected):
                    text, served, seconds = _chat(server.server_address[:2], f"{run_id}/{candidate}", prompt,
                                                  max_tokens)
                    steps.append({"candidate": candidate, "model": served, "seconds": round(seconds, 1),
                                  "reply": text})
                    log(f"crit run switch-check: {candidate} answered in {seconds:.1f} s: {(text or '')[:80]!r}")
            finally:
                server.shutdown()
                server.server_close()
                thread.join(5)
        finally:
            engine.close()
    record = {"selected": selected, "other": other, "prompt": prompt, "max_tokens": max_tokens, "steps": steps,
              "same_reply": steps[0]["reply"] == steps[2]["reply"]}
    path = root / "work" / "switch-check.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    write_json(path, record)
    record["path"] = path.relative_to(root).as_posix()
    if not record["same_reply"]:
        raise ValueError(f"crit serve switched {selected} -> {other} -> {selected} and the selected candidate's greedy "
                         f"reply changed ({record['path']}): something the switch leaves behind or reloads otherwise "
                         "changes the engine")
    return record
