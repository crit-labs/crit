"""crit run ... record-routes: the routing trace of a corpus, recorded by the candidate's own engine. The candidate is
served through crit serve under a check attempt with its routing plugin's route recording on (the plugin in its
selection that declares the options record_routes and trace_path, as expert-offload does), every prompt of the corpus
answered once, and the engine closed, which writes the trace. The trace is what `crit plugins offload --trace`
replays and a residency profile is built from (device-expert-cache's residency_profile).

A content-dependent choice is tuned on a corpus unrelated to Eval (AGENTS.md): general prose, how-tos and code in
several languages, never Eval's prompts, continuations or held-out texts. The corpus is a JSON list whose items are
a prompt's text, {"text": ...} (other keys, such as a language, are kept as labels) or {"messages": [...]}.
Development evidence under a check attempt: it records no check and no benchmark."""
import http.client
import json
from pathlib import Path
import time

from .ablate import _import
from .attempt import Attempt, packaged, remaining, unsealed
from .records import load_run, read_json
from .served import Served

ROUTES = 250000      # the decode routes the plugin keeps (its record_routes): the trace's last this many plans
MAX_TOKENS = 384     # each reply's tokens: decode routes, which a prompt's own rows do not give
OPTIONS = ("record_routes", "trace_path")


def corpus_requests(path):
    """[(label, messages)] of a corpus file (module docstring)."""
    items = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(items, list) or not items:
        raise ValueError(f"{path}: a corpus is a JSON list of prompts")
    out = []
    for index, item in enumerate(items):
        if isinstance(item, str):
            item = {"text": item}
        if not isinstance(item, dict) or not (isinstance(item.get("text"), str) or isinstance(item.get("messages"), list)):
            raise ValueError(f"{path}: item {index} is no prompt text, {{\"text\"}} or {{\"messages\"}}")
        messages = item.get("messages") or [{"role": "user", "content": item["text"]}]
        label = " ".join(str(v) for k, v in item.items() if k not in ("text", "messages") and isinstance(v, str))
        out.append((label or str(index + 1), messages))
    return out


def recording_plugin(selection, registry):
    """The plugin of the selection that records routes: the one whose manifest declares record_routes and
    trace_path."""
    found = [plugin_id for plugin_id in dict.fromkeys(selection["use"].values())
             if plugin_id and plugin_id in registry
             and all(name in ((registry[plugin_id].get("options")) or {}) for name in OPTIONS)]
    if not found:
        raise ValueError("No plugin of the candidate's selection records routes (options record_routes and "
                         "trace_path, as expert-offload declares them)")
    return found[0]


def _reply(port, messages, max_tokens, timeout):
    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=timeout)
    try:
        connection.request("POST", "/v1/chat/completions",
                           json.dumps({"messages": messages, "max_tokens": max_tokens}),
                           {"Content-Type": "application/json"})
        response = connection.getresponse()
        data = response.read().decode("utf-8")
        if response.status != 200:
            raise ValueError(f"crit serve answered {response.status}: {data[:500]}")
        return (json.loads(data).get("usage") or {}).get("completion_tokens") or 0
    finally:
        connection.close()


def record_routes(store, run_id, candidate_id, corpus, out=None, routes=ROUTES, max_tokens=MAX_TOKENS, seconds=None):
    """Serve the candidate with route recording on, answer every prompt of `corpus` (a file, module docstring) once,
    close the engine and return {attempt, trace, plugin, prompts, tokens, routes}. The trace goes to `out` (default
    work/routes/CANDIDATE.json in the run folder)."""
    ablation = _import("crit_plugins.ablation")
    assemble = _import("crit_plugins.assemble")
    root = store.directory(run_id)
    run = load_run(root)
    folder = unsealed(root, candidate_id, "The candidate is sealed and takes no attempt: record routes on a candidate "
                      "before its seal, or on a draft candidate created from its source")
    packaged(folder, "recording routes on", "crit run record-routes serves")
    requests = corpus_requests(corpus)
    selection = read_json(folder / "candidate.json")["configuration"].get("plugins")
    plugins = folder / "package" / "plugins"
    if selection is None or not plugins.is_dir():
        raise ValueError("record-routes serves a candidate built from plugins (configuration.plugins)")
    registry = assemble.Registry.discover(plugins)
    selection = ablation.normalize(selection, registry)
    plugin = recording_plugin(selection, registry)
    trace = Path(out) if out is not None else root / "work" / "routes" / f"{candidate_id}.json"
    trace = trace.resolve()
    trace.parent.mkdir(parents=True, exist_ok=True)
    trace.unlink(missing_ok=True)   # a trace an earlier recording left is no evidence of this one
    options = {plugin_id: dict(given) for plugin_id, given in selection["options"].items()}
    options.setdefault(plugin, {}).update(record_routes=int(routes), trace_path=str(trace))
    configuration = {"plugins": {"use": dict(selection["use"]), "base": selection["base"], "options": options}}
    kv_tokens = run["spec"]["hardware_constraints"]["kv_tokens"]
    attempt = Attempt(store, run_id, candidate_id, "check", seconds, "record-routes", "crit run record-routes",
                      "The route recording")
    tokens = 0
    with attempt.running() as log:
        from crit_core.machine import engine_lock
        log(f"crit run record-routes: {len(requests)} prompts of {corpus}, {plugin} recording {routes:,} routes")
        served = Served(store, run_id, candidate_id, attempt.id, kv_tokens, log, "The route recording",
                        timeout=min(86400, max(1.0, remaining(attempt.deadline_at))))
        with engine_lock(f"crit run record-routes {run_id} {candidate_id}",
                         deadline=time.monotonic() + remaining(attempt.deadline_at)):
            with served.until(attempt.deadline_at):
                _, _, port = served.open(configuration=configuration)
                for index, (label, messages) in enumerate(requests, 1):
                    started = time.monotonic()
                    decoded = _reply(port, messages, max_tokens, served.timeout)
                    tokens += decoded
                    log(f"crit run record-routes: {index}/{len(requests)} {label}: {decoded} tokens in "
                        f"{time.monotonic() - started:.1f} s")
                served.stop()   # the engine closes, and the plugin writes its trace
        if not trace.is_file():
            raise ValueError(f"{plugin} wrote no trace to {trace} when the engine closed")
    recorded = len(json.loads(trace.read_text(encoding="utf-8"))) if trace.is_file() else 0
    result = attempt.settle(evidence=(trace,) if trace.is_file() and trace.is_relative_to(root) else ())
    attempt.finish()
    return {"attempt": result, "trace": str(trace), "plugin": plugin, "prompts": len(requests), "tokens": tokens,
            "routes": recorded}
