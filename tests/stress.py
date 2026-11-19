"""The pre-build stress test: crit's toolkit on real checkpoint tensors, before an engine build or a release starts.

    python tests/stress.py [--tier full|quick] [--cache DIR] [--only SCENARIO ...] [--checkpoints NAME ...] [--out DIR]

It serves layer-cut copies of real checkpoints (tests/support/layer_cut.py: a few decoder layers, every tensor byte
for byte in its stored format) as crit run serves a candidate, and sweeps the scenarios that this toolkit's bugs
showed up in during engine builds: both quality modes, device memory pressures (natural, tight, extreme and floor,
made by a VRAM ballast: crit_plugins.memory's CRIT_VRAM_BALLAST, so expert-offload places experts off the device),
short, 32k and full-context prompts, a ladder of prompt lengths whose prefill rate must rise with the length (where
the experts are off the device), a cancel in the middle of a prefill, reopening at a lower context, drafting from
the draft layer and through the drafter seam, crit smoke, crit run serve --check (with --data where it is
meaningful) with crit-eval's gate pre-check, and on a twelve-layer cut the drift that grows with depth and each
prompt arithmetic's cost. Gates that judge quality against the full model's text are left out (a cut model's text is
nonsense: smoke's `fluent`); self-consistency gates stay. The full tier, the default, runs before an engine build or
a release, and ends with one load of a whole checkpoint at the context engine builds ask for (every gate, none left
out): what a cut cannot show, the first load's expert store, the device memory left at the full context, drift at the
model's own depth. --tier quick, for every commit, serves one load a checkpoint and the tight pressure.

Everything that does not change with the scenario is built once per commit and kept in the cache (default
/llm/sessions-cache/stress): Core's step library, one run a checkpoint and quality mode whose candidate is a package of
the library's plugins at their defaults (what an engine build starts from), crit-eval's prepared data for each cut
model, and expert-offload's store (the plugin cache). The cut checkpoints themselves are made once (--fixtures, default
/llm/models/crit/fixtures) from the checkpoints in --models. Each load serves one checkpoint in one mode at one
pressure; every scenario that does not need its own load runs inside it. Each load takes the machine's shared GPU
lock (--gpu-lock, an flock file: /llm/engines/crit/heavy.lock) and crit's engine lock, and gives them back after.

The report is one JSON file and a markdown table (scenario, state, seconds, detail) in --out (default the cache's
reports/COMMIT-TIME). The exit status is 1 when a scenario failed. Builds and releases start only from a commit where
this and tests/test_checkpoint_manifests.py pass (docs/BUILDING_ENGINES.md)."""
import argparse
import contextlib
import hashlib
import http.client
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
for _folder in ("core/python", "cli", "smoke", "tests", "../crit-eval"):
    if str(ROOT / _folder) not in sys.path:
        sys.path.insert(0, str(ROOT / _folder))

# The cut fixtures: name -> (source checkpoint folder under --models, decoder layers kept (None: layer_cut's
# default, the first ones up to the first of every layer kind: four for Flash-Next)). The deep cut keeps twelve, three
# of each kind's periods, so an error that grows with depth (decode against prefill under the speed default) shows,
# and its experts exceed the device: at the natural pressure they live in host memory.
FULL = "full"   # no cut: the checkpoint itself, every layer (what only the whole model shows: FULL_SIZE below)
CHECKPOINTS = {
    "fnext-nvfp4": ("nvidia-Qwen3.8-Flash-Next-NVFP4", None),
    "fnext-fp4local": ("nvidia-Qwen3.8-Flash-Next-FP4-Local", None),
    "fnext-nvfp4-deep": ("nvidia-Qwen3.8-Flash-Next-NVFP4", list(range(12))),
    "fnext-fp4local-full": ("nvidia-Qwen3.8-Flash-Next-FP4-Local", FULL),
}
# The quality modes a checkpoint's runs are made in (default MODES).
CHECKPOINT_MODES = {"fnext-nvfp4-deep": ("speed",), "fnext-fp4local-full": ("speed",)}
MODES = ("speed", "preserve")
# Device memory pressures: the ballast leaves the cut model's non-expert weights, ACTIVATIONS_GIB and this share of
# its routed experts free (natural: no ballast, the experts fit the device). On an RTX 5090 with the four-layer
# Flash-Next cuts: at 0.58 expert-offload keeps the experts in host memory behind a device cache of about 78% of
# them; at 0.5 the cache is smaller than its decode slots and a prompt ring of a layer's experts together; at 0.08
# a few slots are left, too few to stage a prompt even with the CPU tier: the load must be refused, naming them.
PRESSURES = {"natural": None, "tight": 0.58, "extreme": 0.5, "floor": 0.08}
ACTIVATIONS_GIB = 4.0      # what a pressure leaves beside the weights: KV and states, activations, the CUDA context
KV_TOKENS = 262144         # the cut models' positions: the full context
FULL_CONTEXT = 200000      # the whole model's context: what this machine's engine builds are asked for
CONTEXT_32K = 32768
LOWER_CONTEXT = 65536      # reopening at a lower context
LADDER = (24, 96, 192, 400, 1600, 6400)   # prompt_ladder's prompt lengths in tokens, about
LADDER_SLACK = 0.9         # a rung's prefill rate against the rung's before it: under this share of it fails
PRECHECKS = 3              # with --spread: crit-eval's gate pre-check, run this often in one load
CANCEL_AFTER_S = 1.5       # a cancel this far into a long prompt's prefill
GRACE_S = 1.0              # crit serve's cancel grace (crit_api.residency.CANCEL_GRACE_SECONDS)
# Gates a cut model cannot pass: they judge its text against a full model's.
CUT_MODEL_GATES = {
    "smoke_fluent": "judges perplexity against a working model's text; a cut model's text is nonsense",
    "smoke_recall": "asks the model to read a repeated passage at under 1 nat a token, which only a working model does",
    "tool_call": "parses the arguments of the model's own tool call, which a cut model does not close within its tokens"}
FILLER = ("The river bends around the old mill, and the path follows it past the orchard and the stone wall. ")


def log(message):
    print(time.strftime("%H:%M:%S ") + str(message), flush=True)


def git(folder, *args):
    """git in `folder`, also where it is a worktree whose .git names a Windows path seen from WSL (C:/... ->
    /mnt/c/...)."""
    command = ["git", "-c", "safe.directory=*", *args]
    found = subprocess.run(command, cwd=folder, capture_output=True)
    if found.returncode and (Path(folder) / ".git").is_file():
        text = (Path(folder) / ".git").read_text(encoding="utf-8").strip()
        gitdir = text.split(":", 1)[1].strip() if text.startswith("gitdir:") else ""
        if len(gitdir) > 2 and gitdir[1] == ":":
            gitdir = "/mnt/" + gitdir[0].lower() + gitdir[2:]
            common = (Path(gitdir) / "commondir").read_text(encoding="utf-8").strip()
            env = {**os.environ, "GIT_DIR": gitdir, "GIT_COMMON_DIR": str((Path(gitdir) / common).resolve()),
                   "GIT_WORK_TREE": str(folder)}
            found = subprocess.run(command, cwd=folder, capture_output=True, env=env)
    if found.returncode:
        raise RuntimeError(f"git {' '.join(args)} in {folder}: {found.stderr.decode()[:300]}")
    return found.stdout


# What a stress run builds from: a change outside these (this harness, the docs, the tests) keeps the cache.
BUILT = ("core", "inference", "cli", "api", "smoke", "models", "kernels", "pyproject.toml")


def commit_key():
    """The key of what the builds read: the commit's trees of BUILT, with a digest of their uncommitted changes."""
    trees = git(ROOT, "rev-parse", *[f"HEAD:{path}" for path in BUILT])
    diff = git(ROOT, "diff", "HEAD", "--", *BUILT)
    head = hashlib.sha256(trees).hexdigest()[:12]
    return head + ("-dirty-" + hashlib.sha256(diff).hexdigest()[:8] if diff else "")


# -- the builds ------------------------------------------------------------------------------------------------------
class Cache:
    """The stress test's cache: one folder for each state of the code it builds from (commit_key: Core's build, the
    runs and their packages), and folders kept across them (eval data by crit-eval commit, the plugin cache, the cut
    fixtures)."""

    def __init__(self, root, fixtures, models):
        self.root, self.fixtures, self.models = Path(root), Path(fixtures), Path(models)
        self.key = commit_key()
        self.commit = self.root / "commits" / self.key
        self.commit.mkdir(parents=True, exist_ok=True)
        self.plugin_cache = self.root / "plugin-cache"
        self.timings = {}

    def timed(self, name, built):
        self.timings[name] = round(self.timings.get(name, 0.0) + built, 1)

    # Core's step library, built from this checkout
    def core(self):
        library = self.commit / "core-build" / "libcrit_core_steps.so"
        if not library.is_file():
            began = time.perf_counter()
            log("building Core's step library")
            build = self.commit / "core-build"
            subprocess.run(["cmake", "-G", "Ninja", "-DCMAKE_BUILD_TYPE=Release", "-S", str(ROOT / "core"), "-B",
                            str(build)], check=True, capture_output=True)
            subprocess.run(["cmake", "--build", str(build), "--target", "crit_core_steps"], check=True,
                           capture_output=True)
            found = next(build.rglob("libcrit_core_steps.so"))
            if found != library:
                shutil.copy2(found, library)
            self.timed("core", time.perf_counter() - began)
        os.environ["CRIT_CORE_STEP_LIBRARY"] = str(library)
        return library

    # the cut checkpoints
    def fixture(self, name):
        from support import layer_cut
        source, layers = CHECKPOINTS[name]
        if layers == FULL:
            return self.models / source
        target = self.fixtures / f"{source}-cut{len(layers) if layers else 4}"
        if not (target / "layer_cut.json").is_file():
            began = time.perf_counter()
            if target.exists():
                shutil.rmtree(target)
            log(f"cutting {source}")
            layer_cut.cut(self.models / source, target, layers)
            self.timed("fixtures", time.perf_counter() - began)
        return target

    # crit-eval's prepared data for a cut model, by crit-eval's commit
    def eval_data(self, name, model):
        from crit_eval.quality.prepare import PREPARED
        import crit_eval
        eval_root = Path(crit_eval.__file__).resolve().parents[1]   # crit-eval's checkout, wherever it is imported from
        head = git(eval_root, "rev-parse", "--short=12", "HEAD").decode().strip()
        data = self.root / "eval-data" / head / name
        if not (data / PREPARED).is_file():
            began = time.perf_counter()
            log(f"preparing crit-eval's data for {name}")
            data.mkdir(parents=True, exist_ok=True)
            from crit_eval.cli import main as eval_main
            code = eval_main(["prepare", "--model", str(model), "--data", str(data)])
            if code:
                raise RuntimeError(f"crit eval prepare exited {code}")
            self.timed("eval_data", time.perf_counter() - began)
        return data

    # one run a checkpoint and quality mode, its candidate the library's plugins at their defaults
    def run(self, name, model, mode):
        from crit.run.store import RunStore
        index = self.commit / "runs.json"
        found = json.loads(index.read_text(encoding="utf-8")) if index.is_file() else {}
        store = RunStore(self.commit / "runs", project=ROOT)
        key = f"{name}/{mode}"
        if key in found:
            return store, found[key]["run"], found[key]["candidate"]
        began = time.perf_counter()
        log(f"creating the {mode} run of {name} and packaging its library candidate")
        run_id, candidate_id = build_run(store, model, mode, self.commit / "sources" / name,
                                         FULL_CONTEXT if CHECKPOINTS[name][1] == FULL else KV_TOKENS)
        found[key] = {"run": run_id, "candidate": candidate_id}
        index.write_text(json.dumps(found, indent=2), encoding="utf-8")
        self.timed("runs", time.perf_counter() - began)
        return store, run_id, candidate_id


def selection(model):
    """The library's plugins for a Qwen3.8-Flash-Next-like checkpoint whose experts may live off the device, at their
    defaults: what an engine build's first candidate starts from. The router's options are the checkpoint's rule
    (model-template refuses another), the lookup tables are placed off the device (the n-gram table is larger than
    the device), and the drafter is the draft layer's chain."""
    use = {"model": "model-template", "placement": "device-placement", "projections": "quantized-projections",
           "attention": "partitioned-attention", "kv_cache": "attention-state", "sampler": "token-sampling",
           "norms": "fused-norms", "embeddings": "lookup-table-rows", "recurrence": "sequential-recurrence",
           "router": "topk-router", "experts": "expert-offload", "expert_source": "expert-tensors",
           "drafter": "greedy-head-drafter"}
    options = {"model-template": {"mtp": True},
               "topk-router": {"score_before_selection": True, "renormalize": True},
               "device-placement": {"off_device": "tables."},
               # what crit plugins offload gives every engine whose experts exceed the device (AGENTS.md): the KV
               # pools hold what their open sequences use and the expert cache the rest
               "attention-state": {"elastic": True}}
    return {"use": use, "options": options}


PLUGINS = ["model-template", "device-placement", "quantized-projections", "partitioned-attention", "attention-state",
           "token-sampling", "fused-norms", "lookup-table-rows", "sequential-recurrence", "topk-router",
           "expert-offload", "expert-tensors", "greedy-head-drafter", "mtp-decode"]


def build_run(store, model, mode, sources, kv_tokens=KV_TOKENS):
    """A run of `model` in quality mode `mode` with one candidate packaged from the library's plugins (crit run
    create, recon, capture, resolve, plan, candidate and package, as an engine build does them)."""
    from crit.plugins import add
    from crit.run import recon
    from crit.run.capture import capture
    view = store.create({"model_path": str(model), "correctness_criteria": "crit's pre-build stress test",
                         "kv_tokens": kv_tokens, "quality_mode": mode, "candidate_budget": 64,
                         "workload": {"prompt_tokens": 512, "output_tokens": 64, "repetitions": 1}})
    run_id = view["run"]["run_id"]
    root = store.directory(run_id)
    store.recon(run_id)
    notes = root / "work" / "recon.md"
    text = notes.read_text(encoding="utf-8")
    for title in recon.PLANS + (recon.PLACEMENT_PLAN,):
        text = text.replace(f"## {title}\n\nTODO", f"## {title}\n\nThe library's plugins at their defaults "
                            "(crit's pre-build stress test).")
    notes.write_text(text, encoding="utf-8")
    case = view["run"]["spec"]["cases"][0]["case_id"]
    row = capture(store, run_id, case)
    store.freeze_inputs(run_id, {"hardware_manifest": {"device": "stress test"},
                                 "environment_manifest": {"benchmark": "crit stress"},
                                 "case_manifests": {case: [row["row"] if "row" in row else row]}})
    plan = {"parent_candidate_id": None, "decision": "build",
            "needed": {name: {"needed": True, "evidence": "the stress test serves every path"}
                       for name in ("decode", "prefill")},
            "gaps": [], "levers": [], "rejected": [], "forecast": None, "gates": "the stress test's scenarios",
            "strategy": {"current": {"name": "library defaults", "ceiling": "unknown", "evidence": "stress test"},
                         "alternative": {"name": "none", "ceiling": "unknown", "evidence": "stress test"},
                         "choice": "current", "reason": "the stress test measures the library as it is"}}
    if mode == "speed":
        plan["prompt_activations"] = "stored"
    plan_id = store.plan(run_id, plan)["plan_id"]
    if sources.exists():
        shutil.rmtree(sources)
    (sources / "plugins").mkdir(parents=True)
    add(sources / "plugins", PLUGINS)
    configuration = {"plugins": selection(model)}
    (sources / "configuration.json").write_text(json.dumps(configuration, indent=2), encoding="utf-8")
    candidate = store.candidate(run_id, sources, {
        "label": "library defaults", "rationale": "crit's pre-build stress test", "strategy": "library defaults",
        "configuration": configuration, "plan_id": plan_id, "lever_fates": {}, "store_decisions": {}})
    (sources / "configuration.json").unlink()   # crit writes the candidate's completed configuration into the package
    store.package(run_id, candidate["candidate_id"], sources,
                  {"build": {"toolchain": {"identity": "crit stress"}, "dependencies": [], "command": ["none"]}})
    return run_id, candidate["candidate_id"]


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
        self.model = self.model or self.models()[0]
        body = {"model": self.model, "messages": [{"role": "user", "content": content}], "max_tokens": max_tokens,
                "temperature": 0, **extra}
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


def words(tokens):
    """A user message of about `tokens` prompt tokens (FILLER is about 22 tokens a copy for Qwen's tokenizer)."""
    return FILLER * max(1, tokens // 22)


# -- scenarios -----------------------------------------------------------------------------------------------------------
class Report:
    def __init__(self):
        self.items = []

    def add(self, name, state, seconds=0.0, detail=""):
        item = {"scenario": name, "state": state, "seconds": round(float(seconds), 1), "detail": str(detail)[:1500]}
        self.items.append(item)
        log(f"{state:8s} {name} ({item['seconds']} s) {item['detail'][:200]}")
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


def in_load(report, prefix, served, port, extras, cut=True):
    """The scenarios that run inside a load, on the engine crit run serve serves (its ready hook). cut: the model is
    a layer cut, whose text is nonsense (CUT_MODEL_GATES)."""
    chat = Chat(port)
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
    if "ladder" in extras:
        # Prompts of every length, each a text no earlier request began with (no prompt cache serves it). A request's
        # fixed costs weigh less the longer its prompt, so the rate rises from each rung to the next; a rung slower
        # than the one before it pays something that prompt's length does not ask for (a sweep of every off-device
        # expert for a prompt that routes a few of them), which fails.
        with report.scenario(prefix + "prompt_ladder") as found:
            rungs, slow = [], []
            for tokens in LADDER:
                data, seconds = chat.chat(f"Text {tokens}: " + words(tokens), 1)
                prompt = data["usage"]["prompt_tokens"]
                if rungs and prompt / seconds < LADDER_SLACK * rungs[-1][0] / rungs[-1][1]:
                    slow.append(f"{prompt:,} tokens are prefilled slower than {rungs[-1][0]:,}")
                rungs.append((prompt, seconds))
            found["detail"] = "; ".join(f"{p:,} tokens {s:.2f} s ({p / s:,.0f} tok/s)" for p, s in rungs)
            if slow:
                found["state"], found["detail"] = "failed", "; ".join(slow) + ": " + found["detail"]
    if "32k" in extras:
        with report.scenario(prefix + "prompt_32k") as found:
            data, seconds = chat.chat(words(CONTEXT_32K - 512), 8)
            prompt = data["usage"]["prompt_tokens"]
            found["detail"] = f"{prompt:,} prompt tokens in {seconds:.1f} s ({prompt / seconds:,.0f} tok/s)"
    if "cancel" in extras:
        with report.scenario(prefix + "cancel_in_prefill") as found:
            _, alone = chat.chat("Say yes.", 4)
            _, whole = chat.chat(words(CONTEXT_32K * 2), 1)   # the long prompt's prefill, uncancelled
            chat.cancelled_stream(words(CONTEXT_32K * 2), min(CANCEL_AFTER_S, whole / 2))
            _, after = chat.chat("Say yes.", 4)
            settle = after - alone   # what the request after the cancel waited for it to settle
            found["detail"] = (f"cancelled {min(CANCEL_AFTER_S, whole / 2):.1f} s into a {whole:.1f} s prefill; the "
                               f"next request waited {settle:.2f} s for it to settle (grace {GRACE_S:.0f} s; alone it "
                               f"takes {alone:.2f} s)")
            if settle > GRACE_S:
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
        log(f"smoke at the run's context: {time.perf_counter() - began:.0f} s")
    if "drafting" in extras:
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
                seam.chat("warm " + words(CONTEXT_32K - 512), 1)   # another text: no prompt cache serves the next
                data, seconds = seam.chat(f"{arithmetic} " + words(CONTEXT_32K - 512), 1)
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
    if "lower" in extras:
        with report.scenario(prefix + f"reopen_at_{LOWER_CONTEXT // 1024}k") as found:
            _, _, port2 = served.open(LOWER_CONTEXT)
            data, seconds = Chat(port2).chat(words(CONTEXT_32K - 512), 8)
            _, _, port = served.open()
            chat.port = port
            found["detail"] = f"reopened at {LOWER_CONTEXT:,} tokens, a 32k prompt in {seconds:.1f} s"
    return chat.port


def served_configuration(served):
    from crit.run.records import read_json
    folder = served.store.directory(served.run_id) / "candidates" / served.candidate_id
    return read_json(folder / "candidate.json")["configuration"]


def precheck_spread(report, prefix, served, store, run_id, data, port, count):
    """crit-eval's gate pre-check `count` times on one load: each run's items, and whether they agree."""
    from crit.run.records import load_run
    from crit.run.served import _precheck
    run = load_run(store.directory(run_id))
    runs = []
    for number in range(count):
        began = time.perf_counter()
        items = _precheck(served, run, data, port, log=lambda line: None)
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


def ballast_bytes(model, share):
    """The bytes the cut model needs free at a pressure: its weights that stay on the device (its lookup tables do
    not), ACTIVATIONS_GIB, and `share` of its routed experts."""
    from support import manifests
    found = manifests.tensors(manifests.read(model))
    experts = sum(t.nbytes for n, t in found.items() if ".mlp.experts." in n)
    # the lookup tables the selection places off the device (off_device "tables."): the n-gram table and the token
    # embeddings, read where the checkpoint stores them
    tables = sum(t.nbytes for n, t in found.items() if "ngram_embedding" in n or n.endswith("embed_tokens.weight"))
    rest = sum(t.nbytes for t in found.values()) - experts - tables
    return int(rest + ACTIVATIONS_GIB * 2 ** 30 + share * experts), experts


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


def load(report, cache, name, model, mode, pressure, data, extras, plugin_tests, store_cache=True):
    """One load: crit run serve --check of the checkpoint's run in `mode` at `pressure`, the in-load scenarios in its
    ready hook. Its serving checks join the report, named NAME/MODE/PRESSURE/ITEM."""
    import crit.run.external as external
    from crit.run.served import serve
    store, run_id, candidate_id = cache.run(name, model, mode)
    prefix = f"{name}/{mode}/{pressure}/"
    cut = CHECKPOINTS[name][1] != FULL
    share = PRESSURES[pressure]
    leave = None
    if share is not None:
        leave, experts = ballast_bytes(model, share)
        report.add(prefix + "ballast", "info", 0.0, f"the ballast leaves {leave / 2**30:.1f} GiB free "
                   f"({share:.0%} of {experts / 2**30:.1f} GiB of routed experts beside the rest)")
    captured = {}
    wrapped = external.serving

    def serving(*args, **kwargs):
        captured["served"] = found = wrapped(*args, **kwargs)
        return found

    def ready(port):
        served = captured["served"]
        report.add(prefix + "ready", "info", time.perf_counter() - began,
                   "the engine answered this long after crit run serve --check began (its plugin tests, then the load)")
        port = in_load(report, prefix, served, port, extras, cut)
        if "spread" in extras and data is not None:
            captured["prechecks"] = precheck_spread(report, prefix, served, store, run_id, data, port,
                                                    PRECHECKS - 1)[1]

    began = time.perf_counter()
    external.serving = serving
    try:
        with environment(CRIT_VRAM_BALLAST=f"leave:{leave}" if leave else None,
                         CRIT_PLUGIN_CACHE=str(cache.plugin_cache) if store_cache else ""):
            result = serve(store, run_id, candidate_id, ready=ready, check=True,
                           data=str(data) if data is not None and "precheck" in extras else None,
                           smoke="smoke" not in extras, plugin_tests=plugin_tests)
    except Exception as error:
        refused = pressure == "floor" and "holds no prompt ring" in str(error)
        # At the floor no cache can stage a prompt: the right outcome is a refusal at load naming what it needs.
        report.add(prefix + "load", "passed" if refused else "failed", time.perf_counter() - began,
                   ("refused at load, as it should be: " if refused else "") + f"{type(error).__name__}: {error}")
        if not refused:
            traceback.print_exc()
        return
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


def plan_loads(checkpoints, only, tier="full", spread=False, arithmetic=False):
    """[(name, mode, pressure, extras, plugin tests)]: the tier's scenarios in as few loads as they take.

    full (before an engine build or a release): every cut checkpoint in both modes at every pressure, and the whole
    checkpoint once (speed, no ballast: its experts exceed the device as they are), with the ladder. The first
    checkpoint's speed load also drafts through the drafter seam and its preserve load runs the pre-check; the
    pressure loads and the deep cut run smoke once, at the run's context ("smoke": in the load, no reopen); the deep
    cut also measures each prompt arithmetic with the CPU tier computing decode misses where `arithmetic` asks (its
    expert store, built in host memory at every load, takes ten minutes).
    quick (every commit): one speed load per checkpoint at the natural pressure (serve --check, smoke, the pre-check on
    the first) and the first checkpoint at the tight pressure (the experts in host memory); no deep cut.
    spread: the first load repeats the pre-check twice more and reports whether the runs agree (the pre-check samples
    each case itself, so this is for looking into a flaky gate)."""
    loads = []
    for n, name in enumerate(checkpoints):
        if CHECKPOINTS[name][1] == FULL:   # the whole model, once: what no cut shows
            if tier == "full":
                loads.append((name, "speed", "natural", {"short", "ladder", "32k", "cancel", "precheck"}, False))
            continue
        if name in CHECKPOINT_MODES:   # the deep cut: smoke's drift with depth, and each prompt arithmetic's
            if tier == "full":
                loads.append((name, "speed", "natural", {"short", "32k", "cancel", "smoke"}, False))
                if arithmetic:   # its 23 GB expert store is built in host memory: ten minutes a load
                    loads.append((name, "speed", "tight", {"short", "cancel", "smoke", "arithmetic"}, False))
            continue
        first = ({"drafting"} | ({"spread"} if spread else set())) if n == 0 else set()
        loads.append((name, "speed", "natural", {"short", "32k", "cancel"} | ({"precheck"} if n == 0 or tier == "full"
                                                                           else set()) | first, n == 0))
        if tier == "quick":
            if n == 0:
                loads.append((name, "speed", "tight", {"short", "ladder", "32k", "cancel", "smoke"}, False))
            continue
        loads.append((name, "preserve", "natural", {"short", "cancel"} | ({"precheck"} if n == 0 else set()), False))
        loads.append((name, "speed", "tight", {"short", "ladder", "32k", "cancel", "smoke"}, False))
        loads.append((name, "speed", "extreme", {"short", "32k", "cancel", "smoke"}, False))
        loads.append((name, "speed", "floor", {"short", "smoke"}, False))
    if only:
        loads = [load for load in loads if any(o in "/".join(load[:3]) for o in only)]
    return loads


def write_report(report, out, started, cache):
    out.mkdir(parents=True, exist_ok=True)
    total = time.perf_counter() - started
    summary = {"commit": git(ROOT, "rev-parse", "--short=12", "HEAD").decode().strip(), "key": cache.key,
               "seconds": round(total, 1), "builds": cache.timings,
               "passed": sum(i["state"] == "passed" for i in report.items),
               "failed": sum(i["state"] == "failed" for i in report.items),
               "skipped": sum(i["state"] == "skipped" for i in report.items), "items": report.items}
    (out / "stress.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    lines = [f"# crit stress test at {summary['commit']} (cache key {cache.key})", "",
             f"{summary['passed']} passed, {summary['failed']} failed, {summary['skipped']} skipped in "
             f"{total / 60:.1f} min (builds: {', '.join(f'{k} {v:.0f} s' for k, v in cache.timings.items()) or 'cached'})",
             "", "| Scenario | State | Seconds | Detail |", "| --- | --- | --- | --- |"]
    for item in report.items:
        detail = item["detail"].replace("|", "\\|").replace("\n", " ")[:300]
        lines.append(f"| {item['scenario']} | {item['state']} | {item['seconds']} | {detail} |")
    (out / "stress.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary


def environment_for(cache_root, shared):
    """The Python environment the stress test runs in, made once a checkout: a venv whose third-party packages are
    those of `shared` (a venv with torch and crit's dependencies, read through a .pth) and whose crit, crit-core and
    crit-eval are this checkout's and the crit-eval beside it, installed editable. crit serve starts its engine host
    with the running interpreter, so the host imports this checkout's code too, never another install's."""
    import crit_eval
    key = hashlib.sha256(str(ROOT).encode()).hexdigest()[:12]
    venv = Path(cache_root) / "venvs" / key
    python = venv / "bin" / "python"
    eval_root = Path(crit_eval.__file__).resolve().parents[1]
    want = {"crit": str(ROOT), "crit-eval": str(eval_root), "shared": str(shared)}
    marker = venv / "stress.json"
    if marker.is_file() and json.loads(marker.read_text(encoding="utf-8")) == want:
        return python
    log(f"making the stress test's Python environment in {venv}")
    if venv.exists():
        shutil.rmtree(venv)
    subprocess.run([sys.base_prefix + "/bin/python3", "-m", "venv", str(venv)], check=True)
    site = next((venv / "lib").glob("python3*/site-packages"))
    shared_site = next(Path(shared).glob("lib/python3*/site-packages"))
    (site / "shared.pth").write_text(str(shared_site) + chr(10), encoding="utf-8")
    subprocess.run([str(python), "-m", "pip", "install", "-q", "--no-deps", "--no-build-isolation", "-e",
                    str(ROOT / "core" / "python"), "-e", str(ROOT), "-e", str(eval_root)], check=True)
    marker.write_text(json.dumps(want), encoding="utf-8")
    return python


@contextlib.contextmanager
def gpu(shared, label):
    """The GPU for one load: the machine's shared GPU lock (`shared`, an flock file other work on the GPU takes too;
    empty for none) and crit's engine lock (crit_core.machine), taken before the load and given back after it, so
    work waiting for the GPU runs between loads."""
    from crit_core.machine import engine_lock
    handle = None
    if shared:
        import fcntl
        handle = open(shared, "a+")
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            log(f"waiting for the GPU lock {shared}")
            fcntl.flock(handle, fcntl.LOCK_EX)
    try:
        with engine_lock("crit stress test " + label, log=log):
            yield
    finally:
        if handle is not None:
            handle.close()   # closing the file releases the flock


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--cache", default="/llm/sessions-cache/stress")
    parser.add_argument("--fixtures", default="/llm/models/crit/fixtures")
    parser.add_argument("--models", default="/llm/models/crit")
    parser.add_argument("--checkpoints", nargs="*", default=list(CHECKPOINTS))
    parser.add_argument("--only", nargs="*", help="run the loads whose NAME/MODE/PRESSURE contains one of these")
    parser.add_argument("--out", help="report folder (default: the cache's reports/COMMIT-TIME)")
    parser.add_argument("--build-only", action="store_true", help="build and cache everything, serve nothing")
    parser.add_argument("--tier", choices=("full", "quick"), default="full",
                        help="full (before a build or a release): every scenario; quick (every commit): one load a "
                             "checkpoint and the tight pressure (plan_loads)")
    parser.add_argument("--spread", action="store_true", help="repeat the first load's pre-check for its spread")
    parser.add_argument("--arithmetic", action="store_true",
                        help="also measure each prompt arithmetic on the deep cut with its experts in host memory")
    parser.add_argument("--no-store-cache", action="store_true",
                        help="keep no expert store on disk (expert-offload builds it in host memory at every load)")
    parser.add_argument("--share", action="append", default=[], metavar="PRESSURE=SHARE",
                        help="the share of the routed experts a pressure leaves room for (PRESSURES)")
    parser.add_argument("--gpu-lock", default=os.environ.get("CRIT_GPU_LOCK", "/llm/engines/crit/heavy.lock"),
                        help="the machine's shared GPU lock, an flock file taken for each load (empty: none)")
    parser.add_argument("--shared-venv", default="/llm/engines/crit/crit-venv",
                        help="a venv with torch and crit's dependencies, whose packages the stress test's own reads")
    args = parser.parse_args(argv)
    if os.name != "nt" and not os.environ.get("CRIT_STRESS_ENVIRONMENT"):
        python = environment_for(args.cache, args.shared_venv)
        os.environ["CRIT_STRESS_ENVIRONMENT"] = str(python)
        os.execv(str(python), [str(python), "-B", str(Path(__file__).resolve()), *(argv or sys.argv[1:])])
    for given in args.share:
        pressure, _, share = given.partition("=")
        if pressure not in PRESSURES or pressure == "natural":
            parser.error(f"--share names tight, extreme or floor, not {pressure!r}")
        PRESSURES[pressure] = float(share)
    started = time.perf_counter()
    cache = Cache(args.cache, args.fixtures, args.models)
    report = Report()
    out = Path(args.out) if args.out else cache.root / "reports" / f"{cache.key}-{time.strftime('%Y%m%dT%H%M%S')}"
    log(f"crit stress test at {cache.key}; cache {cache.commit}")
    cache.core()
    models, data = {}, {}
    for name in args.checkpoints:
        models[name] = cache.fixture(name)
        data[name] = cache.eval_data(name, models[name])
        for mode in CHECKPOINT_MODES.get(name, MODES):
            cache.run(name, models[name], mode)
    log(f"builds: {cache.timings or 'all cached'}")
    if not args.build_only:
        for name, mode, pressure, extras, tests in plan_loads(args.checkpoints, args.only, args.tier,
                                                                  args.spread, args.arithmetic):
            with gpu(args.gpu_lock, f"{name}/{mode}/{pressure}"):
                load(report, cache, name, models[name], mode, pressure, data[name], extras, tests,
                     store_cache=not args.no_store_cache)
    summary = write_report(report, out, started, cache)
    log(f"report: {out / 'stress.md'} ({summary['passed']} passed, {summary['failed']} failed, "
        f"{summary['skipped']} skipped in {summary['seconds'] / 60:.1f} min)")
    return 1 if summary["failed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
