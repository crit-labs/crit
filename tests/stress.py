"""The release-time stress test: crit's toolkit on real checkpoint tensors, before a release starts.

    python tests/stress.py [--tier full|quick] [--cache DIR] [--only SCENARIO ...] [--checkpoints NAME ...] [--out DIR]

It serves layer-cut copies of the checkpoints crit targets (crit.preflight.cut: a few decoder layers, every tensor byte
for byte in its stored format) as crit run serve serves a candidate, and sweeps the scenarios of
crit.preflight.scenarios that this toolkit's bugs showed up in during engine builds: both quality modes, device memory
pressures (natural, tight, extreme and floor, made by a VRAM ballast: crit_plugins.memory's CRIT_VRAM_BALLAST, so
expert-offload places experts off the device), short, 32k and full-context prompts, a ladder of prompt lengths whose
prefill rate must rise with the length (where the experts are off the device), a cancel in the middle of a prefill,
reopening at a lower context, drafting from the draft layer and through the drafter seam, crit smoke, crit run serve
--check (with --data where it is meaningful) with crit-eval's gate pre-check, and on a twelve-layer cut the drift that
grows with depth and each prompt arithmetic's cost. Gates that judge quality against the full model's text are left
out (a cut model's text is nonsense: smoke's `fluent`); self-consistency gates stay. The full tier, the default, runs
before a release, and ends with one load of a whole checkpoint at the context engine builds ask for (every gate,
none left out): what a cut cannot show, the first load's expert store, the device memory left at the full context,
drift at the model's own depth. --tier quick serves one load a checkpoint and the tight pressure.

This is the check before a release, over fixed fixtures (CHECKPOINTS) at pressures and contexts chosen for them on
one machine. The check before an engine build is the run's own preflight (crit run ... preflight, crit.preflight): the
same code on the run's checkpoint, every size taken from the run.

Everything that does not change with the scenario is built once per state of the engine code and kept in the cache
(default /llm/sessions-cache/stress): Core's step library, one run a checkpoint and quality mode whose candidate is a
package of the library's plugins at their defaults (crit.preflight.candidate: what an engine build starts from),
crit-eval's prepared data for each cut model, and expert-offload's store (the plugin cache). The cut checkpoints
themselves are made once (--fixtures, default /llm/models/crit/fixtures) from the checkpoints in --models. Each load
serves one checkpoint in one mode at one pressure; every scenario that does not need its own load runs inside it. Each
load takes the machine's shared GPU lock (--gpu-lock, an flock file: /llm/engines/crit/heavy.lock) and crit's engine
lock, and gives them back after.

The report is one JSON file and a markdown table (scenario, state, seconds, detail) in --out (default the cache's
reports/KEY-TIME). The exit status is 1 when a scenario failed. A release starts only from a commit where this and
tests/test_checkpoint_manifests.py pass (docs/CONTRIBUTING.md)."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
for _folder in ("inference/core/python", "models/core/python", "cli", "smoke", "tests", "../crit-eval"):
    if str(ROOT / _folder) not in sys.path:
        sys.path.insert(0, str(ROOT / _folder))

from crit.preflight import records, scenarios  # noqa: E402
from crit.preflight.scenarios import Report, Sizes, gpu, log  # noqa: E402

# The cut fixtures: name -> (source checkpoint folder under --models, decoder layers kept (None: the cut's default,
# the first ones up to the first of every layer kind: four for Flash-Next)). The deep cut keeps twelve, three of each
# kind's periods, so an error that grows with depth (decode against prefill under the speed default) shows, and its
# experts exceed the device: at the natural pressure they live in host memory.
FULL = "full"   # no cut: the checkpoint itself, every layer (what only the whole model shows)
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
FLOOR_REFUSAL = "holds no prompt ring"   # what the floor's load is refused with
ACTIVATIONS_GIB = 4.0      # what a pressure leaves beside the weights: KV and states, activations, the CUDA context
KV_TOKENS = 262144         # the cut models' positions: the full context
FULL_CONTEXT = 200000      # the whole model's context: what this machine's engine builds are asked for
LOWER_CONTEXT = 65536      # reopening at a lower context


def sizes(name):
    """A fixture's scenario sizes: its context, the 32k prompt, the lower context and a 64k cancelled prompt."""
    return Sizes(context=FULL_CONTEXT if CHECKPOINTS[name][1] == FULL else KV_TOKENS, long=32768,
                 lower=LOWER_CONTEXT, cancel=2 * 32768)


# -- the builds ------------------------------------------------------------------------------------------------------
class Cache:
    """The stress test's cache: one folder for each state of the code it builds from (crit.preflight.records.crit_key:
    Core's build, the runs and their packages), and folders kept across them (eval data by crit-eval commit, the
    plugin cache, the cut fixtures)."""

    def __init__(self, root, fixtures, models):
        self.root, self.fixtures, self.models = Path(root), Path(fixtures), Path(models)
        self.key = records.crit_key(ROOT)
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
            subprocess.run(["cmake", "-G", "Ninja", "-DCMAKE_BUILD_TYPE=Release", "-S", str(ROOT / "inference" / "core"), "-B",
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
        from crit.preflight import cut
        source, layers = CHECKPOINTS[name]
        if layers == FULL:
            return self.models / source
        target = self.fixtures / f"{source}-cut{len(layers) if layers else 4}"
        if not (target / "layer_cut.json").is_file():
            began = time.perf_counter()
            if target.exists():
                shutil.rmtree(target)
            log(f"cutting {source}")
            cut.cut(self.models / source, target, layers)
            self.timed("fixtures", time.perf_counter() - began)
        return target

    # crit-eval's prepared data for a cut model, by crit-eval's commit
    def eval_data(self, name, model):
        from crit_eval.quality.prepare import PREPARED
        import crit_eval
        eval_root = Path(crit_eval.__file__).resolve().parents[1]   # crit-eval's checkout, wherever it is imported from
        head = records.git(eval_root, "rev-parse", "--short=12", "HEAD").decode().strip()
        data = self.root / "eval-data" / head / name
        if not (data / PREPARED).is_file():
            began = time.perf_counter()
            log(f"preparing crit-eval's data for {name}")
            scenarios.prepare_data(model, data)
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
        run_id, candidate_id = build_run(store, model, mode, self.commit / "sources" / name, sizes(name).context)
        found[key] = {"run": run_id, "candidate": candidate_id}
        index.write_text(json.dumps(found, indent=2), encoding="utf-8")
        self.timed("runs", time.perf_counter() - began)
        return store, run_id, candidate_id


def build_run(store, model, mode, sources, kv_tokens=KV_TOKENS):
    """A run of `model` in quality mode `mode` with one candidate of the library's plugins at their defaults for it
    (crit.preflight.candidate), its lookup tables off the device: the Flash-Next checkpoints' n-gram table is larger
    than the device."""
    from crit.preflight.candidate import build_run as build, library_selection
    selection, plugins = library_selection(model, tables_off_device=True)
    return build(store, model, mode, sources, selection, plugins, {"kv_tokens": kv_tokens},
                 criteria="crit's pre-build stress test")


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


def load(report, cache, name, model, mode, pressure, data, extras, plugin_tests, store_cache=True):
    """One load of a fixture: crit.preflight.scenarios.load of the checkpoint's run in `mode` at `pressure`, its items
    named NAME/MODE/PRESSURE/ITEM."""
    store, run_id, candidate_id = cache.run(name, model, mode)
    prefix = f"{name}/{mode}/{pressure}/"
    share = PRESSURES[pressure]
    leave = None
    if share is not None:
        leave, experts = ballast_bytes(model, share)
        report.add(prefix + "ballast", "info", 0.0, f"the ballast leaves {leave / 2**30:.1f} GiB free "
                   f"({share:.0%} of {experts / 2**30:.1f} GiB of routed experts beside the rest)")
    return scenarios.load(report, store, run_id, candidate_id, prefix, extras, sizes(name), data, plugin_tests,
                          cut=CHECKPOINTS[name][1] != FULL, leave=leave,
                          plugin_cache=str(cache.plugin_cache) if store_cache else "",
                          refusal=FLOOR_REFUSAL if pressure == "floor" else None)


def plan_loads(checkpoints, only, tier="full", spread=False, arithmetic=False):
    """[(name, mode, pressure, extras, plugin tests)]: the tier's scenarios in as few loads as they take.

    full (before a release): every cut checkpoint in both modes at every pressure, and the whole checkpoint once
    (speed, no ballast: its experts exceed the device as they are), with the ladder. The first checkpoint's speed load
    also drafts through the drafter seam and reopens at a lower context and back and at the same context (in both
    tiers), and its preserve load
    runs the pre-check; the pressure loads and the deep cut run smoke once, at the run's context ("smoke": in the load,
    no reopen); the deep cut also measures each prompt arithmetic with the CPU tier computing decode misses where
    `arithmetic` asks (its expert store, built in host memory at every load, takes ten minutes).
    quick: one speed load per checkpoint at the natural pressure (serve --check, smoke, the pre-check on the first) and
    the first checkpoint at the tight pressure (the experts in host memory); no deep cut.
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
        # The first checkpoint's speed load also reopens at a lower context and back (reopen_at_64k: a closed
        # engine's elastic ranges must not outlive it) and at the same context (reopen_same_context: each close
        # returns its memory), in both tiers: two reopens and a 32k prompt.
        first = ({"drafting", "lower", "same"} | ({"spread"} if spread else set())) if n == 0 else set()
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
    commit = records.crit_commit(ROOT)
    return records.write(report.items, out, name="stress", title=f"crit stress test at {commit} (cache key {cache.key})",
                         commit=commit, key=cache.key, seconds=round(time.perf_counter() - started, 1),
                         builds=cache.timings)


def environment_for(cache_root, shared):
    """The Python environment the stress test runs in, made once a checkout: a venv whose third-party packages are
    those of `shared` (a venv with torch and crit's dependencies, read through a .pth) and whose crit, crit-core,
    crit-models and crit-eval are this checkout's and the crit-eval beside it, installed editable. crit serve starts
    its engine host with the running interpreter, so the host imports this checkout's code too, never another
    install's."""
    import crit_eval
    key = hashlib.sha256(str(ROOT).encode()).hexdigest()[:12]
    venv = Path(cache_root) / "venvs" / key
    python = venv / "bin" / "python"
    eval_root = Path(crit_eval.__file__).resolve().parents[1]
    want = {"crit": str(ROOT), "crit-models": str(ROOT / "models" / "core" / "python"), "crit-eval": str(eval_root),
            "shared": str(shared)}
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
                    str(ROOT / "inference" / "core" / "python"), "-e", str(ROOT / "models" / "core" / "python"),
                    "-e", str(ROOT), "-e", str(eval_root)], check=True)
    marker.write_text(json.dumps(want), encoding="utf-8")
    return python


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--cache", default="/llm/sessions-cache/stress")
    parser.add_argument("--fixtures", default="/llm/models/crit/fixtures")
    parser.add_argument("--models", default="/llm/models/crit")
    parser.add_argument("--checkpoints", nargs="*", default=list(CHECKPOINTS))
    parser.add_argument("--only", nargs="*", help="run the loads whose NAME/MODE/PRESSURE contains one of these")
    parser.add_argument("--out", help="report folder (default: the cache's reports/KEY-TIME)")
    parser.add_argument("--build-only", action="store_true", help="build and cache everything, serve nothing")
    parser.add_argument("--tier", choices=("full", "quick"), default="full",
                        help="full (before a release): every scenario; quick: one load a checkpoint and the tight "
                             "pressure (plan_loads)")
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
            with gpu(args.gpu_lock, f"crit stress test {name}/{mode}/{pressure}"):
                load(report, cache, name, models[name], mode, pressure, data[name], extras, tests,
                     store_cache=not args.no_store_cache)
    summary = write_report(report, out, started, cache)
    log(f"report: {out / 'stress.md'} ({summary['passed']} passed, {summary['failed']} failed, "
        f"{summary['skipped']} skipped in {summary['seconds'] / 60:.1f} min)")
    return 1 if summary["failed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
