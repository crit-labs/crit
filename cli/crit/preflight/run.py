"""crit run ... preflight: the library at its defaults on the run's own checkpoint, before the build.

    crit run --project CRIT_ROOT --root RUN_ROOT --id RUN_ID preflight [--budget-minutes 15] [--gpu-lock FILE]

On a created run it cuts the run's checkpoint (cut's default cut: the first layer of every kind, the draft layer, the
lookup tables; hard links where a shard is kept whole) into the run's cache (RUN/.plugin-cache/preflight/REVISION/),
prepares crit-eval's data for the cut, packages a candidate of the library's plugins at their defaults for it
(candidate.library_selection: what a build starts from) in a run of its own beside it, and serves it in one load per
quality mode the run asks, the scenarios plan.plan_loads finds meaningful for the checkpoint running inside, every size
from the run. A load takes the machine's GPU lock (gpu_lock: --gpu-lock, else CRIT_GPU_LOCK, else settings.toml's
gpu_lock; empty for none) and crit's engine lock, and gives them back after. Each load prints its ETA, and a scenario
the time budget has no room for is left out and says so.

The report (preflight.json and preflight.md: scenario, state, seconds, detail) is kept in RUN/preflight/KEY/, KEY the
crit code and the checkpoint revision it ran (records.key). A failed scenario is toolkit feedback at the library's
defaults: what the build will meet first and may fix in its own candidate or report, never a gate the engine must
pass. The command records nothing else in the run and changes none of its records; it exits 0 when the report is
written, whatever it holds."""
import json
import os
from pathlib import Path
import time
import traceback

from . import candidate, plan, records, scenarios

GIB = 1 << 30


class StopPreflight(Exception):
    """The preflight cannot go on; its report's last item says why."""


def gpu_lock(given=None, settings=None):
    """The machine's shared GPU lock file: `given` (an option), else CRIT_GPU_LOCK (gpu-job sets it empty for the
    jobs it runs, which hold the lock already), else settings.toml's gpu_lock; '' for none."""
    if given is not None:
        return given
    if "CRIT_GPU_LOCK" in os.environ:
        return os.environ["CRIT_GPU_LOCK"]
    return (settings or {}).get("gpu_lock", "")


def companion_drafters(root, run):
    """The companion drafter folders the run names: its request's (hardware_constraints.context.companion_drafters)
    and those recon recorded (recon --drafter)."""
    from ..run.records import read_json
    found = [row.get("path") for row in ((run["spec"]["hardware_constraints"].get("context") or {})
                                         .get("companion_drafters") or [])]
    path = Path(root) / "work" / "recon.json"
    if path.is_file():
        found += [row.get("path") for row in (read_json(path).get("gathered") or {}).get("drafters") or []]
    return [path for path in dict.fromkeys(found) if path]


def placement_of(root, run):
    """crit-eval's placement of the whole model on the run's device at its context: recon.json's, else computed
    (crit.run.recon.placement); {"scenario": None, ...} where the run names no device file."""
    from ..run import recon
    from ..run.records import read_json
    path = Path(root) / "work" / "recon.json"
    if path.is_file():
        found = (read_json(path).get("gathered") or {}).get("placement")
        if found:
            return found
    return recon.placement(run)


def cut_basis(model, run):
    """crit-eval's sizing of the cut on the run's device at its context: (basis, kv_bytes_per_token), or None where the
    run names no device file or Eval cannot size the cut."""
    from ..run.records import read_json
    from ..run.store import run_device_file
    device = run_device_file(run)
    if not device or not Path(device).is_file():
        return None
    try:
        from crit_eval import EVAL_DIR
        from crit_eval.limits.context import GIB as EVAL_GIB, max_context
        settings = read_json(EVAL_DIR / "protocol.json")["context"]
    except (ImportError, OSError, KeyError):
        return None
    hardware = run["spec"]["hardware_constraints"]
    reserve = (hardware.get("context") or {}).get("reserve_bytes")
    if reserve is not None:
        settings = {**settings, "reserve_gib": reserve / EVAL_GIB}
    value = max_context(model, read_json(Path(device)), hardware["vram_headroom_bytes"], settings,
                        hardware["kv_tokens"])
    return (value["basis"], value.get("kv_bytes_per_token")) if value.get("basis") else None


def _request(run):
    """The request of the cut's run: the run's context, device file, headrooms and quality mode."""
    from ..run.store import run_device_file
    hardware = run["spec"]["hardware_constraints"]
    request = {"kv_tokens": hardware["kv_tokens"], "vram_headroom_gib": hardware["vram_headroom_bytes"] / GIB,
               "ram_headroom_gib": hardware["ram_headroom_bytes"] / GIB}
    device = run_device_file(run)
    if device and Path(device).is_file():
        request["device"] = device
    return request


def _built(cache, project, model, run, mode, selection, plugins, log):
    """The cut's run in `mode` with its library candidate, made once per cache (runs.json): (store, run_id, id)."""
    from ..run.store import RunStore
    index = cache / "runs.json"
    found = json.loads(index.read_text(encoding="utf-8")) if index.is_file() else {}
    store = RunStore(cache / "runs", project)
    if mode in found:
        return store, found[mode]["run"], found[mode]["candidate"]
    log(f"crit run preflight: creating the cut's {mode} run and packaging the library's plugins at their defaults")
    run_id, candidate_id = candidate.build_run(store, model, mode, cache / "sources" / mode, selection, plugins,
                                               {**_request(run), "quality_mode": mode},
                                               criteria=f"crit run preflight of {run['run_id']}")
    found[mode] = {"run": run_id, "candidate": candidate_id}
    index.write_text(json.dumps(found, indent=2), encoding="utf-8")
    return store, run_id, candidate_id


def preflight(store, run_id, budget_minutes=plan.BUDGET_MINUTES, lock=None, plugin_tests=False, log=scenarios.log):
    """crit run ... preflight on the run `run_id` of `store`: the report's summary ({key, report, passed, failed,
    skipped, ...}), its files written in RUN/preflight/KEY/."""
    from ..run import replay
    from ..run.records import load_run
    from ..settings import load as load_settings
    root = store.directory(run_id)
    replay.refuse(root)
    run = load_run(root)
    model = Path(run["spec"]["model_path"])
    if not records.cuttable(model):
        raise ValueError(f"{model} is no folder of safetensors weights with a config.json: the preflight cuts the "
                         "checkpoint's decoder layers, which a single weight file or a GGUF does not give it")
    lock = gpu_lock(lock, load_settings(store.project))
    started, revision = time.perf_counter(), records.checkpoint_revision(model)
    key = records.key(model)
    out = records.folder(root, key)
    cache = root / ".plugin-cache" / records.CACHE / revision
    report = scenarios.Report(log)
    previous = records.reports(root)
    budget = plan.Budget(budget_minutes * 60, plan.measured(previous[-1][1] if previous else None))
    extra = {"run_id": run_id, "model": str(model), "key": key, "crit_key": records.crit_key(),
             "commit": records.crit_commit(), "checkpoint_revision": revision, "budget_minutes": budget_minutes,
             "gpu_lock": lock or None, "started_at": records.now()}
    log(f"crit run preflight of {run_id} at {key}; cache {cache}")
    try:
        plugin = candidate.model_plugin(model)
        found = candidate.facts(model, plugin)
        extra["facts"] = found
        cut_folder = cache / "cut"
        if not (cut_folder / "layer_cut.json").is_file():
            began = time.perf_counter()
            if cut_folder.exists():
                import shutil
                shutil.rmtree(cut_folder)
            from .cut import DRAFT, cut
            summary = cut(model, cut_folder, None, True, (found["draft_prefix"],) if found["draft_prefix"] else DRAFT)
            report.add("cut", "info", time.perf_counter() - began,
                       f"decoder layers {summary['layers']} of the checkpoint, {summary['tensors']} tensors, "
                       f"{summary['bytes'] / GIB:.1f} GiB ({summary['bytes_written'] / GIB:.1f} GiB written, "
                       f"{len(summary['linked'])} shards hard-linked)")
        extra["cut"] = json.loads((cut_folder / "layer_cut.json").read_text(encoding="utf-8"))
        placement = placement_of(root, run)
        companions = companion_drafters(root, run)
        from ..models import companion_draft
        read = next((path for path in companions if companion_draft(plugin, path)), None)   # the plugin's own
        tables_off = ((placement or {}).get("lookup_tables") or {}).get("tier") not in (None, "device")
        selection = plugins = None
        with report.scenario("candidate") as item:
            item["state"] = "info"
            selection, plugins = candidate.library_selection(cut_folder, candidate.facts(cut_folder, plugin),
                                                             tables_off_device=tables_off, companion=bool(companions))
            item["detail"] = json.dumps(selection)
        extra["selection"] = selection
        if selection is None:   # no candidate to serve: the failed item says why (no model plugin serves it)
            raise StopPreflight
        data = None
        with report.scenario("eval_data") as item:
            item["state"] = "info"
            data = scenarios.prepare_data(cut_folder, cache / "eval-data")
            item["detail"] = f"crit eval prepare of the cut in {data}"
        if report.items[-1]["state"] == "failed":
            data = None
        planned = plan.plan_loads(run, found, placement, bool(companions), data is not None)
        for name, why in planned.left_out:
            report.add(name, "skipped", 0.0, "not planned: " + why)
        basis = cut_basis(cut_folder, run) if any(load.share is not None for load in planned.loads) else None
        for load in planned.loads:
            prefix = load.name + "/"
            eta = budget.estimate(load)
            if eta > budget.left() and load.share is not None:
                report.add(prefix + "load", "skipped", 0.0, f"{scenarios.OVER_BUDGET} (ETA about {eta / 60:.1f} min, "
                           f"{max(0.0, budget.left()) / 60:.1f} min left of {budget_minutes} min)")
                continue
            if load.precheck and budget.guess("precheck") + budget.guess("load") > budget.left():
                load.precheck = False
                report.add(prefix + "precheck", "skipped", 0.0, f"{scenarios.OVER_BUDGET} ({budget.describe('precheck')})")
            leave = None
            if load.share is not None:
                if basis is None:
                    report.add(prefix + "load", "skipped", 0.0, "Eval cannot size the cut on the run's device, so no "
                               "ballast can make the whole model's pressure")
                    continue
                leave = plan.leave_bytes(basis[0], basis[1], load.sizes.context, load.share)
                report.add(prefix + "ballast", "info", 0.0, f"the ballast leaves {leave / GIB:.1f} GiB free: the "
                           f"cut beside {load.share:.0%} of its {basis[0]['expert_bytes'] / GIB:.1f} GiB of routed "
                           "experts, the whole model's share on the device at the run's context")
            store_, cut_run, candidate_id = _built(cache, store.project, cut_folder, run, load.mode, selection, plugins, log)
            log(f"crit run preflight: load {load.name}, ETA about {eta / 60:.1f} min "
                f"({max(0.0, budget.left()) / 60:.1f} min left of {budget_minutes} min)")
            with scenarios.gpu(lock, f"crit run preflight {run_id} {load.name}", log=log):
                scenarios.load(report, store_, cut_run, candidate_id, prefix, load.extras, load.sizes,
                               data if load.precheck else None, plugin_tests, cut=True, leave=leave,
                               plugin_cache=str(cache / "plugin-cache"), budget=budget, smoke=load.smoke,
                               companion=read)
    except StopPreflight:
        pass
    except Exception as error:   # what stopped the preflight is its report's last item
        traceback.print_exc()
        report.add("preflight", "failed", time.perf_counter() - started, f"{type(error).__name__}: {error}")
    value = records.write(report.items, out, title=f"crit preflight of {run_id} at {extra['commit']} ({key})",
                          seconds=round(time.perf_counter() - started, 1), finished_at=records.now(), **extra)
    return {"key": key, "report": str(out / f"{records.REPORT}.md"),
            **{state: value[state] for state in ("passed", "failed", "skipped")},
            "failed_scenarios": [item["scenario"] for item in value["items"] if item["state"] == "failed"],
            "seconds": value["seconds"]}


def render(found):
    """crit run ... preflight's result as the lines it prints on stderr."""
    lines = [f"preflight {found['key']}: {found['passed']} passed, {found['failed']} failed, {found['skipped']} "
             f"skipped in {found['seconds'] / 60:.1f} min; {found['report']}"]
    if found["failed"]:
        lines.append("Failed at the library's defaults (toolkit feedback, not a gate; the build may go on): "
                     + ", ".join(found["failed_scenarios"]))
    return "\n".join(lines)
