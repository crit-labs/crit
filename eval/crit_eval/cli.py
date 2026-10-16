"""`crit eval` command line. Also runs as `python eval/crit_eval` from a crit checkout."""
import argparse
import json
from pathlib import Path
import sys


def main(argv=None):
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(prog="crit eval", description="Run the common crit speed, quality and speculative-exactness suite and report the engine against its ceilings")
    subs = parser.add_subparsers(dest="command", required=True)

    run = subs.add_parser("run", help="Evaluate one engine (prepares data and reference first when missing)")
    run.add_argument("--adapter", required=True, help="Python file exporting create_eval_engine(context)")
    run.add_argument("--model", required=True, help="Model directory passed to the adapter")
    run.add_argument("--data", required=True, help="Prepared suite and reference directory for this model")
    run.add_argument("--out", required=True, help="Result JSON path")
    run.add_argument("--config", default="{}", help="JSON configuration overrides passed to the adapter")
    run.add_argument("--only", choices=["speed", "quality", "exactness", "context", "heldout"], action="append",
                     help="Run only these parts (repeatable); default is all. context: the long-context curve up to "
                          "the required context and the speed cases again at a lower context")
    run.add_argument("--repetitions", type=int, help="Override the protocol's measured repetitions")
    run.add_argument("--warmups", type=int, help="Override the protocol's warmup runs")
    run.add_argument("--label", help="Name shown in comparisons")
    run.add_argument("--device", metavar="JSON", help="Device ceilings from `crit eval probe` (bandwidth and GEMM rates)")
    run.add_argument("--bandwidth", type=float, metavar="GBS",
                     help="Achievable device memory bandwidth in GB/s; overrides the --device value")
    run.add_argument("--baseline", metavar="RESULT", help="Earlier result of the same data to show KPI changes against")
    run.add_argument("--quality-budget", metavar="JSON",
                     help='Drift limits over the protocol defaults, e.g. \'{"ppl_ratio_max": 1.02}\'; '
                          'keys ppl_ratio_max, kld_mean_max, kld_p99_max')
    run.add_argument("--exactness", choices=["reported", "required"],
                     help="reported (default): agreement with plain greedy is shown and the quality budget judges "
                          "the output; required: any difference fails the gate")
    run.add_argument("--context", type=int, metavar="TOKENS",
                     help="Context (prompt plus output) every engine is built and measured for: the run's kv_tokens. "
                          "Default: Eval's max context for the model on the --device")
    run.add_argument("--context-reserve-gib", type=float, default=None,
                     help="Device memory kept out of the KV cache when Eval computes the max context: activations, "
                          "workspaces, graphs and speed techniques (default: settings.toml context_reserve_gib, else the "
                          "protocol's reserve_gib)")
    run.add_argument("--vram-headroom-gib", type=float, default=None,
                     help="Device memory left free when Eval computes the max context (default 1)")

    context = subs.add_parser("context", help="The max context for a model on a device: the model's positions and "
                                              "the KV cache that fits next to its weights")
    context.add_argument("--model", required=True, help="Model directory")
    context.add_argument("--device", required=True, metavar="JSON", help="Device file from `crit eval probe`")
    context.add_argument("--vram-headroom-gib", type=float, default=None,
                         help="Device memory left free (default: settings.toml vram_headroom_gib, else 1)")
    context.add_argument("--context-reserve-gib", type=float, default=None,
                         help="Device memory kept out of the KV cache for workspaces and speed techniques "
                              "(default: settings.toml context_reserve_gib, else the protocol's reserve_gib)")
    context.add_argument("--json", action="store_true", help="Print the computation as JSON")

    probe = subs.add_parser("probe", help="Measure device bandwidth and GEMM rates per weight format (needs nvcc)")
    probe.add_argument("--out", required=True, help="Device JSON path for `run --device`")
    probe.add_argument("--nvcc", help="nvcc to build the probe with (default: PATH, CUDA_HOME, /usr/local/cuda*)")
    probe.add_argument("--repeats", type=int, default=3, help="Probe runs; each value is the median")

    status = subs.add_parser("status", help="Gates, goals, KPIs against their physical limits, gaps and missing inputs")
    status.add_argument("result")
    status.add_argument("--baseline", metavar="RESULT", help="Earlier result to show KPI changes against")
    status.add_argument("--json", action="store_true", help="Print the status as JSON")

    prepare = subs.add_parser("prepare", help="Freeze the suite as token ids for one model")
    prepare.add_argument("--model", required=True)
    prepare.add_argument("--data", required=True)

    reference = subs.add_parser("reference", help="Build reference log-probabilities (torch + transformers, GPU)")
    reference.add_argument("--model", required=True)
    reference.add_argument("--data", required=True)

    compare = subs.add_parser("compare", help="Print a side-by-side table of result files")
    compare.add_argument("results", nargs="+")

    pin = subs.add_parser("pin", help="Record one engine's plain greedy output of every speed case as the data "
                                      "directory's pinned continuation (decode is then measured on that text)")
    pin.add_argument("--adapter", required=True, help="Python file exporting create_eval_engine(context)")
    pin.add_argument("--model", required=True, help="Model directory passed to the adapter")
    pin.add_argument("--data", required=True, help="Prepared data directory to pin the continuation in")
    pin.add_argument("--config", default="{}", help="JSON configuration overrides passed to the adapter")
    pin.add_argument("--context", type=int, metavar="TOKENS", help="Context the engine is built for (the run's kv_tokens)")

    ab = subs.add_parser("ab", help="Alternate two engines' speed runs in one session and report paired ratios")
    ab.add_argument("--adapter", required=True, help="Adapter of engine A")
    ab.add_argument("--config", default="{}", help="JSON configuration of engine A")
    ab.add_argument("--adapter-b", help="Adapter of engine B (default: engine A's adapter)")
    ab.add_argument("--config-b", default="{}", help="JSON configuration of engine B")
    ab.add_argument("--model", required=True)
    ab.add_argument("--data", required=True)
    ab.add_argument("--out", required=True, help="Result JSON path")
    ab.add_argument("--rounds", type=int, default=2, help="Rounds; each round runs both engines, order alternating")
    ab.add_argument("--repetitions", type=int, help="Measured repetitions per engine and round")
    ab.add_argument("--warmups", type=int, help="Warmup runs per engine and round")
    ab.add_argument("--context", type=int, metavar="TOKENS", help="Context both engines are built for (the run's kv_tokens)")

    kernels = subs.add_parser("kernels", help="Trace every kernel of the speed cases (CUPTI, Linux) and rank them "
                                              "per phase by time, launch width and share of peak")
    kernels.add_argument("--adapter", required=True, help="Python file exporting create_eval_engine(context)")
    kernels.add_argument("--model", required=True, help="Model directory passed to the adapter")
    kernels.add_argument("--data", required=True, help="Prepared suite directory for this model")
    kernels.add_argument("--out", required=True, help="Kernel report JSON path")
    kernels.add_argument("--config", default="{}", help="JSON configuration overrides passed to the adapter")
    kernels.add_argument("--case", action="append", help="Trace only these speed cases (repeatable); default all")
    kernels.add_argument("--device", metavar="JSON", help="Device ceilings from `crit eval probe` for share of peak")
    kernels.add_argument("--nvcc", help="nvcc of a CUDA toolkit that ships CUPTI")
    kernels.add_argument("--top", type=int, default=12, help="Kernels listed per phase")
    kernels.add_argument("--context", type=int, metavar="TOKENS", help="Context the engine is built for (the run's kv_tokens)")

    args = parser.parse_args(argv)
    if args.command == "kernels":
        from . import load_json as load
        from .kernels import render as render_kernels, trace
        from .prepare import PREPARED
        if not (Path(args.data) / PREPARED).is_file():
            from .prepare import prepare as do_prepare
            do_prepare(args.model, args.data)
        result = trace(args.adapter, args.model, args.data, args.out, json.loads(args.config), args.case, args.nvcc,
                       load(args.device) if args.device else None, context_tokens=args.context)
        print(f"wrote {args.out}")
        print(render_kernels(result, args.top))
        return 0
    if args.command == "prepare":
        from .prepare import prepare as do_prepare
        prepared = do_prepare(args.model, args.data)
        print(f"prepared {len(prepared['speed'])} speed cases and {len(prepared['quality'])} quality conversations in {args.data}")
        return 0
    if args.command == "reference":
        return _reference(args.model, args.data)
    if args.command == "context":
        from . import load_json as load
        from .context import GIB, max_context, render as render_context
        settings = context_settings(load(Path(__file__).resolve().parents[1] / "protocol.json")["context"], args)
        value = max_context(args.model, load(args.device), headroom_gib(args) * GIB, settings)
        print(json.dumps(value, indent=1) if args.json else render_context(value))
        return 0 if value["tokens"] else 1
    if args.command == "compare":
        from .report import compare as do_compare
        print(do_compare(args.results))
        return 0
    if args.command == "pin":
        from .pin import CONTINUATIONS, pin as do_pin
        from .prepare import PREPARED
        if not (Path(args.data) / PREPARED).is_file():
            from .prepare import prepare as do_prepare
            do_prepare(args.model, args.data)
        pinned = do_pin(args.adapter, args.model, args.data, json.loads(args.config), context_tokens=args.context)
        print(f"pinned {len(pinned['cases'])} speed cases in {Path(args.data) / CONTINUATIONS} (sha256 {pinned['sha256'][:12]})")
        return 0
    if args.command == "ab":
        from .ab import render as render_ab, run_ab
        result = run_ab(args.adapter, json.loads(args.config), args.adapter_b or args.adapter, json.loads(args.config_b),
                        args.model, args.data, args.out, args.rounds, args.warmups, args.repetitions,
                        context_tokens=args.context)
        print(f"wrote {args.out}")
        print(render_ab(result))
        return 0
    if args.command == "probe":
        from .probe import probe as do_probe
        device = do_probe(args.out, args.nvcc, args.repeats)
        print(f"wrote {args.out}: {device['bandwidth_gbs']:.0f} GB/s, "
              + ", ".join(f"{k} {v:.0f} TFLOPS" for k, v in device["tflops"].items())
              + (f", {device['memory_free_bytes'] / 1e9:.1f} of {device['memory_total_bytes'] / 1e9:.1f} GB free"
                 if device.get("memory_free_bytes") and device.get("memory_total_bytes") else ""))
        return 0
    if args.command == "status":
        from . import load_json as load
        from .status import render, status_of
        protocol = load(Path(__file__).resolve().parents[1] / "protocol.json")
        current = status_of(args.result, protocol, args.baseline)
        print(json.dumps(current, indent=1) if args.json else render(current))
        return 0
    from .prepare import PREPARED
    from .reference import REFERENCE
    from .runner import run as do_run
    parts = tuple(args.only or ("speed", "quality", "exactness", "context", "heldout"))
    data = Path(args.data)
    if not (data / PREPARED).is_file():
        from .prepare import prepare as do_prepare
        do_prepare(args.model, data)
    if "quality" in parts and not (data / REFERENCE).is_file():
        _reference(args.model, data)
    from . import load_json as load
    from .status import render
    device = load(args.device) if args.device else None
    baseline = load(args.baseline) if args.baseline else None
    result = do_run(args.adapter, args.model, data, args.out, json.loads(args.config), parts,
                    args.warmups, args.repetitions, args.label, bandwidth_gbs=args.bandwidth, device=device,
                    baseline=baseline, quality_budget=json.loads(args.quality_budget) if args.quality_budget else None,
                    exactness=args.exactness, context_tokens=args.context,
                    vram_headroom_bytes=round(headroom_gib(args) * (1 << 30)),
                    context_reserve_bytes=round(reserve_gib(args) * (1 << 30)) if reserve_gib(args) is not None else None)
    print(f"wrote {args.out}")
    print(render(result["status"]))
    return 0


def headroom_gib(args):
    """--vram-headroom-gib, else settings.toml's vram_headroom_gib, else 1 GiB."""
    from . import machine_settings
    return args.vram_headroom_gib if args.vram_headroom_gib is not None else machine_settings().get("vram_headroom_gib", 1.0)


def reserve_gib(args):
    """--context-reserve-gib, else settings.toml's context_reserve_gib, else None (the protocol's reserve)."""
    from . import machine_settings
    return args.context_reserve_gib if args.context_reserve_gib is not None else machine_settings().get("context_reserve_gib")


def context_settings(settings, args):
    reserve = reserve_gib(args)
    return settings if reserve is None else {**settings, "reserve_gib": reserve}


def _reference(model, data):
    from . import load_json
    from .reference import build_reference
    protocol = load_json(Path(__file__).resolve().parents[1] / "protocol.json")
    out = build_reference(model, data, protocol["quality"]["top_k"])
    print(f"reference perplexity {out['perplexity']:.4f} over {sum(len(c['targets']) for c in out['conversations'])} targets")
    return 0
