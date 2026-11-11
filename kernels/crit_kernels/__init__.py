"""crit's kernels: templates with knobs, built and tuned on the engine's device (list, build, sweep), and the
workbench for a kernel an engine writes (new). Nothing is downloaded."""
import argparse
import json
from pathlib import Path
import sys

from crit.records import read_json


def parser():
    """The arguments of crit kernels (crit mcp offers its tools from them)."""
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command")
    make = sub.add_parser("new", help="Scaffold a workbench folder for a kernel: reference, shapes, yardstick, harness "
                                      "(measure it with crit eval workbench)")
    make.add_argument("name")
    make.add_argument("--op", required=True, help="gemv or gemm (a working start in bf16, fp8 or nvfp4), or any other "
                                                   "operation name for a harness to fill in")
    make.add_argument("--format", dest="fmt", help="bf16, fp8 or nvfp4 (NVFP4 weights against bf16 activations) for gemv and gemm")
    make.add_argument("--shape", action="append", default=[], help="N,K (repeatable)")
    make.add_argument("--model", type=Path, help="take the shapes from this checkpoint's matrices in the format")
    make.add_argument("--columns", help="column (token) counts, comma-separated")
    make.add_argument("--from", dest="source", type=Path,
                      help="a kernel project to start src/ from (one of crit's kernels in kernels/techniques/<family>/<id>, a "
                           "provider's kernel "
                           "source); without it, kernel-builder init starts one")
    make.add_argument("--yardstick", help="the provider to time beside the kernel (provider syntax of plan levers)")
    make.add_argument("--column-invariant", choices=("yes", "no"))
    make.add_argument("--repeatable", choices=("yes", "no"), default="yes",
                      help="no: the kernel may give other bits on a repeat (atomics, a split the workload chooses); "
                           "the check judges repeats within the tolerance. For quality mode speed engines")
    make.add_argument("--tolerance", type=float)
    make.add_argument("--root", type=Path, default=Path("work/kernels"))
    make.add_argument("--starter", action="store_true",
                      help="start from crit's starter matmul (kernels/starters/workbench-starter, or the kernel --from "
                           "names: a simple, correct "
                           "gemv/gemm in bf16 and fp8); allowed from a run's fourth candidate on")
    make.add_argument("--run", type=Path, help="the run folder, for --starter")
    sub.add_parser("list", help="crit's kernels (kernels/techniques/<family>/<id>/kernel.json): their operations and structures")
    built = sub.add_parser("build", help="Build one of crit's kernels for this machine's GPU (with a tuning table, or "
                                         "its default knobs) and print where it went")
    built.add_argument("kernel")
    built.add_argument("--tuning", type=Path, help="a tuning file crit kernels sweep wrote")
    built.add_argument("--verbose", action="store_true")
    built.add_argument("--arch", action="append", help="compile for this architecture instead of the current GPU's, "
                                                      "to check the build (repeatable; nvcc names: 80, 86, 89, 90a, "
                                                      "100a, 120a)")
    tune = sub.add_parser("sweep", help="Tune one of crit's kernels for a model on this GPU: build every knob "
                                        "configuration, check each against the reference, time it on the model's "
                                        "shapes, and write the tuning table")
    tune.add_argument("kernel")
    tune.add_argument("--model", type=Path, help="the checkpoint folder whose shapes to tune for")
    tune.add_argument("--shapes", type=Path, help="instead of --model: a JSON file {structure: [shape, ...]}")
    tune.add_argument("--structure", action="append", help="sweep only this structure (repeatable)")
    tune.add_argument("--out", type=Path, required=True, help="folder for <kernel>.json and <kernel>.report.json")
    tune.add_argument("--jobs", type=int, help="parallel builds (default: CPU count less 2)")
    tune.add_argument("--lock", type=Path, help="a file to hold an exclusive lock on while timing (a shared GPU)")
    tune.add_argument("--repeat", type=int, default=25)
    tune.add_argument("--chunk", type=int, default=24, help="configurations a build holds (each picked at run time)")
    return parser


def main(argv=None):
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    command = parser()
    args = command.parse_args(argv)
    try:
        if args.command is None:
            print("crit Kernels: crit's kernel templates (list, build, sweep) and the kernel workbench (new)")
            print("Guide: kernels/README.md in the crit repository")
            print("Use kernels --help. Nothing is downloaded; build and sweep compile crit's kernel sources here.")
            return 0
        if args.command == "new":
            from .workbench import new
            result = new(args.name, args.op, args.root, args.fmt, args.shape, args.model,
                         [int(c) for c in args.columns.split(",")] if args.columns else None, args.source, args.yardstick,
                         None if args.column_invariant is None else args.column_invariant == "yes", args.tolerance,
                         args.starter, args.run, repeatable=args.repeatable == "yes")
        elif args.command == "list":
            from .build import manifests
            result = {kid: {"family": m["family"], "title": m.get("title"), "language": m["language"],
                            "operations": m.get("operations", []), "structures": sorted(m.get("structures", {})),
                            "folder": str(folder)} for kid, (folder, m) in manifests().items()}
        elif args.command == "build":
            from .build import build, read_tuning
            out, package, key = build(args.kernel, tuning=read_tuning(args.tuning) if args.tuning else None,
                                      verbose=args.verbose, archs=args.arch)
            result = {"kernel": args.kernel, "folder": str(out), "package": package, "build": key}
        elif args.command == "sweep":
            from .sweep import sweep
            if not args.model and not args.shapes:
                raise ValueError("name the checkpoint (--model) or the shapes (--shapes)")
            shapes = read_json(args.shapes) if args.shapes else None
            tuning, report = sweep(args.kernel, model=args.model, shapes=shapes, structures=args.structure,
                                   out=args.out, jobs=args.jobs, lock=args.lock, repeat=args.repeat, chunk=args.chunk,
                                   log=lambda line: print(line, file=sys.stderr))
            result = {"tuning": str(args.out / f"{args.kernel}.json"), "report": report}
        print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False))
        return 0
    except (ValueError, OSError, TypeError, KeyError, RuntimeError) as exc:
        print("crit kernels: " + str(exc), file=sys.stderr)
        return 2
