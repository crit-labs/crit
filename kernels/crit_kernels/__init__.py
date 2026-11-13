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
    tune.add_argument("--matrices", type=Path, help="with --model: the matrix shapes the engine loaded instead of the "
                      "checkpoint's tensors and their likely stacks: a candidate's serve --check report "
                      "(logs/serve-check-attempt-N.json), an engine's facts, or a list of {N, K, format}")
    gate = sub.add_parser("gates", help="Run the gates beyond the reference check over a kernel's structures: NaN and "
                                        "infinities, repeated calls and untouched inputs, the rows at tile edges, FP32 "
                                        "sums, and every architecture's register and shared-memory limits, each gate "
                                        "proven by a planted control (--controls)")
    gate.add_argument("kernel", nargs="?")
    gate.add_argument("--model", type=Path, help="the checkpoint folder whose shapes to run")
    gate.add_argument("--shapes", type=Path, help="instead of --model: a JSON file {structure: [shape, ...]}")
    gate.add_argument("--structure", action="append", help="only this structure (repeatable)")
    gate.add_argument("--tuning", type=Path, help="a tuning file crit kernels sweep wrote (default: the default knobs)")
    gate.add_argument("--arch", action="append", help="the limits gate's architectures (default: the kernel.json "
                                                      "capabilities; nvcc names)")
    gate.add_argument("--no-limits", action="store_true", help="leave out the limits gate (it builds for every architecture)")
    gate.add_argument("--limits-only", action="store_true", help="only the limits gate: build for every architecture "
                                                                  "and check each one's limits, with no GPU (CI's check)")
    gate.add_argument("--controls", action="store_true", help="run the planted controls instead: each must fail its gate")
    rep = sub.add_parser("report", help="Every kernel's gates and speed against its yardstick library on the shapes of "
                                        "the models given (each structure's distinct shapes over them, at most --most)")
    rep.add_argument("kernel", nargs="*", help="only these kernels (default: all of crit's)")
    rep.add_argument("--model", type=Path, action="append", required=True, help="a checkpoint folder (repeatable)")
    rep.add_argument("--most", type=int, default=3, help="shapes a structure: the smallest, middle and largest")
    rep.add_argument("--no-gates", action="store_true")
    rep.add_argument("--no-speed", action="store_true")
    rep.add_argument("--limits", action="store_true", help="also the limits gate (builds for every architecture)")
    rep.add_argument("--repeat", type=int, default=25)
    rep.add_argument("--out", type=Path, help="write the JSON report here")
    rep.add_argument("--markdown", type=Path, help="write the markdown summary here")
    return parser


def main(argv=None):
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    command = parser()
    args = command.parse_args(argv)
    try:
        if args.command is None:
            print("crit Kernels: crit's kernel templates (list, build, sweep, gates, report) and the kernel workbench (new)")
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
            if args.matrices and not args.model:
                raise ValueError("--matrices replaces the checkpoint's matrices: name the checkpoint too (--model)")
            from .model import loaded_matrices
            matrices = loaded_matrices(args.matrices) if args.matrices else None
            tuning, report = sweep(args.kernel, model=args.model, shapes=shapes, structures=args.structure,
                                   out=args.out, jobs=args.jobs, lock=args.lock, repeat=args.repeat, chunk=args.chunk,
                                   matrices=matrices, log=lambda line: print(line, file=sys.stderr))
            result = {"tuning": str(args.out / f"{args.kernel}.json"), "report": report}
        elif args.command == "gates":
            from .gates import controls, gates
            if args.controls:
                result = controls()
                print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False))
                return 0 if all(c["ok"] for c in result.values()) else 1
            if not args.kernel:
                raise ValueError("name the kernel (or --controls)")
            if args.limits_only:
                from .gates import gate_limits
                limits = gate_limits(args.kernel, args.arch)
                result = {"kernel": args.kernel, "structures": {}, "limits": limits,
                          "passed": limits["verdict"] != "fail"}
                print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False))
                return 0 if result["passed"] else 1
            if not args.model and not args.shapes:
                raise ValueError("name the checkpoint (--model) or the shapes (--shapes)")
            from .build import read_tuning
            result = gates(args.kernel, model=args.model, shapes=read_json(args.shapes) if args.shapes else None,
                           structures=args.structure, tuning=read_tuning(args.tuning) if args.tuning else None,
                           archs=args.arch, limits=not args.no_limits, log=lambda line: print(line, file=sys.stderr))
            print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False))
            return 0 if result["passed"] else 1
        elif args.command == "report":
            from .report import markdown, report
            result = report(args.model, args.kernel or None, args.most, not args.no_gates, not args.no_speed,
                            args.limits, args.repeat, log=lambda line: print(line, file=sys.stderr))
            if args.out:
                args.out.write_text(json.dumps(result, indent=2, ensure_ascii=False, default=str), encoding="utf-8",
                                    newline="\n")
            text = markdown(result)
            if args.markdown:
                args.markdown.write_text(text, encoding="utf-8", newline="\n")
            print(text)
            return 0 if all(e.get("passed", True) for e in result["kernels"].values()) else 1
        print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False))
        return 0
    except (ValueError, OSError, TypeError, KeyError, RuntimeError) as exc:
        print("crit kernels: " + str(exc), file=sys.stderr)
        return 2
