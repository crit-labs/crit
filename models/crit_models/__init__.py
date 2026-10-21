"""crit models: produce transformed weights (PTQ first) and record how they were made.

`plan` reads only checkpoint headers (standard library). `ptq` needs torch and safetensors, and with gptq also
transformers; `eval` runs crit Eval from a checkout. Nothing is installed or downloaded.
"""
import argparse
import json
from pathlib import Path
import sys

from crit.run.records import read_json


def _recipe(path):
    from .recipe import load_recipe
    return load_recipe(read_json(Path(path)))


def main(argv=None):
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(prog="crit models", description="Transform model weights (post-training "
                                     "quantization) and record recipe, calibration, seed, hashes and cost")
    subs = parser.add_subparsers(dest="command", required=True)
    plan = subs.add_parser("plan", help="Resolve a recipe against a checkpoint's headers: targets and bytes")
    plan.add_argument("--recipe", required=True, help="models_recipe JSON")
    plan.add_argument("--source", required=True, help="Unquantized source model directory")
    plan.add_argument("--json", action="store_true", help="Print the resolved recipe and plan as JSON")
    ptq = subs.add_parser("ptq", help="Quantize a checkpoint and write its models_transform record")
    ptq.add_argument("--recipe", required=True, help="models_recipe JSON")
    ptq.add_argument("--source", required=True, help="Unquantized source model directory")
    ptq.add_argument("--out", required=True, help="New or empty directory for the transformed checkpoint")
    ptq.add_argument("--record", required=True, help="New models_transform JSON outside the source and output")
    ptq.add_argument("--calibration", help="JSONL of {\"text\": ...} documents unrelated to Eval (gptq only)")
    ptq.add_argument("--device", help="torch device (default: cuda when available, else cpu)")
    evaluation = subs.add_parser("eval", help="Bind the Eval result of an engine serving the output to its record, "
                                              "with its quality replies scored on the source checkpoint")
    evaluation.add_argument("--record", required=True, help="models_transform JSON")
    evaluation.add_argument("--data", required=True, help="Eval data directory prepared from the source checkpoint")
    evaluation.add_argument("--result", required=True, help="crit eval run --url result of the engine serving the "
                                                            "output; stop the server first")
    evaluation.add_argument("--evaluation", required=True, help="New models_evaluation JSON")
    evaluation.add_argument("--baseline", metavar="RESULT", help="Eval result of the engine on the source checkpoint")
    check = subs.add_parser("verify", help="Re-hash a transform's output and source against its record")
    check.add_argument("--record", required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "plan":
            from crit_core.model import inspect_model
            from .recipe import plan as make_plan
            recipe = _recipe(args.recipe)
            resolved = make_plan(recipe, inspect_model(args.source))
            if args.json:
                print(json.dumps({"recipe": recipe, "plan": resolved}, indent=1))
            else:
                for target in resolved["targets"]:
                    print(f"{target['module']}  {target['dtype']} {target['shape'][0]}x{target['shape'][1]}")
                print(f"{len(resolved['targets'])} modules: {resolved['stored_bytes'] / 2**30:.3f} GiB stored -> "
                      f"{resolved['transformed_bytes'] / 2**30:.3f} GiB {recipe['format']}; checkpoint "
                      f"{resolved['checkpoint_bytes'] / 2**30:.3f} GiB -> "
                      f"{(resolved['checkpoint_bytes'] - resolved['stored_bytes'] + resolved['transformed_bytes']) / 2**30:.3f} GiB")
            return 0
        if args.command == "ptq":
            from .ptq import run_ptq
            run_ptq(_recipe(args.recipe), args.source, args.out, args.record, args.calibration, args.device)
            return 0
        if args.command == "eval":
            from crit_eval import SUITE_DIR, load_json
            if not SUITE_DIR.is_dir():
                raise ValueError("crit models eval runs Eval from a crit checkout")
            from crit_eval.status import render
            from .evaluate import evaluate
            result = evaluate(args.record, args.data, args.result, args.evaluation,
                              load_json(args.baseline) if args.baseline else None)
            print(render(load_json(args.result)["status"]))
            print(f"wrote {args.evaluation} (gates: " + ", ".join(f"{k} {v}" for k, v in result["summary"]["gates"].items()) + ")")
            return 0
        from .evaluate import load_record
        from .ptq import verify
        findings = verify(load_record(args.record))
        for finding in findings:
            print(finding)
        print("output and source match the record" if not findings else f"{len(findings)} findings")
        return 1 if findings else 0
    except ValueError as error:
        raise SystemExit(f"crit models: {error}") from None
