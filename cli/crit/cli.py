"""Small product CLI; implementation code is never imported by navigation."""
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]


def main(argv=None):
    # Route subcommands with their own parsers before the maintenance parser.
    arguments = list(sys.argv[1:] if argv is None else argv)
    if arguments and arguments[0] == "run":
        from .run.cli import main as run_main
        return run_main(arguments[1:])
    if arguments and arguments[0] == "serve":
        from crit_api.cli import main as serve_main
        return serve_main(arguments[1:])
    if arguments and arguments[0] == "kernels":
        from crit_kernels import main as kernels_main
        return kernels_main(arguments[1:])
    if arguments and arguments[0] == "models":
        from crit_models import main as models_main
        return models_main(arguments[1:])
    if arguments and arguments[0] == "eval":
        try:
            from crit_eval.cli import main as eval_main
        except ImportError:
            raise SystemExit("crit eval needs crit Eval (https://github.com/crit-labs/crit-eval): install it with "
                             "python -m pip install -e ../crit-eval beside this checkout") from None
        return eval_main(arguments[1:])
    parser = argparse.ArgumentParser(description="crit Core, Inference, Kernels, Models, Eval, run records and the serving API")
    from . import __version__
    parser.add_argument("--version", action="version", version="crit " + __version__)
    subs = parser.add_subparsers(dest="command", required=True)
    subs.add_parser("contracts", help="Print native core contract locations")
    subs.add_parser("kernels", help="Inspect kernel applicability and retain verified dependency bytes")
    subs.add_parser("models", help="Transform model weights (PTQ) and record how; use crit models --help")
    subs.add_parser("serve", help="Serve a sealed candidate as an OpenAI-compatible API; use crit serve --help")
    subs.add_parser("run", help="Publish retained run/candidate records; use crit run --help")
    subs.add_parser("eval", help="Run the common speed, quality and exactness suite (crit Eval); use crit eval --help")
    sub = subs.add_parser("inference", help="Query or check the inference techniques")
    sub.add_argument("arguments", nargs=argparse.REMAINDER)
    args = parser.parse_args(arguments)
    repository = ROOT if (ROOT / "core/include/crit/core.h").is_file() else Path.cwd()
    if not (repository / "core/include/crit/core.h").is_file():
        parser.error("Run repository maintenance commands from the crit repository")
    if args.command == "inference":
        from . import inference
        return inference.main(args.arguments, repository)
    print("core/include/crit/core.h\ncore/include/crit/cuda.h\ncore/contracts/README.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
