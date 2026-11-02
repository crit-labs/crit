"""Small product CLI; implementation code is never imported by navigation."""
import argparse
import importlib
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]

# The command families crit routes to: name, module (with main(argv), and parser() for crit mcp's tools) and help.
FAMILIES = (
    ("run", "crit.run.cli", "Publish retained run/candidate records; use crit run --help"),
    ("eval", "crit_eval.cli", "Run the common speed, quality and exactness suite (crit Eval); use crit eval --help"),
    ("serve", "crit_api.cli", "Serve a sealed candidate as an OpenAI-compatible API; use crit serve --help"),
    ("pull", "crit.pull.cli", "Install a crit-hub engine for crit serve --engine; use crit pull --help"),
    ("bench", "crit.pull.bench", "Measure an installed engine with crit Eval beside its card; use crit bench --help"),
    ("hub", "crit.pull.report", "Add a crit bench result to its crit-hub card (crit hub report); use crit hub --help"),
    ("kernels", "crit_kernels", "Inspect kernel applicability and retain verified dependency bytes"),
    ("models", "crit_models", "Transform model weights (PTQ) and record how; use crit models --help"),
    ("mcp", "crit_mcp.server", "Serve these commands to an agent as MCP tools over stdio; use crit mcp --help"),
)


def default_project():
    """The crit repository a command uses unless it names one: this checkout when crit runs from one, else the
    current folder."""
    return ROOT if (ROOT / "core/include/crit/core.h").is_file() else Path.cwd()


def family(module):
    """A command family's module; without crit Eval, crit eval says how to install it."""
    try:
        return importlib.import_module(module)
    except ImportError:
        if module.startswith("crit_eval"):
            raise SystemExit("crit eval needs crit Eval (https://github.com/crit-labs/crit-eval): install it with "
                             "python -m pip install -e ../crit-eval beside this checkout") from None
        raise


def main(argv=None):
    # Route subcommands with their own parsers before the maintenance parser.
    arguments = list(sys.argv[1:] if argv is None else argv)
    routes = {name: module for name, module, _ in FAMILIES}
    if arguments and arguments[0] in routes:
        return family(routes[arguments[0]]).main(arguments[1:])
    parser = argparse.ArgumentParser(description="crit Core, Inference, Kernels, Models, Eval, run records and the serving API")
    from . import __version__
    parser.add_argument("--version", action="version", version="crit " + __version__)
    subs = parser.add_subparsers(dest="command", required=True)
    subs.add_parser("contracts", help="Print native core contract locations")
    for name, _, text in FAMILIES:
        subs.add_parser(name, help=text)
    sub = subs.add_parser("inference", help="Query or check the inference techniques")
    sub.add_argument("arguments", nargs=argparse.REMAINDER)
    args = parser.parse_args(arguments)
    repository = default_project()
    if not (repository / "core/include/crit/core.h").is_file():
        parser.error("Run repository maintenance commands from the crit repository")
    if args.command == "inference":
        from . import inference
        return inference.main(args.arguments, repository)
    print("core/include/crit/core.h\ncore/include/crit/cuda.h\ncore/contracts/README.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
