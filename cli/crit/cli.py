"""Small product CLI; implementation code is never imported by navigation."""
import argparse
from contextlib import contextmanager
import importlib
import os
from pathlib import Path
import sys
from typing import NamedTuple

ROOT = Path(__file__).resolve().parents[2]
# crit's repository: crit X.Y.Z is published from its tag vX.Y.Z (.github/workflows/release.yml), whose AGENTS.md,
# docs/ and READMEs are the guides of an installed crit.
REPOSITORY = "https://github.com/crit-labs/crit"
# The variable that names crit's project to crit-eval, which reads the machine's settings from that project's
# settings.toml (crit_eval.machine_settings): the commands that run Eval name their project while they run.
PROJECT_VARIABLE = "CRIT_PROJECT"

# The command families crit routes to: name, module (with main(argv), and parser() for crit mcp's tools) and help.
FAMILIES = (
    ("run", "crit.run.cli", "Publish retained run/candidate records; use crit run --help"),
    ("eval", "crit_eval.cli", "Run the common speed, quality and exactness suite (crit Eval); use crit eval --help"),
    ("serve", "crit_api.cli", "Serve a sealed candidate as an OpenAI-compatible API; use crit serve --help"),
    ("pull", "crit.pull.cli", "Install a crit-hub engine for crit serve --engine; use crit pull --help"),
    ("smoke", "crit_smoke.cli", "Check a served engine's quality components in seconds; use crit smoke --help"),
    ("plugins", "crit.plugins", "List, check and test an engine's plugins; use crit plugins --help"),
    ("bench", "crit.pull.bench", "Measure an installed engine with crit Eval beside its card; use crit bench --help"),
    ("hub", "crit.pull.report", "Add a crit bench result to its crit-hub card (crit hub report); use crit hub --help"),
    ("kernels", "crit_kernels", "Inspect kernel applicability and retain verified dependency bytes"),
    ("models", "crit_models", "Transform model weights (PTQ) and record how; use crit models --help"),
    ("mcp", "crit_mcp.server", "Serve these commands to an agent as MCP tools over stdio; use crit mcp --help"),
)


def checkout(folder):
    """Whether `folder` is a crit checkout, a crit repository: it holds Core's native headers."""
    return (Path(folder) / "core/include/crit/core.h").is_file()


class Installed(NamedTuple):
    """crit as its distribution installed it: the crit package's folder (in the environment's site folder), its
    version and its guides, crit's repository at the release's tag."""
    package: Path
    version: str
    guides: str


def installed():
    """The installed crit when this crit runs from its distribution rather than from a checkout (installed with -e .
    or not), else None."""
    if checkout(ROOT):
        return None
    from . import __version__
    return Installed(Path(__file__).resolve().parent, __version__, f"{REPOSITORY}/tree/v{__version__}")


def eval_install():
    """How to install the crit Eval this crit works with: beside a checkout, crit-eval's checkout in editable mode; for
    an installed crit, the release its distribution pins (crit-eval==X.Y.Z)."""
    if checkout(ROOT):
        return "python -m pip install -e ../crit-eval beside this checkout"
    from importlib import metadata
    try:
        pins = [r.replace(" ", "") for r in metadata.requires("crit") or () if r.replace(" ", "").startswith("crit-eval==")]
    except metadata.PackageNotFoundError:
        pins = []
    return "python -m pip install " + (pins[0].split(";")[0] if pins else "crit-eval")


def default_project():
    """The crit repository a command uses unless it names one: this checkout when crit runs from one, else the
    current folder."""
    return ROOT if checkout(ROOT) else Path.cwd()


@contextmanager
def named_project(project):
    """`project` named to crit-eval (PROJECT_VARIABLE) for the block, so Eval reads the settings.toml the command
    reads, also where crit is installed without a checkout."""
    before = os.environ.get(PROJECT_VARIABLE)
    os.environ[PROJECT_VARIABLE] = str(Path(project).resolve())
    try:
        yield
    finally:
        if before is None:
            os.environ.pop(PROJECT_VARIABLE, None)
        else:
            os.environ[PROJECT_VARIABLE] = before


def family(module):
    """A command family's module; without crit Eval, crit eval says how to install it."""
    try:
        return importlib.import_module(module)
    except ImportError:
        if module.startswith("crit_eval"):
            raise SystemExit("crit eval needs crit Eval (https://github.com/crit-labs/crit-eval): install it with "
                             + eval_install()) from None
        raise


def main(argv=None):
    # Route subcommands with their own parsers before the maintenance parser.
    arguments = list(sys.argv[1:] if argv is None else argv)
    routes = {name: module for name, module, _ in FAMILIES}
    if arguments and arguments[0] == "eval":   # crit's project, unless the environment names one
        with named_project(os.environ.get(PROJECT_VARIABLE) or default_project()):
            return family(routes["eval"]).main(arguments[1:])
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
    if args.command == "inference":   # crit's plugin library, which an installed crit carries as a checkout does
        from . import inference
        return inference.main(args.arguments)
    if not checkout(default_project()):
        parser.error("Run repository maintenance commands from the crit repository")
    print("core/include/crit/core.h\ncore/include/crit/cuda.h\ncore/contracts/README.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
