"""crit hub report OWNER/NAME RESULT.json --hub CHECKOUT --by ACCOUNT: add a crit Eval result of an installed engine
(crit bench writes one) to the engine's card in a crit-hub checkout, for a pull request to crit-hub.

It checks that the result measured this engine as installed, at the source commit the checkout's card pins (a
result describes the commit it was measured on), then runs crit-hub's own tools/add_result.py in the checkout: the
result file goes into the engine's evidence/ as the next crit-eval-N.json, and the card gets it as a reproduction
of the GPU and model it measured or as a new result, its README and charts rendered again and the folder checked.
"""
import argparse
from pathlib import Path
import subprocess
import sys

from crit.run.records import read_json
from crit.run.store import ROOT

from .cli import log
from .store import InstallStore, engine_id


def report(engine, result_path, hub, by, store, gpu=None, model=None):
    """Add the result to the checkout's card; returns add_result.py's exit code."""
    engine = engine_id(engine)
    hub = Path(hub).expanduser().resolve()
    folder = hub / "engines" / engine
    tool = hub / "tools" / "add_result.py"
    if not (folder / "card.yaml").is_file() or not tool.is_file():
        raise ValueError(f"{hub} is not a crit-hub checkout with engines/{engine}/ and tools/add_result.py")
    from crit.run.hub import _yaml
    card = _yaml().safe_load((folder / "card.yaml").read_text(encoding="utf-8"))
    install = store.get(engine)
    result = read_json(Path(result_path), max_bytes=256 << 20)
    revision = install["source"]["revision"]
    if result.get("server") != f"crit serve --engine {engine}" or not str(result.get("label", "")).endswith(revision[:12]):
        raise ValueError(f"{result_path} is not a crit bench result of {engine} as installed (its server is "
                         f"{result.get('server')!r}, its label {result.get('label')!r})")
    if (card.get("source") or {}).get("revision") != revision:
        raise ValueError(f"The checkout's card pins {engine}'s source at {(card.get('source') or {}).get('revision', '')[:12]}, "
                         f"the installed engine was built from {revision[:12]}: a result describes the commit it was "
                         "measured on (crit pull --force installs the card's commit)")
    command = [sys.executable, "-B", str(tool), str(folder), str(Path(result_path).resolve()), "--by", by]
    command += ["--gpu", gpu] if gpu else []
    command += ["--model", model] if model else []
    log(f"crit hub report: {' '.join(command[1:])}")
    return subprocess.run(command, cwd=hub).returncode


def main(argv=None):
    parser = argparse.ArgumentParser(prog="crit hub", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    add = commands.add_parser("report", help="Add a crit bench result to the engine's card in a crit-hub checkout")
    add.add_argument("engine", metavar="OWNER/NAME", help="The installed engine the result measured")
    add.add_argument("result", type=Path, help="The crit Eval result file crit bench wrote")
    add.add_argument("--hub", type=Path, required=True, help="Your crit-hub checkout (a fork, for the pull request)")
    add.add_argument("--by", required=True, help="Your GitHub account: who measured it")
    add.add_argument("--gpu", help="The GPU id, when the result's device does not name it (e.g. rtx5090)")
    add.add_argument("--model", help="The card's model id, when the result names no checkpoint")
    add.add_argument("--store", type=Path, help="Engine store (default: PROJECT/engines)")
    add.add_argument("--project", type=Path, default=ROOT if (ROOT / "core/include/crit/core.h").is_file() else Path.cwd(),
                     help="crit repository (default: this checkout)")
    args = parser.parse_args(argv)
    try:
        code = report(args.engine, args.result, args.hub, args.by, InstallStore(args.store or args.project / "engines"),
                      args.gpu, args.model)
    except (ValueError, OSError, KeyError) as exc:
        print("error: " + str(exc), file=sys.stderr)
        return 1
    if code == 0:
        log(f"crit hub report: commit engines/{engine_id(args.engine)}/ in {args.hub} and open a pull request to crit-hub")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
