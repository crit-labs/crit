"""Record an engine's measured decisions: `crit plugins decisions`.

A plugin that chooses between two ways by timing them at load (crit_plugins.decisions) can choose otherwise in the
next process. This command loads the package's engine once, lets every such decision measure, and writes what was
decided to a record inside the package. The configuration's "decisions" names that file, and every later load
replays it instead of timing again, so the package Eval measures decides as the package served does.

Run it on the machine and model the engine serves, with nothing else on the device, before the candidate is sealed.
A decision's key (the device, the shapes, the options that bound it) is recorded with it: on another device or
model the record does not answer and the decision is measured there, which the engine's telemetry says
("decisions": "measured")."""
import json
import sys
from pathlib import Path

DEFAULT = "decisions.json"


def record(package, model, context=None, out=None, log=None):
    """Load the engine with no record, so every decision measures, and write the record. Returns {"path", "rows":
    the decisions, "named": whether the configuration names the file}."""
    from crit_core.engine import load, read_configuration
    from crit_plugins import decisions
    say = log or (lambda text: None)
    package = Path(package)
    configuration = read_configuration(package)
    name = configuration.get("decisions")
    path = Path(out) if out else package / (name if isinstance(name, str) else DEFAULT)
    fresh = {key: value for key, value in configuration.items() if key != "decisions"}
    say("loading the engine")
    engine = load(package, model, fresh, context_tokens=context)
    try:
        rows = decisions.report()["decisions"]
    finally:
        engine.close()
    decisions.write(path, rows)
    named = isinstance(name, str) and (package / name).resolve() == path.resolve()
    return {"path": path, "rows": rows, "named": named}


def render(found):
    rows = found["rows"]
    lines = [f"{len(rows)} decision{'s' if len(rows) != 1 else ''} recorded in {found['path']}"]
    for row in rows:
        lines.append(f"  {row['plugin']}.{row['name']} = {json.dumps(row['choice'])}   "
                     f"measured {json.dumps(row['measurement'])}")
    if not rows:
        lines.append("  no plugin of this engine decides by a measurement: it needs no record")
    elif not found["named"]:
        relative = found["path"].name
        lines.append(f'add "decisions": "{relative}" to the package\'s configuration.json so every load replays it')
    return "\n".join(lines)


def main(args):
    found = record(args.package, args.model, args.context, args.out,
                   log=lambda text: print(text, file=sys.stderr, flush=True))
    print(render(found))
    return 0


def add_parser(subs):
    sub = subs.add_parser("decisions", help="Record the choices an engine's plugins take from a measurement at load "
                                            "(who computes a missed expert, ...), so every process of the package "
                                            "decides alike")
    sub.add_argument("package", type=Path, help="The engine's package (its plugins/ and configuration.json)")
    sub.add_argument("--model", type=Path, required=True, help="The checkpoint folder")
    sub.add_argument("--context", type=int, help="The context tokens the engine serves (default: the engine's own)")
    sub.add_argument("--out", type=Path, help=f"The record to write (default: the file the configuration's "
                                              f"\"decisions\" names, else PACKAGE/{DEFAULT})")
    return sub
