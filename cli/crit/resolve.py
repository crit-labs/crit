"""What an engine will be before it loads: `crit plugins resolve`.

Each plugin of a package's selection answers from the checkpoint's headers, its options and the selection alone
(its module's resolve(): crit_plugins.assemble.resolve), with no weights read and no device touched, in seconds: the
library's model plugin states the KV its pools take a token of context (kv_bytes_per_token), priced by the kv_cache
plugin. Core's loader holds every load to the same answer, so a plan budgets the device from it before any engine is
built, and crit run checks a candidate's plan against it when the candidate is created."""
import json
from pathlib import Path

GIB = 1 << 30


def resolve(package, model, context=None, configuration=None):
    """resolve_package's answer for a package and a checkpoint: {"facts", "by"}, with the context it was asked at."""
    from crit_core.engine import resolve_package
    return {**resolve_package(package, model, configuration, context_tokens=context), "context": context}


def render(found, context=None):
    facts, by = found["facts"], found["by"]
    if not facts:
        return "no plugin of the selection resolves anything before a load (none has a resolve())"
    lines = [f"{key}: {json.dumps(value)}  ({by[key]})" for key, value in sorted(facts.items())]
    per_token = facts.get("kv_bytes_per_token")
    if per_token and context:
        lines.append(f"KV at {context:,} tokens: {per_token * context / GIB:.2f} GiB (sliding windows' rings apart)")
    return "\n".join(lines)


def main(args):
    if getattr(args, "spec", False):   # the whole EngineSpec, every value with its source (crit_core.spec)
        from crit_core.spec import describe, engine_spec
        spec = engine_spec(args.package, args.model, context_tokens=args.context)
        print(json.dumps(spec, indent=2) if args.json else "\n".join(describe(spec)))
        return 0
    found = resolve(args.package, args.model, args.context)
    if args.json:
        print(json.dumps(found, indent=2))
    else:
        print(render(found, args.context))
    return 0


def add_parser(subs):
    sub = subs.add_parser("resolve", help="What an engine will state of itself before it loads (its KV a token of "
                                          "context, ...): each plugin's answer from the checkpoint's headers and "
                                          "its options, no weights read, no device touched")
    sub.add_argument("package", type=Path, help="The engine's package (its plugins/ and configuration.json)")
    sub.add_argument("--model", type=Path, required=True, help="The checkpoint folder")
    sub.add_argument("--context", type=int, help="The context tokens the engine serves (default: the engine's own)")
    sub.add_argument("--json", action="store_true", help="Print the answer as JSON")
    sub.add_argument("--spec", action="store_true", help="Print the engine's whole spec: its context, headrooms, "
                                                         "quality mode, exactness, step bounds, every plugin's options "
                                                         "and the resolved facts, each with its source "
                                                         "(crit_core.spec)")
    return sub
