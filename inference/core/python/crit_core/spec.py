"""One description of an engine, resolved before any weight is read: the EngineSpec, every value with its source.

    from crit_core.spec import engine_spec
    spec = engine_spec(package, model)                  # {"schema", "values": {name: {"value", "source", ...}}}
    spec["values"]["kv_bytes_per_token"]                # {"value": 12288, "source": "resolved", "by": "my-model"}

What it holds: the context the engine is built for, the device and host memory it leaves free (headrooms), the quality
mode, the exactness and prompt activations it asks, the step bound and cancel grace it serves inside, every selected
plugin's options, and the facts the plugins state of the engine from the checkpoint's headers (their resolve():
kv_bytes_per_token and whatever else a plugin resolves). Each value carries its source, so a default is never mistaken
for a user's choice when it crosses from one tool to another:

    user        the user set it (a run request, settings.toml, a command's option); the configuration's own
                `provenance` says which of its values those are
    package     the package's configuration names it (an engine built outside a run, or a value whose provenance the
                configuration does not record)
    mode        the quality mode's default for it (a precision or exactness left to the mode)
    default     crit's default, the configuration naming none
    resolved    derived from the checkpoint and the selection by the plugin `by` names (its resolve())
    core        Core's own bound, the same for every engine (the cancel grace)
    measured    measured on the machine (a decision a package records)
    protocol    crit Eval's protocol (its required context, its KV precision)

Core's loader records the spec it loads under (telemetry load.spec) and refuses an engine whose facts differ from the
spec's resolved values (crit_core.engine: the resolved facts it holds the load to); `crit plugins resolve --spec`
prints it. A tool that needs one of these values reads it here instead of deriving it again."""
import json

SCHEMA = "crit-engine-spec-v1"
SOURCES = ("user", "package", "mode", "default", "resolved", "core", "measured", "protocol")
PROVENANCE = "provenance"   # a configuration's {value name: source} for the values crit run wrote into it


def value(item, source, **extra):
    """A spec value: {"value": item, "source": source, ...extra} (`by`, `note`)."""
    if source not in SOURCES:
        raise ValueError(f"a spec value's source is one of {', '.join(SOURCES)}, not {source!r}")
    return {"value": item, "source": source, **extra}


def provenance_of(configuration):
    """The configuration's {value name: source} (crit run writes it for the values a run's request or the machine's
    settings gave). ValueError for a source not in SOURCES."""
    found = dict((configuration or {}).get(PROVENANCE) or {})
    for key, source in found.items():
        if source not in SOURCES:
            raise ValueError(f'configuration "{PROVENANCE}": {key} has the source {source!r}, not one of '
                             f"{', '.join(SOURCES)}")
    return found


def _named(configuration, key, provenance, default=None, default_source="default"):
    """The configuration's `key` with its source: its provenance's, else package; else the default."""
    if configuration.get(key) is not None:
        return value(configuration[key], provenance.get(key, "package"))
    return value(default, default_source)


def _options(registry, chosen, mode, provenance):
    """{plugin: {option: spec value}}: each selected plugin's options, set by the configuration (its provenance, else
    package) or left at the manifest's default (default, or mode where the default is auto and a quality mode is
    stated: the mode resolves it)."""
    out = {}
    named = chosen.get("options") or {}
    for plugin in sorted({p for p in list(chosen["use"].values()) + list(chosen.get("base", {}).values()) if p}):
        manifest = registry[plugin]
        own = named.get(plugin) or {}
        found = {}
        for name, option in sorted(manifest.get("options", {}).items()):
            if name in own:
                found[name] = value(own[name], provenance.get(f"options.{plugin}.{name}", "package"))
            elif option["default"] == "auto" and mode is not None:
                found[name] = value("auto", "mode", note=f"resolved by quality mode {mode}")
            else:
                found[name] = value(option["default"], "default")
        out[plugin] = found
    return out


def engine_spec(package, model, configuration=None, *, context_tokens=None, context_source=None, device=None,
                probe=None, speculative=True):
    """The EngineSpec of the package's engine for `model` (the module's description), from the checkpoint's headers
    and the configuration alone. context_tokens: the context the engine is built for (the loader's caller: a run's KV
    capacity, crit serve's --context), from `context_source` (else the configuration's provenance of context_tokens,
    else user); without it the model plugin's context option."""
    from .engine import read_configuration, resolve_package
    configuration = read_configuration(package) if configuration is None else configuration
    resolved = resolve_package(package, model, configuration, context_tokens=context_tokens, speculative=speculative,
                               device=device, probe=probe)
    return spec_of(package, model, configuration, resolved, context_tokens=context_tokens,
                   context_source=context_source, speculative=speculative)


def spec_of(package, model, configuration, resolved, *, context_tokens=None, context_source=None, speculative=True):
    """engine_spec() over facts already resolved ({"facts", "by"}: crit_core.engine.resolve_package's, or the
    loader's own resolution), so a load records the spec it was held to without resolving again."""
    from pathlib import Path
    from crit_plugins.memory import HEADROOM, HOST_HEADROOM
    from crit_plugins.steps import GRACE, SECONDS
    from .engine import exactness_levels, quality_mode_of, registry_of, selection
    provenance = provenance_of(configuration)
    chosen, registry = selection(configuration, speculative), registry_of(package)
    mode = quality_mode_of(configuration)
    options = _options(registry, chosen, mode, provenance)
    model_plugin = chosen["use"]["model"]
    values = {
        "quality_mode": _named(configuration, "quality_mode", provenance),
        "vram_headroom_bytes": _named(configuration, "vram_headroom_bytes", provenance, HEADROOM),
        "ram_headroom_bytes": _named(configuration, "ram_headroom_bytes", provenance, HOST_HEADROOM),
        "prompt_activations": _named(configuration, "prompt_activations", provenance),
        "cancel_grace_seconds": value(GRACE, "core"),
        "prompt_step_seconds": value(SECONDS, "core", note="the default bound of a prompt step; a model plugin's "
                                                           "span_seconds may set a tighter one"),
    }
    levels = exactness_levels(configuration)
    if configuration.get("exactness") is not None:
        values["exactness"] = value(levels, provenance.get("exactness", "package"))
    else:
        values["exactness"] = value(levels, "mode" if levels is not None else "default")
    if context_tokens:
        values["context_tokens"] = value(int(context_tokens),
                                         context_source or provenance.get("context_tokens", "user"))
    else:
        values["context_tokens"] = options.get(model_plugin, {}).get("context") or value(None, "default")
    for key, item in sorted(resolved["facts"].items()):
        values[key] = value(item, "resolved", by=resolved["by"].get(key))
    return {"schema": SCHEMA, "package": str(Path(package)), "model": str(Path(model)),
            "selection": {slot: plugin for slot, plugin in sorted(chosen["use"].items()) if plugin},
            "values": values, "options": options}


def plain(spec):
    """{name: value} of a spec's values, their sources dropped."""
    return {name: item["value"] for name, item in spec["values"].items()}


def differences(spec, facts):
    """[problem] where an engine's stated facts differ from the spec's resolved values (a price, which sizes the
    engine before it exists, aside: crit_plugins.assemble.PRICES), and where the engine serves less context than the
    spec's context_tokens (its info's `context`)."""
    from crit_plugins.assemble import PRICES
    found = []
    context = (spec["values"].get("context_tokens") or {}).get("value")
    served = facts.get("context")
    if isinstance(context, int) and isinstance(served, int) and served < context:
        found.append(f"context_tokens: the engine serves {served:,} tokens of context, its spec asks {context:,} "
                     f"({spec['values']['context_tokens']['source']})")
    for name, item in sorted(spec["values"].items()):
        if name in PRICES:
            continue
        if item["source"] == "resolved" and facts.get(name) != item["value"]:
            found.append(f"{name}: the engine states {facts.get(name)!r}, its spec resolved {item['value']!r} "
                         f"(by {item.get('by')})")
    return found


def describe(spec):
    """Lines of text: each value with its source."""
    lines = []
    for name, item in sorted(spec["values"].items()):
        by = f" by {item['by']}" if item.get("by") else ""
        lines.append(f"{name:<24} {json.dumps(item['value'], sort_keys=True):<28} {item['source']}{by}")
    return lines
