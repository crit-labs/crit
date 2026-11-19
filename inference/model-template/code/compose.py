"""compose(): a model plugin's object from a layout and the blocks it names.

    from .template.compose import compose               # sources: {"template": "../model-template/code", ...}
    def create(context, options, parts):
        return compose(context, options, parts)         # the layout for the checkpoint's model_type

An engine whose architecture the library's layouts do not cover writes a layout (layouts/*.json is the format) and,
for a layer part no block computes, a block class, and passes both: compose(context, options, parts,
layout="layout.json", blocks={"my-mixer": MyMixer}, folder=Path(__file__).parent)."""
import json
from pathlib import Path

from .blocks import BLOCKS
from .model import ComposedModel
from .residual import RESIDUALS

LAYOUTS = Path(__file__).resolve().parent / "layouts"
SCHEMA = "crit-model-layout-v1"
KEYS = {"schema", "name", "description", "model_types", "norm_offset", "mixers", "ffn", "draft", "pre_norm",
        "post_norm", "residual", "before", "defaults", "embed_scale", "logit_softcap", "routed_if", "requires"}

# The options every composed model reads, with their defaults: the template plugin's plugin.json declares the same
# (tests/test_plugin.py holds them equal), and an engine's own model plugin over compose() declares those it sets.
DEFAULTS = {
    "layout": "", "concurrency": 4, "pending": 2, "context": 4096, "kv_margin": 8192, "kv": "auto",
    "prefill_chunk": 512, "span_seconds": 0.75, "passes": "auto", "pass_mib": 1024, "headroom_mib": 0,
    "activations_mib": 768, "release_below_mib": 640,
    "prompt_cache": 1, "rounds": True, "prompt_buckets": "", "mtp": False, "width": 3, "widths": "",
    "width_policy": "fixed", "mtp_encode": "bf16", "mtp_kv": "bf16", "draft_head": "shared", "draft_head_slices": 8,
    "kernel_tuning": "",
}


def check(layout, where="layout", blocks=None):
    """The layout's problems ([] when none): its schema, keys, model types and the blocks it names."""
    known = set(BLOCKS) | set(blocks or ()) | {"auto"}
    problems = []
    if layout.get("schema") != SCHEMA:
        problems.append(f"{where}: schema is {SCHEMA}")
    unknown = set(layout) - KEYS
    if unknown:
        problems.append(f"{where}: unknown keys {', '.join(sorted(unknown))}")
    types = layout.get("model_types")
    if not isinstance(types, list) or not types or not all(isinstance(t, str) and t for t in types):
        problems.append(f"{where}: model_types lists the config model_types it composes")
    mixers = layout.get("mixers")
    if not isinstance(mixers, dict) or not mixers:
        problems.append(f"{where}: mixers maps each layer kind to its block")
    else:
        for kind, spec in mixers.items():
            if not isinstance(spec, dict) or spec.get("block") not in known:
                problems.append(f"{where}: mixers.{kind} names a block ({', '.join(sorted(known - {'auto'}))})")
    ffn = layout.get("ffn", {"block": "auto"})
    if not isinstance(ffn, dict) or ffn.get("block") not in known or any(
            ffn.get(key, "moe") not in known - {"auto"} for key in ("routed", "dense")):
        problems.append(f"{where}: ffn names a block or auto (with auto, `routed` and `dense` may name the blocks of "
                        "routed and dense layers)")
    draft = layout.get("draft")
    if draft is not None and (not isinstance(draft, dict) or draft.get("block") != "mtp" or not draft.get("prefix")):
        problems.append(f"{where}: draft is {{block: mtp, prefix}}")
    if not isinstance(layout.get("norm_offset", 0.0), (int, float)):
        problems.append(f"{where}: norm_offset is a number")
    residual = layout.get("residual", {"block": "pre-norm"})
    if not isinstance(residual, dict) or residual.get("block") not in RESIDUALS:
        problems.append(f"{where}: residual names how blocks join the residual ({', '.join(sorted(RESIDUALS))})")
    before = layout.get("before", [])
    if not isinstance(before, list) or not all(isinstance(spec, dict) and spec.get("block") in known - {"auto"}
                                               and isinstance(spec.get("layers"), (str, list)) for spec in before):
        problems.append(f"{where}: before lists blocks that add rows into the residual before a layer, each {{block, "
                        "layers: the layers' numbers from 1 or the config key that lists them}}")
    if not isinstance(layout.get("defaults", {}), dict):
        problems.append(f"{where}: defaults gives config values the family's reference assumes where config.json "
                        "is silent")
    return problems


def read(path, blocks=None):
    layout = json.loads(Path(path).read_text(encoding="utf-8"))
    problems = check(layout, Path(path).name, blocks)
    if problems:
        raise ValueError("model-template: " + "; ".join(problems))
    return layout


def builtin():
    """{model_type: layout} of the library's layouts."""
    found = {}
    for path in sorted(LAYOUTS.glob("*.json")):
        layout = read(path)
        for model_type in layout["model_types"]:
            if model_type in found:
                raise ValueError(f"model-template: {model_type} is in layouts {found[model_type]['name']} and "
                                 f"{layout['name']}")
            found[model_type] = layout
    return found


def layout_for(config, layout=None, folder=None, blocks=None):
    """The layout a model composes: `layout` (a dict, or a file beside `folder`), else the library's for the config's
    model_type (its text model_type first)."""
    if isinstance(layout, dict):
        problems = check(layout, "layout", blocks)
        if problems:
            raise ValueError("model-template: " + "; ".join(problems))
        return layout
    if layout:
        path = Path(layout)
        if not path.is_absolute():
            if folder is None:
                raise ValueError("model-template: a layout file is named relative to its plugin's folder")
            path = Path(folder) / path
        return read(path, blocks)
    known = builtin()
    text = config.get("text_config") or {}
    for model_type in (text.get("model_type"), config.get("model_type")):
        if model_type in known:
            return known[model_type]
    raise ValueError(f"model-template: no layout for model_type {config.get('model_type')!r} (the library's: "
                     f"{', '.join(sorted(known))}); give the plugin's option layout, a layout file beside it")


def compose(context, options, parts, layout=None, blocks=None, folder=None):
    """The ComposedModel of the checkpoint: options over DEFAULTS, the layout (option `layout`, else `layout`, else the
    library's for the checkpoint), the library's blocks and `blocks`."""
    options = {**DEFAULTS, **{k: v for k, v in dict(options).items()}}
    chosen = layout_for(context.config, options["layout"] or layout, folder, blocks)
    return ComposedModel(context, options, parts, chosen, blocks, folder)


def tuning(options, folder):
    """The engine's kernel tuning (a folder of crit kernels sweep tables kept beside the plugin, option
    kernel_tuning), named before any plugin builds a kernel: memory() runs before every factory."""
    if options.get("kernel_tuning"):
        from crit_plugins.kernels import use_tuning
        path = Path(folder) / options["kernel_tuning"]
        if not path.is_dir():
            raise ValueError(f"model-template: kernel_tuning {options['kernel_tuning']} is no folder beside the plugin")
        use_tuning(path)
