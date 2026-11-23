"""compose(): a model plugin's object from a layout and the techniques it names.

A model plugin extends this core (plugin.json: "sources": {"core": the folder of model-template}, "extends": "core")
and keeps its layout beside its plugin.json; its plugin.py is

    from pathlib import Path
    from .core.template.compose import model_plugin
    resolve, memory, create = model_plugin(Path(__file__).resolve().parent)        # layout.json in its folder

A layer part no model technique computes is the plugin's own technique, a class written like a library technique's
(models/techniques) in the plugin's code: model_plugin(folder, blocks={"my-mixer": MyMixer}), or
residuals={"my-residual": MyResidual} for a way blocks join the residual stream."""
import json
from pathlib import Path

from .blocks import library
from .model import ComposedModel
from .residual import DEFAULT

SCHEMA = "crit-model-layout-v1"
LAYOUT = "layout.json"   # a model plugin's layout, beside its plugin.json
KEYS = {"schema", "norm_offset", "mixers", "ffn", "draft", "pre_norm",
        "post_norm", "residual", "before", "defaults", "embed_scale", "logit_softcap", "routed_if", "requires",
        "names", "expert_caps", "head_rows", "gguf"}

# The options every composed model reads, with their defaults: this core's plugin.json declares the same
# (tests/test_plugin.py holds them equal), and a model plugin takes them by extending it.
DEFAULTS = {
    "concurrency": 4, "pending": 2, "context": 4096, "kv_margin": 8192, "kv": "auto",
    "prefill_chunk": 512, "span_seconds": 0.75, "passes": "auto", "pass_mib": 1024,
    "activations_mib": 768, "release_below_mib": 640,   # release_below_mib: no longer read (Core's Steps.threshold)
    "prompt_cache": 1, "rounds": True, "prompt_buckets": "", "mtp": False, "width": 3, "widths": "",
    "width_policy": "fixed", "mtp_encode": "auto", "mtp_kv": "auto", "draft_head": "shared", "draft_head_slices": 8,
    "kernel_tuning": "",
}


def check(layout, where="layout", blocks=None, residuals=None):
    """The layout's problems ([] when none): its schema, keys and the blocks and residual it names (the library's
    techniques, and the plugin's own `blocks` and `residuals`)."""
    found, joins, drafts = library()
    known = set(found) | set(blocks or ()) | {"auto"}
    residuals = set(joins) | set(residuals or ())
    problems = []
    if layout.get("schema") != SCHEMA:
        problems.append(f"{where}: schema is {SCHEMA}")
    unknown = set(layout) - KEYS
    if unknown:
        problems.append(f"{where}: unknown keys {', '.join(sorted(unknown))}")
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
    if draft is not None and (not isinstance(draft, dict) or draft.get("block") not in {"mtp", *drafts}
                              or not draft.get("prefix")):
        problems.append(f"{where}: draft is {{block, prefix}}, block mtp or a drafter technique's "
                        f"({', '.join(sorted(drafts)) or 'none mounted'})")
    if not isinstance(layout.get("norm_offset", 0.0), (int, float)):
        problems.append(f"{where}: norm_offset is a number")
    residual = layout.get("residual", {"block": DEFAULT})
    if not isinstance(residual, dict) or residual.get("block") not in residuals:
        problems.append(f"{where}: residual names how blocks join the residual ({', '.join(sorted(residuals))})")
    before = layout.get("before", [])
    if not isinstance(before, list) or not all(isinstance(spec, dict) and spec.get("block") in known - {"auto"}
                                               and isinstance(spec.get("layers"), (str, list)) for spec in before):
        problems.append(f"{where}: before lists blocks that add rows into the residual before a layer, each {{block, "
                        "layers: the layers' numbers from 1 or the config key that lists them}}")
    names = layout.get("names", [])
    if not isinstance(names, list) or not all(isinstance(rule, list) and len(rule) in (2, 3)
                                              and all(isinstance(part, str) for part in rule[:2])
                                              and (len(rule) == 2 or isinstance(rule[2], int)) for rule in names):
        problems.append(f"{where}: names lists [pattern, replacement] rules that rename the checkpoint's tensors to the "
                        "names the blocks read ([pattern, replacement, shift]: the first group's number shifted)")
    else:
        import re
        for rule in names:
            pattern = rule[0]
            try:
                re.compile(pattern)
            except re.error as error:
                problems.append(f"{where}: names pattern {pattern!r}: {error}")
    gguf = layout.get("gguf", {})
    if not isinstance(gguf, dict):
        problems.append(f"{where}: gguf maps each GGUF general.architecture the plugin reads to its entry (its parts, "
                        "outside, draft, experts, constants and transforms: code/gguf.py)")
    else:
        from .gguf import rules_of
        for arch, spec in gguf.items():
            try:
                rules = rules_of(spec)
            except ValueError as error:
                problems.append(f"{where}: gguf.{arch}: {error}")
                continue
            if rules.draft is not None and not isinstance(draft, dict):
                problems.append(f"{where}: gguf.{arch} names a draft block's parts; the layout has no draft")
    if not isinstance(layout.get("defaults", {}), dict):
        problems.append(f"{where}: defaults gives config values the family's reference assumes where config.json "
                        "is silent")
    return problems


def read(path, blocks=None, residuals=None):
    layout = json.loads(Path(path).read_text(encoding="utf-8"))
    problems = check(layout, Path(path).name, blocks, residuals)
    if problems:
        raise ValueError("model-template: " + "; ".join(problems))
    return layout


def layout_of(layout, folder=None, blocks=None, residuals=None):
    """The layout a model plugin composes: `layout`, a dict or a file in the plugin's `folder`."""
    if isinstance(layout, dict):
        problems = check(layout, "layout", blocks, residuals)
        if problems:
            raise ValueError("model-template: " + "; ".join(problems))
        return layout
    if not layout or folder is None:
        raise ValueError("model-template is the core a model plugin extends: a model plugin composes the layout in "
                         "its own folder (models/core/README.md)")
    return read(Path(folder) / layout, blocks, residuals)


def compose(context, options, parts, layout=LAYOUT, blocks=None, folder=None, residuals=None):
    """The ComposedModel of the checkpoint: options over DEFAULTS, the layout (a dict, or a file in the plugin's
    `folder`), the library's techniques and the plugin's own `blocks` and `residuals`."""
    options = {**DEFAULTS, **{k: v for k, v in dict(options).items()}}
    chosen = layout_of(layout, folder, blocks, residuals)
    return ComposedModel(context, options, parts, chosen, blocks, folder, residuals)


def resolve(context, options, dry, layout=LAYOUT, blocks=None, folder=None, residuals=None):
    """The model plugin's dry answer (Core's resolve(): crit_plugins.assemble.resolve), compose()'s twin: the
    checkpoint's Shape over its headers alone (shape.py), its KV pools priced by the kv_cache plugin's position_bytes
    (`dry`: the selection before any plugin exists). {"kv_bytes_per_token": ...}, or {} where the kv_cache plugin
    prices no position: with the layout and techniques compose() takes."""
    from crit_core.model import parse_model_config
    from .checkpoint import open_checkpoint
    from .shape import Shape
    options = {**DEFAULTS, **{k: v for k, v in dict(options).items()}}
    chosen = layout_of(layout, folder, blocks, residuals)
    handed = getattr(context, "checkpoint", None)
    checkpoint = handed if handed is not None else open_checkpoint(context.model, context.config, chosen)[0]
    shape = Shape(parse_model_config(context.config), options, chosen, blocks, checkpoint, residuals=residuals,
                  context_tokens=context.context_tokens, drafter=dry.plugin("drafter") is not None,
                  steps=getattr(context, "steps", None))
    price = getattr(dry.module("kv_cache"), "position_bytes", None)
    return shape.kv_facts(price) if callable(price) else {}


def tuning(options, folder):
    """The engine's kernel tuning (a folder of crit kernels sweep tables kept beside the plugin, option
    kernel_tuning), named before any plugin builds a kernel: memory() runs before every factory."""
    if options.get("kernel_tuning"):
        from crit_plugins.kernels import use_tuning
        path = Path(folder) / options["kernel_tuning"]
        if not path.is_dir():
            raise ValueError(f"model-template: kernel_tuning {options['kernel_tuning']} is no folder beside the plugin")
        use_tuning(path)


_PLUGINS = {}   # a model plugin's folder -> (layout, blocks, residuals), as its plugin.py gave them to model_plugin()


def plugin_layout(folder):
    """The layout of the model plugin in `folder`, checked against the library's techniques and the plugin's own:
    what its tests and crit models check read without a checkpoint, once its plugin.py ran."""
    key = str(Path(folder).resolve())
    if key not in _PLUGINS:
        raise ValueError(f"model-template: the plugin.py in {folder} calls no model_plugin()")
    layout, blocks, residuals = _PLUGINS[key]
    return layout_of(layout, folder, blocks, residuals)


def model_plugin(folder, layout=LAYOUT, blocks=None, residuals=None):
    """(resolve, memory, create): the three functions of the model plugin in `folder` over its layout and its own
    techniques, which its plugin.py gives Core's loader by these names. resolve: the engine's KV before any load,
    priced by the kv_cache plugin (resolve above; Core's loader holds the load to it). memory: the engine's kernel
    tuning named before any plugin builds a kernel. create: compose()."""
    def resolved(context, options, dry):
        return resolve(context, options, dry, layout, blocks, folder, residuals)

    def memory(context, options):
        tuning(options, folder)
        return {"device": 0, "host": 0}

    def create(context, options, parts):
        tuning(options, folder)
        return compose(context, options, parts, layout, blocks, folder, residuals)

    _PLUGINS[str(Path(folder).resolve())] = (layout, blocks, residuals)
    return resolved, memory, create
