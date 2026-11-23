"""crit models: build a model plugin from crit's model techniques, and keep it (models/README.md).

    crit models list                          the model techniques and the finished model plugins
    crit models inspect MODEL                 what the checkpoint needs: the model plugin that serves it, the
                                              techniques that read its config keys and tensors, and what none reads
    crit models tiny MODEL --out DIR          a small random checkpoint of the same architecture, from its reference
    crit models diff --model TINY             the model plugin against the reference on that checkpoint (--plugin
                                              ENGINE/plugins/ID: the one being built)
    crit models cache add ENGINE/plugins/ID   keep a finished model plugin in the user's cache: crit plugins add ID
                                              then puts it into any engine

Nothing here names a model: what an architecture is comes from its checkpoint (config.json, tensor names) and from the
techniques' and model plugins' own files.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import sys

from .inference import ROOT

MODELS = ROOT / "models"
TECHNIQUES = MODELS / "techniques"
CACHE = MODELS / "cache"
TEMPLATE = MODELS / "core" / "model-template"
TECHNIQUE_SCHEMA = "crit-model-technique-v1"
LAYOUT = "layout.json"    # a model plugin's layout, beside its plugin.json
MANIFEST = "plugin.json"
# config.json keys every decoder has, which no technique needs to claim
COMMON_KEYS = {"architectures", "model_type", "vocab_size", "hidden_size", "num_hidden_layers", "num_attention_heads",
               "num_key_value_heads", "head_dim", "intermediate_size", "max_position_embeddings", "rms_norm_eps",
               "torch_dtype", "dtype", "tie_word_embeddings", "bos_token_id", "eos_token_id", "pad_token_id",
               "transformers_version", "quantization_config", "text_config", "rope_theta", "rope_scaling",
               "rope_parameters", "initializer_range", "use_cache", "attention_dropout", "hidden_act",
               "attention_bias", "layer_types", "sliding_window"}


def core_keys():
    """The config.json keys Core's typed reader reads (crit_core.model.config: its sections' fields and the spellings
    config_keys.json lists), which a model plugin gets through the parsed config rather than by name."""
    import dataclasses
    from crit_core.model import config as reader
    from crit_models import config as mechanisms
    keys = set()
    for section in (mechanisms.MoeConfig, mechanisms.MlaConfig, mechanisms.LinearAttentionConfig, reader.ModelConfig,
                    mechanisms.Architecture):
        keys |= {f.name for f in dataclasses.fields(section)}
    for module in (reader, mechanisms):
        spelled = json.loads((Path(module.__file__).parent / "config_keys.json").read_text(encoding="utf-8"))
        for value in spelled.values():
            for item in value if isinstance(value, list) else ():
                keys |= set(item.values()) if isinstance(item, dict) else {item}
    keys |= {"linear_" + k for k in ("num_key_heads", "num_value_heads", "key_head_dim", "value_head_dim",
                                     "conv_kernel_dim")}
    return {k for k in keys if isinstance(k, str)}


def user_cache():
    """The user's cache of model plugins (crit_core.cache_home("models")): what runs on this machine built."""
    from crit_core import cache_home
    return Path(cache_home("models"))


def techniques():
    """[entry] of models/techniques, each with its folder."""
    found = []
    for path in sorted(TECHNIQUES.glob("*/*/entry.json")):
        entry = json.loads(path.read_text(encoding="utf-8"))
        if entry.get("schema") != TECHNIQUE_SCHEMA:
            raise ValueError(f"{path}: schema is {TECHNIQUE_SCHEMA}")
        found.append({**entry, "folder": str(path.parent)})
    return found


def model_plugins(folders=None):
    """[{id, title, model_types, folder, layout, where}] of the finished model plugins: crit's (models/cache,
    where "library"), then the user's cache (where "cache") for the ids crit's does not hold. model_types: the
    config model_types its manifest needs."""
    found, seen = [], set()
    for where, folder in (("library", CACHE), ("cache", user_cache())) if folders is None else folders:
        for path in sorted(Path(folder).glob(f"*/{MANIFEST}")) if Path(folder).is_dir() else ():
            manifest = json.loads(path.read_text(encoding="utf-8"))
            ident = manifest.get("id")
            if manifest.get("fills") != ["model"] or ident in seen:
                continue
            seen.add(ident)
            types = [fact[len("model_type:"):] for need in manifest.get("needs", [])
                     for fact in ([need] if isinstance(need, str) else need) if fact.startswith("model_type:")]
            layout = path.parent / LAYOUT
            found.append({"id": ident, "title": manifest.get("title", ""), "model_types": types,
                          "folder": path.parent, "where": where,
                          "layout": json.loads(layout.read_text(encoding="utf-8")) if layout.is_file() else {}})
    return found


def plugin_for(types, folders=None):
    """The finished model plugin that serves one of the config model_types `types` (model_plugins()), or None."""
    return next((plugin for plugin in model_plugins(folders) if set(types) & set(plugin["model_types"])), None)


def _config(model):
    from crit_core.model import open_model
    from crit_models.config import architecture
    return architecture(open_model(model).config())


def _tensor_names(model):
    """{name: (shape, dtype)} of the checkpoint's language-model tensors, from their headers only."""
    from crit_core.model import open_model
    source = open_model(model)
    weights = source.weights
    tensors = getattr(weights, "tensors", None) or {}
    return {name: (tuple(getattr(t, "shape", ())), str(getattr(t, "dtype", ""))) for name, t in tensors.items()}


def pattern(name):
    """A tensor name with its layer and expert numbers as N: one pattern for every layer's copy of a tensor."""
    return re.sub(r"\.\d+$", ".N", re.sub(r"\.\d+\.", ".N.", re.sub(r"\.\d+\.", ".N.", name)))


def suffix_pattern(suffix):
    """A tensor suffix of a technique's reads.tensors as a regex over a name's pattern: whole dotted parts at the
    name's end or inside it, its N any layer's or expert's number."""
    body = re.sub(r"(?<![A-Za-z0-9])N(?![A-Za-z0-9])", lambda _: r"(?:N|\d+)", re.escape(suffix))
    return re.compile(r"(?:^|\.)" + body + r"(?:\.|$)")


def rename(name, rules):
    """A checkpoint's tensor name under a layout's `names` rules ([pattern, replacement(, shift)], in order), as
    model-template's checkpoint applies them."""
    for rule in rules:
        if len(rule) == 3:
            found = re.search(rule[0], name)
            if found:
                index = int(found.group(1)) + int(rule[2])
                name = re.sub(rule[0], rule[1].replace("\\1", str(index)), name)
        else:
            name = re.sub(rule[0], rule[1], name)
    return name


def _strings(value):
    """Every string in a JSON value (a layout's config key names among them)."""
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for item in value.values():
            yield from _strings(item)
    elif isinstance(value, list):
        for item in value:
            yield from _strings(item)


def fingerprint(config, names):
    """The architecture's fingerprint: its model types, architectures, config keys and tensor-name patterns."""
    section = dict(config.section)
    data = {"model_type": config.model_type, "text_model_type": config.text_model_type,
            "architectures": sorted(config.architectures), "keys": sorted(section),
            "tensors": sorted({pattern(n) for n in names})}
    return hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()[:16]


def inspect(model):
    """What a checkpoint needs: {fingerprint, model_type, plugins: [ids of the finished model plugins that serve
    its model_type], techniques: {id: {config: [its reads.config keys the config has], tensors: [its reads.tensors
    suffixes found]}}, unread: {config: [...], tensors: [...]}}."""
    config = _config(model)
    names = _tensor_names(model)
    section = dict(config.section)
    types = [t for t in (config.text_model_type, config.model_type) if t]
    matched = [plugin for plugin in model_plugins() if set(types) & set(plugin["model_types"])]
    layout_keys = set()
    if matched:   # the names the techniques read, through the plugin's rename rules; the config keys its layout names
        layout = matched[0]["layout"]
        names = {rename(n, layout.get("names", [])): v for n, v in names.items()}
        layout_keys = {v for v in _strings(layout)} | set(layout.get("defaults", {}))
    patterns = sorted({pattern(n) for n in names})
    listed = techniques()
    # What a checkpoint can show of a technique's reads: its config keys at the config's top level (ENTRY.KEY, a key
    # inside an entry, is not one) and its tensor suffixes. reads.layout names layout fields, which no checkpoint has.
    keys_of = {e["id"]: sorted(k for k in e["reads"]["config"] if "." not in k) for e in listed}
    suffixes_of = {e["id"]: {s: suffix_pattern(s) for s in e["reads"]["tensors"]} for e in listed}
    # A suffix tells its technique apart where it is more than one short word and no other technique reads it or a
    # part of it (a norm's weight, a projection every block has).
    distinct = {(ident, s) for ident, own in suffixes_of.items() for s in own if ("." in s or len(s) > 6) and not any(
        read.search(s) for other, theirs in suffixes_of.items() if other != ident for read in theirs.values())}

    claimed_keys, found = set(), {}
    for entry in listed:
        ident = entry["id"]
        keys = [k for k in keys_of[ident] if k in section]
        suffixes = sorted(s for s, read in suffixes_of[ident].items()
                          if (ident, s) in distinct and any(read.search(p) for p in patterns))
        if suffixes or set(keys) - COMMON_KEYS:
            found[ident] = {"config": keys, "tensors": suffixes}
            claimed_keys |= set(keys)
    unread_keys = sorted(set(section) - claimed_keys - COMMON_KEYS - core_keys() - layout_keys)
    # A tensor is read where some technique reads it under one of its suffixes; the names every decoder has aside.
    reads = [read for own in suffixes_of.values() for read in own.values()]
    common = ("embed_tokens", "lm_head", "norm.weight", "input_layernorm", "post_attention_layernorm", "q_proj",
              "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
    unread_tensors = sorted(p for p in patterns if not any(read.search(p) for read in reads)
                            and not any(c in p for c in common))
    return {"fingerprint": fingerprint(config, names), "model_type": types[0] if types else "",
            "plugins": [plugin["id"] for plugin in matched],
            "techniques": found, "unread": {"config": unread_keys, "tensors": unread_tensors},
            "tensors": len(names)}


# -- tiny ---------------------------------------------------------------------------------------------------------
# Size keys and the value a tiny checkpoint gives them; every other key (layer kinds, loops, windows, routing) stays.
TINY = {"hidden_size": 64, "intermediate_size": 128, "moe_intermediate_size": 32, "shared_expert_intermediate_size": 32,
        "num_attention_heads": 4, "num_key_value_heads": 2, "head_dim": 16, "vocab_size": 256,
        "num_experts": 8, "num_local_experts": 8, "n_routed_experts": 8, "num_experts_per_tok": 2,
        "kv_lora_rank": 32, "q_lora_rank": 32, "qk_rope_head_dim": 8, "qk_nope_head_dim": 8, "v_head_dim": 16,
        "linear_num_key_heads": 2, "linear_num_value_heads": 4, "linear_key_head_dim": 16, "linear_value_head_dim": 16,
        "max_position_embeddings": 4096, "n_group": 2, "topk_group": 1}


def tiny_config(config, layers=None):
    """config.json (a dict) shrunk to a tiny checkpoint's: the size keys of TINY where the config has them (in its
    text section too), a per-layer list cut to the layers kept, the rest as it is."""
    out = json.loads(json.dumps(config))
    for section in [out] + ([out["text_config"]] if isinstance(out.get("text_config"), dict) else []):
        total = section.get("num_hidden_layers")
        keep = layers or (min(total, 4) if isinstance(total, int) else None)
        for key, value in TINY.items():
            if isinstance(section.get(key), int) and not isinstance(section.get(key), bool):
                section[key] = min(section[key], value) if key not in ("hidden_size",) else value
        if keep and isinstance(total, int):
            section["num_hidden_layers"] = keep
            for key, value in list(section.items()):
                if isinstance(value, list) and len(value) == total:
                    section[key] = value[:keep]
    return out


def tiny(model, out, layers=None, trust_remote_code=False, seed=0):
    """A tiny random checkpoint of the model's architecture in `out`: its config shrunk (tiny_config), the
    reference's own modeling code (transformers, or the checkpoint's with trust_remote_code) given random weights,
    saved as safetensors with the checkpoint's tokenizer and generation files. Returns out."""
    import torch
    import transformers
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    source = Path(model)
    config = json.loads((source / "config.json").read_text(encoding="utf-8"))
    small = tiny_config(config, layers)
    (out / "config.json").write_text(json.dumps(small, indent=2), encoding="utf-8")
    torch.manual_seed(seed)
    built = transformers.AutoConfig.from_pretrained(str(out), trust_remote_code=trust_remote_code)
    reference = transformers.AutoModelForCausalLM.from_config(built, trust_remote_code=trust_remote_code)
    reference.save_pretrained(str(out), safe_serialization=True)
    for name in ("generation_config.json", "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json",
                 "chat_template.jinja"):
        if (source / name).is_file() and not (out / name).exists():
            shutil.copy2(source / name, out / name)
    return out


# -- diff ---------------------------------------------------------------------------------------------------------
# The engine diff loads: the model plugin, the plain plugin of each slot it asks, and the options of a small engine.
PLAIN = {"placement": "device-placement", "projections": "projections-dense",
         "attention": "softmax-attention", "kv_cache": "attention-state", "sampler": "token-sampling",
         "norms": "torch-norms", "embeddings": "memory-table-rows"}
ROUTED = {"router": "topk-router", "experts": "experts-reference", "expert_source": "expert-tensors"}
RECURRENT = {"recurrence": "sequential-recurrence"}
SMALL = {"context": 1024, "kv_margin": 256, "prefill_chunk": 64, "activations_mib": 64, "concurrency": 4}


MOUNTED = "crit_models_mounted"


def mounted(key):
    """The module of the technique the models core's manifest mounts as `key` (its code/KEY.py), imported the way
    the core's package lays its sources out: for tests and tools that read a technique without loading an engine."""
    import importlib
    import types
    if MOUNTED not in sys.modules:
        manifest = json.loads((TEMPLATE / "plugin.json").read_text(encoding="utf-8"))
        root = types.ModuleType(MOUNTED)
        root.__path__ = [str(TEMPLATE)]
        sys.modules[MOUNTED] = root
        for name, folder in manifest["sources"].items():
            sub = types.ModuleType(f"{MOUNTED}.{name}")
            sub.__path__ = [str((TEMPLATE / folder).resolve())]
            sub.__package__ = sub.__name__
            sys.modules[sub.__name__] = sub
            setattr(root, name, sub)
    return importlib.import_module(f"{MOUNTED}.{key}.{key}")


def _types(model):
    config = _config(model)
    return [t for t in (config.text_model_type, config.model_type) if t]


def options_for(model, use, layout=None):
    """The plugin options the checkpoint states for the selection: the router's rule (the moe technique's
    router_rule, over the defaults of `layout`, else of the layout of the model plugin that serves its model_type)
    where it routes experts."""
    config = _config(model)
    if config.moe is None or use.get("router") != "topk-router":
        return {}
    if layout is None:
        layout = (plugin_for(_types(model)) or {}).get("layout", {})
    return {"topk-router": mounted("moe").router_rule(config.moe, layout.get("defaults"))}


def selection_for(model, plugin=None):
    """The plain selection for a checkpoint: the model plugin `plugin` (an id; default: the finished one that serves
    its model_type), PLAIN, the experts' slots where its config routes experts and the recurrence slot where it has
    linear-attention layers."""
    config = _config(model)
    if plugin is None:
        found = plugin_for(_types(model))
        if found is None:
            raise ValueError(f"no finished model plugin serves model_type {' / '.join(_types(model)) or '?'} (crit "
                             "models list): build one from the model techniques (models/README.md) and name its "
                             "folder with --plugin")
        plugin = found["id"]
    use = {"model": plugin, **PLAIN}
    if config.moe is not None:
        use.update(ROUTED)
    if any(kind == "linear_attention" for kind in config.layer_kinds()):
        use.update(RECURRENT)
    return use


def reference_logprobs(model, ids, trust_remote_code=False):
    """The reference's log-probabilities over `ids` (FP32, its own modeling code)."""
    import torch
    import transformers
    reference = transformers.AutoModelForCausalLM.from_pretrained(str(model), dtype=torch.float32,
                                                                  trust_remote_code=trust_remote_code)
    with torch.no_grad():
        return torch.log_softmax(reference(torch.tensor([ids])).logits[0].float(), -1)


def engine_scores(model, ids, use=None, plugin=None):
    """Each next token's log-probability as an engine of a model plugin scores `ids`: Core's loader over a package of
    the plugin, the models core it extends and the plain plugins of the slots it asks (selection_for), copied with
    crit plugins add. plugin: the folder of a model plugin in an engine (ENGINE/plugins/ID), the one being built:
    the package is that engine's plugins as they are, and what the selection lacks is added from the library."""
    import tempfile
    from .plugins import IGNORE, add
    from crit_core.engine import load
    from crit_plugins.assemble import Registry, plugins_root
    layout = None
    if plugin is not None:
        problems = check_plugin(plugin)
        if problems:
            raise ValueError("; ".join(problems))
        plugin = Path(plugin).resolve()
        layout = json.loads((plugin / LAYOUT).read_text(encoding="utf-8"))
    use = use or selection_for(model, plugin.name if plugin is not None else None)
    with tempfile.TemporaryDirectory() as folder:
        package = Path(folder) / "package"
        if plugin is not None:
            shutil.copytree(plugins_root(plugin), package / "plugins", ignore=IGNORE)
        held = Registry.discover(package / "plugins") if (package / "plugins").is_dir() else ()
        lacking = sorted(set(use.values()) - {manifest["id"] for manifest in held})
        if lacking:
            add(package / "plugins", lacking)
        configuration = {"plugins": {"use": use, "options": {use["model"]: dict(SMALL),
                                                              **options_for(model, use, layout)}}}
        (package / "configuration.json").write_text(json.dumps(configuration), encoding="utf-8")
        import torch
        engine = load(package, model, device="cuda" if torch.cuda.is_available() else "cpu")
        try:
            return [float(x) for x in engine.score(ids, 1, "prefill", top_k=1).target]
        finally:
            engine.close()


def diff(model, use=None, tokens=24, seed=0, trust_remote_code=False, limit=0.25, plugin=None):
    """A model plugin against the reference on `model` (a tiny checkpoint): both score one random token sequence.
    {positions, max_abs, mean_abs, first_divergent}: per position the difference of the next token's
    log-probability, and the first position past `limit` nats (None when every one is within it). plugin: the
    folder of the model plugin being built (engine_scores), else the finished one for the model_type."""
    import random
    config = json.loads((Path(model) / "config.json").read_text(encoding="utf-8"))
    vocab = config.get("vocab_size") or config.get("text_config", {}).get("vocab_size")
    rng = random.Random(seed)
    ids = [rng.randrange(min(vocab, 250)) for _ in range(tokens)]
    logp = reference_logprobs(model, ids, trust_remote_code)
    expected = [float(logp[i, ids[i + 1]]) for i in range(len(ids) - 1)]
    errors = [abs(a - b) for a, b in zip(engine_scores(model, ids, use, plugin), expected)]
    first = next((i for i, e in enumerate(errors) if e > limit), None)
    return {"positions": len(errors), "max_abs": max(errors), "mean_abs": sum(errors) / len(errors),
            "first_divergent": first}


# -- cache --------------------------------------------------------------------------------------------------------
def check_plugin(folder):
    """The problems of a model plugin's folder in an engine ([] when none): ENGINE/plugins/ID holding a plugin.json
    that fills model, names the config model_types it serves (needs) and extends the models core, and a layout
    that composes techniques the core or the plugin gives (crit_models.plugin.layout)."""
    from crit_models import plugin as model_plugin
    from crit_plugins.assemble import plugins_root
    from crit_plugins.manifest import PluginError, load_manifest
    folder = Path(folder).resolve()
    if not (folder / MANIFEST).is_file():
        return [f"{folder}: no {MANIFEST}: name a model plugin's folder (ENGINE/plugins/ID)"]
    try:
        manifest = load_manifest(folder)
    except PluginError as error:
        return [str(error)]
    problems = []
    if manifest["fills"] != ["model"]:
        problems.append(f"{manifest['id']} fills {', '.join(manifest['fills']) or 'no slot'}: a model plugin fills model")
    if (manifest.get("extends") or {}).get("id") != model_plugin.CORE:
        problems.append(f"{manifest['id']} extends {(manifest.get('extends') or {}).get('id') or 'no plugin'}: a "
                        f"model plugin built from the model techniques extends the models core, {model_plugin.CORE}")
    if not any(all(fact.startswith("model_type:") for fact in need) for need in manifest["needs"]):
        problems.append(f"{manifest['id']}: needs names the config model_types it serves ([\"model_type:NAME\", ...])")
    if not (folder / LAYOUT).is_file():
        problems.append(f"{folder}: no {LAYOUT}, the layout it composes")
    elif not problems:
        try:
            model_plugin.layout(folder)
        except (OSError, ValueError, ImportError) as error:
            problems.append(f"{manifest['id']}: {error}")
    if plugins_root(folder) != folder.parent:
        problems.append(f"{folder} lies in a group of its engine's plugins: a model plugin sits at ENGINE/plugins/ID, "
                        "beside the models core crit plugins add put there")
    return problems


def library_sources(folder):
    """{source: path} of a model plugin's sources as crit's library holds them (models/cache/ID): each source of
    the plugin at `folder` (ENGINE/plugins/ID) that is a library plugin's copy or code under the engine's _sources,
    as its path from models/cache/ID to the library's folder; its own code as it is."""
    from .plugins import SOURCES, Library, library
    folder = Path(folder).resolve()
    places, registry, plugins = Library(), library(), folder.parent
    place = {(plugins / places.place(manifest)).resolve(): Path(manifest["folder"]).resolve() for manifest in registry}
    out = {}
    for name, source in json.loads((folder / MANIFEST).read_text(encoding="utf-8")).get("sources", {}).items():
        found = (folder / source).resolve()
        if found.is_relative_to(folder):
            out[name] = source
            continue
        if found.is_relative_to(plugins / SOURCES):
            there = places.folder(found.relative_to(plugins / SOURCES))
        else:
            owner = next((where for where in place if found.is_relative_to(where)), None)
            if owner is None:
                raise ValueError(f"{folder.name}: source {name} ({source}) is neither a library plugin's copy nor "
                                 "code under the engine's _sources: keep the plugin's own code in its folder")
            there = place[owner] / found.relative_to(owner)
        out[name] = Path(os.path.relpath(there, CACHE / folder.name)).as_posix()
    return out


def cache_add(folder, force=False, library=False):
    """Keep a finished model plugin (its folder in an engine, ENGINE/plugins/ID) so that it is built once: in the
    user's cache as it is, from where crit plugins add ID puts it into any engine; with `library`, in crit's own
    models/cache with its sources pointing at the library's folders (a checkout's, for a plugin crit ships).
    Returns the folder it was copied to."""
    from .plugins import IGNORE
    problems = check_plugin(folder)
    if problems:
        raise ValueError("; ".join(problems))
    folder = Path(folder).resolve()
    target = (CACHE if library else user_cache()) / folder.name
    sources = library_sources(folder) if library else None
    if target.exists():
        if not force:
            raise ValueError(f"{target} exists: --force replaces it")
        shutil.rmtree(target)
    shutil.copytree(folder, target, ignore=IGNORE)
    if library:
        manifest = json.loads((target / MANIFEST).read_text(encoding="utf-8"))
        manifest["sources"] = sources
        (target / MANIFEST).write_bytes((json.dumps(manifest, indent=2) + "\n").encode("utf-8"))
    return target


def parser():
    """The arguments of crit models (crit mcp offers its tools from them)."""
    parser = argparse.ArgumentParser(prog="crit models", description=__doc__.split("\n")[0])
    subs = parser.add_subparsers(dest="command", required=True)
    listed = subs.add_parser("list", help="The model techniques and the finished model plugins")
    listed.add_argument("--json", action="store_true")
    shown = subs.add_parser("inspect", help="What a checkpoint needs: the model plugin that serves it, techniques, "
                                            "what none reads")
    shown.add_argument("model", type=Path)
    shown.add_argument("--json", action="store_true")
    made = subs.add_parser("tiny", help="A tiny random checkpoint of the model's architecture, from its reference")
    made.add_argument("model", type=Path)
    made.add_argument("--out", type=Path, required=True)
    made.add_argument("--layers", type=int, help="Decoder layers to keep (default: up to 4)")
    made.add_argument("--trust-remote-code", action="store_true",
                      help="Run the checkpoint's own modeling code (downloaded code: only on the user's word)")
    compared = subs.add_parser("diff", help="A model plugin against the reference on a tiny checkpoint")
    compared.add_argument("--model", type=Path, required=True)
    compared.add_argument("--tokens", type=int, default=24)
    compared.add_argument("--selection", type=Path, help="A JSON {slot: plugin} for the engine (default: the model "
                          "plugin and plain ones)")
    compared.add_argument("--plugin", type=Path, help="The folder of the model plugin being built (ENGINE/plugins/"
                          "ID), served from its engine's plugins; default: the finished model plugin for the "
                          "model_type")
    compared.add_argument("--trust-remote-code", action="store_true")
    compared.add_argument("--json", action="store_true")
    cache = subs.add_parser("cache", help="Keep a finished model plugin, so that it is built once")
    actions = cache.add_subparsers(dest="action", required=True)
    added = actions.add_parser("add", help="Keep a model plugin (ENGINE/plugins/ID) in your cache: crit plugins add "
                                           "ID then puts it into any engine")
    added.add_argument("folder", type=Path)
    added.add_argument("--force", action="store_true", help="Replace the one kept under the same id")
    added.add_argument("--library", action="store_true", help="Keep it in crit's own models/cache (a checkout's), "
                       "its sources pointing at the library: a model plugin crit ships")
    checked = actions.add_parser("check", help="Check a model plugin's folder (ENGINE/plugins/ID)")
    checked.add_argument("folder", type=Path)
    return parser


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        if args.command == "list":
            rows = {"techniques": [{k: e[k] for k in ("id", "category", "provides", "summary")} for e in techniques()],
                    "plugins": [{"id": p["id"], "model_types": p["model_types"], "where": p["where"],
                                 "folder": str(p["folder"])} for p in model_plugins()]}
            if args.json:
                print(json.dumps(rows, indent=2))
            else:
                for e in rows["techniques"]:
                    kind, names = next(iter(e["provides"].items()))
                    print(f"{e['category'] + '/' + e['id']:<32} {kind[:-1]:<9} {', '.join(names):<28} {e['summary'][:70]}")
                for c in rows["plugins"]:
                    kept = "" if c["where"] == "library" else f"  (your cache: {c['folder']})"
                    print(f"{c['id']:<32} {'model':<9} {', '.join(c['model_types'])}{kept}")
            return 0
        if args.command == "inspect":
            found = inspect(args.model)
            if args.json:
                print(json.dumps(found, indent=2))
            else:
                print(f"fingerprint {found['fingerprint']}  model_type {found['model_type']}  {found['tensors']} tensors")
                print("model plugin: " + (", ".join(found["plugins"]) + " (crit plugins add puts it into an engine)"
                                          if found["plugins"] else "none: build one from the techniques "
                                          "(models/README.md), then crit models cache add it"))
                for ident, reads in found["techniques"].items():
                    print(f"  {ident:<22} config {', '.join(reads['config']) or '-'}; tensors {', '.join(reads['tensors']) or '-'}")
                print("read by no technique: config " + (", ".join(found["unread"]["config"]) or "-"))
                for p in found["unread"]["tensors"][:40]:
                    print("  tensor " + p)
            return 0
        if args.command == "tiny":
            print(tiny(args.model, args.out, args.layers, args.trust_remote_code))
            return 0
        if args.command == "diff":
            selection = json.loads(args.selection.read_text(encoding="utf-8")) if args.selection else None
            found = diff(args.model, selection, args.tokens, trust_remote_code=args.trust_remote_code,
                         plugin=args.plugin)
            print(json.dumps(found, indent=2) if args.json else
                  f"{found['positions']} positions: max |dlogp| {found['max_abs']:.4f}, mean {found['mean_abs']:.4f}, "
                  f"first past 0.25 nats: {found['first_divergent']}")
            return 0 if found["first_divergent"] is None else 1
        if args.command == "cache":
            if args.action == "add":
                print(cache_add(args.folder, args.force, args.library))
            else:
                problems = check_plugin(args.folder)
                print("\n".join(problems) or "model plugin checked")
                return 1 if problems else 0
            return 0
    except (OSError, ValueError, ImportError) as error:
        print(f"crit models: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
