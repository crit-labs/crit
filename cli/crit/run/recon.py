"""Model reconnaissance: the machine-readable facts a candidate freezes, gathered with their sources.

`crit run --id RUN recon` reads the run's model folder through Core's text layer and writes work/recon.json (the
facts, the files they came from and the `frozen` block every later candidate takes into configuration["recon"])
and scaffolds work/recon.md, whose model-card and reference-code sections the agent writes.
"""
import hashlib
from pathlib import Path

from .records import file_sha, keys, now, read_json, record, safe_path, text, write_json

RECON_JSON = "work/recon.json"
RECON_MD = "work/recon.md"
FROZEN = ("sampling", "chat_template", "template_variables", "stop_token_ids", "stop_strings")
# config.json fields that shape the engine, read at the top level and in text_config.
ARCHITECTURE_KEYS = (
    "architectures", "model_type", "torch_dtype", "dtype", "vocab_size", "hidden_size", "intermediate_size",
    "moe_intermediate_size", "num_hidden_layers", "num_attention_heads", "num_key_value_heads", "head_dim",
    "max_position_embeddings", "rope_theta", "rope_scaling", "rope_parameters", "partial_rotary_factor",
    "rms_norm_eps", "layer_norm_eps", "sliding_window", "layer_types", "full_attention_interval",
    "attention_bias", "tie_word_embeddings", "num_experts", "num_local_experts", "n_routed_experts",
    "num_experts_per_tok", "n_shared_experts", "kv_lora_rank", "q_lora_rank", "num_nextn_predict_layers",
    "mtp_num_hidden_layers", "quantization_config")
# Variables every chat template receives; the rest are the model's own switches (enable_thinking, ...).
STANDARD_VARIABLES = {"messages", "tools", "documents", "add_generation_prompt", "continue_final_message",
                      "raise_exception", "strftime_now", "bos_token", "eos_token", "unk_token", "sep_token",
                      "pad_token", "cls_token", "mask_token", "namespace", "range", "dict", "loop_messages"}
MODEL_CARD = "README.md"


def _architecture(config):
    found = {}
    for section in (config, config.get("text_config") if isinstance(config.get("text_config"), dict) else {}):
        for key in ARCHITECTURE_KEYS:
            if key in section and section[key] is not None and key not in found:
                found[key] = section[key]
    return found


def _template_variables(source):
    from jinja2 import meta
    from crit_core.text import environment
    try:
        names = meta.find_undeclared_variables(environment(None).parse(source))
    except Exception:  # a template Core cannot parse still gets recorded; rendering reports the error
        return []
    return sorted(names - STANDARD_VARIABLES)


def gather(model):
    """Facts from a model folder: sampling defaults, stop tokens, chat templates and config.json fields, plus
    {path, bytes, sha256} for every file read. A model given as a single file yields no facts."""
    from crit_core.files import model_files
    from crit_core.jsonfile import MAX_CONFIG_BYTES
    from crit_core.text import GenerationConfig, Tokenizer, template_sources
    from crit_core.text.errors import read_json_object
    model = Path(model)
    if not model.is_dir():
        raise ValueError(f"Recon reads a model folder; {model} is not one")
    files = model_files(model)
    tokenizer_config = read_json_object(files, "tokenizer_config.json") or {}
    tokenizer = (Tokenizer.from_files(files, tokenizer_config=tokenizer_config)
                 if files.read("tokenizer.json") is not None else None)
    generation = GenerationConfig.from_files(files, tokenizer)
    templates = template_sources(files, tokenizer_config)
    config = read_json_object(files, "config.json", max_bytes=MAX_CONFIG_BYTES, allow_nonfinite=True) or {}
    card = files.read(MODEL_CARD)
    rows = [row.row() for row in sorted(files.records, key=lambda r: r.path)]
    tokens = {}
    if tokenizer is not None:
        for token_id in generation.stop_token_ids:
            tokens[str(token_id)] = tokenizer.id_to_token(token_id)
    chat = {name: {"sha256": _sha_text(source), "variables": _template_variables(source)}
            for name, source in sorted(templates.items())}
    default = "default" if "default" in chat else (next(iter(chat)) if len(chat) == 1 else None)
    gathered = {
        "generation": {key: value for key, value in (
            ("sampling", generation.sampling_defaults()), ("max_new_tokens", generation.max_new_tokens),
            ("bos_token_id", generation.bos_token_id), ("pad_token_id", generation.pad_token_id)) if value is not None},
        "eos_token_ids": list(generation.eos_token_ids),
        "stop_token_ids": list(generation.stop_token_ids),
        "stop_token_text": tokens,
        "stop_strings": list(generation.stop_strings),
        "special_tokens": dict(tokenizer.special_tokens) if tokenizer is not None else {},
        "chat_templates": chat,
        "architecture": _architecture(config),
        "model_card": MODEL_CARD if card is not None else None,
    }
    frozen = {
        "sampling": generation.sampling_defaults(),
        "chat_template": {"name": default, "sha256": chat[default]["sha256"]} if default else None,
        "template_variables": {},
        "stop_token_ids": list(generation.stop_token_ids),
        "stop_strings": list(generation.stop_strings),
    }
    return rows, gathered, frozen


def _sha_text(source):
    return hashlib.sha256(source.encode("utf-8")).hexdigest()


def validate_frozen(value):
    """The block candidates freeze: sampling {name: number or bool}, chat_template {name, sha256} or null,
    template_variables {name: JSON value}, stop_token_ids [int], stop_strings [str]."""
    keys(value, set(FROZEN))
    if not isinstance(value["sampling"], dict) or not all(
            isinstance(k, str) and isinstance(v, (int, float, bool)) for k, v in value["sampling"].items()):
        raise ValueError("recon frozen.sampling must map names to numbers or booleans")
    template = value["chat_template"]
    if template is not None:
        keys(template, {"name", "sha256"})
        text(template["name"], "chat template name", 200)
    if not isinstance(value["template_variables"], dict):
        raise ValueError("recon frozen.template_variables must be an object")
    if not isinstance(value["stop_token_ids"], list) or not all(
            type(i) is int and 0 <= i < 2 ** 31 for i in value["stop_token_ids"]):
        raise ValueError("recon frozen.stop_token_ids must be a list of token ids")
    if not isinstance(value["stop_strings"], list) or not all(isinstance(s, str) and s for s in value["stop_strings"]):
        raise ValueError("recon frozen.stop_strings must be a list of non-empty strings")
    return value


def load(root, run):
    """(frozen block, {path, sha256}) of the run's work/recon.json, or (None, None) when recon has not run."""
    path = safe_path(root, RECON_JSON)
    if not path.is_file():
        return None, None
    value = read_json(path)
    keys(value, {"schema_version", "record_kind", "run_id", "created_at", "model_path", "sources", "gathered", "frozen"})
    if value["record_kind"] != "model_recon" or value["run_id"] != run["run_id"]:
        raise ValueError(f"{RECON_JSON} belongs to another run")
    if value["model_path"] != run["spec"]["model_path"]:
        raise ValueError(f"{RECON_JSON} describes another model; run recon again with --force")
    return validate_frozen(value["frozen"]), {"path": RECON_JSON, "sha256": file_sha(path)}


def write(root, run, force=False):
    """Gather the run's model, write work/recon.json and scaffold work/recon.md (kept when it exists)."""
    path = safe_path(root, RECON_JSON)
    if path.exists() and not force:
        raise ValueError(f"{RECON_JSON} exists; edit its frozen block, or pass --force to gather again")
    model = run["spec"]["model_path"]
    rows, gathered, frozen = gather(model)
    value = record("model_recon", run["run_id"], created_at=now(), model_path=model, sources=rows,
                   gathered=gathered, frozen=frozen)
    write_json(path, value)
    notes = safe_path(root, RECON_MD)
    if not notes.exists():
        notes.write_text(scaffold(value), encoding="utf-8")
    return value


def _items(mapping):
    return "".join(f"- `{k}`: `{v}`\n" for k, v in mapping.items()) or "- none set\n"


def scaffold(value):
    g, f = value["gathered"], value["frozen"]
    sources = "".join(f"| {r['path']} | {r['bytes']} | `{r['sha256'][:16]}` |\n" for r in value["sources"])
    stops = ", ".join(f"{i}" + (f" `{g['stop_token_text'][str(i)]}`" if str(i) in g["stop_token_text"] else "")
                      for i in g["stop_token_ids"]) or "none"
    templates = "".join(f"- `{name}` (sha256 `{t['sha256'][:16]}`), model variables: "
                        f"{', '.join(t['variables']) or 'none'}\n" for name, t in g["chat_templates"].items()) or "- none\n"
    card = (f"Read `{g['model_card']}` in the model folder (or the model's Hugging Face page)"
            if g["model_card"] else "No README.md in the model folder: read the model's Hugging Face page")
    return f"""# Model recon

Model: `{value['model_path']}`, gathered {value['created_at']} by `crit run ... recon`.
`recon.json` holds the facts below with the files they came from. Every candidate created after it freezes its
`frozen` block into `configuration["recon"]`. Change `frozen` only with a reason under Decisions, before the next
candidate. Eval's speed and exactness cases stay greedy by protocol.

## Sources

| File | Bytes | SHA-256 |
| --- | --- | --- |
{sources}
## Sampling (generation_config.json, else config.json)

{_items(g['generation'].get('sampling', {}))}
## Stop tokens

- eos_token_ids: {g['eos_token_ids']}
- stop_token_ids: {stops}
- stop_strings: {g['stop_strings']}
- special tokens: {g['special_tokens'] or 'none read'}

## Chat templates

{templates}
## Architecture (config.json)

{_items(g['architecture'])}
## Model card

TODO. {card}: recommended sampling per mode (thinking and non-thinking), known issues, context limits. Cite the
source and its revision.

## Reference implementation

TODO. The transformers or vLLM modeling file and its revision: rope scaling, MTP or draft heads, norm and
attention quirks. Cite file and line.

## Decisions

Frozen for candidates: sampling `{f['sampling']}`, chat template `{(f['chat_template'] or {}).get('name')}`, template
variables `{f['template_variables']}`, stop token ids `{f['stop_token_ids']}`, stop strings `{f['stop_strings']}`.
TODO: say why wherever this differs from generation_config.json.
"""
