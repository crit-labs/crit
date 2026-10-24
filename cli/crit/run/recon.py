"""Model reconnaissance: the machine-readable facts a candidate freezes, gathered with their sources.

`crit run --id RUN recon` reads the run's model (crit_core.model.open_model: a folder, or a weight file such as a
lone .gguf whose metadata derives the description files) through Core's text layer and writes work/recon.json (the
facts, the files they came from and the `frozen` block every later candidate takes into configuration["recon"])
and scaffolds work/recon.md, whose model-card, reference-code and drafting-plan sections the agent writes.

The sampling is not the agent's to choose. recon.json records the sampling crit Eval's sampled suite measures (the
model's generation_config.json, else the protocol default, else the user's eval_sampling override from run create),
and the served default (`frozen.sampling`) is the model's sampling with that override applied; a candidate refuses a
frozen block whose sampling differs from it.
"""
import hashlib

from .records import file_sha, keys, now, read_json, record, safe_path, text, write_json

RECON_JSON = "work/recon.json"
RECON_MD = "work/recon.md"
FROZEN = ("sampling", "chat_template", "template_variables", "stop_token_ids", "stop_strings")
# config.json fields that shape the engine, read at the top level and in the text section (crit_core.model's
# TEXT_SECTION_KEYS: text_config, else llm_config).
ARCHITECTURE_KEYS = (
    "architectures", "model_type", "torch_dtype", "dtype", "vocab_size", "hidden_size", "intermediate_size",
    "moe_intermediate_size", "num_hidden_layers", "num_attention_heads", "num_key_value_heads", "head_dim",
    "max_position_embeddings", "rope_theta", "rope_scaling", "rope_parameters", "partial_rotary_factor",
    "rms_norm_eps", "layer_norm_eps", "sliding_window", "layer_types", "full_attention_interval",
    "attention_bias", "tie_word_embeddings", "num_experts", "num_local_experts", "n_routed_experts",
    "num_experts_per_tok", "n_shared_experts", "kv_lora_rank", "q_lora_rank", "num_nextn_predict_layers",
    "mtp_num_hidden_layers", "quantization_config", "layers_block_type", "mtp_layers_block_type",
    "mamba_num_heads", "mamba_head_dim", "ssm_state_size", "n_groups", "conv_kernel", "chunk_size",
    "moe_latent_size", "moe_shared_expert_intermediate_size", "routed_scaling_factor", "norm_topk_prob",
    "layer_norm_epsilon")
# Variables every chat template receives; the rest are the model's own switches (enable_thinking, ...).
STANDARD_VARIABLES = {"messages", "tools", "documents", "add_generation_prompt", "continue_final_message",
                      "raise_exception", "strftime_now", "bos_token", "eos_token", "unk_token", "sep_token",
                      "pad_token", "cls_token", "mask_token", "namespace", "range", "dict", "loop_messages"}
MODEL_CARD = "README.md"


# Per-module lists in quantization_config (ModelOpt lists every quantized and ignored module) are summarized by
# their length and first entries so recon.json stays small; the rest of the block is kept as written.
QUANT_LISTS_KEPT = 8


def _quantization(value):
    if not isinstance(value, dict):
        return value
    out = {}
    for key, item in value.items():
        if isinstance(item, list) and len(item) > QUANT_LISTS_KEPT:
            out[key] = {"count": len(item), "first": item[:QUANT_LISTS_KEPT]}
        elif isinstance(item, dict) and len(item) > 4 * QUANT_LISTS_KEPT:
            out[key] = {"count": len(item), "first": dict(list(item.items())[:QUANT_LISTS_KEPT])}
        elif isinstance(item, dict):
            out[key] = _quantization(item)
        else:
            out[key] = item
    return out


def _architecture(config):
    found = {}
    from crit_core.model.config import TEXT_SECTION_KEYS
    text = next((config.get(key) for key in TEXT_SECTION_KEYS if key in config), None)
    for section in (config, text if isinstance(text, dict) else {}):
        for key in ARCHITECTURE_KEYS:
            if key in section and section[key] is not None and key not in found:
                found[key] = _quantization(section[key]) if key == "quantization_config" else section[key]
    return found


def _template_variables(source):
    from jinja2 import meta
    from crit_core.text import environment
    try:
        names = meta.find_undeclared_variables(environment(None).parse(source))
    except Exception:  # a template Core cannot parse still gets recorded; rendering reports the error
        return []
    return sorted(names - STANDARD_VARIABLES)


SAMPLED_KEYS = ("temperature", "top_p", "top_k", "min_p")


def served_sampling(model_sampling, override=None):
    """The served default sampling: the model's own, or with the user's eval_sampling settings in place of its
    temperature and filters."""
    if not override:
        return dict(model_sampling)
    kept = {key: value for key, value in model_sampling.items() if key not in SAMPLED_KEYS}
    return {**kept, "do_sample": True, **override}


def eval_sampling(generation, run):
    """{settings, source} of crit Eval's sampled suite for this run (crit_eval.prepare.sampled_settings, or the run's
    eval_sampling override)."""
    override = run["spec"].get("eval_sampling")
    if override:
        return {"settings": override["settings"], "source": "override", "approved_by": override["approved_by"]}
    from crit_eval.prepare import sampled_settings, sampling_source
    return {"settings": sampled_settings(generation), "source": sampling_source(generation)}


def gather(model, run=None):
    """Facts from a model's description files: sampling defaults, stop tokens, chat templates and config.json
    fields, plus {path, bytes, sha256} for every file read. A weight file without description files (a lone
    .safetensors) yields no facts."""
    from crit_core.jsonfile import MAX_CONFIG_BYTES
    from crit_core.model import open_model
    from crit_core.text import GenerationConfig, Tokenizer, template_sources
    from crit_core.text.errors import read_json_object
    source = open_model(model)
    if source.folder is None and not source.derived:
        raise ValueError(f"Recon reads a model's description files; {model} has none")
    files = source.files
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
        # Files the weights' own metadata supplied (a GGUF without the base model's files beside it).
        "derived_files": list(source.derived_names),
    }
    if run is not None:
        gathered["eval_sampling"] = eval_sampling(generation, run)
    override = (run or {}).get("spec", {}).get("eval_sampling")
    frozen = {
        "sampling": served_sampling(generation.sampling_defaults(), override and override["settings"]),
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
    frozen = validate_frozen(value["frozen"])
    override = run["spec"].get("eval_sampling")
    expected = served_sampling(value["gathered"].get("generation", {}).get("sampling", {}), override and override["settings"])
    if frozen["sampling"] != expected:
        raise ValueError(f"{RECON_JSON} frozen.sampling {frozen['sampling']} is not the model's sampling {expected}: the "
                         "sampling is not the engine builder's to change. Only the user overrides it, with eval_sampling "
                         "at run create; put frozen.sampling back")
    return frozen, {"path": RECON_JSON, "sha256": file_sha(path)}


def write(root, run, force=False):
    """Gather the run's model, write work/recon.json and scaffold work/recon.md (kept when it exists)."""
    path = safe_path(root, RECON_JSON)
    if path.exists() and not force:
        raise ValueError(f"{RECON_JSON} exists; edit its frozen block, or pass --force to gather again")
    model = run["spec"]["model_path"]
    rows, gathered, frozen = gather(model, run)
    value = record("model_recon", run["run_id"], created_at=now(), model_path=model, sources=rows,
                   gathered=gathered, frozen=frozen)
    write_json(path, value)
    notes = safe_path(root, RECON_MD)
    if not notes.exists():
        notes.write_text(scaffold({**value, "context": required_context(run, gathered)}), encoding="utf-8")
    return value


def _items(mapping):
    return "".join(f"- `{k}`: `{v}`\n" for k, v in mapping.items()) or "- none set\n"


def required_context(run, gathered):
    """The run's required context (kv_tokens) with how it came about and the KV cache it takes at the protocol's
    precision, for recon.md."""
    constraints = run["spec"]["hardware_constraints"]
    basis = constraints.get("context") or {}
    per_token = basis.get("kv_bytes_per_token")
    return {"tokens": constraints["kv_tokens"], **{key: basis.get(key) for key in (
        "requested", "bound", "note", "model_limit", "device_limit", "kv_precision")},
            "kv_bytes": per_token * constraints["kv_tokens"] if per_token else None}


def _context(found):
    if not found.get("tokens"):
        return "- not recorded\n"
    if found.get("bound"):
        why = {"requested": "the context the user wants (settings.toml context_tokens)",
               "model": f"the model's {found['model_limit']:,} positions" if found.get("model_limit") else "the model's positions",
               "device": "the KV cache that fits the device next to the weights, headroom and reserve",
               "crit": "crit's largest supported context"}.get(found["bound"], found["bound"])
        source = f"bounded by {why}" + (f" (requested {found['requested']:,})" if found.get("requested") else "")
    else:
        source = "given explicitly as kv_tokens at run create"
    lines = f"Every candidate must serve {found['tokens']:,} tokens of prompt plus output, {source}.\n"
    if found.get("note"):
        lines += f"Note: {found['note']}.\n"
    if found.get("kv_bytes"):
        lines += (f"Its KV cache takes {found['kv_bytes'] / (1 << 30):.2f} GiB at {found.get('kv_precision') or 'the protocol'} "
                  "precision.\n")
    return lines + ("You cannot change it: only the user does (context_tokens in settings.toml, or kv_tokens at run "
                    "create). Eval's capacity gate fails a candidate that cannot serve it.\n\nTODO. Plan the KV cache "
                    "for it: precision, layout and paging, where it lives next to the weights, drafter state and "
                    "workspaces, and how decode and time to first token hold up at this context.\n")


def _eval_sampling(found):
    if not found:
        return "- not recorded (recon from before crit-eval-v7): run recon again with --force\n"
    source = {"generation_config": "the model's generation_config.json",
              "protocol_default": "the protocol default (the model does not sample by default)",
              "override": f"the user's eval_sampling override (approved by {found.get('approved_by')})"}[found["source"]]
    return (f"crit Eval's sampled speed suite and sampling check run at `{found['settings']}`, from {source}. You cannot "
            "change them: they are the model's, or the user's alone. Only the user overrides them, with `eval_sampling` "
            "at run create (and the data prepared with `crit eval prepare --sampling`). Plan the drafting for them "
            "(below), and keep every other temperature correct: requests choose their own sampling.\n")


def scaffold(value):
    g, f = value["gathered"], value["frozen"]
    sources = "".join(f"| {r['path']} | {r['bytes']} | `{r['sha256'][:16]}` |\n" for r in value["sources"])
    stops = ", ".join(f"{i}" + (f" `{g['stop_token_text'][str(i)]}`" if str(i) in g["stop_token_text"] else "")
                      for i in g["stop_token_ids"]) or "none"
    templates = "".join(f"- `{name}` (sha256 `{t['sha256'][:16]}`), model variables: "
                        f"{', '.join(t['variables']) or 'none'}\n" for name, t in g["chat_templates"].items()) or "- none\n"
    card = (f"Read `{g['model_card']}` in the model folder (or the model's Hugging Face page)"
            if g["model_card"] else "No README.md in the model folder: read the model's Hugging Face page")
    if g.get("derived_files"):
        card += (f". {', '.join(f'`{n}`' for n in g['derived_files'])} came from the weights' own metadata (crit "
                 "derives them from a GGUF); check them against the base model's repository")
    return f"""# Model recon

Model: `{value['model_path']}`, gathered {value['created_at']} by `crit run ... recon`.
`recon.json` holds the facts below with the files they came from. Every candidate created after it freezes its
`frozen` block into `configuration["recon"]`. Change `frozen` only with a reason under Decisions, before the next
candidate; its sampling is not yours to change.

## Sources

| File | Bytes | SHA-256 |
| --- | --- | --- |
{sources}
## Sampling (generation_config.json, else config.json)

{_items(g['generation'].get('sampling', {}))}
## Eval sampling

{_eval_sampling(g.get('eval_sampling'))}
## Required context

{_context(value.get('context') or {})}
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

TODO. The transformers modeling file and its revision: rope scaling, MTP or draft heads, norm and
attention quirks. Cite file and line.

## Drafting plan for both decode modes

TODO. How the engine decodes fast in both modes Eval scores. Greedy (temperature 0): speculative output must stay
token-exact against plain greedy decoding. Sampled, at the Eval sampling above: the headline decode and its goal;
verify drafts by rejection sampling (accept a draft with probability min(1, p/q) under the processed target
distribution p, sample from max(p - q, 0) on the first rejection; a greedy draft is q one-hot) so the output keeps
the model's distribution and passes the `sampling` gate, and sample on the device inside the decode graph. Say which
drafter, draft width and acceptance you expect at these settings, and what a request at other settings gets.

## Decisions

Frozen for candidates: sampling `{f['sampling']}`, chat template `{(f['chat_template'] or {}).get('name')}`, template
variables `{f['template_variables']}`, stop token ids `{f['stop_token_ids']}`, stop strings `{f['stop_strings']}`.
TODO: say why wherever this differs from generation_config.json.
"""
