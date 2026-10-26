"""Model reconnaissance: the machine-readable facts a candidate freezes, gathered with their sources.

`crit run --id RUN recon` reads the run's model (crit_core.model.open_model: a folder, or a weight file such as a
lone .gguf whose metadata derives the description files) through Core's text layer and writes work/recon.json (the
facts, the files they came from and the `frozen` block every later candidate takes into configuration["recon"])
and scaffolds work/recon.md, whose model-card, reference-code, drafting-plan and prefill-arithmetic-plan sections the
agent writes (runs with plan rules 3 refuse a candidate while either plan is unfinished: recon.unfinished). With
`--drafter PATH` (repeatable) it also reads each companion drafter's config.json through crit_core.model.drafters
(EAGLE-3, DFlash, DSpark and crit's own `crit models drafter` heads) and records what the engine wires: kind, taps
into the target, draft vocabulary, block size and mask token, checked against the target's depth.

The sampling is not the agent's to choose. recon.json records the sampling crit Eval's sampled suite measures (the
model's generation_config.json, else the protocol default, else the user's eval_sampling override from run create),
and the served default (`frozen.sampling`) is the model's sampling with that override applied; a candidate refuses a
frozen block whose sampling differs from it.
"""
import hashlib
from pathlib import Path

from .records import file_sha, keys, now, read_json, record, safe_path, text, write_json

RECON_JSON = "work/recon.json"
RECON_MD = "work/recon.md"
# The recon.md sections a run with plan rules 3 needs written before its first candidate: decode's and prefill's plans.
PLANS = ("Drafting plan for both decode modes", "Prefill arithmetic plan")
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


def _drafters(paths, architecture):
    """Each companion drafter as Core's reader describes it, with its config.json's digest."""
    from crit_core.model.config import ConfigError
    from crit_core.model.drafters import read_drafter_file
    out = []
    layers = architecture.get("num_hidden_layers")
    for path in paths:
        path = Path(path).expanduser().resolve()
        config = path / "config.json" if path.is_dir() else path
        try:
            drafter = read_drafter_file(path, target_layers=layers if isinstance(layers, int) and layers > 0 else None)
        except ConfigError as error:
            raise ValueError(f"Drafter {path}: {error}") from None
        if drafter is None:
            raise ValueError(f"Drafter {path}: not a hidden-state drafter crit reads (EAGLE-3, DFlash, DSpark or crit models drafter)")
        hidden = architecture.get("hidden_size")
        if drafter.target_hidden_size and isinstance(hidden, int) and drafter.target_hidden_size != hidden:
            raise ValueError(f"Drafter {path}: its taps are {drafter.target_hidden_size} wide; the target's hidden size is {hidden}")
        out.append({"path": str(path), "config_sha256": file_sha(config), "kind": drafter.kind, "taps": list(drafter.taps),
                    "drafts_per_pass": drafter.drafts_per_pass, "hidden_size": drafter.hidden_size,
                    "target_hidden_size": drafter.target_hidden_size, "draft_vocab_size": drafter.draft_vocab_size,
                    "layers": drafter.layers, "block_size": drafter.block_size, "mask_token_id": drafter.mask_token_id,
                    "sample_from_anchor": drafter.sample_from_anchor, "causal": drafter.causal,
                    "confidence_head": drafter.confidence_head, "speculative_tokens": drafter.speculative_tokens,
                    "verifier": drafter.verifier, "format": drafter.source})
    return out


def gather(model, run=None, drafters=()):
    """Facts from a model's description files: sampling defaults, stop tokens, chat templates and config.json
    fields, plus {path, bytes, sha256} for every file read, and each companion drafter in `drafters`. A weight file
    without description files (a lone .safetensors) yields no facts."""
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
    tokens = {}
    if tokenizer is not None:
        for token_id in generation.stop_token_ids:
            tokens[str(token_id)] = tokenizer.id_to_token(token_id)
    chat = {name: {"sha256": _sha_text(template), "variables": _template_variables(template)}
            for name, template in sorted(templates.items())}
    default = "default" if "default" in chat else (next(iter(chat)) if len(chat) == 1 else None)
    architecture = _architecture(config)
    print_ = fingerprint(source, files, architecture, chat, generation, tokenizer)
    rows = [row.row() for row in sorted(files.records, key=lambda r: r.path)]
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
        "architecture": architecture,
        "fingerprint": print_,
        "model_card": MODEL_CARD if card is not None else None,
        # Files the weights' own metadata supplied (a GGUF without the base model's files beside it).
        "derived_files": list(source.derived_names),
        "drafters": _drafters(drafters, architecture),
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


FINGERPRINT_VERSION = "crit-fingerprint-v1"
# config.json keys about how the weights are stored rather than what the model computes; they belong to the
# fingerprint's weights section, read from the weight headers themselves.
STORAGE_KEYS = ("torch_dtype", "dtype", "quantization_config")
TOKENIZER_FILES = ("tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt", "tokenizer.model",
                   "special_tokens_map.json")
FINGERPRINT_SECTIONS = ("architecture", "weights", "tokenizer", "template")


def _pattern(name):
    """A tensor name with its layer and expert numbers replaced by *, so one pattern covers every layer."""
    return ".".join("*" if part.isdigit() else part for part in name.split("."))


def weight_layout(source):
    """{tensor pattern: [{dtype, shape, count}]} over every weight file, draft layers included: the stored types
    and shapes an engine's loading and arithmetic are written for."""
    layout = {}
    if source.format is None:
        return layout
    for shard in source.weight_files(drafts=True):
        for tensor in shard.tensors:
            variants = layout.setdefault(_pattern(tensor.name), {})
            key = (str(tensor.dtype), tuple(int(d) for d in tensor.shape))
            variants[key] = variants.get(key, 0) + 1
    return {pattern: [{"dtype": dtype, "shape": list(shape), "count": count}
                      for (dtype, shape), count in sorted(variants.items())]
            for pattern, variants in sorted(layout.items())}


def fingerprint(source, files, architecture, chat, generation, tokenizer):
    """What an engine is written against, in four sections, each hashed so two checkpoints compare by section:
    architecture (config.json's model shape), weights (format, quantization and the stored tensor layout),
    tokenizer (its files, special tokens and stops) and template (the chat templates). Two checkpoints with the
    same fingerprint differ in weight values only, such as a finetune or a re-upload."""
    tokenizer_files = {}
    for name in TOKENIZER_FILES:
        data = files.read(name)
        if data is not None:
            tokenizer_files[name] = hashlib.sha256(data).hexdigest()
    sections = {
        "architecture": {key: value for key, value in architecture.items() if key not in STORAGE_KEYS},
        "weights": {"format": source.format, "dtype": architecture.get("torch_dtype", architecture.get("dtype")),
                    "quantization": architecture.get("quantization_config"), "layout": weight_layout(source)},
        "tokenizer": {"files": tokenizer_files, "special_tokens": dict(tokenizer.special_tokens) if tokenizer is not None else {},
                      "eos_token_ids": list(generation.eos_token_ids), "stop_token_ids": list(generation.stop_token_ids),
                      "vocabulary": tokenizer.vocab_size if tokenizer is not None and hasattr(tokenizer, "vocab_size") else None},
        "template": {name: entry["sha256"] for name, entry in chat.items()},
    }
    from .records import digest
    return {"version": FINGERPRINT_VERSION, **sections,
            "hashes": {name: digest(sections[name]) for name in FINGERPRINT_SECTIONS}, "sha256": digest(sections)}


def _diff(section, a, b, prefix=""):
    rows = []
    if isinstance(a, dict) and isinstance(b, dict):
        for key in sorted(a.keys() | b.keys()):
            if key not in a or key not in b or a[key] != b[key]:
                if isinstance(a.get(key), dict) and isinstance(b.get(key), dict):
                    rows.extend(_diff(section, a[key], b[key], prefix + key + "."))
                else:
                    rows.append({"section": section, "key": prefix + key, "a": a.get(key), "b": b.get(key)})
    elif a != b:
        rows.append({"section": section, "key": prefix.rstrip("."), "a": a, "b": b})
    return rows


def compare(a, b):
    """Compare two fingerprints: which sections differ, every differing key with both values, and what each
    difference means for an engine built against `a`."""
    if a.get("version") != FINGERPRINT_VERSION or b.get("version") != FINGERPRINT_VERSION:
        raise ValueError(f"Both fingerprints must be {FINGERPRINT_VERSION}; run recon again with --force")
    sections, differences = {}, []
    for name in FINGERPRINT_SECTIONS:
        same = a["hashes"][name] == b["hashes"][name]
        sections[name] = "same" if same else "differs"
        if not same:
            differences.extend(_diff(name, a[name], b[name]))
    notes = []
    if all(state == "same" for state in sections.values()):
        notes.append("Same architecture, weight layout, tokenizer and chat template: the checkpoints differ in weight "
                     "values only (a finetune or a re-upload). An engine built for one loads the other unchanged; "
                     "its speed should match and its quality needs Eval against the new checkpoint's reference.")
    if sections["architecture"] == "differs":
        keys = sorted({row["key"] for row in differences if row["section"] == "architecture"})
        notes.append("Architecture differs in " + ", ".join(keys) + ": model code, kernels and state layouts depend "
                     "on these, so the engine needs changes before it runs the other checkpoint.")
    if sections["weights"] == "differs":
        rows = [row for row in differences if row["section"] == "weights"]
        patterns = sorted({row["key"].split("layout.", 1)[1] for row in rows if row["key"].startswith("layout.")})
        other = sorted({row["key"] for row in rows if not row["key"].startswith("layout.")})
        parts = []
        if other:
            parts.append("storage differs in " + ", ".join(other))
        if patterns:
            parts.append(f"{len(patterns)} tensor patterns differ in dtype or shape (first: {', '.join(patterns[:4])})")
        notes.append("Weights: " + "; ".join(parts) + ". Loading and the arithmetic paths for those tensors change; "
                     "model code does not.")
    if sections["tokenizer"] == "differs":
        keys = sorted({row["key"] for row in differences if row["section"] == "tokenizer"})
        notes.append("Tokenizer differs in " + ", ".join(keys) + ": Core's text layer reads the new files; the "
                     "embedding and output-head rows follow the vocabulary, and the stop tokens follow the files.")
    if sections["template"] == "differs":
        notes.append("Chat template differs: Core renders it, so no engine code depends on it, but the tool-call and "
                     "reasoning formats the engine declares may change.")
    return {"same": all(state == "same" for state in sections.values()), "sections": sections,
            "differences": differences, "assessment": notes}


def compare_models(a, b):
    """Gather both models and compare their fingerprints: {a, b, comparison}."""
    prints = []
    for model in (a, b):
        _, gathered, _ = gather(model)
        prints.append(gathered["fingerprint"])
    return {"a": {"model": str(a), "fingerprint_sha256": prints[0]["sha256"]},
            "b": {"model": str(b), "fingerprint_sha256": prints[1]["sha256"]}, "comparison": compare(*prints)}


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
    reference = {"path": RECON_JSON, "sha256": file_sha(path)}
    print_ = value["gathered"].get("fingerprint")
    if isinstance(print_, dict) and isinstance(print_.get("sha256"), str):
        # The checkpoint's fingerprint travels with the candidate, so a sealed engine says what it was built against.
        reference["fingerprint_sha256"] = print_["sha256"]
    return frozen, reference


def compare_with_run(root, run, other):
    """Compare the run's model (its recon.json fingerprint) with another checkpoint, without writing anything."""
    path = safe_path(root, RECON_JSON)
    if not path.is_file():
        raise ValueError(f"Run recon first: {RECON_JSON} holds the run's model fingerprint")
    mine = read_json(path)["gathered"].get("fingerprint")
    if not isinstance(mine, dict):
        raise ValueError(f"{RECON_JSON} predates fingerprints; run recon again with --force")
    _, gathered, _ = gather(other)
    theirs = gathered["fingerprint"]
    return {"run_model": {"model": run["spec"]["model_path"], "fingerprint_sha256": mine["sha256"]},
            "other": {"model": str(other), "fingerprint_sha256": theirs["sha256"]}, "comparison": compare(mine, theirs)}


def write(root, run, force=False, drafters=()):
    """Gather the run's model (and companion drafters), write work/recon.json and scaffold work/recon.md (kept when
    it exists)."""
    path = safe_path(root, RECON_JSON)
    if path.exists() and not force:
        raise ValueError(f"{RECON_JSON} exists; edit its frozen block, or pass --force to gather again")
    model = run["spec"]["model_path"]
    rows, gathered, frozen = gather(model, run, drafters)
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


def unfinished(root):
    """The PLANS sections of the run's recon.md that are missing or still start with TODO; [] without a recon.md."""
    notes = safe_path(root, RECON_MD)
    if not notes.is_file():
        return []
    text_ = notes.read_text(encoding="utf-8")
    found = []
    for title in PLANS:
        head = f"## {title}"
        if head not in text_:
            found.append(title)
            continue
        body = text_.split(head, 1)[1].split("\n## ", 1)[0].strip()
        if not body or body.startswith("TODO"):
            found.append(title)
    return found


def scaffold(value):
    g, f = value["gathered"], value["frozen"]
    sources = "".join(f"| {r['path']} | {r['bytes']} | `{r['sha256'][:16]}` |\n" for r in value["sources"])
    stops = ", ".join(f"{i}" + (f" `{g['stop_token_text'][str(i)]}`" if str(i) in g["stop_token_text"] else "")
                      for i in g["stop_token_ids"]) or "none"
    templates = "".join(f"- `{name}` (sha256 `{t['sha256'][:16]}`), model variables: "
                        f"{', '.join(t['variables']) or 'none'}\n" for name, t in g["chat_templates"].items()) or "- none\n"
    card = (f"Read `{g['model_card']}` in the model folder (or the model's Hugging Face page)"
            if g["model_card"] else "No README.md in the model folder: read the model's Hugging Face page")
    found = g.get("drafters") or []
    drafters = "".join(f"- `{d['path']}`: {d['kind']} ({d['format']}), taps {d['taps']}, {d['drafts_per_pass']} draft(s) per "
                       f"pass" + (f", block {d['block_size']}, mask token {d['mask_token_id']}" if d["block_size"] else "")
                       + (f", draft vocabulary {d['draft_vocab_size']} (map through d2t)" if d["draft_vocab_size"] else "")
                       + (f", publisher width {d['speculative_tokens']}" if d["speculative_tokens"] else "") + "\n"
                       for d in found) or ("- none given: `crit run ... recon --force --drafter PATH` records a published "
                                         "EAGLE-3, DFlash or DSpark drafter, or one from `crit models drafter`\n")
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
## Fingerprint

`{g['fingerprint']['sha256']}` ({g['fingerprint']['version']}): architecture `{g['fingerprint']['hashes']['architecture'][:16]}`,
weights `{g['fingerprint']['hashes']['weights'][:16]}`, tokenizer `{g['fingerprint']['hashes']['tokenizer'][:16]}`,
template `{g['fingerprint']['hashes']['template'][:16]}`. Every candidate records it. To see what another checkpoint
(a finetune, a re-quantization, a related model) changes for this engine, run
`crit run ... recon --compare OTHER_MODEL`: same fingerprint means weight values differ only.

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

Companion drafters read (Inference's hidden-state-drafters entry wires them):

{drafters}
## Prefill arithmetic plan

TODO. How the engine prefills fast inside the drift budget, as the drafting plan says for decode. For each matrix
group of the checkpoint (attention projections, MLP, recurrent projections, ...) by its stored format: the activation
format its prompt GEMMs take (bf16, fp16_acc, fp8 or stored, as crit-eval prices them) and the quality measurement
that allows it: `crit eval run --adapter A --only quality --attribute-drift`, the engine declaring
`prompt_activations` and one `exact_variants` entry per group, each group's share of the budget read against the
quality spread (`budget_spread`). Name the prefill ceiling crit-eval prices for this plan next to the stored-format
one.

## Decisions

Frozen for candidates: sampling `{f['sampling']}`, chat template `{(f['chat_template'] or {}).get('name')}`, template
variables `{f['template_variables']}`, stop token ids `{f['stop_token_ids']}`, stop strings `{f['stop_strings']}`.
TODO: say why wherever this differs from generation_config.json.
"""
