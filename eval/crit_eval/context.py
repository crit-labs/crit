"""The longest request an engine must hold: `crit eval context --model DIR --device device.json`.

    max context = min(requested, model limit, device limit)      tokens of prompt plus output

The requested context is the user's (settings.toml context_tokens, default 262,144; `--requested-context`): the
longest request they want served. The model and the device may allow less, and then bound it; the result says
which bound and why.

The model limit is the number of positions the checkpoint declares in its config: `max_position_embeddings`
(or another length key), or with YaRN rope scaling in the config, its original positions times the factor.

The device limit is the KV cache that fits in the device's free memory next to everything that has to
stay there. Weights read on every pass stay in device memory: all of them except gathered embedding
rows (unless the output head is tied to them), the vision tower and mixture-of-experts experts. Experts
are read sparsely, so they can move to host memory or disk; at the maximum context they get only the
device memory the KV cache leaves. Draft layers count as resident. State that does not grow with the
context (linear-attention recurrence, sliding-window KV) is counted once. The KV cache is counted at
the protocol's KV precision (FP8 by default; the quality gate judges it), and a reserve covers
activations, workspaces and graphs. The run's VRAM headroom stays free:

    device limit = (free memory - headroom - reserve - resident weights - fixed state) / KV bytes per token

rounded down to the protocol's granularity. The KV cache never leaves device memory: decode reads all
of it on every pass, so a context beyond the device limit is a cliff, not a slope.

Below the maximum, an engine sized for it must be at least as fast: Eval measures the engine built for
the maximum, a curve of long-context points up to it, and the speed cases again at a lower context.
"""
import re

from .ceiling import model_config, weight_files
from .tensors import tensor_role

GIB = 1 << 30
DRAFT_HEAD = re.compile(r"(^|\.)head\.weight$")   # a draft layer's own output head (DeepSeek-style shared_head.head)


def weight_split(model, config):
    """Stored bytes by where they have to live: `resident` (read on every pass, kept in device memory),
    `experts` (read sparsely, may move to host memory or disk), `draft` (part of resident), `embedding`
    (gathered rows, host memory is enough) and `vision` (not loaded by a text engine). None without
    safetensors files or a GGUF. config: the model's crit_core.model.ModelConfig."""
    shards, drafts = weight_files(model, drafts=True)
    if not shards:
        return None
    split = {"resident": 0, "experts": 0, "draft": 0, "embedding": 0, "vision": 0}
    tied = bool(config.tie_word_embeddings)
    heads = 0
    for shard in shards:
        for tensor in shard.tensors:
            role, size = tensor_role(tensor.name, shard.path.name, drafts), tensor.nbytes
            if role.kind == "other_modality":
                split["vision"] += size
            elif role.kind == "embedding":
                split["embedding"] += size
            elif role.kind == "expert":
                split["experts"] += size
            else:
                split["resident"] += size
                heads += role.kind == "head"
                if role.draft:
                    split["draft"] += size
    if tied and not heads:
        # The output head is the embedding table: every pass reads all of it.
        split["resident"] += split["embedding"]
        split["embedding"] = 0
    return split


def pass_bytes(model, kv_bytes_per_element):
    """What one decode pass of one token reads from the stored model, for an engine that reports no bytes_per_pass:
    the weights read on every pass but draft layers, the routed experts' share (experts_per_token of the experts,
    taken as equally sized) and the fixed state (linear-attention and Mamba recurrence, sliding-window KV at
    kv_bytes_per_element). None without weight headers."""
    config = model_config(model)
    split = weight_split(model, config)
    if split is None:
        return None
    moe = config.moe
    experts = (split["experts"] * min(1.0, moe.experts_per_token / moe.experts)
               if split["experts"] and moe and moe.experts and moe.experts_per_token else 0)
    return int(split["resident"] - split["draft"] + experts + config.fixed_state_bytes(kv_bytes_per_element))


def draft_step_bytes(model):
    """What one draft step of the checkpoint's own draft (MTP) layers reads, for an engine that drafts and reports no
    draft_bytes_per_step: those layers' weights (their routed experts' share as in pass_bytes) and the output head
    their logits go through, the draft layers' own head where they have one, else the model's (the embedding table
    when the head is tied to it). None without draft layers or weight headers."""
    config = model_config(model)
    shards, drafts = weight_files(model, drafts=True)
    layers = experts = draft_head = head = embedding = 0
    for shard in shards:
        for tensor in shard.tensors:
            role, size = tensor_role(tensor.name, shard.path.name, drafts), tensor.nbytes
            if role.draft:
                if role.kind == "expert":
                    experts += size
                elif role.kind == "head" or DRAFT_HEAD.search(tensor.name):
                    draft_head += size
                elif role.kind == "layer":
                    layers += size   # a draft embedding table is gathered by rows, never read whole
            elif role.kind == "head":
                head += size
            elif role.kind == "embedding":
                embedding += size
    if not layers and not experts:
        return None
    moe = config.moe
    if experts and moe and moe.experts and moe.experts_per_token:
        experts *= min(1.0, moe.experts_per_token / moe.experts)
    return int(layers + experts + (draft_head or head or (embedding if config.tie_word_embeddings else 0)))


def max_context(model, device, vram_headroom_bytes, settings, requested=None):
    """The required context of an engine for MODEL on DEVICE (a `crit eval probe` result) with the facts
    behind it. `tokens` is None when an input is missing (`missing`) or the weights read on every pass do
    not fit (`reason`). settings: protocol.json's `context` section. requested: the user's wanted context; the
    model's positions and the device's memory may cap it (`bound`, and `note` says by how much)."""
    config = model_config(model)
    limit, source = config.position_limit()
    element = float(settings["kv_bytes_per_element"])
    granularity = int(settings.get("granularity", 1))
    elements = config.kv_elements_per_token()
    kv = int(round(elements * element)) if elements else None
    out = {"tokens": None, "bound": None, "requested": int(requested) if requested else None, "note": None,
           "model_limit": limit, "model_limit_source": source, "device_limit": None,
           "kv_precision": settings.get("kv_precision"), "kv_bytes_per_token": kv, "missing": [], "reason": None}
    if limit is None:
        out["missing"].append("the model's positions (max_position_embeddings in config.json)")
    if elements is None:
        out["missing"].append("the model's attention layers (config.json)")
    try:
        split, problem = weight_split(model, config), None
    except (OSError, ValueError, KeyError) as error:
        split, problem = None, str(error)
    free = (device or {}).get("memory_free_bytes")
    if split is None:
        out["missing"].append(f"the model's weight headers ({problem})" if problem else
                              "the model's weight headers (safetensors files or a GGUF)")
    if not free:
        out["missing"].append("free device memory (re-run crit eval probe)")
    if out["missing"]:
        return out
    fixed = config.fixed_state_bytes(element)
    reserve = int(float(settings["reserve_gib"]) * GIB)
    budget = free - int(vram_headroom_bytes) - reserve - split["resident"] - fixed
    out["basis"] = {"device": (device or {}).get("device"), "memory_free_bytes": free,
                    "memory_total_bytes": (device or {}).get("memory_total_bytes"),
                    "vram_headroom_bytes": int(vram_headroom_bytes), "reserve_bytes": reserve,
                    "resident_weight_bytes": split["resident"], "draft_weight_bytes": split["draft"],
                    "expert_bytes": split["experts"], "embedding_bytes": split["embedding"],
                    "fixed_state_bytes": fixed, "kv_cache_bytes_available": max(0, budget)}
    if budget <= 0:
        out["device_limit"] = 0
        out["reason"] = (f"the weights read on every pass ({split['resident'] / 1e9:.1f} GB) and fixed state do not fit "
                         f"the device's {free / 1e9:.1f} GB free memory with {vram_headroom_bytes / 1e9:.1f} GB headroom "
                         f"and a {reserve / 1e9:.1f} GB reserve")
        return out
    if kv:
        out["device_limit"] = budget // kv // granularity * granularity
        tokens, bound = min((limit, "model"), (out["device_limit"], "device"))
    else:
        tokens, bound = limit, "model"  # no cache grows with the context
    if requested and int(requested) <= tokens:
        tokens, bound = int(requested), "requested"
    elif requested:
        out["note"] = (f"the requested {int(requested):,} tokens exceed the "
                       + (f"model's {limit:,} positions ({source})" if bound == "model" else
                          f"{out['device_limit']:,} tokens whose KV cache fits the device")
                       + f": the required context is {tokens:,}")
    out.update(tokens=tokens, bound=bound)
    # Experts get the device memory the KV cache leaves at the maximum context.
    out["basis"]["expert_bytes_in_device_memory_at_max"] = max(0, min(split["experts"], budget - tokens * (kv or 0)))
    return out


def lower_context(required, longest_case_tokens, settings):
    """The lower context the speed cases are measured again at, or None when no lower context still holds
    every speed case."""
    granularity = int(settings.get("granularity", 1))
    floor = -(-longest_case_tokens // granularity) * granularity
    lower = max(int(required * float(settings["lower_fraction"])) // granularity * granularity, floor)
    return lower if lower < required else None


def curve_points(required, settings):
    """Context lengths the long-context curve measures: powers of two from `curve_from` below the
    requirement, then the requirement itself."""
    points, point = [], int(settings["curve_from"])
    while point < required:
        points.append(point)
        point *= 2
    return points + [required]


def retrieval_points(required, rule):
    """Context lengths the retrieval check fills: its fixed points below the requirement, then the requirement
    itself (near capacity: prompt plus output equal it). rule: protocol.json's context.retrieval."""
    return sorted({int(p) for p in rule["points"] if int(p) < required} | {int(required)})


def retrieval_prompt(pieces, length, positions, tail=None):
    """A retrieval prompt of exactly `length` tokens: the prepared head, the filler (repeated as needed) with
    one fact at each of `positions` (fractions of the filler), and the tail with the question (`tail`: another
    request after the same head). None when `length` cannot hold the head, tail and facts with filler between them.
    pieces: prepared["retrieval"]."""
    head, facts = pieces["head_ids"], pieces["facts"]
    tail = pieces["tail_ids"] if tail is None else tail
    plain = length - len(head) - len(tail) - sum(len(f["ids"]) for f in facts)
    if plain < len(facts) + 1:
        return None
    filler = pieces["filler_ids"]
    filler = (list(filler) * -(-plain // len(filler)))[:plain]
    out, last = list(head), 0
    for fraction, fact in zip(positions, facts):
        cut = max(last, min(plain, int(plain * float(fraction))))
        out += filler[last:cut] + list(fact["ids"])
        last = cut
    return out + filler[last:] + list(tail)


def curve_prompt(pieces, length, positions):
    """A long-context curve prompt of exactly `length` tokens: the retrieval prompt (retrieval_prompt) whose request
    asks for the facts and then for a summary of the archive (curve_tail_ids), so its reply is long enough to time
    the decode and is judged for retrieval too. None when `length` cannot hold it. pieces: prepared["retrieval"]."""
    return retrieval_prompt(pieces, length, positions, pieces["curve_tail_ids"])


def token_text(tokens, strings, stop_ids=()):
    """The text of `tokens` up to the first stop id, joined from the prepared per-token strings; a token with
    no ASCII string (or outside the table) becomes U+FFFD, so it can never complete an ASCII answer."""
    stops, out = set(stop_ids), []
    for token in tokens:
        if token in stops:
            break
        text = strings[token] if 0 <= token < len(strings) else None
        out.append(text if text is not None else "�")
    return "".join(out)


def render(value):
    lines = []
    if value["tokens"]:
        lines.append(f"max context {value['tokens']:,} tokens ({value['bound']} bound)")
        if value.get("requested"):
            lines.append(f"  requested    {value['requested']:,} tokens (settings.toml context_tokens)")
        if value.get("note"):
            lines.append(f"  note: {value['note']}")
    else:
        lines.append("max context unknown" + (f": {value['reason']}" if value["reason"] else ""))
    if value["model_limit"]:
        lines.append(f"  model limit  {value['model_limit']:,} tokens ({value['model_limit_source']})")
    basis = value.get("basis")
    if basis:
        lines.append(f"  device limit {value['device_limit']:,} tokens: {basis['kv_cache_bytes_available'] / 1e9:.2f} GB for the KV "
                     f"cache at {value['kv_bytes_per_token']:,} bytes per token ({value['kv_precision']})"
                     if value["device_limit"] is not None and value["kv_bytes_per_token"] else
                     "  device limit none: no cache grows with the context")
        lines.append(f"    {basis['memory_free_bytes'] / 1e9:.2f} GB free on {basis['device']} - {basis['vram_headroom_bytes'] / 1e9:.2f} "
                     f"headroom - {basis['reserve_bytes'] / 1e9:.2f} reserve - {basis['resident_weight_bytes'] / 1e9:.2f} weights read "
                     f"every pass - {basis['fixed_state_bytes'] / 1e9:.3f} fixed state")
        if basis["expert_bytes"]:
            lines.append(f"    experts {basis['expert_bytes'] / 1e9:.2f} GB, "
                         f"{basis.get('expert_bytes_in_device_memory_at_max', 0) / 1e9:.2f} GB of them in device memory at the max")
    if value["missing"]:
        lines.append("missing: " + "; ".join(value["missing"]))
    return "\n".join(lines)

