"""The quality reference: `crit eval reference --model DIR --data DIR --result RESULT`.

The reference is the checkpoint the data was prepared from, the one the user chose. It runs the model's official
Hugging Face `transformers` modeling code with FP32 activations and the stored weights dequantized exactly to FP32.
Decoder layers are streamed onto the GPU one at a time, so a model larger than device memory still runs. It scores
the replies a run recorded (protocol quality) teacher-forced after the same prompt ids: for every reply token, the
token's log-probability and the reference's top-k. The conversations use eager attention with an explicit causal
mask; the long prompts the same FP32 arithmetic in blocks of query rows (chunked_attention), since a 32K-token mask
and score matrix would not fit. It needs the device to itself: stop the engine's server first.

Supported stored formats (compressed-tensors): unquantized tensors, FP8 E4M3 with per-channel
or per-tensor scales, and NVFP4 (E2M1 values, E4M3 group-16 scales and an FP32 global scale).
Requires torch, transformers and safetensors. This is the slow, trusted path, run once per
evaluation; the engine's replies are judged against its scores.
"""
import inspect
import time
from contextlib import contextmanager
from pathlib import Path

from . import EVAL_DIR, load_json, metrics, write_json
from .prepare import PREPARED, model_identity

E2M1 = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]
CHUNKED = "crit_chunked"
QUERY_BLOCK = 256


def chunked_attention(module, query, key, value, attention_mask, scaling=None, dropout=0.0, sliding_window=None,
                      **kwargs):
    """Eager causal attention computed QUERY_BLOCK query rows at a time: FP32 scores, softmax over each whole
    row, never a full length x length matrix. Registered with transformers' AttentionInterface as CHUNKED;
    attention_mask is ignored (the layer gets none), causality and any sliding window come from positions. Query heads
    share their key/value head by broadcasting (transformers' repeat_kv without the copy), and a sliding-window layer
    reads only the columns its block can see."""
    import torch
    batch, heads, length, _ = query.shape
    groups = heads // key.shape[1]
    query = query.view(batch, key.shape[1], groups, length, query.shape[-1])
    key, value = key[:, :, None], value[:, :, None]
    scaling = query.shape[-1] ** -0.5 if scaling is None else scaling
    offset = key.shape[3] - length
    columns = torch.arange(key.shape[3], device=query.device)
    out = torch.empty(batch, key.shape[1], groups, length, value.shape[-1], dtype=query.dtype, device=query.device)
    for start in range(0, length, QUERY_BLOCK):
        stop = min(start + QUERY_BLOCK, length)
        rows = torch.arange(start + offset, stop + offset, device=query.device)[:, None]
        width = stop + offset
        first = max(0, start + offset - sliding_window + 1) if sliding_window else 0
        scores = torch.matmul(query[:, :, :, start:stop], key[:, :, :, first:width].transpose(-2, -1)).mul_(scaling)
        allowed = columns[None, first:width] <= rows
        if sliding_window:
            allowed &= columns[None, first:width] > rows - sliding_window
        weights = torch.softmax(scores.masked_fill_(~allowed, float("-inf")), dim=-1, dtype=torch.float32)
        del scores
        out[:, :, :, start:stop] = torch.matmul(weights.to(query.dtype), value[:, :, :, first:width])
    return out.view(batch, heads, length, -1).transpose(1, 2).contiguous(), None


class Checkpoint:
    def __init__(self, model):
        from crit_core.model import inspect_model
        from safetensors import safe_open
        self.model = Path(model)
        self.files = {}
        self.where = {}
        # The files and headers Core validates (crit_core.model.inspect_model) before torch reads any tensor.
        for shard in inspect_model(self.model).shards:
            handle = safe_open(str(shard.path), "pt", device="cpu")
            self.files[shard.path.name] = handle
            for key in handle.keys():
                self.where[key] = handle

    def has(self, key):
        return key in self.where

    def raw(self, key):
        return self.where[key].get_tensor(key)

    def prefix(self):
        for key in self.where:
            if key.endswith("layers.0.input_layernorm.weight"):
                return key[: -len("layers.0.input_layernorm.weight")]
        raise KeyError("cannot find the decoder layer prefix in the checkpoint")

    def dense(self, base, device):
        """FP32 weight for module path `base` (without the .weight suffix)."""
        import torch
        # NVFP4: E2M1 pairs (low nibble first) with E4M3 group-16 scales and an FP32 global factor, which
        # compressed-tensors stores as a divisor (weight_packed, weight_global_scale) and ModelOpt as a
        # multiplier (weight, weight_scale_2).
        packed = None
        if self.has(base + ".weight_packed"):
            packed = self.raw(base + ".weight_packed").to(device)
            scale = self.raw(base + ".weight_scale").to(device).float()
            scale = scale / self.raw(base + ".weight_global_scale").to(device).float()
        elif self.has(base + ".weight_scale_2") and self.raw(base + ".weight").dtype == torch.uint8:
            packed = self.raw(base + ".weight").to(device)
            scale = self.raw(base + ".weight_scale").to(device).float()
            scale = scale * self.raw(base + ".weight_scale_2").to(device).float()
        if packed is not None:
            table = torch.tensor(E2M1, dtype=torch.float32, device=device)
            low, high = packed & 0x0F, packed >> 4
            nibbles = torch.stack((low, high), dim=-1).reshape(packed.shape[0], -1).long()
            values = table[nibbles & 7] * torch.where(nibbles & 8 > 0, -1.0, 1.0)
            values = values.reshape(values.shape[0], scale.shape[1], -1) * scale[..., None]
            return values.reshape(packed.shape[0], -1)
        weight = self.raw(base + ".weight").to(device)
        if weight.dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
            scale = self.raw(base + ".weight_scale").to(device).float()
            return weight.float() * (scale if scale.ndim == 2 else scale.reshape(-1)[0])
        return weight.float()


def _text_config(model):
    """transformers' own config object, which its modeling classes need; config.json must first pass Core's
    rules (crit_core.model.load_model_config), as it did when the suite was prepared."""
    from crit_core.model import load_model_config
    from transformers import AutoConfig
    load_model_config(model)
    config = AutoConfig.from_pretrained(str(model))
    text = getattr(config, "text_config", None) or config
    text._attn_implementation = "eager"
    return text


@contextmanager
def _meta_parameters():
    """Build modules with their parameters on the meta device and their buffers real: the checkpoint supplies
    every parameter, while buffers a module computes (rotary frequencies, an embedding's scale) keep their values."""
    import torch
    register = torch.nn.Module.register_parameter

    def meta(module, name, param):
        register(module, name, None if param is None else
                 torch.nn.Parameter(param.to("meta"), requires_grad=param.requires_grad))
    torch.nn.Module.register_parameter = meta
    try:
        yield
    finally:
        torch.nn.Module.register_parameter = register


class StreamingReference:
    def __init__(self, model, device="cuda"):
        import torch
        from transformers import AutoModel
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        self.torch, self.device = torch, device
        self.checkpoint = Checkpoint(model)
        self.prefix = self.checkpoint.prefix()
        self.config = _text_config(model)
        with _meta_parameters():
            self.skeleton = AutoModel.from_config(self.config)
        self.rotary = self.skeleton.rotary_emb.to(device)  # buffers only: its frequencies, computed at build
        # Models whose layer kinds rotate differently (per-layer-type RoPE parameters) take the layer type.
        self.rotary_by_type = "layer_type" in inspect.signature(self.rotary.forward).parameters

    def layer(self, index):
        """Decoder layer `index` with its parameters and the persistent buffers the checkpoint stores (such as a
        layer scalar), weights dequantized to FP32, on the device."""
        layer = self.skeleton.layers[index]
        state = {}
        for name, value in layer.state_dict(keep_vars=True).items():
            key = f"{self.prefix}layers.{index}.{name}"
            if name.rsplit(".", 1)[-1] == "weight":
                state[name] = self.checkpoint.dense(key[: -len(".weight")], self.device)
            elif self.checkpoint.has(key) or isinstance(value, self.torch.nn.Parameter):
                state[name] = self.checkpoint.raw(key).to(self.device).float()
            else:
                state[name] = value
        layer.load_state_dict(state, strict=True, assign=True)
        layer.to(self.device)
        self._config_dt_limit(layer)
        return layer

    def _embeddings(self, hidden, positions, layer_types):
        """(cos, sin) per layer type (key None when the model rotates every layer alike)."""
        if not self.rotary_by_type:
            return {None: self.rotary(hidden, positions)}
        return {kind: self.rotary(hidden, positions, kind) for kind in sorted(set(layer_types))}

    def _mask(self, length, sliding):
        """The causal mask, banded to the model's sliding window for its sliding-attention layers."""
        torch = self.torch
        rows = torch.arange(length, device=self.device)[:, None]
        columns = torch.arange(length, device=self.device)[None, :]
        allowed = columns <= rows
        window = getattr(self.config, "sliding_window", None)
        if sliding and window:
            allowed &= columns > rows - window
        return torch.zeros((length, length), device=self.device).masked_fill(~allowed, float("-inf"))[None, None]

    def _config_dt_limit(self, layer):
        """Mamba-2 mixers take the model's own dt limit. transformers sets some mixers' limit to
        (time_step_min, inf) although config.json declares time_step_limit (usually [0, inf]) and the model's own
        kernels (mamba_chunk_scan_combined) apply that; the clamp makes long-memory heads forget, which shows
        as retrieval failing beyond ~64K tokens and as drift against engines that follow the config."""
        limit = getattr(self.config, "time_step_limit", None)
        if not isinstance(limit, (list, tuple)) or len(limit) != 2:
            return
        lo, hi = (float(v) for v in limit)
        for module in layer.modules():
            current = getattr(module, "time_step_limit", None)
            if isinstance(current, tuple) and len(current) == 2 and tuple(map(float, current)) != (lo, hi):
                module.time_step_limit = (lo, hi)

    def _attention(self, layer, implementation):
        """Point every attention module of `layer` at `implementation` ("eager" or CHUNKED)."""
        for module in layer.modules():
            config = getattr(module, "config", None)
            if config is not None and hasattr(config, "_attn_implementation"):
                config._attn_implementation = implementation

    def score(self, sequences, top_k, valid_vocab=None, log=print, long=()):
        """sequences: list of (ids, first_target). Returns, per sequence, a list of target records.
        Head rows at or above valid_vocab (padding rows with no token) are excluded. The sequences whose
        indices are in `long` use chunked_attention (no mask or score matrix of the whole sequence in memory,
        which at 32K tokens would not fit); the others eager attention with an explicit causal mask."""
        torch = self.torch
        from transformers import AttentionInterface
        AttentionInterface.register(CHUNKED, chunked_attention)
        embed = self.checkpoint.raw(self.prefix + "embed_tokens.weight")
        # A scaled embedding (for example Gemma's) multiplies its rows by its own embed_scale buffer.
        scale = getattr(self.skeleton.embed_tokens, "embed_scale", None)
        layer_types = getattr(self.config, "layer_types", None) or [None] * self.config.num_hidden_layers
        states, extras = [], []
        for s, (ids, _) in enumerate(sequences):
            hidden = embed[torch.tensor(ids)].float().to(self.device)[None]
            if scale is not None:
                hidden = hidden * float(scale)
            positions = torch.arange(len(ids), device=self.device)[None]
            masks = {} if s in long else {kind: self._mask(len(ids), kind == "sliding_attention")
                                          for kind in set(layer_types) if kind != "linear_attention"}
            states.append(hidden)
            extras.append((self._embeddings(hidden, positions, layer_types), masks, positions, s in long, {}))
        del embed
        with torch.no_grad():
            for index in range(self.config.num_hidden_layers):
                started = time.time()
                layer = self.layer(index)
                kind = layer_types[index]
                # Layers that keep keys and values for later layers (KV sharing) store them per sequence.
                shared = "shared_kv_states" in inspect.signature(layer.forward).parameters
                for s, (embeddings, masks, positions, causal, kv) in enumerate(extras):
                    options = dict(position_embeddings=embeddings[kind if self.rotary_by_type else None],
                                   position_ids=positions, **({"shared_kv_states": kv} if shared else {}))
                    if causal:
                        self._attention(layer, CHUNKED)
                        states[s] = layer(states[s], attention_mask=None, **options)
                        self._attention(layer, "eager")
                    else:
                        states[s] = layer(states[s], attention_mask=masks.get(kind), **options)
                self.skeleton.layers[index] = torch.nn.Identity()
                del layer
                torch.cuda.empty_cache()
                log(f"layer {index + 1}/{self.config.num_hidden_layers} {time.time() - started:.1f}s")
            norm = type(self.skeleton.norm)(self.config.hidden_size, eps=self.config.rms_norm_eps).to(self.device)
            norm.weight.data = self.checkpoint.raw(self.prefix + "norm.weight").to(self.device).float()
            tied = not self.checkpoint.has("lm_head.weight") and getattr(self.config, "tie_word_embeddings", False)
            head = self.checkpoint.dense(self.prefix + "embed_tokens" if tied else "lm_head", self.device)[:valid_vocab]
            cap = getattr(self.config, "final_logit_softcapping", None)
            results = []
            for (ids, first), hidden in zip(sequences, states):
                rows = norm(hidden[0, first - 1:len(ids) - 1])
                logits = rows @ head.T
                if cap:
                    logits = torch.tanh(logits / cap) * cap
                logprobs = torch.log_softmax(logits.double(), dim=-1)
                targets = torch.tensor(ids[first:], device=self.device)
                target_lp = logprobs.gather(1, targets[:, None])[:, 0]
                top = logprobs.topk(top_k, dim=-1)
                results.append([{"target": int(t), "logprob": float(lp), "top_ids": [int(i) for i in ti],
                                 "top_logprobs": [float(v) for v in tv]}
                                for t, lp, ti, tv in zip(targets.tolist(), target_lp.tolist(),
                                                          top.indices.tolist(), top.values.tolist())])
        return results


def judge_sets(replies, scored, thresholds):
    """Every quality set's replies against the reference's scores of them (scored: per reply, StreamingReference.score
    rows), judged by the drift budget on its own: {"sets": {name: metrics, ...}, "passed"}."""
    groups = {}
    for reply, rows in zip(replies, scored):
        if [row["target"] for row in rows] != reply["tokens"]:
            raise ValueError(f"{reply['id']}: the reference scored other tokens than the reply")
        reference_rows, engine_rows, ids = groups.setdefault(reply["set"], ([], [], []))
        reference_rows += rows
        engine_rows += [{"logprob": lp, "top": top} for lp, top in zip(reply["logprobs"], reply["top"])]
        ids.append(reply["id"])
    sets = {}
    for name, (reference_rows, engine_rows, ids) in groups.items():
        row = metrics.reply_quality(reference_rows, engine_rows)
        failures = metrics.judge(row, thresholds)
        sets[name] = {**row, "replies": ids, "passed": not failures, "failures": failures,
                      "budget_used": metrics.budget_used(row, thresholds)}
    return {"sets": sets, "passed": bool(sets) and all(s["passed"] for s in sets.values())}


def score_result(model, data, path, baseline=None, log=print):
    """Score the replies the run at `path` recorded on the checkpoint `data` was prepared from, judge every set by
    the run's drift budget, and write the quality and the status (against `baseline`) back into the result."""
    import transformers
    from . import status
    protocol = load_json(EVAL_DIR / "protocol.json")
    prepared, result = load_json(Path(data) / PREPARED), load_json(path)
    quality = result.get("quality") or {}
    if not quality.get("replies"):
        raise SystemExit(f"{path} holds no quality replies: measure the engine with crit eval run --url and its quality part")
    if (result.get("model_sha256"), result.get("suite_sha256")) != (prepared["model"]["sha256"], prepared["suite_sha256"]):
        raise SystemExit(f"{path} was measured on other prepared data than {data}")
    if model_identity(model)["sha256"] != prepared["model"]["sha256"]:
        raise SystemExit(f"{model} is not the checkpoint {data} was prepared from: the reference is that checkpoint")
    prompts = {p["id"]: p["prompt_ids"] for p in prepared["quality"]}
    replies = quality["replies"]
    started = time.time()
    sequences = [(prompts[r["id"]] + r["tokens"], len(prompts[r["id"]])) for r in replies]
    long = {i for i, r in enumerate(replies) if r["set"] != "conversations"}
    scored = StreamingReference(model).score(sequences, quality["top_k"], prepared["valid_vocab"], log=log, long=long)
    thresholds = result["requirements"]["quality_budget"]
    quality.update(judge_sets(replies, scored, thresholds), status="scored", thresholds=thresholds, reference={
        "implementation": f"transformers {transformers.__version__} official modeling, layer-streamed; "
                          "FP32 activations, stored weights dequantized to FP32, no TF32; Mamba-2 dt limit from "
                          "config.json time_step_limit; per-layer-type RoPE and sliding-window masks, embedding "
                          "scale and final logit soft-capping as the modeling code applies them",
        "seconds": time.time() - started})
    result["status"] = status.build(result, protocol, baseline)
    write_json(path, result)
    return result
