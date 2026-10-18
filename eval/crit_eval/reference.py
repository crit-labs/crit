"""Reference log-probabilities: `crit eval reference --model DIR --data DIR`.

The reference runs the model's official Hugging Face `transformers` modeling code with FP32
activations and the stored weights dequantized exactly to FP32. Decoder layers are streamed
onto the GPU one at a time, so a model larger than device memory still runs. It writes, for
every scored target, the target log-probability and the reference's top-k tokens: the suite's
conversations, and the long prefill_decode prompts (protocol quality.prefill_decode). The conversations
use eager attention with an explicit causal mask; the long prompts the same FP32 arithmetic in blocks of
query rows (chunked_attention), since a 32K-token mask and score matrix would not fit.

Supported stored formats (compressed-tensors): unquantized tensors, FP8 E4M3 with per-channel
or per-tensor scales, and NVFP4 (E2M1 values, E4M3 group-16 scales and an FP32 global scale).
Requires torch, transformers and safetensors. This is the slow, trusted path, run once per
model and suite; engines are compared against its output.
"""
import json
import math
import time
from pathlib import Path

from . import load_json, write_json
from .prepare import PREPARED

REFERENCE = "reference.json"
E2M1 = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]
CHUNKED = "crit_chunked"
QUERY_BLOCK = 256


def chunked_attention(module, query, key, value, attention_mask, scaling=None, dropout=0.0, sliding_window=None,
                      **kwargs):
    """Eager causal attention computed QUERY_BLOCK query rows at a time: FP32 scores, softmax over each whole
    row, never a full length x length matrix. Registered with transformers' AttentionInterface as CHUNKED;
    attention_mask is ignored (the layer gets none), causality and any sliding window come from positions."""
    import torch
    groups = query.shape[1] // key.shape[1]
    key, value = key.repeat_interleave(groups, dim=1), value.repeat_interleave(groups, dim=1)  # transformers' repeat_kv
    scaling = query.shape[-1] ** -0.5 if scaling is None else scaling
    offset = key.shape[2] - query.shape[2]
    columns = torch.arange(key.shape[2], device=query.device)
    out = torch.empty_like(query)
    for start in range(0, query.shape[2], QUERY_BLOCK):
        stop = min(start + QUERY_BLOCK, query.shape[2])
        rows = torch.arange(start + offset, stop + offset, device=query.device)[:, None]
        width = stop + offset
        scores = torch.matmul(query[:, :, start:stop], key[:, :, :width].transpose(2, 3)) * scaling
        allowed = columns[None, :width] <= rows
        if sliding_window:
            allowed &= columns[None, :width] > rows - sliding_window
        weights = torch.softmax(scores.masked_fill(~allowed, float("-inf")), dim=-1, dtype=torch.float32).to(query.dtype)
        out[:, :, start:stop] = torch.matmul(weights, value[:, :, :width])
    return out.transpose(1, 2).contiguous(), None


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
        if self.has(base + ".weight_packed"):  # NVFP4
            packed = self.raw(base + ".weight_packed").to(device)
            table = torch.tensor(E2M1, dtype=torch.float32, device=device)
            low, high = packed & 0x0F, packed >> 4
            nibbles = torch.stack((low, high), dim=-1).reshape(packed.shape[0], -1).long()
            values = table[nibbles & 7] * torch.where(nibbles & 8 > 0, -1.0, 1.0)
            scale = self.raw(base + ".weight_scale").to(device).float()
            global_scale = self.raw(base + ".weight_global_scale").to(device).float()
            values = values.reshape(values.shape[0], scale.shape[1], -1) * (scale / global_scale)[..., None]
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
        with torch.device("meta"):
            self.skeleton = AutoModel.from_config(self.config)
        self.rotary = type(self.skeleton.rotary_emb)(config=self.config, device=device)

    def layer(self, index):
        layer = self.skeleton.layers[index]
        state = {}
        for name, _ in layer.named_parameters():
            base, leaf = name.rsplit(".", 1)
            key = f"{self.prefix}layers.{index}.{base}"
            state[name] = self.checkpoint.dense(key, self.device) if leaf == "weight" else \
                self.checkpoint.raw(f"{key}.{leaf}").to(self.device).float()
        layer.load_state_dict(state, strict=True, assign=True)
        return layer

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
        states, extras = [], []
        for s, (ids, _) in enumerate(sequences):
            hidden = embed[torch.tensor(ids)].float().to(self.device)[None]
            positions = torch.arange(len(ids), device=self.device)[None]
            cos, sin = self.rotary(hidden, positions)
            mask = None if s in long else torch.full((len(ids), len(ids)), float("-inf"), device=self.device).triu(1)[None, None]
            states.append(hidden)
            extras.append(((cos, sin), mask, positions, s in long))
        del embed
        layer_types = getattr(self.config, "layer_types", None)
        with torch.no_grad():
            for index in range(self.config.num_hidden_layers):
                started = time.time()
                layer = self.layer(index)
                full = layer_types is None or layer_types[index] != "linear_attention"
                for s, (embeddings, mask, positions, causal) in enumerate(extras):
                    if causal:
                        self._attention(layer, CHUNKED)
                        states[s] = layer(states[s], position_embeddings=embeddings, attention_mask=None,
                                          position_ids=positions)
                        self._attention(layer, "eager")
                    else:
                        states[s] = layer(states[s], position_embeddings=embeddings,
                                          attention_mask=mask if full else None, position_ids=positions)
                self.skeleton.layers[index] = torch.nn.Identity()
                del layer
                torch.cuda.empty_cache()
                log(f"layer {index + 1}/{self.config.num_hidden_layers} {time.time() - started:.1f}s")
            norm = type(self.skeleton.norm)(self.config.hidden_size, eps=self.config.rms_norm_eps).to(self.device)
            norm.weight.data = self.checkpoint.raw(self.prefix + "norm.weight").to(self.device).float()
            head = self.checkpoint.dense("lm_head", self.device)[:valid_vocab]
            results = []
            for (ids, first), hidden in zip(sequences, states):
                rows = norm(hidden[0, first - 1:len(ids) - 1])
                logprobs = torch.log_softmax((rows @ head.T).double(), dim=-1)
                targets = torch.tensor(ids[first:], device=self.device)
                target_lp = logprobs.gather(1, targets[:, None])[:, 0]
                top = logprobs.topk(top_k, dim=-1)
                results.append([{"target": int(t), "logprob": float(lp), "top_ids": [int(i) for i in ti],
                                 "top_logprobs": [float(v) for v in tv]}
                                for t, lp, ti, tv in zip(targets.tolist(), target_lp.tolist(),
                                                          top.indices.tolist(), top.values.tolist())])
        return results


def build_reference(model, data, top_k, log=print):
    import transformers
    prepared = load_json(Path(data) / PREPARED)
    started = time.time()
    reference = StreamingReference(model)
    # The suite's conversations and the long prefill_decode prompts go through the layers in one stream.
    long, short = prepared.get("prefill_decode") or [], len(prepared["quality"])
    sequences = [(c["ids"], c["first_target"]) for c in prepared["quality"] + long]
    scored = reference.score(sequences, top_k, prepared["valid_vocab"], log=log, long=set(range(short, len(sequences))))
    scored, long_scored = scored[:short], scored[short:]
    out = {
        "prepared_model_sha256": prepared["model"]["sha256"],
        "suite_sha256": prepared["suite_sha256"],
        "implementation": f"transformers {transformers.__version__} official modeling, layer-streamed; "
                          "FP32 activations, stored weights dequantized to FP32, no TF32",
        "top_k": top_k,
        "valid_vocab": prepared["valid_vocab"],
        "seconds": time.time() - started,
        "conversations": [{"id": c["id"], "targets": t} for c, t in zip(prepared["quality"], scored)],
        "prefill_decode": [{"id": c["id"], "prompt_tokens": c["prompt_tokens"], "targets": t}
                           for c, t in zip(long, long_scored)],
    }
    nll = [-r["logprob"] for t in scored for r in t]
    out["perplexity"] = math.exp(sum(nll) / len(nll))
    write_json(Path(data) / REFERENCE, out)
    return out
