"""A small Hugging Face checkpoint written as GGUF's reference converter writes it, for the template's GGUF tests.

The converter's steps, done here in its own direction (the template's code/gguf.py undoes them): tensors renamed by
the convention (blk.<i>.<part>, token_embd, output_norm, output); routed experts stacked [experts][rows][columns];
Llama's query and key rows permuted for its rotary layout; zero-centred norms stored as 1 + w and A_log as -exp(A_log)
(the gated delta-rule families); a gated delta net's value heads tiled where its key heads are fewer; convolution
kernels squeezed to [channels][kernel], a shared expert's gate to a vector; Qwen4Exp's indexer query and key rows split,
its n-gram table renamed and its hash constants moved to metadata. Matrices are stored Q8_0 (32 values a block, one FP16
scale), each one's values handed back so the reference computes what the file stores; the rest F32.

    path, values = write_gguf(hf_folder, out_folder, "qwen35moe")   # values: {hf name: the stored values, FP32}

The folder gets the checkpoint's config.json beside the GGUF, as a model downloaded with its base model's files has it.
A draft (mtp.*) layer is written as the block after the text layers, its own tensors as nextn parts (Qwen's
converters); Gemma 4's norms pair around each block and its experts stay one gate-up stack; DeepSeek's kv_b_proj is
split by heads into attn_k_b (transposed) and attn_v_b."""
import json
from pathlib import Path
import re
import shutil
import struct

ALIGN = 32
F32, Q8_0 = 0, 8

LAYER = {
    "input_layernorm.weight": "attn_norm.weight",
    "post_attention_layernorm.weight": "ffn_norm.weight",
    "self_attn.q_proj.weight": "attn_q.weight", "self_attn.k_proj.weight": "attn_k.weight",
    "self_attn.v_proj.weight": "attn_v.weight", "self_attn.o_proj.weight": "attn_output.weight",
    "self_attn.q_proj.bias": "attn_q.bias", "self_attn.k_proj.bias": "attn_k.bias",
    "self_attn.v_proj.bias": "attn_v.bias",
    "self_attn.q_norm.weight": "attn_q_norm.weight", "self_attn.k_norm.weight": "attn_k_norm.weight",
    "mlp.gate_proj.weight": "ffn_gate.weight", "mlp.up_proj.weight": "ffn_up.weight",
    "mlp.down_proj.weight": "ffn_down.weight", "mlp.gate.weight": "ffn_gate_inp.weight",
    "mlp.shared_expert.gate_proj.weight": "ffn_gate_shexp.weight",
    "mlp.shared_expert.up_proj.weight": "ffn_up_shexp.weight",
    "mlp.shared_expert.down_proj.weight": "ffn_down_shexp.weight",
    "mlp.shared_expert_gate.weight": "ffn_gate_inp_shexp.weight",
    "linear_attn.in_proj_qkv.weight": "attn_qkv.weight", "linear_attn.in_proj_z.weight": "attn_gate.weight",
    "linear_attn.in_proj_a.weight": "ssm_alpha.weight", "linear_attn.in_proj_b.weight": "ssm_beta.weight",
    "linear_attn.A_log": "ssm_a", "linear_attn.conv1d.weight": "ssm_conv1d.weight",
    "linear_attn.dt_bias": "ssm_dt.bias", "linear_attn.norm.weight": "ssm_norm.weight",
    "linear_attn.out_proj.weight": "ssm_out.weight",
    "attn_hyper_connection.input_mix_weight_down.weight": "hc_attn_down.weight",
    "attn_hyper_connection.input_mix_weight_up.weight": "hc_attn_up.weight",
    "attn_hyper_connection.block_inject_weight.weight": "hc_attn_inject.weight",
    "attn_hyper_connection.hc_norm.weight": "hc_attn_norm.weight",
    "mlp_hyper_connection.input_mix_weight_down.weight": "hc_ffn_down.weight",
    "mlp_hyper_connection.input_mix_weight_up.weight": "hc_ffn_up.weight",
    "mlp_hyper_connection.block_inject_weight.weight": "hc_ffn_inject.weight",
    "mlp_hyper_connection.hc_norm.weight": "hc_ffn_norm.weight",
    "ple.conv1d.weight": "ple_conv1d.weight", "ple.key_proj.weight": "ple_key.weight",
    "ple.value_proj.weight": "ple_value.weight", "ple.norm_key.weight": "ple_norm_key.weight",
    "ple.norm_query.weight": "ple_norm_query.weight", "ple.norm_conv.weight": "ple_norm_conv.weight",
    "self_attn.indexer.q_layernorm.weight": "indexer.q_norm.weight",
    "self_attn.indexer.k_layernorm.weight": "indexer.k_norm.weight",
}
OUTSIDE = {"embed_tokens.weight": "token_embd.weight", "norm.weight": "output_norm.weight",
           "hyper_connection_mixer.input_mix_weight_down.weight": "output_hc_down.weight",
           "hyper_connection_mixer.input_mix_weight_up.weight": "output_hc_up.weight",
           "hyper_connection_mixer.hc_norm.weight": "output_hc_norm.weight"}
CONSTANTS = {"layer_multipliers": "ple.layer_multipliers", "ngram_heads_offsets": "ple.head_offsets",
             "ngram_heads_vocab_sizes": "ple.head_vocab_sizes"}
ZERO_CENTRED = ("qwen35", "qwen35moe", "qwen4exp")
# Per architecture, the converter's names beside LAYER's.
ARCH_LAYER = {
    "gemma4": {"post_attention_layernorm.weight": "post_attention_norm.weight",
               "pre_feedforward_layernorm.weight": "ffn_norm.weight",
               "post_feedforward_layernorm.weight": "post_ffw_norm.weight",
               "pre_feedforward_layernorm_2.weight": "pre_ffw_norm_2.weight",
               "post_feedforward_layernorm_1.weight": "post_ffw_norm_1.weight",
               "post_feedforward_layernorm_2.weight": "post_ffw_norm_2.weight",
               "layer_scalar": "layer_output_scale.weight", "router.proj.weight": "ffn_gate_inp.weight",
               "router.scale": "ffn_gate_inp.scale", "router.per_expert_scale": "ffn_down_exps.scale"},
    "deepseek2": {"self_attn.q_a_proj.weight": "attn_q_a.weight",
                  "self_attn.q_a_layernorm.weight": "attn_q_a_norm.weight",
                  "self_attn.q_b_proj.weight": "attn_q_b.weight",
                  "self_attn.kv_a_proj_with_mqa.weight": "attn_kv_a_mqa.weight",
                  "self_attn.kv_a_layernorm.weight": "attn_kv_a_norm.weight",
                  "mlp.gate.e_score_correction_bias": "exp_probs_b.bias",
                  "mlp.shared_experts.gate_proj.weight": "ffn_gate_shexp.weight",
                  "mlp.shared_experts.up_proj.weight": "ffn_up_shexp.weight",
                  "mlp.shared_experts.down_proj.weight": "ffn_down_shexp.weight"},
}
# A draft (mtp.*) layer's own tensors -> the draft block's nextn parts (Qwen's converters: the block after the text
# layers; fc_embedding and fc_hidden joined by inputs into one eh_proj).
NEXTN = {"fc.weight": "nextn.eh_proj.weight", "pre_fc_norm_embedding.weight": "nextn.enorm.weight",
         "pre_fc_norm_hidden.weight": "nextn.hnorm.weight", "norm.weight": "nextn.shared_head_norm.weight",
         "hyper_connection_mixer.hc_norm.weight": "nextn.hc_head_norm.weight",
         "hyper_connection_mixer.input_mix_weight_down.weight": "nextn.hc_head_down.weight",
         "hyper_connection_mixer.input_mix_weight_up.weight": "nextn.hc_head_up.weight"}


def q8_0(rows):
    """FP32 [N][K] -> (Q8_0 blocks uint8 [N][K / 32 * 34], their values FP32 [N][K]), as the format quantizes."""
    import numpy as np
    w = np.asarray(rows, dtype=np.float32)
    blocks = w.reshape(w.shape[0], -1, 32)
    d = (np.abs(blocks).max(axis=-1, keepdims=True) / 127).astype(np.float16)
    scale = d.astype(np.float32)
    with np.errstate(divide="ignore", invalid="ignore"):
        q = np.where(scale == 0, 0, np.round(blocks / np.where(scale == 0, 1, scale)))
    q = np.clip(q, -127, 127).astype(np.int8)
    raw = np.concatenate([d.view(np.uint8), q.view(np.uint8)], axis=-1).reshape(w.shape[0], -1)
    return np.ascontiguousarray(raw), (scale * q.astype(np.float32)).astype(np.float32).reshape(w.shape)


def _llama_permute(n, heads):
    """The source row of each stored row: LlamaModel.permute (pairs of rotary rows interleaved per head)."""
    import numpy as np
    return np.arange(n).reshape(heads, 2, n // heads // 2).swapaxes(1, 2).reshape(n)


def _tiled(count, key_heads, unit, before=0):
    """The source index of each stored index: value heads grouped by key head (k R + r) stored tiled (r K + k)."""
    import numpy as np
    ratio = count // key_heads
    heads = np.array([k * ratio + r for r in range(ratio) for k in range(key_heads)])
    return np.concatenate([np.arange(before), before + (heads[:, None] * unit + np.arange(unit)).reshape(-1)])


class _Tensor:
    def __init__(self, name, kind, shape, data):
        self.name, self.kind, self.shape, self.data = name, kind, tuple(shape), data


def _string(text):
    data = text.encode("utf-8")
    return struct.pack("<Q", len(data)) + data


def _metadata(key, value):
    if isinstance(value, str):
        return _string(key) + struct.pack("<I", 8) + _string(value)
    if isinstance(value, (list, tuple)):
        big = any(v >= 1 << 31 for v in value)
        kind, fmt = (11, "<q") if big else (5, "<i")
        return (_string(key) + struct.pack("<IIQ", 9, kind, len(value))
                + b"".join(struct.pack(fmt, int(v)) for v in value))
    return _string(key) + struct.pack("<I", 4) + struct.pack("<I", int(value))


def _write(path, metadata, tensors):
    entries = [_metadata(k, v) for k, v in metadata.items()]
    table, payload, cursor = [], [], 0
    for t in tensors:
        cursor = -(-cursor // ALIGN) * ALIGN
        ne = tuple(reversed(t.shape))
        table.append(_string(t.name) + struct.pack("<I", len(ne)) + b"".join(struct.pack("<Q", n) for n in ne)
                     + struct.pack("<IQ", t.kind, cursor))
        payload.append((cursor, t.data))
        cursor += len(t.data)
    head = b"GGUF" + struct.pack("<IQQ", 3, len(table), len(entries)) + b"".join(entries) + b"".join(table)
    head += bytes(-len(head) % ALIGN)
    body = bytearray()
    for start, data in payload:
        body += bytes(start - len(body)) + data
    body += bytes(-len(body) % ALIGN)
    path.write_bytes(head + bytes(body))


def write_gguf(folder, out, arch, quantize=True, dense=()):
    """The safetensors checkpoint in `folder` written to out/model.gguf as GGUF's converter writes `arch`; returns
    (the GGUF's path, {hf name: the values the file stores, FP32 in the Hugging Face layout} of every quantized
    tensor). dense: Hugging Face name endings of matrices kept F32 as the file's other tensors (a GGUF stores some
    matrices so: routers, small gates)."""
    import numpy as np
    import torch
    from safetensors.torch import load_file
    folder, out = Path(folder), Path(out)
    out.mkdir(parents=True, exist_ok=True)
    config = json.loads((folder / "config.json").read_text(encoding="utf-8"))
    c = config.get("text_config") or config
    hf = {}
    for shard in sorted(folder.glob("*.safetensors")):
        hf.update(load_file(str(shard)))
    prefix = next(n[:-len("embed_tokens.weight")] for n in hf if n.endswith("embed_tokens.weight"))
    layers = c["num_hidden_layers"]
    drafts = 1 if any(n.startswith("mtp.") for n in hf) else 0
    values, tensors, metadata = {}, [], {"general.architecture": arch, f"{arch}.block_count": layers + drafts}
    if drafts:
        metadata[f"{arch}.nextn_predict_layers"] = drafts
    zero_centred = arch in ZERO_CENTRED
    layer_names = {**LAYER, **ARCH_LAYER.get(arch, {})}
    kh, vh = c.get("linear_num_key_heads"), c.get("linear_num_value_heads")
    kd, vd = c.get("linear_key_head_dim"), c.get("linear_value_head_dim")

    def stored(hf_name, w, rows=None, inputs=None):
        """A matrix's stored tensor data and kind: Q8_0 (its values to the reference) or F32, orders applied."""
        w = w.float().numpy()
        matrix = quantize and w.ndim == 2 and w.shape[1] % 32 == 0 and w.shape[0] > 1 and not hf_name.endswith(dense)
        if matrix:
            raw, back = q8_0(w)
            values[hf_name] = back
            data = raw
        else:
            data = w.astype(np.float32)
        if rows is not None:
            data = data[rows]
        if inputs is not None:
            if matrix:   # whole value heads move: their 34-byte blocks
                per = vd // 32 * 34
                blocks = data.reshape(data.shape[0], -1, per)
                data = blocks[:, inputs.reshape(-1, vd)[:, 0] // vd].reshape(data.shape[0], -1)
            else:
                data = data[:, inputs]
        return (Q8_0 if matrix else F32), np.ascontiguousarray(data)

    def add(gguf_name, kind, data, shape):
        tensors.append(_Tensor(gguf_name, kind, shape, data.tobytes()))

    experts, fc = {}, {}
    for name, w in hf.items():
        rest = name[len(prefix):] if name.startswith(prefix) else name
        if name.startswith("mtp."):
            draft = name[len("mtp."):]
            if draft.startswith("layers.0."):
                rest = f"layers.{layers}." + draft[len("layers.0."):]
            elif draft in ("fc_embedding.weight", "fc_hidden.weight"):
                fc[draft] = (name, w)
                continue
            elif draft in NEXTN:
                kind, data = stored(name, w)
                if zero_centred and NEXTN[draft].endswith("norm.weight"):
                    data = data + 1
                add(f"blk.{layers}.{NEXTN[draft]}", kind, data, w.shape)
                continue
            else:
                raise ValueError(f"the fixture writes no GGUF tensor for {name}")
        if name == "lm_head.weight":
            kind, data = stored(name, w)
            add("output.weight", kind, data, w.shape)
            continue
        if rest in OUTSIDE:
            kind, data = stored(name, w)
            if zero_centred and rest.endswith("norm.weight"):
                data = data + 1
            add(OUTSIDE[rest], kind, data, w.shape)
            continue
        found = re.fullmatch(r"layers\.(\d+)\.(.+)", rest)
        if found is None:
            raise ValueError(f"the fixture writes no GGUF tensor for {name}")
        layer, part = int(found[1]), found[2]
        at = f"blk.{layer}."
        expert = re.fullmatch(r"mlp\.experts\.(\d+)\.(gate|up|down)_proj\.weight", part)
        if expert:
            experts.setdefault((layer, expert[2]), {})[int(expert[1])] = (name, w)
            continue
        if part in ("mlp.experts.gate_up_proj", "mlp.experts.down_proj", "experts.gate_up_proj", "experts.down_proj"):
            if arch == "gemma4":   # stored [experts][2 * inter][hidden] and [experts][hidden][inter]
                stack = w
            else:
                stack = w if part.endswith("down_proj") or w.shape[1] != c["hidden_size"] else w.transpose(1, 2)
                if part.endswith("down_proj") and stack.shape[1] != c["hidden_size"]:
                    stack = stack.transpose(1, 2)
            if part.endswith("gate_up_proj"):
                inter = stack.shape[1] // 2
                for e in range(stack.shape[0]):
                    experts.setdefault((layer, "gate"), {})[e] = (f"{name}:{e}:gate", stack[e, :inter])
                    experts.setdefault((layer, "up"), {})[e] = (f"{name}:{e}:up", stack[e, inter:])
            else:
                for e in range(stack.shape[0]):
                    experts.setdefault((layer, "down"), {})[e] = (f"{name}:{e}", stack[e])
            continue
        constant = part.removeprefix("ple.ple_embedding.")
        if constant in CONSTANTS:
            metadata[f"{arch}.{CONSTANTS[constant]}"] = [int(v) for v in w.tolist()]
            continue
        if part.startswith("ple.ple_embedding.ngram_embedding."):
            continue   # the table: written after the loop, its shards joined
        if part == "self_attn.indexer.index_qk_proj.weight":
            q_rows = c["indexer_n_heads"] * c["indexer_head_dim"]
            kind, data = stored(name, w)
            add(at + "indexer.q_proj.weight", kind, data[:q_rows], (q_rows, w.shape[1]))
            add(at + "indexer.k_proj.weight", kind, data[q_rows:], (w.shape[0] - q_rows, w.shape[1]))
            continue
        if arch == "deepseek2" and part == "self_attn.kv_b_proj.weight":
            heads, nope, v = c["num_attention_heads"], c["qk_nope_head_dim"], c["v_head_dim"]
            kv = w.float().reshape(heads, nope + v, -1)
            key, value = kv[:, :nope].transpose(1, 2).contiguous(), kv[:, nope:].contiguous()
            back = []
            for gguf_part, t in (("attn_k_b", key), ("attn_v_b", value)):
                raw, got = q8_0(t.reshape(-1, t.shape[-1]).numpy())
                add(at + gguf_part + ".weight", Q8_0, raw, t.shape)
                back.append(torch.from_numpy(got).reshape(t.shape))
            values[name] = torch.cat([back[0].transpose(1, 2), back[1]], 1).reshape(w.shape).numpy()
            continue
        if part not in layer_names:
            raise ValueError(f"the fixture writes no GGUF tensor for {name}")
        rows = inputs = None
        if arch == "llama" and part in ("self_attn.q_proj.weight", "self_attn.k_proj.weight"):
            heads = c["num_attention_heads"] if ".q_proj" in part else c.get("num_key_value_heads",
                                                                              c["num_attention_heads"])
            rows = _llama_permute(w.shape[0], heads)
        if kh and vh and kh != vh and part.startswith("linear_attn."):
            if part in ("linear_attn.in_proj_qkv.weight", "linear_attn.conv1d.weight"):
                rows = _tiled(vh, kh, vd, before=2 * kh * kd)
            elif part == "linear_attn.in_proj_z.weight":
                rows = _tiled(vh, kh, vd)
            elif part in ("linear_attn.in_proj_a.weight", "linear_attn.in_proj_b.weight", "linear_attn.A_log",
                          "linear_attn.dt_bias"):
                rows = _tiled(vh, kh, 1)
            elif part == "linear_attn.out_proj.weight":
                inputs = _tiled(vh, kh, vd)
        shape = list(w.shape)
        if part.endswith("conv1d.weight"):
            w, shape = w.reshape(w.shape[0], w.shape[-1]), [w.shape[0], w.shape[-1]]
        if part == "mlp.shared_expert_gate.weight":
            w, shape = w.reshape(-1), [w.numel()]
        if w.dim() == 0:
            w, shape = w.reshape(1), [1]
        kind, data = stored(name, w, rows, inputs)
        if zero_centred and (part.endswith("norm.weight") and part != "linear_attn.norm.weight"
                             or part in ("ple.norm_key.weight", "ple.norm_query.weight", "ple.norm_conv.weight")):
            data = data + 1
        if part == "linear_attn.A_log":
            data = -np.exp(data)
        add(at + layer_names[part], kind, data.astype(np.float32) if kind == F32 else data, shape)
    if fc:   # the draft layer's input halves, joined by inputs as the converter writes eh_proj
        (e_name, e), (h_name, h) = fc["fc_embedding.weight"], fc["fc_hidden.weight"]
        kind, data = stored("mtp.fc", torch.cat([e, h], 1))
        if "mtp.fc" in values:
            back = values.pop("mtp.fc")
            values[e_name], values[h_name] = back[:, :e.shape[1]], back[:, e.shape[1]:]
        add(f"blk.{layers}.nextn.eh_proj.weight", kind, data, (e.shape[0], e.shape[1] + h.shape[1]))
    stacks = {}
    for (layer, which), found in sorted(experts.items()):
        parts = [stored(*found[e]) for e in sorted(found)]
        stacks[layer, which] = (parts[0][0], np.stack([p[1] for p in parts]), tuple(found[0][1].shape))
    for (layer, which), (kind, data, (rows, columns)) in sorted(stacks.items()):
        if arch == "gemma4" and which in ("gate", "up"):   # one gate-up stack, the gate's rows first
            if which == "up":
                continue
            data, rows, which = np.concatenate([data, stacks[layer, "up"][1]], axis=1), 2 * rows, "gate_up"
        add(f"blk.{layer}.ffn_{which}_exps.weight", kind, data, (data.shape[0], rows, columns))
    table = [n for n in hf if ".ple.ple_embedding.ngram_embedding." in n and n.endswith(".weight")]
    if table:
        def shard(n):
            got = re.search(r"shard_(\d+)\.weight$", n)
            return int(got[1]) if got else -1
        rows = torch.cat([hf[n].float() for n in sorted(table, key=shard)])
        add("per_layer_token_embd.weight", F32, rows.numpy().astype(np.float32), tuple(rows.shape))
    if arch == "qwen4exp":
        layers = [int(i) - 1 for i in c.get("ple_layer_ids") or ()]
        metadata[f"{arch}.ple.layers"] = layers
    path = out / "model.gguf"
    _write(path, metadata, tensors)
    shutil.copy(folder / "config.json", out / "config.json")
    for name in ("generation_config.json",):
        if (folder / name).is_file():
            shutil.copy(folder / name, out / name)
    return path, values


def stored_reference(model, values):
    """The reference model's tensors set to the values the GGUF stores (its quantized matrices), so it computes
    what the file holds. A fused experts tensor gets each expert's part back."""
    import torch
    params = dict(model.named_parameters())
    with torch.no_grad():
        for name, value in values.items():
            if ":" in name:   # a fused stack's expert part: name:expert[:gate|up]
                base, *where = name.split(":")
                p = params[base]
                e = int(where[0])
                v = torch.from_numpy(value).to(p.dtype)
                hidden_last = p.shape[-1] != v.shape[-1]
                target = p[e].t() if hidden_last else p[e]
                if len(where) == 2:
                    inter = target.shape[0] // 2
                    target = target[:inter] if where[1] == "gate" else target[inter:]
                target.copy_(v)
                continue
            if name in params:
                params[name].copy_(torch.from_numpy(value).to(params[name].dtype))
