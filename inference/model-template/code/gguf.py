"""A GGUF checkpoint read as the composed model reads a safetensors one: its tensors under the Hugging Face names the
blocks read, its quantized matrices as stored (format gguf:<TYPE>, the rows the file holds, for gguf-projections and an
experts plugin that takes GGUF experts), its other tensors decoded on the host by Core's decoders.

GGUF's reference converter renames every tensor of the Hugging Face checkpoint by one convention (blk.<layer>.<part>,
token_embd, output_norm, output) and, for some architectures, reorders or transforms a tensor on the way. Both are
undone here from the file's own metadata (general.architecture, the config Core derives from it), never from what one
model's checkpoint holds:

    names       a table of the convention's parts for every layer (attn_q -> self_attn.q_proj, ffn_up_exps -> the
                routed experts' up_proj, ssm_out -> linear_attn.out_proj, hc_attn_down -> the attention's
                hyper-connection input_mix_weight_down, ...) and of the tensors outside the layers, the architecture's
                parts beside the common ones (ARCHITECTURES)
    stacks      a routed layer's experts, stored as one [experts][rows][row bytes] stack a matrix, read as each
                expert's gate_proj, up_proj and down_proj (a fused gate-up stack split by rows)
    merges      tensors the converter writes as one whose rows are two (attn_qkv_gate, ssm_beta_alpha,
                hc_*_down_inject) read as their halves, and two it splits (the indexer's query and key rows) read as
                one
    transforms  the converter's elementwise steps inverted in FP32, one rounding: zero-centred norms stored as 1 + w
                read as w, A_log stored as -exp(A_log) read as log(-a); squeezed convolution kernels given their
                [channels][1][kernel] shape back; a vector stored for a [1][width] matrix given its row
    orders      moves of whole rows (or, along a matrix's inputs, of whole blocks of the stored format), so they apply
                to quantized matrices without decoding them: Llama's query and key rows (permuted for its rotary
                layout), a gated delta net's value heads (tiled by the converter, grouped in the Hugging Face layout)
                where its key heads are fewer
    constants   the n-gram hash constants a converter moves to metadata, read back as int64 tensors
    latent      DeepSeek's kv_b_proj, which the converter splits by heads into attn_k_b (each head's key rows
                transposed) and attn_v_b, read joined back and decoded (latent attention reads it as a plain matrix)
    drafts      the first draft (nextn) block after the text layers read as the draft layer under mtp. (its decoder
                layer's parts as any layer's, eh_proj as the MTP layer's fc or, where the model takes them, its two
                input halves fc_embedding and fc_hidden as whole blocks of their columns, enorm, hnorm and
                shared_head_norm as its norms); a draft block's own embedding and head copies are not read (the
                template drafts with the model's own)

A tensor of the file that maps to no name is refused (so a converter's new part fails here, not later as a wrong
answer), the file's rotary frequency table (read by the config) excepted.

    ck = GgufCheckpoint(path, config)         # path: the .gguf (a split set's first part) or its folder
    ck.get("model.layers.0.input_layernorm.weight", "cuda")
    matrices = GgufMatrices(ck, projections, None, "cuda")
    matrices.format("model.layers.0.self_attn.q_proj")            # "gguf:Q6_K", or "bf16" for a dense tensor
    matrices.load(["model.layers.0.self_attn.q_proj", ...], "qkv")
"""
from dataclasses import dataclass, field
from pathlib import Path
import re

PREFIX = "model."
DENSE = ("F32", "F16", "BF16")
BLOCK = re.compile(r"blk\.(\d+)\.(.+?)(\.weight|\.bias)?$")

# The convention's parts of a layer -> the Hugging Face name under model.layers.<i>. (a ".weight" or ".bias" suffix
# carries over; a part stored without one keeps its own name).
COMMON = {
    "attn_norm": "input_layernorm", "ffn_norm": "post_attention_layernorm",
    "post_attention_norm": "post_attention_layernorm",
    "attn_q": "self_attn.q_proj", "attn_k": "self_attn.k_proj", "attn_v": "self_attn.v_proj",
    "attn_output": "self_attn.o_proj", "attn_q_norm": "self_attn.q_norm", "attn_k_norm": "self_attn.k_norm",
    "ffn_gate": "mlp.gate_proj", "ffn_up": "mlp.up_proj", "ffn_down": "mlp.down_proj",
    "ffn_gate_inp": "mlp.gate",
    "ffn_gate_shexp": "mlp.shared_expert.gate_proj", "ffn_up_shexp": "mlp.shared_expert.up_proj",
    "ffn_down_shexp": "mlp.shared_expert.down_proj", "ffn_gate_inp_shexp": "mlp.shared_expert_gate",
}
GATED_DELTA = {
    "attn_qkv": "linear_attn.in_proj_qkv", "attn_gate": "linear_attn.in_proj_z",
    "ssm_alpha": "linear_attn.in_proj_a", "ssm_beta": "linear_attn.in_proj_b", "ssm_a": "linear_attn.A_log",
    "ssm_conv1d": "linear_attn.conv1d", "ssm_dt.bias": "linear_attn.dt_bias", "ssm_dt": "linear_attn.dt_bias",
    "ssm_norm": "linear_attn.norm", "ssm_out": "linear_attn.out_proj",
}
STREAMS = {
    "hc_attn_down": "attn_hyper_connection.input_mix_weight_down",
    "hc_attn_up": "attn_hyper_connection.input_mix_weight_up",
    "hc_attn_inject": "attn_hyper_connection.block_inject_weight", "hc_attn_norm": "attn_hyper_connection.hc_norm",
    "hc_ffn_down": "mlp_hyper_connection.input_mix_weight_down",
    "hc_ffn_up": "mlp_hyper_connection.input_mix_weight_up",
    "hc_ffn_inject": "mlp_hyper_connection.block_inject_weight", "hc_ffn_norm": "mlp_hyper_connection.hc_norm",
    "ple_conv1d": "ple.conv1d", "ple_key": "ple.key_proj", "ple_value": "ple.value_proj",
    "ple_norm_key": "ple.norm_key", "ple_norm_query": "ple.norm_query", "ple_norm_conv": "ple.norm_conv",
    "indexer.q_norm": "self_attn.indexer.q_layernorm", "indexer.k_norm": "self_attn.indexer.k_layernorm",
}
# Tensors outside the layers -> their Hugging Face names (lm_head outside the model's prefix).
OUTSIDE = {"token_embd.weight": PREFIX + "embed_tokens.weight", "output_norm.weight": PREFIX + "norm.weight",
           "output.weight": "lm_head.weight"}
OUTSIDE_STREAMS = {"output_hc_down.weight": PREFIX + "hyper_connection_mixer.input_mix_weight_down.weight",
                   "output_hc_up.weight": PREFIX + "hyper_connection_mixer.input_mix_weight_up.weight",
                   "output_hc_norm.weight": PREFIX + "hyper_connection_mixer.hc_norm.weight"}
# Read by the config rule (Llama 3's rotary scaling), not by the model.
IGNORED = ("rope_freqs.weight",)
# Merged tensors: GGUF part -> the halves' parts (the first half's rows from the config key, or "split": the config
# key's value is the first half's rows).
MERGES = {"attn_qkv_gate": ("attn_qkv", "attn_gate", "linear_conv_rows"),
          "ssm_beta_alpha": ("ssm_beta", "ssm_alpha", "linear_num_value_heads"),
          "hc_attn_down_inject": ("hc_attn_down", "hc_attn_inject", "hc_lowrank"),
          "hc_ffn_down_inject": ("hc_ffn_down", "hc_ffn_inject", "hc_lowrank")}
EXPERT_STACKS = {"ffn_gate_exps": "gate_proj", "ffn_up_exps": "up_proj", "ffn_down_exps": "down_proj"}
PLE_CONSTANTS = {"layer_multipliers": ".ple.layer_multipliers", "ngram_heads_offsets": ".ple.head_offsets",
                 "ngram_heads_vocab_sizes": ".ple.head_vocab_sizes"}
# Gemma 4: each block between two norms (attn_norm / post_attention_norm, ffn_norm / post_ffw_norm), a scalar a layer
# (layer_scalar, stored with a .weight suffix), and beside the MLP a mixture of experts at the layer's own prefix: its
# router (proj, scale), each expert's scale stored as the down stack's .scale, and three norms of its own.
GEMMA4 = {
    "ffn_norm": "pre_feedforward_layernorm", "post_ffw_norm": "post_feedforward_layernorm",
    "pre_ffw_norm_2": "pre_feedforward_layernorm_2", "post_ffw_norm_1": "post_feedforward_layernorm_1",
    "post_ffw_norm_2": "post_feedforward_layernorm_2", "layer_output_scale.weight": "layer_scalar",
    "ffn_gate_inp": "router.proj", "ffn_gate_inp.scale": "router.scale",
    "ffn_down_exps.scale": "router.per_expert_scale",
}
# DeepSeek V2/V3 (general.architecture deepseek2): latent attention's low-rank matrices and norms, the shared experts
# (plural), the router's score correction stored as exp_probs_b.bias. The converter splits kv_b_proj into attn_k_b
# (each head's key rows, transposed) and attn_v_b; they are read joined back (an older file's attn_kv_b as is).
LATENT = {
    "attn_q_a": "self_attn.q_a_proj", "attn_q_a_norm": "self_attn.q_a_layernorm", "attn_q_b": "self_attn.q_b_proj",
    "attn_kv_a_mqa": "self_attn.kv_a_proj_with_mqa", "attn_kv_a_norm": "self_attn.kv_a_layernorm",
    "attn_kv_b": "self_attn.kv_b_proj",
    "ffn_gate_shexp": "mlp.shared_experts.gate_proj", "ffn_up_shexp": "mlp.shared_experts.up_proj",
    "ffn_down_shexp": "mlp.shared_experts.down_proj", "exp_probs_b.bias": "mlp.gate.e_score_correction_bias",
}
# A draft (nextn) block's own parts beside a decoder layer's -> the draft layer's Hugging Face names under DRAFT (the
# layout's draft prefix; Qwen's converters move mtp.* to the block after the text layers). eh_proj is the MTP layer's
# fc over [embedding ; hidden], read as one matrix or, where the model takes the halves (fc_embedding, fc_hidden: the
# converter joined them by inputs), as its two halves of input columns.
DRAFT = "mtp."
NEXTN = {"nextn.enorm": "pre_fc_norm_embedding", "nextn.hnorm": "pre_fc_norm_hidden",
         "nextn.shared_head_norm": "norm", "nextn.hc_head_norm": "hyper_connection_mixer.hc_norm",
         "nextn.hc_head_down": "hyper_connection_mixer.input_mix_weight_down",
         "nextn.hc_head_up": "hyper_connection_mixer.input_mix_weight_up"}
# A draft block's own copies of the embedding and the head (DeepSeek's): the template drafts with the model's own.
NEXTN_SHARED = ("nextn.embed_tokens", "nextn.shared_head_head")


@dataclass(frozen=True)
class Rules:
    parts: dict                     # the layers' parts beside COMMON
    outside: dict = field(default_factory=dict)
    zero_centred: bool = False      # norms stored as 1 + w (every *norm.weight but the delta net's own, PLE's norms)
    llama_rotary: bool = False      # query and key rows permuted for the rotary layout
    gated_delta: bool = False       # A_log stored -exp, value heads tiled where the key heads are fewer
    streams: bool = False           # the n-gram table, its constants and the indexer's split query and key
    experts: str = "mlp.experts."   # where a layer's routed experts are, under the layer
    latent: bool = False            # kv_b_proj stored split by heads into attn_k_b (transposed) and attn_v_b
    split_fc: bool = False          # a draft layer's fc read as fc_embedding and fc_hidden (its input halves)


# general.architecture -> the converter's rules for it.
ARCHITECTURES = {
    "llama": Rules({}, llama_rotary=True),
    "qwen2": Rules({}), "qwen3": Rules({}), "qwen2moe": Rules({}), "qwen3moe": Rules({}),
    "qwen35": Rules(GATED_DELTA, zero_centred=True, gated_delta=True),
    "qwen35moe": Rules(GATED_DELTA, zero_centred=True, gated_delta=True),
    "qwen4exp": Rules({**GATED_DELTA, **STREAMS}, OUTSIDE_STREAMS, zero_centred=True, gated_delta=True, streams=True,
                      split_fc=True),
    "gemma4": Rules(GEMMA4, experts="experts."),
    "deepseek2": Rules(LATENT, latent=True),
}


@dataclass
class Entry:
    """One Hugging Face tensor: where its values are and what turns them into it."""
    shape: tuple
    sources: tuple                  # GGUF TensorInfos (two for a joined tensor), () for a constant
    dtype: str                      # the source's ggml type ("I64" for a constant)
    rows: object = None             # a slice of the source's rows (a merge's half), or an expert index
    expert: int = None
    order: tuple = None             # (index of output rows from source rows, or of input blocks, "rows" | "inputs")
    transform: str = None           # "minus_one", "log_neg"
    values: tuple = None            # a constant's values
    columns: slice = None           # a slice of the source's input columns (a draft fc's half), whole blocks
    latent: tuple = None            # (heads, key rows a head): kv_b_proj joined from attn_k_b and attn_v_b, decoded

    @property
    def stored(self):
        return self.dtype not in DENSE + ("I64",) and self.latent is None

    @property
    def nbytes(self):
        if not self.sources:
            return 8 * len(self.values)
        total = sum(s.nbytes for s in self.sources)
        if self.expert is not None:
            total //= self.sources[0].shape[0]
        if isinstance(self.rows, slice):
            n = self.sources[0].shape[-2] if len(self.sources[0].shape) > 1 else self.sources[0].shape[0]
            total = total * (self.rows.stop - self.rows.start) // n
        if self.columns is not None:
            total = total * (self.columns.stop - self.columns.start) // self.sources[0].shape[-1]
        return total

    # The safetensors header's fields the template reads of a table (its rows read in place from the file).
    @property
    def shard(self):
        return self.sources[0].shard

    @property
    def data_begin(self):
        return self.sources[0].data_begin


def _cfg(config):
    return (config.get("text_config") or config) if isinstance(config, dict) else {}


def _llama_rows(n, heads):
    """The source row of each Hugging Face query or key row: the converter interleaved each head's rotary pairs."""
    import numpy as np
    per = n // heads
    return np.arange(n).reshape(heads, per // 2, 2).transpose(0, 2, 1).reshape(n)


def _grouped_heads(count, key_heads, unit, before=0):
    """The source index of each Hugging Face index along a dimension whose last `count` x `unit` entries (after
    `before` others) are value heads the converter tiled (head r K + k) and the Hugging Face layout groups by key
    head (k R + r)."""
    import numpy as np
    ratio = count // key_heads
    heads = np.arange(count).reshape(ratio, key_heads).T.reshape(count)   # grouped position -> tiled head
    within = (heads[:, None] * unit + np.arange(unit)).reshape(-1)
    return np.concatenate([np.arange(before), before + within])


class GgufCheckpoint:
    """A GGUF model's tensors under Hugging Face names (the template's Checkpoint interface): `name in ck`, names(),
    nbytes(), shape(), get(name, device), headers[name] (shape, dtype, nbytes, shard, data_begin), close()."""

    def __init__(self, path, config):
        from crit_core.model.gguf import inspect_gguf_model
        from crit_core.model.gguf_blocks import GgufTensors
        self.folder = Path(path)
        self.model = inspect_gguf_model(path)
        self.arch = self.model.architecture
        self.rules = ARCHITECTURES.get(self.arch)
        if self.rules is None:
            raise ValueError(f"model-template reads GGUF architectures {', '.join(sorted(ARCHITECTURES))}; this "
                             f"file's general.architecture is {self.arch!r}")
        self.config = _cfg(config)
        self.reader = GgufTensors(self.model)
        self.headers = {}
        self._map()

    # -- the names -------------------------------------------------------------------------------------------------
    def _setting(self, key):
        value = self.config.get(key)
        if value is None:
            raise ValueError(f"model-template: reading this GGUF's {self.arch} tensors takes the config's {key}")
        return int(value)

    def _add(self, name, entry):
        if name in self.headers:
            raise ValueError(f"model-template: two tensors of the GGUF map to {name}")
        self.headers[name] = entry

    def _map(self):
        rules, tensors = self.rules, self.model.tensors
        parts = {**COMMON, **rules.parts}
        layers = int(self.config.get("num_hidden_layers") or self.model.get(".block_count") or 0)
        unmapped = []
        ple = [int(i) - 1 for i in self.config.get("ple_layer_ids") or ()]
        for gguf_name, info in tensors.items():
            if gguf_name in IGNORED:
                continue
            outside = {**OUTSIDE, **rules.outside}
            if gguf_name in outside:
                self._add(outside[gguf_name], self._entry(outside[gguf_name], info))
                continue
            if rules.streams and gguf_name == "per_layer_token_embd.weight":
                if len(ple) != 1:
                    raise ValueError(f"model-template: the GGUF's n-gram table serves one layer; the config names "
                                     f"{ple or 'none'}")
                self._add(f"{PREFIX}layers.{ple[0]}.ple.ple_embedding.ngram_embedding.weight", self._entry("", info))
                continue
            found = BLOCK.fullmatch(gguf_name)
            if found is None:
                unmapped.append(gguf_name)
                continue
            layer, part, suffix = int(found[1]), found[2], found[3] or ""
            at = f"{PREFIX}layers.{layer}."
            if layer >= layers:   # a draft (nextn) block: the draft layer's names (the first draft block's alone)
                if layer > layers or part in NEXTN_SHARED:
                    continue
                at = f"{DRAFT}layers.0."
                if part in NEXTN:
                    name = DRAFT + NEXTN[part] + suffix
                    self._add(name, self._entry(name, info))
                    continue
                if part == "nextn.eh_proj":
                    self._fc(info, suffix)
                    continue
            if part in EXPERT_STACKS:
                self._experts(at, EXPERT_STACKS[part], info)
            elif part == "ffn_gate_up_exps":
                half = info.shape[1] // 2
                self._experts(at, "gate_proj", info, slice(0, half))
                self._experts(at, "up_proj", info, slice(half, 2 * half))
            elif part in MERGES:
                first, second, key = MERGES[part]
                rows = (2 * self._setting("linear_num_key_heads") * self._setting("linear_key_head_dim")
                        + self._setting("linear_num_value_heads") * self._setting("linear_value_head_dim")
                        if key == "linear_conv_rows" else self._setting(key))
                for half, cut in ((first, slice(0, rows)), (second, slice(rows, info.shape[0]))):
                    name = at + parts[half] + suffix
                    self._add(name, self._entry(name, info, rows=cut))
            elif rules.latent and part in ("attn_k_b", "attn_v_b"):
                if part == "attn_k_b":
                    value = tensors.get(f"blk.{layer}.attn_v_b{suffix}")
                    if value is None:
                        raise ValueError(f"model-template: blk.{layer} has attn_k_b without attn_v_b")
                    self._latent(at + "self_attn.kv_b_proj" + suffix, info, value)
            elif rules.streams and part in ("indexer.q_proj", "indexer.k_proj"):
                if part == "indexer.q_proj":
                    key = tensors.get(f"blk.{layer}.indexer.k_proj{suffix}")
                    if key is None:
                        raise ValueError(f"model-template: blk.{layer} has the indexer's query rows without its key")
                    name = at + "self_attn.indexer.index_qk_proj" + suffix
                    self._add(name, self._entry(name, info, joined=key))
            elif part + suffix in parts:   # a part stored with its own name (ssm_dt.bias)
                self._add(at + parts[part + suffix], self._entry(at + parts[part + suffix], info))
            elif part in parts:
                name = at + parts[part] + suffix
                self._add(name, self._entry(name, info))
            else:
                unmapped.append(gguf_name)
        if rules.streams and ple:
            for hf, key in PLE_CONSTANTS.items():
                values = self.model.get(key)
                if isinstance(values, tuple):
                    self._add(f"{PREFIX}layers.{ple[0]}.ple.ple_embedding.{hf}",
                              Entry((len(values),), (), "I64", values=tuple(int(v) for v in values)))
        if unmapped:
            raise ValueError(f"model-template: {len(unmapped)} tensors of this {self.arch} GGUF map to no name the "
                             f"template reads: {', '.join(sorted(unmapped)[:8])}")

    def _experts(self, at, part, info, rows=None):
        for e in range(info.shape[0]):
            shape = info.shape[1:] if rows is None else (rows.stop - rows.start, info.shape[2])
            self._add(f"{at}{self.rules.experts}{e}.{part}.weight",
                      Entry(tuple(shape), (info,), info.dtype, rows=rows, expert=e))

    def _fc(self, info, suffix):
        """A draft block's eh_proj [hidden][2 * hidden] (inputs: the normed embedding, then the normed hidden row): the
        MTP layer's fc, or where the rules split it its two input halves."""
        if not self.rules.split_fc:
            self._add(DRAFT + "fc" + suffix, self._entry(DRAFT + "fc" + suffix, info))
            return
        half = info.shape[-1] // 2
        for name, cut in (("fc_embedding", slice(0, half)), ("fc_hidden", slice(half, 2 * half))):
            entry = Entry((info.shape[0], half), (info,), info.dtype, columns=cut)
            if entry.stored:
                from crit_core.model.gguf_blocks import BLOCKS, TYPES
                if half % BLOCKS[TYPES[info.dtype]][0]:
                    raise ValueError(f"model-template: {info.name} is {info.dtype}, whose blocks do not split at its "
                                     f"{half}-wide halves")
            self._add(DRAFT + name + suffix, entry)

    def _latent(self, name, key, value):
        """kv_b_proj [heads * (nope + v)][rank] from attn_k_b [heads][rank][nope] (each head's key rows transposed)
        and attn_v_b [heads][v][rank]: decoded, read as a plain matrix (latent attention multiplies its per-head rows
        so)."""
        heads, rank, nope = key.shape
        if tuple(value.shape[::2]) != (heads, rank):
            raise ValueError(f"model-template: {key.name} {list(key.shape)} and {value.name} {list(value.shape)} are "
                             "not one latent's key and value rows")
        self._add(name, Entry((heads * (nope + value.shape[1]), rank), (key, value), key.dtype, latent=(heads, nope)))

    def _entry(self, name, info, rows=None, joined=None):
        """The entry of HF tensor `name` from GGUF `info` (and `joined`, rows after it)."""
        shape = list(info.shape)
        if rows is not None:
            shape[0] = rows.stop - rows.start
        sources = (info,) if joined is None else (info, joined)
        if joined is not None:
            shape[0] += joined.shape[0]
        entry = Entry(tuple(shape), sources, info.dtype, rows=rows)
        rules, c = self.rules, self.config
        if name.endswith("conv1d.weight") and len(shape) == 2:      # squeezed [channels][1][kernel]
            entry.shape = (shape[0], 1, shape[1])
        if name.endswith("shared_expert_gate.weight") and len(shape) == 1:
            entry.shape = (1, shape[0])
        if rules.zero_centred and (name.endswith("norm.weight") and not name.endswith("linear_attn.norm.weight")
                                   or name.endswith((".ple.norm_key.weight", ".ple.norm_query.weight",
                                                     ".ple.norm_conv.weight", ".pre_fc_norm_embedding.weight",
                                                     ".pre_fc_norm_hidden.weight"))):
            entry.transform = "minus_one"
        if rules.gated_delta and name.endswith("linear_attn.A_log"):
            entry.transform = "log_neg"
        if rules.llama_rotary and name.endswith(("self_attn.q_proj.weight", "self_attn.q_proj.bias",
                                                 "self_attn.k_proj.weight", "self_attn.k_proj.bias")):
            heads = self._setting("num_attention_heads")
            if ".k_proj." in name:
                heads = int(c.get("num_key_value_heads") or heads)
            entry.order = (_llama_rows(entry.shape[0], heads), "rows")
        if rules.gated_delta and ".linear_attn." in name:
            entry.order = self._value_heads(name, entry)
        return entry

    def _value_heads(self, name, entry):
        """The tiled value heads' order where the key heads are fewer than the value heads, else None."""
        k, v = int(self.config.get("linear_num_key_heads") or 0), int(self.config.get("linear_num_value_heads") or 0)
        if not k or not v or k == v:
            return None
        kd, vd = self._setting("linear_key_head_dim"), self._setting("linear_value_head_dim")
        if name.endswith((".in_proj_qkv.weight", ".conv1d.weight")):
            return _grouped_heads(v, k, vd, before=2 * k * kd), "rows"
        if name.endswith(".in_proj_z.weight"):
            return _grouped_heads(v, k, vd), "rows"
        if name.endswith((".in_proj_a.weight", ".in_proj_b.weight", ".A_log", ".dt_bias")):
            return _grouped_heads(v, k, 1), "rows"
        if name.endswith(".out_proj.weight"):
            return _grouped_heads(v, k, vd), "inputs"
        return None

    # -- the Checkpoint interface ---------------------------------------------------------------------------------
    def __contains__(self, name):
        return name in self.headers

    def names(self):
        return self.headers.keys()

    def nbytes(self, name):
        return self.headers[name].nbytes

    def shape(self, name):
        return tuple(self.headers[name].shape)

    def rows(self, name):
        """The stored rows of a quantized tensor, uint8 numpy [N][row bytes] (an expert's, a merge's half, the
        orders applied): what gguf-projections loads."""
        import numpy as np
        e = self.headers[name]
        parts = []
        for info in e.sources:
            raw = self.reader.rows(info.name)
            if e.expert is not None:
                raw = raw[e.expert]
            if isinstance(e.rows, slice):
                raw = raw[e.rows]
            if e.columns is not None:   # whole blocks of the stored format
                from crit_core.model.gguf_blocks import BLOCKS, TYPES
                values, size = BLOCKS[TYPES[e.dtype]]
                raw = raw[:, e.columns.start // values * size:e.columns.stop // values * size]
            parts.append(raw)
        raw = parts[0] if len(parts) == 1 else np.concatenate(parts)
        if e.order is not None:
            index, along = e.order
            if along == "rows":
                raw = raw[index]
            else:
                raw = self._move_inputs(name, raw, index)
        raw = np.ascontiguousarray(raw)
        return raw if raw.flags.writeable else raw.copy()   # the file's mapping is read-only; a load takes a copy

    def _move_inputs(self, name, raw, index):
        """A matrix's input columns reordered (whole value heads) by moving whole blocks of its stored format."""
        from crit_core.model.gguf_blocks import BLOCKS, TYPES
        e = self.headers[name]
        values, size = BLOCKS[TYPES[e.dtype]] if e.stored else (1, {"F32": 4, "F16": 2, "BF16": 2}[e.dtype])
        unit = int(self.config["linear_value_head_dim"])
        if unit % values:
            raise ValueError(f"model-template: {name} is {e.dtype}, whose {values}-value blocks do not align with the "
                             f"{unit}-wide value heads the converter reordered; the inputs cannot be moved as stored")
        heads = index.reshape(-1, unit)[:, 0] // unit
        per = unit // values * size
        blocks = raw.reshape(raw.shape[0], -1, per)
        return blocks[:, heads].reshape(raw.shape[0], -1)

    def values(self, name):
        """FP32 (int64 for a constant) torch tensor of HF tensor `name` on the CPU, its Hugging Face shape and layout."""
        import numpy as np
        import torch
        from crit_core.model.gguf_blocks import TYPES, stored_values
        e = self.headers[name]
        if not e.sources:
            return torch.tensor(e.values, dtype=torch.int64)
        if e.latent is not None:
            heads, nope = e.latent
            key, value = (self._decoded(info) for info in e.sources)
            return torch.cat([key.transpose(1, 2), value], dim=1).reshape(e.shape)
        if e.stored:
            raw = self.rows(name)
            flat = raw.reshape(-1, raw.shape[-1])
            k = e.shape[-1] if len(e.shape) > 1 else e.shape[0]
            out = np.empty((flat.shape[0], k), dtype=np.float32)
            for start in range(0, flat.shape[0], 8192):   # bounded temporaries
                out[start:start + 8192] = stored_values(flat[start:start + 8192], TYPES[e.dtype], k)
            tensor = torch.from_numpy(out)
        else:
            parts = []
            for info in e.sources:
                kind = {"F32": torch.float32, "F16": torch.float16, "BF16": torch.bfloat16}[info.dtype]
                t = torch.from_numpy(np.array(self.reader.raw(info.name))).view(kind).reshape(info.shape)
                if e.expert is not None:
                    t = t[e.expert]
                if isinstance(e.rows, slice):
                    t = t[e.rows]
                if e.columns is not None:
                    t = t[:, e.columns]
                parts.append(t.float())
            tensor = parts[0] if len(parts) == 1 else torch.cat(parts)
            if e.order is not None:
                index, along = e.order
                tensor = tensor[torch.from_numpy(index)] if along == "rows" else tensor[:, torch.from_numpy(index)]
        if e.transform == "minus_one":
            tensor = tensor - 1
        elif e.transform == "log_neg":
            tensor = torch.log(-tensor)
        return tensor.reshape(e.shape)

    def _decoded(self, info):
        """FP32 torch tensor of a whole GGUF tensor, its stored shape (quantized rows decoded)."""
        import numpy as np
        import torch
        from crit_core.model.gguf_blocks import TYPES, stored_values
        if info.dtype in DENSE:
            kind = {"F32": torch.float32, "F16": torch.float16, "BF16": torch.bfloat16}[info.dtype]
            return torch.from_numpy(np.array(self.reader.raw(info.name))).view(kind).reshape(info.shape).float()
        raw = self.reader.rows(info.name)
        flat = raw.reshape(-1, raw.shape[-1])
        return torch.from_numpy(stored_values(flat, TYPES[info.dtype], info.shape[-1])).reshape(info.shape)

    def get(self, name, device):
        return self.values(name).to(device)

    def close(self):
        self.reader.close()


class GgufMatrices:
    """A GGUF checkpoint's matrices as the projections slot's load() and the experts slot's load() take them (the
    template's Matrices interface): quantized ones as gguf:<TYPE> with their stored rows, dense ones as bf16."""

    def __init__(self, checkpoint, projections, quantization, device):
        import torch
        self.ck, self.proj, self.device, self.torch = checkpoint, projections, device, torch
        self.stored_experts = True   # the model sets it: whether the experts plugin takes GGUF experts as stored

    def _entry(self, base):
        e = self.ck.headers.get(base + ".weight")
        if e is None:
            raise KeyError(f"model-template: no matrix {base} in the checkpoint")
        return e

    def has(self, base):
        return base + ".weight" in self.ck

    def format(self, base):
        e = self._entry(base)
        return f"gguf:{e.dtype}" if e.stored else "bf16"

    def stored(self, base, device=None):
        torch = self.torch
        e = self._entry(base)
        if e.stored:
            return f"gguf:{e.dtype}", {"weight": torch.from_numpy(self.ck.rows(base + ".weight"))}
        where = self.device if device is None else device
        return "bf16", {"weight": self.ck.get(base + ".weight", where).to(torch.bfloat16)}

    def stacked(self, bases, device=None):
        torch = self.torch
        found = [self.stored(b, device) for b in bases]
        fmt = found[0][0]
        if any(f != fmt for f, _ in found):
            raise ValueError(f"model-template: {bases} are stored in different formats; stack only one")
        if len(found) == 1:
            return fmt, found[0][1]
        return fmt, {"weight": torch.cat([t["weight"] for _, t in found])}

    def load(self, bases, role, encode=None, rows=None):
        fmt, tensors = self.stacked(bases)
        if rows is not None and tensors["weight"].shape[0] > rows:
            raise ValueError(f"model-template: {bases[0]} stores {tensors['weight'].shape[0]} rows, the head predicts "
                             f"{rows}: a GGUF's rows are taken whole")
        if encode is not None:
            return self.proj.load(tensors, fmt, role=role, encode=encode)
        return self.proj.load(tensors, fmt, role=role)

    def together(self, bases, role, encode=None):
        if len({self.format(b) for b in bases}) == 1 and len({self.ck.shape(b + ".weight")[-1] for b in bases}) == 1:
            return [self.load(list(bases), role, encode)]
        return [self.load([b], role, encode) for b in bases]

    def bias(self, bases):
        names = [b + ".bias" for b in bases]
        if not any(n in self.ck for n in names):
            return None
        if not all(n in self.ck for n in names):
            raise ValueError(f"model-template: only some of {bases} store a bias")
        return self.torch.cat([self.ck.get(n, self.device).float() for n in names])

    def rows(self, base):
        return self._entry(base).shape[0]

    def expert(self, gate, up, down, want=None, device=None):
        """(format, gate_up, down) of one routed expert: bf16 matrices where the file stores them dense or the
        experts plugin takes no GGUF experts (stored_experts: their stored values, rounded to BF16), else "gguf" with
        each part's stored rows and type ({"weight": uint8 [rows][row bytes], "type": "IQ2_S"}): the gate's and the up
        projection's rows stacked (one type), the down projection's as stored. `want` (the format a draft layer's
        experts are re-encoded to) is never given for a GGUF: each of its layers keeps the formats it is stored in.
        device: where decoded values are handed over (the model's by default); stored rows are host bytes."""
        if want is not None:
            raise ValueError(f"model-template: a GGUF's experts keep their stored formats, not {want}")
        torch = self.torch
        if not self.stored_experts:
            where = self.device if device is None else device
            values = [self.ck.get(b + ".weight", where).to(torch.bfloat16) for b in (gate, up, down)]
            return "bf16", torch.cat(values[:2]), values[2]
        fmt, gate_up = self.stacked([gate, up], device)
        down_fmt, down_t = self.stacked([down], device)
        if fmt == "bf16" and down_fmt == "bf16":
            return "bf16", gate_up["weight"], down_t["weight"]
        if fmt == "bf16" or down_fmt == "bf16":
            raise ValueError(f"model-template: {gate} and {down} are stored one dense and one quantized")
        return "gguf", {"weight": gate_up["weight"], "type": fmt[5:]}, {"weight": down_t["weight"], "type": down_fmt[5:]}

    def expert_formats(self, base):
        """(gate_up type, down type) of the routed experts under `base` (a layer's experts prefix), from expert 0."""
        return (self._entry(base + "0.gate_proj").dtype, self._entry(base + "0.down_proj").dtype)


def is_gguf(path):
    """Whether `path` is a GGUF model (a .gguf file, or a folder with one and no safetensors)."""
    from crit_core.model.gguf import SUFFIX, find_gguf
    path = Path(path)
    if path.is_file():
        return path.suffix == SUFFIX
    return path.is_dir() and not any(path.glob("*.safetensors")) and find_gguf(path) is not None


def generation_ends(path):
    """The end tokens of a GGUF's derived generation_config.json (Core's derivation from its metadata), or None."""
    import json
    from crit_core.model import open_model
    try:
        found = json.loads(open_model(path).files.read("generation_config.json")).get("eos_token_id")
    except (OSError, ValueError, KeyError):
        return None
    if found is None:
        return None
    return sorted(set(found if isinstance(found, list) else [found]))


__all__ = ["ARCHITECTURES", "GgufCheckpoint", "GgufMatrices", "generation_ends", "is_gguf"]
