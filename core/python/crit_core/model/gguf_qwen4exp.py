"""config.json for a qwen4exp GGUF (Qwen3.8-Flash-Next; transformers' qwen4_exp_text), from its metadata (standard
library only): the inverse of the reference converter's Qwen4ExpTextModel.set_gguf_parameters (conversion/qwen4exp.py),
for the model as GGUF's reference runtime defines it (src/models/qwen4exp.cpp).

The architecture is Qwen3.5 MoE's (gated delta net, interleaved MRoPE, routed and shared experts), whose keys
crit_core.model.gguf_files derives, plus:

    hyper-connections   hc_count, hc_lowrank (hyper_connection.count, .low_rank)
    QSA indexer         indexer_n_heads, indexer_head_dim, indexer_budget (attention.indexer.head_count, .key_length,
                        .top_k), indexer_compress_ratio (the one nonzero attention.compress_ratios value) and
                        indexer_kv_heads 1 (the converter splits index_qk_proj into a query and one key head)
    PLE                 ple_layer_ids (ple.layers, 0-based in the GGUF), ngram_size, heads_per_ngram,
                        ple_conv_kernel_size, ple_embed_dim (the n-gram table's row width times its heads) and
                        eos_token_id (ple.eos_token_id, the token the n-gram hash restarts after)
    runtime facts       output_gate_type "sigmoid" (the runtime gates the delta net's output with a sigmoid) and
                        norm_topk_prob (it renormalises the top-k expert weights)

The modeling code computes the n-gram hash constants (Qwen4ExpTextNGramEmbedding's layer_multipliers,
ngram_heads_vocab_sizes and ngram_heads_offsets buffers) and the table's rows from seed, ngram_vocab_size_base and
make_ngram_vocab_size_divisible_by, which the GGUF does not keep: it keeps the constants (ple.layer_multipliers,
.head_vocab_sizes, .head_offsets). Those three keys are derived so that the modeling code computes exactly the stored
constants and the stored table's rows, else the GGUF is refused: the base is 20,000,000 (the config default) when it
gives the head vocabularies, else the smallest base that does; the divisor 128 (the default) when it gives the rows,
else the smallest that does; the seed 1234 (the default) when it gives the multipliers, else the seed found by undoing
the hash (splitmix64 is a bijection, so each multiplier leaves 2^64 / bound candidates, checked against the others).

The runtime holds one set of hash constants and one n-gram table, so a GGUF with several PLE layers is refused as it
refuses it; and an attention layer without a compress ratio runs plain attention in the runtime, which transformers'
Qwen4Exp (QSA on every attention layer) cannot, so it is refused too. The converter writes such ratios when it reads
config.json through a transformers that renames the released "full_attention" layer types to "indexed_attention".
"""
import math

from .gguf import GgufError

DEFAULT_SEED = 1234
DEFAULT_NGRAM_BASE = 20_000_000
DEFAULT_DIVISOR = 128
# transformers.models.qwen4_exp.modeling_qwen4_exp: the n-gram hash.
_MASK64 = (1 << 64) - 1
_GAMMA = 0x9E3779B97F4A7C15
_M1, _M2 = 0xBF58476D1CE4E5B9, 0x94D049BB133111EB
_M1_INVERSE, _M2_INVERSE = pow(_M1, -1, 1 << 64), pow(_M2, -1, 1 << 64)
_PRIME_1 = 10007


def _splitmix64(value):
    value = (value + _GAMMA) & _MASK64
    value = ((value ^ (value >> 30)) * _M1) & _MASK64
    value = ((value ^ (value >> 27)) * _M2) & _MASK64
    return (value ^ (value >> 31)) & _MASK64


def _unsplitmix64(value):
    """The value _splitmix64 maps to `value`."""
    value ^= (value >> 31) ^ (value >> 62)
    value = (value * _M2_INVERSE) & _MASK64
    value ^= (value >> 27) ^ (value >> 54)
    value = (value * _M1_INVERSE) & _MASK64
    value ^= (value >> 30) ^ (value >> 60)
    return (value - _GAMMA) & _MASK64


def _half_bound(vocab_size):
    return max(1, ((1 << 63) - 1) // max(vocab_size, 1) // 2)


def layer_multipliers(vocab_size, ngram_size, ple_layer_index, seed):
    """_build_layer_multipliers: the n-gram hash's multiplier per token position."""
    half, base = _half_bound(vocab_size), seed + _PRIME_1 * ple_layer_index
    return tuple(2 * (_splitmix64((base + _GAMMA * (i + 1)) & _MASK64) % half) + 1 for i in range(ngram_size))


def _is_prime(value):
    if value < 2:
        return False
    if value % 2 == 0:
        return value == 2
    return all(value % divisor for divisor in range(3, math.isqrt(value) + 1, 2))


def head_vocab_sizes(base, heads):
    """The prime vocabulary of each n-gram head of the first PLE layer (Qwen4ExpTextNGramEmbedding: head h's is the
    (h + 1)-th prime from base on)."""
    sizes, prime = [], base - 1
    for _ in range(heads):
        prime += 1
        while not _is_prime(prime):
            prime += 1
        sizes.append(prime)
    return tuple(sizes)


def _seed(multipliers, vocab_size, index):
    """A seed whose multipliers are `multipliers`: the default when it gives them, else one found by undoing the hash
    of the first multiplier for each value its residue leaves, or None."""
    count = len(multipliers)
    if layer_multipliers(vocab_size, count, index, DEFAULT_SEED) == multipliers:
        return DEFAULT_SEED
    half = _half_bound(vocab_size)
    residue = (multipliers[0] - 1) // 2
    if multipliers[0] % 2 == 0 or residue >= half:
        return None
    for value in range(residue, 1 << 64, half):
        base = (_unsplitmix64(value) - _GAMMA) & _MASK64
        # The second multiplier rules out almost every candidate at the cost of one hash.
        if count > 1 and 2 * (_splitmix64((base + 2 * _GAMMA) & _MASK64) % half) + 1 != multipliers[1]:
            continue
        seed = (base - _PRIME_1 * index) & _MASK64
        if layer_multipliers(vocab_size, count, index, seed) == multipliers:
            return seed
    return None


def _base(sizes):
    """An ngram_vocab_size_base whose head vocabularies are `sizes`: the default when it gives them, else the smallest
    that does (one past the prime before the first head's), or None."""
    if head_vocab_sizes(DEFAULT_NGRAM_BASE, len(sizes)) == sizes:
        return DEFAULT_NGRAM_BASE
    below = sizes[0] - 1
    while below > 1 and not _is_prime(below):
        below -= 1
    base = below + 1 if below > 1 else 0
    return base if head_vocab_sizes(base, len(sizes)) == sizes else None


def _divisor(total, rows):
    """A make_ngram_vocab_size_divisible_by that pads `total` n-gram ids to the table's `rows`: the default when it
    does, else the smallest that does, or None."""
    if math.ceil(total / DEFAULT_DIVISOR) * DEFAULT_DIVISOR == rows:
        return DEFAULT_DIVISOR
    return next((d for d in range(max(1, rows - total + 1), rows + 1)
                 if rows % d == 0 and math.ceil(total / d) * d == rows), None)


def _compress_ratio(model, get, kinds):
    """The one compress ratio of the QSA layers, checked to cover every attention layer of `kinds` (the layer types)."""
    ratios = get(".attention.compress_ratios")
    ratios = tuple(ratios) if isinstance(ratios, tuple) else (ratios or 0,) * get(".block_count")
    plain = [i for i, kind in enumerate(kinds) if kind == "full_attention" and not ratios[i]]
    if plain:
        raise GgufError(f"attention layers {plain} have no compress ratio ({model.architecture}.attention."
                        "compress_ratios), so GGUF's runtime runs them without the QSA indexer, which transformers' "
                        "Qwen4Exp always applies; the converter writes this when transformers renamed the released "
                        "'full_attention' layer types to 'indexed_attention' before it read them")
    values = {r for r in ratios if r}
    if len(values) != 1:
        raise GgufError(f"{model.architecture}.attention.compress_ratios holds {sorted(values) or 'no'} QSA block "
                        "sizes; the architecture has one")
    return values.pop()


def _array(model, get, key):
    value = get(key)
    if not isinstance(value, tuple):
        raise GgufError(f"the GGUF has no {model.architecture}{key} array; the PLE hash needs it")
    return value


def _ple(model, get, config):
    """The PLE keys, checked to make the modeling code compute the GGUF's hash constants and table rows."""
    layers = get(".ple.layers")
    if not layers:
        return {"ple_layer_ids": []}
    if len(layers) != 1:
        raise GgufError(f"{model.architecture}.ple.layers names {len(layers)} layers; the GGUF holds one set of n-gram "
                        "hash constants and one table, as GGUF's runtime requires")
    ngram, per_ngram = get.require(".ple.ngram_size"), get.require(".ple.heads_per_ngram")
    heads = (ngram - 1) * per_ngram
    multipliers = _array(model, get, ".ple.layer_multipliers")
    sizes, offsets = _array(model, get, ".ple.head_vocab_sizes"), _array(model, get, ".ple.head_offsets")
    if len(multipliers) != ngram or len(sizes) != heads or len(offsets) != heads:
        raise GgufError(f"the PLE constants hold {len(multipliers)} multipliers and {len(sizes)}/{len(offsets)} head "
                        f"sizes/offsets for {ngram}-grams with {heads} heads")
    if offsets != tuple(sum(sizes[:h]) for h in range(heads)):
        raise GgufError("the PLE head offsets are not the running sums of the head vocabularies, as transformers' "
                        "Qwen4Exp lays the n-gram table out")
    table = model.tensors.get("per_layer_token_embd.weight")
    if table is None:
        raise GgufError("the GGUF has PLE keys but no n-gram table (per_layer_token_embd.weight)")
    rows, width = table.shape
    key = model.tensors.get(f"blk.{layers[0]}.ple_key.weight")
    if key is not None and key.shape[-1] != width * heads:
        raise GgufError(f"the PLE key projection reads {key.shape[-1]} features; {heads} heads of the table's "
                        f"{width} make {width * heads}")
    base, divisor = _base(sizes), _divisor(sum(sizes), rows)
    seed = _seed(multipliers, config["vocab_size"], 0)
    for value, what in ((base, "head vocabularies are not consecutive primes"),
                        (divisor, f"table's {rows} rows do not pad its {sum(sizes)} n-gram ids to a multiple"),
                        (seed, "hash multipliers come from no seed")):
        if value is None:
            raise GgufError(f"the GGUF's PLE {what}, as transformers' Qwen4Exp computes them")
    return {"ple_layer_ids": [layers[0] + 1], "ple_embed_dim": width * heads,
            "ple_conv_kernel_size": get.require(".ple.conv_kernel"), "ngram_size": ngram, "heads_per_ngram": per_ngram,
            "ngram_vocab_size_base": base, "make_ngram_vocab_size_divisible_by": divisor, "seed": seed,
            "eos_token_id": get.require(".ple.eos_token_id")}


def qwen4exp_config(model, get, qwen35moe):
    """config.json for a qwen4exp GgufModel; `qwen35moe` is what the Qwen3.5 MoE rule derives from its metadata."""
    config = dict(qwen35moe)
    config.pop("attn_output_gate", None)   # Qwen4Exp's attention always gates its output (q_proj holds the gate)
    ratio = _compress_ratio(model, get, config["layer_types"])
    head_dim = get.require(".attention.indexer.key_length")
    for i, kind in enumerate(config["layer_types"]):
        key = model.tensors.get(f"blk.{i}.indexer.k_proj.weight")
        if kind == "full_attention" and key is not None and key.shape[0] != head_dim:
            raise GgufError(f"blk.{i}.indexer.k_proj has {key.shape[0]} rows; QSA has one key head of {head_dim}")
    config.update({
        "architectures": ["Qwen4ExpForCausalLM"], "model_type": "qwen4_exp_text",
        "hc_count": get.require(".hyper_connection.count"), "hc_lowrank": get.require(".hyper_connection.low_rank"),
        "indexer_n_heads": get.require(".attention.indexer.head_count"), "indexer_kv_heads": 1,
        "indexer_head_dim": head_dim, "indexer_budget": get.require(".attention.indexer.top_k"),
        "indexer_compress_ratio": ratio, "output_gate_type": "sigmoid", "norm_topk_prob": True})
    config.update(_ple(model, get, config))
    return config
