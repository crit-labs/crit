"""Rows of a hashed n-gram table gated into the residual ahead of a layer, with a dilated causal convolution.

A model technique: the model plugin (models/core/model-template) mounts this code as its source `ngram_rows`
and composes the block ngram-rows a layout names (README.md)."""
import math
from ..template.blocks import _tensor, row_sum


NGRAM = {"size": "ngram_size", "heads": "heads_per_ngram", "base": "ngram_vocab_size_base", "dim": "ple_embed_dim",
         "kernel": "ple_conv_kernel_size", "seed": "seed", "layers": "ple_layer_ids"}


_MASK64 = (1 << 64) - 1


_GAMMA, _M1, _M2, _PRIME = 0x9E3779B97F4A7C15, 0xBF58476D1CE4E5B9, 0x94D049BB133111EB, 10007


def _splitmix64(value):
    value = (value + _GAMMA) & _MASK64
    value = ((value ^ (value >> 30)) * _M1) & _MASK64
    value = ((value ^ (value >> 27)) * _M2) & _MASK64
    return (value ^ (value >> 31)) & _MASK64


def ngram_hash(vocabulary, size, heads, base, index, seed):
    """(multipliers, sizes) of the `index`-th hashed n-gram table of a model, as its reference derives them where the
    checkpoint does not store them: one odd multiplier a token of the n-gram (splitmix64 of the seed, below the bound
    that keeps token * multiplier inside 63 bits) and each head's table size, the next primes from `base` on (the
    tables of a model take them in turn)."""
    bound = max(1, (((1 << 63) - 1) // max(vocabulary, 1)) // 2)
    start = seed + _PRIME * index
    multipliers = [2 * (_splitmix64((start + _GAMMA * (i + 1)) & _MASK64) % bound) + 1 for i in range(size)]

    def prime(n):
        return n >= 2 and all(n % d for d in range(2, math.isqrt(n) + 1))
    sizes, at = [], base - 1
    for _ in range((index + 1) * heads):
        at += 1
        while not prime(at):
            at += 1
        sizes.append(at)
    return multipliers, sizes[index * heads:]


class NgramRows:
    """Rows of a hashed n-gram table gated into every residual stream ahead of a layer (a `before` block over a
    streams residual).

    A token's n-grams, of 2 .. size tokens ending at it, are hashed to table rows, several heads an n-gram: the tokens
    times one odd multiplier each, XORed, modulo each head's own prime size, at the head's offset in one table. An
    n-gram never reaches past the end token before it: the tokens from that one back count as end tokens, as those
    before the sequence do. The rows side by side are the token's embedding e, and with n the streams' norms

        key = rms_norm(key_proj(e))           one a stream            value = value_proj(e)
        g = key . rms_norm(stream) / sqrt(hidden),  then sign(g) * sqrt(max(|g|, 1e-6))
        v = sigmoid(g) * value                one a stream
        stream += v + silu(conv(rms_norm(v)))  a causal depthwise convolution over the token's sequence, dilated by
                                               the n-gram size (the recurrence slot's, its window a pool the block
                                               declares)

    The table (hundreds of millions of rows where the n-gram vocabulary is large) is planned as a lookup table: on
    the device where the placement plugin puts it there, else read where the checkpoint stores it (its shards'
    files) through an embeddings plugin that provides embeddings:host; stored NVFP4 rows (their global scale
    weight_scale_2) and FP8 rows (E4M3 codes with one multiplier for the table, weight_scale) are decoded on lookup. The
    multipliers, sizes and offsets are the checkpoint's where it stores them, else derived as its reference derives
    them. Config keys (the spec's `config` renames them): ngram_size, heads_per_ngram, ngram_vocab_size_base,
    ple_embed_dim (the hidden width by default), ple_conv_kernel_size, seed, ple_layer_ids (the layers, for the
    table's turn among a model's tables), and eos_token_id. Spec: embedding (ple_embedding.) and table
    (ngram_embedding.), the tensors' prefixes under the block's."""

    @classmethod
    def _config(cls, model, spec):
        names = {**NGRAM, **(spec.get("config") or {})}
        found = {key: model.setting(name) for key, name in names.items()}
        found["dim"] = int(found["dim"] or model.hidden)
        for key in ("size", "heads", "base", "kernel"):
            if not isinstance(found[key], int) or found[key] < 1:
                raise ValueError(f"model-template: hashed n-gram rows take the config's {names[key]} (a positive "
                                 f"integer), not {found[key]!r}")
        if found["size"] < 2 or found["dim"] % ((found["size"] - 1) * found["heads"]):
            raise ValueError(f"model-template: {found['dim']} n-gram values over {found['size'] - 1} n-grams of "
                             f"{found['heads']} heads")
        return found

    @classmethod
    def _names(cls, model, prefix, spec):
        """The table's row tensors in row order: one tensor, or its shards."""
        ck = model.ck
        at = prefix + spec.get("embedding", "ple_embedding.") + spec.get("table", "ngram_embedding.")
        if at + "weight" in ck:
            return at, [at + "weight"]
        names = []
        while f"{at}shard_{len(names)}.weight" in ck:
            names.append(f"{at}shard_{len(names)}.weight")
        if not names:
            raise KeyError(f"model-template: the checkpoint has no {at}weight or {at}shard_0.weight")
        return at, names

    @classmethod
    def tables(cls, model, prefix, spec):
        """{plan name: (bytes, the tensor-name prefix the plan covers)}."""
        at, _ = cls._names(model, prefix, spec)
        return {"tables." + at.rstrip("."): (sum(model.ck.nbytes(n) for n in model.ck.names() if n.startswith(at)),
                                             at)}

    @classmethod
    def pools(cls, model, spec, layers):
        found = cls._config(model, spec)
        count = getattr(model.residual, "count", 0)
        if not count:
            raise ValueError("model-template: hashed n-gram rows join residual streams (the layout's residual: "
                             "streams)")
        return [{"key": "ngram", "kind": "window", "layers": list(layers), "channels": count * model.hidden,
                 "kernel": found["kernel"], "dilation": found["size"]}]

    def __init__(self, model, layer, prefix, spec):
        m = self.m = model
        torch = m.torch
        self.layer, self.pool = layer, "ngram"
        found = self._config(m, spec)
        self.size, self.per, self.dim = found["size"], found["heads"], found["dim"]
        self.token_window = self.size   # the ids a token's rows are hashed from: its own and the size - 1 before it
        self.captures_prompts = False   # a chunk's windows of ids come from the request, on the host
        heads = (self.size - 1) * self.per
        self.count, hidden = m.residual.count, m.hidden
        eos = m.config.get("eos_token_id")
        eos = eos[0] if isinstance(eos, (list, tuple)) and eos else eos
        if not isinstance(eos, int):
            raise ValueError("model-template: hashed n-gram rows stop at the end token: the config's eos_token_id")
        self.eos = eos
        ep = prefix + spec.get("embedding", "ple_embedding.")
        stored = [ep + n for n in ("layer_multipliers", "ngram_heads_vocab_sizes", "ngram_heads_offsets")]
        if all(n in m.ck for n in stored):
            multipliers, sizes, offsets = (m.ck.get(n, "cpu").tolist() for n in stored)
        else:
            turn = sorted(found["layers"] or [layer + 1]).index(layer + 1)
            multipliers, sizes = ngram_hash(m.vocabulary, self.size, heads, found["base"], turn,
                                            int(found["seed"] if found["seed"] is not None else 1234))
            offsets = [sum(sizes[:i]) for i in range(heads)]
        if len(multipliers) != self.size or len(sizes) != heads or len(offsets) != heads:
            raise ValueError(f"model-template: {ep} hashes {len(multipliers)} tokens into {len(sizes)} heads, the "
                             f"config's n-grams take {self.size} and {heads}")
        long = dict(dtype=torch.int64, device=m.device)
        self.multipliers = torch.tensor(multipliers, **long)
        self.sizes, self.offsets = torch.tensor(sizes, **long), torch.tensor(offsets, **long)
        self.table = self._table(prefix, spec, sizes[-1] + offsets[-1])
        self.key_value = m.matrices.together([prefix + "key_proj", prefix + "value_proj"], "ngram")
        if m.matrices.rows(prefix + "key_proj") != self.count * hidden:
            raise ValueError(f"model-template: {prefix}key_proj gives {m.matrices.rows(prefix + 'key_proj')} values, "
                             f"a key a stream is {self.count * hidden}")
        self.norm_key, self.norm_query, self.norm_conv = (
            _tensor(m, prefix + name + ".weight").float().view(self.count, hidden).contiguous()
            for name in ("norm_key", "norm_query", "norm_conv"))
        self.conv = _tensor(m, prefix + "conv1d.weight")[:, 0].float().contiguous()

    def _table(self, prefix, spec, rows):
        """The table through the embeddings slot: on the device where its plan put it, else read in place from the
        checkpoint's files (embeddings:host)."""
        m, ck, torch = self.m, self.m.ck, self.m.torch
        at, names = self._names(m, prefix, spec)
        width = self.dim // ((self.size - 1) * self.per)
        if sum(ck.shape(n)[0] for n in names) < rows:
            raise ValueError(f"model-template: {at} stores {sum(ck.shape(n)[0] for n in names)} rows, its heads hash "
                             f"into {rows}")
        if getattr(ck.headers[names[0]], "stored", False):   # a GGUF's quantized rows, decoded on lookup
            return self._gguf_table(names, width)
        stored = ck.headers[names[0]].dtype
        nvfp4, fp8 = stored == "U8", stored == "F8_E4M3"
        scales = [n + "_scale" for n in names] if nvfp4 else []
        scale = 1.0
        if nvfp4 or fp8:   # NVFP4's global scale, or FP8's one multiplier for the whole table (ModelOpt's weight_scale)
            name = at + ("weight_scale_2" if nvfp4 else "weight_scale")
            if name not in ck or ck.headers[name].nbytes > 8:
                raise ValueError(f"model-template: {at} stores {'NVFP4' if nvfp4 else 'FP8'} rows, read with one "
                                 f"scale for the table ({name}), which the checkpoint does not store as one value")
            scale = float(ck.get(name, "cpu").float().reshape(-1)[0])
        if stored not in ("U8", "F8_E4M3", "BF16", "F16", "F32"):
            raise ValueError(f"model-template: {at} stores {stored} rows; a table is BF16, F16, F32, NVFP4 or FP8")
        if m.placed.get("tables." + at.rstrip("."), "device") == "device":
            found = torch.cat([ck.get(n, m.device) for n in names])
            if nvfp4:
                return m.embeddings.store(found, format="nvfp4", scale=scale, dtype=torch.bfloat16,
                                          scales=torch.cat([ck.get(n, m.device) for n in scales]))
            if fp8:
                return m.embeddings.store(found, format="fp8", scale=scale, dtype=torch.bfloat16)
            return m.embeddings.store(found.to(torch.bfloat16))
        if "embeddings:host" not in m.parts.facts("embeddings"):
            raise ValueError(f"model-template: the table {at} lies off the device, and the embeddings plugin does "
                             "not provide embeddings:host (lookup-table-rows reads a table where the checkpoint "
                             "stores it)")

        def files(tensors):
            return {"paths": [str(ck.headers[n].shard) for n in tensors],
                    "offsets": [int(ck.headers[n].data_begin) for n in tensors],
                    "rows": [int(ck.shape(n)[0]) for n in tensors]}
        if nvfp4:
            return m.embeddings.open(files(names), width, torch.bfloat16, m.device, format="nvfp4",
                                     scales=files(scales), scale=scale)
        if fp8:
            return m.embeddings.open(files(names), width, torch.bfloat16, m.device, format="fp8", scale=scale)
        kind = {"BF16": torch.bfloat16, "F16": torch.float16, "F32": torch.float32}[ck.headers[names[0]].dtype]
        return m.embeddings.open(files(names), width, kind, m.device)

    def _gguf_table(self, names, width):
        """The table as a GGUF stores it (rows of one block format), decoded a lookup (the model's gguf_table)."""
        at = names[0].rsplit("weight", 1)[0]
        return self.m.gguf_table(names, width, "tables." + at.rstrip("."))

    def rows(self, windows):
        """The table rows of each token ([tokens][heads] int64) from its window of ids ([tokens][size], the token's
        own last, -1 before the sequence)."""
        torch = self.m.torch
        ids = torch.where(windows < 0, self.eos, windows)
        mixed = ids[:, -1] * self.multipliers[0]
        ended = torch.zeros_like(mixed, dtype=torch.bool)
        found = []
        for back in range(1, self.size):
            earlier = ids[:, -1 - back]
            ended = ended | (earlier == self.eos)   # an end token from here back: the n-gram stops at it
            mixed = mixed ^ (torch.where(ended, self.eos, earlier) * self.multipliers[back])
            heads = slice((back - 1) * self.per, back * self.per)
            found.append(torch.remainder(mixed[:, None], self.sizes[None, heads]) + self.offsets[None, heads])
        return torch.cat(found, dim=1)

    def forward(self, streams, ctx):
        m, norms = self.m, self.m.norms
        tokens, count, hidden = streams.shape
        if ctx.windows is None:
            raise RuntimeError("model-template: hashed n-gram rows read each token's window of ids, and this forward "
                               "has none")
        e = self.table.lookup(self.rows(ctx.windows).reshape(-1)).reshape(tokens, self.dim)
        found = [ctx.mm(e, matrix) for matrix in self.key_value]
        kv = found[0] if len(found) == 1 else m.torch.cat(found, dim=-1)
        key = norms.rms_norm(kv[:, :count * hidden].reshape(tokens, count, hidden), self.norm_key, m.eps,
                             offset=m.norm_offset)
        query = norms.rms_norm(streams, self.norm_query, m.eps, offset=m.norm_offset)
        gate = row_sum(key.float() * query.float()) * hidden ** -0.5
        gate = gate.abs().clamp_min(1e-6).sqrt() * gate.sign()
        value = norms.sigmoid_mul(kv[:, None, count * hidden:].expand(tokens, count, hidden).contiguous(),
                                  gate[:, :, None].expand(tokens, count, hidden).contiguous(),
                                  out_dtype=streams.dtype)
        normed = norms.rms_norm(value, self.norm_conv, m.eps, offset=m.norm_offset)
        conv = m.recurrence.convolve(self.layer, normed.reshape(tokens, count * hidden), self.conv, None,
                                     m.pools[self.pool], ctx.batch)
        return (streams.float() + value.float() + conv.view(tokens, count, hidden).float()).to(streams.dtype)


BLOCKS = {"ngram-rows": NgramRows}
