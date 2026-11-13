"""A GGUF checkpoint's quantized matrices as stored: their bytes (GgufTensors, mapped from the inspected files, never
copied) and the block formats' exact host decoders, the reference every device form of them is checked against.

    with GgufTensors(inspect_gguf_model(path)) as reader:
        raw = reader.rows("blk.0.ffn_up.weight")      # uint8 [N][row bytes] (or [experts][N][row bytes])
        q, s0, s1, m = subblocks(raw, TYPES["Q4_K"], k)  # each 32-value sub-block's codes and FP32 scales
        w = stored_values(raw, TYPES["Q4_K"], k)          # W [N][K] FP32: the stored values

A row of a quantized matrix is its blocks in order; a block holds 32 values, or a 256-value superblock of eight 32-value
sub-blocks. subblocks() gives every sub-block's 32 values as integer codes and its scales: value e is
fsub(fmul(s0 if e < 16 else s1, q_e), m), the scales being the stored FP16 scale times the sub-block's integer scale,
rounded once, so stored_values() is the stored value bit for bit as the format's own reference decoder computes it in
FP32. crit's kernel core decoders (crit_kcore/gguf.cuh) give the same codes and scales.

Formats (TYPES: GGUF type ids): Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, Q2_K, Q3_K, Q4_K, Q5_K, Q6_K, IQ1_S, IQ1_M, IQ2_XXS,
IQ2_XS, IQ2_S, IQ3_XXS, IQ3_S, IQ4_NL, IQ4_XS, MXFP4 (E2M1 doubled, its E8M0 byte e as 2^(e - 128), exact). IQ1's
offset of an eighth is in the codes (8 g +- 1, the scale d l / 8,
exact). Q4_1's
and Q5_1's stored offset is added to s q, so m is its negation (exact). Q2_K has a minimum a 16 values:
subblocks(..., halves=True) gives m0 (values 0..15) and m1 (16..31); without halves a Q2_K matrix is refused. Requires
numpy.
"""
from pathlib import Path
import mmap

from .gguf import GgufError

TYPES = {"Q4_0": 2, "Q4_1": 3, "Q5_0": 6, "Q5_1": 7, "Q8_0": 8, "Q2_K": 10, "Q3_K": 11, "Q4_K": 12, "Q5_K": 13, "Q6_K": 14,
         "IQ2_XXS": 16, "IQ2_XS": 17, "IQ3_XXS": 18, "IQ1_S": 19, "IQ4_NL": 20, "IQ3_S": 21, "IQ2_S": 22, "IQ4_XS": 23,
         "IQ1_M": 29, "MXFP4": 39}
# Per type id: (values, bytes) a block.
BLOCKS = {2: (32, 18), 3: (32, 20), 6: (32, 22), 7: (32, 24), 8: (32, 34), 10: (256, 84), 11: (256, 110), 12: (256, 144),
          13: (256, 176), 14: (256, 210), 16: (256, 66), 17: (256, 74), 18: (256, 98), 19: (256, 50), 20: (32, 18),
          21: (256, 110), 22: (256, 82), 23: (256, 136), 29: (256, 56), 39: (32, 17)}
# Where each format keeps its FP16 scales in a block.
SCALE_BYTES = {2: (0,), 3: (0, 2), 6: (0,), 7: (0, 2), 8: (0,), 10: (80, 82), 11: (108,), 12: (0, 2), 13: (0, 2), 14: (208,),
               16: (0,), 17: (0,), 18: (0,), 19: (0,), 20: (0,), 21: (0,), 22: (0,), 23: (0,), 29: (), 39: ()}
# Whether value e's scale is s0 for e < 16 and s1 for e >= 16, whether the format has a minimum, and whether its
# minimum is m0 for e < 16 and m1 for e >= 16.
TWO_SCALES = frozenset({10, 11, 14, 17, 22, 29})
HAS_MIN = frozenset({3, 7, 10, 12, 13})
TWO_MINS = frozenset({10})
IQ4_VALUES = (-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113)
MXFP4_VALUES = (0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12)
_GRIDS = {}


def grid(name):
    """An IQ format's grid from gguf_grids.json, int16 [entries][values]: iq3_s [512][4] (1, 3, ..., 15), iq3_xxs
    [256][4], iq2_xxs [256][8], iq2_xs [512][8], iq2_s [1024][8] (8, 25, 43), iq1_s [2048][8] (-1, 0, 1)."""
    if name not in _GRIDS:
        import json
        import numpy as np
        hexes = json.loads(Path(__file__).with_name("gguf_grids.json").read_text(encoding="utf-8"))[name]
        width = len(hexes[0]) // 2
        words = np.array([int(w, 16) for w in hexes], dtype=np.uint64)
        values = ((words[:, None] >> (8 * np.arange(width, dtype=np.uint64))) & np.uint64(0xff)).astype(np.uint8)
        _GRIDS[name] = values.view(np.int8).astype(np.int16) if name == "iq1_s" else values.astype(np.int16)
    return _GRIDS[name]


def iq3s_grid():
    """IQ3_S's grid [512][4] (uint8 values 1, 3, ..., 15), from gguf_grids.json."""
    import numpy as np
    return grid("iq3_s").astype(np.uint8)


def _even_signs(index):
    """IQ2's and IQ3_XXS's eight sign bits (1: negative) [..., 8] from 7-bit indices: the eighth makes them even."""
    import numpy as np
    index = index.astype(np.int64)
    bits = (index[..., None] >> np.arange(7)) & 1
    return np.concatenate([bits, bits.sum(-1, keepdims=True) & 1], axis=-1)


def _u16(blocks, at):
    """The little-endian 16-bit words at byte at (an int or an array of offsets) of each block."""
    return blocks[..., at].astype("int64") | (blocks[..., at + 1].astype("int64") << 8)


def row_bytes(type_id, k):
    """Bytes of a row of k values in the format (k a multiple of its block)."""
    values, size = BLOCKS[type_id]
    if k % values:
        raise GgufError(f"a row of {k} values is no whole number of {values}-value blocks")
    return k // values * size


def _f16(blocks, at):
    import numpy as np
    return blocks[..., at:at + 2].copy().view(np.float16)[..., 0].astype(np.float32)


def subblocks(raw, type_id, k, halves=False):
    """(q int8 [N][J][32], s0, s1, m FP32 [N][J]) of a matrix's stored blocks (raw uint8 [N][row bytes]), J = K / 32;
    with halves, (q, s0, s1, m0, m1), m0 the minimum of values 0..15 and m1 of 16..31 (Q2_K's; m0 == m1 elsewhere)."""
    import numpy as np
    if type_id in TWO_MINS and not halves:
        raise GgufError(f"GGUF type {type_id} has a minimum a 16 values: subblocks(..., halves=True)")
    if type_id not in BLOCKS:
        raise GgufError(f"GGUF type {type_id} is not one of {', '.join(f'{n} ({t})' for n, t in TYPES.items())}")
    values, size = BLOCKS[type_id]
    raw = np.ascontiguousarray(raw, dtype=np.uint8)
    n = raw.shape[0]
    if raw.ndim != 2 or raw.shape[1] != row_bytes(type_id, k):
        raise GgufError(f"a row of {k} values holds {row_bytes(type_id, k)} bytes, not {raw.shape[1:]}")
    blocks = raw.reshape(n, k // values, size)
    per = values // 32
    zero = np.zeros((n, k // values, per), dtype=np.float32)
    table = np.array(IQ4_VALUES, dtype=np.int8)
    if type_id == 8:
        d = _f16(blocks, 0)[..., None]
        q = blocks[..., 2:].view(np.int8).reshape(n, -1, 1, 32)
        s0 = s1 = d
        m = zero
    elif type_id in (2, 3, 6, 7):
        # Values 0..15 the 16 bytes' low nibbles, 16..31 their high nibbles; Q5's bit 4 of value e is bit e of qh.
        five, offset = type_id in (6, 7), type_id in (3, 7)
        d = _f16(blocks, 0)[..., None]
        m = -_f16(blocks, 2)[..., None] if offset else zero
        at = (4 if offset else 2) + (4 if five else 0)
        qs = blocks[..., at:at + 16]
        codes = np.concatenate([qs & 15, qs >> 4], axis=-1).astype(np.int16)
        if five:
            qh = blocks[..., at - 4:at].copy().view(np.uint32)[..., 0].astype(np.int64)
            codes = codes | (((qh[..., None] >> np.arange(32)) & 1) << 4).astype(np.int16)
        q = (codes - (0 if offset else 16 if five else 8)).astype(np.int8).reshape(n, -1, 1, 32)
        s0 = s1 = d
    elif type_id == 10:
        d, dmin = _f16(blocks, 80)[..., None], _f16(blocks, 82)[..., None]
        sc = blocks[..., 0:16].astype(np.int32)
        s = (d * (sc & 15).astype(np.float32)).astype(np.float32)
        mins = (dmin * (sc >> 4).astype(np.float32)).astype(np.float32)
        s0, s1, m0, m1 = s[..., 0::2], s[..., 1::2], mins[..., 0::2], mins[..., 1::2]
        qs = blocks[..., 16:80].reshape(n, -1, 2, 1, 32)                             # half h, l
        q = ((qs >> (2 * np.arange(4, dtype=np.uint8)).reshape(1, 1, 1, 4, 1)) & 3).astype(np.int8).reshape(n, -1, 8, 32)
        m = None
    elif type_id == 11:
        d = _f16(blocks, 108)[..., None]
        sc = blocks[..., 96:108].astype(np.int32)
        i = np.arange(16)
        low = np.where(i < 8, sc[..., i % 8] & 15, sc[..., i % 8] >> 4)
        scale = (low | (((sc[..., 8 + i % 4] >> (2 * (i // 4))) & 3) << 4)) - 32
        s = (d * scale.astype(np.float32)).astype(np.float32)
        s0, s1 = s[..., 0::2], s[..., 1::2]
        qs = blocks[..., 32:96].reshape(n, -1, 2, 1, 32).astype(np.int16)        # half h, l
        hm = blocks[..., 0:32].reshape(n, -1, 1, 32).astype(np.int16)
        g = np.arange(4).reshape(1, 1, 1, 4, 1)
        low = (qs >> (2 * g)) & 3                                                   # [n][blocks][h][g][32]
        ib = np.arange(8).reshape(1, 1, 8, 1)
        high = (hm >> ib) & 1                                                       # [n][blocks][ib][32]
        q = (low.reshape(n, -1, 8, 32) + 4 * high - 4).astype(np.int8)
        m = zero
    elif type_id == 39:
        e = blocks[..., 0].astype(np.uint32)
        bits = np.where(e < 2, np.uint32(0x00200000) << e, (e - 1) << np.uint32(23)).astype(np.uint32)
        d = bits.view(np.float32)[..., None]
        qs = blocks[..., 1:17]
        q = np.array(MXFP4_VALUES, dtype=np.int8)[np.concatenate([qs & 15, qs >> 4], axis=-1)].reshape(n, -1, 1, 32)
        s0 = s1 = d
        m = zero
    elif type_id == 20:
        d = _f16(blocks, 0)[..., None]
        qs = blocks[..., 2:18]
        q = table[np.concatenate([qs & 15, qs >> 4], axis=-1)].reshape(n, -1, 1, 32)
        s0 = s1 = d
        m = zero
    elif type_id == 23:
        d = _f16(blocks, 0)[..., None]
        scales_h = blocks[..., 2:4].copy().view(np.uint16)[..., 0].astype(np.int32)
        scales_l = blocks[..., 4:8].astype(np.int32)
        ib = np.arange(8)
        ls = ((scales_l[..., ib // 2] >> (4 * (ib % 2))) & 15) | (((scales_h[..., None] >> (2 * ib)) & 3) << 4)
        s0 = s1 = (d * (ls - 32).astype(np.float32)).astype(np.float32)
        qs = blocks[..., 8:].reshape(n, -1, 8, 16)
        q = table[np.concatenate([qs & 15, qs >> 4], axis=-1)]
        m = zero
    elif type_id in (12, 13):
        d, dmin = _f16(blocks, 0)[..., None], _f16(blocks, 2)[..., None]
        sc = blocks[..., 4:16].astype(np.int32)
        j = np.arange(8)
        lo = j < 4
        scale = np.where(lo, sc[..., j % 4] & 63, (sc[..., 8 + j % 4] & 15) | ((sc[..., j % 4] >> 6) << 4))
        mins = np.where(lo, sc[..., 4 + j % 4] & 63, (sc[..., 8 + j % 4] >> 4) | ((sc[..., 4 + j % 4] >> 6) << 4))
        s0 = s1 = (d * scale.astype(np.float32)).astype(np.float32)
        m = (dmin * mins.astype(np.float32)).astype(np.float32)
        at = 16 if type_id == 12 else 48
        qs = blocks[..., at:at + 128].reshape(n, -1, 4, 1, 32) >> np.array([0, 4], dtype=np.uint8).reshape(1, 1, 1, 2, 1)
        q = (qs & 15).astype(np.int8).reshape(n, -1, 8, 32)
        if type_id == 13:   # bit 4 of sub-block ib's value l: bit ib of qh[l]
            qh = blocks[..., 16:48].reshape(n, -1, 1, 32)
            q = q | (((qh >> np.arange(8, dtype=np.uint8).reshape(1, 1, 8, 1)) & 1) << 4).astype(np.int8)
    elif type_id == 14:
        d = _f16(blocks, 208)[..., None]
        scales = blocks[..., 192:208].view(np.int8).astype(np.float32)
        s0 = (d * scales[..., 0::2]).astype(np.float32)
        s1 = (d * scales[..., 1::2]).astype(np.float32)
        ql = blocks[..., 0:128].reshape(n, -1, 2, 2, 32)        # half, (g & 1), l
        qh = blocks[..., 128:192].reshape(n, -1, 2, 32)         # half, l
        q = np.empty((n, blocks.shape[1], 8, 32), dtype=np.int8)
        for ib in range(8):
            half, g = ib // 4, ib % 4
            low = (ql[:, :, half, g & 1] >> (4 * (g >> 1))) & 15
            high = (qh[:, :, half] >> (2 * g)) & 3
            q[:, :, ib] = ((low | (high << 4)).astype(np.int16) - 32).astype(np.int8)
        m = zero
    elif type_id in (16, 17, 22, 18):
        d = _f16(blocks, 0)[..., None]
        j = np.arange(8)
        if type_id in (16, 18):   # a word a sub-block: four 7-bit sign indices, the 4-bit scale in bits 28..31
            at = 2 + 8 * j + 4 if type_id == 16 else 66 + 4 * j
            aux = sum(blocks[..., at + i].astype(np.int64) << (8 * i) for i in range(4))   # [n][blocks][8]
            ls = (aux >> 28).astype(np.float32)
            s0 = s1 = ((d * (np.float32(0.5) + ls)).astype(np.float32) * np.float32(0.25 if type_id == 16 else 0.5)
                       ).astype(np.float32)
            signs = _even_signs((aux[..., None] >> (7 * np.arange(4))) & 127)                # [n][blocks][8][4][8]
            if type_id == 16:
                index = blocks[..., 2:66].reshape(n, -1, 8, 8)[..., :4].astype(np.int64)
                vals = grid("iq2_xxs")[index]                                                # [n][blocks][8][4][8]
            else:
                index = blocks[..., 2:66].reshape(n, -1, 8, 8).astype(np.int64)
                vals = grid("iq3_xxs")[index].reshape(n, -1, 8, 4, 8)
        else:
            scales = blocks[..., 66 if type_id == 17 else 74:][..., :8].astype(np.int64)
            s0 = ((d * (np.float32(0.5) + (scales & 15).astype(np.float32))).astype(np.float32) * np.float32(0.25)
                  ).astype(np.float32)
            s1 = ((d * (np.float32(0.5) + (scales >> 4).astype(np.float32))).astype(np.float32) * np.float32(0.25)
                  ).astype(np.float32)
            if type_id == 17:
                entry = blocks[..., 2:66].reshape(n, -1, 8, 4, 2).astype(np.int64)
                entry = entry[..., 0] | (entry[..., 1] << 8)                                 # [n][blocks][8][4]
                vals = grid("iq2_xs")[entry & 511]
                signs = _even_signs(entry >> 9)
            else:
                qh = blocks[..., 66:74].astype(np.int64)
                index = blocks[..., 2:34].reshape(n, -1, 8, 4).astype(np.int64) | (
                    ((qh[..., None] >> (2 * np.arange(4))) & 3) << 8)
                vals = grid("iq2_s")[index]
                sb = blocks[..., 34:66].reshape(n, -1, 8, 4).astype(np.int64)
                signs = (sb[..., None] >> np.arange(8)) & 1
        q = np.where(signs == 1, -vals, vals).astype(np.int8).reshape(n, -1, 8, 32)
        m = zero
    elif type_id in (19, 29):
        # Codes 8 g +- 1 (the offset of an eighth), scales d l / 8: exact, so s q is d l (g +- 1/8) rounded once.
        j = np.arange(8)
        if type_id == 19:
            d = _f16(blocks, 0)[..., None]
            qh = _u16(blocks, 34 + 2 * j)                                                     # [n][blocks][8]
            dl = (d * (2 * ((qh >> 12) & 7) + 1).astype(np.float32)).astype(np.float32)
            s0 = s1 = (dl * np.float32(0.125)).astype(np.float32)
            index = blocks[..., 2:34].reshape(n, -1, 8, 4).astype(np.int64) | (
                ((qh[..., None] >> (3 * np.arange(4))) & 7) << 8)
            negative = np.broadcast_to(((qh >> 15) & 1)[..., None], index.shape)
        else:
            w = [_u16(blocks, 48 + 2 * u) for u in range(4)]                                 # [n][blocks]
            dbits = (w[0] >> 12) | ((w[1] >> 8) & 0xf0) | ((w[2] >> 4) & 0xf00) | (w[3] & 0xf000)
            d = dbits.astype(np.uint16).view(np.float16).astype(np.float32)[..., None]
            word = np.stack(w, axis=-1)[..., j // 2]                                          # [n][blocks][8]
            shift = 6 * (j % 2)
            s0 = ((d * (2 * ((word >> shift) & 7) + 1).astype(np.float32)).astype(np.float32) * np.float32(0.125)
                  ).astype(np.float32)
            s1 = ((d * (2 * ((word >> (shift + 3)) & 7) + 1).astype(np.float32)).astype(np.float32) * np.float32(0.125)
                  ).astype(np.float32)
            qh = blocks[..., 32:48].reshape(n, -1, 8, 2).astype(np.int64)
            nib = np.stack([qh & 15, qh >> 4], axis=-1).reshape(n, -1, 8, 4)                 # group l's nibble
            index = blocks[..., 0:32].reshape(n, -1, 8, 4).astype(np.int64) | ((nib & 7) << 8)
            negative = (nib >> 3) & 1
        vals = grid("iq1_s")[index]                                                           # [n][blocks][8][4][8]
        q = (8 * vals + np.where(negative[..., None] == 1, -1, 1)).astype(np.int8).reshape(n, -1, 8, 32)
        m = zero
    else:   # 21, IQ3_S
        d = _f16(blocks, 0)[..., None]
        table = iq3s_grid()
        qs = blocks[..., 2:66].reshape(n, -1, 8, 8).astype(np.int32)
        qh = blocks[..., 66:74].astype(np.int32)
        index = qs | (((qh[..., None] >> np.arange(8)) & 1) << 8)
        vals = table[index].reshape(n, -1, 8, 32).astype(np.int16)
        signs = blocks[..., 74:106].reshape(n, -1, 8, 4)
        bits = (signs[..., None] >> np.arange(8, dtype=np.uint8)) & 1     # byte b, bit i: value 8 b + i
        vals = np.where(bits.reshape(n, -1, 8, 32) == 1, -vals, vals)
        q = vals.astype(np.int8)
        ib = np.arange(8)
        scale = (blocks[..., 106:110][..., ib // 2].astype(np.int32) >> (4 * (ib % 2))) & 15
        s0 = s1 = (d * (1 + 2 * scale).astype(np.float32)).astype(np.float32)
        m = zero

    def flat(a):
        return np.broadcast_to(a, (n, blocks.shape[1], per)).reshape(n, -1).astype(np.float32)
    if not halves:
        return q.reshape(n, -1, 32), flat(s0), flat(s1), flat(m)
    if m is not None:
        m0 = m1 = m
    return q.reshape(n, -1, 32), flat(s0), flat(s1), flat(m0), flat(m1)


def stored_values(raw, type_id, k):
    """W [N][K] FP32 from subblocks(): the stored values, bit for bit."""
    import numpy as np
    q, s0, s1, m0, m1 = subblocks(raw, type_id, k, halves=True)
    s = np.concatenate([np.repeat(s0[..., None], 16, -1), np.repeat(s1[..., None], 16, -1)], axis=-1)
    m = np.concatenate([np.repeat(m0[..., None], 16, -1), np.repeat(m1[..., None], 16, -1)], axis=-1)
    return ((s * q.astype(np.float32)).astype(np.float32) - m).astype(np.float32).reshape(raw.shape[0], k)


class GgufTensors:
    """Read-only access to an inspected GGUF model's tensor bytes (crit_core.model.GgufModel), each file mapped once.
    rows(name) is a numpy view uint8 [rows][row bytes] of a 2-D tensor ([experts][rows][row bytes] of a 3-D one, as
    GGUF stores expert stacks), without copying; info(name) its TensorInfo. close() drops the mappings (views handed
    out keep theirs)."""

    def __init__(self, model):
        self.model = model
        self._maps = {}

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def info(self, name):
        try:
            return self.model.tensors[name]
        except KeyError:
            raise KeyError(f"{self.model.source} holds no tensor {name!r}") from None

    def _map(self, path):
        if path not in self._maps:
            with open(path, "rb") as f:
                self._maps[path] = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
        return self._maps[path]

    def raw(self, name):
        """The tensor's bytes as a read-only uint8 numpy view."""
        import numpy as np
        info = self.info(name)
        view = self._map(info.shard)
        if info.data_end > len(view):
            raise GgufError(f"{info.shard}: tensor {name} ends at byte {info.data_end}, past the file's {len(view)}")
        return np.frombuffer(view, dtype=np.uint8, count=info.nbytes, offset=info.data_begin)

    def rows(self, name):
        """The tensor's bytes shaped as its rows: [rows][row bytes], or [experts][rows][row bytes]."""
        info = self.info(name)
        if len(info.shape) not in (2, 3):
            raise GgufError(f"{name} is {len(info.shape)}-D, not a matrix or a stack of them")
        raw = self.raw(name)
        return raw.reshape(*info.shape[:-1], info.nbytes // max(1, _count(info.shape[:-1])))

    def close(self):
        for view in self._maps.values():
            try:
                view.close()
            except BufferError:   # a view handed out still holds it; the mapping goes with the last one
                pass
        self._maps = {}


def _count(dims):
    out = 1
    for d in dims:
        out *= d
    return out


__all__ = ["BLOCKS", "GgufTensors", "HAS_MIN", "TWO_MINS", "TWO_SCALES", "TYPES", "grid", "iq3s_grid", "row_bytes", "stored_values",
           "subblocks"]
