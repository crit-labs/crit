"""Host model of the stored tiles and of how the GEMM kernels turn tile bytes into MMA fragments.

Pure Python, no device. It packs small matrices exactly as `repack_fp8` / `repack_fp4` and the activation
packers do, assembles each lane's registers exactly as the kernels do (byte permutes, the lane t <-> t ^ 2 word
exchange), and evaluates the MMA instructions from their fragment definitions. `check_fp8` and `check_fp4`
compare that result with a direct dot product: they hold exactly when the k permutation, the exchange and the
packed activation order agree.

Fragment definitions (PTX ISA, row.col, lane l = 4 g + t):
  m16n8k32 e4m3:     a0 = (row g, k 4t..4t+3), a1 = (row g+8, same), a2 = (row g, k 16+4t..), a3 = (row g+8, same);
                     b0 = (k 4t..4t+3, col g), b1 = (k 16+4t.., col g).
  m16n8k64 e2m1:     a0 = (row g, k 8t..8t+7), a1 = (row g+8, same), a2 = (row g, k 32+8t..), a3 = (row g+8, same);
                     b0 = (k 8t..8t+7, col g), b1 = (k 32+8t.., col g); nibble i of a register is element i.
  block scales (scale_vec::4X, selectors 0): byte j of a scale word covers k 16j..16j+15; row g's word comes from
                     lane 4g, row g+8's from lane 4g+1, column g's from lane 4g.
  accumulators:      c0 = (row g, col 2t), c1 = (row g, 2t+1), c2 = (row g+8, 2t), c3 = (row g+8, 2t+1).
The scale-operand mapping is the part documentation states least clearly; `fragment_probe.cu` measures it on a
device (one-hot scale probes per lane and byte) and checks the value layout against a host reference.
"""
import random

E2M1 = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)


def e4m3(code):
    """Value of an E4M3FN byte (no infinities; 0x7F/0xFF are NaN)."""
    sign = -1.0 if code & 0x80 else 1.0
    exponent, mantissa = (code >> 3) & 0x0F, code & 0x07
    if exponent == 0x0F and mantissa == 0x07:
        return float("nan")
    if exponent == 0:
        return sign * mantissa / 8.0 * 2.0 ** -6
    return sign * (1.0 + mantissa / 8.0) * 2.0 ** (exponent - 7)


def e2m1(code):
    return (-1.0 if code & 8 else 1.0) * E2M1[code & 7]


def byte_perm(x, y, selector):
    """CUDA __byte_perm without sign replication: result byte k = byte (selector >> 4k) & 7 of (y:x)."""
    pool = x.to_bytes(4, "little") + y.to_bytes(4, "little")
    return int.from_bytes(bytes(pool[(selector >> (4 * k)) & 7] for k in range(4)), "little")


def words(data):
    """Four little-endian 32-bit words of a lane's 16 tile bytes (the kernel's uint4)."""
    return [int.from_bytes(bytes(data[4 * i:4 * i + 4]), "little") for i in range(4)]


# ----------------------------------------------------------------------------------------------- FP8
def fp8_tile(codes, row0=0, col0=0):
    """Lane bytes of the 16 x 32 tile at (row0, col0) of a row-major code matrix, as repack_fp8 writes them."""
    lanes = []
    for lane in range(32):
        g, t = lane >> 2, lane & 3
        data = []
        for s in range(2):
            c = col0 + 16 * s + 2 * t
            a, b = codes[row0 + g], codes[row0 + g + 8]
            data += [a[c], a[c + 1], b[c], b[c + 1], a[c + 8], a[c + 9], b[c + 8], b[c + 9]]
        lanes.append(data)
    return lanes


def fp8_slot_column(t, i):
    """Tile column (within a 16-column half) held by MMA k slot 4 t + i."""
    return 2 * t + (i if i < 2 else 6 + i)


def fp8_a_fragments(lane_bytes, selectors=(0x5410, 0x7632)):
    """The kernel's register assembly: a0/a1 from words 0-1, a2/a3 from words 2-3 with byte permutes."""
    w = words(lane_bytes)
    low, high = selectors
    return [byte_perm(w[0], w[1], low), byte_perm(w[0], w[1], high),
            byte_perm(w[2], w[3], low), byte_perm(w[2], w[3], high)]


def fp8_b_fragments(activation_codes, token, col0=0):
    """b0/b1 of lane (token, t) for t = 0..3: columns {2t, 2t+1, 2t+8, 2t+9} of each 16-column half."""
    out = []
    for t in range(4):
        regs = []
        for h in range(2):
            c = col0 + 16 * h + 2 * t
            regs.append(int.from_bytes(bytes(activation_codes[token][c + d] for d in (0, 1, 8, 9)), "little"))
        out.append(regs)
    return out


def mma_e4m3(a_by_lane, b_by_lane):
    """D[16][8] of mma.m16n8k32 e4m3 from per-lane registers (a: 4 words, b: 2 words)."""
    A = [[0.0] * 32 for _ in range(16)]
    B = [[0.0] * 8 for _ in range(32)]
    for lane in range(32):
        g, t = lane >> 2, lane & 3
        for reg, (row, k0) in enumerate(((g, 4 * t), (g + 8, 4 * t), (g, 16 + 4 * t), (g + 8, 16 + 4 * t))):
            for i in range(4):
                A[row][k0 + i] = e4m3((a_by_lane[lane][reg] >> (8 * i)) & 0xFF)
        for reg, k0 in enumerate((4 * t, 16 + 4 * t)):
            for i in range(4):
                B[k0 + i][g] = e4m3((b_by_lane[lane][reg] >> (8 * i)) & 0xFF)
    return [[sum(A[r][k] * B[k][c] for k in range(32)) for c in range(8)] for r in range(16)]


def check_fp8(seed=0, selectors=(0x5410, 0x7632)):
    """Pack a random 16 x 32 weight tile and 8 tokens, evaluate the kernel's fragments, compare with dot products.
    Returns the largest absolute difference (0.0 when the layout is consistent)."""
    rng = random.Random(seed)
    finite = [c for c in range(256) if (c & 0x7F) != 0x7F]
    W = [[rng.choice(finite) for _ in range(32)] for _ in range(16)]
    X = [[rng.choice(finite) for _ in range(32)] for _ in range(8)]
    tile = fp8_tile(W)
    a = [fp8_a_fragments(tile[lane], selectors) for lane in range(32)]
    b = [fp8_b_fragments(X, lane >> 2)[lane & 3] for lane in range(32)]
    D = mma_e4m3(a, b)
    worst = 0.0
    for r in range(16):
        for c in range(8):
            want = sum(e4m3(W[r][k]) * e4m3(X[c][k]) for k in range(32))
            worst = max(worst, abs(D[r][c] - want))
    return worst


# ----------------------------------------------------------------------------------------------- NVFP4
def fp4_tile(codes, row0=0, col0=0):
    """Lane bytes of the 16 x 64 tile at (row0, col0) of a code matrix (one code per element), packed as
    repack_fp4 writes them: low nibble = even column."""
    def pair(row, c):
        return codes[row][c] | (codes[row][c + 1] << 4)
    lanes = []
    for lane in range(32):
        g, t = lane >> 2, lane & 3
        data = []
        for s in range(4):
            c = col0 + 16 * s + 2 * t
            data += [pair(row0 + g, c), pair(row0 + g + 8, c), pair(row0 + g, c + 8), pair(row0 + g + 8, c + 8)]
        lanes.append(data)
    return lanes


def fp4_scale_tile(scales, row0=0, group0=0):
    """64 scale bytes of a tile: [g][row g: groups 0..3][row g + 8: groups 0..3]."""
    out = []
    for g in range(8):
        out += [scales[row0 + g][group0 + s] for s in range(4)] + [scales[row0 + g + 8][group0 + s] for s in range(4)]
    return out


def fp4_mma_column(t, r, i):
    """Tile column held by slot (lane t, register pair r, nibble i) after the exchange."""
    s, tp, ii = (t >> 1) + 2 * r, (t if i < 4 else t ^ 2), i & 3
    return 16 * s + 2 * tp + (ii if ii < 2 else 6 + ii)


def fp4_a_fragments(tile, partner_xor=2):
    """The kernel's register assembly for all 32 lanes: own words for groups t >> 1 and t >> 1 + 2, the partner's
    (lane ^ 2) words for the same groups, merged with byte permutes."""
    w = [words(tile[lane]) for lane in range(32)]
    out = []
    for lane in range(32):
        hi = (lane & 3) >> 1
        own0, own2 = (w[lane][1], w[lane][3]) if hi else (w[lane][0], w[lane][2])
        partner = lane ^ partner_xor
        phi = (partner & 3) >> 1
        got0, got2 = (w[partner][0], w[partner][2]) if phi else (w[partner][1], w[partner][3])
        out.append([byte_perm(own0, got0, 0x6420), byte_perm(own0, got0, 0x7531),
                    byte_perm(own2, got2, 0x6420), byte_perm(own2, got2, 0x7531)])
    return out


def mma_nvfp4(a_by_lane, b_by_lane, sa_by_lane, sb_by_lane):
    """D[16][8] of mma.m16n8k64 kind::mxf4nvf4 scale_vec::4X from per-lane registers and scale words."""
    A = [[0.0] * 64 for _ in range(16)]
    B = [[0.0] * 8 for _ in range(64)]
    for lane in range(32):
        g, t = lane >> 2, lane & 3
        for reg, (row, k0) in enumerate(((g, 8 * t), (g + 8, 8 * t), (g, 32 + 8 * t), (g + 8, 32 + 8 * t))):
            for i in range(8):
                A[row][k0 + i] = e2m1((a_by_lane[lane][reg] >> (4 * i)) & 0xF)
        for reg, k0 in enumerate((8 * t, 32 + 8 * t)):
            for i in range(8):
                B[k0 + i][g] = e2m1((b_by_lane[lane][reg] >> (4 * i)) & 0xF)
    row_scale = [[e4m3((sa_by_lane[4 * (r % 8) + (r >= 8)] >> (8 * j)) & 0xFF) for j in range(4)] for r in range(16)]
    col_scale = [[e4m3((sb_by_lane[4 * c] >> (8 * j)) & 0xFF) for j in range(4)] for c in range(8)]
    return [[sum(A[r][k] * B[k][c] * row_scale[r][k // 16] * col_scale[c][k // 16] for k in range(64))
             for c in range(8)] for r in range(16)]


def check_fp4(seed=0, partner_xor=2):
    """Pack a random 16 x 64 NVFP4 tile with its scales and 8 tokens of NVFP4 activations, evaluate the kernel's
    fragments, compare with dot products. Returns the largest absolute difference (0.0 when consistent)."""
    rng = random.Random(seed)
    W = [[rng.randrange(16) for _ in range(64)] for _ in range(16)]
    WS = [[rng.randrange(0x30, 0x48) for _ in range(4)] for _ in range(16)]
    X = [[rng.randrange(16) for _ in range(64)] for _ in range(8)]
    XS = [[rng.randrange(0x30, 0x48) for _ in range(4)] for _ in range(8)]
    tile, scale_tile = fp4_tile(W), fp4_scale_tile(WS)
    a = fp4_a_fragments(tile, partner_xor)
    # The kernel loads 4 scale bytes at g * 8 + (t & 1) * 4: row g (t even) or row g + 8 (t odd).
    sa = [int.from_bytes(bytes(scale_tile[(lane >> 2) * 8 + (lane & 1) * 4:(lane >> 2) * 8 + (lane & 1) * 4 + 4]), "little")
          for lane in range(32)]
    b, sb = [], []
    for lane in range(32):
        g, t = lane >> 2, lane & 3
        regs = []
        for r in range(2):
            word = 0
            for i in range(8):
                word |= X[g][fp4_mma_column(t, r, i)] << (4 * i)
            regs.append(word)
        b.append(regs)
        sb.append(int.from_bytes(bytes(XS[g]), "little"))
    D = mma_nvfp4(a, b, sa, sb)
    worst = 0.0
    for r in range(16):
        for c in range(8):
            want = sum(e2m1(W[r][k]) * e4m3(WS[r][k // 16]) * e2m1(X[c][k]) * e4m3(XS[c][k // 16]) for k in range(64))
            worst = max(worst, abs(D[r][c] - want))
    return worst


def fp4_activation_columns():
    """Every lane/register/nibble slot's column; the slots cover each tile column exactly once per token."""
    return {(t, r, i): fp4_mma_column(t, r, i) for t in range(4) for r in range(2) for i in range(8)}
