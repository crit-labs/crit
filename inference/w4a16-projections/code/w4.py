"""Host reference for published 4-bit weight formats with BF16 activations (W4A16): their exact unpacking, the one
canonical layout the kernels read, and the projection.

Canonical layout, for an N x K weight (N output rows, K inputs):
    codes   [N][K / 2] bytes, the even column in the low nibble
    int4 groups: value = (q - zero) * scale with scale and zero per row and group of `group` columns
    MXFP4:       value = e2m1(q) * 2^(e - 127) with one E8M0 exponent e per row and 32 columns
    perm    optional [K]: canonical column j is the checkpoint's column perm[j] (GPTQ act-order), so activations are
            gathered x'[j] = x[perm[j]] before the projection and every group is contiguous

Formats (as their reference packers write them):
    GPTQ           qweight int32 [K / 8][N], column k at nibble k % 8; qzeros int32 [G][N / 8], row n at nibble n % 8;
                   scales [G][N]; g_idx [K] the group of each input column (act-order), else k // group.
                   checkpoint_format "gptq" stores zero - 1 (zero_offset 1); "gptq_v2" stores the zero.
    AWQ            qweight int32 [K][N / 8] and qzeros int32 [G][N / 8], packed along N in the order
                   (0, 2, 4, 6, 1, 3, 5, 7): nibble i holds column 8j + ORDER[i]; scales [G][N].
    compressed-tensors pack-quantized
                   weight_packed int32 [N][K / 8], column k at nibble k % 8, unsigned codes; weight_scale [N][G];
                   symmetric (zero 8) unless weight_zero_point [N][G] is given.
    MXFP4 (gpt-oss) blocks uint8 [N][K / 32][16], the even element in the low nibble; scales uint8 [N][K / 32] (E8M0).
"""
E2M1 = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0)
AWQ_ORDER = (0, 2, 4, 6, 1, 3, 5, 7)


def nibble(word, i):
    return (word >> (4 * i)) & 0xF


def from_gptq(qweight, qzeros, scales, K, N, group, g_idx=None, zero_offset=1):
    """(codes[N][K], scales[N][K], zeros[N][K], perm) with per-element group values, canonical column order."""
    g_idx = list(g_idx) if g_idx is not None else [k // group for k in range(K)]
    perm = sorted(range(K), key=lambda k: (g_idx[k], k))
    q = [[nibble(qweight[k // 8][n], k % 8) for k in range(K)] for n in range(N)]
    z = [[nibble(qzeros[g][n // 8], n % 8) + zero_offset for g in range(max(g_idx) + 1)] for n in range(N)]
    s = [[float(scales[g][n]) for g in range(max(g_idx) + 1)] for n in range(N)]
    codes = [[q[n][perm[j]] for j in range(K)] for n in range(N)]
    groups = [g_idx[perm[j]] for j in range(K)]
    return codes, [[s[n][groups[j]] for j in range(K)] for n in range(N)], [[z[n][groups[j]] for j in range(K)] for n in range(N)], perm


def from_awq(qweight, qzeros, scales, K, N, group):
    unpack = lambda word, c: nibble(word, AWQ_ORDER.index(c % 8))  # noqa: E731
    codes = [[unpack(qweight[k][n // 8], n) for k in range(K)] for n in range(N)]
    zeros = [[unpack(qzeros[k // group][n // 8], n) for k in range(K)] for n in range(N)]
    s = [[float(scales[k // group][n]) for k in range(K)] for n in range(N)]
    return codes, s, zeros, list(range(K))


def from_compressed(packed, scale, K, N, group, zero_point=None):
    codes = [[nibble(packed[n][k // 8], k % 8) for k in range(K)] for n in range(N)]
    zeros = [[(zero_point[n][k // group] if zero_point else 8) for k in range(K)] for n in range(N)]
    return codes, [[float(scale[n][k // group]) for k in range(K)] for n in range(N)], zeros, list(range(K))


def int4_weights(codes, scales, zeros):
    """The represented weight matrix: (q - zero) * scale."""
    return [[(q - z) * s for q, s, z in zip(rq, rs, rz)] for rq, rs, rz in zip(codes, scales, zeros)]


def mxfp4_weights(blocks, exponents, K, N):
    """MXFP4: e2m1(code) * 2^(e - 127), 32 values per exponent."""
    out = []
    for n in range(N):
        row = []
        for k in range(K):
            byte = blocks[n][k // 32][(k % 32) // 2]
            code = byte & 0xF if k % 2 == 0 else byte >> 4
            row.append(E2M1[code] * 2.0 ** (exponents[n][k // 32] - 127))
        out.append(row)
    return out


def pack_canonical(codes):
    """codes[N][K] -> bytes [N][K / 2], the even column in the low nibble."""
    return [bytes(r[k] | (r[k + 1] << 4) for k in range(0, len(r), 2)) for r in codes]


def project(weights, x, perm=None):
    """y[n] = sum_j w[n][j] x[perm[j]]: the projection over canonical columns."""
    xs = [x[p] for p in perm] if perm else list(x)
    return [sum(w * v for w, v in zip(row, xs)) for row in weights]
