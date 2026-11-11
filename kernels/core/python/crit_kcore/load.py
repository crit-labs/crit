"""crit's kernel core for Triton: tile scheduling and loads (kernels/core/README.md, "load")."""
import triton
import triton.language as tl


@triton.jit
def grouped_tile(pid, M, N, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, GROUP_M: tl.constexpr):
    """Program pid's output tile (tm, tn) of an [M][N] output in BLOCK_M x BLOCK_N tiles, grouped: GROUP_M row tiles
    walk the column tiles together, so the programs running at once share the operand rows and columns they read in
    L2. Which tile a program takes, never what an output computes."""
    tiles_m, tiles_n = tl.cdiv(M, BLOCK_M), tl.cdiv(N, BLOCK_N)
    group = pid // (GROUP_M * tiles_n)
    first = group * GROUP_M
    size = tl.minimum(tiles_m - first, GROUP_M)
    tm = first + (pid % (GROUP_M * tiles_n)) % size
    tn = (pid % (GROUP_M * tiles_n)) // size
    return tm, tn
