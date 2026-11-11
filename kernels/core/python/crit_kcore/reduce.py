"""crit's kernel core for Triton: reductions in a fixed order (kernels/core/README.md, "reduce").

tl.sum reduces in the order the tile's layout gives it (its warps, its threads' elements), so a sum's bits can change
when a tuning picks another tile or warp count. The trees here add element pairs explicitly, a shape-determined
order: the same bits whatever the layout, num_warps or tile (crit's fixed-order rule, kernels/TEMPLATES.md). They cost
log2(N) elementwise adds over shrinking tensors, a little more than tl.sum.
"""
import triton
import triton.language as tl


@triton.jit
def tree_sum(x, N: tl.constexpr):
    """The sum of x [N] (N a power of two, at most 2^16) by the pairwise tree of adjacent elements: ((x0 + x1) +
    (x2 + x3)) + ..., whatever the layout."""
    y = tl.reshape(x, (1, N))
    for _ in tl.static_range(16):
        if y.shape[1] > 1:
            a, b = tl.split(tl.reshape(y, (1, y.shape[1] // 2, 2)))
            y = a + b
    return tl.sum(tl.reshape(y, (1,)), axis=0)   # one element: no order


@triton.jit
def tree_sum_rows(x, ROWS: tl.constexpr, N: tl.constexpr):
    """Each row's sum of x [ROWS, N] (N a power of two, at most 2^16) by the pairwise tree of adjacent elements ->
    [ROWS]."""
    y = x
    for _ in tl.static_range(16):
        if y.shape[1] > 1:
            a, b = tl.split(tl.reshape(y, (ROWS, y.shape[1] // 2, 2)))
            y = a + b
    return tl.reshape(y, (ROWS,))


@triton.jit
def sum_partials(part_ptr, stride, offs, mask, PARTS: tl.constexpr):
    """part[0] + part[1] + ... + part[PARTS - 1] at offs (partials `stride` elements apart, e.g. split-K partial sums),
    added in part order in FP32."""
    acc = tl.load(part_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    for s in tl.static_range(1, PARTS):
        acc += tl.load(part_ptr + s * stride + offs, mask=mask, other=0.0).to(tl.float32)
    return acc
