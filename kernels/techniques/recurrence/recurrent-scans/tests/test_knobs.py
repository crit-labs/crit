"""Other knobs, the same bits: one build holding several configurations of each tunable structure
(crit_kernels.build.load(configs=...), switched with select()) against its defaults and against the default build.
pool-delta's rows give the same outputs, states, stashes and pending counts in every configuration (lanes of 4, 8 or 16
value columns), also when the configuration changes from one call to the next (a pending count left by one
block layout is consumed by another);
delta-prefill-rb's outputs and final state are the same values (a zero's sign aside, kernel.json); delta-prefill-tc's
the same bits in every configuration (its knobs regroup whole warps' work, stage the same loads, or overlap the two
passes in one launch)."""
import pytest
import torch

from crit_kernels import build as kbuild

PREFILL = [{"COLS": 16, "THREADS": 256, "MINB": 2}, {"COLS": 32, "THREADS": 128, "MINB": 0},
           {"COLS": 64, "THREADS": 1024, "MINB": 0}, {"COLS": 32, "THREADS": 512, "MINB": 2}]
TC = [{"WARPS": 1, "STAGES": 1, "OVERLAP": 0}, {"WARPS": 2, "STAGES": 2, "OVERLAP": 1},
      {"WARPS": 8, "STAGES": 2, "OVERLAP": 1}, {"WARPS": 4, "STAGES": 1, "OVERLAP": 0}]
POOL = [{"SLOTS": 4, "PREFETCH": 0, "MINB": 0}, {"SLOTS": 2, "PREFETCH": 1, "MINB": 3},
        {"SLOTS": 1, "PREFETCH": 0, "MINB": 6}, {"SLOTS": 1, "PREFETCH": 1, "MINB": 0, "LANE_COLS": 8},
        {"SLOTS": 2, "PREFETCH": 1, "MINB": 0, "LANE_COLS": 16}, {"SLOTS": 4, "PREFETCH": 0, "MINB": 3, "LANE_COLS": 8},
        {"SLOTS": 1, "PREFETCH": 0, "MINB": 6, "LANE_COLS": 16}]
PIPE = [{"LANE_COLS": 4, "ITEMS": 1, "STAGES": 1, "MINB": 0}, {"LANE_COLS": 4, "ITEMS": 2, "STAGES": 1, "MINB": 4},
        {"LANE_COLS": 4, "ITEMS": 4, "STAGES": 2, "MINB": 0}, {"LANE_COLS": 16, "ITEMS": 8, "STAGES": 2, "MINB": 2}]
rs = kbuild.load("recurrent-scans")
kn = kbuild.load("recurrent-scans", configs={"delta-prefill-rb": PREFILL, "delta-prefill-tc": TC, "pool-delta": POOL,
                                             "pool-delta-pipe": PIPE})
dev = torch.device("cuda")


def same(x, y):
    return torch.equal(x + 0.0, y + 0.0)   # + 0.0: -0 and +0 alike


def prefill_inputs(T, kh, vh, dv, seed):
    g = torch.Generator(device="cpu").manual_seed(seed)
    q = torch.nn.functional.normalize(torch.randn(T, kh, 128, generator=g), dim=-1)
    k = torch.nn.functional.normalize(torch.randn(T, kh, 128, generator=g), dim=-1)
    v = torch.randn(T, vh, dv, generator=g)
    logd = -torch.rand(T, vh, generator=g) * 0.2
    beta = torch.rand(T, vh, generator=g)
    s0 = torch.randn(vh, 128, dv, generator=g) * 0.1
    return [t.to(dev) for t in (q, k, v, logd, beta, s0)]


def prefill(module, x):
    q, k, v, logd, beta, s0 = x
    state = s0.clone()
    return module.gated_delta_prefill(q, k, v, logd, beta, state, scale=0.25), state


@pytest.mark.kernels_ci
@pytest.mark.parametrize("T", [37, 200])
def test_prefill_configurations_give_the_default_values(T):
    x = prefill_inputs(T, 2, 4, 128, seed=T)
    kbuild.select(kn, "delta-prefill-rb", 0)
    base = prefill(kn, x)
    assert all(torch.equal(a, b) for a, b in zip(base, prefill(rs, x)))   # the configurations build's defaults
    for i in range(1, len(PREFILL) + 1):
        kbuild.select(kn, "delta-prefill-rb", i)
        found = prefill(kn, x)
        assert same(found[0], base[0]) and same(found[1], base[1]), PREFILL[i - 1]
    kbuild.select(kn, "delta-prefill-rb", -1)


@pytest.mark.kernels_ci
@pytest.mark.parametrize("T,dv", [(37, 128), (700, 80)])
def test_tensor_core_prefill_configurations_give_the_default_bits(T, dv):
    x = prefill_inputs(T, 2, 6, dv, seed=T)

    def run(module):
        q, k, v, logd, beta, s0 = x
        state = s0.clone()
        return module.gated_delta_prefill(q, k, v, logd, beta, state, scale=128 ** -0.5, tensor_cores=True), state
    kbuild.select(kn, "delta-prefill-tc", 0)
    base = run(kn)
    assert all(torch.equal(a, b) for a, b in zip(base, run(rs)))
    for i in range(1, len(TC) + 1):
        kbuild.select(kn, "delta-prefill-tc", i)
        found = run(kn)
        assert torch.equal(found[0], base[0]) and torch.equal(found[1], base[1]), TC[i - 1]
    kbuild.select(kn, "delta-prefill-tc", -1)


def pool_inputs(T, kh, vh, dk, dv, seed):
    g = torch.Generator(device="cpu").manual_seed(seed)
    q, k = (torch.randn(T, kh, dk, generator=g).bfloat16() for _ in range(2))
    v = torch.randn(T, vh, dv, generator=g).bfloat16()
    logd = -torch.rand(T, vh, generator=g) * 0.5
    beta = torch.rand(T, vh, generator=g)
    return [t.to(dev) for t in (q, k, v, logd, beta)]


def pool_run(module, shape, schedule, pipelined=lambda step: False):
    """Two sequences through verify rows of 3 tokens, the deferred commit of 2 and 1 of them, again, then a flush;
    schedule(step) is the configuration index of each of the two pool_delta calls, of pool-delta-pipe where
    pipelined(step). Returns the outputs and the pool."""
    kh, vh, dk, dv = shape
    capacity = 3
    states = torch.randn(1, 2, vh, dk, dv, generator=torch.Generator().manual_seed(1)).mul_(0.1).to(dev)
    stash = torch.zeros(1, 2, 2 * module.pool_half_floats(kh, vh, dk, dv, capacity), device=dev)
    blocks = module.pool_blocks(vh, dv)
    control = torch.zeros(1, 2, 2 * blocks, dtype=torch.int32, device=dev)
    q, k, v, g, b = pool_inputs(12, kh, vh, dk, dv, seed=dk + dv)
    outs = []
    for step in range(2):
        kbuild.select(module, "pool-delta-pipe" if pipelined(step) else "pool-delta", schedule(step))
        rows = slice(6 * step, 6 * step + 6)
        meta = torch.tensor([[0, 3, 0, 0, -1], [3, 3, 1, 0, -1]], dtype=torch.int32, device=dev)
        out = torch.zeros(6, vh, dv, device=dev)
        module.pool_delta(out, states, stash, control, None, q[rows], k[rows], v[rows], g[rows], b[rows], meta=meta, rows=2,
                          kind=module.STASH, layer=0, capacity=capacity, scale=dk ** -0.5, pipelined=pipelined(step))
        outs.append(out)
        control[0, 0, :blocks], control[0, 1, :blocks] = 2, 1
    kbuild.select(module, "pool-delta", -1)
    kbuild.select(module, "pool-delta-pipe", -1)
    meta = torch.tensor([[0, 0, 0, 4, -1], [0, 0, 1, 4, -1]], dtype=torch.int32, device=dev)
    module.pool_delta_replay(states, stash, control, None, None, meta=meta, rows=2, capacity=capacity, key_heads=kh,
                             mode=module.FLUSH_PENDING)
    return outs, states, stash, control


@pytest.mark.kernels_ci
@pytest.mark.parametrize("shape", [(16, 48, 128, 128), (2, 4, 128, 96), (1, 2, 64, 100)])
def test_pool_configurations_give_the_default_bits(shape):
    base = pool_run(kn, shape, lambda step: 0)
    plain = pool_run(rs, shape, lambda step: -1)
    assert all(torch.equal(a, b) for a, b in zip(base[0] + list(base[1:]), plain[0] + list(plain[1:])))
    n = len(POOL) + 1
    for schedule in [lambda step, i=i: i for i in range(1, n)] + [lambda step: step + 1, lambda step: n - 1 - step]:
        found = pool_run(kn, shape, schedule)
        assert all(torch.equal(a, b) for a, b in zip(found[0] + list(found[1:]), base[0] + list(base[1:])))


@pytest.mark.kernels_ci
@pytest.mark.parametrize("shape", [(16, 48, 128, 128), (2, 4, 128, 96), (1, 2, 64, 100), (1, 3, 200, 70)])
def test_pipelined_rows_give_the_default_bits(shape):
    """pool-delta-pipe in every configuration, alone and taking over from pool-delta (and back) between calls."""
    base = pool_run(kn, shape, lambda step: 0)
    n = len(PIPE) + 1
    for i in range(n):
        for pipelined in (lambda step: True, lambda step: step == 1, lambda step: step == 0):
            found = pool_run(kn, shape, lambda step, i=i: i, pipelined)
            assert all(torch.equal(a, b) for a, b in zip(found[0] + list(found[1:]), base[0] + list(base[1:]))), (i, PIPE[i - 1] if i else "defaults")
