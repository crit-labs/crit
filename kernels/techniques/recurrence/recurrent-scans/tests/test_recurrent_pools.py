"""The pool kernels (pool_delta, pool_delta_replay, pool_conv, conv_segments, pool_conv_prompt) against a float64 recurrence:
rows of many sequences in one launch, each from its own state; a row's bits independent of the rows beside it; the three
commit strategies bit for bit the same as decoding the accepted tokens one by one, for every accepted count; and
capture in a CUDA graph."""
import pytest
import torch

from crit_kernels import build as kbuild

rs = kbuild.load("recurrent-scans")   # built for this GPU with its default knobs (cache: CRIT_KERNEL_CACHE)
dev = torch.device("cuda")


@pytest.fixture(autouse=True, params=["pool-delta", "pool-delta-pipe"])
def structure(request, monkeypatch):
    """Every pool_delta test on both structures (pool-delta-pipe: pool_delta(pipelined=True), the same bits); the
    convolution tests once."""
    if request.param == "pool-delta-pipe":
        if "convol" in request.node.name:
            pytest.skip("no pool_delta call")
        plain = rs.pool_delta
        monkeypatch.setattr(rs, "pool_delta", lambda *a, **k: plain(*a, **dict(k, pipelined=True)))
    return request.param


def normal(x, eps):
    return x / torch.sqrt((x * x).sum(-1, keepdim=True) + eps)


def reference(q, k, v, g, b, state, scale, eps=1e-6, normalize=(True, True)):
    """float64 recurrence over raw q, k [T][kh][dk] (normalized here where asked), v [T][vh][dv]; state [vh][dk][dv]."""
    k = normal(k.double(), eps) if normalize[0] else k.double()
    q = normal(q.double(), eps) if normalize[1] else q.double()
    v, g, b, S = v.double(), g.double(), b.double(), state.double().clone()
    vh, kh = v.shape[1], q.shape[1]
    out = torch.empty(v.shape, dtype=torch.float64, device=v.device)
    for t in range(v.shape[0]):
        for h in range(vh):
            j = h // (vh // kh)
            e = torch.exp(g[t, h])
            delta = b[t, h] * (v[t, h] - e * S[h].T @ k[t, j])
            S[h] = e * S[h] + torch.outer(k[t, j], delta)
            out[t, h] = scale * S[h].T @ q[t, j]
    return out, S


class Pool:
    def __init__(self, layers, sequences, kh, vh, dk, dv, capacity, slots=8):
        self.shape = (kh, vh, dk, dv)
        self.capacity = capacity
        self.states = torch.zeros(layers, sequences, vh, dk, dv, device=dev)
        half = rs.pool_half_floats(kh, vh, dk, dv, capacity)
        self.stash = torch.zeros(layers, sequences, 2 * half, device=dev)
        self.control = torch.zeros(layers, sequences, 2 * rs.pool_blocks(vh, dv), dtype=torch.int32, device=dev)
        self.checkpoints = torch.zeros(layers, slots, capacity + 1, vh * dk * dv, device=dev)
        self.starts = torch.zeros(layers, slots, vh * dk * dv, device=dev)
        self.table = torch.full((slots,), -1, dtype=torch.int32, device=dev)

    def rows(self, layer, q, k, v, g, b, rows, scale, scratch=None, epsilon=1e-6, normalize=(True, True)):
        """rows: (first, tokens, sequence, kind[, scratch slot])."""
        meta = torch.tensor([list(r) + [-1] * (5 - len(r)) for r in rows], dtype=torch.int32, device=dev)
        out = torch.zeros(v.shape, device=dev)
        rs.pool_delta(out, self.states, self.stash, self.control, scratch, q, k, v, g, b, meta=meta, rows=len(rows),
                      kind=0, layer=layer, capacity=self.capacity, epsilon=epsilon, scale=scale,
                      normalize_keys=normalize[0], normalize_queries=normalize[1])
        return out

    def record(self, accepted, sequences):
        """The deferred commit (the caller's bookkeeping): each sequence's pending counts."""
        blocks = rs.pool_blocks(self.shape[1], self.shape[3])
        for n, s in zip(accepted, sequences):
            self.control[:, s, :blocks] = n

    def flush(self, sequences):
        meta = torch.tensor([[0, 0, s, 4, -1] for s in sequences], dtype=torch.int32, device=dev)
        rs.pool_delta_replay(self.states, self.stash, self.control, None, None, meta=meta, rows=len(sequences),
                             capacity=self.capacity, key_heads=self.shape[0], mode=rs.FLUSH_PENDING)

    def commit(self, strategy, accepted, slots, columns):
        """A round's commit at accepted ([rows] int64 on the device), rows in round slots `slots`."""
        kh, vh, dk, dv = self.shape
        sequences = self.table[slots].long()
        if strategy == "deferred":
            self.control[:, sequences, :rs.pool_blocks(vh, dv)] = accepted.clamp(0, columns).int()[None, :, None]
        elif strategy == "checkpoints":
            chosen = self.checkpoints[:, slots, accepted.clamp(0, columns)]
            self.states[:, sequences] = chosen.view(self.states.shape[0], -1, *self.states.shape[2:])
        else:
            rs.pool_delta_replay(self.states, self.stash, self.control, self.starts, accepted, slots=slots,
                                 table=self.table, rows=len(slots), columns=columns, capacity=self.capacity,
                                 key_heads=kh, mode=rs.REPLAY)


def inputs(T, kh, vh, dk, dv, seed, dtype=torch.float32, strong=False):
    g = torch.Generator(device="cpu").manual_seed(seed)
    q = torch.randn(T, kh, dk, generator=g)
    k = torch.randn(T, kh, dk, generator=g)
    v = torch.randn(T, vh, dv, generator=g)
    logd = -(torch.rand(T, vh, generator=g) * 9 + 3) if strong else -torch.rand(T, vh, generator=g) * 0.5
    beta = torch.rand(T, vh, generator=g)
    return [t.to(dev) for t in (q.to(dtype), k.to(dtype), v.to(dtype), logd, beta)]


SHAPES = [(2, 4, 64, 64), (16, 48, 128, 128), (1, 2, 4, 3), (2, 2, 100, 40), (1, 1, 256, 32)]


@pytest.mark.kernels_ci
@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("strong", [False, True])
def test_rows_of_two_sequences_match_the_recurrence(shape, strong):
    kh, vh, dk, dv = shape
    pool = Pool(1, 3, kh, vh, dk, dv, capacity=4)
    q, k, v, g, b = inputs(7, kh, vh, dk, dv, seed=3, strong=strong)
    scale = dk ** -0.5
    out = pool.rows(0, q, k, v, g, b, [(0, 3, 0, 0), (3, 4, 2, 0)], scale)
    pool.record([3, 4], [0, 2])
    pool.flush([0, 2])
    for first, count, seq in ((0, 3, 0), (3, 4, 2)):
        rows = slice(first, first + count)
        ref_out, ref_state = reference(q[rows], k[rows], v[rows], g[rows], b[rows], torch.zeros(vh, dk, dv, device=dev), scale)
        torch.testing.assert_close(out[rows].double(), ref_out, rtol=1e-4, atol=1e-5)
        torch.testing.assert_close(pool.states[0, seq].double(), ref_state, rtol=1e-4, atol=1e-5)
    assert torch.count_nonzero(pool.states[0, 1]) == 0


@pytest.mark.kernels_ci
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_a_row_alone_has_the_bits_it_has_beside_others(dtype):
    kh, vh, dk, dv = 2, 6, 128, 64
    q, k, v, g, b = inputs(10, kh, vh, dk, dv, seed=5, dtype=dtype)
    together = Pool(1, 4, kh, vh, dk, dv, capacity=4)
    out = together.rows(0, q, k, v, g, b, [(0, 4, 0, 0), (4, 1, 1, 0), (5, 3, 3, 0), (8, 2, 2, 0)], 0.3)
    for first, count, seq in ((0, 4, 0), (4, 1, 1), (5, 3, 3), (8, 2, 2)):
        alone = Pool(1, 4, kh, vh, dk, dv, capacity=4)
        rows = slice(first, first + count)
        found = alone.rows(0, q[rows], k[rows], v[rows], g[rows], b[rows], [(0, count, seq, 0)], 0.3)
        assert torch.equal(found, out[rows])


def strategy_round(strategy, pool, q, k, v, g, b, n, slot=0, sequence=1):
    """One verify row of len(q) columns from sequence `sequence` in round slot `slot`, committed at n."""
    pool.table[slot] = sequence
    cols = q.shape[0]
    starts = torch.tensor([0, cols], dtype=torch.int32, device=dev)
    slots = torch.tensor([slot], dtype=torch.int64, device=dev)
    accepted = torch.tensor([n], dtype=torch.int64, device=dev)
    kind = {"deferred": rs.STASH, "checkpoints": rs.CHECKPOINTS, "recompute": rs.RECOMPUTE}[strategy]
    scratch = {"deferred": None, "checkpoints": pool.checkpoints[0], "recompute": pool.starts[0]}[strategy]
    out = torch.zeros(v.shape, device=dev)
    rs.pool_delta(out, pool.states, pool.stash, pool.control, scratch, q, k, v, g, b, starts=starts, slots=slots,
                  table=pool.table, rows=1, kind=kind, layer=0, capacity=pool.capacity, scale=0.25)
    pool.commit(strategy, accepted, slots, cols)
    return out


@pytest.mark.kernels_ci
@pytest.mark.parametrize("strategy", ["deferred", "checkpoints", "recompute"])
def test_every_accepted_count_continues_as_the_serial_decode(strategy):
    kh, vh, dk, dv, W = 2, 4, 128, 64, 4
    q, k, v, g, b = inputs(2 + (W + 1) + 3, kh, vh, dk, dv, seed=11)
    for n in range(W + 2):
        # Serial: tokens one by one, each accepted.
        serial = Pool(1, 2, kh, vh, dk, dv, capacity=W + 1)
        outs = []
        for t in range(2 + n + 3):
            row = slice(t, t + 1)
            outs.append(serial.rows(0, q[row], k[row], v[row], g[row], b[row], [(0, 1, 1, 0)], 0.25))
            serial.record([1], [1])
        serial.flush([1])
        # Speculative: 2 tokens, a verify window of W + 1 with n accepted, then the 3 tokens after the accepted ones.
        spec = Pool(1, 2, kh, vh, dk, dv, capacity=W + 1)
        for t in range(2):
            row = slice(t, t + 1)
            spec.rows(0, q[row], k[row], v[row], g[row], b[row], [(0, 1, 1, 0)], 0.25)
            spec.record([1], [1])
        window = slice(2, 2 + W + 1)
        verified = strategy_round(strategy, spec, q[window], k[window], v[window], g[window], b[window], n)
        for c in range(n):
            assert torch.equal(verified[c:c + 1], outs[2 + c]), (strategy, n, c)
        tail = [2 + n + i for i in range(3)]
        for i, t in enumerate(tail):
            # the continuation reads the inputs the serial decode read at that position
            row = slice(t, t + 1)
            found = spec.rows(0, q[row], k[row], v[row], g[row], b[row], [(0, 1, 1, 0)], 0.25)
            spec.record([1], [1])
            assert torch.equal(found, outs[t]), (strategy, n, i)
        spec.flush([1])
        assert torch.equal(spec.states, serial.states), (strategy, n)


@pytest.mark.kernels_ci
@pytest.mark.parametrize("strategy", ["deferred", "checkpoints", "recompute"])
def test_a_commit_runs_again_with_a_smaller_count(strategy):
    kh, vh, dk, dv = 1, 2, 64, 64
    q, k, v, g, b = inputs(5, kh, vh, dk, dv, seed=13)
    once, twice = (Pool(1, 2, kh, vh, dk, dv, capacity=5) for _ in range(2))
    strategy_round(strategy, once, q, k, v, g, b, 2)
    strategy_round(strategy, twice, q, k, v, g, b, 4)
    accepted = torch.tensor([2], dtype=torch.int64, device=dev)
    slots = torch.tensor([0], dtype=torch.int64, device=dev)
    twice.commit(strategy, accepted, slots, 5)
    once.flush([1])
    twice.flush([1])
    assert torch.equal(once.states, twice.states)


@pytest.mark.kernels_ci
def test_a_row_of_no_sequence_writes_zeros_and_touches_no_state():
    kh, vh, dk, dv = 2, 4, 64, 64
    pool = Pool(1, 3, kh, vh, dk, dv, capacity=4)
    pool.table[1] = 2   # slot 0 holds no sequence
    pool.states.normal_()
    before = pool.states.clone()
    q, k, v, g, b = inputs(8, kh, vh, dk, dv, seed=4)
    out = torch.full(v.shape, float("nan"), device=dev)
    rs.pool_delta(out, pool.states, pool.stash, pool.control, None, q, k, v, g, b,
                  starts=torch.tensor([0, 4, 8], dtype=torch.int32, device=dev),
                  slots=torch.arange(2, dtype=torch.int64, device=dev), table=pool.table, rows=2, kind=rs.STASH, layer=0,
                  capacity=4, scale=0.25)
    assert torch.equal(out[:4], torch.zeros_like(out[:4])) and torch.isfinite(out[4:]).all()
    assert torch.equal(pool.states, before)   # sequence 2 verified (stashed), nothing committed
    window = torch.randn(1, 3, 40, 3, device=dev)
    kept = window.clone()
    x = torch.randn(8, 40, device=dev)
    y = torch.full_like(x, float("nan"))
    rs.pool_conv(y, window, torch.zeros(1, 3, 4, 40, device=dev), torch.zeros(1, 3, 1, dtype=torch.int32, device=dev),
                 None, x, torch.randn(40, 4, device=dev), starts=torch.tensor([0, 4, 8], dtype=torch.int32, device=dev),
                 slots=torch.arange(2, dtype=torch.int64, device=dev), table=pool.table, rows=2, capacity=4)
    assert torch.equal(y[:4], torch.zeros_like(y[:4])) and torch.isfinite(y[4:]).all()
    assert torch.equal(window, kept)


@pytest.mark.kernels_ci
def test_a_round_captures_and_replays():
    kh, vh, dk, dv, W, R = 2, 4, 64, 64, 3, 2
    pool = Pool(1, 4, kh, vh, dk, dv, capacity=W + 1)
    pool.table[0], pool.table[1] = 3, 1
    q, k, v, g, b = (torch.zeros_like(t) for t in inputs(R * (W + 1), kh, vh, dk, dv, seed=1))
    starts = torch.arange(0, R * (W + 1) + 1, W + 1, dtype=torch.int32, device=dev)
    slots = torch.arange(R, dtype=torch.int64, device=dev)
    accepted = torch.zeros(R, dtype=torch.int64, device=dev)
    out = torch.zeros(v.shape, device=dev)

    def round_():
        rs.pool_delta(out, pool.states, pool.stash, pool.control, None, q, k, v, g, b, starts=starts, slots=slots,
                      table=pool.table, rows=R, kind=rs.STASH, layer=0, capacity=W + 1, scale=0.25)
        pool.commit("deferred", accepted, slots, W + 1)
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        round_()
    torch.cuda.synchronize()
    pool.states.zero_()
    pool.control.zero_()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        round_()
    eager = Pool(1, 4, kh, vh, dk, dv, capacity=W + 1)
    eager.table.copy_(pool.table)
    for seed, counts in ((2, [2, 4]), (3, [0, 1]), (4, [4, 3])):
        new = inputs(R * (W + 1), kh, vh, dk, dv, seed=seed)
        for t, n in zip((q, k, v, g, b), new):
            t.copy_(n)
        accepted.copy_(torch.tensor(counts, device=dev))
        graph.replay()
        expected = torch.zeros(v.shape, device=dev)
        rs.pool_delta(expected, eager.states, eager.stash, eager.control, None, q, k, v, g, b, starts=starts, slots=slots,
                      table=eager.table, rows=R, kind=rs.STASH, layer=0, capacity=W + 1, scale=0.25)
        eager.commit("deferred", accepted, slots, W + 1)
        torch.cuda.synchronize()
        assert torch.equal(out, expected)
    assert torch.equal(pool.states, eager.states) and torch.equal(pool.control, eager.control)


def conv_reference(x, weight, bias, window, silu=True):
    padded = torch.cat([window.T.double(), x.double()])
    w = weight.shape[1]
    out = torch.stack([(padded[t:t + w].T * weight.double()).sum(1) + bias.double() for t in range(x.shape[0])])
    return torch.nn.functional.silu(out) if silu else out, padded[-(w - 1):].T


@pytest.mark.kernels_ci
@pytest.mark.parametrize("width", [2, 4, 8])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_pool_convolution_keeps_each_sequences_window(width, dtype):
    g = torch.Generator(device="cpu").manual_seed(width)
    C, cap = 300, 4
    x = torch.randn(9, C, generator=g).to(dev).to(dtype)
    weight = torch.randn(C, width, generator=g).to(dev)
    bias = torch.randn(C, generator=g).to(dev)
    window = torch.zeros(1, 3, C, width - 1, device=dev)
    window[0, 2] = torch.randn(C, width - 1, generator=g).to(dev)
    start = window[0, 2].clone()
    stash = torch.zeros(1, 3, cap, C, device=dev)
    pending = torch.zeros(1, 3, rs.pool_conv_blocks(C), dtype=torch.int32, device=dev)
    out = torch.zeros_like(x)
    meta = torch.tensor([[0, 3, 2, rs.STASH, -1], [3, 2, 0, rs.STASH, -1]], dtype=torch.int32, device=dev)
    rs.pool_conv(out, window, stash, pending, None, x, weight, bias, meta=meta, rows=2, capacity=cap)
    ref, ref_window = conv_reference(x[:3].float(), weight, bias, start)
    torch.testing.assert_close(out[:3].double(), ref.to(dtype).double(), rtol=1e-2 if dtype != torch.float32 else 1e-5,
                               atol=1e-2 if dtype != torch.float32 else 1e-5)
    assert torch.equal(window[0, 2], start)   # pending until committed
    pending[0, 2] = 2   # accept 2 of the 3 inputs
    flush = torch.tensor([[0, 0, 2, rs.FLUSH, -1]], dtype=torch.int32, device=dev)
    rs.pool_conv(out[:0], window, stash, pending, None, x[:0], weight, bias, meta=flush, rows=1, capacity=cap)
    _, expected = conv_reference(x[:2].float(), weight, bias, start)
    torch.testing.assert_close(window[0, 2].double(), expected, rtol=0, atol=0)
    # A long row through the segments kernel has the row kernel's bits.
    long = torch.randn(70, C, generator=g).to(dev).to(dtype)
    seg_out, seg_window = torch.empty_like(long), torch.empty(C, width - 1, device=dev)
    rs.conv_segments(seg_out, window[0, 2].clone(), seg_window, long, weight, bias)
    scratch = torch.zeros(1, C, width - 1, device=dev)
    row_out = torch.empty_like(long)
    meta = torch.tensor([[0, 70, 2, rs.FINAL, 0]], dtype=torch.int32, device=dev)
    rs.pool_conv(row_out, window, stash, pending, scratch, long, weight, bias, meta=meta, rows=1, capacity=cap)
    assert torch.equal(seg_out, row_out) and torch.equal(seg_window, scratch[0])
    # rows shorter than the window
    short = long[:1]
    s_out, s_window = torch.empty_like(short), torch.empty(C, width - 1, device=dev)
    rs.conv_segments(s_out, window[0, 2].clone(), s_window, short, weight, bias)
    _, expected = conv_reference(short.float(), weight, bias, window[0, 2])
    torch.testing.assert_close(s_window.double(), expected, rtol=0, atol=0)


@pytest.mark.kernels_ci
@pytest.mark.parametrize("variant", ["no-norm-keys", "no-norm-queries", "epsilon", "fp16"])
def test_geometry_variants_match_the_recurrence(variant):
    kh, vh, dk, dv = 2, 4, 64, 64
    normalize = (variant != "no-norm-keys", variant != "no-norm-queries")
    eps = 1e-3 if variant == "epsilon" else 1e-6
    dtype = torch.float16 if variant == "fp16" else torch.float32
    q, k, v, g, b = inputs(6, kh, vh, dk, dv, seed=21, dtype=dtype)
    if not normalize[0]:
        k = (k.float() / (k.float().norm(dim=-1, keepdim=True) * 2)).to(dtype)   # keys of norm 1/2 keep the state bounded
    pool = Pool(1, 2, kh, vh, dk, dv, capacity=6)
    out = pool.rows(0, q, k, v, g, b, [(0, 6, 1, rs.STASH)], 0.3, epsilon=eps, normalize=normalize)
    pool.record([6], [1])
    pool.flush([1])
    ref_out, ref_state = reference(q.float(), k.float(), v.float(), g, b, torch.zeros(vh, dk, dv, device=dev), 0.3, eps,
                                   normalize)
    torch.testing.assert_close(out.double(), ref_out, rtol=1e-4, atol=1e-5)
    torch.testing.assert_close(pool.states[0, 1].double(), ref_state, rtol=1e-4, atol=1e-5)


@pytest.mark.kernels_ci
def test_final_rows_keep_the_state_after_their_last_token_in_their_scratch_slot():
    kh, vh, dk, dv = 1, 2, 32, 32
    q, k, v, g, b = inputs(11, kh, vh, dk, dv, seed=23)
    pool = Pool(1, 3, kh, vh, dk, dv, capacity=2)
    S = vh * dk * dv
    scratch = torch.full((2, S), float("nan"), device=dev)
    before = pool.states.clone()
    # a decode row, then two FINAL rows in scratch slots 1 and 0 (the slots need not follow the rows)
    out = pool.rows(0, q, k, v, g, b, [(0, 1, 0, rs.STASH, -1), (1, 4, 1, rs.FINAL, 1), (5, 6, 2, rs.FINAL, 0)], 0.25,
                    scratch=scratch)
    assert torch.equal(pool.states, before)   # FINAL rows leave the live state untouched
    for first, count, seq, slot in ((1, 4, 1, 1), (5, 6, 2, 0)):
        rows = slice(first, first + count)
        ref_out, ref_state = reference(q[rows], k[rows], v[rows], g[rows], b[rows], torch.zeros(vh, dk, dv, device=dev), 0.25)
        torch.testing.assert_close(out[rows].double(), ref_out, rtol=1e-4, atol=1e-5)
        torch.testing.assert_close(scratch[slot].view(vh, dk, dv).double(), ref_state, rtol=1e-4, atol=1e-5)
    # a FINAL row without scratch writes zeros and touches nothing
    out = pool.rows(0, q, k, v, g, b, [(0, 4, 1, rs.FINAL, -1)], 0.25, scratch=None)
    assert torch.equal(out[:4], torch.zeros_like(out[:4])) and torch.equal(pool.states, before)


@pytest.mark.kernels_ci
def test_a_flush_of_more_rows_by_layers_than_a_grid_dimension_holds():
    layers, sequences = 48, 1400   # rows x layers past 65,535
    pool = Pool(layers, sequences, 1, 1, 4, 4, capacity=1)
    pool.stash.normal_()
    pool.control[:, :, 0] = 1   # every block of every sequence: one pending step
    before = pool.states.clone()
    pool.flush(list(range(sequences)))
    torch.cuda.synchronize()
    assert int(pool.control[:, :, 0].abs().sum()) == 0 and not torch.equal(pool.states, before)
    window = torch.zeros(layers, sequences, 8, 3, device=dev)
    stash = torch.randn(layers, sequences, 1, 8, device=dev)
    pending = torch.ones(layers, sequences, 1, dtype=torch.int32, device=dev)
    meta = torch.tensor([[0, 0, s, rs.FLUSH, -1] for s in range(sequences)], dtype=torch.int32, device=dev)
    x = torch.zeros(0, 8, device=dev)
    rs.pool_conv(torch.zeros(0, 8, device=dev), window, stash, pending, None, x, torch.zeros(8, 4, device=dev), meta=meta,
                 rows=sequences, kind=rs.FLUSH, layer=0, layers=layers, capacity=1)
    torch.cuda.synchronize()
    assert int(pending.sum()) == 0 and torch.equal(window[:, :, :, 2], stash[:, :, 0])


@pytest.mark.kernels_ci
def test_a_convolution_into_its_own_input_is_refused():
    x = torch.randn(4, 8, device=dev)
    with pytest.raises(RuntimeError, match="out must not be x"):
        rs.conv_segments(x, torch.zeros(8, 3, device=dev), torch.zeros(8, 3, device=dev), x, torch.zeros(8, 4, device=dev))


@pytest.mark.kernels_ci
@pytest.mark.parametrize("width", [2, 4, 8])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_a_padded_prompt_chunk_convolves_its_real_rows_and_moves_the_window_past_them(width, dtype):
    """pool_conv_prompt over a bucket of 64 rows has the bits of a pending flush then conv_segments over the chunk's
    real rows alone, outputs zeros for the padding, and writes the window after the real inputs in place; a sequence
    below 0 touches nothing. Captured once, it replays for any count."""
    g = torch.Generator(device="cpu").manual_seed(40 + width)
    C, cap, bucket = 300, 4, 64
    x = torch.randn(bucket, C, generator=g).to(dev).to(dtype)
    weight, bias = torch.randn(C, width, generator=g).to(dev), torch.randn(C, generator=g).to(dev)
    window = torch.randn(1, 3, C, width - 1, generator=g).to(dev)
    stash = torch.randn(1, 3, cap, C, generator=g).to(dev)
    pending = torch.zeros(1, 3, rs.pool_conv_blocks(C), dtype=torch.int32, device=dev)
    meta = torch.zeros(1, 5, dtype=torch.int32, device=dev)
    out = torch.full_like(x, float("nan"))
    graph = torch.cuda.CUDAGraph()
    rs.pool_conv_prompt(out, window, stash, pending, x, weight, bias, meta=meta, capacity=cap)   # warm-up: count 0
    with torch.cuda.graph(graph):
        rs.pool_conv_prompt(out, window, stash, pending, x, weight, bias, meta=meta, capacity=cap)
    for count, waiting in ((0, 2), (1, 0), (2, 3), (width - 1, 1), (23, 0), (64, 2)):
        pending.fill_(waiting)
        expected_window, expected_pending = window.clone(), pending.clone()
        flush = torch.tensor([[0, 0, 1, rs.FLUSH, -1]], dtype=torch.int32, device=dev)
        rs.pool_conv(out[:0], expected_window, stash, expected_pending, None, x[:0], weight, bias, meta=flush, rows=1,
                     capacity=cap)
        expected = torch.zeros_like(x)
        after = expected_window[0, 1].clone()
        if count:
            rs.conv_segments(expected[:count], expected_window[0, 1].clone(), after, x[:count], weight, bias)
        meta.copy_(torch.tensor([[0, count, 1, rs.FINAL, 1]], dtype=torch.int32))
        graph.replay()
        assert torch.equal(out, expected), count
        assert torch.equal(window[0, 1], after), count
        assert torch.equal(pending[0, 1], torch.zeros_like(pending[0, 1]))
        assert torch.equal(pending[0, 0], expected_pending[0, 0])
        assert torch.equal(window[0, 0], expected_window[0, 0]) and torch.equal(window[0, 2], expected_window[0, 2])
    before, waiting = window.clone(), pending.clone()
    meta.copy_(torch.tensor([[0, 0, -1, rs.FINAL, -1]], dtype=torch.int32))
    graph.replay()
    assert torch.equal(out, torch.zeros_like(out)) and torch.equal(window, before) and torch.equal(pending, waiting)


@pytest.mark.kernels_ci
def test_a_final_row_into_its_own_live_state_continues_it():
    """A FINAL row whose scratch slot is its sequence's live state (a captured prompt chunk's pool_delta): the pending
    count applied, the row's tokens scanned, the live state left as decoding them one by one would."""
    kh, vh, dk, dv = 2, 4, 64, 64
    q, k, v, g, b = inputs(9, kh, vh, dk, dv, seed=29)
    whole, serial = Pool(1, 3, kh, vh, dk, dv, capacity=2), Pool(1, 3, kh, vh, dk, dv, capacity=2)
    for pool in (whole, serial):   # a decode row of two tokens, committed (deferred)
        pool.rows(0, q[:2], k[:2], v[:2], g[:2], b[:2], [(0, 2, 1, rs.STASH, -1)], 0.25)
        pool.record([2], [1])
    live = whole.states[0].view(whole.states.shape[1], -1)
    out = whole.rows(0, q[2:], k[2:], v[2:], g[2:], b[2:], [(0, 7, 1, rs.FINAL, 1)], 0.25, scratch=live)
    steps = []
    for t in range(2, 9):   # the same tokens decoded one by one, each committed
        steps.append(serial.rows(0, q[t:t + 1], k[t:t + 1], v[t:t + 1], g[t:t + 1], b[t:t + 1], [(0, 1, 1, rs.STASH, -1)],
                                 0.25))
        serial.record([1], [1])
    serial.flush([1])
    assert torch.equal(out, torch.cat(steps))
    assert torch.equal(whole.states[0, 1], serial.states[0, 1])
    assert torch.equal(whole.states[0, 0], serial.states[0, 0]) and torch.equal(whole.states[0, 2], serial.states[0, 2])


def dilated_reference(x, weight, bias, window, dilation, silu=True):
    """FP64 outputs [T][C] of a causal depthwise convolution at `dilation` from `window` [C][(width - 1) * dilation]
    (oldest first), and the window after the rows."""
    padded = torch.cat([window.T.double(), x.double()])
    w = weight.shape[1]
    out = torch.stack([sum(padded[t + i * dilation] * weight[:, i].double() for i in range(w)) + bias.double()
                       for t in range(x.shape[0])])
    return torch.nn.functional.silu(out) if silu else out, padded[-(w - 1) * dilation:].T


@pytest.mark.kernels_ci
@pytest.mark.parametrize("shape", [(10240, 4, 3), (300, 2, 4), (300, 3, 2), (300, 4, 1)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_a_dilated_convolution_follows_the_fp64_one_on_every_path(shape, dtype):
    """(channels, taps, dilation): Flash-Next's PLE convolution (4 x 2,560 channels, 4 taps at dilation 3) and others.
    Decode and verify rows from the window (the window kept until the commit), a flush of a pending count, a long row
    through the segments kernel and through a FINAL row (the same bits), and a padded prompt chunk (the bits of a flush
    then the segments kernel over its real rows), each against an FP64 dilated convolution."""
    C, width, dilation = shape
    span, cap = (width - 1) * dilation, 4
    tol = dict(rtol=1e-2, atol=1e-2) if dtype != torch.float32 else dict(rtol=1e-5, atol=1e-5)
    g = torch.Generator(device="cpu").manual_seed(C + width * 10 + dilation)
    x = torch.randn(90, C, generator=g).to(dev).to(dtype)
    weight, bias = torch.randn(C, width, generator=g).to(dev), torch.randn(C, generator=g).to(dev)
    window = torch.randn(1, 3, C, span, generator=g).to(dev)
    start = window.clone()
    stash = torch.zeros(1, 3, cap, C, device=dev)
    pending = torch.zeros(1, 3, rs.pool_conv_blocks(C), dtype=torch.int32, device=dev)
    assert rs.pool_conv_supported(width, dilation)
    # A verify row of 3 inputs (sequence 2) beside a decode row (sequence 0): outputs from the windows, which stay.
    out = torch.zeros(4, C, dtype=dtype, device=dev)
    meta = torch.tensor([[0, 3, 2, rs.STASH, -1], [3, 1, 0, rs.STASH, -1]], dtype=torch.int32, device=dev)
    rs.pool_conv(out, window, stash, pending, None, x[:4], weight, bias, meta=meta, rows=2, capacity=cap)
    for first, count, seq in ((0, 3, 2), (3, 1, 0)):
        ref, _ = dilated_reference(x[first:first + count].float(), weight, bias, start[0, seq], dilation)
        torch.testing.assert_close(out[first:first + count].double(), ref.to(dtype).double(), **tol)
    assert torch.equal(window, start)
    # Two of the three accepted, then flushed: the window after those two inputs, exact.
    pending[0, 2] = 2
    flush = torch.tensor([[0, 0, 2, rs.FLUSH, -1]], dtype=torch.int32, device=dev)
    rs.pool_conv(out[:0], window, stash, pending, None, x[:0], weight, bias, meta=flush, rows=1, capacity=cap)
    _, expected = dilated_reference(x[:2].float(), weight, bias, start[0, 2], dilation)
    torch.testing.assert_close(window[0, 2].double(), expected, rtol=0, atol=0)
    # A long row: the segments kernel and a FINAL row give the same bits, and the FP64 values.
    long = x[4:74]
    seg_out, seg_window = torch.empty_like(long), torch.empty(C, span, device=dev)
    rs.conv_segments(seg_out, window[0, 1].clone(), seg_window, long, weight, bias)
    scratch = torch.zeros(1, C, span, device=dev)
    row_out = torch.empty_like(long)
    final = torch.tensor([[0, 70, 1, rs.FINAL, 0]], dtype=torch.int32, device=dev)
    rs.pool_conv(row_out, window, stash, pending, scratch, long, weight, bias, meta=final, rows=1, capacity=cap)
    assert torch.equal(seg_out, row_out) and torch.equal(seg_window, scratch[0])
    ref, ref_window = dilated_reference(long.float(), weight, bias, window[0, 1], dilation)
    torch.testing.assert_close(seg_out.double(), ref.to(dtype).double(), **tol)
    torch.testing.assert_close(seg_window.double(), ref_window, rtol=0, atol=0)
    # Rows shorter than the window: its older inputs carry over.
    short_window = torch.empty(C, span, device=dev)
    rs.conv_segments(torch.empty_like(x[:1]), window[0, 1].clone(), short_window, x[:1], weight, bias)
    _, expected = dilated_reference(x[:1].float(), weight, bias, window[0, 1], dilation)
    torch.testing.assert_close(short_window.double(), expected, rtol=0, atol=0)
    # A padded prompt chunk of 23 real rows in 64, a pending count of 1 first: a flush then the segments kernel's bits.
    stash[0, 0, 0] = x[80].float()
    pending[0, 0] = 1
    flushed, flushed_pending = window.clone(), pending.clone()
    rs.pool_conv(out[:0], flushed, stash, flushed_pending, None, x[:0], weight, bias,
                 meta=torch.tensor([[0, 0, 0, rs.FLUSH, -1]], dtype=torch.int32, device=dev), rows=1, capacity=cap)
    chunk = x[10:74]
    expected, after = torch.zeros_like(chunk), torch.empty(C, span, device=dev)
    rs.conv_segments(expected[:23], flushed[0, 0].clone(), after, chunk[:23], weight, bias)
    found = torch.full_like(chunk, float("nan"))
    rs.pool_conv_prompt(found, window, stash, pending, chunk, weight, bias,
                        meta=torch.tensor([0, 23, 0, rs.FINAL, 0], dtype=torch.int32, device=dev), capacity=cap)
    assert torch.equal(found, expected) and torch.equal(window[0, 0], after)
    assert int(pending[0, 0].abs().sum()) == 0


@pytest.mark.kernels_ci
def test_a_convolution_the_kernels_do_not_build_is_refused():
    assert not rs.pool_conv_supported(5, 2) and not rs.pool_conv_supported(4, 5) and rs.pool_conv_supported(8, 1)
    C = 16
    x, weight = torch.randn(4, C, device=dev), torch.randn(C, 5, device=dev)
    with pytest.raises(RuntimeError, match="dilation 2 is none the kernels run"):
        rs.conv_segments(torch.empty_like(x), torch.zeros(C, 8, device=dev), torch.zeros(C, 8, device=dev), x, weight)
    with pytest.raises(RuntimeError, match="holds \(width - 1\) \* dilation"):
        rs.conv_segments(torch.empty_like(x), torch.zeros(C, 6, device=dev), torch.zeros(C, 6, device=dev), x, weight)
