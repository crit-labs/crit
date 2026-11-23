"""sparse-indexer against attention-state's float64 reference (sparse_index.py): scores over 16-bit and E4M3 keys at
the head counts and widths it takes, causal ends; picks with ties, fewer valid keys than k, empty rows and NaN; a
token's bits the same alone and beside other tokens; select in groups the same as one call; what it refuses."""
import importlib.util
from pathlib import Path

import pytest
import torch

from crit_kernels.build import load

si = load("sparse-indexer")
dev = torch.device("cuda")
_spec = importlib.util.spec_from_file_location(
    "sparse_reference", Path(__file__).resolve().parents[5] / "inference" / "techniques" / "attention" / "attention-state" / "code" /
    "sparse_index.py")
ref = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ref)


def problem(t, h, d, n, seed=0, fp8=False, dtype=torch.bfloat16):
    g = torch.Generator().manual_seed(seed)
    q = (torch.randn(t, h, d, generator=g) * 0.5).to(dev).to(dtype)
    keys = (torch.randn(n, d, generator=g) * 0.5).to(dev).to(dtype)
    w = (torch.randn(t, h, generator=g) * h ** -0.5).to(dev)
    ends = torch.randint(0, n + 1, (t,), generator=g).to(torch.int32).to(dev)
    ends[0] = n
    scales, values = None, keys.float()
    if fp8:
        scales = (keys.float().abs().amax(1) / 448.0).clamp_min(1e-30)
        keys = (keys.float() / scales[:, None]).to(torch.float8_e4m3fn)
        values = keys.float() * scales[:, None]
    return q, keys, scales, values, w, ends


@pytest.mark.kernels_ci
@pytest.mark.parametrize("h,d", [(64, 128), (32, 64), (16, 256), (8, 128)])
@pytest.mark.parametrize("fp8", [False, True])
def test_scores_match_the_reference(h, d, fp8):
    q, keys, scales, values, w, ends = problem(37, h, d, 1000, seed=h + d, fp8=fp8)
    got = si.scores(q, keys, w, ends, key_scales=scales)
    want = ref.sparse_scores(q, values, w, ends, d ** -0.5)
    bound = ref.sparse_scores(q.abs(), values.abs(), w.abs(), ends, d ** -0.5) * 1e-4 + 1e-6
    finite = torch.isfinite(want)
    assert torch.equal(finite, torch.isfinite(got))
    assert bool(((got.double() - want).abs() <= bound)[finite].all())


@pytest.mark.kernels_ci
def test_a_tokens_scores_are_its_own():
    q, keys, scales, _, w, ends = problem(33, 64, 128, 700, seed=3)
    all_rows = si.scores(q, keys, w, ends)
    for t in (0, 1, 16, 32):
        alone = si.scores(q[t:t + 1], keys, w[t:t + 1], ends[t:t + 1])
        assert torch.equal(alone[0].view(torch.int32), all_rows[t].view(torch.int32)), t
    again = si.scores(q, keys, w, ends)
    assert torch.equal(again.view(torch.int32), all_rows.view(torch.int32))


@pytest.mark.kernels_ci
def test_the_tile_changes_no_bit():
    """TM, BN and WARPS are speed knobs: the head sum is an explicit tree, so a build with another tile gives every
    token the same scores, alone (a one-token tile) and in a chunk."""
    other = load("sparse-indexer", knobs={"scores": {"TM": 1, "BN": 128, "WARPS": 8}})
    third = load("sparse-indexer", knobs={"scores": {"TM": 2, "BN": 32, "WARPS": 4}})
    for h, d, fp8 in ((64, 128, False), (8, 64, True), (128, 256, False), (1, 64, False)):
        q, keys, scales, _, w, ends = problem(5, h, d, 900, seed=h, fp8=fp8)
        want = si.scores(q, keys, w, ends, key_scales=scales)
        for build in (other, third):
            got = build.scores(q, keys, w, ends, key_scales=scales)
            assert torch.equal(got.view(torch.int32), want.view(torch.int32)), (h, d, fp8)
        alone = si.scores(q[3:4], keys, w[3:4], ends[3:4], key_scales=scales)
        assert torch.equal(alone[0].view(torch.int32), want[3].view(torch.int32)), (h, d, fp8)


@pytest.mark.kernels_ci
@pytest.mark.parametrize("n,k", [(5000, 512), (700, 64), (300, 512), (1, 1), (4097, 2048)])
def test_topk_picks_the_references_positions(n, k):
    g = torch.Generator().manual_seed(n + k)
    t = 9
    s = torch.randn(t, n, generator=g)
    s[:, torch.randint(0, n, (n // 3,), generator=g)] = 0.0          # many ties
    s[2] = 1.0                                                        # one row all ties
    s = s.to(dev)
    ends = torch.tensor([n, n - 1, n, k // 2, 0, max(1, n - 300), k, n, 1], dtype=torch.int32, device=dev).clamp(0, n)
    got = si.topk(s, ends, k)
    assert torch.equal(got.long(), ref.sparse_index(s, ends, k))


@pytest.mark.kernels_ci
def test_a_nan_ranks_below_every_number():
    s = torch.tensor([[1.0, float("nan"), float("-inf"), 2.0, float("nan")]], device=dev)
    end = torch.tensor([5], dtype=torch.int32, device=dev)
    assert si.topk(s, end, 4).tolist() == [[0, 1, 2, 3]]   # 2, 1, -inf, then the first NaN
    assert si.topk(s, end, 3).tolist() == [[0, 2, 3]]
    for k in (3, 4):
        assert torch.equal(si.topk(s, end, k).long(), ref.sparse_index(s, end, k))


@pytest.mark.kernels_ci
def test_select_in_groups_is_one_call_and_matches_the_reference():
    q, keys, scales, values, w, ends = problem(50, 64, 128, 3000, seed=9, fp8=True)
    whole = si.select(q, keys, w, ends, 256, key_scales=scales, workspace_bytes=1 << 30)
    grouped = si.select(q, keys, w, ends, 256, key_scales=scales, workspace=torch.empty(7 * 3000, device=dev))
    assert torch.equal(whole, grouped)
    want = ref.sparse_scores(q, values, w, ends, 128 ** -0.5)
    picked = ref.sparse_index(want, ends, 256)
    # where the exact and the kernel's picks differ, the kernel's picks score within the products' error of the k-th
    got_scores = si.scores(q, keys, w, ends, key_scales=scales)
    assert torch.equal(si.topk(got_scores, ends, 256), whole)
    for t in range(q.shape[0]):
        a, b = set(whole[t].tolist()) - {-1}, set(picked[t].tolist()) - {-1}
        if a != b:
            kth = want[t][list(b)].min()
            assert float(want[t][list(a - b)].min()) >= float(kth) - 1e-3, t


@pytest.mark.kernels_ci
@pytest.mark.parametrize("fp8", [False, True])
@pytest.mark.parametrize("kind", [torch.int32, torch.int64])
def test_keys_read_in_place_score_as_their_gathered_copy(fp8, kind):
    # A paged pool: a sequence's keys lie at its slots of a larger store. Read there, the scores have the bits of
    # the gathered keys', for one sequence and for a round's rows (a group of tokens a sequence).
    rows, each, n = 3, 4, 700
    q, _, _, _, w, ends = problem(rows * each, 64, 128, n, seed=11)
    g = torch.Generator().manual_seed(5)
    store = (torch.randn(5000, 128, generator=g) * 0.5).to(dev).to(torch.bfloat16)
    scales = None
    if fp8:
        scales = (store.float().abs().amax(1) / 448.0).clamp_min(1e-30)
        store = (store.float() / scales[:, None]).to(torch.float8_e4m3fn)
    slots = torch.stack([torch.randperm(5000, generator=g)[:n] for _ in range(rows)]).to(dev).to(kind)
    got = si.scores(q, store, w, ends, key_scales=scales, slots=slots)
    for r in range(rows):
        span = slice(r * each, (r + 1) * each)
        keys = store[slots[r].long()]
        want = si.scores(q[span], keys, w[span], ends[span], key_scales=None if scales is None else scales[slots[r].long()])
        assert torch.equal(got[span], want), r
    one = si.scores(q[:each], store, w[:each], ends[:each], key_scales=scales, slots=slots[0])
    assert torch.equal(one, got[:each])                                   # [n]: one sequence
    picked = si.select(q[:each], store, w[:each], ends[:each], 16, key_scales=scales, slots=slots[0])
    assert torch.equal(picked, si.topk(got[:each], ends[:each], 16))
    with pytest.raises(ValueError):
        si.scores(q[:5], store, w[:5], ends[:5], key_scales=scales, slots=slots)   # 5 tokens are not 3 groups
    with pytest.raises(ValueError):
        si.scores(q, store, w, ends, key_scales=scales, slots=slots.float())


@pytest.mark.kernels_ci
def test_graphs_capture_a_select():
    q, keys, scales, _, w, ends = problem(4, 64, 128, 2048, seed=4)
    out = torch.empty(4, 512, dtype=torch.int32, device=dev)
    space = torch.empty(4 * 2048, device=dev)
    si.select(q, keys, w, ends, 512, out=out, workspace=space)
    want = out.clone()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        si.select(q, keys, w, ends, 512, out=out, workspace=space)
    out.zero_()
    graph.replay()
    assert torch.equal(out, want)


@pytest.mark.kernels_ci
def test_it_refuses_what_it_does_not_take():
    q, keys, scales, _, w, ends = problem(2, 64, 128, 10)
    with pytest.raises(ValueError):
        si.scores(q.float(), keys, w, ends)
    with pytest.raises(ValueError):
        si.scores(q, keys.half(), w, ends)
    with pytest.raises(ValueError):
        si.scores(q, keys.to(torch.float8_e4m3fn), w, ends)              # E4M3 keys without scales
    with pytest.raises(ValueError):
        si.scores(q, keys, w, ends.long())
    with pytest.raises(ValueError):
        si.scores(q[:, :48], keys, w[:, :48], ends)                      # heads not a power of two
    with pytest.raises(ValueError):
        si.topk(torch.zeros(2, 10, device=dev), ends, 0)


@pytest.mark.kernels_ci
@pytest.mark.parametrize("n,k", [(50000, 512), (20000, 2048), (9000, 64)])
def test_slices_pick_what_one_pass_picks(n, k):
    """A few tokens over many keys select in slices then over the slices' picks; the same rows in a batch wide enough
    for one pass a row pick the same positions, and the reference's."""
    g = torch.Generator().manual_seed(n)
    s = torch.randn(200, n, generator=g)
    s[:, torch.randint(0, n, (n // 4,), generator=g)] = 0.0
    s[1, 100:] = 0.5
    s = s.to(dev)
    ends = torch.full((200,), n, dtype=torch.int32, device=dev)
    ends[2], ends[3] = n - 777, k + 5
    batch = si.topk(s, ends, k)
    for t in range(4):
        alone = si.topk(s[t:t + 1].contiguous(), ends[t:t + 1], k)
        assert torch.equal(alone[0], batch[t]), t
    assert torch.equal(batch[:4].long(), ref.sparse_index(s[:4], ends[:4], k))


@pytest.mark.kernels_ci
@pytest.mark.parametrize("n,k,t", [(200000, 512, 1), (60000, 4096, 2), (8192, 512, 3), (5, 512, 1), (33000, 16, 4),
                                   (4096, 1, 2)])
def test_the_cuda_select_picks_what_the_triton_programs_pick(n, k, t):
    """Calls of few tokens select in the CUDA kernel (one launch, slices and a second launch over their picks, fewer
    and wider slices where the candidates would not fit a block's shared memory, the Triton programs where a slice
    would not): the positions a build without it gives, with ties, NaNs, -inf, rows cut short and empty rows."""
    triton_only = load("sparse-indexer", knobs={"topk": {"CUDA": 0}})
    g = torch.Generator().manual_seed(n * 7 + k)
    s = torch.randn(t, n, generator=g)
    s[:, torch.randint(0, n, (n // 3,), generator=g)] = 0.0
    s[:, torch.randint(0, n, (max(1, n // 50),), generator=g)] = float("nan")
    s[:, torch.randint(0, n, (max(1, n // 40),), generator=g)] = float("-inf")
    s[-1, : n // 2] = 2.5   # more ties at the threshold than a slice keeps
    s = s.to(dev)
    for cut in (0, 1, 3, n // 2, n):
        ends = torch.full((t,), n, dtype=torch.int32, device=dev)
        ends[0] = max(0, n - cut)
        got = si.topk(s, ends, k)
        assert torch.equal(got, triton_only.topk(s, ends, k)), (n, k, t, cut)
    if n <= 33000:
        assert torch.equal(got.long(), ref.sparse_index(s, ends, k))
