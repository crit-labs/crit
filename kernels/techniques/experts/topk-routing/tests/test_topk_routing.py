"""The fused top-k routing against an FP64 reference of the same rule: the same ids (ties to the lower id), weights
within FP32 rounding, rows of NaN logits routed to the lowest ids with weights 0 (NaN below a real -inf) and other rows
untouched, a captured call replaying new logits, and a non-default configuration of the knobs giving the same bits."""
import math

import pytest
import torch

from crit_kernels.build import load

tr = load("topk-routing")
dev = torch.device("cuda")
RULES = [dict(softmax=True, before=False, renormalize=False), dict(softmax=True, before=True, renormalize=True),
         dict(softmax=False, before=True, renormalize=True, scaling=2.5), dict(softmax=False, before=False)]


def reference(logits, top, bias, softmax, before, renormalize=False, epsilon=0.0, floor=0.0, scaling=1.0):
    ids, weights = [], []
    for row in logits.double().tolist():
        if before:
            if softmax:
                m = max(row)
                e = [math.exp(v - m) for v in row]
                s = math.fsum(e)
                scores = [v / s for v in e]
            else:
                scores = [1 / (1 + math.exp(-v)) for v in row]
        else:
            scores = row
        choice = [s + (bias[i] if bias is not None else 0.0) for i, s in enumerate(scores)]
        picked = sorted(range(len(row)), key=lambda i: (-choice[i], i))[:top]
        if before:
            w = [scores[i] for i in picked]
        elif softmax:
            m = max(row[i] for i in picked)
            e = [math.exp(row[i] - m) for i in picked]
            w = [v / math.fsum(e) for v in e]
        else:
            w = [1 / (1 + math.exp(-row[i])) for i in picked]
        if renormalize:
            d = max(math.fsum(w) + epsilon, floor)
            w = [v / d for v in w]
        ids.append(picked)
        weights.append([v * scaling for v in w])
    return ids, weights


@pytest.mark.kernels_ci
@pytest.mark.parametrize("experts,top", [(8, 2), (64, 6), (128, 8), (256, 8), (512, 10), (1024, 32)])
def test_the_rule_routes_as_the_reference(experts, top):
    g = torch.Generator().manual_seed(experts + top)
    logits = torch.randn(9, experts, generator=g) * 3
    bias = (torch.rand(experts, generator=g) - 0.5) * 0.1
    for rule in RULES:
        for b in (None, bias):
            ids, weights = tr.route(logits.to(dev), top, None if b is None else b.to(dev), **rule)
            want_ids, want_w = reference(logits, top, None if b is None else b.tolist(), **rule)
            assert ids.cpu().tolist() == want_ids, (rule, b is not None)
            assert torch.allclose(weights.cpu().double(), torch.tensor(want_w, dtype=torch.float64), atol=2e-6), rule


@pytest.mark.kernels_ci
def test_ties_go_to_the_lower_id_and_nan_rows_still_route():
    logits = torch.tensor([[1.0, 2.0, 2.0, 0.5, 2.0, 0.5, 0.5, 0.5], [0.0] * 8, [float("nan")] * 8,
                           [1.0, float("nan"), 3.0, 0.0, 0.0, 0.0, 0.0, 0.0]], device=dev)
    ids, weights = tr.route(logits, 3, None, softmax=True, before=False)
    assert ids[0].tolist() == [1, 2, 4] and ids[1].tolist() == [0, 1, 2] and ids[2].tolist() == [0, 1, 2]
    assert ids[3].tolist() == [2, 0, 3] and weights[2].tolist() == [0.0, 0.0, 0.0]
    ids, weights = tr.route(logits, 3, None, softmax=True, before=True, renormalize=True)
    assert ids[2].tolist() == [0, 1, 2] and ids[3].tolist() == [0, 1, 2]   # a NaN poisons a softmax over the row
    assert not bool(torch.isnan(weights).any())


@pytest.mark.kernels_ci
def test_nan_ranks_below_a_real_negative_infinity():
    inf, nan = float("inf"), float("nan")
    logits = torch.tensor([[nan, -inf, 1.0, nan, -inf, nan, nan, nan], [-0.0, 0.0, nan, -inf, -1.0, nan, nan, nan]],
                          device=dev)
    ids, weights = tr.route(logits, 4, None, softmax=False, before=False)
    assert ids[0].tolist() == [2, 1, 4, 0]   # 1, then the two -inf ahead of every NaN
    assert ids[1].tolist() == [0, 1, 4, 3]   # -0 and +0 tie: the lower id first
    assert weights[0].tolist()[1:3] == [0.0, 0.0] and weights[0, 3].item() == 0.0


@pytest.mark.kernels_ci
def test_a_captured_call_replays_new_logits():
    g = torch.Generator().manual_seed(3)
    logits = (torch.randn(6, 512, generator=g)).to(dev)
    ids = torch.empty(6, 10, dtype=torch.int64, device=dev)
    weights = torch.empty(6, 10, device=dev)
    tr.route(logits, 10, None, True, True, True, ids=ids, weights=weights)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        tr.route(logits, 10, None, True, True, True, ids=ids, weights=weights)
    logits.copy_(torch.randn(6, 512, generator=g))
    graph.replay()
    eager = tr.route(logits.clone(), 10, None, True, True, True)
    assert torch.equal(ids, eager[0]) and torch.equal(weights, eager[1])


@pytest.mark.kernels_ci
def test_other_knobs_route_the_same_bits():
    other = load("topk-routing", knobs={"warp-rows": {"WARPS": 1, "ROWS": 4}})
    g = torch.Generator().manual_seed(11)
    for experts, top in ((64, 6), (512, 10)):
        logits = (torch.randn(37, experts, generator=g) * 3).to(dev)
        for rule in RULES:
            a, b = tr.route(logits, top, None, **rule), other.route(logits, top, None, **rule)
            assert torch.equal(a[0], b[0]) and torch.equal(a[1].view(torch.int32), b[1].view(torch.int32)), rule
