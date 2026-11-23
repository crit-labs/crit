"""crit_kernels.contracts, CI's check of the operation contract (CPU): every operation of crit's techniques has a
contract, every structure of every kernel maps to defined operations, and the FP64 references agree with the
techniques' own host forms (their operations.json references) on small inputs."""
import importlib.util
import math
from pathlib import Path

import pytest
import torch

from crit_kernels import contracts

ROOT = Path(__file__).resolve().parents[3]


def _module(relative):
    path = ROOT / relative
    spec = importlib.util.spec_from_file_location("ref_" + path.stem, path)
    module = importlib.util.module_from_spec(spec)
    import sys
    sys.path.insert(0, str(path.parent))
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path.remove(str(path.parent))
    return module


def test_every_operation_has_a_contract_and_every_structure_maps_to_defined_operations():
    assert contracts.problems() == []


def test_problems_name_a_kernel_without_operations_and_an_undefined_one(tmp_path):
    # A copy of the tree with one kernel naming none and one naming an undefined operation.
    import json
    import shutil
    shutil.copytree(ROOT / "inference" / "techniques", tmp_path / "inference" / "techniques", ignore=shutil.ignore_patterns("*.py", "tests"))
    shutil.copytree(ROOT / "kernels", tmp_path / "kernels", ignore=shutil.ignore_patterns("tests", "*.cu", "*.py",
                                                                                          "*.cuh", "*.md"))
    a = tmp_path / "kernels/techniques/experts/topk-routing/kernel.json"
    data = json.loads(a.read_text(encoding="utf-8"))
    data["operations"] = []
    a.write_text(json.dumps(data), encoding="utf-8")
    b = tmp_path / "kernels/techniques/projections/gguf-rows/kernel.json"
    data = json.loads(b.read_text(encoding="utf-8"))
    data["operations"] = ["no-such-operation"]
    data["structures"]["gemv"]["operations"] = ["gated-delta"]
    b.write_text(json.dumps(data), encoding="utf-8")
    found = contracts.problems(tmp_path)
    assert any("topk-routing: kernel.json names no operation" in p for p in found), found
    assert any("gguf-rows: operation no-such-operation is defined by no technique" in p for p in found), found
    assert any("gguf-rows/gemv: operation gated-delta is not one of the kernel's" in p for p in found), found


def test_tolerances():
    want = torch.tensor([1.0, -2.0, 0.0])
    assert contracts.Close(1e-3, 1e-3).check(want + 1e-4, want) is None
    assert "beyond" in contracts.Close(1e-3, 1e-3).check(want + 1e-2, want)
    assert "NaN" in contracts.Close(1e-3, 1e-3).check(torch.tensor([float("nan"), -2.0, 0.0]), want)
    assert contracts.Sums().check(want + 1e-7, (want, torch.ones(3))) is None
    assert contracts.Sums().check(want + 1e-2, (want, torch.ones(3))) is not None
    assert contracts.Bits("x").check(torch.tensor([1, 2]), torch.tensor([1, 2])) is None
    assert contracts.Bits("x").check(torch.tensor([0.0]), torch.tensor([-0.0])) is not None
    assert contracts.check("paged-attention", want, want) is None
    assert contracts.Close(0.0, 0.0, 0.1).check(want + 0.15, want) is None   # 0.1 of the largest |want|, 2
    assert contracts.check("paged-attention", want + 5e-2, want, arithmetic="int8-queries") is None
    assert contracts.check("paged-attention", want + 5e-2, want) is not None
    with pytest.raises(KeyError):
        contracts.contract("nothing")
    with pytest.raises(ValueError, match="declared order"):
        contracts.reference("integer-block-rows")


def test_attention_is_the_softmax_definition_and_merges_to_itself():
    g = torch.Generator().manual_seed(1)
    q, k, v = torch.randn(3, 4, 8, generator=g), torch.randn(10, 2, 8, generator=g), torch.randn(10, 2, 5, generator=g)
    seen = torch.arange(10)[None, :] <= torch.tensor([6, 8, 9])[:, None]
    sinks = torch.randn(4, generator=g)
    out, lse = contracts.attention(q, k, v, seen, 0.3, sinks)
    for t in range(3):
        for h in range(4):
            s = [0.3 * float(q[t, h].double() @ k[n, h // 2].double()) for n in range(10) if seen[t, n]]
            z = math.log(math.fsum(math.exp(x) for x in s) + math.exp(float(sinks[h])))
            o = sum(math.exp(x - z) * v[n, h // 2].double() for x, n in zip(s, range(10)))
            assert abs(lse[t, h] - z) < 1e-12 and float((out[t, h] - o).abs().max()) < 1e-12
    # two partitions' states (each its normalized output, its log-sum-exp as the maximum and a sum of 1) merged equal
    # the whole row's attention
    parts = [contracts.attention(q, k, v, seen & part, 0.3) for part in (torch.arange(10) < 5, torch.arange(10) >= 5)]
    outputs = torch.stack([o for o, _ in parts]).nan_to_num()
    maxima = torch.stack([l for _, l in parts])
    merged, merged_lse = contracts.merge(outputs, maxima, torch.ones_like(maxima))
    whole, whole_lse = contracts.attention(q, k, v, seen, 0.3)
    assert float((merged - whole).abs().max()) < 1e-12 and float((merged_lse - whole_lse).abs().max()) < 1e-12


def test_experts_are_the_techniques_moe_block():
    ref = _module("inference/techniques/experts/expert-streamed-prefill/code/reference.py")
    g = torch.Generator().manual_seed(2)
    E, H, I = 3, 8, 4
    gu = [torch.randn(2 * I, H, generator=g, dtype=torch.float64) for _ in range(E)]
    dn = [torch.randn(H, I, generator=g, dtype=torch.float64) for _ in range(E)]
    x = torch.randn(4, H, generator=g, dtype=torch.float64)
    ids = torch.tensor([[0, 2], [1, -1], [2, 1], [0, 1]])
    w = torch.rand(4, 2, generator=g, dtype=torch.float64)
    for activation in ("silu", "gelu_tanh"):
        got = contracts.experts(x, ids, w, gu, dn, activation)
        want = ref.moe_block(x.tolist(), ids.tolist(), w.tolist(), lambda e: gu[e].tolist(), lambda e: dn[e].tolist(),
                             I, activation=activation)
        assert float((got - torch.tensor(want, dtype=torch.float64)).abs().max()) < 1e-12, activation


@pytest.mark.parametrize("softmax,before,renormalize,bias", [(True, False, False, False), (True, True, True, True),
                                                             (False, True, True, False), (False, False, False, True)])
def test_routing_is_the_techniques_route(softmax, before, renormalize, bias):
    ref = _module("inference/techniques/experts/heterogeneous-expert-dispatch/code/routing.py")
    g = torch.Generator().manual_seed(3)
    logits = torch.randn(5, 16, generator=g, dtype=torch.float64)
    logits[2, 3] = logits[2, 9]   # a tie: the lower id first
    b = torch.randn(16, generator=g, dtype=torch.float64) * 0.1 if bias else None
    ids, w = contracts.routing(logits, 4, b, softmax, before, renormalize, 0.01, 0.5, 2.5)
    rule = ref.Routing(scoring="softmax" if softmax else "sigmoid", score_before_selection=before,
                       correction_bias=bias, renormalize=renormalize, renormalize_epsilon=0.01, renormalize_floor=0.5,
                       scaling=2.5)
    if renormalize and softmax and not before:
        return
    for t in range(5):
        want_ids, want_w = ref.route(logits[t].tolist(), 4, rule, b.tolist() if bias else None)
        assert tuple(ids[t].tolist()) == want_ids
        assert max(abs(a - c) for a, c in zip(w[t].tolist(), want_w)) < 1e-12


def test_gated_delta_is_the_techniques_sequential_recurrence():
    ref = _module("inference/techniques/recurrence/recurrent-state-recovery/code/chunked_prefill.py")
    g = torch.Generator().manual_seed(4)
    T, dk, dv = 6, 4, 3
    q = torch.nn.functional.normalize(torch.randn(T, 1, dk, generator=g, dtype=torch.float64), dim=-1)
    k = torch.nn.functional.normalize(torch.randn(T, 1, dk, generator=g, dtype=torch.float64), dim=-1)
    v = torch.randn(T, 1, dv, generator=g, dtype=torch.float64)
    gl = -torch.rand(T, 1, generator=g, dtype=torch.float64) * 0.3
    b = torch.rand(T, 1, generator=g, dtype=torch.float64)
    S0 = torch.randn(1, dk, dv, generator=g, dtype=torch.float64) * 0.1
    out, S = contracts.gated_delta(q, k, v, gl, b, S0, 0.7)
    geometry = ref.Geometry(dk, dv, query_scale=0.7, normalize_keys=False, normalize_queries=False)
    steps = [ref.Step(tuple(q[t, 0].tolist()), tuple(k[t, 0].tolist()), tuple(v[t, 0].tolist()), float(gl[t, 0]),
                      float(b[t, 0])) for t in range(T)]
    outs, matrix = ref.sequential(tuple(S0[0].reshape(-1).tolist()), steps, geometry)
    assert float((out[:, 0] - torch.tensor(outs, dtype=torch.float64)).abs().max()) < 1e-12
    assert float((S[0].reshape(-1) - torch.tensor(matrix, dtype=torch.float64)).abs().max()) < 1e-12


def test_ssd_and_conv_are_the_techniques_host_forms():
    ref = _module("inference/techniques/recurrence/recurrent-state-recovery/code/reference.py")
    g = torch.Generator().manual_seed(5)
    T, H, P, G, N = 5, 4, 3, 2, 6
    x = torch.randn(T, H, P, generator=g, dtype=torch.float64)
    dt = torch.rand(T, H, generator=g, dtype=torch.float64)
    a = -torch.rand(H, generator=g, dtype=torch.float64)
    bm, cm = torch.randn(T, G, N, generator=g, dtype=torch.float64), torch.randn(T, G, N, generator=g, dtype=torch.float64)
    S0, d = torch.randn(H, P, N, generator=g, dtype=torch.float64), torch.randn(H, generator=g, dtype=torch.float64)
    y, S = contracts.ssd(x, dt, bm, cm, a, d, S0)
    want_y, want_S = ref.ssd_sequential(S0.tolist(), x.tolist(), dt.tolist(), a.tolist(), bm.tolist(), cm.tolist(),
                                        d.tolist())
    assert float((y - torch.tensor(want_y, dtype=torch.float64)).abs().max()) < 1e-12 and float((S - torch.tensor(want_S, dtype=torch.float64)).abs().max()) < 1e-12
    rows, w, window = torch.randn(7, 5, generator=g), torch.randn(5, 4, generator=g), torch.randn(3, 5, generator=g)
    bias = torch.randn(5, generator=g)
    got = contracts.causal_conv(rows, w, window, bias)
    want = ref.causal_conv_rows(rows.double().tolist(), w.double().tolist(), window.double().tolist(),
                                bias.double().tolist())
    assert float((got - torch.tensor(want, dtype=torch.float64)).abs().max()) < 1e-12


def test_projection_gives_the_products_magnitudes():
    x, w = torch.tensor([[1.0, -2.0]]), torch.tensor([[3.0, 4.0], [-1.0, 0.5]])
    out, magnitude = contracts.projection(x, w)
    assert out.tolist() == [[-5.0, -2.0]] and magnitude.tolist() == [[11.0, 2.0]]
