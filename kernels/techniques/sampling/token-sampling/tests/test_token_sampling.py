"""crit's kernel token-sampling on its own: greedy rows on the fast path are torch's argmax (the lower id on ties); the
full law's draws are the same bits spread over every PARTS (8, 16, 32) and in one block a row (CRIT_SAMPLING_ONE_BLOCK),
in a CUDA graph too; verify rounds run; the Request rows pack Core's rules (a 64-bit seed). Core's draws themselves are
checked against Core's sampler by the C++ suites (tests/inference/token_law_tests.cpp, token_sampling_tests.cpp) and
the token-sampling plugin's tests."""
import os

import pytest
import torch

from crit_kernels import build as kbuild

ts = kbuild.load("token-sampling")
dev = torch.device("cuda")
VOCAB = 50000
PARTS = [{"PARTS": 8}, {"PARTS": 16}, {"PARTS": 32}]


def rows(n, seed, scale=0.5):
    return (torch.randn(n, VOCAB, generator=torch.Generator().manual_seed(seed)) * scale).to(dev)


def requests(n, **rule):
    return torch.cat([ts.pack_request(seed=900 + r, position=r, **rule) for r in range(n)]).to(dev)


def draw(module, x, req):
    out = torch.full((x.shape[0],), -7, dtype=torch.int32, device=dev)
    scratch = torch.empty(module.scratch_bytes(VOCAB, x.shape[0]), dtype=torch.uint8, device=dev)
    module.sample(x, VOCAB, req, x.shape[0], 1, 0, 0, out, None, None, scratch, 0, None, None, None, None, None)
    return out


def test_greedy_rows_are_the_argmax():
    x = rows(9, 1)
    x[3, 100] = x[3, 7] = x[3].max() + 1.0   # a tie: the lower id
    assert torch.equal(draw(ts, x, requests(9, temperature=0.0)), torch.argmax(x, dim=1).to(torch.int32))


@pytest.mark.parametrize("rule", [dict(temperature=1.0, top_p=0.95), dict(temperature=0.7, top_k=200, top_p=0.9),
                                  dict(temperature=1.0, min_p=0.05)])
def test_the_full_law_has_the_same_bits_over_every_parts_and_in_one_block(rule):
    x, req = rows(4, 2), requests(4, **rule)
    want = draw(ts, x, req)
    assert bool(((want >= 0) & (want < VOCAB)).all())
    os.environ["CRIT_SAMPLING_ONE_BLOCK"] = "1"
    try:
        assert torch.equal(draw(ts, x, req), want)
    finally:
        os.environ.pop("CRIT_SAMPLING_ONE_BLOCK")
    kn = kbuild.load("token-sampling", configs={"spread-law": PARTS})
    for i in range(len(PARTS) + 1):
        kbuild.select(kn, "spread-law", i)
        assert torch.equal(draw(kn, x, req), want), PARTS[i - 1] if i else "default"
    kbuild.select(kn, "spread-law", -1)


def test_a_captured_draw_replays_the_eager_bits():
    x, req = rows(3, 3), requests(3, temperature=1.0, top_p=0.95)
    want = draw(ts, x, req)
    out = torch.empty(3, dtype=torch.int32, device=dev)
    scratch = torch.empty(ts.scratch_bytes(VOCAB, 3), dtype=torch.uint8, device=dev)
    call = lambda: ts.sample(x, VOCAB, req, 3, 1, 0, 0, out, None, None, scratch, 0, None, None, None, None,  # noqa: E731
                             None)
    call()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        call()
    out.fill_(-1)
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(out, want)


def test_a_verify_round_of_greedy_drafts_predicts_the_targets_draws():
    count, columns = 2, 3
    x = rows(count * columns, 4)
    req = requests(count, temperature=0.0)
    drafts = torch.argmax(x, dim=1).view(count, columns)[:, :columns - 1].to(torch.int32).contiguous()
    predictions = torch.full((count, columns), -7, dtype=torch.int32, device=dev)
    scratch = torch.empty(ts.scratch_bytes(VOCAB, count * columns), dtype=torch.uint8, device=dev)
    ts.verify(x, VOCAB, req, count, columns, drafts, None, None, predictions, scratch, 0, None, None, None)
    assert torch.equal(predictions.view(-1), torch.argmax(x, dim=1).to(torch.int32))


def test_requests_pack_the_rules_and_a_64_bit_seed():
    shape = ts.layout()
    assert shape["max_top_k"] == 64 and shape["request"] > shape["position"] >= 0
    big = ts.pack_request(temperature=1.0, seed=(1 << 64) - 1, position=5)
    assert big.dtype == torch.uint8 and big.numel() == shape["request"]
    at = shape["position"]
    assert int.from_bytes(bytes(big[at:at + 8].tolist()), "little") == 5
