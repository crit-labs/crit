"""A NaN in one token's query reaches that token's output and no other, on every structure (the split launch, fused,
flash, flash fused, the CUDA tile split and fused, and the three prompt-chunk paths): a fault upstream shows instead
of turning into zeros."""
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_paged_kv_attention import Cache, _q, dev, pk  # noqa: E402

WINDOWS = [{}, {"fused": True}, {"flash": True}, {"flash": True, "fused": True}, {"flash": "cuda"},
           {"flash": "cuda", "fused": True}]


@pytest.mark.parametrize("family", WINDOWS, ids=["split", "fused", "flash", "flash-fused", "cuda", "cuda-fused"])
def test_a_nan_query_reaches_its_own_tokens_output_only(family):
    cache = Cache("bf16", 2, 128, 128, 16, [300, 300])
    q = _q(2, 8, 128, 1)
    clean = cache.attend(q, [0, 1], [0, 1], [200, 200], 1, **family)
    q[0, 0, 0] = float("nan")
    lse = torch.empty(2, 8, device=dev)
    out = cache.attend(q, [0, 1], [0, 1], [200, 200], 1, lse=lse, **family)
    assert torch.isnan(out[0, 0]).all() and torch.isnan(lse[0, 0])
    assert torch.equal(out[0, 1:], clean[0, 1:]) and torch.equal(out[1], clean[1])


@pytest.mark.parametrize("flash", [False, True, "cuda"])
def test_a_nan_query_in_a_prompt_chunk_reaches_its_own_tokens_output_only(flash):
    cache = Cache("bf16", 2, 128, 128, 16, [300])
    q = _q(4, 8, 128, 2).to(torch.bfloat16)
    clean = pk.prefill(q, cache.k, cache.v, "bf16", cache.tables[0], 200, 16, flash=flash)
    q[1, 0, 0] = float("nan")
    out = pk.prefill(q, cache.k, cache.v, "bf16", cache.tables[0], 200, 16, flash=flash)
    assert torch.isnan(out[1, 0]).all()
    assert torch.equal(out[0], clean[0]) and torch.equal(out[2:], clean[2:]) and torch.equal(out[1, 1:], clean[1, 1:])
