"""prefill_rows (several sequences' prompt chunks in one launch of the prompt reader) against prefill() row by row:
every token the bits its row's own launch gives it, over every page kind, with log-sum-exps, fresh rows, sinks,
sliding windows and INT8 queries, in rows of one token and of many, at position 0 and past it."""
import pytest
import torch

from test_paged_kv_attention import KINDS, Cache, _q, dev, pk


def _rows(lengths, counts, positions):
    starts, at = [], 0
    for n in counts:
        starts.append(at)
        at += n
    return starts, at


@pytest.mark.kernels_ci
@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("hq,hkv,d", [(8, 2, 128), (24, 2, 256), (16, 4, 64)])
def test_one_launch_gives_each_row_its_own_launchs_bits(kind, hq, hkv, d):
    lengths = [900, 300, 64, 1200]
    cache = Cache(kind, hkv, d, d, 16, lengths, seed=3)
    counts, positions = [100, 1, 64, 257], [800, 299, 0, 943]
    starts, T = _rows(lengths, counts, positions)
    q_dtype = torch.float16 if kind == "fp16" else torch.bfloat16
    q = _q(T, hq, d, 4).to(q_dtype)
    g = torch.Generator().manual_seed(5)
    fk = torch.randn(T, hkv, d, generator=g).to(dev).to(q_dtype)
    fv = torch.randn(T, hkv, d, generator=g).to(dev).to(q_dtype)
    sinks = torch.randn(hq, generator=g).to(dev)
    common = dict(key_scales=cache.ks, value_scales=cache.vs, multipliers=cache.multipliers)
    for fresh, given, sliding, int8 in ((False, None, 0, False), (True, None, 0, False), (False, sinks, 0, False),
                                        (False, None, 100, False), (True, sinks, 33, False),
                                        (False, None, 0, kind == "int8")):
        out, lse = torch.empty_like(q), torch.empty(T, hq, device=dev)
        pk.prefill_rows(q, cache.k, cache.v, kind, cache.tables, starts, counts, positions, 16, out=out, lse=lse,
                        fresh=(fk, fv) if fresh else None, sinks=given, sliding=sliding, int8_qk=int8, **common)
        for s, n, p, r in zip(starts, counts, positions, range(len(counts))):
            span = slice(s, s + n)
            want, want_lse = torch.empty_like(q[span]), torch.empty(n, hq, device=dev)
            pk.prefill(q[span], cache.k, cache.v, kind, cache.tables[r], p, 16, out=want, lse=want_lse,
                       fresh=(fk[span], fv[span]) if fresh else None, sinks=given, sliding=sliding, int8_qk=int8,
                       **common)
            assert torch.equal(out[span], want) and torch.equal(lse[span], want_lse), (kind, r, fresh, sliding, int8)


@pytest.mark.kernels_ci
def test_rows_are_checked():
    cache = Cache("bf16", 2, 128, 128, 16, [100, 100])
    q = _q(10, 8, 128, 1).to(torch.bfloat16)
    with pytest.raises(ValueError, match="tables are int32"):
        pk.prefill_rows(q, cache.k, cache.v, "bf16", cache.tables[:1], [0, 5], [5, 5], [0, 0], 16)
    with pytest.raises(ValueError, match="lie in q"):
        pk.prefill_rows(q, cache.k, cache.v, "bf16", cache.tables, [0, 6], [5, 5], [0, 0], 16)
