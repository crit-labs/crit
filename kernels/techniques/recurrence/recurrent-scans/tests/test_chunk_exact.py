"""The chunk-aligned exact form of delta-prefill-tc: a token's output depends on its 64-token chunk (aligned to absolute
positions) only through the chunk's tokens up to it and the state at the chunk's start, with the same bits however
many of the chunk's later tokens the launch also holds. So a decode step that recomputes its chunk so far from the
boundary state gives the prompt's bits, and a prompt continued token by token gives the bits of one prompt."""
import pytest
import torch

from crit_kernels import build as kbuild

rs = kbuild.load("recurrent-scans")
dev = torch.device("cuda")
CHUNK = 64


def inputs(T, kh, vh, dk, dv, seed):
    g = torch.Generator(device="cpu").manual_seed(seed)
    q = torch.nn.functional.normalize(torch.randn(T, kh, dk, generator=g), dim=-1)
    k = torch.nn.functional.normalize(torch.randn(T, kh, dk, generator=g), dim=-1)
    v = torch.randn(T, vh, dv, generator=g)
    logd = -torch.rand(T, vh, generator=g) * 0.2
    beta = torch.rand(T, vh, generator=g)
    s0 = torch.randn(vh, dk, dv, generator=g) * 0.1
    return [t.to(dev) for t in (q, k, v, logd, beta, s0)]


def tc(xs, state, lo, hi, scale):
    q, k, v, logd, beta = xs
    return rs.gated_delta_prefill(q[lo:hi], k[lo:hi], v[lo:hi], logd[lo:hi], beta[lo:hi], state, scale=scale,
                                  tensor_cores=True)


@pytest.mark.parametrize("dk,dv,kh,vh", [(128, 128, 16, 32), (128, 128, 2, 6), (64, 48, 1, 3)])
@pytest.mark.parametrize("T", [64, 130, 200])
def test_chunk_recompute_gives_the_prompts_bits(dk, dv, kh, vh, T):
    *xs, s0 = inputs(T, kh, vh, dk, dv, seed=T + dk + vh)
    scale = dk ** -0.5
    full_state = s0.clone()
    full = tc(xs, full_state, 0, T, scale)
    boundary = s0.clone()
    for t in range(T):
        b = t // CHUNK * CHUNK
        state = boundary.clone()
        out = tc(xs, state, b, t + 1, scale)
        assert torch.equal(out[-1], full[t]), f"token {t}"
        if (t + 1) % CHUNK == 0:
            boundary = state
    assert torch.equal(state, full_state)


def boundary_states(xs, s0, T, scale):
    """The prompt's state at every CHUNK-aligned position up to T (index m: after m chunks)."""
    found, state = [s0.clone()], s0.clone()
    for m in range(T // CHUNK):
        tc(xs, state, m * CHUNK, (m + 1) * CHUNK, scale)
        found.append(state.clone())
    return found


@pytest.mark.parametrize("dk,dv,kh,vh", [(128, 128, 16, 48), (64, 48, 1, 3)])
def test_chunk_rows_give_every_sequence_its_prompts_bits(dk, dv, kh, vh):
    """Sequences at other points of their chunks in one launch: decode rows (the chunk so far plus the new token),
    a verify window that crosses a chunk boundary, a row that completes its chunk; outputs against each sequence's
    full prompt, the boundary states written for complete chunks only, a CUDA graph replay with the same bits."""
    scale = dk ** -0.5
    lengths = (1, 40, 64, 100, 130, 192)   # tokens of each sequence's prompt; its row ends at its last token
    window = (1, 1, 1, 5, 5, 3)            # tokens the row adds (decode 1, verify windows; the fifth crosses 128)
    seqs = [inputs(T, kh, vh, dk, dv, seed=3 * i + dk) for i, T in enumerate(lengths)]
    fulls, bounds = [], []
    for (*xs, s0), T in zip(seqs, lengths):
        st = s0.clone()
        fulls.append(tc(xs, st, 0, T, scale))
        bounds.append(boundary_states(xs, s0, T, scale))
    # each row: from the last boundary before its window's first token to its last token
    packed, meta, widest = [[] for _ in range(5)], [], 0
    states_in = torch.stack([bounds[i][(T - w) // CHUNK] for i, (T, w) in enumerate(zip(lengths, window))])
    first = 0
    for i, ((*xs, s0), T, w) in enumerate(zip(seqs, lengths, window)):
        b = (T - w) // CHUNK * CHUNK
        for j, x in enumerate(xs):
            packed[j].append(x[b:T])
        meta.append([first, T - b, i, i])
        widest = max(widest, T - b)
        first += T - b
    packed = [torch.cat(p) for p in packed]
    meta_t = torch.tensor(meta, dtype=torch.int32, device=dev)
    states_out = torch.full_like(states_in, float("nan"))
    out = rs.gated_delta_chunk_rows(*packed, meta_t, states_in, states_out, scale=scale, widest=widest)
    for i, (T, w) in enumerate(zip(lengths, window)):
        f, n = meta[i][0], meta[i][1]
        b = T - n
        assert torch.equal(out[f:f + n], fulls[i][b:T]), f"sequence {i}"
        done = T // CHUNK
        if done > b // CHUNK:   # the row completed a chunk: the state at its last complete boundary
            assert torch.equal(states_out[i], bounds[i][done]), f"sequence {i} state"
        else:
            assert torch.isnan(states_out[i]).all(), f"sequence {i} state written"
    # slot_out < 0 writes nothing; a captured replay gives the same bits
    meta_t[:, 3] = -1
    again = torch.full_like(states_in, float("nan"))
    graph_out = torch.empty_like(out)
    rs.gated_delta_chunk_rows(*packed, meta_t, states_in, again, scale=scale, widest=widest, out=graph_out)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        rs.gated_delta_chunk_rows(*packed, meta_t, states_in, again, scale=scale, widest=widest, out=graph_out)
    graph_out.zero_()
    g.replay()
    torch.cuda.synchronize()
    assert torch.equal(graph_out, out)
    assert torch.isnan(again).all()


def test_chunk_rows_decode_a_prompt_token_by_token_in_place():
    """A decode loop on chunk rows alone: the sequence keeps its boundary state (written in place when a step completes
    a chunk) and its chunk's tokens; every step's output has the full prompt's bits."""
    dk, dv, kh, vh, T = 128, 128, 4, 8, 150
    *xs, s0 = inputs(T, kh, vh, dk, dv, seed=11)
    scale = dk ** -0.5
    st = s0.clone()
    full = tc(xs, st, 0, T, scale)
    states = s0.clone()[None]
    for t in range(T):
        b = t // CHUNK * CHUNK
        meta = torch.tensor([[0, t + 1 - b, 0, 0]], dtype=torch.int32, device=dev)
        out = rs.gated_delta_chunk_rows(*(x[b:t + 1] for x in xs), meta, states, scale=scale)
        assert torch.equal(out[-1], full[t]), f"token {t}"
    st = s0.clone()
    tc(xs, st, 0, T // CHUNK * CHUNK, scale)
    assert torch.equal(states[0], st)
