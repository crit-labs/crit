"""Core's sampling law on the device (crit's kernel token-sampling, which crit's token-sampling plugin loads for its
captured rounds, built on this GPU with the engine's tuning or the default knobs): one draw per row of logits by Core's
rules and Philox draws, rejection-sampling verify predictions, log-probabilities, the fast path for greedy and top-k
rows and the full law for the rest (spread over several blocks a row where the grid fits, sampling_cuda/sampling.cu).
The functions keep the plugin's call shapes: an empty tensor stands for an absent buffer.

    requests = pack_request(temperature, top_p, min_p, top_k, seed, stream, position).cuda()
    sample(logits, vocab, requests, count, columns, position_base, purpose, out, sparse, dense, scratch, lp_n,
           chosen, top_ids, top_values, tree, tree_tokens)
"""
import torch

from ._ops import ops

__all__ = ["layout", "logprob_scratch_bytes", "logprobs", "pack_request", "sample", "scratch_bytes", "verify"]


def pack_request(temperature=0.0, top_p=1.0, min_p=0.0, top_k=0, seed=0, stream=0, position=0) -> torch.Tensor:
    """One Request row of these rules (no bias, penalties or suppression) as CPU bytes: a round stages it on the
    device. seed: its 64 bits (an int of up to 2^64 - 1)."""
    seed = int(seed) & ((1 << 64) - 1)
    signed = seed - (1 << 64) if seed >= 1 << 63 else seed
    return ops.pack_request(float(temperature), float(top_p), float(min_p), int(top_k), signed, int(stream),
                            int(position))


def layout() -> dict:
    """The device structs' layout: {request, position, sparse, sparse_ids, sparse_probabilities, max_top_k}."""
    names = ("request", "position", "sparse", "sparse_ids", "sparse_probabilities", "max_top_k")
    return dict(zip(names, (int(v) for v in ops.layout())))


_EMPTY = {}


def _t(x, like):
    """x, or for None the empty tensor that stands for an absent buffer (on like's device)."""
    if x is not None:
        return x
    key = str(like.device)
    if key not in _EMPTY:
        _EMPTY[key] = torch.empty(0, device=like.device)
    return _EMPTY[key]


def scratch_bytes(vocab, rows) -> int:
    return int(ops.scratch_bytes(int(vocab), int(rows)))


def logprob_scratch_bytes(vocab, rows, n) -> int:
    return int(ops.logprob_scratch_bytes(int(vocab), int(rows), int(n)))


def verify(logits, vocab, requests, count, columns, drafts, sparse, dense, predictions, scratch, lp_n, chosen, top_ids,
           top_values):
    e = lambda x: _t(x, logits)  # noqa: E731
    ops.verify(logits, int(vocab), requests, int(count), int(columns), e(drafts), e(sparse), e(dense), predictions,
               e(scratch), int(lp_n), e(chosen), e(top_ids), e(top_values))


def sample(logits, vocab, requests, count, columns, position_base, purpose, out, sparse, dense, scratch, lp_n, chosen,
           top_ids, top_values, tree, tree_tokens):
    e = lambda x: _t(x, logits)  # noqa: E731
    ops.sample(logits, int(vocab), requests, int(count), int(columns), int(position_base), int(purpose), out,
               e(sparse), e(dense), e(scratch), int(lp_n), e(chosen), e(top_ids), e(top_values), e(tree),
               e(tree_tokens))


def logprobs(logits, vocab, chosen, n, chosen_values, top_ids, top_values, scratch, given):
    e = lambda x: _t(x, logits)  # noqa: E731
    ops.logprobs(logits, int(vocab), chosen, int(n), e(chosen_values), e(top_ids), e(top_values), e(scratch),
                 bool(given))
