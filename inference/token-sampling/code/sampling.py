"""Python mirror of the device token selection (same rules as sampling.cu).

The host oracle is Core's crit::core::Sampler; this mirror lets Python tools and the
worked example reproduce device tokens. Candidates rank by (logit descending, index
ascending). Greedy (temperature <= 0) returns the first. Otherwise the top_k
candidates (0 means 64) get weights exp((logit - best) / temperature); top-p keeps
the smallest rank prefix whose probability reaches top_p (at least one); Core's draw
u = uniform(seed, position, stream) picks the first kept token whose cumulative
weight exceeds u times the kept weight. The device computes weights in FP32, this
mirror in FP64, so a draw that lands within rounding of a boundary may differ.
"""
import math

MAX_TOP_K = 64
_WORD = 0xFFFFFFFF
_SAMPLE = 0   # crit::core::DrawPurpose::sample


def philox4x32(seed, position, stream=0, purpose=_SAMPLE):
    """Core's Philox4x32-10 block (crit/sampling.h): key = seed, counter = (position, stream, purpose)."""
    if not (0 <= seed < 1 << 64 and 0 <= position < 1 << 64 and 0 <= stream <= _WORD and 0 <= purpose <= _WORD):
        raise ValueError("seed and position are unsigned 64-bit, stream and purpose unsigned 32-bit")
    c0, c1, c2, c3 = position & _WORD, position >> 32, stream, purpose
    k0, k1 = seed & _WORD, seed >> 32
    for round_ in range(10):
        if round_:
            k0, k1 = (k0 + 0x9E3779B9) & _WORD, (k1 + 0xBB67AE85) & _WORD
        p0, p1 = 0xD2511F53 * c0, 0xCD9E8D57 * c2
        c0, c1, c2, c3 = (p1 >> 32) ^ c1 ^ k0, p1 & _WORD, (p0 >> 32) ^ c3 ^ k1, p0 & _WORD
    return c0, c1, c2, c3


def uniform(seed, position, stream=0):
    """Core's sampling draw in [0, 1) with 53 random bits (bit-identical to crit::core::uniform)."""
    word = philox4x32(seed, position, stream)
    return ((word[0] << 32 | word[1]) >> 11) * 2.0 ** -53


def candidates(logits, k):
    """Indices of the k best logits in rank order."""
    if not logits:
        raise ValueError("empty logits")
    if any(math.isnan(v) for v in logits):
        raise ValueError("NaN logit")
    return sorted(range(len(logits)), key=lambda i: (-logits[i], i))[:k]


def _kept(logits, temperature, top_p, top_k):
    if not temperature > 0:
        raise ValueError("nucleus needs a positive temperature")
    if not 0 < top_p <= 1:
        raise ValueError("top_p must be in (0, 1]")
    if top_k < 0:
        raise ValueError("top_k must be non-negative")
    k = max(1, min(top_k or MAX_TOP_K, MAX_TOP_K))
    best = candidates(logits, k)
    w = [math.exp((logits[i] - logits[best[0]]) / temperature) for i in best]
    total = math.fsum(w)
    keep, acc = len(best), 0.0
    for r, x in enumerate(w):
        acc += x / total
        if acc >= top_p:
            keep = r + 1
            break
    return best[:keep], w[:keep]


def nucleus(logits, temperature, top_p=1.0, top_k=0):
    """(token, probability) pairs the draw chooses from, in rank order."""
    tokens, w = _kept(logits, temperature, top_p, top_k)
    kept = math.fsum(w)
    return [(token, x / kept) for token, x in zip(tokens, w)]


def select(logits, temperature=0.0, top_p=1.0, top_k=0, seed=0, position=0, stream=0):
    """Selected token index for one logits row at draw `position` (committed outputs before it)."""
    if not temperature > 0:
        return candidates(logits, 1)[0]
    tokens, w = _kept(logits, temperature, top_p, top_k)
    target, c = uniform(seed, position, stream) * math.fsum(w), 0.0
    for token, x in zip(tokens, w):
        c += x
        if target < c:
            return token
    return tokens[-1]


if __name__ == "__main__":
    row = [2.0, 1.0, 2.0, -1.0, 0.5]
    print("greedy", select(row))
    print("nucleus", nucleus(row, 0.7, top_p=0.9))
    print("draws", [select(row, 0.7, 0.9, 0, seed=1, position=c) for c in range(8)])
