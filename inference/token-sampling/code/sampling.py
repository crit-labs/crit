"""Python mirror of the device token selection and verification (same rules as sampling.cu).

The host oracle is Core's crit::core::Sampler; this mirror lets Python tools and the
worked example reproduce device tokens. Processing follows Core's steps 2 to 5 (bias,
min_tokens suppression, repetition, frequency and presence over committed plus staged
outputs). Candidates rank by (value descending, index ascending). Greedy (temperature
<= 0) returns the first. Otherwise weights are exp((value - best) / temperature); top_k
(0 = the full vocabulary) keeps a rank prefix, top-p the shortest rank prefix whose
weight reaches top_p of the top-k weight (at least one), min_p the weights >= min_p;
Core's draw u = uniform(seed, position, stream) picks the first kept token whose
cumulative weight exceeds u times the kept weight. The device processes values in FP64
as Core does and computes the full law's weights in FP32, this mirror everything in
FP64, so a draw that lands within rounding of a boundary may differ.

`verify` is the speculative rule of enqueue_verify. Greedy drafts: every column writes the
target's own draw, so a draft is accepted exactly when it equals it (probability p_c(d)) and the
emitted text equals plain sampling. Sampled drafts (draft_laws): draft d at column c is accepted
with u_accept * q_c(d) < p_c(d), otherwise column c draws from max(p_c - q_c, 0) with
u_residual; the last column draws the bonus with u_sample, and a cut column (draft -1) takes the
target's own draw like the bonus.
"""
import math

MAX_TOP_K = 64   # the device's fast path (greedy rows and 1 <= top_k <= 64); results are Core's law
_WORD = 0xFFFFFFFF
SAMPLE, DRAFT, ACCEPT, RESIDUAL, BLOCK = 0, 1, 2, 3, 4   # crit::core::DrawPurpose


def philox4x32(seed, position, stream=0, purpose=SAMPLE):
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


def uniform(seed, position, stream=0, purpose=SAMPLE):
    """Core's draw in [0, 1) with 53 random bits (bit-identical to crit::core::uniform)."""
    word = philox4x32(seed, position, stream, purpose)
    return ((word[0] << 32 | word[1]) >> 11) * 2.0 ** -53


def candidates(logits, k):
    """Indices of the k best logits in rank order."""
    if not logits:
        raise ValueError("empty logits")
    if any(math.isnan(v) for v in logits):
        raise ValueError("NaN logit")
    return sorted(range(len(logits)), key=lambda i: (-logits[i], i))[:k]


def process(logits, *, bias=None, repetition=1.0, frequency=0.0, presence=0.0, counts=None, prompt=(),
            staged=(), min_tokens=0, suppress=(), outputs=0, yields=False):
    """Core's steps 2 to 5: values after bias, suppression and penalties.

    counts maps tokens to committed output counts; `staged` tokens count as outputs too (the drafts
    before a verify column); `outputs` is Core's outputs() + column for the min_tokens rule. `yields` skips
    suppression, as Core does for a constraint that allows only suppressed tokens (suppression_yields)."""
    n = dict(counts or {})
    for t in staged:
        n[t] = n.get(t, 0) + 1
    seen = set(prompt)
    out = []
    for t, x in enumerate(logits):
        v = float(x) + (bias or {}).get(t, 0.0)
        if v != -math.inf:
            if (n.get(t, 0) > 0 or t in seen) and repetition != 1.0:
                v = min(v / repetition, 1.7976931348623157e308) if v > 0 else max(v * repetition, -1.7976931348623157e308)
            if n.get(t, 0) > 0:
                v -= frequency * n[t] + presence
        out.append(v)
    if outputs < min_tokens and not yields:
        for t in suppress:
            if 0 <= t < len(out):
                out[t] = -math.inf
    return out


def law(values, temperature, top_p=1.0, top_k=0, min_p=0.0):
    """Kept (token, weight) pairs in rank order; a greedy law is [(argmax, 1.0)]."""
    if not 0 < top_p <= 1:
        raise ValueError("top_p must be in (0, 1]")
    if top_k < 0 or not 0 <= min_p <= 1:
        raise ValueError("top_k must be non-negative and min_p in [0, 1]")
    order = [t for t in candidates(values, len(values)) if values[t] != -math.inf]
    if not order:
        raise ValueError("no selectable token")
    if not temperature > 0:
        return [(order[0], 1.0)]
    if 0 < top_k < len(order):
        order = order[:top_k]
    w = [math.exp((values[t] - values[order[0]]) / temperature) for t in order]
    if top_p < 1:
        target, acc = top_p * math.fsum(w), 0.0
        for r, x in enumerate(w):
            acc += x
            if acc >= target:
                order, w = order[:r + 1], w[:r + 1]
                break
    kept = [(t, x) for t, x in zip(order, w) if not (min_p > 0 and x < min_p)]
    return kept


def nucleus(logits, temperature, top_p=1.0, top_k=0, min_p=0.0):
    """(token, probability) pairs the draw chooses from, in rank order."""
    if not temperature > 0:
        raise ValueError("nucleus needs a positive temperature")
    kept = law(logits, temperature, top_p, top_k, min_p)
    total = math.fsum(x for _, x in kept)
    return [(token, x / total) for token, x in kept]


def walk(pairs, target):
    """Core's CDF walk: the first token whose cumulative weight exceeds target, else the last."""
    c = 0.0
    for token, x in pairs:
        c += x
        if target < c:
            return token
    return pairs[-1][0] if pairs else None


def select(logits, temperature=0.0, top_p=1.0, top_k=0, seed=0, position=0, stream=0, min_p=0.0):
    """Selected token index for one values row at draw `position` (committed outputs before it)."""
    kept = law(logits, temperature, top_p, top_k, min_p)
    if not temperature > 0:
        return kept[0][0]
    return walk(kept, uniform(seed, position, stream) * math.fsum(x for _, x in kept))


def verify(rows, drafts, temperature, top_p=1.0, top_k=0, min_p=0.0, seed=0, position=0, stream=0, draft_laws=None):
    """enqueue_verify's predictions for K drafts over K + 1 processed rows (column c already counts drafts[:c]).

    draft_laws, when given, holds K dicts token -> q probability the drafts were drawn from."""
    out = []
    for c, values in enumerate(rows):
        kept = law(values, temperature, top_p, top_k, min_p)
        total = math.fsum(x for _, x in kept)
        at = position + c
        # The bonus column, greedy drafts and a cut column (draft id < 0: the drafter stopped) take the target's own draw.
        if c == len(drafts) or not temperature > 0 or not draft_laws or drafts[c] < 0:
            out.append(kept[0][0] if not temperature > 0 else walk(kept, uniform(seed, at, stream) * total))
            continue
        d = drafts[c]
        p = {t: x / total for t, x in kept}
        q = draft_laws[c]
        if uniform(seed, at, stream, ACCEPT) * q.get(d, 0.0) < p.get(d, 0.0):
            out.append(d)
            continue
        residual = [(t, max(p[t] - q.get(t, 0.0), 0.0)) for t, _ in kept]
        residual = [(t, x) for t, x in residual if x > 0]
        mass = math.fsum(x for _, x in residual)
        out.append(walk(residual, uniform(seed, at, stream, RESIDUAL) * mass) if residual else d)
    return out


if __name__ == "__main__":
    row = [2.0, 1.0, 2.0, -1.0, 0.5]
    print("greedy", select(row))
    print("nucleus", nucleus(row, 0.7, top_p=0.9))
    print("draws", [select(row, 0.7, 0.9, 0, seed=1, position=c) for c in range(8)])
    print("verify", verify([row, row, row], [0, 3], 0.7, seed=1))
