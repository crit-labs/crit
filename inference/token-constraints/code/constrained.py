"""Host reference for constrained speculative laws: mask first, then filter; accept and repair with those laws.

Mirrors constrained_logits in constrained_state.h (mtp-decode's Distribution filters applied to the allowed ids).
`allowed` is a matcher's sorted allowed set (Core's TokenMatcher::allowed()).
"""
import math


def constrained_law(scores, allowed, temperature=1.0, top_k=0, top_p=1.0, min_p=0.0):
    """Full-vocabulary probabilities with zero mass outside `allowed`; filters see only the allowed ids."""
    allowed = list(allowed)
    if not allowed or allowed != sorted(set(allowed)) or allowed[0] < 0 or allowed[-1] >= len(scores):
        raise ValueError("allowed ids must be nonempty, sorted, unique and in the vocabulary")
    if not all(math.isfinite(s) for s in scores) or top_k > len(scores):
        raise ValueError("finite logits and top_k within the vocabulary required")
    compact = [scores[i] for i in allowed]
    best = max(range(len(compact)), key=lambda i: (compact[i], -i))  # lowest id on ties
    weights = [0.0] * len(compact)
    if temperature == 0:
        weights[best] = 1.0
    else:
        order = sorted(range(len(compact)), key=lambda i: (-compact[i], i))
        order = order[:min(top_k, len(compact))] if top_k else order
        for i in order:
            weights[i] = math.exp((compact[i] - compact[best]) / temperature)
        if top_p < 1:
            target, accumulated = top_p * sum(weights), 0.0
            for rank, i in enumerate(order):
                accumulated += weights[i]
                if accumulated >= target:
                    for j in order[rank + 1:]:
                        weights[j] = 0.0
                    break
        weights = [w if w >= min_p else 0.0 for w in weights]
    total = sum(weights)
    law = [0.0] * len(scores)
    for i, w in zip(allowed, weights):
        law[i] = w / total
    return law


def accept_probability(target, draft, token):
    """Speculative acceptance of a drafted token: min(1, p/q) with both laws already constrained."""
    return 1.0 if draft[token] <= target[token] else target[token] / draft[token]


def residual(target, draft):
    """The repair law after a rejection: normalized max(0, p - q). Both laws are zero outside the language, so the
    repair stays inside it; renormalizing unconstrained laws afterwards would not."""
    positive = [max(0.0, p - q) for p, q in zip(target, draft)]
    total = sum(positive)
    if total <= 0:
        raise ValueError("no residual mass: the draft law equals the target law")
    return [p / total for p in positive]
