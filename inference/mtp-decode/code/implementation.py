"""Longest verified speculative prefix, bounded correction and committed-token accounting."""
from dataclasses import dataclass


@dataclass(frozen=True)
class CommitPlan:
    tokens: tuple[int, ...]
    accepted_drafts: int
    correction: int | None


def verify(drafts, target_tokens, remaining, stop_tokens=()):
    drafts, target_tokens = tuple(drafts), tuple(target_tokens)
    if type(remaining) is not int or remaining < 0:
        raise ValueError("nonnegative remaining output budget required")
    if len(target_tokens) != len(drafts) + 1:
        raise ValueError("one target prediction per draft plus a bonus prediction required")
    if any(type(t) is not int or not 0 <= t < 1 << 31 for t in drafts + target_tokens):
        raise ValueError("valid native vocabulary IDs required")
    accepted, tokens = 0, []
    for draft, target in zip(drafts, target_tokens):
        if len(tokens) == remaining:
            return CommitPlan(tuple(tokens), accepted, None)
        if draft != target:
            tokens.append(target)
            return CommitPlan(tuple(tokens), accepted, target)
        tokens.append(draft)
        accepted += 1
        if draft in stop_tokens:
            return CommitPlan(tuple(tokens), accepted, None)
    correction = None
    if len(tokens) < remaining:
        correction = target_tokens[-1]
        tokens.append(correction)
    return CommitPlan(tuple(tokens), accepted, correction)


def commit(plan, apply):
    """apply updates only accepted/correction state; core publishes after success."""
    apply(plan.tokens, plan.accepted_drafts)
    return plan.tokens
