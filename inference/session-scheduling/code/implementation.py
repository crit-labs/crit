"""Extension-owned chunk cursor and borrowed progress staging; core owns publication."""
from dataclasses import dataclass


def chunk_plan(length, chunk_limit, fold_limit=0):
    """Chunk lengths for a prompt of `length` tokens: full chunks of `chunk_limit`, except that a remainder of at
    most `fold_limit` tokens joins the previous chunk instead of running alone. A near-empty last chunk re-reads
    every weight of the model for a handful of tokens; folding it costs one slightly larger chunk instead."""
    if type(chunk_limit) is not int or chunk_limit <= 0 or type(fold_limit) is not int or fold_limit < 0:
        raise ValueError("positive chunk limit and non-negative fold limit required")
    if type(length) is not int or length < 0:
        raise ValueError("non-negative prompt length required")
    plan, left = [], length
    while left > 0:
        take = min(left, chunk_limit)
        if 0 < left - take <= fold_limit:
            take = left
        plan.append(take)
        left -= take
    return plan


class PrefillCursor:
    def __init__(self, prompt, chunk_limit, fold_limit=0):
        self.prompt = tuple(prompt)
        if type(chunk_limit) is not int or chunk_limit <= 0:
            raise ValueError("positive chunk limit required")
        if type(fold_limit) is not int or fold_limit < 0:
            raise ValueError("non-negative fold limit required")
        self.chunk_limit = chunk_limit
        self.fold_limit = fold_limit
        self.committed = 0
        self.pending = None

    def next_chunk(self, budget):
        if self.pending is not None:
            raise ValueError("commit the borrowed chunk before advancing")
        if type(budget) is not int or budget <= 0:
            raise ValueError("positive work budget required")
        end = min(len(self.prompt), self.committed + min(budget, self.chunk_limit))
        rest = len(self.prompt) - end
        # Fold a short remainder into this chunk when the work budget allows it.
        if 0 < rest <= self.fold_limit and end + rest - self.committed <= budget:
            end += rest
        self.pending = end
        return self.prompt[self.committed:end]

    def commit(self):
        if self.pending is None:
            raise ValueError("no staged chunk")
        self.committed = self.pending
        self.pending = None

    def rollback(self):
        self.pending = None

    @property
    def complete(self):
        return self.committed == len(self.prompt)


@dataclass(frozen=True)
class StagedProgress:
    tokens: tuple[int, ...]
    charged_bytes: int


def stage_progress(borrowed_tokens, output_budget, reserve, readiness=None):
    if readiness is not None:
        readiness.wait()
    tokens = tuple(borrowed_tokens)  # copy before model commit invalidates storage
    if type(output_budget) is not int or output_budget < 0 or len(tokens) > output_budget:
        raise ValueError("progress exceeds admitted output budget")
    if any(type(t) is not int or not -(1 << 31) <= t < 1 << 31 for t in tokens):
        raise ValueError("native token representation exceeded")
    owner = reserve(4 * len(tokens))
    return StagedProgress(tokens, 4 * len(tokens)), owner
