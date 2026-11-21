"""Two serving contracts of a model plugin, kept here once so that no model plugin writes them again.

Exact rows (RowGroups). A batched engine's reply must not depend on the requests beside it, and a prompt row's bits
must follow from the prompt alone: every prompt chunk is multiplied alone with `prompt=True` (the prompt rows'
arithmetic at every length), and one-token rows (decode rows, a round's columns, draft steps) in calls of at most
`invariant_rows(projections)` rows, so each row has the bits it has alone (docs/PLUGINS.md, row invariance).

Prompt spans' first guess (fit). Core cancels a request between steps, and crit serve retires the engine when a
cancel does not settle within its grace (one second): every prompt step must finish well inside it, at every context
the engine serves. The spans are Core's prompt-step budget (crit_plugins.steps.StepBudget): it times every prompt step
the engine runs and sizes each chunk by what such steps took at that depth. Its first guess is this model's
calibration at load: three timed prompt steps (a short and a full chunk at position 0, a full chunk at the deepest
position the engine serves, its KV pages written with zeros) and their peaks, fitted as

    seconds(rows, position) = c + a * rows + b * rows * (position + rows)
    bytes(rows)             = m * rows

a measured decision (crit_plugins.decisions.settle) that `crit plugins decisions` records in the package. Where the
engine asks its prompt rows below row-exact (a chunk's boundaries then change its rows' bits), Core keeps the
calibration's spans, so a prompt's chunk boundaries are the same in every process that replays the record.
"""

from crit_plugins.steps import GRACE, QUANTUM, SECONDS  # noqa: F401 - the spans' units, Core's (crit_plugins.steps)


class RowGroups:
    """The token ranges a step's matrices multiply apart, and the matmul over them."""

    def __init__(self, projections):
        from crit_plugins.device import invariant_rows
        self.proj = projections
        self.decode_rows = invariant_rows(projections)

    def of(self, batch, via_prompt=()):
        """[(first, end, prompt)] of a StepBatch, RoundBatch or PromptBatch: each prompt chunk alone, one-token rows
        decode_rows at a time; the one-token rows `via_prompt` names (the batch's row indices: decode rows of a
        request that asks decode_via_prompt) each alone as a one-token prompt chunk, the prompt rows' arithmetic."""
        step = self.decode_rows
        if getattr(batch, "prompt", False):   # a captured prompt chunk: one chunk, its bucket's rows
            return [(0, batch.tokens_t.shape[0], True)]
        if batch.round:
            return [(first, min(first + step, batch.tokens), False) for first in range(0, batch.tokens, step)]
        groups, run = [], None
        for r in range(batch.rows):
            first, end = batch.starts[r], batch.starts[r + 1]
            if end - first > 1 or r in via_prompt:
                groups.append((first, end, True))
                run = None
            elif run is not None and run[1] == first and run[1] - run[0] < step:
                run = groups[-1] = (run[0], end, False)
            else:
                run = (first, end, False)
                groups.append(run)
        return groups

    def rows(self, count):
        """Groups of `count` one-token rows (logits rows, draft steps)."""
        step = self.decode_rows
        return [(f, min(f + step, count), False) for f in range(0, count, step)]

    def matmul(self, x, matrix, groups, out_dtype=None):
        if len(groups) == 1:
            first, end, prompt = groups[0]
            return self.proj.matmul(x, matrix, out_dtype=out_dtype, prompt=prompt)
        import torch
        return torch.cat([self.proj.matmul(x[first:end], matrix, out_dtype=out_dtype, prompt=prompt)
                          for first, end, prompt in groups])


from crit_plugins.steps import fit  # noqa: E402,F401 - the spans' cost model, Core's (Steps.calibrated fits with it)
