"""Two serving contracts of a model plugin, kept here once so that no model plugin writes them again.

Exact rows (RowGroups). A batched engine's reply must not depend on the requests beside it, and a prompt row's bits
must follow from the prompt alone: every prompt chunk (a prefill row of the batch, however short) is multiplied alone
with `prompt=True` (the prompt rows' arithmetic at every length), and decode tokens (decode rows and verify windows
of any width, a round's columns, draft steps) in calls of at most `invariant_rows(projections)` rows, so each row has
the bits it has alone, in a step as in a captured round (docs/PLUGINS.md, row invariance). The phase is the batch's
(StepBatch.prompt_row), never a row's length.

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

    def of(self, batch):
        """[(first, end, prompt)] of a StepBatch, RoundBatch or PromptBatch, by each row's phase: a prompt chunk
        alone (a prefill row of any length: a one-token chunk, a prompt-cache hit's remainder, and a decode row of a
        request that asks decode_via_prompt, which the model builds a prefill row), decode tokens decode_rows at a
        time (decode rows and verify windows of any width, cut as a captured round cuts its columns: a window's tokens
        the bits they have in the round): Core's crit_plugins.device.row_groups."""
        from crit_plugins.device import row_groups
        return row_groups(batch, self.decode_rows)

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
