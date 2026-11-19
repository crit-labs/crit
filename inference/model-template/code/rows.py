"""Two serving contracts of a model plugin, kept here once so that no model plugin writes them again.

Exact rows (RowGroups). A batched engine's reply must not depend on the requests beside it, and a prompt row's bits
must follow from the prompt alone: every prompt chunk is multiplied alone with `prompt=True` (the prompt rows'
arithmetic at every length), and one-token rows (decode rows, a round's columns, draft steps) in calls of at most
`invariant_rows(projections)` rows, so each row has the bits it has alone (docs/PLUGINS.md, row invariance).

Prompt spans (Spans). Core cancels a request between steps, and crit serve retires the engine when a cancel does not
settle within its grace (one second): every prompt step must finish well inside it, at every context the engine
serves, and leave the run's device memory headroom free at its peak. A span is a prompt chunk's rows, a function of
the position it starts at only, so a prompt's chunk boundaries are the same whatever shares its steps and in every
process that replays the recorded calibration. The calibration times three prompt steps at load (a short and a full
chunk at position 0, a full chunk at the deepest position the engine serves, its KV pages written with zeros),
measures their peaks, fits

    seconds(rows, position) = c + a * rows + b * rows * (position + rows)
    bytes(rows)             = m * rows

and gives each position the longest span (a multiple of QUANTUM, at most the prefill chunk) whose predicted seconds
stay within the budget and whose peak leaves the headroom free. It is a measured decision
(crit_plugins.decisions.settle): `crit plugins decisions` records it in the package, so the package Eval measured
spans its prompts as the package served does.

A shorter span lowers a step's time only by the part that grows with its rows, and pays the fixed part c again for
fewer tokens. With an experts plugin that copies a chunk's routed experts to the device every step (expert-offload
over host memory: gigabytes a step, whatever the chunk's size past a few dozen tokens) c is most of a step, so the
budget decides the prefill rate: with c 0.45 s and 0.48 ms a token, a budget of 0.5 s gives 96-token steps and 190
tokens a second, one of 0.75 s gives 624-token steps and 830. The budget is therefore the largest that still leaves
the cancel grace a margin (SECONDS: three quarters of it), not a small one. Where c alone passes the budget no span
meets it, and every shorter span is slower still (16-token steps prefill a 200,000-token prompt in most of an hour):
the span there is the one that prefills fastest by the fit, among those predicted to end inside the cancel grace
where any does. The spans' facts give each step's predicted seconds and rate and say when a step passes the budget
(over_budget): the lever is then the step's fixed part, not a smaller chunk.

A model that runs its prompt spans layer by layer (passes.py) spreads a pass over as many steps as its work takes,
and a step of it ends between a chunk through one layer and the next: what has to end within the budget is that
unit, a chunk through a layer, not the chunk through the model. Its spans are sized by that (`units`: the model's
layers, a span's time is the fit's over them). The fit times whole chunks, experts' copies included, and those copies
do not follow c + a * rows (an experts plugin copies a short chunk's routed experts alone and a long chunk's whole
layers), so the fit's a may hold copies that a pass pays once: a unit's time read off it is an upper bound, and a
chunk stays as long as the device computes well."""

QUANTUM = 16   # spans are multiples of this many rows: few distinct shapes, the attention reader's tiles
GRACE = 1.0    # crit serve's cancel grace in seconds (fixed there): a prompt step ends inside it
SECONDS = 0.75 * GRACE   # the default budget of a prompt step: a quarter of the grace for what the fit does not see


class RowGroups:
    """The token ranges a step's matrices multiply apart, and the matmul over them."""

    def __init__(self, projections):
        from crit_plugins.device import invariant_rows
        self.proj = projections
        self.decode_rows = invariant_rows(projections)

    def of(self, batch):
        """[(first, end, prompt)] of a StepBatch, RoundBatch or PromptBatch: each prompt chunk alone, one-token rows
        decode_rows at a time."""
        step = self.decode_rows
        if getattr(batch, "prompt", False):   # a captured prompt chunk: one chunk, its bucket's rows
            return [(0, batch.tokens_t.shape[0], True)]
        if batch.round:
            return [(first, min(first + step, batch.tokens), False) for first in range(0, batch.tokens, step)]
        groups, run = [], None
        for r in range(batch.rows):
            first, end = batch.starts[r], batch.starts[r + 1]
            if end - first > 1:
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


def fit(timings, peaks):
    """The cost model of three timed prompt steps {(rows, position): seconds} (rows r1 < r2 at position 0, r2 at
    position p > 0) and their peaks {rows: bytes}: {c, a, b, m}, each rounded to four significant figures so a recorded
    decision reads back as it was written."""
    (r1, _), (r2, _), (_, p) = sorted(timings, key=lambda k: (k[1], k[0]))
    t1, t2, t3 = timings[(r1, 0)], timings[(r2, 0)], timings[(r2, p)]
    b = max(0.0, (t3 - t2) / (r2 * p))
    a = max(0.0, (t2 - t1) / (r2 - r1) - b * (r1 + r2))
    c = max(0.0, t1 - a * r1 - b * r1 * r1)
    m = max(v / r for r, v in peaks.items()) if peaks else 0.0
    return {k: float(f"{v:.4g}") for k, v in {"c": c, "a": a, "b": b, "m": m}.items()}


class Spans:
    """Prompt spans: rows(position), the rows of a prompt chunk that starts at `position`."""

    def __init__(self, chunk, costs=None, seconds=0.0, room=None, deepest=0, units=1):
        """deepest: the deepest position a prompt chunk of the engine starts at (for the spans' facts). units: the
        equal parts of a chunk's work between which a step of the model can end (1: a step runs a chunk through the
        model; a model's passes end a step between a chunk through one layer and the next: its layers), so a span's
        time is the fit's over them."""
        self.chunk, self.costs, self.seconds, self.room = int(chunk), costs, float(seconds), room
        self.deepest, self.units = int(deepest), max(1, int(units))
        self._rows = {}

    def predicted(self, rows, position):
        c = self.costs
        return (c["c"] + c["a"] * rows + c["b"] * rows * (position + rows)) / self.units

    def rows(self, position):
        """The span at `position`: the prefill chunk without a calibration; else the longest span (the chunk less a
        multiple of QUANTUM, QUANTUM at least) within the memory room whose predicted time is within the budget.
        Where none is (the module's description), the span that prefills fastest by the fit, among those predicted to
        end inside the cancel grace where any does."""
        if self.costs is None:
            return self.chunk
        key = position // QUANTUM
        found = self._rows.get(key)
        if found is None:
            found = self._rows[key] = self._span(position)
        return found

    def _span(self, position):
        spans = list(range(self.chunk, QUANTUM, -QUANTUM)) + [min(self.chunk, QUANTUM)]   # longest first
        roomy = [r for r in spans if self.room is None or self.costs["m"] * r <= self.room] or spans[-1:]
        within = [r for r in roomy if self.predicted(r, position) <= self.seconds]
        if within:
            return within[0]
        if len(roomy) == 1:
            return roomy[0]
        graceful = [r for r in roomy if self.predicted(r, position) < GRACE] or roomy
        return max(graceful, key=lambda r: (r / max(self.predicted(r, position), 1e-12), r))

    def stop(self, position):
        """The position a prompt chunk starting at `position` does not pass (crit_core.batch.step_item's stop)."""
        return position + self.rows(position)

    def facts(self):
        """The spans for the engine's facts: the fit, and the span and its predicted seconds at position 0 and at the
        deepest position; over_budget where a step is predicted past the budget there (its fixed part, costs c, is
        what to lower: a shorter span does not)."""
        if self.costs is None:
            return {"rows": self.chunk, "calibrated": False}
        found = {"rows": self.chunk, "calibrated": True, "seconds": self.seconds, "costs": self.costs,
                 **({"room_bytes": int(self.room)} if self.room is not None else {}),
                 **({} if self.units == 1 else {"units": self.units})}
        steps = {position: (self.rows(position), self.predicted(self.rows(position), position))
                 for position in sorted({0, self.deepest})}
        found["steps"] = [{"position": p, "rows": r, "seconds": float(f"{s:.4g}"),
                           "tokens_per_second": float(f"{r / max(s * self.units, 1e-12):.4g}")}
                          for p, (r, s) in steps.items()]
        if any(s > self.seconds for _, s in steps.values()):
            found["over_budget"] = (f"a prompt step is predicted to take {max(s for _, s in steps.values()):.2f} s, "
                                    f"{self.costs['c']:.2f} s of it whatever its span: no span meets span_seconds "
                                    f"{self.seconds:g}, so spans are the ones that prefill fastest")
        return found
