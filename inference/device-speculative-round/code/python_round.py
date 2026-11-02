"""A Python driver for device speculative rounds over many rows at once, with torch.

A Python engine decoding several requests together (a Core step engine's step(), crit_core.native.StepEngine, or a
single-request engine with one row) runs one round for all of them:

1. the drafter catches up on each row's previous round (catch_up_tokens, the tokens it emitted, at the fixed width
   widest + 1 with junk past catch_up_count; catch_up.py's plan) and proposes `width` drafts per row;
2. the target verifies each row's pending token and its drafts in width + 1 columns at positions frontier ..
   frontier + width, and writes the token each column predicts: its argmax for greedy rows, or for sampled rows the
   predictions of rejection sampling (rejection_predictions), under which the same accept rule keeps the target's
   distribution;
3. acceptance runs on the device: the drafts that match the predictions, then the correction or bonus token, bounded
   by the row's output allowance and ending after a stop token, for every row at once;
4. the engine's deferred layers keep each row's state after its last emitted column (commit; again after a
   truncation, from the round's start);
5. the round's results are packed into one block, copied to pinned host memory, which the host reads after one event
   per round.

Every tensor a round touches is allocated once, so a round of (rows, width, mode) is captured as one CUDA graph at load
(capture), and no request waits for a capture. Rows join and leave between rounds (start, release, move); a round runs
on the first `rows` slots, so the engine keeps its decoding rows packed at the front.

Modes: "greedy" and "sampled" (the verify hook writes the predictions), "replay" (a fixed continuation is the
prediction: crit Eval's replay, which accepts a draft while it equals the continuation) and "forced" (the continuation
is also the draft: scoring a fixed text width + 1 positions per target pass).

The engine supplies three hooks, the parts that differ between engines (drafter, target, state):

    draft(view)    write view.drafts[:, :view.width], after ingesting view.catch_up_tokens / view.catch_up_count
    verify(view)   run the target over view.columns() at positions view.frontier + column and write
                   view.predictions[:, :view.width + 1] (and view.logprobs when the driver keeps them)
    commit(view)   keep each row's state after column view.n_emitted - 1 (nothing where it is 0); idempotent from the
                   round's start, since a truncation calls it again

and optionally reset_capture(), called before each warm-up of a capture and after it, to put its own state back.
Hooks must be capture-safe: fixed shapes for a (rows, width), no host synchronization.
"""
import torch

MODES = ("greedy", "sampled", "replay", "forced")


class RoundView:
    """One round's tensors, the driver's buffers cut to its rows and width; the hooks read and write them."""

    def __init__(self, rounds, rows, width, mode):
        self.rows, self.width, self.mode = rows, width, mode
        self.pending = rounds.pending[:rows]                # each row's input at column 0, its last emitted token
        self.frontier = rounds.frontier[:rows]              # the position of column 0
        self.allowance = rounds.allowance[:rows]            # output tokens the row may still emit (0: inactive)
        self.drafts = rounds.drafts[:rows]                  # [rows, widest]: the draft hook writes [:, :width]
        self.predictions = rounds.predictions[:rows]        # [rows, widest + 1]: the verify hook writes [:, :width + 1]
        self.logprobs = None if rounds.logprobs is None else rounds.logprobs[:rows]
        self.catch_up_tokens = rounds.catch_up[:rows]       # [rows, widest + 1]: the previous round's tokens, 0 past
        self.catch_up_count = rounds.catch_up_count[:rows]  # how many of them it emitted
        self.n_emitted = rounds.n_emitted[:rows]            # this round's count, set before commit
        self.emitted = rounds.emitted[:rows]                # this round's tokens, -1 past n_emitted

    def columns(self):
        """The verify columns' input tokens: the pending token, then the drafts ([rows, width + 1])."""
        return torch.cat([self.pending[:, None], self.drafts[:, :self.width]], dim=1)


class SpeculativeRounds:
    """Device state and rounds for up to `rows` rows with drafts up to `widest` wide on `device`; `stops` stop ids per
    row at most, `continuation` tokens per row for replay and forced rounds, and `logprobs` to pack each emitted
    column's log-probability (the verify hook writes view.logprobs)."""

    def __init__(self, hooks, rows, widest, device, stops=4, continuation=0, logprobs=False):
        if rows < 1 or widest < 0 or stops < 0 or continuation < 0:
            raise ValueError("at least one row, and non-negative widths, stop and continuation counts")
        self.hooks, self.rows, self.widest, self.device = hooks, rows, widest, torch.device(device)
        cols = widest + 1
        long = dict(dtype=torch.long, device=self.device)
        self.pending, self.frontier, self.allowance, self.out_count = (torch.zeros(rows, **long) for _ in range(4))
        self.n_emitted, self.n_matched, self.catch_up_count = (torch.zeros(rows, **long) for _ in range(3))
        # The round's start, which a truncation goes back to.
        self.start = {name: torch.zeros(rows, **long) for name in ("pending", "frontier", "allowance", "out_count")}
        self.drafts = torch.zeros(rows, max(widest, 1), **long)
        self.predictions = torch.zeros(rows, cols, **long)
        self.emitted = torch.full((rows, cols), -1, **long)
        self.catch_up = torch.zeros(rows, cols, **long)
        self.stops = torch.full((rows, max(stops, 1)), -1, **long)
        self.continuation = torch.zeros(rows, max(continuation, 1), **long)
        self.logprobs = torch.zeros(rows, cols, dtype=torch.float32, device=self.device) if logprobs else None
        self.columns = torch.arange(cols, **long)
        # The readback block: n_emitted, n_matched, frontier after the round, then the emitted tokens.
        self.block = torch.zeros(rows, 3 + cols, **long)
        pinned = self.device.type == "cuda"
        self.host = torch.zeros(rows, 3 + cols, dtype=torch.long, pin_memory=pinned)
        self.host_logprobs = (torch.zeros(rows, cols, dtype=torch.float32, pin_memory=pinned) if logprobs else None)
        self.graphs = {}
        self.event = torch.cuda.Event() if pinned else None
        self.launched = None   # (rows, width) of the round in flight
        self.last = None       # (rows, width) of the last round read, which a truncation applies to

    # Rows -----------------------------------------------------------------------------------------------------------

    def start_row(self, slot, frontier, token, allowance, stops=(), continuation=()):
        """Row `slot` starts decoding after its prefill: `token` is the first output token the prefill produced, at
        position `frontier` (the prompt's length), `allowance` the tokens it may still emit after it. The drafter's
        first catch-up ingests that token (a one-column seed). `stops` are its stop token ids; `continuation`, for
        replay and forced rounds, the fixed text its output follows, from the first output token on."""
        stops, continuation = list(stops), list(continuation)
        if len(stops) > self.stops.shape[1] or len(continuation) > self.continuation.shape[1]:
            raise ValueError("more stop ids or continuation tokens than the driver was sized for")
        values = {"pending": token, "frontier": frontier, "allowance": allowance, "out_count": 1, "n_emitted": 1,
                  "catch_up_count": 1}
        for name, value in values.items():
            getattr(self, name)[slot] = int(value)
        self.catch_up[slot].zero_()
        self.catch_up[slot, 0] = int(token)
        self.emitted[slot].fill_(-1)
        self.stops[slot].fill_(-1)
        if stops:
            self.stops[slot, :len(stops)] = torch.tensor(stops, dtype=torch.long)
        self.continuation[slot].zero_()
        if continuation:
            self.continuation[slot, :len(continuation)] = torch.tensor(continuation, dtype=torch.long)

    def release(self, slot):
        """Row `slot` stops decoding: it emits nothing in later rounds until it starts again."""
        self.allowance[slot] = 0

    def move(self, source, target):
        """Row `source`'s state moves to slot `target` (to keep the decoding rows at the front)."""
        for tensor in (self.pending, self.frontier, self.allowance, self.out_count, self.n_emitted, self.n_matched,
                       self.catch_up_count, self.catch_up, self.emitted, self.stops, self.continuation, self.predictions,
                       self.drafts, *self.start.values(), *([self.logprobs] if self.logprobs is not None else [])):
            tensor[target] = tensor[source]
        self.allowance[source] = 0

    # Rounds ---------------------------------------------------------------------------------------------------------

    def view(self, rows, width, mode):
        return RoundView(self, rows, width, mode)

    def _from_continuation(self, rows, out, count):
        """Each row's continuation from its output count on, `count` tokens, into `out` (clamped at its end)."""
        index = (self.out_count[:rows, None] + self.columns[None, :count]).clamp(max=self.continuation.shape[1] - 1)
        out.copy_(self.continuation[:rows].gather(1, index))

    def _round(self, rows, width, mode):
        view = self.view(rows, width, mode)
        for name, tensor in self.start.items():
            tensor[:rows].copy_(getattr(self, name)[:rows])
        if mode == "forced":
            self._from_continuation(rows, view.drafts[:, :width], width)
        elif width:
            self.hooks.draft(view)
        self.hooks.verify(view)
        if mode in ("replay", "forced"):
            self._from_continuation(rows, view.predictions[:, :width + 1], width + 1)
        self._accept(rows, width)
        self.hooks.commit(view)
        self._pack(rows)

    def _accept(self, rows, width):
        """Greedy acceptance for every row on the device: the drafts that match the predictions, then the correction
        or bonus token, ending after the first predicted stop token and bounded by the allowance."""
        predictions = self.predictions[:rows, :width + 1]
        if width:
            matched = (self.drafts[:rows, :width] == predictions[:, :width]).long().cumprod(1).sum(1)
        else:
            matched = torch.zeros_like(self.n_emitted[:rows])
        stop = (predictions[:, :, None] == self.stops[:rows, None, :]).any(2)
        first_stop = torch.where(stop, self.columns[None, :width + 1], width + 1).min(1).values
        count = torch.minimum(torch.minimum(matched + 1, first_stop + 1), self.allowance[:rows]).clamp(min=0)
        self.n_matched[:rows] = torch.minimum(matched, (count - 1).clamp(min=0))
        self._emit(rows, width, count)

    def _emit(self, rows, width, count):
        """Each row keeps `count` tokens of its predictions, from the round's start."""
        predictions = self.predictions[:rows, :width + 1]
        self.n_emitted[:rows] = count
        kept = self.columns[None, :width + 1] < count[:, None]
        self.emitted[:rows].fill_(-1)
        self.emitted[:rows, :width + 1] = torch.where(kept, predictions, -1)
        last = predictions.gather(1, (count - 1).clamp(min=0)[:, None]).squeeze(1)
        self.pending[:rows] = torch.where(count > 0, last, self.start["pending"][:rows])
        self.frontier[:rows] = self.start["frontier"][:rows] + count
        self.allowance[:rows] = self.start["allowance"][:rows] - count
        self.out_count[:rows] = self.start["out_count"][:rows] + count
        self.catch_up[:rows] = self.emitted[:rows].clamp(min=0)
        self.catch_up_count[:rows] = count

    def _pack(self, rows):
        self.block[:rows, 0] = self.n_emitted[:rows]
        self.block[:rows, 1] = self.n_matched[:rows]
        self.block[:rows, 2] = self.frontier[:rows]
        self.block[:rows, 3:] = self.emitted[:rows]
        self.host[:rows].copy_(self.block[:rows], non_blocking=True)
        if self.logprobs is not None:
            self.host_logprobs[:rows].copy_(self.logprobs[:rows], non_blocking=True)

    def capture(self, rows, width, mode):
        """Capture the round of (rows, width, mode) as a CUDA graph (two warm-ups first), the driver's buffers put back
        around it and the hooks' own by their reset_capture()."""
        if self.device.type != "cuda":
            raise ValueError("CUDA graphs need a CUDA device")
        self._check(rows, width, mode)
        saved = {name: tensor.clone() for name, tensor in self._state().items()}
        reset = getattr(self.hooks, "reset_capture", None)

        def restore():
            for name, tensor in self._state().items():
                tensor.copy_(saved[name])
            if callable(reset):
                reset()
        stream = torch.cuda.Stream(self.device)
        stream.wait_stream(torch.cuda.current_stream(self.device))
        with torch.cuda.stream(stream):
            for _ in range(2):
                restore()
                self._round(rows, width, mode)
            stream.synchronize()
            restore()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream):
                self._round(rows, width, mode)
        torch.cuda.current_stream(self.device).wait_stream(stream)
        torch.cuda.synchronize(self.device)
        restore()
        self.graphs[(rows, width, mode)] = graph
        return graph

    def _state(self):
        names = ("pending", "frontier", "allowance", "out_count", "n_emitted", "n_matched", "catch_up_count", "catch_up",
                 "emitted", "drafts", "predictions")
        return {**{name: getattr(self, name) for name in names}, **{"start_" + k: v for k, v in self.start.items()}}

    def _check(self, rows, width, mode):
        if not 1 <= rows <= self.rows or not 0 <= width <= self.widest or mode not in MODES:
            raise ValueError(f"a round runs 1..{self.rows} rows at width 0..{self.widest} in one of {', '.join(MODES)}")

    def launch(self, rows, width, mode="greedy"):
        """Run one round of the first `rows` rows at `width` drafts: its graph when captured, else eagerly."""
        self._check(rows, width, mode)
        if self.launched is not None:
            raise ValueError("read the round in flight before launching another")
        graph = self.graphs.get((rows, width, mode))
        if graph is not None:
            graph.replay()
        else:
            self._round(rows, width, mode)
        if self.event is not None:
            self.event.record()
        self.launched = (rows, width)

    def read(self):
        """The round in flight's results, after one wait: per row {"tokens", "matched", "frontier"} (and "logprobs")."""
        if self.launched is None:
            raise ValueError("no round in flight")
        rows, width = self.last = self.launched
        self.launched = None
        if self.event is not None:
            self.event.synchronize()
        block = self.host[:rows].tolist()
        logprobs = self.host_logprobs[:rows].tolist() if self.host_logprobs is not None else None
        out = []
        for row, values in enumerate(block):
            count = values[0]
            found = {"tokens": values[3:3 + count], "matched": values[1], "frontier": values[2]}
            if logprobs is not None:
                found["logprobs"] = logprobs[row][:count]
            out.append(found)
        return out

    def truncate(self, keep):
        """Core kept fewer tokens of the last round than it emitted (a stop it did not know, an output limit): each
        row keeps at most keep[row] (a sequence over the round's rows), and the engine's state is selected again."""
        if self.last is None:
            raise ValueError("no round read to truncate")
        rows, width = self.last
        keep = torch.as_tensor(list(keep), dtype=torch.long, device=self.device)
        count = torch.minimum(self.n_emitted[:rows], keep.clamp(min=0))
        self.n_matched[:rows] = torch.minimum(self.n_matched[:rows], (count - 1).clamp(min=0))
        self._emit(rows, width, count)
        self.hooks.commit(self.view(rows, width, "greedy"))


def draw(probabilities, uniforms):
    """A token from each distribution of `probabilities` ([..., vocabulary], non-negative, any scale) by inverse CDF
    with `uniforms` ([...] in [0, 1)): capture-safe, no host synchronization."""
    cdf = probabilities.cumsum(-1)
    target = (uniforms * cdf[..., -1]).unsqueeze(-1)
    return torch.searchsorted(cdf, target, right=True).squeeze(-1).clamp(max=probabilities.shape[-1] - 1)


def rejection_predictions(target, drafts, draft_probs, uniforms, residual_uniforms):
    """Predictions under which the greedy accept rule is rejection sampling, so sampled rounds keep the target's
    distribution. target: [rows, width + 1, vocabulary] the processed target distribution at each column (temperature
    and filters applied); drafts: [rows, width]; draft_probs: [rows, width, vocabulary] the drafter's distribution each
    draft came from (one-hot for a greedy drafter); uniforms: [rows, width] for the accept tests; residual_uniforms:
    [rows, width + 1] for the draws. Column j < width predicts its draft when u_j < p_j(d_j) / q_j(d_j), else a draw
    from max(p_j - q_j, 0) (which cannot be the draft, so acceptance stops there); the last column draws from p."""
    width = drafts.shape[1]
    predictions = draw(target, residual_uniforms)
    if width:
        p = target[:, :width].gather(2, drafts[..., None]).squeeze(2)
        q = draft_probs.gather(2, drafts[..., None]).squeeze(2)
        accept = uniforms * q < p
        residual = (target[:, :width] - draft_probs).clamp(min=0)
        # Where p <= q everywhere (the drafter's distribution is the target's) the residual is empty; then the accept
        # test always passes, and the draw is never used.
        residual = torch.where(residual.sum(-1, keepdim=True) > 0, residual, target[:, :width])
        rejected = draw(residual, residual_uniforms[:, :width])
        predictions[:, :width] = torch.where(accept, drafts, rejected)
    return predictions
