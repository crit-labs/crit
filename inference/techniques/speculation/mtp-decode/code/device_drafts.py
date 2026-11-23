"""Drafting inside a captured decode round, in torch on the device: the drafter slot's "draft:device" extension
(crit_plugins.device.DEVICE_EXTENSIONS) for a chain over the model's device draft head, and prompt_copy.py's copy
proposals computed from each slot's token history on the device, so copies need no host between rounds (native rounds).

The model's draft head (fact draft:head-device; docs/PLUGINS.md, The model plugin):

    head.catch_up(view)                  ingest the previous round's emitted tokens (view.catch_up_tokens / _count)
    head.logits(depth, tokens, view)     [rows][vocab] FP32: the next draft's logits after `tokens` [rows] at `depth`
    head.commit_round(view)              (optional; the round driver calls it) keep the accepted prefix of its state

head_chain drafts view.width tokens per row: depth 0 after the pending token, each later one after the previous draft;
the argmax in greedy rounds (the lower id on a tie), and in sampled rounds the sampler's draft_device (a draw from the
row's processed law with Core's `draft` uniform at output position out_count + depth, its law kept in
view.draft_laws; greedy rows still take the argmax). So in sampled rounds the device form samples its drafts where the
host form (propose) drafts greedily: both keep the target's law (the verify's rejection rule), and their tokens differ.

HistoryCopies keeps each round slot's tokens (prompt and outputs, the pending token last) in a ring of `capacity`
positions on the device and proposes, for every row at once, prompt_copy.PromptCopyIndex.propose_matched's copy: the
longest continuation of an earlier occurrence of the context's last `window` tokens among their 64 most recent
occurrences (an occurrence in the outputs counts only where the last `history_window` tokens match too), the most
recent of equally long ones, at most `width` tokens, and how many tokens before it match the context (at most
`longest`). It searches the last capacity - (widest + 1) positions, so a round's commit (which writes up to widest + 1
positions past the frontier, and again fewer after a truncation) never touches a position a later round reads: it
equals the host index wherever the sequence fits that span. A copied draft is drawn from no law: in sampled rounds its
column is verified by the target's own draw (the sampler's laws cleared there). override() puts a long copy over a
neural chain (device-speculative-round's override_drafts semantics: the first count drafts of each row whose match
covers at least `threshold` tokens become the copy).
"""
import torch

PAD = -1


def head_chain(view, head, sampler=None):
    """view.drafts[:, :width] from the model's draft head: catch-up first, then one head call per depth."""
    rows, width = view.rows, view.width
    head.catch_up(view)
    if not width:
        return
    draw = None
    if view.mode == "sampled":
        draw = getattr(sampler, "draft_device", None)
        if not callable(draw):
            raise ValueError("a sampled round draws its drafts from their laws: the sampler gives draft_device "
                             "(rounds:sampler)")
    tokens = view.pending[:rows]
    for depth in range(width):
        logits = head.logits(depth, tokens, view)
        tokens = draw(view, logits, depth) if draw is not None else torch.argmax(logits, dim=-1)
        view.drafts[:rows, depth] = tokens


class HistoryCopies:
    """Token histories of up to `slots` round slots in rings of `capacity` tokens on `device`, and their copies for
    rounds of up to `widest` drafts."""

    def __init__(self, slots, capacity, window, history_window, widest, device, longest=64, max_matches=64):
        if slots < 1 or window < 1 or history_window < window or longest < 1 or widest < 0:
            raise ValueError("at least one slot, a window of one or more and a history window >= the window")
        if capacity - (widest + 1) <= max(history_window, longest):
            raise ValueError(f"a ring of {capacity} tokens leaves {capacity - widest - 1} to search past a round's "
                             f"{widest + 1} columns: more than the history window and the {longest} matched")
        self.slots, self.capacity, self.window, self.history_window = slots, capacity, window, history_window
        self.longest, self.max_matches = longest, max_matches
        self.span = capacity - (widest + 1)   # the positions a round reads: never ones a commit may rewrite
        self.device = torch.device(device)
        long = dict(dtype=torch.int64, device=self.device)
        self.ring = torch.full((slots, capacity), PAD, **long)
        self.prompt_end = torch.zeros(slots, **long)
        self.offsets = torch.arange(capacity, **long)
        self.back = torch.arange(1, max(longest, history_window) + 1, **long)

    # Slots ------------------------------------------------------------------------------------------------------
    def bind(self, slot, tokens):
        """Round slot `slot` holds a sequence whose tokens so far are `tokens` (the pending token last): its
        prompt for the copies' prompt rule (outside a capture)."""
        tokens = list(tokens)
        n = len(tokens)
        found = torch.full((self.capacity,), PAD, dtype=torch.int64)
        if n:
            keep = tokens[-self.capacity:]
            positions = torch.arange(n - len(keep), n) % self.capacity
            found[positions] = torch.tensor(keep, dtype=torch.int64)
        self.ring[slot].copy_(found)
        self.prompt_end[slot] = n

    def move(self, source, target):
        self.ring[target].copy_(self.ring[source])
        self.prompt_end[target] = self.prompt_end[source]

    def commit(self, view):
        """The round's emitted tokens into each row's history: positions frontier - n_emitted + 1 .. frontier (the
        frontier after the round; the new pending token at it). Idempotent from the round's start (a truncation)."""
        rows, cols = view.rows, view.width + 1
        count = view.n_emitted[:rows]
        column = self.offsets[None, :cols]
        positions = (view.frontier[:rows] - count + 1)[:, None] + column
        keep = column < count[:, None]
        index = positions.clamp(min=0) % self.capacity
        current = self.ring[:rows].gather(1, index)
        self.ring[:rows].scatter_(1, index, torch.where(keep, view.emitted[:rows, :cols], current))

    # Proposals ---------------------------------------------------------------------------------------------------
    def propose(self, view, width):
        """(copies [rows][width] int64, PAD past each row's count; counts [rows]; matched [rows]) for the round's rows,
        from their histories up to the pending token at view.frontier; capture-safe."""
        rows, cap, w, span = view.rows, self.capacity, self.window, self.span
        long = dict(dtype=torch.int64, device=self.device)
        length = view.frontier[:rows] + 1                            # tokens so far, the pending one included
        first = (length - span).clamp(min=0)                         # the oldest position a round reads
        positions = first[:, None] + self.offsets[None, :span]       # [rows][span]: absolute positions, ascending

        # The held context once, PAD past its end; shifted views of it read neighbouring positions.
        held = self.ring[:rows].gather(1, positions % cap)
        held = torch.where(positions < length[:, None], held, torch.full_like(held, PAD))

        def at(offset):
            """tokens at positions + offset ([rows][cap]), PAD outside the held context."""
            if offset >= 0:
                return torch.nn.functional.pad(held[:, offset:], (0, offset), value=PAD)
            return torch.nn.functional.pad(held[:, :offset], (-offset, 0), value=PAD)

        reach = max(w, self.history_window)
        back_index = (length[:, None] - self.back[None, :reach]).clamp(min=0) % cap
        tail = self.ring[:rows].gather(1, back_index)                 # tail[:, k - 1]: the token k back from the end

        def last(k):
            """the context's token k back from the end (k = 1: the pending token), [rows]."""
            return tail[:, k - 1]

        # Earlier occurrences of the last `window` tokens: windows ending before the pending token (the host index
        # holds a window once a token follows it).
        end = positions + w
        occurrence = (end < length[:, None]) & (length[:, None] > w)
        for k in range(w):
            occurrence &= at(k) == last(w - k)[:, None]
        # Only the max_matches most recent occurrences are known (the host index's bounded deques).
        recent = occurrence.long().flip(1).cumsum(1).flip(1) <= self.max_matches
        valid = occurrence & recent
        # An occurrence in the outputs needs the last history_window tokens to match before its end.
        hw = self.history_window
        history = positions >= self.prompt_end[:rows, None]
        agree = (end >= hw) & (length[:, None] >= hw)
        for k in range(1, hw + 1):
            agree &= at(w - k) == last(k)[:, None]
        valid &= ~history | agree
        limit = min(width, 64)
        reach_left = torch.clamp(length[:, None] - end, max=limit)
        key = torch.where(valid, reach_left * (cap + 1) + self.offsets[None, :span] + 1, torch.zeros_like(reach_left))
        best = key.argmax(1)
        found = key.gather(1, best[:, None])[:, 0] > 0
        count = torch.where(found, reach_left.gather(1, best[:, None])[:, 0], torch.zeros_like(best))
        source = positions.gather(1, best[:, None])[:, 0] + w                  # the copy's first position
        column = torch.arange(max(width, 1), **long)[None, :width]
        copied = self.ring[:rows].gather(1, ((source[:, None] + column).clamp(min=0)) % cap)
        copies = torch.where(column < count[:, None], copied, torch.full_like(copied, PAD))
        # How many tokens before the copy's source match the context's last ones (window at least, longest at most).
        back = self.back[None, :self.longest]
        before = self.ring[:rows].gather(1, ((source[:, None] - back).clamp(min=0)) % cap)
        ours = self.ring[:rows].gather(1, ((length[:, None] - back).clamp(min=0)) % cap)
        same = (before == ours) & (source[:, None] - back >= first[:, None]) & (back <= source[:, None])
        same |= back <= w
        matched = torch.where(found, same.long().cumprod(1).sum(1), torch.zeros_like(count))
        return copies, count, matched


def override(view, copies, counts, matched, threshold):
    """Where a row's copy match covers at least `threshold` tokens, its first counts[row] drafts become the copied
    tokens, and in sampled rounds their laws are cleared (a copy is drawn from no law: its column is verified by the
    target's own draw); capture-safe."""
    rows, width = view.rows, view.width
    if not width:
        return
    column = torch.arange(width, device=copies.device)[None, :]
    use = (matched >= threshold)[:, None] & (column < counts[:, None])
    drafts = view.drafts[:rows, :width]
    drafts.copy_(torch.where(use, copies[:, :width], drafts))
    if view.mode == "sampled":
        view.draft_laws.clear(rows, width, use)
