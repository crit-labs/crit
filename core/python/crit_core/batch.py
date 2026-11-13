"""Batched steps for Python engines: the twin of Core's crit/batch.h (core/src/batch/README.md).

A step model (crit_core.native.StepEngine) gets one row per scheduled request; it lays the rows out as a C++ engine
does: decode rows first, then prompt chunks, each phase in the order given, and runs its work once over all of them.
A request's tokens then do not depend on the requests sharing its step when the model's arithmetic follows the batch
invariance rules (core/src/batch/README.md).

    items = [step_item(row, len(session.prompt), session.frontier) for row, session in rows]
    layout = StepLayout(items)          # layout.placements in row order, layout.logit_rows for the logits pass,
                                        # layout.read_rows for the last layer's rows past its state writes
"""
from collections import namedtuple

DECODE, PREFILL = "decode", "prefill"
READY, DEFERRED, INFEASIBLE = "ready", "deferred", "infeasible"
MAX_STEP_ROWS = 2 ** 31 - 1

# One item's rows in a step: a decode item stages its pending token (with a verify's drafts), a prefill item its prompt
# chunk; the last `logits` rows produce logits. `readers`: something besides the logits pass reads the item's last-layer
# output on every row (a drafter's or MTP head's taps, a hidden-state readout).
StepItem = namedtuple("StepItem", "key phase rows logits readers", defaults=(False,))
# Where StepLayout put an item: step rows [first, first + rows), gathered logits rows [first_logit, first_logit + logits);
# `item` is its index in the input; `read` counts its last rows whose last-layer output anything reads (its logits
# rows, or every row with readers): the last layer's work past its state writes is needed on those rows alone.
Placement = namedtuple("Placement", "key phase item first rows first_logit logits read", defaults=(0,))


def step_item(row, prompt_tokens, frontier, decode_rows=1, key=None):
    """The StepItem of a scheduled row (prefill_tokens, token_work, prefill_complete, as crit_core.native.StepRow):
    a decode row stages decode_rows rows (1, or a verify's drafts + 1), at most its token work and at least one; a
    prefill row computes the prompt left after `frontier`, at most its grant and token work, with logits when it ends
    the prompt."""
    if row.prefill_complete:
        if decode_rows < 1:
            raise ValueError("a decode item stages at least one row")
        rows = max(1, min(decode_rows, row.token_work) if row.token_work else decode_rows)
        return StepItem(key, DECODE, rows, rows)
    if frontier >= prompt_tokens:
        raise ValueError("a prefill item past the end of its prompt")
    rows = min(row.prefill_tokens, prompt_tokens - frontier)
    if row.token_work:
        rows = min(rows, row.token_work)
    if rows < 1:
        raise ValueError("a prefill item without prompt rows")
    return StepItem(key, PREFILL, rows, 1 if frontier + rows == prompt_tokens else 0)


class StepLayout:
    """One step's rows: decode items first, then prefill items, each phase in input order."""

    def __init__(self, items=()):
        self.arrange(items)

    def arrange(self, items):
        items = list(items)
        total = 0
        for item in items:
            if item.rows < 1:
                raise ValueError("a step item needs rows")
            if item.logits > item.rows:
                raise ValueError("a step item has more logits rows than rows")
            if item.phase == DECODE and item.logits < 1:
                raise ValueError("a decode item needs logits")
            if item.phase not in (DECODE, PREFILL):
                raise ValueError("unknown row phase")
            total += item.rows
            if total > MAX_STEP_ROWS:
                raise ValueError("a step holds at most 2^31 - 1 rows")
        placements, logit_rows, read_rows, by_item = [], [], [], [0] * len(items)
        rows = logits = decode_rows = decode_items = 0
        for phase in (DECODE, PREFILL):
            for index, item in enumerate(items):
                if item.phase != phase:
                    continue
                by_item[index] = len(placements)
                read = item.rows if item.readers else item.logits
                placements.append(Placement(item.key, phase, index, rows, item.rows, logits, item.logits, read))
                logit_rows.extend(range(rows + item.rows - item.logits, rows + item.rows))
                read_rows.extend(range(rows + item.rows - read, rows + item.rows))
                rows += item.rows
                logits += item.logits
                if phase == DECODE:
                    decode_rows += item.rows
                    decode_items += 1
        self.placements, self.logit_rows, self.read_rows, self._by_item = placements, logit_rows, read_rows, by_item
        self.rows, self.decode_rows, self.decode_items = rows, decode_rows, decode_items
        return self

    def placement(self, item):
        """The placement of input item `item`."""
        return self.placements[self._by_item[item]]


def row_pages(rows, page_tokens):
    """Pages of page_tokens rows that hold `rows` rows."""
    if page_tokens < 1:
        raise ValueError("page_tokens must be positive")
    return -(-rows // page_tokens)


class PageReservations:
    """Page-reserving admission over a pool of `pages` pages: a request reserves every row it can stage (prompt,
    output and a verify's extra columns, at most max_tokens) without credit for shared prefix pages."""

    def __init__(self, pages, page_tokens, max_tokens, verify_rows=1):
        if pages < 1 or page_tokens < 1 or max_tokens < 1:
            raise ValueError("page reservations need pages, page_tokens and max_tokens")
        if not 1 <= verify_rows <= max_tokens:
            raise ValueError("verify_rows must be in [1, max_tokens]")
        if pages * page_tokens < max_tokens:
            raise ValueError("max_tokens exceeds pages x page_tokens")
        self.pages, self.page_tokens, self.max_tokens, self.verify_rows = pages, page_tokens, max_tokens, verify_rows
        self.reserved = 0

    def need(self, prompt_tokens, output_tokens):
        return row_pages(min(prompt_tokens + output_tokens + self.verify_rows - 1, self.max_tokens), self.page_tokens)

    def admission(self, prompt_tokens, output_tokens):
        """READY, or INFEASIBLE for an empty prompt or prompt plus output beyond max_tokens, or DEFERRED while the
        reservations leave too few pages (the answers a step model's admission() gives)."""
        if prompt_tokens < 1 or prompt_tokens + output_tokens > self.max_tokens:
            return INFEASIBLE
        return READY if self.need(prompt_tokens, output_tokens) <= self.pages - self.reserved else DEFERRED

    def reserve(self, prompt_tokens, output_tokens):
        """Reserves the need and returns it, the count release() takes back."""
        if self.admission(prompt_tokens, output_tokens) != READY:
            raise RuntimeError("page reservation for a request admission does not admit")
        pages = self.need(prompt_tokens, output_tokens)
        self.reserved += pages
        return pages

    def release(self, pages):
        if pages > self.reserved:
            raise RuntimeError("releasing more pages than are reserved")
        self.reserved -= pages


__all__ = ["DECODE", "PREFILL", "READY", "DEFERRED", "INFEASIBLE", "StepItem", "Placement", "StepLayout", "step_item",
           "row_pages", "PageReservations"]
