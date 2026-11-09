"""Copy drafts in the drafter slot: mtp-decode's bounded exact-window index over accepted tokens (prompt_copy.py) as the
host form; inside captured decode rounds the same copies from each round slot's token history on the device
(device_drafts.HistoryCopies, the draft:device extension), so they need no host between rounds. With `override`, the
model's draft head drafts the chain and a copy matching at least `override` tokens replaces its first drafts."""
from .mtp import prompt_copy

# The host index bounds its context; the device ring searches its last `capacity` - (widest + 1) positions.
MAX_TOKENS = 1 << 30


def argmax(row):
    return max(range(len(row)), key=lambda t: (row[t], -t))


class PromptCopyDrafter:
    def __init__(self, options):
        self.options, self.indices = options, {}
        self.copies, self._threshold = None, options["override"]

    def start(self, sequence, tokens):
        window = self.options["window"]
        self.indices[sequence] = prompt_copy.PromptCopyIndex(
            list(tokens), window=window, history_window=max(window, self.options["history_window"]), max_draft=64,
            max_tokens=MAX_TOKENS)

    def propose(self, sequence, width, head=None):
        index = self.indices[sequence]
        if self._threshold and head is None:
            raise ValueError("mtp-decode with override drafts the chain from the engine's draft head: pass head")
        if self._threshold:
            path = []
            for _ in range(width):
                path.append(argmax(head([list(path)])[0]))
            copied, matched = index.propose_matched(width) if width else ((), 0)
            if matched >= self._threshold:
                path[:len(copied)] = copied
            drafts = path
        else:
            drafts = list(index.propose(width)) if width else []
        return {"tokens": [index.tokens[-1]] + list(drafts), "parents": list(range(-1, len(drafts))), "laws": None}

    def observe(self, sequence, emitted):
        self.indices[sequence].append(list(emitted))

    def release(self, sequence):
        self.indices.pop(sequence, None)

    # draft:device (crit_plugins.device.DEVICE_EXTENSIONS)
    def reserve(self, rows, widest, vocab, device):
        """Each of `rows` round slots' token history ring for rounds of up to `widest` drafts, once, before any bind
        or capture (the round driver calls it)."""
        from .mtp import device_drafts
        w = self.options["window"]
        self.copies = device_drafts.HistoryCopies(rows, self.options["capacity"], w,
                                                  max(w, self.options["history_window"]), widest, device)
        return self.copies

    def _ring(self):
        if self.copies is None:
            raise ValueError("mtp-decode: reserve(rows, widest, vocab, device) its round buffers first")
        return self.copies

    def bind(self, slot, sequence):
        """Round slot `slot` holds `sequence` (the round driver's bind): its history is empty until seed(). Outside a
        capture."""
        self._ring().bind(slot, [])

    def seed(self, slot, tokens):
        """Round slot `slot`'s tokens so far, positions 0 .. frontier: the prompt and the outputs, the pending token
        last (the round driver seeds the slot as it binds it). Outside a capture."""
        self._ring().bind(slot, list(tokens))

    def unbind(self, slot):
        self._ring().bind(slot, [])

    def move(self, source, target):
        self._ring().move(source, target)

    def draft(self, view, head=None, sampler=None):
        """view.drafts[:, :width]: each row's copy (PAD past it), or with `override` the head's chain with long copies
        over its first drafts; in sampled rounds a copied draft has no law (its column is verified by the target's own
        draw); capture-safe."""
        import torch
        from .mtp import device_drafts
        rows, width = view.rows, view.width
        ring = self._ring()
        if self._threshold:
            if head is None:
                raise ValueError("mtp-decode with override drafts the chain from the model's draft head")
            device_drafts.head_chain(view, head, sampler)
        if not width:
            return
        copies, counts, matched = ring.propose(view, width)
        if self._threshold:
            device_drafts.override(view, copies, counts, matched, self._threshold)
        else:
            view.drafts[:rows, :width] = copies
            if view.mode == "sampled":   # every column a copy or PAD: none was drawn from a law
                view.draft_laws.clear(rows, width, torch.ones_like(copies, dtype=torch.bool))

    def commit_round(self, view):
        """Each row's history takes the round's emitted tokens (a row that emitted none keeps its own); idempotent
        from the round's start. The round driver commits the draft head itself."""
        self._ring().commit(view)


def create(context, options, parts):
    return PromptCopyDrafter(options)
