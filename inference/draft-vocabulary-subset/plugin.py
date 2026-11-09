"""Subset drafts in the drafter slot: draft-vocabulary-subset's row subset of the draft head (subset.py). The head
computes only the subset's rows (head(paths, rows=...)), which is the saving on the device. Inside captured decode
rounds (the draft:device extension) the subset is a device tensor the model's device head reads as head.rows: each
depth's draft is the best of its rows (sampled rounds: the sampler's draft draw over them, mapped to target ids), and
the emitted tokens are counted on the device, which chooses the subset again every `refresh` rows' rounds from those
counts, with no host between rounds."""
from .subset import subset as subsets


class SubsetHeadDrafter:
    def __init__(self, options):
        self.options, self.pending, self.head, self.counts, self.rounds = options, {}, None, {}, 0

    def start(self, sequence, tokens):
        self.pending[sequence] = list(tokens)[-1]

    def _subset(self, head):
        if self.head is None:
            vocabulary = len(head([[]])[0])
            rows = subsets.prior_subset(vocabulary, min(self.options["size"], vocabulary))
            self.head = subsets.SubsetHead(rows, vocabulary)
        return self.head

    def propose(self, sequence, width, head=None):
        if head is None:
            raise ValueError("draft-vocabulary-subset drafts from the engine's draft head: pass head")
        subset, path = self._subset(head), []
        for _ in range(width):
            row = head([list(path)], rows=subset.rows)[0]
            path.append(subset.token(max(range(len(row)), key=lambda i: (row[i], -i))))
        return {"tokens": [self.pending[sequence]] + path, "parents": list(range(-1, width)), "laws": None}

    def observe(self, sequence, emitted):
        for token in emitted:
            self.counts[token] = self.counts.get(token, 0) + 1
        self.rounds += 1
        if self.head is not None and self.rounds % self.options["refresh"] == 0:
            vocabulary, size = self.head.vocabulary, len(self.head)
            counts = {t: self.counts.get(t, 0) * vocabulary + (vocabulary - t) for t in range(vocabulary)}
            self.head = subsets.SubsetHead(subsets.choose_subset(counts, size), vocabulary)
        if emitted:
            self.pending[sequence] = emitted[-1]

    def release(self, sequence):
        self.pending.pop(sequence, None)

    # draft:device (crit_plugins.device.DEVICE_EXTENSIONS)
    def reserve(self, rows, widest, vocab, device):
        """The subset (the lowest ids at first, subset.prior_subset), the emitted counts, the rows' round counter and
        the last round's staged tokens on the device, once, before capturing rounds (the round driver calls it)."""
        import torch
        size = min(self.options["size"], vocab)
        long = dict(dtype=torch.int64, device=device)
        self._round = {"rows": torch.arange(size, **long), "counts": torch.zeros(vocab, **long),
                       "rounds": torch.zeros((), **long), "vocab": vocab, "ids": torch.arange(vocab, **long),
                       "staged": torch.full((rows, widest + 1), -1, **long), "emitted": torch.zeros(rows, **long)}
        return self._round

    def _device(self):
        found = getattr(self, "_round", None)
        if found is None:
            raise ValueError("draft-vocabulary-subset: reserve(rows, widest, vocab, device) its round buffers first")
        return found

    def _fold(self, state):
        """The last round's emitted tokens into the counts and the rows' rounds, then the subset chosen again where
        the rounds passed a multiple of `refresh` (the `size` most emitted, lower ids first on a tie, ascending; in
        place: the head's rows tensor keeps its address); capture-safe."""
        import torch
        staged = state["staged"]
        state["counts"].scatter_add_(0, staged.clamp(min=0).flatten(), (staged >= 0).long().flatten())
        before = state["rounds"].clone()
        state["rounds"] += (state["emitted"] > 0).long().sum()
        refresh = self.options["refresh"]
        due = torch.div(state["rounds"], refresh, rounding_mode="floor") > torch.div(before, refresh,
                                                                                     rounding_mode="floor")
        vocab = state["vocab"]
        key = state["counts"] * vocab + (vocab - state["ids"])
        chosen = torch.sort(key.topk(state["rows"].numel()).indices).values
        state["rows"].copy_(torch.where(due, chosen, state["rows"]))
        staged.fill_(-1)
        state["emitted"].zero_()

    def draft(self, view, head, sampler=None):
        """view.drafts[:, :width]: per depth the best of the head's subset rows (head.rows), mapped to their ids (in
        sampled rounds the sampler's draft draw over them); the last round's tokens counted first; capture-safe."""
        import torch
        if head is None:
            raise ValueError("draft-vocabulary-subset drafts from the model's device draft head: pass head")
        state = self._device()
        self._fold(state)
        head.rows = state["rows"]
        head.catch_up(view)
        rows, width = view.rows, view.width
        draw = getattr(sampler, "draft_device", None) if view.mode == "sampled" else None
        if view.mode == "sampled" and not callable(draw):
            raise ValueError("a sampled round draws its drafts from their laws: the sampler gives draft_device "
                             "(rounds:sampler)")
        tokens = view.pending[:rows]
        for depth in range(width):
            logits = head.logits(depth, tokens, view)
            if draw is not None:
                tokens = draw(view, logits, depth, state["rows"])
            else:
                tokens = state["rows"][torch.argmax(logits, dim=-1)]
            view.drafts[:rows, depth] = tokens

    def commit_round(self, view):
        """The round's emitted tokens, staged (overwritten by a truncation's second commit) and counted at the next
        round's draft; the round driver commits the draft head itself."""
        state = self._device()
        rows, cols = view.rows, view.width + 1
        state["staged"].fill_(-1)
        state["staged"][:rows, :cols] = view.emitted[:rows, :cols]
        state["emitted"].zero_()
        state["emitted"][:rows] = view.n_emitted[:rows]


def create(context, options, parts):
    return SubsetHeadDrafter(options)
