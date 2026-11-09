"""drafting.py's rules for every row of a captured decode round, in torch on the device: an EAGLE-style head's drafts
over its draft vocabulary mapped to target ids by d2t, and ConfidenceCut learned and applied on the device, so a round
needs no host between rounds (the drafter slot's "draft:device" extension).

Per round, after head.catch_up(view): at each depth the head's logits over the draft vocabulary ([rows][draft vocab],
head.logits(depth, tokens, view)); the draft is the argmax (sampled rounds: the sampler's draft draw over the draft
vocabulary, its law mapped to target ids), mapped by target = draft + d2t[draft], with its confidence the argmax's
softmax probability (FP64, softmax_best's). From the first draft whose confidence is below the cut's threshold on,
drafts become PAD (truncate), which no prediction matches.

DeviceCut is ConfidenceCut over device tensors: bins of `bin_width` holding decayed (tested, accepted) counts in FP64.
A round's commit stages its labels (idempotent: a truncation's second commit overwrites them) and the next round's
draft folds them in as the host form's observe() labels each row that emitted (each kept draft while the emitted token
equals it, the first mismatch included, never a PAD), decaying once per such row; threshold() is ConfidenceCut's
(the lowest edge of the top bins with at least `min_count` labels that all accept at least `target`; -inf until
`burn_in` labels, +inf when even the best populated bin falls short). A round's rows decay in one product (the host
decays after each row's labels, so their sums differ by FP64 rounding only). Confidences fall into bins
floor(c / bin_width) for c in [0, 1].
"""
import math

import torch

PAD = -1


def confidences(logits):
    """softmax_best's probability of each row's argmax ([rows][vocab] -> [rows] FP64)."""
    x = logits.double()
    top = x.max(dim=-1, keepdim=True).values
    return 1.0 / torch.exp(x - top).sum(-1)


def truncate(drafts, confidence, threshold):
    """drafts [rows][width] in place: PAD from the first draft whose confidence is below threshold ([] tensor)."""
    cut = (confidence < threshold).long().cumsum(1) > 0
    drafts.copy_(torch.where(cut, torch.full_like(drafts, PAD), drafts))


class DeviceCut:
    """ConfidenceCut's state and rule on `device`, for rounds of up to `rows` rows at up to `widest` drafts.

    A round's labels are staged by stage() (commit_round: overwritten, so a truncation's second commit leaves the
    round's final emission) and folded into the bins by fold() at the next round's draft, before its threshold()."""

    def __init__(self, target, rows, widest, device, bin_width=0.05, decay=0.995, min_count=8.0, burn_in=64.0):
        if not 0.0 < target < 1.0:
            raise ValueError("target acceptance must be in (0, 1)")
        self.target, self.bin_width, self.decay = target, bin_width, decay
        self.min_count, self.burn_in = min_count, burn_in
        self.bins = int(math.floor(1.0 / bin_width)) + 1
        double = dict(dtype=torch.float64, device=device)
        long = dict(dtype=torch.int64, device=device)
        self.tested = torch.zeros(self.bins, **double)
        self.accepted = torch.zeros(self.bins, **double)
        self.total = torch.zeros((), **double)
        self.index = torch.arange(self.bins, device=device)
        self.edges = self.index.double() * bin_width
        widest = max(widest, 1)
        self.drafts = torch.full((rows, widest), PAD, **long)        # the last round's drafts, PAD past its own
        self.confidence = torch.zeros(rows, widest, **double)
        self.emitted = torch.full((rows, widest), -1, **long)
        self.n_emitted = torch.zeros(rows, **long)                   # 0: the row emitted nothing (or was not in it)
        self.columns = torch.arange(widest, **long)

    def threshold(self):
        """ConfidenceCut.threshold() as a [] FP64 tensor; capture-safe."""
        eligible = self.tested >= self.min_count
        fail = eligible & (self.accepted / self.tested.clamp(min=1e-300) < self.target)
        last_fail = torch.where(fail, self.index, torch.full_like(self.index, -1)).max()
        keep = eligible & (self.index > last_fail)
        lowest = torch.where(keep, self.index, torch.full_like(self.index, self.bins)).min()
        edge = self.edges.index_select(0, lowest.clamp(max=self.bins - 1).view(1)).view(())   # no host read
        inf = torch.full_like(self.total, math.inf)
        cut = torch.where(lowest < self.bins, edge, inf)
        populated = eligible.any()
        return torch.where((self.total < self.burn_in) | ~populated, -inf, cut)

    def stage(self, drafts, confidence, emitted, n_emitted):
        """The round's drafts and confidences [rows][width] as drafted (PAD where cut), its emitted tokens and counts:
        kept until the next fold (overwritten by a second commit of the same round); capture-safe."""
        rows, width = drafts.shape
        self.drafts.fill_(PAD)
        self.n_emitted.zero_()
        if width:
            self.drafts[:rows, :width] = drafts
            self.confidence[:rows, :width] = confidence
            self.emitted[:rows, :width] = emitted[:, :width]
        self.n_emitted[:rows] = n_emitted

    def fold(self):
        """The staged round's labels into the bins, as the host form's observe() labels each row that emitted (each
        kept draft while the emitted token equals it, the first mismatch included, never a PAD), decaying once per such
        row; then nothing is staged. Capture-safe."""
        active = self.n_emitted > 0
        drafts, column = self.drafts, self.columns[None, :]
        hit = drafts == self.emitted
        before = torch.cat([torch.ones_like(hit[:, :1]), hit[:, :-1]], dim=1).long().cumprod(1).bool()
        labeled = before & (drafts >= 0).long().cumprod(1).bool() & (column < self.n_emitted[:, None]) & active[:, None]
        weight = self._weights(active)   # row r's labels decay by the rows after it (the host decays after each row)
        b = torch.floor(self.confidence / self.bin_width).long().clamp(0, self.bins - 1)
        w = torch.where(labeled, weight[:, None].expand_as(self.confidence), torch.zeros_like(self.confidence))
        tested = torch.zeros_like(self.tested).scatter_add_(0, b.flatten(), w.flatten())
        accepted = torch.zeros_like(self.accepted).scatter_add_(0, b.flatten(), (w * hit).flatten())
        for t, added in ((self.tested, tested), (self.accepted, accepted)):
            t.mul_(weight[0]).add_(added)
        self.total.mul_(weight[0]).add_(w.sum())
        self.n_emitted.zero_()

    def _weights(self, active):
        """weight[r] = decay ** (active rows from r on): the decay a label of row r gets by the round's end; weight[0]
        decays the state from before the round."""
        after = active.long().flip(0).cumsum(0).flip(0)
        return torch.pow(torch.full_like(after, self.decay, dtype=torch.float64), after.double())
