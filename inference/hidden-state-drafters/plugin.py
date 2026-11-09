"""Hidden-state drafts in the drafter slot: hidden-state-drafters' d2t mapping and confidence cut (drafting.py) as the
host form. The engine's head runs its drafter over the target's taps; its rows are over the draft vocabulary, mapped by
head.d2t. Inside captured decode rounds the same drafts, mapping and cut on the device (device_drafting.py, the
draft:device extension): the cut's bins are learned from each round's emitted tokens without the host."""
import math

from .drafters import drafting


def softmax_best(row):
    best = max(range(len(row)), key=lambda t: (row[t], -t))
    top = row[best]
    return best, 1.0 / math.fsum(math.exp(v - top) for v in row)


class HiddenStateDrafter:
    def __init__(self, options):
        self.options, self.pending, self.last = options, {}, {}
        self.cut = drafting.ConfidenceCut(options["cut_target"]) if options["cut_target"] > 0 else None
        self.round, self.targets = None, None

    def start(self, sequence, tokens):
        self.pending[sequence] = list(tokens)[-1]

    def propose(self, sequence, width, head=None):
        if head is None:
            raise ValueError("hidden-state-drafters drafts from the engine's drafter: pass head")
        d2t = getattr(head, "d2t", None)
        path, confidence = [], []
        for _ in range(width):
            token, p = softmax_best(head([list(path)])[0])
            path.append(drafting.map_tokens([token], d2t)[0] if d2t is not None else token)
            confidence.append(p)
        drafts = drafting.truncate(path, confidence, self.cut.threshold()) if self.cut else path
        self.last[sequence] = (drafts, confidence)
        return {"tokens": [self.pending[sequence]] + drafts, "parents": list(range(-1, width)), "laws": None}

    def observe(self, sequence, emitted):
        drafts, confidence = self.last.pop(sequence, ((), ()))
        if self.cut is not None:
            for d, c, e in zip(drafts, confidence, emitted):
                if d == drafting.PAD:
                    break
                self.cut.label(c, d == e)
                if d != e:
                    break
            self.cut.decay_step()
        if emitted:
            self.pending[sequence] = emitted[-1]

    def release(self, sequence):
        self.pending.pop(sequence, None)
        self.last.pop(sequence, None)

    # draft:device (crit_plugins.device.DEVICE_EXTENSIONS)
    def reserve(self, rows, widest, vocab, device):
        """The round's confidences and the device cut, once, before capturing rounds."""
        import torch
        from .drafters import device_drafting
        self.round = {"confidence": torch.zeros(rows, max(widest, 1), dtype=torch.float64, device=device),
                      "cut": device_drafting.DeviceCut(self.options["cut_target"], rows, widest, device)
                      if self.options["cut_target"] > 0 else None}
        return self.round

    def _targets(self, head, vocab, device):
        """Each draft-vocabulary id's target id (id + d2t[id]), made once per head."""
        import torch
        if self.targets is None or self.targets.numel() != vocab or getattr(self, "_mapped", None) is not head:
            d2t = getattr(head, "d2t", None)
            ids = torch.arange(vocab, dtype=torch.int64, device=device)
            self.targets = ids if d2t is None else ids + torch.as_tensor(d2t, dtype=torch.int64, device=device)
            self._mapped = head
        return self.targets

    def draft(self, view, head, sampler=None):
        """view.drafts[:, :width]: the head's draft-vocabulary choices mapped to target ids, PAD from the first below
        the cut's threshold; capture-safe."""
        import torch
        from .drafters import device_drafting
        if head is None:
            raise ValueError("hidden-state-drafters drafts from the model's device draft head: pass head")
        if self.round is None:
            raise ValueError("hidden-state-drafters: reserve(rows, widest, vocab, device) its round buffers first")
        rows, width = view.rows, view.width
        cut = self.round["cut"]
        if cut is not None:
            cut.fold()   # the last round's labels, before this round's threshold
        head.catch_up(view)
        if not width:
            return
        draw = getattr(sampler, "draft_device", None) if view.mode == "sampled" else None
        if view.mode == "sampled" and not callable(draw):
            raise ValueError("a sampled round draws its drafts from their laws: the sampler gives draft_device "
                             "(rounds:sampler)")
        confidence = self.round["confidence"][:rows, :width]
        tokens = view.pending[:rows]
        for depth in range(width):
            logits = head.logits(depth, tokens, view)
            targets = self._targets(head, logits.shape[-1], logits.device)
            if draw is not None:
                tokens = draw(view, logits, depth, targets)
            else:
                tokens = targets[torch.argmax(logits, dim=-1)]
            confidence[:, depth] = device_drafting.confidences(logits)
            view.drafts[:rows, depth] = tokens
        if cut is not None:
            device_drafting.truncate(view.drafts[:rows, :width], confidence, cut.threshold())

    def commit_round(self, view):
        """The round's labels for the cut, staged (idempotent from the round's start: a truncation commits again) and
        folded in at the next round's draft; the round driver commits the draft head itself."""
        if self.round is not None and self.round["cut"] is not None:
            rows, width = view.rows, view.width
            # Staged, not folded: a truncation commits again; a row that emitted nothing gets no label and no decay.
            self.round["cut"].stage(view.drafts[:rows, :width], self.round["confidence"][:rows, :width],
                                    view.emitted[:rows], view.n_emitted[:rows])


def create(context, options, parts):
    return HiddenStateDrafter(options)
