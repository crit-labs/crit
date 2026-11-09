"""Constrained sampling in the sampler slot: token-constraints' laws (constrained.py) drawn with Core's uniform
(token-sampling's sampling.py), so structured output stays exact under sampled speculation.

The device form masks each constrained row's logits by its Allowed set before any rule, then draws by token-sampling's
device law: sample(logits, params) per step, and staged(rows, device, vocabulary) for a step captured in a CUDA graph,
whose masks sit in a buffer at a fixed address that set(params) fills from the host's bitmask words before the
launch. Inside captured decode rounds (rounds:sampler) it is token-sampling's round form (device_rounds.py) with each
slot's mask staged beside its rules (RoundMasks): every logits row of a constrained slot masked before the law, and
its drafts cut, so it emits one token a round, its column 0's, whose allowed set is the one Core staged. Core decodes
constrained requests plainly (one token per step, Core's StepRequest.plain): a verify window under a constraint needs
the matcher's mask at every draft position, which Core computes on the host only once the tokens before it are known;
predictions() stays the host form of that."""
from .constraints import constrained
from .token_sampling import device_law, sampling


def rows_params(rows, params):
    found = params if isinstance(params, list) else [params] * len(rows)
    if len(found) != len(rows) or not all(isinstance(p, dict) for p in found):
        raise ValueError("params is one dict for every row or a list of one dict per row")
    return found


class ConstrainedSampler:
    _rounds, kernel = None, "native"

    def _full(self, row, params, allowed):
        """The full-vocabulary law of a row under `allowed` (every token when None)."""
        allowed = sorted(set(allowed)) if allowed is not None else list(range(len(row)))
        return constrained.constrained_law([float(v) for v in row], allowed, params.get("temperature", 0.0),
                                           params.get("top_k", 0), params.get("top_p", 1.0), params.get("min_p", 0.0))

    @staticmethod
    def _ranked(row, probabilities):
        order = sorted(range(len(row)), key=lambda t: (-row[t], t))
        return [(t, probabilities[t]) for t in order if probabilities[t] > 0]

    def law(self, row, params):
        return self._ranked(row, self._full(row, params, params.get("allowed")))

    def _draw(self, row, params, allowed, position, purpose=sampling.SAMPLE):
        pairs = self._ranked(row, self._full(row, params, allowed))
        if not params.get("temperature", 0.0) > 0:
            return pairs[0][0]
        u = sampling.uniform(params.get("seed", 0), position, params.get("stream", 0), purpose)
        return sampling.walk(pairs, u * sum(p for _, p in pairs))

    def select(self, rows, params):
        return [self._draw(row, p, p.get("allowed"), p.get("position", 0))
                for row, p in zip(rows, rows_params(rows, params))]

    def predictions(self, rows, drafts, params, draft_laws=None):
        allowed = params.get("allowed")
        per_column = allowed if allowed and isinstance(allowed[0], (list, tuple, set)) else [allowed] * len(rows)
        sampled = params.get("temperature", 0.0) > 0
        out = []
        for c, (row, a) in enumerate(zip(rows, per_column)):
            at = params.get("position", 0) + c
            if c == len(drafts) or not sampled or not draft_laws or drafts[c] < 0:
                out.append(self._draw(row, params, a, at))
                continue
            d, target = drafts[c], self._full(row, params, a)
            draft = [draft_laws[c].get(t, 0.0) for t in range(len(row))]
            u = sampling.uniform(params.get("seed", 0), at, params.get("stream", 0), sampling.ACCEPT)
            if u < constrained.accept_probability(target, draft, d):
                out.append(d)
                continue
            try:
                repair = constrained.residual(target, draft)
            except ValueError:   # no residual mass: the draft law equals the target law
                out.append(d)
                continue
            pairs = self._ranked(row, repair)
            u = sampling.uniform(params.get("seed", 0), at, params.get("stream", 0), sampling.RESIDUAL)
            out.append(sampling.walk(pairs, u * sum(p for _, p in pairs)))
        return out

    # -- the device form ---------------------------------------------------------------------------------------------------
    def sample(self, logits, params):
        """One token per logits row ([rows][vocabulary], a tensor on the device) by Core's law, each constrained row
        (params' allowed: a crit_plugins.Allowed, or sorted ids) masked to its allowed tokens before the rules."""
        return device_law.sample(logits, params)

    def staged(self, rows, device, vocabulary):
        """Buffers for up to `rows` rows over `vocabulary` tokens at fixed addresses (StagedMasks): set(params) before
        a step, sample(logits) inside it with no host work, the constrained rows masked before the rules."""
        return StagedMasks(rows, device, vocabulary)

    # rounds:sampler (crit_plugins.device.DEVICE_EXTENSIONS): token-sampling's round form under RoundMasks.
    def _round_sampling(self):
        if self._rounds is None:
            from .token_sampling import device_rounds
            inner = device_rounds.RoundSampling()
            inner.kernel = self.kernel
            self._rounds = RoundMasks(inner)
        return self._rounds

    def reserve(self, rows, widest, vocab, device, logprobs=0):
        """The round buffers for up to `rows` rows at up to `widest` drafts over `vocab` logits, the slots' masks
        among them, once, before capturing rounds."""
        return self._round_sampling().reserve(rows, widest, vocab, device, logprobs)

    def stage(self, slot, params):
        """The rules of the request in round slot `slot` (select()'s params; its allowed, a crit_plugins.Allowed or
        sorted ids over the vocabulary, the mask of its next token), in the round's device buffers. Outside a capture."""
        self._round_sampling().stage(slot, params)

    def laws(self, rows, width):
        return self._round_sampling().inner.laws(rows, width)

    def draft_device(self, view, logits, depth, targets=None):
        """token-sampling's draft (a constrained slot's drafts are cut when its round verifies them)."""
        return self._round_sampling().inner.draft_device(view, logits, depth, targets)

    def predictions_device(self, view, logits, draft_laws=None):
        """Each verify column's prediction by token-sampling's round form over the logits, every row of a
        constrained slot masked to its allowed set first and its drafts cut (it emits its column 0's token);
        capture-safe."""
        return self._round_sampling().predictions_device(view, logits, draft_laws)

    def logprobs_device(self, view, logits, tokens):
        """Core's TokenLogprob of the emitted tokens over the masked logits rows (a constrained slot's law)."""
        return self._round_sampling().logprobs_device(view, logits, tokens)

    def facts(self):
        return self._rounds.inner_facts() if self._rounds is not None else {}


class StagedMasks:
    """A step's sampling rules and allowed masks in device buffers: set(params) copies each row's rules (token-sampling's
    staged form) and its Allowed bitmask words (crit::core::TokenMatcher::allowed_bits: bit t % 32 of word t // 32) into
    pinned host buffers and from them to the device in one asynchronous copy; sample(logits) unpacks the words, puts
    every disallowed token of a constrained row at -inf, and draws by the staged law. Capture-safe."""

    def __init__(self, rows, device, vocabulary):
        import torch
        from .token_sampling import device_staged
        self.torch, self.rows, self.vocabulary = torch, int(rows), int(vocabulary)
        self.device = torch.device(device)
        self.words = (self.vocabulary + 31) // 32
        pinned = self.device.type == "cuda"
        self.h_words = torch.zeros(self.rows, self.words, dtype=torch.int32, pin_memory=pinned)
        self.h_masked = torch.zeros(self.rows, dtype=torch.bool, pin_memory=pinned)
        self.d_words = torch.zeros(self.rows, self.words, dtype=torch.int32, device=self.device)
        self.d_masked = torch.zeros(self.rows, dtype=torch.bool, device=self.device)
        self.bits = torch.arange(32, dtype=torch.int32, device=self.device)
        self.law = device_staged.Staged(self.rows, self.device)
        self._copied = None

    def set(self, params):
        """The step's rows in row order: dicts as sample() takes them, a constrained row's allowed a
        crit_plugins.Allowed over this vocabulary."""
        torch = self.torch
        if len(params) > self.rows:
            raise ValueError(f"{len(params)} rows for staged buffers of {self.rows}")
        if self._copied is not None:
            self._copied.synchronize()
        rules = []
        self.h_masked[len(params):] = False     # rows past this step's: no mask left from an earlier one
        for r, p in enumerate(params):
            allowed = p.get("allowed")
            self.h_masked[r] = allowed is not None
            if allowed is not None:
                if not hasattr(allowed, "words"):
                    from crit_plugins import Allowed
                    allowed = Allowed.of(allowed, self.vocabulary)
                if allowed.vocabulary != self.vocabulary:
                    raise ValueError(f"an allowed set over {allowed.vocabulary} tokens for a vocabulary of "
                                     f"{self.vocabulary}")
                if not allowed:
                    raise ValueError("an allowed set holds at least one token")
                self.h_words[r] = torch.frombuffer(bytearray(allowed.words.tobytes()), dtype=torch.int32)
            rules.append({k: v for k, v in p.items() if k != "allowed"})
        self.law.set(rules)
        self.d_words.copy_(self.h_words, non_blocking=True)
        self.d_masked.copy_(self.h_masked, non_blocking=True)
        if self.device.type == "cuda":
            self._copied = torch.cuda.Event()
            self._copied.record()

    def mask(self, rows, width):
        """The allowed tokens of the first `rows` rows over `width` logit columns ([rows][width] bool): every token of
        the vocabulary for an unconstrained row, its allowed ones for a constrained one, none past the vocabulary."""
        return _allowed(self.torch, self.d_words, self.d_masked, self.bits, rows, width, self.vocabulary)

    def sample(self, logits):
        """One token per row of logits ([rows][width]): each constrained row masked, then the staged law."""
        rows, width = logits.shape
        masked = logits.masked_fill(~self.mask(rows, width), float("-inf"))
        return self.law.sample(masked)


class RoundMasks:
    """token-sampling's round form (device_rounds.RoundSampling) with each round slot's allowed mask: stage(slot,
    params) keeps the slot's rules in the inner form and its Allowed bitmask words in device buffers at fixed addresses;
    a round masks every logits row of a constrained slot before the inner form predicts or scores, and cuts its drafts
    (-1: a draft never matched, the target's own draw taken), so a constrained slot emits its column 0's token, whose
    allowed set is the one staged. Capture-safe."""

    def __init__(self, inner):
        self.inner, self.rows, self.vocabulary = inner, 0, 0
        self.d_words = self.d_masked = None
        self._masks = {}    # slot -> its Allowed words (int32 tensor), kept across a reserve()

    def reserve(self, rows, widest, vocab, device, logprobs=0):
        import torch
        state = self.inner.reserve(rows, widest, vocab, device, logprobs)
        device = torch.device(device)
        if self.d_words is None or (self.rows, self.vocabulary, self.d_words.device) != (rows, vocab, device):
            self.rows, self.vocabulary = int(rows), int(vocab)
            self.d_words = torch.zeros(self.rows, (self.vocabulary + 31) // 32, dtype=torch.int32, device=device)
            self.d_masked = torch.zeros(self.rows, dtype=torch.bool, device=device)
            self.bits = torch.arange(32, dtype=torch.int32, device=device)
            for slot, words in sorted(self._masks.items()):
                if slot < self.rows:
                    self.d_words[slot].copy_(words)
                    self.d_masked[slot] = True
        return state

    def stage(self, slot, params):
        import torch
        if self.d_words is None:
            raise ValueError("reserve(rows, widest, vocab, device) the sampler's round buffers before staging")
        if not 0 <= slot < self.rows:
            raise ValueError(f"round slot {slot} outside the {self.rows} reserved")
        allowed = params.get("allowed")
        self.inner.stage(slot, {k: v for k, v in params.items() if k != "allowed"})
        if allowed is None:
            self._masks.pop(int(slot), None)
            self.d_masked[slot] = False
            return
        if not hasattr(allowed, "words"):
            from crit_plugins import Allowed
            allowed = Allowed.of(allowed, self.vocabulary)
        if allowed.vocabulary != self.vocabulary:
            raise ValueError(f"an allowed set over {allowed.vocabulary} tokens for round logits of {self.vocabulary}")
        if not allowed:
            raise ValueError("an allowed set holds at least one token")
        words = torch.frombuffer(bytearray(allowed.words.tobytes()), dtype=torch.int32)
        self._masks[int(slot)] = words
        self.d_words[slot].copy_(words)
        self.d_masked[slot] = True

    def _masked(self, view, logits):
        import torch
        rows = view.rows
        x = logits.reshape(rows, -1, logits.shape[-1])
        allowed = _allowed(torch, self.d_words, self.d_masked, self.bits, rows, x.shape[-1], self.vocabulary)
        return x.masked_fill(~allowed[:, None, :], float("-inf")).reshape(logits.shape)

    def predictions_device(self, view, logits, draft_laws=None):
        if view.width:
            view.drafts[:view.rows, :view.width].masked_fill_(self.d_masked[:view.rows, None], -1)
        return self.inner.predictions_device(view, self._masked(view, logits), draft_laws)

    def logprobs_device(self, view, logits, tokens):
        return self.inner.logprobs_device(view, self._masked(view, logits), tokens)

    def inner_facts(self):
        state = self.inner._round
        return {"round_sampling": "native" if state.native is not None else "torch"} if state is not None else {}


def _allowed(torch, words, masked, bits, rows, width, vocabulary):
    """[rows][width] bool: each row's allowed columns from its bitmask words (bit t % 32 of word t // 32), every
    column for a row not masked, none past the vocabulary."""
    found = ((words[:rows, :, None] >> bits) & 1).view(rows, -1).bool()
    if found.shape[1] >= width:
        found = found[:, :width]
    else:
        found = torch.cat([found, found.new_zeros(rows, width - found.shape[1])], dim=1)
    inside = torch.arange(width, device=words.device) < vocabulary
    return (found | ~masked[:rows, None]) & inside[None, :]


def create(context, options, parts):
    found = ConstrainedSampler()
    found.kernel = options["rounds"]
    return found
