"""Core's sampling law in the sampler slot: token-sampling's FP64 mirror of the device sampler (sampling.py) as the
host form, and the same law over logits rows in torch (device_law.py) as the device form; inside captured decode
rounds, the rounds:sampler extension (device_rounds.py): drafts drawn and verify columns predicted on the device by
the CUDA sampler (sampling.cu), rejection sampling with Core's draws."""
import math

from .token_sampling import device_law, sampling


def rows_params(rows, params):
    """One params dict per row: `params` itself for every row, or its own list."""
    found = params if isinstance(params, list) else [params] * len(rows)
    if len(found) != len(rows) or not all(isinstance(p, dict) for p in found):
        raise ValueError("params is one dict for every row or a list of one dict per row")
    return found


def masked(row, allowed):
    """The row with every token outside `allowed` at -inf (all of it without an allowed set)."""
    if allowed is None:
        return [float(v) for v in row]
    keep = set(allowed)
    if not keep:
        raise ValueError("an allowed set holds at least one token")
    return [float(v) if t in keep else -math.inf for t, v in enumerate(row)]


def rule(p):
    return (p.get("temperature", 0.0), p.get("top_p", 1.0), p.get("top_k", 0), p.get("min_p", 0.0))


class CoreSampler:
    _rounds = None

    def sample(self, logits, params):
        """The device form: one token per logits row (a tensor on the device), by Core's law."""
        return device_law.sample(logits, params)

    def staged(self, rows, device):
        """The device form's staged buffers for up to `rows` rows (device_staged.Staged): set(params) before a step,
        sample(logits) inside it with no host read, so a model can capture its sampling in a CUDA graph."""
        from .token_sampling import device_staged
        return device_staged.Staged(rows, device)

    # rounds:sampler (crit_plugins.device.DEVICE_EXTENSIONS): device_rounds.RoundSampling, imported with torch.
    def _round_sampling(self):
        if self._rounds is None:
            from .token_sampling import device_rounds
            self._rounds = device_rounds.RoundSampling()
            self._rounds.kernel = getattr(self, "kernel", "native")
        return self._rounds

    def reserve(self, rows, widest, vocab, device, logprobs=0):
        """The round buffers for up to `rows` rows at up to `widest` drafts over `vocab` logits, once, before
        capturing rounds (device_rounds.RoundSampling.reserve)."""
        return self._round_sampling().reserve(rows, widest, vocab, device, logprobs)

    def stage(self, slot, params):
        """The rules of the request in round slot `slot` (select()'s params), in the round's device buffers."""
        self._round_sampling().stage(slot, params)

    def laws(self, rows, width):
        """The round's draft laws (device_rounds.DraftLaws): what a drafter writes and view.draft_laws holds."""
        return self._round_sampling().laws(rows, width)

    def draft_device(self, view, logits, depth, targets=None):
        """Each row's draft at `depth` from its draft logits: argmax, or in sampled rounds Core's draft draw from the
        processed law with the law kept in view.draft_laws; targets maps a draft vocabulary's columns to target ids;
        capture-safe."""
        return self._round_sampling().draft_device(view, logits, depth, targets)

    def predictions_device(self, view, logits, draft_laws=None):
        """Each verify column's prediction: argmax for greedy rows, rejection sampling against draft_laws for
        sampled ones, by Core's law and draws as predictions(); capture-safe."""
        return self._round_sampling().predictions_device(view, logits, draft_laws)

    def logprobs_device(self, view, logits, tokens):
        """A round's emitted tokens' log-probabilities into view.logprobs (and their ranked alternatives): torch's FP32
        log_softmax of the logits rows, the step and score paths' arithmetic (code/device_rounds.py)."""
        return self._round_sampling().logprobs_device(view, logits, tokens)

    def facts(self):
        """The engine facts of its rounds: what samples inside them (round_sampling: native, the sampling law built
        for the device, or torch, the torch law), once reserve() made their buffers."""
        state = self._rounds._round if self._rounds is not None else None
        if state is None:
            return {}
        return {"round_sampling": "native" if state.native is not None else "torch"}

    def select(self, rows, params):
        out = []
        for row, p in zip(rows, rows_params(rows, params)):
            temperature, top_p, top_k, min_p = rule(p)
            out.append(sampling.select(masked(row, p.get("allowed")), temperature, top_p, top_k, p.get("seed", 0),
                                       p.get("position", 0), p.get("stream", 0), min_p))
        return out

    def law(self, row, params):
        temperature, top_p, top_k, min_p = rule(params)
        values = masked(row, params.get("allowed"))
        if not temperature > 0:
            return [(sampling.law(values, 0.0)[0][0], 1.0)]
        return sampling.nucleus(values, temperature, top_p, top_k, min_p)

    def predictions(self, rows, drafts, params, draft_laws=None):
        allowed = params.get("allowed")
        per_column = allowed if allowed and isinstance(allowed[0], (list, tuple, set)) else [allowed] * len(rows)
        values = [masked(row, a) for row, a in zip(rows, per_column)]
        temperature, top_p, top_k, min_p = rule(params)
        return sampling.verify(values, list(drafts), temperature, top_p, top_k, min_p, params.get("seed", 0),
                               params.get("position", 0), params.get("stream", 0), draft_laws)


def create(context, options, parts):
    found = CoreSampler()
    found.kernel = options["rounds"]
    return found
