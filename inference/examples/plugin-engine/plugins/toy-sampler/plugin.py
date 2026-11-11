"""The toy's draw: the highest logit the row's allowed set keeps (all of them without one), the lower id first on
ties, whatever the sampling; over rows of numbers (host form) or a logits tensor (device form)."""


def _best(row, allowed=None):
    keep = range(len(row)) if allowed is None else list(allowed)
    return max(keep, key=lambda i: (row[i], -i))


def _allowed(params, rows):
    found = params if isinstance(params, list) else [params] * rows
    return [p.get("allowed") for p in found]


class Sampler:
    def select(self, rows, params):
        return [_best(row, allowed) for row, allowed in zip(rows, _allowed(params, len(rows)))]

    def law(self, row, params):
        return [(_best(row, params.get("allowed")), 1.0)]

    def predictions(self, rows, drafts, params, draft_laws=None):
        return [_best(row, params.get("allowed")) for row in rows]

    def sample(self, logits, params):
        import torch
        out = torch.empty(logits.shape[0], dtype=torch.int64, device=logits.device)
        for r, allowed in enumerate(_allowed(params, logits.shape[0])):
            values = logits[r].float()
            if allowed is not None:
                keep = (allowed.mask(values.device, values.shape[-1]) if hasattr(allowed, "mask") else
                        torch.zeros(values.shape[-1], dtype=torch.bool, device=values.device).index_fill_(
                            0, torch.tensor(sorted(allowed), dtype=torch.int64, device=values.device), True))
                values = values.masked_fill(~keep, float("-inf"))
            out[r] = torch.argmax(values)   # the first of equal maxima: the lower id
        return out


def create(context, options, parts):
    return Sampler()
