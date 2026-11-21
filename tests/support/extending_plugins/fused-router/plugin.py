"""topk-router extended: a subclass of its TopKRouter whose device form selects the default rule's experts in one
top-k over keys of topk-router's own order (order_keys, then the lower id among equals), and leaves every other rule,
and the host reference, to topk-router's code."""
from .base.plugin import TopKRouter, order_keys


class FusedRouter(TopKRouter):
    def __init__(self, options):
        options = dict(options)
        self.fused = options.pop("fused")
        super().__init__(options)

    def select(self, logits, top, bias=None):
        rule = self.routing
        plain = (rule.scoring == "softmax" and not rule.score_before_selection and not rule.renormalize
                 and rule.selected_groups >= rule.groups and bias is None)
        if not self.fused or not plain:
            return super().select(logits, top, bias)
        import torch
        tokens, experts = logits.shape
        rule.validate(experts, top)
        x = logits.float()
        lower = torch.arange(experts - 1, -1, -1, dtype=torch.int64, device=x.device)   # the lower id ranks higher
        keys = order_keys(torch, x).long() * experts + lower
        ids = keys.topk(top, -1).indices
        weights = torch.softmax(x.gather(1, ids), -1) * rule.scaling
        return ids, torch.where(torch.isnan(weights), torch.zeros_like(weights), weights)


def create(context, options, parts):
    return FusedRouter(options)
