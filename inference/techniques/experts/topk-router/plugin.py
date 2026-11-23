"""Expert routing in the router slot: heterogeneous-expert-dispatch's route (routing.py) under one Routing, and its
device form, the same rule over a step's router logits (FP32, ties to the lower expert and group id): one launch of
crit's kernel topk-routing (a warp per token row; built on this GPU, tuned or default knobs) where it builds and the
rule has no group limit, else torch.

The device form is capture-safe: no host synchronization, fixed shapes for a fixed token count, a bias moved to the
device once, at its first (eager) call, and kept. A row with NaN logits (a captured round's junk positions) still
routes: NaN ranks below every number, -inf included, so such a row picks valid distinct experts (the lowest ids when
the whole row is NaN) and its weights are 0. So a captured prompt chunk's padding (prompts:router) routes to valid
experts with finite weights, which the model masks to ids -1 by the batch's valid_t before the experts see them."""
from dataclasses import replace
from pathlib import Path

from .dispatch import routing

HERE = Path(__file__).resolve().parent
KERNEL = "topk-routing"
NAN_KEY, MASKED_KEY = -(1 << 31) + 1, -(1 << 31)   # selection keys: NaN below -inf, experts a group limit drops below NaN
BIASES = 256   # device copies of biases kept


def order_keys(torch, values):
    """values (FP32) as int32 keys of the same order (-0 and +0 equal), NaN below every number, -inf included."""
    bits = torch.where(values == 0, torch.zeros_like(values), values).view(torch.int32)
    keys = torch.where(bits < 0, bits ^ 0x7FFFFFFF, bits)
    return torch.where(torch.isnan(values), torch.full_like(keys, NAN_KEY), keys)


class TopKRouter:
    def __init__(self, options):
        options = dict(options)
        use_kernel = options.pop("kernel", True)
        self.routing = routing.Routing(**options)
        self._bias = {}
        self.kernel = None
        if use_kernel:
            try:
                import torch
                cuda = torch.cuda.is_available()
            except ImportError:
                cuda = False
            if cuda:
                from crit_plugins.kernels import load_kernel
                self.kernel = load_kernel(KERNEL, folder=HERE)

    def route(self, logits, top, bias=None):
        rule = replace(self.routing, correction_bias=bias is not None)
        return [tuple(list(part) for part in routing.route(row, top, rule, bias)) for row in logits]

    def _device_bias(self, bias, device):
        """A bias as an FP32 device tensor: one on the device as given; another tensor (a CPU one, another dtype) or a
        list converted at its first call and kept, keyed by the tensor and its version (the tensor is held, so its key
        is never reused), so a later or captured call copies nothing."""
        import torch
        if isinstance(bias, torch.Tensor):
            if bias.device == device and bias.dtype == torch.float32:
                return bias
            key = (str(device), id(bias), bias._version)
            if key not in self._bias:
                self._keep(key, (bias, bias.detach().to(device, torch.float32)))
            return self._bias[key][1]
        key = (str(device), tuple(float(b) for b in bias))
        if key not in self._bias:
            self._keep(key, (None, torch.tensor(key[1], dtype=torch.float32, device=device)))
        return self._bias[key][1]

    def _keep(self, key, value):
        self._bias[key] = value
        while len(self._bias) > BIASES:   # a model has a bias a layer: more means fresh tensors each call
            self._bias.pop(next(iter(self._bias)))

    def select(self, logits, top, bias=None):
        """The device form: (ids [tokens][top] int64, weights [tokens][top] FP32) by route()'s rule."""
        import torch
        rule = replace(self.routing, correction_bias=bias is not None)
        tokens, experts = logits.shape
        rule.validate(experts, top)
        x = logits.float()
        b = self._device_bias(bias, x.device) if bias is not None else None
        if (self.kernel is not None and x.is_cuda and rule.selected_groups >= rule.groups
                and experts <= self.kernel.MAX_EXPERTS and top <= self.kernel.MAX_TOP):
            x = x if x.stride(-1) == 1 else x.contiguous()
            return self.kernel.route(x, top, b, rule.scoring == "softmax", rule.score_before_selection,
                                     rule.renormalize, rule.renormalize_epsilon, rule.renormalize_floor, rule.scaling)
        if not rule.score_before_selection:
            scores = x
        else:
            scores = torch.softmax(x, -1) if rule.scoring == "softmax" else torch.sigmoid(x)
        choice = scores + b if b is not None else scores
        keys = order_keys(torch, choice)   # NaN below every number, -inf included: a junk row still routes
        if rule.selected_groups < rule.groups:
            size = experts // rule.groups
            low = torch.full_like(choice, float("-inf"))
            ranked = torch.where(torch.isnan(choice), low, choice)   # a group's score counts a NaN as -inf
            rank = ranked.view(tokens, rule.groups, size).topk(rule.group_score_top, -1).values.sum(-1)
            kept = torch.sort(order_keys(torch, rank), dim=-1, descending=True, stable=True).indices
            kept = kept[:, :rule.selected_groups]
            allowed = torch.zeros(tokens, rule.groups, dtype=torch.bool, device=x.device).scatter_(1, kept, True)
            keys = keys.masked_fill(~allowed.repeat_interleave(size, 1), MASKED_KEY)
        ids = torch.sort(keys, dim=-1, descending=True, stable=True).indices[:, :top]
        if rule.score_before_selection:
            weights = scores.gather(1, ids)
        elif rule.scoring == "softmax":
            weights = torch.softmax(x.gather(1, ids), -1)
        else:
            weights = torch.sigmoid(x.gather(1, ids))
        if rule.renormalize:
            weights = weights / torch.clamp(weights.sum(-1, keepdim=True) + rule.renormalize_epsilon,
                                            min=rule.renormalize_floor)
        weights = weights * rule.scaling
        return ids, torch.where(torch.isnan(weights), torch.zeros_like(weights), weights)


def create(context, options, parts):
    return TopKRouter(options)
