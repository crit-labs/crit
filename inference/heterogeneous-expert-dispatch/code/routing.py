"""One description of top-k expert routing and its host reference.

`Routing` states how one token's router logits become selected expert IDs and weights:
the scoring function and whether it runs before or after selection, an optional
per-expert correction bias that only changes the selection, an optional group limit,
renormalization and a routed scaling factor. `route` is the host reference an engine's
device routing kernel is checked against for the same fields, and `routing_from_config`
maps the config.json fields Core parses onto it. A model's own values belong to its
engine package; nothing here is chosen per model family.
"""
from dataclasses import dataclass
import math

SCORINGS = ("softmax", "sigmoid")
# topk_method -> how many of a group's best choice scores rank that group; 0 ignores groups.
GROUP_SCORE_TOP = {"greedy": 0, "group_limited_greedy": 1, "noaux_tc": 2}
# Values a config may leave to its reference code; the engine supplies them in defaults.
CONFIG_FIELDS = ("scoring_func", "norm_topk_prob", "routed_scaling_factor", "n_group", "topk_group")


@dataclass(frozen=True)
class Routing:
    """Router semantics; the defaults are softmax over the top-k raw logits.

    scoring: "softmax" or "sigmoid".
    score_before_selection: score every expert, then select on the scores; otherwise
        select on raw logits and score only the selected (softmax over them, or sigmoid).
    correction_bias: a per-expert bias is added to the selection value only; weights
        use the unbiased scores.
    groups, selected_groups, group_score_top: experts form `groups` equal contiguous
        groups; each group ranks by the sum of its `group_score_top` best selection
        values and only experts of the best `selected_groups` groups can be selected.
    renormalize: divide the selected weights by max(sum + renormalize_epsilon,
        renormalize_floor).
    scaling: multiply every final weight.
    Ties select the lower expert ID, and the lower group index for groups.
    """
    scoring: str = "softmax"
    score_before_selection: bool = False
    correction_bias: bool = False
    groups: int = 1
    selected_groups: int = 1
    group_score_top: int = 1
    renormalize: bool = False
    renormalize_epsilon: float = 0.0
    renormalize_floor: float = 0.0
    scaling: float = 1.0

    def validate(self, experts, top):
        if self.scoring not in SCORINGS:
            raise ValueError(f"scoring must be one of {SCORINGS}, got {self.scoring!r}")
        for name in ("score_before_selection", "correction_bias", "renormalize"):
            if not isinstance(getattr(self, name), bool):
                raise ValueError(f"{name} must be a bool")
        for name in ("groups", "selected_groups", "group_score_top"):
            if not isinstance(getattr(self, name), int) or isinstance(getattr(self, name), bool) or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer")
        if not isinstance(experts, int) or not isinstance(top, int) or experts < 1 or not 1 <= top <= experts:
            raise ValueError("Routing needs 1 <= top <= experts")
        if experts % self.groups or self.selected_groups > self.groups:
            raise ValueError("Groups must divide the experts and selected_groups cannot exceed groups")
        size = experts // self.groups
        if self.group_score_top > size or top > self.selected_groups * size:
            raise ValueError("group_score_top exceeds a group, or top exceeds the selectable experts")
        if self.renormalize and self.scoring == "softmax" and not self.score_before_selection:
            raise ValueError("Softmax after selection is already normalized; renormalize would only change rounding")
        values = (self.renormalize_epsilon, self.renormalize_floor, self.scaling)
        if not all(isinstance(v, (int, float)) and math.isfinite(v) for v in values) or min(values[:2]) < 0 or self.scaling <= 0:
            raise ValueError("Renormalization terms must be finite and non-negative, and scaling finite and positive")
        return self


def _sigmoid(value):
    if value >= 0:
        return 1.0 / (1.0 + math.exp(-value))
    exponent = math.exp(value)
    return exponent / (1.0 + exponent)


def _softmax(values):
    peak = max(values)
    exponents = [math.exp(v - peak) for v in values]
    total = math.fsum(exponents)
    return [e / total for e in exponents]


def route(logits, top, routing=Routing(), bias=None):
    """Select `top` experts for one token; return (expert IDs in selection order, weights)."""
    logits = [float(v) for v in logits]
    experts = len(logits)
    routing.validate(experts, top)
    if (bias is not None) != routing.correction_bias or (bias is not None and len(bias) != experts):
        raise ValueError("A correction bias of one value per expert is required exactly when correction_bias is set")
    if not all(math.isfinite(v) for v in logits + list(bias or ())):
        raise ValueError("Router logits and bias must be finite")
    if not routing.score_before_selection:
        scores = logits
    elif routing.scoring == "softmax":
        scores = _softmax(logits)
    else:
        scores = [_sigmoid(v) for v in logits]
    choice = [s + float(b) for s, b in zip(scores, bias)] if bias is not None else list(scores)
    candidates = range(experts)
    if routing.selected_groups < routing.groups:
        size = experts // routing.groups
        rank = [sum(sorted(choice[g * size:(g + 1) * size], reverse=True)[:routing.group_score_top])
                for g in range(routing.groups)]
        kept = set(sorted(range(routing.groups), key=lambda g: (-rank[g], g))[:routing.selected_groups])
        candidates = [e for e in candidates if e // size in kept]
    ids = sorted(candidates, key=lambda e: (-choice[e], e))[:top]
    if routing.score_before_selection:
        weights = [scores[e] for e in ids]
    elif routing.scoring == "softmax":
        weights = _softmax([logits[e] for e in ids])
    else:
        weights = [_sigmoid(logits[e]) for e in ids]
    if routing.renormalize:
        denominator = max(math.fsum(weights) + routing.renormalize_epsilon, routing.renormalize_floor)
        weights = [w / denominator for w in weights]
    return tuple(ids), tuple(w * routing.scaling for w in weights)


def routing_from_config(moe, *, correction_bias=False, defaults=None):
    """Map Core's MoeConfig (crit_core.model.config.MoeConfig) to a Routing.

    Each field in CONFIG_FIELDS comes from the config, else from `defaults` (the value
    the model's reference code assumes when config.json omits it, keyed by the config
    field name); a field neither declares raises, because references disagree on it.
    `defaults` must also give "score_before_selection", which no config states, and
    may give "renormalize_epsilon" and "renormalize_floor" (0 when absent).
    topk_method is needed only when topk_group < n_group. `correction_bias` says
    whether the checkpoint supplies a per-expert e_score_correction_bias tensor.
    """
    defaults = dict(defaults or {})
    unknown = set(defaults) - set(CONFIG_FIELDS) - {"topk_method", "score_before_selection",
                                                    "renormalize_epsilon", "renormalize_floor"}
    if unknown:
        raise ValueError(f"Unknown routing defaults: {sorted(unknown)}")

    def value(name, required=True):
        declared = getattr(moe, name, None)
        if declared is None or declared == "":
            declared = defaults.get(name)
        if declared is None and required:
            raise ValueError(f"The config omits {name} and no default was supplied")
        return declared

    if "score_before_selection" not in defaults:
        raise ValueError("defaults must state score_before_selection")
    groups, selected = value("n_group"), value("topk_group")
    method = value("topk_method", required=selected < groups)
    if method is not None and method not in GROUP_SCORE_TOP:
        raise ValueError(f"Unsupported topk_method {method!r}")
    if method == "noaux_tc" and not correction_bias:
        raise ValueError("topk_method noaux_tc selects with e_score_correction_bias, which is absent")
    top_scores = GROUP_SCORE_TOP.get(method, 1)
    if top_scores == 0 or selected >= groups:
        groups = selected = top_scores = 1
    return Routing(scoring=value("scoring_func"), score_before_selection=defaults["score_before_selection"],
                   correction_bias=correction_bias, groups=groups, selected_groups=selected,
                   group_score_top=top_scores, renormalize=bool(value("norm_topk_prob")),
                   renormalize_epsilon=float(defaults.get("renormalize_epsilon", 0.0)),
                   renormalize_floor=float(defaults.get("renormalize_floor", 0.0)),
                   scaling=float(value("routed_scaling_factor"))).validate(moe.experts, moe.experts_per_token)

