"""A test stand-in for the router slot's round form (rounds:router), which topk-router also provides: the
package's topk-router in every respect, its device selection the same rule over the router logits with nothing read
back to the host (topk-router first checks the logits finite on the host, which a captured round cannot do)."""
from pathlib import Path

from crit_plugins import Registry
from crit_plugins.assemble import load_module
from crit_plugins.manifest import options_for


def library_plugin(context, plugin_id, parts, options=None):
    """The package's plugin `plugin_id` (copied in beside this stand-in), created as Core's loader creates a plugin."""
    manifest = Registry.discover(Path(context.package) / "plugins")[plugin_id]
    factory = getattr(load_module(manifest), manifest["entry"][1])
    return factory(context, options_for(manifest, options or {}, f"plugins.options.{plugin_id}"), parts)


class RoundRouter:
    def __init__(self, inner):
        self.inner = inner

    def __getattr__(self, name):
        return getattr(self.inner, name)

    def select(self, logits, top, bias=None):
        """topk-router's device form (ids [tokens][top] int64, weights [tokens][top] FP32), capture-safe: its rule
        without groups or a correction bias, which this stand-in does not take."""
        import torch
        rule = self.inner.routing
        if bias is not None or rule.selected_groups < rule.groups:
            raise ValueError("this stand-in routes without a correction bias or expert groups")
        rule.validate(logits.shape[1], top)
        x = logits.float()
        if not rule.score_before_selection:
            scores = x
        else:
            scores = torch.softmax(x, -1) if rule.scoring == "softmax" else torch.sigmoid(x)
        ids = torch.sort(scores, dim=-1, descending=True, stable=True).indices[:, :top]
        if rule.score_before_selection:
            weights = scores.gather(1, ids)
        elif rule.scoring == "softmax":
            weights = torch.softmax(x.gather(1, ids), -1)
        else:
            weights = torch.sigmoid(x.gather(1, ids))
        if rule.renormalize:
            weights = weights / torch.clamp(weights.sum(-1, keepdim=True) + rule.renormalize_epsilon,
                                            min=rule.renormalize_floor)
        return ids, weights * rule.scaling


def create(context, options, parts):
    inner = {k: options[k] for k in ("score_before_selection", "renormalize")}
    return RoundRouter(library_plugin(context, options["inner"], parts, inner))
