"""A test stand-in for the sampler slot's round form (rounds:sampler), which token-sampling also provides: the
package's token-sampling plugin in every respect, with greedy device predictions for a round (each column's argmax,
the lower id among equals, as token-sampling's device form picks a greedy row's token) and the staged rules of each
round slot, greedy only."""
from crit_plugins.compose import package_plugin


class RoundSampler:
    def __init__(self, inner):
        self.inner, self.staged = inner, {}

    def __getattr__(self, name):
        return getattr(self.inner, name)

    def predictions_device(self, view, logits, draft_laws=None):
        if draft_laws is not None or view.mode == "sampled":
            raise ValueError("this stand-in predicts greedy rounds only")
        return logits.argmax(-1)

    def stage(self, slot, params):
        if float(params.get("temperature", 0.0)) > 0:
            raise ValueError("this stand-in stages greedy rows only")
        self.staged[slot] = dict(params)


def create(context, options, parts):
    return RoundSampler(package_plugin(context, options["inner"], parts))
