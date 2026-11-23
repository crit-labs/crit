"""Chain verification in the verifier slot: mtp-decode's verify (implementation.py) as the host form, and the same rule
for every row of a captured decode round on the device (device_verify.py, the verify:device extension); the round
driver counts acceptance by draft depth (crit_plugins.rounds)."""
from .mtp import implementation as mtp


class ChainVerifier:
    def __init__(self):
        self._columns = {}

    def accept(self, proposal, predictions, remaining, stops=()):
        tokens, parents = list(proposal["tokens"]), list(proposal["parents"])
        if parents != list(range(-1, len(tokens) - 1)):
            raise ValueError("chain-verifier verifies a chain of drafts: a tree drafter needs draft-trees")
        drafts = tokens[1:]
        cut = next((i for i, d in enumerate(drafts) if d < 0), len(drafts))   # a cut draft (PAD) matches nothing
        plan = mtp.verify(drafts[:cut], list(predictions[:cut + 1]), remaining, tuple(stops))
        return {"tokens": list(plan.tokens), "accepted": plan.accepted_drafts,
                "path": list(range(plan.accepted_drafts + 1))}

    # verify:device (crit_plugins.device.DEVICE_EXTENSIONS)
    def accept_device(self, view):
        """Each row's emitted tokens from view.predictions and view.drafts (view.n_emitted, view.n_matched,
        view.emitted), ending after a stop token (view.stops) and within the row's allowance (view.allowance), a held
        row (view.hold) emitting nothing; on the device and capture-safe."""
        from .mtp import device_verify
        device_verify.accept_chain(view, self._columns)


def create(context, options, parts):
    return ChainVerifier()
