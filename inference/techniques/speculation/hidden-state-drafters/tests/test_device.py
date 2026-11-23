"""hidden-state-drafters' draft:device maps draft-vocabulary choices by d2t and learns and applies the confidence cut
on the device, round for round as its host form does."""
from crit_plugins.testing import PluginTestCase


def round_view(rows, width, mode, device, stops=1):
    """A round view as Core's round driver gives one (crit_plugins.rounds.RoundView over its buffers)."""
    from crit_plugins.rounds import SpeculativeRounds
    return SpeculativeRounds(None, rows, width, device, stops=stops).view(rows, width, mode)


FACTS = ["draft:head", "draft:head-device", "taps:hidden-states"]
SURE, UNSURE = [5.0, 0.0, 0.0, 0.0], [0.1, 0.0, 0.2, 0.0]


class Head:
    """The host test's head on the device: a 4-token draft vocabulary mapped to 10..13, sure of its first draft."""
    d2t = [10, 10, 10, 10]

    def __init__(self, torch, device):
        self.torch, self.on = torch, device

    def catch_up(self, view):
        pass

    def logits(self, depth, tokens, view, parents=None):
        return self.torch.tensor(SURE if depth == 0 else UNSURE, device=self.on).expand(tokens.shape[0], 4).contiguous()

    def __call__(self, paths, rows=None):
        return [SURE if not p else UNSURE for p in paths]


class DeviceDraftTests(PluginTestCase):
    def test_it_provides_draft_device(self):
        self.assertImplements(self.assemble(facts=FACTS))

    def test_the_device_cut_learns_as_the_host_cut(self):
        torch = __import__("torch") if self.device() else None
        drafter = self.assemble(options={"cut_target": 0.8}, facts=FACTS)["drafter"]
        rows, width = 2, 3
        drafter.reserve(rows, width, 14, self.device())
        head = Head(torch, self.device())
        view = round_view(rows, width, "greedy", self.device())
        view.allowance.fill_(9)
        view.n_emitted.fill_(2)
        view.emitted[:, :2] = torch.tensor([10, 11])
        for r in range(rows):
            drafter.start(r, [4])
        for round_ in range(80):
            drafter.draft(view, head)
            self.assertEqual(view.drafts.tolist(), [drafter.propose(r, width, head)["tokens"][1:] for r in range(rows)],
                             round_)
            drafter.commit_round(view)
            drafter.commit_round(view)   # a truncation's second commit: the round counts once
            for r in range(rows):
                drafter.observe(r, [10, 11])
        self.assertEqual(view.drafts[0].tolist(), [10, -1, -1])
