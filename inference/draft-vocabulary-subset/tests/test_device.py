"""draft-vocabulary-subset's draft:device drafts the best of its subset rows from a device head (head.rows), counts the
emitted tokens on the device and chooses its subset again there as its host form does, a truncation's second commit
counting the round once."""
from crit_plugins.testing import PluginTestCase


def round_view(rows, width, mode, device, stops=1):
    """A round view as Core's round driver gives one (crit_plugins.rounds.RoundView over its buffers)."""
    from crit_plugins.rounds import SpeculativeRounds
    return SpeculativeRounds(None, rows, width, device, stops=stops).view(rows, width, mode)


FACTS = ["draft:head", "draft:head-device", "draft:head-rows"]


class Head:
    """After n tokens the best of 12 is (n * 5 + 2) % 12 (the host test's head), on the device over head.rows."""

    def __init__(self, torch, device):
        self.torch, self.on, self.rows = torch, device, None

    def catch_up(self, view):
        pass

    def logits(self, depth, tokens, view, parents=None):
        row = self.torch.zeros(12, device=self.on)
        row[(depth * 5 + 2) % 12] = 1.0
        found = row.expand(tokens.shape[0], 12)
        return found if self.rows is None else found[:, self.rows]

    def __call__(self, paths, rows=None):
        full = [[1.0 if t == (len(p) * 5 + 2) % 12 else 0.0 for t in range(12)] for p in paths]
        return full if rows is None else [[r[t] for t in rows] for r in full]


class DeviceSubsetTests(PluginTestCase):
    def test_it_provides_draft_device(self):
        self.assertImplements(self.assemble(facts=FACTS))

    def test_device_subset_drafts_follow_the_host_form(self):
        torch = __import__("torch") if self.device() else None
        drafter = self.assemble(options={"size": 4, "refresh": 2}, facts=FACTS)["drafter"]
        width = 3
        drafter.reserve(1, width, 12, self.device())
        head = Head(torch, self.device())
        view = round_view(1, width, "greedy", self.device())
        view.allowance.fill_(1)
        drafter.start(0, [3])
        for emitted in ([7, 7], [7], [1, 2, 3], [7, 0], [11]):
            drafter.draft(view, head)
            self.assertEqual(view.drafts[0].tolist(), drafter.propose(0, width, head)["tokens"][1:])
            for tokens in (emitted + [7] * (width + 1 - len(emitted)), emitted):   # committed, then truncated
                view.emitted.fill_(-1)
                view.emitted[0, :len(tokens)] = torch.tensor(tokens)
                view.n_emitted.fill_(len(tokens))
                drafter.commit_round(view)
            drafter.observe(0, emitted)
        self.assertIn(7, view.drafts[0].tolist())
