"""greedy-head-drafter's draft:device drafts from a device draft head what its host form drafts from the same head."""
from crit_plugins.testing import PluginTestCase


def round_view(rows, width, mode, device, stops=1):
    """A round view as Core's round driver gives one (crit_plugins.rounds.RoundView over its buffers)."""
    from crit_plugins.rounds import SpeculativeRounds
    return SpeculativeRounds(None, rows, width, device, stops=stops).view(rows, width, mode)


class Head:
    """A toy device draft head: after token t at depth d its logits are table[t] + bias[d]."""

    def __init__(self, torch, device, vocab=40, depths=6):
        g = torch.Generator().manual_seed(4)
        self.table, self.bias = torch.randn(vocab, vocab, generator=g), torch.randn(depths, vocab, generator=g)
        self.on = (self.table.to(device), self.bias.to(device))
        self.caught = 0

    def catch_up(self, view):
        self.caught += 1

    def logits(self, depth, tokens, view, parents=None):
        return self.on[0][tokens] + self.on[1][depth]

    def host(self, pending):
        return lambda paths, rows=None: [(self.table[p[-1] if p else pending] + self.bias[len(p)]).tolist()
                                         for p in paths]


class DeviceDraftTests(PluginTestCase):
    def test_it_provides_draft_device(self):
        self.assertImplements(self.assemble(facts=["draft:head", "draft:head-device"]))

    def test_device_drafts_are_the_host_drafts(self):
        torch = __import__("torch") if self.device() else None
        drafter = self.assemble(facts=["draft:head", "draft:head-device"])["drafter"]
        head = Head(torch, self.device())
        pending = [3, 17, 39]
        for width in (0, 1, 5):
            view = round_view(3, width, "greedy", self.device())
            view.pending.copy_(torch.tensor(pending))
            drafter.draft(view, head)
            for r, token in enumerate(pending):
                drafter.start(r, [token])
                self.assertEqual(view.drafts[r, :width].tolist(), drafter.propose(r, width, head.host(token))["tokens"][1:])
        self.assertEqual(head.caught, 3)
