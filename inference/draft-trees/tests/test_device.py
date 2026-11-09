"""draft-trees' draft:device and verify:device: the device tree of every row is the host form's tree from the same
head, and the device walk emits what the host walk emits."""
import random

from crit_plugins.testing import PluginTestCase


def round_view(rows, width, mode, device, stops=1):
    """A round view as Core's round driver gives one (crit_plugins.rounds.RoundView over its buffers)."""
    from crit_plugins.rounds import SpeculativeRounds
    return SpeculativeRounds(None, rows, width, device, stops=stops).view(rows, width, mode)


FACTS = ["draft:head", "draft:head-device"]


class Head:
    """A toy device draft head: after token t at depth d its logits are table[t] + bias[d], whatever the path."""

    def __init__(self, torch, device, vocab=20, depths=4):
        g = torch.Generator().manual_seed(6)
        self.table, self.bias = torch.randn(vocab, vocab, generator=g), torch.randn(depths, vocab, generator=g)
        self.on = (self.table.to(device), self.bias.to(device))

    def catch_up(self, view):
        pass

    def logits(self, depth, tokens, view, parents=None):
        return self.on[0][tokens] + self.on[1][depth]

    def host(self, pending):
        return lambda paths, rows=None: [(self.table[p[-1] if p else pending] + self.bias[len(p)]).tolist()
                                         for p in paths]


class DeviceTreeTests(PluginTestCase):
    def view(self, torch, rows, width):
        view = round_view(rows, width, "greedy", self.device())
        view.allowance.fill_(9)
        return view

    def test_it_provides_draft_and_verify_device(self):
        self.assertImplements(self.assemble(facts=FACTS))

    def test_device_trees_and_walks_are_the_host_ones(self):
        torch = __import__("torch") if self.device() else None
        trees = self.assemble(options={"depth": 3, "branch": 3, "beam": 2}, facts=FACTS)["drafter"]
        trees.reserve(3, 8, 20, self.device())
        head = Head(torch, self.device())
        rng = random.Random(1)
        for width in (1, 4, 8):
            view = self.view(torch, 3, width)
            pending = [rng.randrange(20) for _ in range(3)]
            view.pending.copy_(torch.tensor(pending))
            trees.draft(view, head)
            predictions = []
            for r in range(3):
                trees.start(r, [pending[r]])
                host = trees.propose(r, width, head.host(pending[r]))
                count = len(host["tokens"]) - 1
                self.assertEqual(view.drafts[r, :count].tolist(), host["tokens"][1:])
                self.assertEqual(view.tree_parents[r, :count].tolist(), host["parents"][1:])
                # The target agrees with the first child of each node it reaches, then differs.
                p = [rng.randrange(20) for _ in range(width + 1)]
                if count:
                    p[0] = host["tokens"][1]
                predictions.append((p, host))
            view.predictions.copy_(torch.tensor([p for p, _ in predictions]))
            trees.accept_device(view)
            for r, (p, host) in enumerate(predictions):
                walk = trees.accept(host, p[:len(host["tokens"])], 9)
                count = int(view.n_emitted[r])
                self.assertEqual(view.emitted[r, :count].tolist(), walk["tokens"])
                self.assertEqual(view.predictions[r, :count].tolist(), walk["tokens"])   # in path order
                self.assertEqual(view.tree_path[r, :count].tolist(), walk["path"][:count])
