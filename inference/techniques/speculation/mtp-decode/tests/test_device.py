"""mtp-decode's draft:device: copies from each round slot's token history on the device equal the host index's, round
after round, as the rounds' emitted tokens enter the histories on the device."""
import random

from crit_plugins.testing import PluginTestCase


def round_view(rows, width, mode, device, stops=1):
    """A round view as Core's round driver gives one (crit_plugins.rounds.RoundView over its buffers)."""
    from crit_plugins.rounds import SpeculativeRounds
    return SpeculativeRounds(None, rows, width, device, stops=stops).view(rows, width, mode)


class DeviceCopyTests(PluginTestCase):
    def test_it_provides_draft_device(self):
        self.assertImplements()

    def test_device_copies_follow_the_host_index(self):
        torch = __import__("torch") if self.device() else None
        long = dict(dtype=torch.int64, device=self.device())
        for window, history in ((2, 4), (3, 6)):
            drafter = self.assemble(options={"window": window, "history_window": history, "capacity": 256})["drafter"]
            rows, width, vocab = 3, 4, 5
            drafter.reserve(rows, width, vocab, self.device())
            rng = random.Random(window)
            view = round_view(rows, width, "greedy", self.device())
            for r in range(rows):
                prompt = [rng.randrange(vocab) for _ in range(rng.randrange(3, 40))]
                drafter.start(r, prompt)
                drafter.seed(r, prompt)
                view.frontier[r] = len(prompt) - 1
            for _ in range(25):
                drafter.draft(view)
                emitted = []
                for r in range(rows):
                    host = drafter.propose(r, width)["tokens"][1:]
                    got = view.drafts[r].tolist()
                    self.assertEqual(got, host + [-1] * (width - len(host)))
                    emitted.append(host[:rng.randrange(0, len(host) + 1)] + [rng.randrange(vocab)])
                view.emitted.fill_(-1)
                for r, tokens in enumerate(emitted):
                    view.emitted[r, :len(tokens)] = torch.tensor(tokens)
                    drafter.observe(r, tokens)
                view.n_emitted.copy_(torch.tensor([len(t) for t in emitted], **long))
                view.frontier += view.n_emitted
                drafter.commit_round(view)
