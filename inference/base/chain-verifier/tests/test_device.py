"""chain-verifier's verify:device accepts on the device what its host form accepts, for every row of a round at once,
a held row emitting nothing."""
from crit_plugins.testing import PluginTestCase


def round_view(rows, width, mode, device, stops=1):
    """A round view as Core's round driver gives one (crit_plugins.rounds.RoundView over its buffers)."""
    from crit_plugins.rounds import SpeculativeRounds
    return SpeculativeRounds(None, rows, width, device, stops=stops).view(rows, width, mode)


CASES = (([4, 5, 6, 7, 8], 10, ()), ([3, 5, 6, 7, 8], 10, ()), ([4, 5, 2, 7, 8], 10, ()), ([4, 5, 6, 7, 8], 2, ()),
         ([4, 5, 6, 7, 8], 10, (5,)), ([4, 5, 6, 7, 8], 0, ()))


class DeviceAcceptTests(PluginTestCase):
    def test_it_provides_verify_device(self):
        self.assertImplements()

    def test_a_round_accepts_what_the_host_form_accepts(self):
        torch = __import__("torch") if self.device() else None
        verifier = self.assemble()["verifier"]
        rows, width = len(CASES), 4
        view = round_view(rows, width, "greedy", self.device(), stops=2)
        long = dict(dtype=torch.int64, device=self.device())
        view.drafts[:, :width] = torch.tensor([[4, 5, 6, 7]] * rows, **long)
        view.predictions[:, :width + 1] = torch.tensor([c[0] for c in CASES], **long)
        view.allowance.copy_(torch.tensor([c[1] for c in CASES], **long))
        view.stops.copy_(torch.tensor([list(c[2]) + [-1] * (2 - len(c[2])) for c in CASES], **long))
        view.hold[0] = 1
        verifier.accept_device(view)
        self.assertEqual(int(view.n_emitted[0]), 0)
        view.hold[0] = 0
        verifier.accept_device(view)
        for r, (predictions, remaining, stops) in enumerate(CASES):
            host = verifier.accept({"tokens": [9, 4, 5, 6, 7], "parents": [-1, 0, 1, 2, 3]}, predictions, remaining, stops)
            count = int(view.n_emitted[r])
            self.assertEqual(view.emitted[r, :count].tolist(), host["tokens"])
            self.assertEqual(int(view.n_matched[r]), min(host["accepted"], max(count - 1, 0)))
