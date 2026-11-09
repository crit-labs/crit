from crit_plugins.testing import PluginTestCase


class RecurrentStateRecoveryTests(PluginTestCase):
    def test_it_has_the_recurrence_interface(self):
        self.assertImplements()

    def test_it_equals_the_token_by_token_recurrence_for_every_chunk_size(self):
        for chunk_size in (1, 2, 3, 4, 7, 64):
            with self.subTest(chunk_size=chunk_size):
                self.assertMatchesReference(options={"chunk_size": chunk_size}, atol=1e-10, rtol=1e-9)

    def test_no_steps_keep_the_state(self):
        geometry = {"keys": 2, "values": 1}
        self.assertEqual(self.assemble()["recurrence"].prefill((0.5, 0.25), [], geometry), ([], (0.5, 0.25)))
