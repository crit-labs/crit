from crit_plugins.testing import PluginTestCase


class RecurrentStateRecoveryTests(PluginTestCase):
    def test_it_has_the_recurrence_interface(self):
        self.assertImplements()

    def test_it_equals_the_token_by_token_recurrence_for_every_chunk_size(self):
        for chunk_size in (1, 2, 3, 4, 7, 64):
            with self.subTest(chunk_size=chunk_size):
                self.assertMatchesReference(options={"chunk_size": chunk_size}, atol=1e-10, rtol=1e-9)

    def test_prompt_rows_auto_follows_the_level_the_engine_asks_of_its_prompt_rows(self):
        from crit_plugins import kernels
        self.addCleanup(kernels.use_exactness, None)
        for asked, prompt in ((None, "two_pass"), ("repeatable", "two_pass"), ("row-exact", "scan"), ("exact", "scan")):
            kernels.use_exactness(asked and {"decode": "row-exact", "prompt": asked})
            plugin = self.assemble()["recurrence"]
            self.assertEqual(plugin.prompt, prompt, asked)
            self.assertEqual(plugin.exactness(), {})   # no pool made: it states no level yet
        # An option set to one way is that way, whatever is asked.
        self.assertEqual(self.assemble(options={"prompt_rows": "chunked"})["recurrence"].prompt, "two_pass")
        kernels.use_exactness("repeatable")
        self.assertEqual(self.assemble(options={"prompt_rows": "scan"})["recurrence"].prompt, "scan")

    def test_no_steps_keep_the_state(self):
        geometry = {"keys": 2, "values": 1}
        self.assertEqual(self.assemble()["recurrence"].prefill((0.5, 0.25), [], geometry), ([], (0.5, 0.25)))
