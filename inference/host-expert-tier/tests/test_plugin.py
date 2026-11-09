from crit_plugins.slots import CASES
from crit_plugins.testing import PluginTestCase

USE = {"expert_source": "expert-tensors"}


class HostExpertTierTests(PluginTestCase):
    def test_it_has_the_experts_interface(self):
        self.assertImplements(self.assemble(USE))

    def test_it_is_the_reference_within_fp32_rounding(self):
        for max_tokens in (1, 2, 8):
            with self.subTest(max_tokens=max_tokens):
                self.assertMatchesReference(USE, {"max_tokens": max_tokens}, atol=2e-6, rtol=2e-6)

    def test_the_chunks_hold_at_most_max_tokens_routes_of_one_expert(self):
        parts = self.assemble(USE, {"max_tokens": 1})
        CASES["experts"](parts)
        self.assertEqual(parts["experts"].chunks, 9 + 5 + 5)
        parts = self.assemble(USE, {"max_tokens": 8})
        CASES["experts"](parts)
        self.assertLess(parts["experts"].chunks, 19)
