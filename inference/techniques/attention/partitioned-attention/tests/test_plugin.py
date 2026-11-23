from crit_plugins.testing import PluginTestCase

USE = {"kv_cache": "attention-state"}


class PartitionedAttentionTests(PluginTestCase):
    def test_it_has_the_attention_interface(self):
        self.assertImplements(self.assemble(USE))

    def test_merged_partials_equal_attention_over_the_whole_sequence(self):
        for world in (1, 2, 3, 5):
            self.assertMatchesReference(USE, {"world": world}, atol=1e-12, rtol=1e-10)
