from crit_plugins.testing import PluginTestCase

# The toy engine's plugins beside this one, which its slot interface's methods may call.
USE = {'placement': 'toy-placement', 'projections': 'toy-projections', 'attention': 'toy-attention', 'kv_cache': 'toy-kv', 'sampler': 'toy-sampler'}


class ToyTests(PluginTestCase):
    def test_it_implements_its_slot(self):
        own = self.registry()[self.plugin_id()]["fills"]
        self.assertImplements(self.assemble(use={slot: plugin for slot, plugin in USE.items() if slot not in own}))
