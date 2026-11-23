"""dense-layer-offload in the placement slot: what every placement does, whole layers kept together, and the planner's
plan with measured rates."""
from crit_plugins.slots import PLACES
from crit_plugins.testing import PluginTestCase


class PlacementChecks:
    """What every placement must do (as device-placement's tests check)."""

    def placement(self, **options):
        return self.assemble(options=options)["placement"]

    def test_it_has_the_placement_interface(self):
        self.assertImplements()

    def test_every_tensor_gets_one_place(self):
        tensors = {"embed": 600, "layers.0": 300, "head": 600, "kv": 1000}
        plan = self.placement().plan(dict(tensors), {"device": 10_000, "host": 10_000})
        self.assertEqual(set(plan), set(tensors))
        self.assertTrue(all(place in PLACES for place in plan.values()))

    def test_what_fits_sits_on_the_device(self):
        tensors = {"embed": 600, "layers.0": 300, "head": 600}
        self.assertEqual(self.placement().plan(dict(tensors), {"device": 1500, "host": 0}),
                         {name: "device" for name in tensors})

TENSORS = {"model.embed_tokens.weight": 100, "lm_head.weight": 100, "model.norm.weight": 1,
           "model.layers.0.self_attn.q_proj.weight": 30, "model.layers.0.mlp.down_proj.weight": 50,
           "model.layers.1.self_attn.q_proj.weight": 30, "model.layers.1.mlp.down_proj.weight": 50,
           "model.layers.2.self_attn.q_proj.weight": 30, "model.layers.2.mlp.down_proj.weight": 50}


class LayerOffloadTests(PlacementChecks, PluginTestCase):
    def test_it_equals_device_placement_where_everything_fits(self):
        self.assertMatchesReference()

    def test_the_layers_that_do_not_fit_go_to_host_memory_whole(self):
        plan = self.placement().plan(dict(TENSORS), {"device": 201 + 160, "host": 1 << 30})
        self.assertEqual({n for n, p in plan.items() if p == "host"},
                         {"model.layers.2.self_attn.q_proj.weight", "model.layers.2.mlp.down_proj.weight"})
        self.assertEqual(plan["lm_head.weight"], "device")

    def test_measured_rates_choose_the_planners_fastest_plan(self):
        # 160 bytes of device room for three 80-byte layers. Uploading at 50 GB/s, streaming all three (two slots of
        # 80 take the room) costs 3 x 1.6 ns, running the third on the host at 20 GB/s 2 x 0.08 + 4 ns: the host wins.
        # At 500 GB/s streaming costs 3 x 0.16 ns and wins.
        for upload, expected in ((50.0, {0: "device", 1: "device", 2: "host"}),
                                 (500.0, {0: "stream", 1: "stream", 2: "stream"})):
            rates = {"device_gbs": 1000.0, "upload_gbs": upload, "host_gbs": 20.0}
            places, _ = self.placement(**rates).layers(dict(TENSORS), {"device": 201 + 160})
            self.assertEqual(places, expected, upload)

    def test_what_is_outside_the_layers_must_fit_the_device(self):
        with self.assertRaisesRegex(MemoryError, "outside the layers need"):
            self.placement().plan(dict(TENSORS), {"device": 150, "host": 1 << 30})
