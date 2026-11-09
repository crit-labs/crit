from crit_plugins.slots import PLACES
from crit_plugins.testing import PluginTestCase


class PlacementChecks:
    """What every placement must do."""

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


class DevicePlacementTests(PlacementChecks, PluginTestCase):
    def test_what_does_not_fit_is_refused(self):
        with self.assertRaisesRegex(MemoryError, "need 0.00 GiB on the device"):
            self.placement().plan({"a": 600, "b": 600}, {"device": 1000, "host": 1 << 40})

    def test_the_reserve_stays_out_of_the_plan(self):
        tensors = {"a": 600 << 20, "b": 300 << 20}
        self.assertEqual(set(self.placement(reserve_mib=100).plan(dict(tensors), {"device": 1000 << 20}).values()),
                         {"device"})
        with self.assertRaisesRegex(MemoryError, "0.10 GiB kept free"):
            self.placement(reserve_mib=101).plan(dict(tensors), {"device": 1000 << 20})


class OffDeviceTests(PluginTestCase):
    """off_device: routed experts' bundles and lookup tables off the device, host memory first, then files."""

    def placement(self, **options):
        return self.assemble(options={"off_device": "experts.,tables.", **options})["placement"]

    def test_off_device_groups_go_to_host_memory_in_preference_order_then_files(self):
        tensors = {"embed": 100, "layers.0": 200, "tables.0": 300, "experts.0": 400, "experts.1": 400, "kv": 300}
        plan = self.placement().plan(dict(tensors), {"device": 600, "host": 1000})
        self.assertEqual(list(plan), list(tensors))
        self.assertEqual(plan, {"embed": "device", "layers.0": "device", "tables.0": "files", "experts.0": "host",
                                "experts.1": "host", "kv": "device"})
        self.assertEqual(self.placement().plan(dict(tensors), {"device": 600, "host": 2000})["tables.0"], "host")
        # a prefix's groups stay together: experts that do not all fit host memory all go to files
        plan = self.placement().plan(dict(tensors), {"device": 600, "host": 700})
        self.assertEqual((plan["experts.0"], plan["experts.1"], plan["tables.0"]), ("files", "files", "host"))
        with self.assertRaisesRegex(MemoryError, "files are off"):
            self.placement(files=False).plan(dict(tensors), {"device": 600, "host": 1000})
        with self.assertRaisesRegex(MemoryError, "no host bytes"):
            self.placement().plan(dict(tensors), {"device": 600})

    def test_the_device_groups_alone_must_fit(self):
        tensors = {"layers.0": 500, "experts.0": 10_000, "kv": 200}
        self.assertEqual(self.placement().plan(dict(tensors), {"device": 700, "host": 20_000})["experts.0"], "host")
        with self.assertRaisesRegex(MemoryError, "need 0.00 GiB on the device"):
            self.placement().plan(dict(tensors), {"device": 699, "host": 20_000})

    def test_slots_fill_what_the_device_groups_leave(self):
        tensors = {"layers.0": 500, "experts.0": 10_000, "kv": 200, "expert_slots": 999}
        self.assertEqual(self.placement().slots(dict(tensors), {"device": 1700, "host": 0}, 100), 9)   # 1 scratch
        self.assertEqual(self.placement(reserve_mib=0).slots(dict(tensors), {"device": 650, "host": 0}, 100), 0)
        plan = self.placement().plan({**tensors, "expert_slots": 1000}, {"device": 1700, "host": 20_000})
        self.assertEqual(plan["expert_slots"], "device")
