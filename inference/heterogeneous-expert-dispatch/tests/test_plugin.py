from crit_plugins.slots import CASES
from crit_plugins.testing import PluginTestCase

USE = {"expert_source": "expert-tensors", "expert_residency": "device-expert-cache"}


class HeterogeneousExpertDispatchTests(PluginTestCase):
    def test_it_has_the_experts_interface(self):
        self.assertImplements(self.assemble(USE))

    def test_it_is_bitwise_the_reference_however_the_routes_split(self):
        for residency, policy in (("device-expert-cache", {"policy": "clock"}), ("expert-residency", {}),
                                  ("device-expert-cache", {"policy": "ranked"})):
            for slots, device_slots in ((1, 0), (2, 1), (3, 4), (64, 64)):
                with self.subTest(residency=residency, **policy, slots=slots, device_slots=device_slots):
                    use = dict(USE, expert_residency=residency)
                    self.assertMatchesReference(use, {"device_slots": device_slots}, atol=0.0, rtol=0.0,
                                                use_options={residency: {**policy, "slots": slots}})

    def test_resident_experts_run_on_the_device_up_to_its_slots(self):
        for slots, device_slots, sides in ((64, 64, "device"), (64, 0, "host"), (2, 64, "both")):
            with self.subTest(slots=slots, device_slots=device_slots):
                parts = self.assemble(USE, {"device_slots": device_slots},
                                      use_options={"device-expert-cache": {"slots": slots}})
                CASES["experts"](parts)
                routes = parts["experts"].routes
                self.assertEqual(routes["device"] + routes["host"], 9 + 5 + 5)
                if sides == "device":
                    self.assertEqual(routes["host"], 0)
                elif sides == "host":
                    self.assertEqual(routes["device"], 0)
                else:
                    self.assertGreater(routes["device"], 0)
                    self.assertGreater(routes["host"], 0)
