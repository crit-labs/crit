from crit_plugins.slots import CASES
from crit_plugins.testing import PluginTestCase

GEOMETRY = {"keys": 1, "values": 1, "epsilon": 0.0, "query_scale": 1.0, "normalize_keys": False,
            "normalize_queries": False}


class SequentialRecurrenceTests(PluginTestCase):
    def test_it_has_the_recurrence_interface(self):
        self.assertImplements()

    def test_one_dimension_is_the_delta_rule(self):
        recurrence = self.assemble()["recurrence"]
        step = {"query": [1.0], "key": [1.0], "value": [2.0], "log_decay": 0.0, "beta": 0.5}
        outputs, state = recurrence.prefill((0.0,), [step, step], GEOMETRY)
        self.assertClose(state, [1.5], atol=1e-12)       # 0 -> 1.0 -> 1.5
        self.assertClose([o[0] for o in outputs], [1.0, 1.5], atol=1e-12)

    def test_a_split_prefill_continues_from_its_state(self):
        out = CASES["recurrence"](self.assemble())
        whole, state = self.assemble()["recurrence"].prefill(
            (0.0,) * 6, *self._steps_and_geometry())
        self.assertClose(out["outputs"], whole, atol=1e-12)
        self.assertClose(out["state"], list(state), atol=1e-12)

    def test_a_state_of_the_wrong_size_is_refused(self):
        with self.assertRaisesRegex(ValueError, "keys"):
            self.assemble()["recurrence"].prefill((0.0, 0.0), [], GEOMETRY)

    def _steps_and_geometry(self):
        captured = {}

        class Capture:
            def prefill(self, state, steps, geometry):
                captured.setdefault("steps", []).extend(steps)
                captured["geometry"] = geometry
                return [], state

        CASES["recurrence"]({"recurrence": Capture()})
        return captured["steps"], captured["geometry"]
