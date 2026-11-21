"""Measured decisions (crit_plugins.decisions): a recorded choice is replayed, another key measures again."""
import itertools
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "core/python"))

from crit_plugins import decisions, policy  # noqa: E402


class Decisions(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.addCleanup(decisions.use_record, None)
        self.path = Path(self.tmp.name) / "decisions.json"
        decisions.use_record(None)
        decisions.use_record(None)          # the process's second load: what it measures is a warm load's
        self.calls = 0

    def settle(self, cpu, key=None):
        def measure():
            self.calls += 1
            return {"cpu": cpu, "link": 100.0}
        return decisions.settle("offload", "misses", key=key or {"device": "a", "hidden": 8}, measure=measure,
                                decide=lambda found: "cpu" if found["cpu"] < 0.8 * found["link"] else "link")

    def test_a_near_tie_measured_twice_can_part_and_a_record_holds_it(self):
        self.assertEqual(self.settle(79.0), "cpu")
        first = decisions.report()
        self.assertEqual((first["measured"], first["decisions"][0]["source"]), (1, "measured"))
        decisions.write(self.path)
        decisions.use_record(None)                      # another process, no record: the tie falls the other way
        self.assertEqual(self.settle(81.0), "link")
        self.assertNotEqual(decisions.report()["digest"], first["digest"])
        decisions.use_record(self.path)                 # another process with the record
        self.assertEqual(self.settle(81.0), "cpu")
        again = decisions.report()
        self.assertEqual((again["measured"], again["decisions"][0]["source"], again["digest"]),
                         (0, "recorded", first["digest"]))
        self.assertEqual(again["decisions"][0]["measurement"], {"cpu": 79.0, "link": 100.0})
        self.assertEqual(self.calls, 2)                 # the recorded load timed nothing

    def test_a_record_for_another_key_is_measured_again_and_says_so(self):
        self.settle(50.0)
        decisions.write(self.path)
        decisions.use_record(self.path)
        self.assertEqual(self.settle(90.0, key={"device": "b", "hidden": 8}), "link")
        row = decisions.settled("offload", "misses")
        self.assertEqual(row["source"], "measured")
        self.assertIn("another key", row["stale"])

    def test_a_first_loads_measurement_is_not_replayed_and_a_warm_loads_is(self):
        with mock.patch.object(policy, "_LOADS", itertools.count(1)):
            decisions.use_record(None)                  # a process's first load: stores built, pages read cold
            self.settle(50.0)
            self.assertEqual(decisions.settled("offload", "misses")["conditions"], {"load": 1, "first_load": True})
            decisions.write(self.path)
            decisions.use_record(self.path)             # its second load reads the record and measures again
            self.assertEqual(self.settle(90.0), "link")
            row = decisions.settled("offload", "misses")
            self.assertEqual((row["source"], row["conditions"]), ("measured", {"load": 2, "first_load": False}))
            self.assertIn("first load", row["stale"])
            decisions.write(self.path)
            decisions.use_record(self.path)             # the warm load's record answers
            self.assertEqual(self.settle(10.0), "link")
            self.assertEqual(decisions.settled("offload", "misses")["source"], "recorded")
        self.assertEqual(self.calls, 2)

    def test_a_record_that_does_not_say_how_it_was_measured_is_measured_again(self):
        self.settle(50.0)
        decisions.write(self.path)
        record = json.loads(self.path.read_text(encoding="utf-8"))
        for row in record["decisions"]:
            del row["conditions"]                       # a record written before decisions carried conditions
        self.path.write_text(json.dumps(record), encoding="utf-8")
        decisions.use_record(self.path)
        self.assertEqual(self.settle(90.0), "link")
        self.assertIn("does not say how", decisions.settled("offload", "misses")["stale"])

    def test_a_process_settles_a_decision_once(self):
        self.assertEqual(self.settle(50.0), "cpu")
        self.assertEqual(self.settle(95.0), "cpu")
        self.assertEqual(self.calls, 1)

    def test_tuples_and_lists_are_one_key(self):
        decisions.settle("p", "n", key={"shape": (2, 3)}, measure=dict, decide=lambda _: [1, 2])
        decisions.write(self.path)
        decisions.use_record(self.path)
        choice = decisions.settle("p", "n", key={"shape": [2, 3]}, measure=lambda: self.fail("measured"),
                                  decide=lambda _: None)
        self.assertEqual(choice, [1, 2])

    def test_what_is_not_a_record_or_not_json_is_refused(self):
        self.path.write_text(json.dumps({"version": 0, "decisions": []}), encoding="utf-8")
        decisions.use_record(self.path)
        with self.assertRaises(decisions.DecisionError):
            self.settle(50.0)
        decisions.use_record(None)
        with self.assertRaises(decisions.DecisionError):
            decisions.settle("p", "n", key={"a": 1}, measure=lambda: {"x": float("nan")}, decide=lambda _: 1)
        with self.assertRaises(decisions.DecisionError):
            decisions.settle("p", "n", key=["a"], measure=dict, decide=lambda _: 1)

    def test_a_package_names_its_record(self):
        from crit_core.engine import package_decisions
        from crit_plugins import PluginError
        package = Path(self.tmp.name)
        self.assertIsNone(package_decisions(package, {}))
        with self.assertRaises(PluginError):
            package_decisions(package, {"decisions": "decisions.json"})     # the file is not there
        self.settle(50.0)
        decisions.write(self.path)
        self.assertEqual(package_decisions(package, {"decisions": "decisions.json"}), self.path.resolve())
        self.assertEqual(decisions.record_file(), self.path.resolve())
        with self.assertRaises(PluginError):
            package_decisions(package, {"decisions": "../decisions.json"})  # outside the package


if __name__ == "__main__":
    unittest.main()
