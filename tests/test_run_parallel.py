"""tests/run_parallel.py's shards: every unit in exactly one shard, the shards' recorded times close, and the recorded
times folded back into durations.json."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import run_parallel


class ShardTests(unittest.TestCase):
    def test_every_unit_runs_in_exactly_one_shard(self):
        chosen = [(f"test_{number}", float(number % 7 + 1)) for number in range(40)]
        shards = [run_parallel.shard(chosen, index, 4) for index in range(1, 5)]
        names = [unit for shard in shards for unit, _ in shard]
        self.assertEqual(sorted(names), sorted(unit for unit, _ in chosen))

    def test_shards_fill_by_recorded_time_longest_first(self):
        chosen = [("long", 10.0), ("a", 4.0), ("b", 3.0), ("c", 3.0), ("d", 1.0)]
        first, second = run_parallel.shard(chosen, 1, 2), run_parallel.shard(chosen, 2, 2)
        # long fills the first shard; a, b and c the second up to the same time; d goes to the first on that tie.
        self.assertEqual([unit for unit, _ in first], ["long", "d"])
        self.assertEqual([unit for unit, _ in second], ["a", "b", "c"])
        loads = [sum(seconds for _, seconds in shard) for shard in (first, second)]
        self.assertLessEqual(max(loads) - min(loads), 1.0)

    def test_the_guards_left_to_the_lint_job_are_the_scripts_it_runs(self):
        workflow = (run_parallel.TESTS.parent / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
        for module in run_parallel.LINT:
            self.assertTrue((run_parallel.TESTS / f"{module}.py").is_file(), module)
            self.assertIn(f"run: python -B tests/{module}.py", workflow)

    def test_a_shard_can_be_empty(self):
        self.assertEqual(run_parallel.shard([("only", 5.0)], 2, 3), [])

    def test_units_split_the_modules_the_recorded_times_split(self):
        times = {"linux": {"test_a.One": 5.0, "test_b": 2.0}, "windows": {"test_a.Two": 9.0}}
        with mock.patch.object(run_parallel, "PLATFORM", "linux"), \
                mock.patch.object(run_parallel, "classes",
                                  return_value={"test_a": ["test_a.One", "test_a.Two", "test_a.New"]}) as classes:
            found = run_parallel.units(["test_a", "test_b", "test_c"], times)
        classes.assert_called_once_with(["test_a"])
        self.assertEqual(found, [("test_a.One", 5.0), ("test_a.Two", 9.0), ("test_a.New", run_parallel.UNRECORDED),
                                 ("test_b", 2.0), ("test_c", run_parallel.UNRECORDED)])

    def test_newer_times_lie_over_durations_and_a_missing_file_is_none(self):
        with tempfile.TemporaryDirectory() as folder:
            durations, newer = Path(folder) / "durations.json", Path(folder) / "newer.json"
            durations.write_text(json.dumps({"linux": {"test_a": 1.0, "test_b": 2.0}}), encoding="utf-8")
            with mock.patch.object(run_parallel, "DURATIONS", durations):
                self.assertEqual(run_parallel.recorded(newer), {"linux": {"test_a": 1.0, "test_b": 2.0}})
                newer.write_text(json.dumps({"linux": {"test_a": 300.0, "test_c": 5.0}, "windows": {"test_a": 600.0}}),
                                 encoding="utf-8")
                self.assertEqual(run_parallel.recorded(newer), {"linux": {"test_a": 300.0, "test_b": 2.0, "test_c": 5.0},
                                                                "windows": {"test_a": 600.0}})
                newer.write_text("{}", encoding="utf-8")   # CI's times job without a cache
                self.assertEqual(run_parallel.recorded(newer), {"linux": {"test_a": 1.0, "test_b": 2.0}})

    def test_short_classes_of_a_module_join_into_units_of_bounded_time(self):
        chosen = [("test_a.Long", 40.0), ("test_a.One", 5.0), ("test_a.Two", 9.0), ("test_b", 2.0), ("test_c.Only", 3.0),
                  ("test_a.New", run_parallel.UNRECORDED), ("test_a.Three", 15.0)]
        with mock.patch.object(run_parallel, "GROUP", 20.0):
            found = run_parallel.grouped(chosen)
        # One and Two fill a group of 20 s; Three starts the next; a module, a long class and an unrecorded one stay alone.
        self.assertEqual(found, [("test_a.Long", 40.0), ("test_a.One test_a.Two", 14.0), ("test_b", 2.0),
                                 ("test_c.Only", 3.0), ("test_a.New", run_parallel.UNRECORDED), ("test_a.Three", 15.0)])
        self.assertEqual(sorted(name for unit, _ in found for name in unit.split()), sorted(unit for unit, _ in chosen))
        self.assertEqual(run_parallel.label("test_a.One test_a.Two"), "test_a (2 short classes: One, Two)")
        self.assertEqual(run_parallel.label("test_b"), "test_b")

    def test_merge_folds_records_into_durations(self):
        with tempfile.TemporaryDirectory() as folder:
            durations, record = Path(folder) / "durations.json", Path(folder) / "record.json"
            durations.write_text(json.dumps({"linux": {"test_a": 1.0, "test_b": 2.0}}), encoding="utf-8")
            record.write_text(json.dumps({"linux": {"test_a": 3.0}, "windows": {"test_a": 6.0}}), encoding="utf-8")
            with mock.patch.object(run_parallel, "DURATIONS", durations):
                run_parallel.merge([record])
            self.assertEqual(json.loads(durations.read_text(encoding="utf-8")),
                             {"linux": {"test_a": 3.0, "test_b": 2.0}, "windows": {"test_a": 6.0}})


if __name__ == "__main__":
    unittest.main()
