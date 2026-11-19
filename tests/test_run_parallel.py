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
