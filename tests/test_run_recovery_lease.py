"""Two recoverers must never remove each other's live coordinator lease."""
import json
import subprocess
import sys
import threading
import unittest
from unittest.mock import patch

from test_run_records import RunFixture
from crit.run import store as run_store


class RecoveryLeaseTests(RunFixture, unittest.TestCase):
    def test_delayed_recoverer_cannot_steal_a_replacement_lease(self):
        self.candidate()
        with subprocess.Popen([sys.executable, "-B", "-c", "pass"]) as process:
            process.wait(timeout=20)
            stale = run_store.lease_owner() | {"pid": process.pid}
            (self.directory / ".coordinator.lock").write_text(json.dumps(stale), encoding="utf-8")
            self.exercise_race()

    def exercise_race(self):
        first_read = threading.Event()
        resume_first = threading.Event()
        second_progress = threading.Event()
        release_second = threading.Event()
        first_done = threading.Event()
        second_inside = threading.Event()
        overlaps, outcomes = [], {}
        read_owner = run_store.read_lease_owner
        refresh = self.store._refresh

        def delayed_read(path):
            owner = read_owner(path)
            if threading.current_thread().name == "first-recoverer":
                first_read.set()
                if not resume_first.wait(10):
                    raise AssertionError("first recoverer was not resumed")
            return owner

        def guarded_refresh(*args):
            if threading.current_thread().name == "second-recoverer":
                second_inside.set()
                second_progress.set()
                if not release_second.wait(10):
                    raise AssertionError("second recoverer was not released")
            elif second_inside.is_set():
                overlaps.append("both recoverers entered the coordinator")
            return refresh(*args)

        def recover(name):
            try:
                outcomes[name] = self.store.recover(self.run_id)
            except BaseException as exc:
                outcomes[name] = exc
            finally:
                (first_done if name == "first" else second_progress).set()

        first = threading.Thread(target=recover, args=("first",), name="first-recoverer")
        second = threading.Thread(target=recover, args=("second",), name="second-recoverer")
        with patch.object(run_store, "read_lease_owner", side_effect=delayed_read), \
             patch.object(self.store, "_refresh", side_effect=guarded_refresh):
            try:
                first.start()
                self.assertTrue(first_read.wait(10))
                second.start()
                self.assertTrue(second_progress.wait(10))
                resume_first.set()
                self.assertTrue(first_done.wait(10))
            finally:
                resume_first.set()
                release_second.set()
                first.join(timeout=10)
                if second.ident is not None:
                    second.join(timeout=10)
        self.assertFalse(first.is_alive() or second.is_alive())
        self.assertEqual(overlaps, [])
        self.assertIsInstance(outcomes["second"], ValueError)
        self.assertEqual(outcomes["first"]["errors"], [])
        self.assertEqual(self.store.get(self.run_id)["errors"], [])


if __name__ == "__main__":
    unittest.main()
