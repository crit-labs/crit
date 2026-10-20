"""crit_core.machine: one engine-loading process per machine."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "core/python"))
from crit_core import machine  # noqa: E402

HOLD = """
import sys, time
sys.path.insert(0, sys.argv[2])
from crit_core.machine import engine_lock
with engine_lock("crit eval run (test holder)"):
    print("held", flush=True)
    time.sleep(float(sys.argv[1]))
"""


class EngineLockTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name) / "engine.lock"
        self.env = {key: os.environ.get(key) for key in (machine.LOCK_ENV, machine.HELD_ENV, machine.LEASE_ENV)}
        os.environ[machine.LOCK_ENV] = str(self.path)
        os.environ.pop(machine.HELD_ENV, None)
        os.environ.pop(machine.LEASE_ENV, None)
        self.lease = Path(str(self.path) + ".lease")

    def tearDown(self):
        for key, value in self.env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        self.tmp.cleanup()

    def holder_process(self, seconds):
        env = dict(os.environ)
        env.pop(machine.HELD_ENV, None)
        proc = subprocess.Popen([sys.executable, "-c", HOLD, str(seconds), str(ROOT / "core/python")], env=env,
                                stdout=subprocess.PIPE, text=True)
        self.assertEqual(proc.stdout.readline().strip(), "held")
        return proc

    def test_holder_is_named_and_children_do_not_wait(self):
        with machine.engine_lock("crit run measure r c"):
            self.assertIn("crit run measure r c", machine.holder(self.path))
            self.assertEqual(os.environ.get(machine.HELD_ENV), "1")
            with machine.engine_lock("nested"):   # a child of the holder never waits for it
                pass
        self.assertIsNone(os.environ.get(machine.HELD_ENV))
        self.assertIsNone(machine.holder(self.path))

    def test_a_second_process_waits_or_refuses(self):
        proc = self.holder_process(3)
        try:
            with self.assertRaisesRegex(RuntimeError, "test holder"):
                with machine.engine_lock("second", wait=False):
                    pass
            lines = []
            started = time.time()
            with machine.engine_lock("second", poll_s=0.1, log=lines.append):
                waited = time.time() - started
            self.assertTrue(lines and "test holder" in lines[0], lines)
            self.assertGreater(waited, 0.5)
        finally:
            proc.wait(timeout=30)
            proc.stdout.close()

    def foreign_lease(self, age_s):
        self.lease.write_text(json.dumps({"id": "other", "pid": 7, "label": "crit serve (WSL test)", "system": "wsl",
                                          "at": time.time() - age_s}), encoding="utf-8")

    def test_the_lease_is_held_and_removed(self):
        self.assertEqual(machine.lease_path(), self.lease)
        with machine.engine_lock("crit eval run"):
            lease = json.loads(self.lease.read_text(encoding="utf-8"))
            self.assertEqual(lease["label"], "crit eval run")
        self.assertFalse(self.lease.exists())

    def test_a_fresh_lease_of_another_os_instance_blocks(self):
        self.foreign_lease(age_s=1)
        with self.assertRaisesRegex(RuntimeError, "WSL test"):
            with machine.engine_lock("second", wait=False):
                pass
        # The file lock was released with the refusal.
        os.environ[machine.LEASE_ENV] = "off"
        with machine.engine_lock("third", wait=False):
            pass

    def test_a_stale_lease_does_not_block(self):
        self.foreign_lease(age_s=machine.LEASE_STALE_S + 5)
        with machine.engine_lock("second", wait=False):
            self.assertEqual(json.loads(self.lease.read_text(encoding="utf-8"))["label"], "second")

    def test_off_disables_the_lock(self):
        os.environ[machine.LOCK_ENV] = "off"
        with machine.engine_lock("anything"):
            self.assertIsNone(os.environ.get(machine.HELD_ENV))

    def test_deadline_bounds_wait_for_file_lock(self):
        proc = self.holder_process(10)
        try:
            started = time.monotonic()
            with self.assertRaises(TimeoutError):
                with machine.engine_lock("bounded waiter", deadline=started + .15, poll_s=5):
                    self.fail("busy lock admitted a waiter")
            self.assertLess(time.monotonic() - started, 2)
        finally:
            proc.terminate()
            proc.wait(timeout=10)
            proc.stdout.close()

    def test_deadline_bounds_wait_for_shared_lease(self):
        self.foreign_lease(age_s=1)
        started = time.monotonic()
        with self.assertRaises(TimeoutError):
            with machine.engine_lock("bounded waiter", deadline=started + .15, poll_s=5):
                self.fail("busy lease admitted a waiter")
        self.assertLess(time.monotonic() - started, 2)
        self.assertEqual(json.loads(self.lease.read_text())["id"], "other")
        os.environ[machine.LEASE_ENV] = "off"
        with machine.engine_lock("next caller", wait=False):
            pass

    def test_expiry_during_lease_acquisition_releases_both_locks(self):
        with self.assertRaises(TimeoutError):
            with machine.engine_lock("expires during handshake", deadline=time.monotonic() + .05):
                self.fail("expired caller admitted")
        self.assertFalse(self.lease.exists())
        self.assertNotIn(machine.HELD_ENV, os.environ)
        with machine.engine_lock("next caller", wait=False):
            pass


if __name__ == "__main__":
    unittest.main()
