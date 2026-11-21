"""Core's host memory admission (crit_plugins.host.memory): what the OS can back, from physical memory, the commit
headroom where the OS enforces commit and the cgroup's headroom; memory_bytes() answers with the tightest, so every
host allocation sized from it is one the OS admits."""
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "core/python"))

from crit_plugins import host  # noqa: E402
from crit_plugins.host import memory  # noqa: E402


class AdmissionTests(unittest.TestCase):
    def tree(self, root, files):
        for path, text in files.items():
            target = Path(root) / path
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(text)

    def test_linux_observations_and_admission(self):
        m = memory
        with tempfile.TemporaryDirectory() as root:
            self.tree(root, {"proc/meminfo": "MemTotal: 1000 kB\nMemAvailable: 800 kB\nCommitLimit: 900 kB\nCommitted_AS: 600 kB\n",
                             "proc/sys/vm/overcommit_memory": "2\n", "proc/self/cgroup": "0::/app/worker\n",
                             "sys/fs/cgroup/app/worker/memory.max": "max\n", "sys/fs/cgroup/app/worker/memory.current": "1\n",
                             "sys/fs/cgroup/app/memory.max": "500000\n", "sys/fs/cgroup/app/memory.current": "450000\n",
                             "sys/fs/cgroup/app/memory.stat": "file 60000\nfile_dirty 15000\nfile_writeback 5000\n"})
            seen = m.observe_linux(root)
            self.assertEqual((seen.physical_available, seen.commit_available, seen.cgroup_available), (819200, 307200, 90000))
            self.assertTrue(m.admit(80000, seen, 10000).admitted)
            denied = m.admit(80001, seen, 10000)
            self.assertEqual((denied.admitted, denied.binding, denied.limit), (False, "cgroup", 80000))
            self.assertEqual(m.admit(0, seen, 90001).limit, 0)
            self.tree(root, {"proc/sys/vm/overcommit_memory": "0\n", "sys/fs/cgroup/app/memory.max": "max\n"})
            seen = m.observe_linux(root)
            self.assertIsNone(seen.commit_available)
            self.assertIsNone(seen.cgroup_available)
            self.assertEqual(len(seen.notes), 2)
            self.tree(root, {"proc/self/cgroup": "4:memory:/job\n1:cpu:/job\n",
                             "sys/fs/cgroup/memory/job/memory.usage_in_bytes": "700\n",
                             "sys/fs/cgroup/memory/job/memory.stat": "cache 100\ndirty 30\nwriteback 10\nhierarchical_memory_limit 1000\n"})
            self.assertEqual(m.observe_linux(root).cgroup_available, 360)
            # Hybrid host: an unlimited v2 tree and a v1 memory hierarchy; the v1 limit applies, as cgroup v1 hybrid hosts count it.
            self.tree(root, {"proc/self/cgroup": "0::/app/worker\n4:memory:/job\n"})
            self.assertEqual(m.observe_linux(root).cgroup_available, 360)
        with tempfile.TemporaryDirectory() as empty:
            unknown = m.observe_linux(empty)
            self.assertEqual(unknown.limits(), {})
            self.assertFalse(m.admit(1, unknown).admitted)
            self.assertTrue(m.admit(1, unknown, require_known=False).admitted)
        with self.assertRaises(ValueError):
            m.admit(-1, unknown)

    def test_cgroup_reclaimable_pages_and_memory_high(self):
        m = memory
        gib = 1 << 30
        with tempfile.TemporaryDirectory() as root:
            app = "sys/fs/cgroup/app/"
            self.tree(root, {"proc/meminfo": "MemAvailable: 800 kB\n", "proc/self/cgroup": "0::/app/worker\n",
                             app + "memory.max": "500000\n", app + "memory.current": "450000\n",
                             app + "memory.stat": "file 60000\nshmem 50000\nfile_dirty 1000\nfile_writeback 0\n"})
            # tmpfs/shmem pages sit in file but are swap-backed: only the file LRUs, else file - shmem, are reclaimable.
            self.assertEqual(m.observe_linux(root).cgroup_available, 59000)
            self.tree(root, {app + "memory.stat": "file 60000\nshmem 50000\nactive_file 4000\ninactive_file 6000\n"
                                                  "file_dirty 1000\nfile_writeback 500\n"})
            self.assertEqual(m.observe_linux(root).cgroup_available, 58500)
            self.tree(root, {app + "memory.max": "%d\n" % (8 * gib), app + "memory.current": "%d\n" % (8 * gib),
                             app + "memory.stat": "anon %d\nfile %d\nshmem %d\nactive_file 0\ninactive_file 0\n"
                                                  % (2 * gib, 6 * gib, 6 * gib)})
            seen = m.observe_linux(root)
            self.assertEqual(seen.cgroup_available, 0)
            denied = m.admit(5 * gib, seen)
            self.assertEqual((denied.admitted, denied.binding), (False, "cgroup"))
            # memory.high limits too: the effective limit is min(memory.max, memory.high).
            self.tree(root, {app + "memory.max": "max\n", app + "memory.high": "460000\n", app + "memory.current": "450000\n",
                             app + "memory.stat": "active_file 4000\ninactive_file 6000\nfile_dirty 1000\nfile_writeback 500\n"})
            self.assertEqual(m.observe_linux(root).cgroup_available, 18500)
            self.tree(root, {app + "memory.max": "455000\n"})
            self.assertEqual(m.observe_linux(root).cgroup_available, 13500)
            self.tree(root, {app + "memory.max": "max\n", app + "memory.high": "max\n"})
            seen = m.observe_linux(root)
            self.assertIsNone(seen.cgroup_available)
            self.assertIn("no cgroup v2 memory limit", seen.notes)
            # Malformed counters count nothing as reclaimable, as the kernel would not reclaim them; malformed limits skip the level.
            self.tree(root, {app + "memory.max": "1000\n", app + "memory.high": "max\n", app + "memory.current": "400\n"})
            for stat in ("file 100\nfile_dirty lots\n", "file 100\nshmem x\n", "active_file 100\ninactive_file -1\n",
                         "active_file 100\nfile_writeback 1.5\n"):
                self.tree(root, {app + "memory.stat": stat})
                seen = m.observe_linux(root)
                self.assertEqual(seen.cgroup_available, 600)
                self.assertTrue(any("memory.stat malformed" in n for n in seen.notes))
            for name in ("memory.max", "memory.high", "memory.current"):
                self.tree(root, {app + "memory.max": "1000\n", app + "memory.high": "max\n", app + "memory.current": "400\n",
                                 app + name: "12z\n"})
                seen = m.observe_linux(root)
                self.assertIsNone(seen.cgroup_available)
                self.assertTrue(any("memory.max, memory.high or memory.current malformed" in n for n in seen.notes))
            # v1: shmem is in cache but not reclaimable; each key prefers its hierarchical total_ form.
            job = "sys/fs/cgroup/memory/job/"
            self.tree(root, {"proc/self/cgroup": "4:memory:/job\n", job + "memory.usage_in_bytes": "700\n",
                             job + "memory.stat": "cache 500\nshmem 300\ndirty 50\nhierarchical_memory_limit 1000\n"})
            self.assertEqual(m.observe_linux(root).cgroup_available, 450)
            self.tree(root, {job + "memory.stat": "cache 500\nshmem 300\ntotal_cache 900\ntotal_shmem 800\ndirty 50\n"
                                                  "hierarchical_memory_limit 1000\n"})
            self.assertEqual(m.observe_linux(root).cgroup_available, 350)
            self.tree(root, {job + "memory.stat": "cache 500\nshmem 300\ntotal_active_file 100\ntotal_inactive_file 60\n"
                                                  "inactive_file 1\ntotal_dirty 20\nhierarchical_memory_limit 1000\n"})
            self.assertEqual(m.observe_linux(root).cgroup_available, 440)
            self.tree(root, {job + "memory.stat": "hierarchical_memory_limit 100\ncache x\n", job + "memory.usage_in_bytes": "10\n"})
            self.assertEqual(m.observe_linux(root).cgroup_available, 90)
            for stat, usage, note in (("hierarchical_memory_limit 1e9\n", "10\n", "hierarchical_memory_limit malformed"),
                                      ("hierarchical_memory_limit 100\n", "ten\n", "memory.usage_in_bytes malformed")):
                self.tree(root, {job + "memory.stat": stat, job + "memory.usage_in_bytes": usage})
                seen = m.observe_linux(root)
                self.assertIsNone(seen.cgroup_available)
                self.assertTrue(any(note in n for n in seen.notes))

    def test_live_observation_reports_physical_memory(self):
        m = memory
        seen = m.observe()
        if sys.platform.startswith("linux") or sys.platform == "win32":
            self.assertGreater(seen.physical_available, 0)


class MemoryBytesTests(unittest.TestCase):
    def test_available_is_the_tightest_observation(self):
        seen = memory.Observation(physical_available=900, commit_available=400, cgroup_available=600)
        with mock.patch.object(memory, "observe", return_value=seen):
            found = host.memory_bytes()
        self.assertEqual((found["available"], found["binding"]), (400, "commit"))
        with mock.patch.object(memory, "observe", return_value=memory.Observation(notes=("nothing",))):
            self.assertEqual((host.memory_bytes()["available"], host.memory_bytes()["binding"]), (0, None))


if __name__ == "__main__":
    unittest.main()
