"""crit_plugins.disk_cache: the machine-wide ledger of files plugins keep, its cap, its free-space floor and its
least-recently-used eviction."""
import os
from pathlib import Path
import sys
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "core" / "python"))
from crit_plugins import disk_cache  # noqa: E402

GIB = disk_cache.GIB


class DiskCacheTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(tempfile.mkdtemp())
        self.env = patch.dict(os.environ, {disk_cache.LEDGER_ENV: str(self.root / "ledger.json"),
                                           disk_cache.CAP_ENV: "3", disk_cache.FLOOR_ENV: "0"})
        self.env.start()
        self.free = 100 * GIB
        self.usage = patch("shutil.disk_usage", side_effect=lambda _: SimpleNamespace(free=self.free))
        self.usage.start()

    def tearDown(self):
        self.usage.stop()
        self.env.stop()

    def keep(self, name, gib):
        path = self.root / name
        admitted, reason = disk_cache.admit(path, gib * GIB, [str(path) + ".json"])
        if admitted:
            path.write_bytes(b"x")
            Path(str(path) + ".json").write_text("{}")
        return admitted, reason

    def test_the_cap_evicts_the_least_recently_used_file_and_its_index(self):
        self.assertEqual(self.keep("a.bin", 1), (True, None))
        time.sleep(0.01)
        self.assertTrue(self.keep("b.bin", 1)[0])
        time.sleep(0.01)
        disk_cache.touch(self.root / "a.bin")   # a reopened: b is now the oldest
        self.assertTrue(self.keep("c.bin", 2)[0])
        self.assertTrue((self.root / "a.bin").exists())
        self.assertFalse((self.root / "b.bin").exists() or (self.root / "b.bin.json").exists())

    def test_the_floor_evicts_and_then_refuses(self):
        os.environ[disk_cache.FLOOR_ENV] = "10"
        self.assertTrue(self.keep("a.bin", 1)[0])
        self.free = 10.5 * GIB   # a 1 GiB file would leave 9.5 GiB: a goes, and the disk still is short
        admitted, reason = self.keep("b.bin", 1)
        self.assertFalse(admitted)
        self.assertIn("floor", reason)
        self.assertFalse((self.root / "a.bin").exists())

    def test_every_cache_lives_under_one_home(self):
        from crit_core import cache_home
        from crit_plugins import native, testing
        home = self.root / "home"
        with patch.dict(os.environ, {"CRIT_CACHE_HOME": str(home)}):
            for name in (disk_cache.LEDGER_ENV, "CRIT_NATIVE_CACHE", "TORCH_EXTENSIONS_DIR", "CRIT_PLUGIN_TEST_CACHE"):
                os.environ.pop(name, None)
            self.assertEqual(disk_cache.ledger_path(), home / "plugin-cache.json")
            self.assertEqual((native.cache_root(), testing.cache_folder()), (home / "native", home / "plugin-tests"))
            self.assertEqual(cache_home("kernels"), home / "kernels")

    def test_a_file_over_the_cap_is_refused_without_evicting(self):
        self.assertTrue(self.keep("a.bin", 1)[0])
        self.assertFalse(self.keep("big.bin", 4)[0])
        self.assertTrue((self.root / "a.bin").exists())


if __name__ == "__main__":
    unittest.main()
