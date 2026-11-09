"""crit_plugins.files: positioned reads and writes, through os.pread / os.pwrite and through the seek-under-a-lock path
Windows takes (forced here on any platform)."""
from concurrent.futures import ThreadPoolExecutor
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "core" / "python"))
from crit_plugins import files  # noqa: E402

BINARY = getattr(os, "O_BINARY", 0)


class PositionedFileTests(unittest.TestCase):
    def roundtrip(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "rows.bin"
            fd = os.open(path, os.O_RDWR | os.O_CREAT | BINARY, 0o644)
            try:
                rows = [bytes([i]) * 7 + b"\r\n\x1a" for i in range(64)]   # bytes text mode would change
                for i in reversed(range(64)):
                    self.assertEqual(files.pwrite(fd, rows[i], i * 10), 10)
                with ThreadPoolExecutor(8) as pool:
                    got = list(pool.map(lambda i: files.pread(fd, 10, i * 10), range(64)))
                self.assertEqual(got, rows)
                self.assertEqual(files.pread(fd, 10, 635), rows[63][5:])   # short at the end of the file
                self.assertEqual(files.pread(fd, 10, 640), b"")
            finally:
                os.close(fd)

    def test_reads_and_writes_land_at_their_offsets(self):
        self.roundtrip()

    def test_without_positioned_calls_threads_seek_and_read_in_turn(self):
        # Windows has no os.pread or os.pwrite: the helper seeks under the descriptor's lock instead.
        windows_os = SimpleNamespace(lseek=os.lseek, read=os.read, write=os.write, SEEK_SET=os.SEEK_SET)
        with patch.object(files, "os", windows_os):
            self.assertFalse(hasattr(files.os, "pread") or hasattr(files.os, "pwrite"))
            self.roundtrip()


if __name__ == "__main__":
    unittest.main()
