"""The host's file reads (crit_plugins.host.files): positioned reads at a queue depth, direct or buffered."""
import os
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "inference/core/python"))

from crit_plugins.host import Host  # noqa: E402


class Reads(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name) / "store.bin"
        self.data = os.urandom(3_000_000 + 123)          # ends inside a block
        self.path.write_bytes(self.data)
        self.host = Host(cores=[0])

    def reads(self, **options):
        found = self.host.reads("test", self.path, **options)
        self.addCleanup(found.close)
        return found

    def test_every_range_comes_back_whole_at_any_depth_direct_or_buffered(self):
        ranges = [(0, 10), (4095, 3), (5000, 70000), (len(self.data) - 50, 50), (1 << 20, 1 << 20), (123457, 4096)]
        for direct in (True, False):
            for depth in (1, 4):
                with self.subTest(direct=direct, depth=depth):
                    reads = self.reads(depth=depth, direct=direct)
                    buffers = [(offset, bytearray(size)) for offset, size in ranges]
                    self.assertEqual(reads.read(buffers), sum(size for _, size in ranges))
                    for offset, buffer in buffers:
                        self.assertEqual(bytes(buffer), self.data[offset:offset + len(buffer)])
                    seen = reads.stats()
                    self.assertEqual((seen["reads"], seen["depth"]), (len(ranges), depth))
                    # Direct where the file system allows it; a refusal is said, and the reads are buffered.
                    self.assertEqual(seen["direct"], direct and seen["fallback"] is None)

    def test_a_read_past_the_end_fails_after_the_others_finished(self):
        reads = self.reads(depth=4)
        good = bytearray(100)
        with self.assertRaises(OSError):
            reads.read([(len(self.data) - 10, bytearray(20)), (50, good)])
        self.assertEqual(bytes(good), self.data[50:150])

    def test_the_hosts_report_lists_open_reads(self):
        reads = self.reads()
        reads.read([(0, bytearray(8))])
        listed = self.host.report()["reads"]
        self.assertEqual([found["bytes"] for found in listed.values()], [8])
        reads.close()
        self.assertNotIn("reads", self.host.report())
        with self.assertRaises(ValueError):
            reads.read([(0, bytearray(8))])

    def test_whole_blocks_into_memory_that_starts_on_a_block_come_back_whole(self):
        # A mapping filled piece by piece (a store read into host memory): a direct read lands where it goes.
        import mmap
        target = mmap.mmap(-1, 1 << 20)
        view = memoryview(target)
        for direct in (True, False):
            reads = self.reads(direct=direct, depth=2)
            reads.read([(4096 * 3, view[:8192]), (4096 * 100, view[8192:8192 + 4096 * 16])])
            self.assertEqual(bytes(view[:8192]), self.data[4096 * 3:4096 * 5])
            self.assertEqual(bytes(view[8192:8192 + 4096 * 16]), self.data[4096 * 100:4096 * 116])
        view.release()

    def test_memory_written_to_a_file_is_the_file_whatever_its_last_block(self):
        # A store's memory written out past the page cache: whole blocks direct, a last part of a block buffered.
        import ctypes
        import mmap
        from crit_plugins.host import write_memory
        for size in (4096 * 5, 4096 * 5 + 1234, 100):
            source = mmap.mmap(-1, 4096 * 6)
            source[:size] = self.data[:size]
            address = ctypes.addressof(ctypes.c_char.from_buffer(source))
            target, seen = Path(self.tmp.name) / f"written-{size}.bin", []
            write_memory(target, address, size, progress=lambda done, total: seen.append((done, total)), piece=8192)
            self.assertEqual(target.read_bytes(), self.data[:size])
            self.assertEqual(seen[-1], (size, size))
            write_memory(target, address, 50)   # a file written again is cut to what is written
            self.assertEqual(target.read_bytes(), self.data[:50])


if __name__ == "__main__":
    unittest.main()
