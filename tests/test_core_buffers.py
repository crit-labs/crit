"""crit_core.buffers, the Python twin of Core's plan_buffers and BufferArena (crit/device_memory.h): the same layout
as tests/core/device_memory_tests.cpp checks for the C++ library (memory_buffer_plan, memory_buffer_arena)."""
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "core" / "python"))

from crit_core.buffers import BufferRequest, ScratchArena, plan_buffers  # noqa: E402

REQUESTS = [BufferRequest("weights", 1000, 256), BufferRequest("kv", 4096, 4096), BufferRequest("logits", 10, 64)]


class BufferPlanTests(unittest.TestCase):
    def test_the_plan_matches_the_native_layout(self):
        plan = plan_buffers(REQUESTS, available=64 * 1024, reserve=8 * 1024, scratch_bytes=300, slot_bytes=1000)
        self.assertEqual([b.offset for b in plan.buffers], [0, 4096, 8192])
        self.assertEqual((plan.scratch_offset, plan.slots_offset, plan.slot_stride), (8448, 8960, 1024))
        self.assertEqual(plan.slots, (56 * 1024 - 8960) // 1024)
        self.assertEqual(plan.total_bytes + plan.unused_bytes + plan.reserve_bytes, 64 * 1024)
        self.assertLess(plan.unused_bytes, plan.slot_stride)

    def test_bytes_a_smaller_plan_frees_become_slots(self):
        args = dict(available=64 * 1024, reserve=8 * 1024, scratch_bytes=300, slot_bytes=1000)
        lower = [REQUESTS[0], BufferRequest("kv", 1024, 4096), REQUESTS[2]]
        self.assertGreater(plan_buffers(lower, **args).slots, plan_buffers(REQUESTS, **args).slots)
        self.assertEqual(plan_buffers(REQUESTS, max_slots=3, **args).slots, 3)
        with self.assertRaisesRegex(MemoryError, "short by"):
            plan_buffers(REQUESTS, min_slots=1000, **args)

    def test_bad_requests_are_refused(self):
        for requests, message in (([BufferRequest("a", 8), BufferRequest("a", 8)], "named twice"),
                                  ([BufferRequest("a", 0)], "no bytes"), ([BufferRequest("a", 8, 3)], "power of two")):
            with self.assertRaisesRegex(ValueError, message):
                plan_buffers(requests, available=4096)
        with self.assertRaises(KeyError):
            plan_buffers(REQUESTS, available=1 << 20).buffer("missing")

    def test_views_slots_and_scratch_are_slices_of_one_block(self):
        plan = plan_buffers([BufferRequest("a", 3), BufferRequest("b", 5, 16)], available=8192, scratch_bytes=64,
                            slot_bytes=100, slot_alignment=128)
        block = bytearray(plan.total_bytes)
        view = memoryview(block)
        plan.view(view, "b")[:] = b"\x07" * 5
        self.assertEqual(block[16:21], b"\x07" * 5)
        plan.slot(view, 1)[:] = b"\x01" * 100
        self.assertEqual(block[plan.slots_offset + 128:plan.slots_offset + 228], b"\x01" * 100)
        with self.assertRaises(IndexError):
            plan.slot(view, plan.slots)
        arena = ScratchArena(plan.scratch(view))
        mark = arena.mark()
        self.assertEqual(len(arena.allocate(40, 16)), 40)
        arena.rewind(mark)
        with self.assertRaises(MemoryError):
            arena.allocate(65, 1)
        self.assertEqual(arena.peak, 40)
        with self.assertRaises(ValueError):
            plan_buffers([BufferRequest("a", 3)], available=8192).scratch(view)


if __name__ == "__main__":
    unittest.main()
