"""crit_plugins.memory: one object a device. Its ledger hands out what the device lacks, leaves free what the givers
keep and lets them grow back once the engine serves; an elastic range grows and shrinks with its addresses and its
contents in place; the census counts torch's bytes for the plugin whose objects hold them (CUDA)."""
import sys
import types
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "inference" / "core" / "python"))

from crit_plugins import memory  # noqa: E402
from crit_plugins.observe import Calls, Observed  # noqa: E402

try:
    import torch
except ImportError:
    torch = None


class Readings:
    """A device of `free` bytes whose torch allocator holds `unused` bytes in blocks nothing uses."""
    total = 10_000

    def __init__(self, free, unused=0):
        self.bytes, self.unused, self.released = free, unused, 0

    def free(self):
        return self.bytes

    def torch(self):
        return {"allocated": 2000, "reserved": 2000 + self.unused, "unused": self.unused}

    def release(self):
        self.bytes, self.unused, self.released = self.bytes + self.unused, 0, self.released + 1

    def segments(self):
        return []


class Giver:
    """A tier of `held` bytes that gives back whole units of `unit` bytes."""

    def __init__(self, readings, held, unit):
        self.readings, self.held, self.unit = readings, held, unit

    def release(self, nbytes):
        freed = min(self.held, -(-nbytes // self.unit) * self.unit)
        self.held -= freed
        self.readings.bytes += freed
        return freed

    def regrow(self, room):
        taken = room // self.unit * self.unit
        self.held += taken
        self.readings.bytes -= taken
        return taken


class LedgerTests(unittest.TestCase):
    def setUp(self):
        self.readings = Readings(1000)
        self.memory = memory.DeviceMemory("fake:0", self.readings)
        self.memory.headroom = 0   # the ledger alone: a device of 10,000 bytes keeps no headroom

    def test_nothing_is_taken_while_the_device_has_room(self):
        cache = Giver(self.readings, 5000, 100)
        self.memory.give("cache", cache.release, cache.regrow)
        self.assertEqual(self.memory.need(900), 0)
        self.assertEqual(cache.held, 5000)

    def test_givers_release_what_the_device_lacks_in_their_own_units(self):
        cache = Giver(self.readings, 5000, 300)
        self.memory.give("cache", cache.release, cache.regrow)
        self.assertEqual(self.memory.need(1500), 600)             # 500 missing: two units of 300
        self.assertEqual((cache.held, self.readings.bytes), (4400, 1600))

    def test_a_second_giver_gives_what_the_first_could_not(self):
        small, large = Giver(self.readings, 200, 100), Giver(self.readings, 5000, 100)
        self.memory.give("small", small.release, small.regrow)
        self.memory.give("large", large.release, large.regrow)
        self.assertEqual(self.memory.need(1700), 700)
        self.assertEqual((small.held, large.held), (0, 4500))

    def test_a_need_leaves_what_the_givers_keep_free(self):
        cache = Giver(self.readings, 5000, 100)
        kept = [300]
        self.memory.give("cache", cache.release, cache.regrow, keep=lambda: kept[0])
        self.assertEqual(self.memory.need(700), 0)                # 700 and the 300 kept are the 1000 free
        self.assertEqual(self.memory.need(900), 200)
        self.assertEqual(self.readings.bytes, 1200)
        kept[0] = 500                                             # a step in flight: the giver keeps more
        self.assertEqual(self.memory.need(0), 0)
        kept[0] = 1500
        self.assertEqual(self.memory.need(0), 300)                # nothing to take: only what it keeps is made free
        self.assertEqual(self.readings.bytes, 1500)

    def test_what_the_givers_cannot_release_is_asked_of_torchs_allocator(self):
        self.readings.unused = 400
        cache = Giver(self.readings, 200, 100)
        self.memory.give("cache", cache.release, cache.regrow)
        self.assertEqual(self.memory.need(1100), 100)
        self.assertEqual(self.readings.released, 0)
        self.assertEqual(self.memory.need(5000), 100)             # the giver has 100 left: torch's unused blocks next
        self.assertEqual((self.readings.released, self.readings.bytes), (1, 1600))

    def test_givers_grow_back_only_once_the_engine_serves(self):
        self.readings.unused = 100
        cache = Giver(self.readings, 4000, 300)
        self.memory.give("cache", cache.release, cache.regrow, keep=250)
        self.memory.load()
        self.assertEqual(self.memory.settle(), 0)                 # the load's memory is the load's
        self.assertEqual(cache.held, 4000)
        self.assertEqual(self.memory.serve(), 600)                # torch's unused blocks first: 850 of room, two units
        self.assertEqual((cache.held, self.readings.bytes, self.readings.released), (4600, 500, 1))
        self.assertEqual(self.memory.settle(), 0)
        self.memory.give("cache", cache.release, cache.regrow, keep=lambda: 150)   # what it keeps, read when asked
        self.assertEqual(self.memory.settle(), 300)
        self.assertEqual(self.readings.bytes, 200)
        self.memory.load()                                        # another engine loads on the device
        self.readings.bytes = 2000
        self.assertEqual(self.memory.settle(), 0)

    def test_a_settle_grows_the_givers_back_for_a_bounded_time_and_owes_the_rest(self):
        from unittest import mock
        unit = memory.SETTLE_SLICE // 4
        self.readings.total, self.readings.bytes = 100 * unit, 0
        cache = Giver(self.readings, 10 * unit, unit)
        self.memory.give("cache", cache.release, cache.regrow, keep=0)
        self.memory.serve()
        self.readings.bytes = 40 * unit                           # a prompt's room came back
        ticks = iter(range(100))
        with mock.patch.object(memory.time, "perf_counter", lambda: next(ticks) * 0.03):
            self.assertEqual(self.memory.settle(), 8 * unit)      # two slices of four units inside SETTLE_S
        self.assertTrue(self.memory.owed)
        self.assertEqual(self.memory.settle(None), 32 * unit)     # no bound: the rest
        self.assertFalse(self.memory.owed)
        self.assertEqual(self.memory.settle(), 0)

    def test_free_is_never_more_than_the_card_has_left(self):
        # A driver whose free reading is the process's own budget (Windows, Linux under WSL) does not see what
        # another program took: the card's count of what every program holds bounds the reading.
        import sys
        import types
        from unittest import mock
        fake = types.SimpleNamespace(cuda=types.SimpleNamespace(mem_get_info=lambda index: (900, 1000)))
        readings = memory._Torch(0)
        with mock.patch.dict(sys.modules, {"torch": fake}):
            for left, free in ((None, 900), (950, 900), (600, 600), (0, 0)):
                with mock.patch.object(memory, "card_free", lambda index, left=left: left):
                    self.assertEqual(readings.free(), free, left)

    def test_memory_another_program_takes_is_given_back_between_steps_and_not_taken_again_at_once(self):
        now = [100.0]
        self.memory.clock = lambda: now[0]
        slack, hold = memory.OUTSIDE_SLACK, memory.HOLD_S
        self.readings.total, self.readings.bytes = 40 * slack, 2 * slack
        unit = slack // 8
        cache = Giver(self.readings, 20 * slack, unit)
        self.memory.give("cache", cache.release, cache.regrow, keep=slack)
        self.assertEqual(self.memory.relieve(), 0)                # the engine still loads: the load's memory
        self.memory.serve()                                       # the cache takes what is free past what it keeps
        self.assertEqual((cache.held, self.readings.bytes), (21 * slack, slack))
        self.assertEqual(self.memory.relieve(), 0)                # nothing was taken: nothing to give
        # A desktop takes two slacks of memory: half of what the driver then gives it is system memory, until the
        # cache gives its own back.
        self.readings.bytes -= 2 * slack
        self.assertEqual(self.memory.relieve(), 0)                # read at most every RELIEVE_S
        now[0] += memory.RELIEVE_S
        self.assertEqual(self.memory.relieve(), 2 * slack)
        self.assertEqual((cache.held, self.readings.bytes), (19 * slack, slack))
        # The desktop gives one back: the cache waits before it takes it again.
        self.readings.bytes += slack
        self.assertEqual(self.memory.settle(), 0)
        now[0] += hold
        self.assertEqual(self.memory.settle(), slack)
        self.assertEqual(self.memory.report()["outside"], {"relieved": 1, "bytes": 2 * slack, "held": 1})
        # A giver that asks for more room itself (a callable keep) is given it without a hold: nobody else grew.
        self.memory.give("cache", cache.release, cache.regrow, keep=lambda: slack + 3 * unit)
        now[0] += memory.RELIEVE_S
        self.assertEqual(self.memory.relieve(), 3 * unit)
        self.assertEqual(self.memory.report()["outside"]["held"], 1)
        self.memory.give("cache", cache.release, cache.regrow, keep=slack)
        self.assertEqual(self.memory.settle(), 3 * unit)

    def test_a_watch_asks_for_a_step_once_when_another_program_leaves_less_than_the_givers_keep(self):
        import threading
        cache = Giver(self.readings, 5000, 100)
        self.memory.give("cache", cache.release, cache.regrow, keep=300)
        asked, fell = [], threading.Event()
        self.memory.watch(lambda: (asked.append(self.readings.bytes), fell.set()), every=0.005)
        self.addCleanup(self.memory.unwatch)
        self.assertFalse(fell.wait(0.05))                         # the engine loads: nothing to watch
        self.memory.serve()                                       # the cache takes 700 of the 1,000 free
        self.assertEqual(self.readings.bytes, 300)
        self.assertFalse(fell.wait(0.05))                         # free is what the givers keep
        self.readings.bytes = 120                                 # a desktop takes 180
        self.assertTrue(fell.wait(2))
        fell.clear()
        self.assertFalse(fell.wait(0.05))                         # asked once, while the device stays as short
        self.assertEqual(asked, [120])
        self.memory.clock = lambda: 1e9                           # the engine's step: the cache gives 200 back
        self.assertEqual(self.memory.relieve(), 200)
        self.assertFalse(fell.wait(0.05))
        self.readings.bytes = 250                                 # and takes more: asked again
        self.assertTrue(fell.wait(2))
        self.memory.unwatch()
        self.assertIsNone(self.memory._watch)

    def test_a_giver_that_left_is_not_asked(self):
        cache = Giver(self.readings, 4000, 100)
        self.memory.give("cache", cache.release, cache.regrow)
        self.memory.ungive("cache")
        self.assertEqual(self.memory.need(5000), 0)
        self.assertEqual(cache.held, 4000)

    def test_the_report_is_the_whole_device(self):
        self.readings.unused = 300
        found = self.memory.report()
        self.assertEqual(found.pop("steps")["arena"], 0)   # the steps' memory: Core's allocator, the arena, the counts
        self.assertEqual(found, {
            "device": "fake:0", "total": 10_000, "free": 1000, "elastic": {},
            "torch": {"allocated": 2000, "reserved": 2300, "unused": 300}, "other": 6700})

    def test_a_device_whose_memory_cannot_be_read_takes_and_gives_nothing(self):
        cpu = memory.of("cpu")
        self.assertIs(cpu, memory.of("cpu"))
        self.assertEqual((cpu.total, cpu.free(), cpu.torch(), cpu.report(), cpu.census({})), (None,) * 5)
        self.assertEqual((cpu.made(), cpu.pool("rounds", "1 rows")), (None, None))
        self.assertEqual((cpu.need(1 << 40), cpu.serve(), cpu.release(), cpu.elastic_ranges), (0, 0, None, False))


MIB = 1 << 20


class StepReadings:
    """A device of `free` MiB whose torch allocator reserves blocks as a step's tensors come (take) and keeps them
    unused when they go (drop), with its peak since the last mark."""
    total = 32 << 30

    def __init__(self, free):
        self.bytes, self.reserved, self.unused = free * MIB, 2048 * MIB, 0
        self.top = self.reserved

    def free(self):
        return self.bytes

    def torch(self):
        return {"allocated": self.reserved - self.unused, "reserved": self.reserved, "unused": self.unused}

    def release(self):
        self.bytes, self.reserved, self.unused = self.bytes + self.unused, self.reserved - self.unused, 0

    def mark(self):
        self.top = self.reserved

    def peak(self):
        return self.top

    def take(self, mib):   # a tensor of the step: from the unused blocks first, else a new block from the device
        nbytes = mib * MIB
        reused = min(self.unused, nbytes)
        self.unused -= reused
        self.reserved += nbytes - reused
        self.bytes -= nbytes - reused
        self.top = max(self.top, self.reserved)

    def drop(self, mib):   # it goes: the allocator keeps its block
        self.unused += mib * MIB

    def segments(self):
        return []


class RangeGiver(Giver):
    """An elastic expert cache: its slots are an elastic range of the device (the ledger's ranges hold what it backs),
    so what it gives and takes is never a step's."""

    @property
    def mapped(self):
        return self.held


def cache_on(device, unit=4 * MIB):
    cache = RangeGiver(device.readings, 0, unit)
    device.ranges["cache/slots"] = cache
    device.give("cache", cache.release, cache.regrow)
    return cache


class StepMemoryTests(unittest.TestCase):
    """Every byte a step takes is admitted by the device's ledger before it exists (Core's allocator calls _admit): the
    givers keep the headroom and the step arena free, a step's segments come out of the arena, and what passes it or
    the headroom is counted. Nothing is forecast and nothing is looked at during a step."""

    def setUp(self):
        self.readings = StepReadings(free=24 * 1024)
        self.memory = memory.DeviceMemory("fake:0", self.readings)
        self.memory.headroom = 256 * MIB
        self.cache = cache_on(self.memory)   # an elastic expert cache: it releases and regrows, no keep of its own
        self.memory.load()

    def segment(self, mib):
        """Core's allocator hands the step pool a segment: the ledger admits it, the driver makes it."""
        ok = self.memory._admit(mib * MIB)
        if ok:
            self.readings.bytes -= mib * MIB
        return ok

    def free_segment(self, mib):
        self.readings.bytes += mib * MIB
        self.memory._freed(mib * MIB)

    def test_the_arena_is_what_the_load_s_steps_took_or_the_model_s_word_whichever_is_more(self):
        self.memory.begin_step(37)
        self.segment(40)
        self.memory.end_step(False)
        self.memory.begin_step(0, score=True)
        self.segment(100)
        self.memory.end_step(False)
        self.assertEqual(self.memory.most, {"prompt": 40 * MIB, "score": 100 * MIB})
        self.assertEqual(self.memory.size_arena(), int(100 * MIB * memory.ARENA_SLACK))
        self.assertEqual(self.memory.size_arena(declared=400 * MIB), int(400 * MIB * memory.ARENA_SLACK))

    def test_the_givers_keep_the_headroom_and_the_arena_free_and_a_step_takes_it_without_asking(self):
        self.memory.arena = 512 * MIB
        self.memory.serve()                     # the cache took every free byte but the headroom and the arena
        self.assertEqual(self.readings.bytes, (256 + 512) * MIB)
        held = self.cache.held
        self.memory.begin_step(512)
        for _ in range(4):
            self.assertTrue(self.segment(100))   # within the arena: no giver is asked, the headroom stays free
        self.assertEqual(self.cache.held, held)
        self.assertGreaterEqual(self.readings.bytes, 256 * MIB)
        self.memory.end_step(False)             # no prompt in flight: the pool retires, its segments go back
        for _ in range(4):
            self.free_segment(100)
        self.memory.settle(None)
        self.assertEqual(self.cache.held, held)   # the arena is free again, kept for the next step
        steps = self.memory.steps()
        self.assertEqual((steps["overflows"], steps["breaches"], steps["refused"]), (0, 0, 0))
        self.assertEqual((steps["most"]["prompt"], steps["granted"]), (400 * MIB, 0))

    def test_what_passes_the_arena_or_the_headroom_is_counted_and_what_the_device_lacks_is_refused(self):
        self.memory.arena = 128 * MIB
        self.memory.serve()
        self.memory.begin_step(512)
        self.assertTrue(self.segment(100))
        self.assertTrue(self.segment(100))       # past the arena: lent from free memory, counted, and under the
        self.assertTrue(self.segment(100))       # headroom: counted too
        steps = self.memory.steps()
        self.assertEqual((steps["overflows"], steps["breaches"]), (2, 2))
        self.assertLess(steps["least_free"], 256 * MIB)
        self.assertFalse(self.memory._admit(self.readings.bytes + MIB))   # more than the device has: refused
        self.assertEqual(self.memory.steps()["refused"], 1)
        self.memory.end_step(False)

    def test_each_serving_segment_is_kept_with_the_memory_outside_the_engine(self):
        # Core's segment readings for the check: the free bytes a segment left and what others held on the card then,
        # with what they held as serving began, so the check judges the engine's own use (crit.run.checks).
        self.memory.arena = 128 * MIB
        self.memory.serve()
        at_serve = self.memory.steps()["outside_at_serve"]
        self.assertIsNotNone(at_serve)
        self.memory.begin_step(512)
        self.assertTrue(self.segment(100))
        self.readings.reserved += 100 * MIB       # torch holds the segment it was given
        self.readings.bytes -= 50 * MIB           # another program takes 50 MiB of the card
        self.assertTrue(self.segment(20))
        self.memory.end_step(False)
        steps = self.memory.steps()
        (first_left, first_other), (second_left, second_other) = steps["segment_memory"]
        self.assertEqual(second_other - first_other, 50 * MIB)
        self.assertEqual(first_left - second_left, 70 * MIB)    # the 20 MiB segment and the other program's 50
        self.assertEqual(first_other, at_serve)

    def test_a_segment_a_kept_tensor_holds_is_the_engine_s_past_its_pool(self):
        self.memory.arena = 256 * MIB
        self.memory.serve()
        self.memory.begin_step(64)
        self.segment(64)                         # a workspace a plugin makes at its first call and keeps
        self.segment(32)
        self.memory.end_step(False)              # the pool retires; torch frees what nothing holds
        self.free_segment(32)
        steps = self.memory.steps()
        self.assertEqual((steps["granted"], steps["retired"]), (64 * MIB, 64 * MIB))
        self.assertEqual(self.memory.kept(), 256 * MIB + 256 * MIB)   # the next pool starts with the whole arena
        self.free_segment(64)                    # the workspace goes too
        self.assertEqual((self.memory.steps()["granted"], self.memory.steps()["retired"]), (0, 0))

    def test_a_long_prompt_keeps_its_pool_while_another_chunk_fits_the_arena(self):
        # torch reuses the pool's cached blocks for a like chunk and a deeper chunk adds segments of its own: the pool
        # retires once the next chunk's most would pass the arena, and when the prompt ends in any case.
        self.memory.arena = 512 * MIB
        self.memory.serve()
        self.memory.begin_step(512)
        self.segment(200)
        self.memory.end_step(True)               # 200 and another 200 fit 512: kept
        self.assertEqual(self.memory.steps()["retired"], 0)
        self.memory.begin_step(512)
        self.segment(150)                        # a deeper chunk's larger blocks
        self.memory.end_step(True)               # 350 and another 200 pass 512: retired
        self.assertEqual(self.memory.steps()["retired"], 350 * MIB)
        self.free_segment(200)
        self.free_segment(150)
        self.memory.begin_step(512)
        self.segment(100)
        self.memory.end_step(False)              # the prompt ended: retired whatever it holds
        self.assertEqual(self.memory.steps()["retired"], 100 * MIB)
        self.assertEqual(self.memory.steps()["overflows"], 0)

    def test_a_measured_step_that_runs_out_has_the_givers_make_room_and_runs_again(self):
        # While the engine loads the givers keep only the headroom free (an expert cache opened before a model's
        # calibration takes the rest); the allocator cannot ask them for a segment, so a calibration chunk whose
        # segment it refused (torch's out-of-memory error) has them release room, and the chunk runs again.
        self.cache.regrow(self.readings.bytes - 256 * MIB)   # the cache took every byte past the headroom
        runs = []

        def chunk():
            runs.append(len(runs))
            if not self.segment(300):
                raise RuntimeError("CUDA out of memory. Tried to allocate 300.00 MiB")
        held = self.cache.held
        step = self.memory.measure(chunk)
        self.assertEqual((len(runs), step.took), (2, 300 * MIB))
        self.assertEqual(held - self.cache.held, 600 * MIB)   # twice the refused segment, given back
        self.assertEqual(self.memory.most["prompt"], 300 * MIB)
        self.free_segment(300)
        with self.assertRaisesRegex(ValueError, "not memory"):   # any other failure is the step's own
            self.memory.measure(lambda: (_ for _ in ()).throw(ValueError("not memory")))
        self.cache.regrow(self.readings.bytes - 256 * MIB)
        self.cache.held = 0          # a giver with nothing left to give: the refusal is raised
        with self.assertRaisesRegex(RuntimeError, "out of memory"):
            self.memory.measure(chunk)

    def test_a_device_without_core_s_allocator_runs_its_steps_on_torch_s_and_says_so(self):
        bare = memory.DeviceMemory("fake:1", StepReadings(free=1024))
        with bare.step_pool():   # fake readings: no CUDA, no pool
            pass
        self.assertIsNone(bare._pool)


@unittest.skipIf(torch is None or not torch.cuda.is_available(), "no CUDA device")
class DeviceTests(unittest.TestCase):
    def test_a_step_s_tensors_come_from_core_s_allocator_and_go_back_with_its_pool(self):
        # Core's step engine runs each step in the device's step pool: its segments are admitted before they exist,
        # counted as the step's, and handed back to the driver when the pool retires (no prompt in flight).
        import gc
        device = memory.of("cuda")
        before = device.steps()["granted"]
        device.begin_step(16)
        with device.step_pool():
            made = torch.empty(64 << 20, dtype=torch.uint8, device="cuda")
            made.fill_(1)
        if device.steps().get("unaccounted"):
            self.skipTest(device.steps()["unaccounted"])
        self.assertGreaterEqual(device.steps()["granted"] - before, 64 << 20)
        del made
        device.end_step(False)
        self.assertGreaterEqual(device.most.get("prompt", 0), 64 << 20)   # what the step took, kept by its kind
        gc.collect()
        torch.cuda.synchronize()
        self.assertEqual(device.steps()["granted"], before)

    def test_a_step_a_plugin_measures_at_load_is_counted_as_a_step_of_its_kind(self):
        # A calibration's prompt chunk at the deepest position: no warm request reaches it, so the plugin runs it as
        # Core's measured step and the arena holds what it took from Core's allocator.
        device = memory.of("cuda")
        with device.measured("prompt") as step:
            made = torch.empty(48 << 20, dtype=torch.uint8, device="cuda")
            made.fill_(1)
            del made
        if device.steps().get("unaccounted"):
            self.skipTest(device.steps()["unaccounted"])
        self.assertGreaterEqual(step.took, 48 << 20)
        self.assertGreaterEqual(device.most["prompt"], step.took)
        device.begin_step(16)
        try:
            with self.assertRaises(RuntimeError):   # never inside another step
                with device.measured():
                    pass
        finally:
            device.end_step(False)

    def test_a_thread_reading_torch_s_allocator_never_waits_on_a_step_s_new_segments(self):
        # Core's allocator runs inside torch's allocation, under torch's allocator lock, on a step thread that let the
        # GIL go: it calls no Python, so a thread holding the GIL and reading torch's allocator (memory_stats, as a
        # telemetry read does) while the step makes segments is never stuck behind it.
        import threading
        device = memory.of("cuda")
        stop, done = threading.Event(), []

        def steps():
            for i in range(64):
                device.begin_step(16)
                with device.step_pool():
                    made = torch.empty((1 + i % 5) << 22, dtype=torch.uint8, device="cuda")   # a new segment
                    made.fill_(1)
                    del made
                device.end_step(False)   # the pool retires: the next step's segments are new
            done.append(True)

        def read():
            while not stop.is_set():
                torch.cuda.memory_stats()

        reader, stepper = threading.Thread(target=read, daemon=True), threading.Thread(target=steps, daemon=True)
        reader.start()
        stepper.start()
        stepper.join(120)
        stop.set()
        reader.join(10)
        if device.steps().get("unaccounted"):
            self.skipTest(device.steps()["unaccounted"])
        self.assertEqual(done, [True], "a step making segments and a thread reading torch's allocator deadlocked")
        self.assertFalse(reader.is_alive())

    def test_the_card_counts_at_least_what_this_process_holds(self):
        held = torch.zeros(8 << 20, dtype=torch.uint8, device="cuda")
        left = memory.card_free(0)
        if left is None:
            self.skipTest("the card's management library is not reachable here")
        self.assertLessEqual(left, memory.of("cuda").total - torch.cuda.memory_reserved())
        self.assertLessEqual(memory.of("cuda").free(), left + (64 << 20))   # read a moment apart
        del held

    def setUp(self):
        self.memory = memory.of("cuda")
        if not self.memory.elastic_ranges:
            self.skipTest("the driver's virtual memory API is not reachable here")
        self.gran = memory.granularity(torch.cuda.current_device())

    def test_one_object_a_device_reads_it(self):
        self.assertIs(self.memory, memory.of(torch.device("cuda", torch.cuda.current_device())))
        self.assertGreater(self.memory.total, self.memory.free())
        held = torch.zeros(8 << 20, dtype=torch.uint8, device="cuda")
        before = self.memory.torch()
        del held
        after = self.memory.torch()
        self.assertEqual(before["allocated"] - after["allocated"], 8 << 20)
        self.assertGreaterEqual(after["unused"], 8 << 20)          # the block is torch's, unused
        self.assertGreaterEqual(self.memory.release(), 8 << 20)    # and goes back to the driver
        made = self.memory.made()                                  # allocations are counted
        held = torch.zeros(1 << 20, dtype=torch.uint8, device="cuda")
        self.assertEqual(self.memory.made() - made, 1)
        held.fill_(1)                                              # ... and work in a tensor that exists makes none
        self.assertEqual(self.memory.made() - made, 1)
        self.assertEqual(self.memory.release(below=0), 0)

    def test_a_ballast_takes_a_share_of_the_device_away_from_every_reading(self):
        # A test's ballast: an elastic range of the driver's memory, outside torch's allocator, that every reading
        # of free memory counts as taken until it is closed. leave: all but that much of what is free.
        gran, before = self.gran, self.memory.free(release=True)
        leave = before // 2 // gran * gran
        held = self.memory.ballast(leave=leave)
        try:
            self.assertIn("ballast/test", self.memory.report()["elastic"])
            self.assertLess(abs(self.memory.free() - leave), (256 << 20))
            left = memory.card_free(torch.cuda.current_device())
            if left is not None:
                self.assertLess(left, leave + (256 << 20))
        finally:
            held.close()
        self.assertGreater(self.memory.free(), before - (256 << 20))
        with self.assertRaisesRegex(ValueError, "leave:SIZE or take:SIZE"):
            self.memory._ballast_from("half")
        held = self.memory._ballast_from("take:0.5GiB")
        self.assertGreaterEqual(held.mapped, (1 << 29) - gran)
        held.close()

    def test_a_range_grows_and_shrinks_with_its_addresses_and_contents_in_place(self):
        gran = self.gran
        r = self.memory.elastic("test", "range", 16 * gran, initial=1)
        self.addCleanup(r.close)
        self.assertEqual(self.memory.report()["elastic"]["test/range"], {"backed": gran, "reserved": 16 * gran})
        other = self.memory.elastic("test", "range", gran)          # a second of the name is numbered
        self.assertIn("test/range#2", self.memory.ranges)
        other.close()
        self.assertNotIn("test/range#2", self.memory.ranges)
        base = r.tensor.data_ptr()
        self.assertEqual(r.mapped, gran)                       # the first granule: a tensor over the range needs it
        self.assertEqual(r.grow(3 * gran + 1), 4 * gran)        # whole granules
        r.tensor[:4 * gran].copy_(torch.arange(4 * gran, device="cuda").to(torch.uint8))
        self.assertEqual(r.shrink(2 * gran), 2 * gran)
        self.assertEqual(r.grow(6 * gran), 6 * gran)
        self.assertEqual(r.tensor.data_ptr(), base)             # every address stays
        kept = r.tensor[:2 * gran].cpu()
        self.assertTrue(torch.equal(kept, torch.arange(2 * gran).to(torch.uint8)))   # what stayed backed kept its bytes
        self.assertEqual(int(r.tensor[2 * gran:6 * gran].max()), 0)                   # new backing comes zeroed
        self.assertEqual(r.grow(64 * gran), 16 * gran)          # never past the range
        r.close()
        self.assertEqual(r.mapped, 0)
        self.assertNotIn("test/range", self.memory.ranges)      # a closed range leaves the census

    def test_a_range_is_backed_in_places_and_released_by_whole_granules(self):
        gran = self.gran
        r = self.memory.elastic("test", "places", 32 * gran)
        self.addCleanup(r.close)
        self.assertEqual(r.back(10 * gran + 5, 12 * gran + 1), 3 * gran)     # the granules holding those bytes
        self.assertEqual((r.mapped, r.prefix), (4 * gran, gran))             # the first granule and three far from it
        r.tensor[10 * gran:13 * gran].fill_(7)
        self.assertEqual(r.back(11 * gran, 14 * gran), gran)                 # only what was not backed
        self.assertEqual(int(r.tensor[10 * gran:13 * gran].min()), 7)        # what was backed keeps its bytes
        self.assertEqual(int(r.tensor[13 * gran:14 * gran].max()), 0)
        self.assertEqual(r.release(10 * gran + 1, 13 * gran), 2 * gran)      # granules wholly inside, not the first
        self.assertEqual(int(r.tensor[10 * gran:11 * gran].min()), 7)
        self.assertEqual(r.release(0, 4 * gran), 0)                          # the range's first granule stays
        self.assertEqual(r.mapped, 3 * gran)

    def test_a_growing_buffer_holds_what_its_largest_user_asked_for_at_one_address(self):
        gran = self.gran
        g = self.memory.growing("test", "rows", 8 * gran)
        self.addCleanup(g.close)
        small = g.take(100)
        small.fill_(3)
        self.assertEqual((g.bytes, small.numel()), (gran, 100))
        large = g.take(2 * gran + 1)                                  # grown: the same address, the bytes kept
        self.assertEqual((g.bytes, large.data_ptr(), int(large[:100].min())), (3 * gran, small.data_ptr(), 3))
        self.assertEqual(g.take(50).data_ptr(), small.data_ptr())     # a smaller user reads the same rows
        self.assertEqual(g.bytes, 3 * gran)
        with self.assertRaises(ValueError):
            g.take(8 * gran + 1)
        self.assertIn("test/rows", self.memory.report()["elastic"])

    def test_a_scratch_is_one_buffer_that_every_call_takes_from_its_start(self):
        gran = self.gran
        scratch = self.memory.scratch("test", poison=False)
        self.addCleanup(scratch.close)
        first = scratch.take(100)                                  # one call's bytes
        first.fill_(3)
        second = scratch.take(3 * gran)                            # another call's: the same bytes, the buffer grown
        self.assertEqual((second.data_ptr(), second.numel(), scratch.bytes), (first.data_ptr(), 3 * gran, 3 * gran))
        y, h = scratch.take(100, 4 * gran)                         # a call's buffers together, each 64-byte aligned
        self.assertEqual((y.data_ptr(), y.numel(), h.data_ptr(), h.numel()),
                         (first.data_ptr(), 100, first.data_ptr() + 128, 4 * gran))
        self.assertEqual((scratch.bytes, scratch.most), (5 * gran, 128 + 4 * gran))
        self.assertEqual(self.memory.report()["elastic"]["test/scratch"]["backed"], 5 * gran)
        graph, stream = torch.cuda.CUDAGraph(), torch.cuda.Stream()
        with torch.cuda.graph(graph, stream=stream):
            scratch.take(2 * gran).fill_(5)                        # a captured call takes what a run before it took
            with self.assertRaises(RuntimeError):                  # and none grows the scratch
                scratch.take(6 * gran)
        first.fill_(0)
        graph.replay()
        self.assertEqual(int(first.min()), 5)
        with self.assertRaises(ValueError):
            scratch.take(memory.SCRATCH_BYTES + 1)

    def test_a_poisoned_scratch_hands_every_call_bytes_no_call_before_it_wrote(self):
        from unittest import mock
        with mock.patch.dict("os.environ", {memory.POISON: "1"}):
            scratch = self.memory.scratch("test")                  # as the plugin tests run
        self.addCleanup(scratch.close)
        scratch.take(64).fill_(7)
        taken = scratch.take(64)
        self.assertEqual((int(taken.min()), int(taken.view(torch.int32)[0]), bool(taken.view(torch.float32).isnan().all())),
                         (255, -1, True))                          # what the call before wrote is gone
        graph, stream = torch.cuda.CUDAGraph(), torch.cuda.Stream()
        with torch.cuda.graph(graph, stream=stream):               # a captured call is poisoned at every replay
            scratch.take(64)[:8].zero_()
        taken.fill_(7)
        graph.replay()
        self.assertEqual(taken.tolist(), [0] * 8 + [255] * 56)
        self.assertFalse(self.memory.scratch("test").poison)       # only where the environment asks for it

    def test_a_paged_range_backs_its_planes_units_as_they_come_and_refuses_what_the_device_lacks(self):
        from unittest import mock
        gran = self.gran
        planes = [(0, gran, 4), (8 * gran, gran // 2, 6)]        # 4 units of a granule, 6 of half a granule
        p = self.memory.paged("test", "paged", 16 * gran, planes, kept=[(15 * gran, 16 * gran)])
        self.addCleanup(p.close)
        self.assertEqual((p.units, p.backed, p.bytes), (6, 0, 2 * gran))   # the range's first granule and the kept one
        self.assertTrue(p.back(2))
        self.assertEqual((p.backed, p.bytes), (2, 4 * gran))               # granules 0, 1 and 8 beside the kept one
        p.tensor[gran:2 * gran].fill_(5)
        self.assertTrue(p.back(6))                                         # the shorter plane stops at its 4 units
        self.assertEqual((p.backed, p.bytes), (6, 8 * gran))
        self.assertEqual(int(p.tensor[gran:2 * gran].min()), 5)            # what was backed keeps its bytes
        self.assertEqual(p.trim(1), 5 * gran)                              # whole granules past unit 1 of each plane
        self.assertEqual((p.backed, p.bytes), (1, 3 * gran))
        with mock.patch.object(self.memory.readings, "free", lambda: 0):   # a full device, nobody to give
            self.assertFalse(p.back(3))
        self.assertEqual((p.backed, p.bytes), (1, 3 * gran))               # nothing was backed for it
        self.assertIn("test/paged", self.memory.report()["elastic"])

    def test_a_paged_range_asks_the_ledger_for_the_granules_it_will_map(self):
        # A unit smaller than a granule backs a whole granule: the givers are asked for that, or the device's free
        # memory falls under what they keep by a granule a plane (KV pools of many layers: megabytes a step).
        from unittest import mock
        gran = self.gran
        planes = [(0, gran // 4, 8), (4 * gran, gran // 4, 8), (8 * gran, 0, 8)]   # two planes of quarter units
        p = self.memory.paged("test", "granules", 12 * gran, planes)
        self.addCleanup(p.close)
        asked = []
        with mock.patch.object(self.memory, "need", asked.append):
            self.assertTrue(p.back(1))      # granule 0 is the range's first, backed already: granule 4 is new
            self.assertTrue(p.back(3))      # within the granules backed: nothing new
            self.assertTrue(p.back(5))      # a granule more in each plane
        self.assertEqual(asked, [gran, 0, 2 * gran])
        self.assertEqual(p.bytes, 4 * gran)

    def test_units_the_device_has_no_memory_for_are_backed_from_host_memory_and_read_through_the_range(self):
        from unittest import mock
        gran = self.gran
        p = self.memory.paged("test", "hosted", 16 * gran, [(0, gran, 8)], host_bytes=3 * gran)
        self.addCleanup(p.close)
        self.assertTrue(p.back(2))
        p.tensor[:2 * gran].fill_(3)
        device_bytes = p.bytes
        with mock.patch.object(self.memory.readings, "free", lambda: 0):   # a full device, nobody to give
            self.assertFalse(p.back(4))
            self.assertTrue(p.back(4, host=True))                          # two units from host memory
            self.assertFalse(p.back(8, host=True))                         # four more are past the range's host bytes
        self.assertEqual((p.backed, p.bytes, p.range.host_mapped), (4, device_bytes, 2 * gran))
        hosted = p.tensor[2 * gran:4 * gran]
        self.assertEqual(int(hosted.max()), 0)                             # backed zeroed, as device units are
        values = torch.arange(2 * gran, device="cuda", dtype=torch.int64).to(torch.uint8)
        hosted.copy_(values)                                               # kernels write and read it in place
        self.assertTrue(torch.equal(p.tensor[2 * gran:4 * gran], values))
        self.assertEqual(int(p.tensor[:2 * gran].min()), 3)                # the device's units beside it kept theirs
        self.assertEqual(int(p.tensor[:4 * gran].long().sum()), 3 * 2 * gran + int(values.long().sum()))
        self.assertEqual(self.memory.report()["elastic"]["test/hosted"]["host"], 2 * gran)
        self.assertTrue(p.back(5))                                         # the device has memory again: units go there
        self.assertEqual((p.range.host_mapped, p.bytes), (2 * gran, device_bytes + gran))
        self.assertEqual(p.trim(2), 3 * gran)                              # host and device units go back alike
        self.assertEqual((p.range.host_mapped, p.bytes), (0, device_bytes))

    def test_the_census_counts_a_tensor_for_the_plugin_whose_object_holds_it(self):
        def plugin_class(package, name):   # a class as a plugin's package defines it (crit_plugins.assemble's names)
            return type(name, (), {"__module__": f"crit_plugin_{package}.plugin"})
        Matrix = plugin_class("projections_0a", "Matrix")
        projections, model = plugin_class("projections_0a", "Projections")(), plugin_class("model_0b", "Model")()
        matrix = Matrix()
        matrix.tensors = {"weight": torch.zeros(6 << 20, dtype=torch.uint8, device="cuda")}
        projections.buffers = [torch.zeros(2 << 20, dtype=torch.uint8, device="cuda")]
        model.blocks = [{"q": matrix}]                            # made by the projections plugin, kept by the model
        behind = Matrix()                                         # another, behind the loader's counting proxy
        behind.tensors = {"weight": torch.zeros(1 << 20, dtype=torch.uint8, device="cuda")}
        model.blocks.append({"k": Observed(behind, "projections", Calls())})
        model.embed = torch.zeros(3 << 20, dtype=torch.uint8, device="cuda")
        model.view = model.embed[:16]                             # a second view of one storage counts once
        model.namespace = types.SimpleNamespace(state=torch.zeros(1 << 20, dtype=torch.uint8, device="cuda"))
        model.host = torch.zeros(1 << 20, dtype=torch.uint8)      # not on the device
        r = self.memory.elastic("test", "census", self.gran)
        self.addCleanup(r.close)
        model.range = r.tensor                                    # an elastic range is the census's own entry
        stray = torch.zeros(4 << 20, dtype=torch.uint8, device="cuda")   # held by no plugin object
        graph, stream = torch.cuda.CUDAGraph(), torch.cuda.Stream()
        with torch.cuda.graph(graph, stream=stream):                     # a capture's pool holds its temporaries
            total = (stray.float() * 2).sum()
            model.captured = torch.zeros(2 << 20, dtype=torch.uint8, device="cuda")   # made in it, kept by a plugin
        found = self.memory.census({"projections": projections, "the-model": model})
        self.assertEqual(found["plugins"], {"projections": 9 << 20, "the-model": 6 << 20})
        self.assertEqual(found["tensors"][:2],                    # the largest storages, each by its shape
                         [{"plugin": "projections", "bytes": 6 << 20, "shape": [6 << 20], "dtype": "uint8"},
                          {"plugin": "the-model", "bytes": 3 << 20, "shape": [3 << 20], "dtype": "uint8"}])
        self.assertGreaterEqual(found["graphs"], 16 << 20)        # the 4 MiB as FP32: its room is the graph's pool's
        self.assertGreaterEqual(found["unattributed"], 4 << 20)
        del total
        report = self.memory.report()
        self.assertEqual(report["plugins"], found["plugins"])
        self.assertEqual(report["elastic"]["test/census"]["backed"], self.gran)
        backed = sum(e["backed"] for e in report["elastic"].values())   # this range, and any an earlier test left open
        self.assertEqual(report["total"], report["free"] + report["torch"]["reserved"] + backed + report["other"])
        del stray



class KvFitTests(unittest.TestCase):
    """memory.kv_fit, the one arithmetic of the KV cache that fits the device (crit Eval's max context calls it)."""

    def test_the_budget_is_free_memory_less_headroom_reserve_and_what_stays_in_order(self):
        found = memory.kv_fit(10_000, 7, headroom=1_000, reserve=500, held=(2_000, 0.5, 100), granularity=16)
        self.assertEqual(found["budget"], 10_000 - 1_000 - 500 - 2_000 - 0.5 - 100)
        self.assertEqual(found["tokens"], (10_000 - 1_000 - 500 - 2_000 - 0.5 - 100) // 7 // 16 * 16)
        self.assertEqual(memory.kv_fit(10_000, 0)["tokens"], None)   # nothing grows with the context
        self.assertEqual(memory.kv_fit(100, 1, headroom=200), {"budget": -100, "tokens": 0})
        self.assertEqual(memory.kv_fit(memory.HEADROOM + 4096, 2)["tokens"], 2048)   # Core's headroom by default


class ProcessTensorTests(unittest.TestCase):
    def test_a_name_the_process_keeps_is_one_tensor_whatever_copy_of_a_plugin_asks(self):
        # every engine's package imports its own copy of a plugin's code; each copy's module cache asks for the
        # process's buffers again, and gets the one kept, not a second tensor that would outlive every engine
        import torch
        device = memory.DeviceMemory("fake:0", Readings(1000))
        first = device.process("plugin", "table", torch.zeros(4))
        self.assertIs(device.process("plugin", "table", torch.ones(4)), first)
        made = []
        self.assertIs(device.process("plugin", "table", lambda: made.append(1) or torch.ones(4)), first)
        self.assertEqual(made, [])
        other = device.process("plugin", "table", torch.zeros(8))   # another shape: the new one is kept
        self.assertIsNot(other, first)
        self.assertIs(device.process_tensors["plugin/table"], other)
        self.assertEqual(tuple(device.process("plugin", "fresh", lambda: torch.zeros(2)).shape), (2,))


class KvNeedTests(unittest.TestCase):
    def test_the_need_at_a_context_is_what_kv_fit_fits_that_context_in(self):
        held = (2_000, 100)
        need = memory.kv_need(7, 4096, headroom=1_000, reserve=500, held=held)
        self.assertEqual(need, 1_000 + 500 + 2_000 + 100 + 7 * 4096)
        self.assertEqual(memory.kv_fit(need, 7, headroom=1_000, reserve=500, held=held)["tokens"], 4096)


class StepPoolNameTests(unittest.TestCase):
    def test_step_pools_are_named_and_the_names_of_those_holding_nothing_go(self):
        # Core's step pools are listed as core/steps among the device's pools, as a graph's pool is listed by its
        # driver, so the memory audit names their segments. Every STEP_POOLS_NAMED new pools the names of those the
        # allocator holds no segment of go: a pool that still holds one (the first step's, which keeps a workspace)
        # and the newest stay named.
        class Held(Readings):
            def __init__(self):
                super().__init__(free=1000)
                self.held = set()

            def segments(self):   # (first, end, allocated, private, pool) as _Torch.segments() gives them
                return [(0, 1, 1, True, pool) for pool in sorted(self.held)]
        readings = Held()
        device = memory.DeviceMemory("fake:0", readings)
        first, newest = (0, 1), (0, memory.STEP_POOLS_NAMED + 1)
        readings.held.add(first)
        for n in range(1, memory.STEP_POOLS_NAMED + 1):
            device._name_step_pool((0, n))
        self.assertEqual(len(device.pools), memory.STEP_POOLS_NAMED)   # none gone yet
        self.assertEqual(device.pools[(0, 2)], "core/steps")
        device._name_step_pool(newest)
        self.assertEqual(device.pools, {first: "core/steps", newest: "core/steps"})


if __name__ == "__main__":
    unittest.main()
