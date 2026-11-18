"""crit_plugins.memory: one object a device. Its ledger hands out what the device lacks, leaves free what the givers
keep and lets them grow back once the engine serves; an elastic range grows and shrinks with its addresses and its
contents in place; the census counts torch's bytes for the plugin whose objects hold them (CUDA)."""
import sys
import types
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "core" / "python"))

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
        self.assertEqual(self.memory.report(), {
            "device": "fake:0", "total": 10_000, "free": 1000, "elastic": {},
            "torch": {"allocated": 2000, "reserved": 2300, "unused": 300}, "other": 6700})

    def test_a_device_whose_memory_cannot_be_read_takes_and_gives_nothing(self):
        cpu = memory.of("cpu")
        self.assertIs(cpu, memory.of("cpu"))
        self.assertEqual((cpu.total, cpu.free(), cpu.torch(), cpu.report(), cpu.census({})), (None,) * 5)
        self.assertEqual((cpu.made(), cpu.pool("rounds", "1 rows")), (None, None))
        self.assertEqual((cpu.need(1 << 40), cpu.serve(), cpu.release(), cpu.elastic_ranges), (0, 0, None, False))


@unittest.skipIf(torch is None or not torch.cuda.is_available(), "no CUDA device")
class DeviceTests(unittest.TestCase):
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
        self.assertEqual(report["total"], report["free"] + report["torch"]["reserved"] + self.gran + report["other"])
        del stray


if __name__ == "__main__":
    unittest.main()
