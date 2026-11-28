"""A closed engine proves it returned its memory (crit_plugins.lifetime): Core's close drops every reference it holds,
collects, empties torch's cache and measures the device against its level before the load. A close that leaves an
elastic range, a giver, a captured graph or torch's bytes behind fails with a report naming what holds it, and the
next load on the device refuses to start beside it. Fake device readings (as tests/test_memory.py's) count the bytes
of the objects a test says are on the device for as long as they live."""
import gc
import json
import os
import sys
import tempfile
import unittest
import weakref
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
for _folder in ("inference/core/python", "tests"):
    sys.path.insert(0, str(ROOT / _folder))

from core_text_fixtures import llama_files, write_files  # noqa: E402
from crit_plugins import lifetime, memory  # noqa: E402
from support.engines import STEP_LIBRARY, write_package  # noqa: E402

try:
    import torch
except ImportError:
    torch = None

GIB = 1 << 30
KEPT = []   # a plugin module's cache across engines: what keeps a closed engine's objects alive in these tests


class Device:
    """Readings of a device whose torch allocator holds `base` bytes and the bytes of every object hold() put there,
    for as long as the object lives; `cached` bytes are blocks nothing uses, which release() hands back."""
    total = 1 << 40

    def __init__(self, base=64 << 20):
        self.base, self.cached, self.live, self.segs = base, 0, {}, []
        self.segmented = 0   # reserved bytes of segments tensors keep, beyond the tensors (no release frees them)
        self.released = self.cleared = 0

    def hold(self, found, nbytes):
        key = id(found)
        self.live[key] = int(nbytes)
        weakref.finalize(found, self.live.pop, key, None)
        return found

    def free(self):
        return self.total - self.torch()["reserved"]

    def torch(self):
        allocated = self.base + sum(self.live.values())
        return {"allocated": allocated, "reserved": allocated + self.cached + self.segmented, "unused": self.cached}

    def release(self):
        self.cached, self.released = 0, self.released + 1

    def clear_workspaces(self):
        self.cleared += 1

    def segments(self):
        return list(self.segs)

    def mark(self):
        pass

    def peak(self):
        return None

    def made(self):
        return None

    def pool(self):
        return None

    def blocks(self):
        return []


class Range:
    """An elastic range without a driver (crit_plugins.memory.ElasticRange's surface): its backed bytes are the
    device's while it is open."""

    def __init__(self, max_bytes, device="cpu", initial=0):
        self.size, self.mapped, self.gran, self.host_mapped = int(max_bytes), int(initial), 4096, 0
        self.tensor = torch.zeros(1 << 16, dtype=torch.uint8) if torch is not None else None
        self.closed = None

    @property
    def prefix(self):
        return self.mapped

    def back(self, first, end, strict=True, host=False):
        self.mapped = max(self.mapped, int(end))

    def unbacked(self, spans):
        return sum(max(0, int(end) - max(int(first), self.mapped)) for first, end in spans)

    def close(self):
        closed, self.closed = self.closed, None
        if closed is not None:
            closed()


class Graph:
    """A captured graph: its pool's bytes are the device's while something holds it."""


class Model:
    pass


class Drafter:
    def __init__(self, model):
        self.model = model


class Rounds:
    """A round driver: its captured graphs and the model they run (crit_plugins.rounds' SpeculativeRounds)."""
    runs = (0, 0, 0)

    def __init__(self, graphs, model):
        self.graphs, self.model = graphs, model

    def close(self):
        pass


class LifetimeCase(unittest.TestCase):
    def setUp(self):
        self.device = Device()
        self.memory = memory.DeviceMemory("cuda:0", self.device)
        KEPT.clear()
        self.addCleanup(KEPT.clear)

    def opened(self):
        found = lifetime.open(self.memory)
        self.addCleanup(lambda: found in self.memory.opened and self.memory.opened.remove(found))
        return found

    def close(self, opened):
        with mock.patch.dict(os.environ, {lifetime.CHECK: "raise"}):
            return opened.close()

    def leak(self, opened):
        with mock.patch.dict(os.environ, {lifetime.CHECK: "raise"}), self.assertRaises(lifetime.LeakingClose) as raised:
            opened.close()
        return raised.exception.report


class CloseTests(LifetimeCase):
    def test_a_close_that_drops_everything_returns_its_memory(self):
        opened = self.opened()
        model = self.device.hold(Model(), 2 * GIB)
        graph = self.memory.captured(self.device.hold(Graph(), GIB), self.memory.pool("rounds", "every round", (1, 1)))
        self.memory.give("expert-cache", release=lambda n: 0)
        opened.watch({"model": model})
        self.device.cached = GIB   # torch's cache of the engine's temporaries: the close empties it
        self.memory.ungive("expert-cache")
        del model, graph
        report = self.close(opened)
        self.assertTrue(report["returned"], report["message"])
        self.assertEqual(report["message"], "returned")
        self.assertEqual((report["ranges"], report["givers"], report["graphs"], report["objects"]), ([], [], [], []))
        self.assertEqual(report["torch"]["grown"], {"allocated": 0, "reserved": 0})
        self.assertGreater(self.device.released, 0)
        self.assertGreater(self.device.cleared, 0)   # torch's cuBLAS workspaces are freed, not tolerated
        self.assertIsNone(self.memory.leaking)
        lifetime.open(self.memory).close()   # the next load starts

    def test_a_plugin_that_keeps_the_model_alive_fails_naming_the_chain_that_holds_it(self):
        # The drafter case (11-25): a drafter's head kept the model alive past the close.
        opened = self.opened()
        model = self.device.hold(Model(), 3 * GIB)
        drafter = Drafter(model)
        KEPT.append(drafter)
        opened.watch({"model": model, "toy-drafter": drafter})
        del model, drafter
        report = self.leak(opened)
        self.assertFalse(report["returned"])
        self.assertEqual(report["torch"]["grown"]["allocated"], 3 * GIB)
        held = {found["name"]: found["held_by"] for found in report["objects"]}
        self.assertIn(f"module {__name__} global KEPT -> list[0] -> Drafter.model -> Model", held["model"])
        self.assertIn("torch holds 3.00 GiB allocated", report["message"])
        self.assertIn("plugin object model (Model) alive, held by module", report["message"])
        self.assertIs(self.memory.leaking, opened)

    def test_a_scratch_a_plugin_made_for_itself_outlives_the_engine_and_the_reopen_refuses(self):
        # The 097f97e0 case (11-23): token-constraints made a scratch of its own, which no close closed: the
        # reopened engine found token-sampling/scratch still held.
        if torch is None:
            self.skipTest("torch is not installed")
        with mock.patch.object(memory, "ElasticRange", Range), \
                mock.patch.object(memory.DeviceMemory, "elastic_ranges", property(lambda self: True)):
            opened = self.opened()
            engines = self.memory.scratch()                   # the engine's: Core closes it with the engine
            own = self.memory.scratch("token-sampling")       # the plugin's own
            engines.take(4096), own.take(8192)
            engines.close()
            report = self.leak(opened)
            self.assertEqual(report["ranges"], [{"name": "token-sampling/scratch", "backed": 8192,
                                                 "reserved": memory.SCRATCH_BYTES}])
            self.assertIn("elastic range token-sampling/scratch is open", report["message"])
            with self.assertRaisesRegex(lifetime.LeakingClose, "no engine loads on cuda:0 beside the last engine's "
                                        "close.*token-sampling/scratch"):
                lifetime.open(self.memory)
            own.close()   # the holder lets go: the next load measures again and starts
            lifetime.open(self.memory).close()
            self.assertIsNone(self.memory.leaking)

    def test_an_elastic_range_and_a_giver_left_behind_fail(self):
        # The 11-21 case: an elastic range outlived its engine, and its giver stayed listed on the device.
        with mock.patch.object(memory, "ElasticRange", Range):
            opened = self.opened()
            pages = self.memory.paged("attention-state", "pages", 4 * GIB, [(0, GIB, 4)])
            pages.back(2)
            self.memory.give("attention-state", release=lambda n: 0)
            report = self.leak(opened)
        self.assertEqual([found["name"] for found in report["ranges"]], ["attention-state/pages"])
        self.assertEqual(report["givers"], ["attention-state"])
        self.assertIn("giver(s) still listed on the device: attention-state", report["message"])
        self.assertEqual(report["torch"]["grown"], {"allocated": 0, "reserved": 0})   # an elastic range is not torch's

    def test_ranges_and_graphs_made_before_the_load_are_not_the_engines(self):
        with mock.patch.object(memory, "ElasticRange", Range):
            earlier = self.memory.elastic("expert-offload", "slots", GIB)
            ballast = self.memory.elastic("ballast", "test", GIB)
            graph = self.memory.captured(Graph(), self.memory.pool("prompts", "every bucket", (2, 2)))
            opened = self.opened()
            ballast_late = self.memory.elastic("ballast", "test", GIB)   # a test's share of the device: the process's
            report = self.close(opened)
        self.assertTrue(report["returned"], report["message"])
        del earlier, ballast, graph, ballast_late

    def test_the_process_state_is_named_apart_and_not_counted(self):
        if torch is None:
            self.skipTest("torch is not installed")
        opened = self.opened()
        # A plugin's constant table the process keeps for every engine, in a segment of its own, and the capture
        # stream's cuBLAS workspaces in the process's pool: made by this engine's load, kept past its close.
        table = self.device.hold(torch.zeros(1024, dtype=torch.float32), 4096)
        self.memory.process("quantized-projections", "e2m1", table)
        at = table.untyped_storage().data_ptr()
        workspaces = (9, 9)
        self.memory.pool("captures", "matrix workspaces", workspaces, process=True)
        workspace = self.device.hold(Graph(), 32 << 20)   # the workspace block itself
        self.device.segs = [(at, at + (2 << 20), 4096, False, (0, 0)),
                            (1 << 40, (1 << 40) + (32 << 20), 32 << 20, True, workspaces)]
        self.device.segmented = (2 << 20) - 4096   # the table's segment: torch reserves it whole
        report = self.close(opened)
        self.assertTrue(report["returned"], report["message"])
        self.assertEqual(report["torch"]["excluded"]["by"],
                         {"captures/matrix workspaces": 32 << 20, "quantized-projections/e2m1": 4096})
        self.assertEqual(self.memory.report()["process"], ["quantized-projections/e2m1"])
        del workspace, table

    def test_a_hosts_device_lease_is_set_aside_by_its_bytes_and_counts_once_unlisted(self):
        # crit serve's model pool keeps a device lease's allocation past the engine that asked for it (the model's,
        # until it unloads): listed with memory.lease, a closed engine is measured without its bytes.
        opened = self.opened()
        allocation = self.device.hold(Model(), 3 << 20)
        self.memory.lease("model-pool", "weights", 3 << 20)
        report = self.close(opened)
        self.assertTrue(report["returned"], report["message"])
        self.assertEqual(report["torch"]["excluded"]["by"], {"model-pool/weights": 3 << 20})
        self.assertEqual(self.memory.report()["leased"], {"model-pool/weights": 3 << 20})
        self.memory.unlease("model-pool", "weights")
        opened = self.opened()   # a lease made after the load and not listed is the engine's
        late = self.device.hold(Model(), 3 << 20)
        report = self.leak(opened)
        self.assertFalse(report["returned"])
        del allocation, late

    def test_a_close_beside_another_open_engine_is_not_judged_and_hands_its_objects_on(self):
        first = self.opened()
        second = self.opened()
        model = self.device.hold(Model(), GIB)
        KEPT.append(model)
        second.watch({"second-model": model})
        del model
        report = self.close(second)
        self.assertFalse(report["judged"])
        self.assertIsNone(report["returned"])
        self.assertIsNone(self.memory.leaking)
        report = self.leak(first)   # the engine opened first, judged alone, names what the second left
        self.assertEqual([found["name"] for found in report["objects"]], ["second-model"])

    def test_a_production_close_warns_and_the_next_load_refuses(self):
        opened = self.opened()
        graph = self.memory.captured(self.device.hold(Graph(), GIB), self.memory.pool("rounds", "every round", (1, 1)))
        KEPT.append(graph)
        del graph
        with mock.patch.dict(os.environ, {lifetime.CHECK: ""}), mock.patch("sys.stderr") as stderr:
            report = opened.close()
        self.assertFalse(report["returned"])
        self.assertIn("the closed engine did not return its memory: 1 captured graph(s) alive (rounds/every round), "
                      f"held by module {__name__} global KEPT -> list[0] -> Graph",
                      "".join(call.args[0] for call in stderr.write.call_args_list))
        with self.assertRaises(lifetime.LeakingClose) as raised:
            lifetime.open(self.memory)
        self.assertIn("1 captured graph(s) alive", str(raised.exception))
        self.assertEqual(self.memory.report()["leaking"], report["message"])

    def test_an_unreadable_device_judges_the_ledger_and_the_graphs(self):
        device = memory.DeviceMemory("cpu")
        opened = lifetime.open(device)
        report = opened.close()
        self.assertTrue(report["returned"])
        self.assertIsNone(report["torch"])
        self.assertIn("torch's bytes cannot be read here", report["message"])


@unittest.skipUnless(STEP_LIBRARY, "Core's runtime crit_core_steps is not built: `crit core build` builds it")
class EngineCloseTests(unittest.TestCase):
    """Core's engine (crit_core.engine.load on the toy package) on a device of fake readings."""

    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        root = Path(temp.name)
        self.model = write_files(root / "model", llama_files(**{"config.json": {
            "model_type": "llama", "eos_token_id": 1, "text_config": {"eos_token_id": 4, "vocab_size": 420}}}))
        self.package = write_package(root / "package", drafting=True)
        self.device = Device()
        patches = [mock.patch.dict(memory._DEVICES, {"cpu": memory.DeviceMemory("cpu", self.device)}),
                   mock.patch.dict(os.environ, {lifetime.CHECK: "raise"})]
        for patch in patches:
            patch.start()
            self.addCleanup(patch.stop)
        self.memory = memory.of("cpu")
        KEPT.clear()
        self.addCleanup(KEPT.clear)

    def load(self):
        from crit_core.engine import load
        return load(self.package, self.model, device="cpu")

    def test_a_closed_engine_returns_its_memory_and_says_so_in_its_telemetry(self):
        engine = self.load()
        self.device.hold(engine.model, GIB)
        engine.generate([5, 6, 7], 3, stop_tokens=[])
        engine.close()
        report = engine.close_report
        self.assertTrue(report["returned"], report["message"])
        self.assertEqual(engine.telemetry()["close"], report)
        self.assertEqual(self.memory.opened, [])
        self.load().close()   # the reopen starts

    def test_an_engine_dropped_without_close_leaves_the_device_and_is_measured_at_the_next_load(self):
        # An Opened whose engine was collected without close() stayed open on the device, so every later close was
        # beside it and went unjudged. Now the engine's collection takes it off the device, and the next load
        # measures it as its close would have.
        import gc
        engine = self.load()
        self.assertEqual(len(self.memory.opened), 1)
        del engine
        gc.collect()
        self.assertEqual((self.memory.opened, len(self.memory.abandoned)), ([], 1))
        again = self.load()   # measured first: it returned its memory, so the load starts
        self.assertEqual(self.memory.abandoned, [])
        again.close()
        self.assertTrue(again.close_report["judged"], "the next close is judged alone")
        self.assertTrue(again.close_report["returned"], again.close_report["message"])

    def test_compile_recording_lasts_the_engines_life_and_ends_with_its_close(self):
        from crit_plugins import compiles
        before = len(compiles._recorders)   # engines another module left open record too
        engine = self.load()
        self.assertTrue(compiles.recording(), "an open engine records the process's compiles")
        self.assertEqual(len(compiles._recorders), before + 1)
        engine.close()
        self.assertEqual(len(compiles._recorders), before, "a closed engine's recorder ends with it")
        self.assertEqual(compiles.recording(), before > 0)

    def test_an_engine_that_keeps_its_rounds_fails_naming_them(self):
        # 1830b16d reverted: the closed engine kept its rounds, whose round driver holds the captured graphs and
        # the model, so four graphs and their pools survived the close.
        from crit_core.native import StepEngine
        engine = self.load()
        pool = self.memory.pool("core", "graphs", (3, 3))
        graphs = {key: self.memory.captured(self.device.hold(Graph(), GIB), pool) for key in range(4)}
        engine.rounds = Rounds(graphs, engine.model)
        self.device.hold(engine.model, 2 * GIB)
        del graphs
        close = StepEngine.close

        def keeping_rounds(self):   # StepEngine.close before 1830b16d: its rounds closed and kept
            rounds, drop = self.rounds, self._drop

            def kept():
                self.rounds = rounds
                drop()
            self._drop = kept
            close(self)
        with mock.patch.object(StepEngine, "close", keeping_rounds), \
                self.assertRaises(lifetime.LeakingClose) as raised:
            engine.close()
        report = raised.exception.report
        self.assertEqual(len(report["graphs"]), 4)
        self.assertTrue(any("StepEngine.rounds -> Rounds.graphs" in path
                            for found in report["graphs"] for path in found["held_by"]), report["graphs"])
        self.assertEqual(report["torch"]["grown"]["allocated"], 6 * GIB)
        model = next(found for found in report["objects"] if found["name"] == "model")
        self.assertTrue(any("StepEngine.rounds -> Rounds.model" in path for path in model["held_by"]), model)
        # The next load refuses beside it, with the report as its message, until the engine object goes.
        with self.assertRaisesRegex(lifetime.LeakingClose, "4 captured graph\\(s\\) alive \\(core/graphs\\)"):
            self.load()
        del engine, raised, report, model
        self.load().close()

    def test_a_drafter_that_keeps_the_model_fails_and_the_next_load_refuses(self):
        engine = self.load()
        drafter = engine.plugins["toy-drafter"]
        KEPT.append(Drafter(engine.model))   # the drafter's head, kept by its module across engines
        self.device.hold(engine.model, 2 * GIB)
        del drafter
        with self.assertRaises(lifetime.LeakingClose) as raised:
            engine.close()
        self.assertIn("plugin object model", raised.exception.report["message"])
        self.assertIn("Drafter.model", raised.exception.report["message"])
        with self.assertRaisesRegex(lifetime.LeakingClose, "beside the last engine's close"):
            self.load()
        KEPT.clear()
        self.load().close()

    def test_a_load_that_fails_closes_what_it_made_and_measures_it(self):
        configuration = json.loads((self.package / "configuration.json").read_text(encoding="utf-8"))
        configuration["plugins"]["options"] = {"toy-model": {"skip": "attention"}}
        from crit_core.engine import load
        from crit_plugins import PluginError
        with self.assertRaises(PluginError):
            load(self.package, self.model, configuration, device="cpu")
        gc.collect()
        self.assertEqual(self.memory.opened, [])
        self.load().close()



class HarnessTests(unittest.TestCase):
    """crit_plugins.testing's PluginTestCase closes every assembly a plugin's test makes and fails the test where the
    close did not return its memory, so every library plugin's own tests hold its close to Core's."""

    def run_case(self, leaks):
        from crit_plugins import testing

        class Plugin:
            def __init__(self, device):
                device.give("toy-cache", release=lambda n: 0)
                self.device = device

            def close(self):
                if not leaks:
                    self.device.ungive("toy-cache")

        class Assembly:
            def __init__(self, device):
                self.instances = {"toy-cache": Plugin(device)}

            def close(self):
                for plugin in self.instances.values():
                    plugin.close()
                self.instances = {}

        device = memory.DeviceMemory("cpu")

        class Case(testing.PluginTestCase):
            def plugin_id(self):
                return "toy-cache"

            def registry(self):
                return {"toy-cache": {"fills": ["expert_cache"]}}

            def runTest(self):
                self.assemble(context={"device": "cpu"})
        result = unittest.TestResult()
        with mock.patch.dict(memory._DEVICES, {"cpu": device}),                 mock.patch.object(testing, "assemble", lambda *args, **kwargs: Assembly(device)):
            Case().run(result)
        self.assertEqual(device.opened, [])
        self.assertIsNone(device.leaking)   # the leak is the test's failure, not the next test's refusal
        return result

    def test_a_plugin_whose_close_returns_its_memory_passes(self):
        result = self.run_case(leaks=False)
        self.assertTrue(result.wasSuccessful(), result.failures + result.errors)

    def test_a_plugin_whose_close_leaves_its_giver_fails_its_own_test_naming_it(self):
        result = self.run_case(leaks=True)
        self.assertEqual(len(result.failures), 1, result.errors)
        message = result.failures[0][1]
        self.assertIn("toy-cache: closing what the test assembled did not return its memory: giver(s) still listed "
                      "on the device: toy-cache", message)


if __name__ == "__main__":
    unittest.main()
