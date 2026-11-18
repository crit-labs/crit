"""crit_plugins.audit: readings of a device's pages say which buffers a workload wrote, which stayed zero and how
far the writes reach, with no word from the code that holds them; a captured graph's pool is named by its driver and
by the code whose allocations fill it; the bytes Core knows to be scratch are judged by a mark (CUDA)."""
import sys
import types
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for _folder in ("core/python", "cli"):
    sys.path.insert(0, str(ROOT / _folder))

from crit_core.native import step_library  # noqa: E402
from crit_plugins import audit, memory  # noqa: E402

try:
    import torch
except ImportError:
    torch = None
try:
    import safetensors  # noqa: F401
    import transformers
    ENGINE = torch is not None and torch.cuda.is_available() and step_library() is not None
except ImportError:
    ENGINE = False

MIB = 1 << 20


def plugin_class(package, name):   # a class as a plugin's package defines it (crit_plugins.assemble's names)
    return type(name, (), {"__module__": f"crit_plugin_{package}.plugin"})


@unittest.skipIf(torch is None or not torch.cuda.is_available(), "no CUDA device")
class AuditTests(unittest.TestCase):
    def setUp(self):
        self.memory = memory.of("cuda")
        if not self.memory.elastic_ranges:
            self.skipTest("the driver's virtual memory API is not reachable here")
        self.gran = memory.granularity(torch.cuda.current_device())
        self.audit = audit.Audit("cuda")
        self.addCleanup(self.audit.close)

    def bytes_of(self, size, fill=None):
        made = torch.zeros(size, dtype=torch.uint8, device="cuda")
        return made if fill is None else made.fill_(fill)

    def test_holdings_name_each_tensor_by_its_plugin_and_where_it_is_held(self):
        Pool, Model = plugin_class("state_0a", "Pool"), plugin_class("model_0b", "Model")
        pool, model = Pool(), Model()
        pool.windows = self.bytes_of(MIB)
        model.blocks = [{"q": self.bytes_of(MIB)}]
        model.pool = pool                                    # made by the state plugin, kept by the model
        module = types.ModuleType("crit_plugin_state_0a.buffers")
        module._BUFFERS = {"gemm": self.bytes_of(MIB)}       # a buffer the plugin's module keeps for all its objects
        sys.modules[module.__name__] = module
        self.addCleanup(sys.modules.pop, module.__name__)
        found = {(plugin, name) for plugin, name, _ in self.memory.holdings({"state": pool, "the-model": model})}
        self.assertEqual(found, {("state", "Pool.windows"), ("the-model", "Model.blocks[0][q]"),
                                 ("state", "buffers._BUFFERS[gemm]")})
        held = self.memory.census({"state": pool, "the-model": model})["plugins"]
        self.assertEqual(held, {"state": 2 * MIB, "the-model": MIB})

    def test_readings_tell_dead_oversized_written_and_constant_buffers_apart(self):
        Work = plugin_class("work_0a", "Work")
        work = Work()
        work.dead = self.bytes_of(4 * MIB)
        work.oversized = self.bytes_of(8 * MIB)
        work.scratch = self.bytes_of(2 * MIB)
        work.table = self.bytes_of(2 * MIB, 7)
        work.stale = torch.empty(8 * MIB, dtype=torch.uint8, device="cuda").fill_(9)   # a workspace of old bytes
        self.audit.plugins = {"work": work}
        self.audit.read("load")
        self.audit.mark()
        for label, value in (("first", 1), ("second", 2)):
            work.oversized[:MIB].fill_(value)
            work.scratch.fill_(value)
            work.stale[:2 * MIB].fill_(value)
            self.audit.read(label)
        work.late = self.bytes_of(2 * MIB)                    # made after the last reading
        self.audit.read("third")
        report = self.audit.report()
        found = {row["name"]: row for row in report["buffers"] if row["plugin"] == "work"}
        self.assertEqual({name: (row["verdict"], row["certain"]) for name, row in found.items()},
                         {"Work.dead": ("dead", True), "Work.oversized": ("oversized", True),
                          "Work.scratch": ("written", True), "Work.table": ("constant", False),
                          "Work.stale": ("partly written", False), "Work.late": ("new", True)})
        self.assertEqual((found["Work.oversized"]["written"], found["Work.oversized"]["reach"],
                          found["Work.oversized"]["zero"]), (MIB, MIB, 7 * MIB))
        self.assertEqual((found["Work.stale"]["written"], found["Work.stale"]["reach"], found["Work.stale"]["constant"]),
                         (2 * MIB, 2 * MIB, 6 * MIB))
        self.assertEqual((found["Work.table"]["shape"], found["Work.table"]["dtype"], found["Work.late"]["since"]),
                         ([2 * MIB], "uint8", "third"))
        self.assertEqual(report["plugins"]["work"]["buffers"], 6)
        self.assertEqual((report["totals"]["dead"], report["totals"]["oversized"]) >= (4 * MIB, 7 * MIB), True)
        self.assertEqual(int(work.table.min()), 7)            # a reading writes nothing
        self.assertEqual(report["readings"], ["load", "first", "second", "third"])

    def test_a_graphs_pool_is_named_and_its_temporaries_are_judged_by_the_mark(self):
        Model = plugin_class("model_0c", "Model")
        model = Model()
        model.x = self.bytes_of(4 * MIB, 1)
        graph, stream = torch.cuda.CUDAGraph(), torch.cuda.Stream()
        pool = self.memory.pool("rounds", "1 rows, width 0, greedy")
        with self.memory.capturing(pool), torch.cuda.graph(graph, pool=pool, stream=stream):
            wide = model.x.float() * 2                        # 16 MiB: a temporary of the capture
            model.out = wide.sum()
            del wide
            unused = torch.zeros(6 * MIB, dtype=torch.uint8, device="cuda")   # written by a kernel the capture holds
            del unused
        self.audit.plugins = {"the-model": model}
        self.audit.read("load")
        self.audit.mark()
        self.assertGreaterEqual(sum(end - first for first, end in self.audit.marks), 22 * MIB)
        graph.replay()
        self.audit.read("first")
        report = self.audit.report()
        held = next(found for found in report["pools"] if found["name"] == "rounds/1 rows, width 0, greedy")
        self.assertGreaterEqual(held["temporaries"], 22 * MIB)
        self.assertGreaterEqual(held["written"], 22 * MIB)    # the replay wrote the product and the zeros
        self.assertLess(held["written"], held["temporaries"])
        made = {found["bytes"]: found["where"] for found in held["allocations"]}
        self.assertIn("test_audit.py", made[16 * MIB])
        self.assertIn("test_a_graphs_pool_is_named", made[16 * MIB])
        self.assertEqual(self.memory.census({"the-model": model})["pools"]["rounds/1 rows, width 0, greedy"],
                         held["bytes"])
        self.assertEqual(float(model.out), 2.0 * 4 * MIB)     # the mark changed nothing the graph reads

    def test_a_growing_buffer_is_scratch_and_a_paged_range_reports_its_units(self):
        gran = self.gran
        rows = self.memory.growing("test", "rows", 8 * gran)
        self.addCleanup(rows.close)
        rows.take(4 * gran).fill_(3)
        planes = [(0, gran, 4), (8 * gran, gran // 2, 6)]
        paged = self.memory.paged("test", "states", 16 * gran, planes)
        self.addCleanup(paged.close)
        self.assertTrue(paged.back(2))
        self.audit.read("load")
        self.audit.mark()
        for label, value in (("first", 1), ("second", 2)):
            rows.take(gran).fill_(value)                       # a call of a smaller shape
            paged.tensor[:gran].fill_(value)                   # unit 0 of the first plane
            paged.tensor[8 * gran:8 * gran + gran // 2].fill_(value)
            self.audit.read(label)
        found = {row["name"]: row for row in self.audit.report()["elastic"]}
        self.assertEqual((found["test/rows"]["written"], found["test/rows"]["zero"], found["test/rows"]["verdict"],
                          found["test/rows"]["certain"]), (gran, 3 * gran, "oversized", True))
        self.assertEqual(found["test/states"]["planes"],
                         [{"unit_bytes": gran, "units": 4, "backed": 2, "written": 1},
                          {"unit_bytes": gran // 2, "units": 6, "backed": 2, "written": 1}])
        # Units backed from host memory are not the device's bytes: the audit counts them and reads none of them.
        hosted = self.memory.paged("test", "hosted", 8 * gran, [(0, gran, 8)], host_bytes=4 * gran)
        self.addCleanup(hosted.close)
        self.assertTrue(hosted.back(2))
        if hosted.back(4, host=True):     # a host with memory to spare beside its headroom
            hosted.tensor[:4 * gran].fill_(7)
            self.audit.read("third")
            row = next(row for row in self.audit.report()["elastic"] if row["name"] == "test/hosted")
            self.assertEqual((row["backed"], row["host"]), (2 * gran, 2 * gran))
            self.assertEqual(row["planes"], [{"unit_bytes": gran, "units": 8, "backed": 2, "written": 0}])
        self.assertNotIn("audit/pages", found)

    def test_with_an_engine_readings_and_the_mark_wait_for_a_request_to_end(self):
        engine = types.SimpleNamespace(plugins={}, audit=None)
        self.audit.attach(engine)
        self.audit.read("load")
        self.audit.mark()
        self.audit.read("first")
        self.assertEqual(self.audit.readings, [])              # nothing is read from the caller's thread
        with self.assertRaises(ValueError):
            self.audit.report()
        engine.audit.step([])()                                # the engine's thread: a step's start, then its end
        self.assertEqual(self.audit.readings, [])
        engine.audit.retire()                                  # ... and a request's end, before the model retires it
        self.assertEqual([reading.label for reading in self.audit.readings], ["load", "first"])
        self.assertEqual((self.audit.marked_at, len(self.audit.steps)), (1, 1))
        self.assertEqual(self.audit.report()["readings"], ["load", "first"])

    def test_steps_are_counted_by_kind(self):
        self.audit.steps = [(30, 1, 0, 4000, 12 * MIB, MIB), (40, 1, 0, 5000, 80 * MIB, 3 * MIB),
                            (512, 1, 1, 10, 0, 0), (0, 1, 0, 300, MIB, 0)]
        self.audit.engine = types.SimpleNamespace(rounds=types.SimpleNamespace(runs=(120, 120, 0)), audit=None)
        self.assertEqual(self.audit._steps(), [
            {"kind": "prompt step, 17 to 32 tokens", "steps": 1, "replays": 0.0, "allocations": 4000,
             "most_allocations": 4000, "peak": 12 * MIB, "kept": MIB},
            {"kind": "prompt step, 33 to 64 tokens", "steps": 1, "replays": 0.0, "allocations": 5000,
             "most_allocations": 5000, "peak": 80 * MIB, "kept": 3 * MIB},
            {"kind": "prompt step, 257 to 512 tokens", "steps": 1, "replays": 1.0, "allocations": 10,
             "most_allocations": 10, "peak": 0, "kept": 0},
            {"kind": "decode step (the model's step)", "steps": 1, "replays": 0.0, "allocations": 300,
             "most_allocations": 300, "peak": MIB, "kept": 0},
            {"kind": "native round (no Python)", "steps": 120, "replays": 1.0, "allocations": 0,
             "most_allocations": 0, "peak": 0, "kept": 0}])


@unittest.skipUnless(ENGINE, "a GPU engine: CUDA, transformers, safetensors and CRIT_CORE_STEP_LIBRARY")
class EngineAuditTests(unittest.TestCase):
    def test_the_command_audits_a_whole_engine_with_no_word_from_its_plugins(self):
        import json
        import shutil
        import tempfile
        from crit.audit import audit as run, render
        from crit.plugins import add
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        root = Path(temp.name)
        torch.manual_seed(0)
        config = transformers.LlamaConfig(vocab_size=256, hidden_size=128, intermediate_size=256, num_hidden_layers=2,
                                          num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=1024,
                                          rms_norm_eps=1e-6, tie_word_embeddings=False)
        model = transformers.LlamaForCausalLM(config)
        with torch.no_grad():
            model.lm_head.weight.mul_(30)   # sharp logits: a greedy choice no rounding flips
        model.save_pretrained(root / "model", safe_serialization=True)
        package = root / "package"
        add(package / "plugins", ["attention-state", "softmax-attention", "projections-dense", "token-sampling",
                                  "device-placement", "fused-norms"])
        shutil.copytree(ROOT / "tests" / "support" / "device_plugins" / "llama-model", package / "plugins" / "llama-model")
        selection = {"use": {"model": "llama-model", "placement": "device-placement", "projections": "projections-dense",
                             "attention": "softmax-attention", "kv_cache": "attention-state",
                             "sampler": "token-sampling", "norms": "fused-norms"}, "base": {"norms": "torch-norms"}}
        (package / "configuration.json").write_text(json.dumps({"plugins": selection}), encoding="utf-8")
        report = run(package, root / "model", prompt_tokens=300, reply_tokens=8, least=1 << 16,
                     encode=lambda text: [1 + byte % 255 for byte in text.encode()][:300])
        work = report["workload"]
        self.assertEqual((work["sets"], work["requests"], work["prompt_tokens"], work["repeated"]),
                         (2, 10, [300, 300], True))
        self.assertEqual(report["readings"], ["load", "first", "second"])
        self.assertEqual((len(work["replies"]), work["replies"][0]), (6, work["replies"][-1]))
        self.assertGreater(report["telemetry"]["core"]["steps"], 0)   # the engine's own counters come with it
        self.assertTrue({"llama-model", "attention-state"} <= set(report["plugins"]))
        scratch = next(row for row in report["elastic"] if row["name"] == "core/scratch")
        self.assertTrue(0 < scratch["written"] <= scratch["backed"])   # the engine's scratch: marked, then written
        self.assertTrue(scratch["certain"])
        self.assertTrue(any(step["kind"].startswith("prompt step") and step["steps"] for step in report["steps"]))
        named = [row for row in report["buffers"] if row["plugin"] and row["verdict"] != "new"]
        self.assertTrue(named and all("." in row["name"] and row["bytes"] >= 1 << 16 for row in named))
        self.assertTrue(all(pool["name"].split("/")[0] in ("core", "rounds", "prompts", "captures")
                            for pool in report["pools"]))
        self.assertIn("the same reply", render(report))
        self.assertIsNone(memory.of("cuda").recorded)          # the audit ended its recording
        self.assertNotIn("audit/pages", memory.of("cuda").ranges)


class RenderTests(unittest.TestCase):
    def test_the_command_renders_a_report_with_what_is_unused_first(self):
        from crit.audit import prompts, render
        buffer = {"plugin": "softmax-attention", "name": "DeviceAttention.split_workspace", "shape": [128 * MIB],
                  "dtype": "uint8", "pool": None, "since": "load", "bytes": 128 * MIB, "written": 48 * MIB,
                  "reach": 48 * MIB, "zero": 80 * MIB, "constant": 0, "verdict": "oversized", "certain": True}
        report = {"memory": {"device": "cuda:0", "total": 32 << 30, "free": 1 << 28,
                             "torch": {"reserved": 6 << 30, "allocated": 6 << 30, "unused": 0},
                             "elastic": {"expert-offload/slots": {"backed": 22 << 30, "reserved": 65 << 30}},
                             "other": 2 << 30},
                  "workload": {"sets": 2, "prompt_tokens": [2040, 2046], "reply_tokens": 96, "requests": 10,
                               "at_once": 1, "repeated": True, "seconds": 61.0,
                               "replies": ["2cf54ab6", "896fe95a"]},
                  "readings": ["load", "first", "second"], "audit_bytes": 33 * MIB, "marked_bytes": 253 * MIB,
                  "buffers": [buffer, {**buffer, "name": "Matrix.tensors[weight]", "verdict": "constant", "zero": 0,
                                       "written": 0, "reach": 0, "constant": 128 * MIB, "certain": False}],
                  "plugins": {"softmax-attention": {"buffers": 2, "bytes": 256 * MIB, "written": 48 * MIB,
                                                    "zero": 80 * MIB, "constant": 128 * MIB}},
                  "pools": [{"name": "rounds/1 rows, width 3, sampled", "bytes": 270 * MIB, "kept": 17 * MIB,
                             "temporaries": 253 * MIB, "written": 250 * MIB,
                             "allocations": [{"bytes": 98 * MIB, "count": 60, "where": "code/pages.py:484 rows_at"}]}],
                  "elastic": [{"name": "sequential-recurrence/states", "reserved": 1 << 30, "backed": 170 * MIB,
                               "written": 40 * MIB, "zero": 130 * MIB, "constant": 0, "verdict": "oversized",
                               "certain": True, "planes": [{"unit_bytes": 3 * MIB, "units": 48, "backed": 3,
                                                            "written": 1}] * 36}],
                  "steps": [{"kind": "prompt step, 257 to 512 tokens", "steps": 8, "replays": 0.0, "allocations": 4083,
                             "most_allocations": 4100, "peak": 286 * MIB, "kept": 4 * MIB}],
                  "totals": {"dead": 0, "oversized": 80 * MIB, "new": 0}}
        text = render(report)
        self.assertLess(text.index("Oversized"), text.index("By plugin"))
        for part in ("softmax-attention", "128.0 MiB  written    48.0 MiB, zero    80.0 MiB",
                     "DeviceAttention.split_workspace [134217728]", "the same reply; greedy replies 2cf54ab6 896fe95a",
                     "98.0 MiB x60   code/pages.py:484 rows_at", "36 planes of 48 units of 3.0 MiB: 3 backed, 1 written by the workload",
                     "allocations  4,083 (most 4,100), peak 286.0 MiB"):
            self.assertIn(part, text)
        self.assertNotIn("Matrix.tensors", text)              # constants are counted by plugin, not listed
        ids = prompts(lambda text: text.split(), 300)          # a tokenizer of words
        self.assertEqual(len(ids), 2)
        self.assertTrue(all(250 < len(one["long"]) <= 300 and len(one["chats"]) == 3 for one in ids))
        self.assertNotIn("valve", " ".join(ids[1]["long"]))    # the two sets share no words but the frame's


if __name__ == "__main__":
    unittest.main()
