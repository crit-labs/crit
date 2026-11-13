"""Core's engine loader (crit_core.engine): a package is its plugins and the configuration that selects them, and an
engine is Core running what the selection assembles. The loader refuses a package that is not a whole engine and a
model plugin that cannot batch or serve the contract, runs the toy package (inference/examples/plugin-engine/plugins) on Core's concurrent
runtime, and Core's eval-v1 and chat-resident-v1 adapters serve it with its plugins in their facts."""
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import shutil
import sys
import tempfile
import threading
import types
import unittest

ROOT = Path(__file__).resolve().parents[1]
for _folder in ("core/python", "tests"):
    sys.path.insert(0, str(ROOT / _folder))

from core_text_fixtures import llama_files, write_files  # noqa: E402
from crit_core import engine as core_engine  # noqa: E402
from crit_core.engine import (capabilities, check_package, eval_v1, load, resident, selection,  # noqa: E402
                              serving_problems)
from crit_core.native import NativeError  # noqa: E402
from crit_core.serving import CONCURRENT_CAPABILITY, LOGPROBS_CAPABILITY, PLAIN_CAPABILITY, SCORE_CAPABILITY  # noqa: E402
from crit_plugins import PluginError  # noqa: E402
from support.engines import STEP_LIBRARY, USE, write_package  # noqa: E402

VOCAB = 420   # the llama fixture's vocab_size, which the toy model reads from config.json


def successor(seen):
    h = 1469598103934665603
    for position, token in enumerate(seen):
        h = ((h ^ ((token + 1) * (position + 1))) * 1099511628211) % 2 ** 64
    return h % VOCAB


def oracle(prompt, output, stops=(1, 4)):
    rows, out = list(prompt), []
    for _ in range(output):
        out.append(successor(rows))
        rows.append(out[-1])
        if out[-1] in stops:
            break
    return out


class PackageFixture:
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.model = write_files(self.root / "model", llama_files(**{"config.json": {
            "model_type": "llama", "eos_token_id": 1, "text_config": {"eos_token_id": 4, "vocab_size": VOCAB}}}))
        self.package = write_package(self.root / "package")

    def configuration(self):
        return json.loads((self.package / "configuration.json").read_text(encoding="utf-8"))


class PackageCheckTests(PackageFixture, unittest.TestCase):
    """What the loader refuses before any plugin code runs (check_package, which crit run package uses too)."""

    def test_a_package_is_its_plugins_and_a_selection_that_is_a_whole_engine(self):
        use, order, _ = check_package(self.package, self.model, device="cpu")
        self.assertEqual(use, USE)
        self.assertEqual(order[-1], "toy-model")
        shutil.rmtree(self.package / "plugins")
        with self.assertRaisesRegex(PluginError, "has no plugins/ folder: an engine package is its plugins"):
            check_package(self.package, self.model, device="cpu")

    def test_a_configuration_without_a_selection_is_refused(self):
        with self.assertRaisesRegex(PluginError, "the configuration names no plugins"):
            check_package(self.package, self.model, {"batch": 1}, device="cpu")
        (self.package / "configuration.json").unlink()
        with self.assertRaisesRegex(PluginError, "holds no configuration.json"):
            check_package(self.package, self.model, device="cpu")

    def test_an_engine_fills_every_required_slot_on_its_device(self):
        configuration = self.configuration()
        del configuration["plugins"]["use"]["sampler"]
        with self.assertRaisesRegex(PluginError, "leaves sampler empty"):
            check_package(self.package, self.model, configuration, device="cpu")
        with self.assertRaisesRegex(PluginError, "runs on cpu, not on cuda"):
            check_package(self.package, self.model, device="cuda")
        self.assertEqual(core_engine.engine_device(core_engine.registry_of(self.package), USE, {}), "cpu")

    def test_eval_s_plain_engine_drafts_nothing(self):
        chosen = selection({"plugins": {"use": {**USE, "drafter": "toy-drafter", "verifier": "toy-verifier"},
                                        "options": {"toy-drafter": {}}}}, speculative=False)
        self.assertEqual((chosen["use"]["drafter"], chosen["use"]["verifier"], chosen["options"]), (None, None, {}))

    def test_the_model_s_architecture_is_a_fact_its_plugin_needs(self):
        manifest = self.package / "plugins" / "toy-model" / "plugin.json"
        data = json.loads(manifest.read_text(encoding="utf-8"))
        manifest.write_text(json.dumps({**data, "needs": [["model_type:qwen3_5", "model_type:qwen3_5_text"]]}),
                            encoding="utf-8")
        with self.assertRaisesRegex(PluginError, "toy-model needs model_type:qwen3_5 or model_type:qwen3_5_text"):
            check_package(self.package, self.model, device="cpu")
        manifest.write_text(json.dumps({**data, "needs": ["model_type:llama"]}), encoding="utf-8")
        check_package(self.package, self.model, device="cpu")

    def test_every_engine_serves_the_same_contract(self):
        found = capabilities(self.package, self.model)
        for capability in (CONCURRENT_CAPABILITY, LOGPROBS_CAPABILITY, SCORE_CAPABILITY,
                           "chat-sampling-temperature-v1", "chat-sampling-min-p-v1", "chat-sampling-seed-v1"):
            self.assertIn(capability, found)
        self.assertNotIn(PLAIN_CAPABILITY, found)
        self.assertNotIn("chat-sampling-presence-penalty-v1", found)
        drafting = write_package(self.root / "drafting", drafting=True)
        self.assertIn(PLAIN_CAPABILITY, capabilities(drafting, self.model))

    def test_a_model_short_of_the_contract_is_named(self):
        info = {"token_logprobs": True, "max_top_logprobs": 20, "scoring": True, "query_scoring": True,
                "replay": True, "sampling": dict.fromkeys(core_engine.CORE_SAMPLING, True), "speculative": False}
        fields = core_engine.CORE_SAMPLING
        self.assertEqual(serving_problems(info, False, fields), [])
        self.assertEqual(serving_problems({**info, "max_top_logprobs": 5}, False, fields),
                         ["info() reports token_logprobs with max_top_logprobs of at least 20 (chat-logprobs-v1)"])
        self.assertEqual(serving_problems({**info, "replay": False}, False, fields),
                         ["info() declares no replay: every engine scores fixed text (chat-score-v1)"])
        self.assertEqual(serving_problems({**info, "sampling": {"temperature": True}}, False, fields),
                         ["info()'s sampling does not apply top_p, top_k, min_p, seed"])
        self.assertEqual(serving_problems(info, True, fields),
                         ["info() declares speculative False, but the selection drafts",
                          "info() declares no plain: an engine that drafts also decodes one token per target pass on "
                          "request (chat-plain-v1)"])


@unittest.skipUnless(STEP_LIBRARY, "Core's runtime crit_core_steps is not built: `crit core build` builds it")
def _cuda():
    try:
        import torch
    except ImportError:
        return False
    return torch.cuda.is_available()


@unittest.skipUnless(STEP_LIBRARY and _cuda(), "Core's runtime and a CUDA device for captured prompt chunks")
class ReplayedSlotTests(PackageFixture, unittest.TestCase):
    """The loader's "the engine is its plugins" request reaches a slot through a graph replay too: a replay counts as
    the calls its capture made through the loader's proxies (here the toy's projection work, captured as prompt
    chunks). A capture that never called a slot's plugin does not count for it."""

    def load(self, mode):
        return load(write_package(self.root / mode, options={"toy-model": {"prompt_graphs": mode}}), self.model)

    def test_a_model_whose_steps_replay_captured_chunks_passes_the_check(self):
        engine = self.load("on")
        try:
            tokens, _ = engine.generate([5, 6, 7], 4, stop_tokens=[])
            self.assertEqual(tokens, oracle([5, 6, 7], 4, stops=()))
            counts = engine.model.prompts.telemetry()["prompt_graphs"]
            self.assertGreater(counts["graph_chunks"], 0)
            self.assertEqual(counts["eager_chunks"], 0, "the projections plugin was reached through replays only")
        finally:
            engine.close()

    def test_a_replay_whose_capture_skipped_a_slot_does_not_count_for_it(self):
        with self.assertRaisesRegex(PluginError, "never called the projections plugin"):
            self.load("skip")


@unittest.skipUnless(STEP_LIBRARY and _cuda(), "Core's runtime and a CUDA device for captured prompt chunks")
class SideBySideEngineTests(PackageFixture, unittest.TestCase):
    """Two engines in one process: one closed and freed leaves the other's captured prompt chunks their bits, and their
    replays write no memory the other engine gave back. Every capture runs on Core's one capture stream, and torch frees
    the stream's cuBLAS workspaces when a graph captured on it goes (CUDAGraph.reset); a graph's workspace is made where
    nothing else is (crit_plugins.rounds.capture_workspaces)."""

    def test_an_engine_freed_beside_another_leaves_its_captured_chunks_their_bits_and_memory(self):
        import gc
        import weakref
        import torch
        from crit_plugins.rounds import capture_stream
        # Rows of 4,096 columns: cuBLAS sums each in parts through its workspace.
        options = {"toy-model": {"prompt_graphs": "on", "prompt_width": 4096}}
        kept = load(write_package(self.root / "kept", options=options), self.model)
        try:
            freed = load(write_package(self.root / "freed", options=options), self.model)
            chunks, tokens = kept.model.prompts, list(range(5, 21))
            before = chunks.run(1, 0, tokens).clone()
            self.assertEqual(before.view(-1).tolist(), [4096.0 * t for t in tokens])
            gone = weakref.ref(freed.model.prompts)
            freed.close()
            del freed
            gc.collect()
            self.assertIsNone(gone(), "the closed engine's captured chunks are freed")
            # Fresh tensors on the capture stream take the memory freed there (its cache gives a block back to it).
            with torch.cuda.stream(capture_stream(torch.device("cuda"))):
                taken = [torch.full((8 << 20,), 7.0, device="cuda") for _ in range(4)]
            after = chunks.run(1, 0, tokens)
            torch.cuda.synchronize()
            self.assertTrue(torch.equal(after, before), "the captured chunk gives other bits")
            self.assertTrue(all(bool((t == 7.0).all()) for t in taken), "a replay wrote memory its engine gave back")
            self.assertEqual(kept.generate([5, 6, 7], 4, stop_tokens=[])[0], oracle([5, 6, 7], 4, stops=()))
        finally:
            kept.close()


class NativeReplayCheckTests(unittest.TestCase):
    def test_native_rounds_of_a_request_count_as_the_calls_their_captures_made(self):
        from crit_core.engine import slot_problems
        from crit_plugins.observe import Calls
        calls = Calls()

        class Rounds:   # Core's native rounds: their runs, and the slots the captures called
            runs, captured_slots = (0, 0), ["attention"]

        class Engine:
            rounds = Rounds()

            def info(self):
                return {"vocabulary": 10}

            def generate(self, prompt, count, stop_tokens, sampling, constraint=None):
                if constraint:
                    return [constraint["tokens"][0]], {}
                calls["projections"] += 1          # the prompt, through the proxies
                self.rounds.runs = (self.rounds.runs[0] + self.native, 1)   # the decode, natively
                return [3, 4], {}
        use = {"model": "m", "projections": "p", "attention": "a"}
        engine = Engine()
        engine.native = 1
        self.assertEqual(slot_problems(engine, use, calls), [])
        engine.native = 0   # no native round ran: nothing reached attention
        self.assertIn("never called the attention plugin", slot_problems(engine, use, calls)[0])


class PlanCheckTests(unittest.TestCase):
    """Core's check that an engine holds on the device only what its placement plans and its plugins' own memory put
    there."""

    def test_declared_memory_counts_as_planned_as_far_as_it_was_measured(self):
        from crit_core.engine import plan_problems
        plans = [({"weights": 1 << 30, "spill": 1 << 30}, {"weights": "device", "spill": "host"})]
        held = (1 << 30) + (300 << 20)
        self.assertTrue(plan_problems(plans, held), "300 MiB past the plan, beyond its slack")
        self.assertEqual(plan_problems(plans, held, declared=256 << 20), [], "a workspace measured and declared")
        problem = plan_problems(plans, held + (1 << 30), declared=256 << 20, unmeasured=["attn"])[0]
        self.assertIn("unmeasured, counted at their declarations: attn", problem)
        self.assertEqual(plan_problems(plans, None), [], "a device whose memory cannot be read")


class FactTests(unittest.TestCase):
    """The facts Core's loader gives plugins' needs: the checkpoint's stored formats and the device's features."""

    def test_a_checkpoint_says_how_its_matrices_are_stored(self):
        from crit_core.engine import stored_facts
        for config, facts in (({}, ["stored:plain"]),
                              ({"quantization_config": {"quant_method": "modelopt", "quant_algo": "NVFP4"}},
                               ["stored:nvfp4"]),
                              ({"quantization_config": {"quant_method": "compressed-tensors",
                                                        "format": "nvfp4-pack-quantized"}}, ["stored:nvfp4"]),
                              ({"quantization_config": {"quant_method": "compressed-tensors", "config_groups": {
                                  "group_0": {"format": "float-quantized"}}}}, ["stored:fp8"]),
                              ({"quantization_config": {"quant_method": "fp8", "weight_block_size": [128, 128]}},
                               ["stored:fp8"]),
                              ({"quantization_config": {"quant_method": "awq", "bits": 4}}, ["stored:awq"]),
                              # ModelOpt's mixed precision names each layer's format.
                              ({"quantization_config": {"quant_method": "modelopt", "quant_algo": "MIXED_PRECISION",
                                                        "quantized_layers": {
                                                            "layers.0.mlp.experts": {"quant_algo": "NVFP4"},
                                                            "layers.0.mlp.shared": {"quant_algo": "W4A16_NVFP4"},
                                                            "layers.0.attn": {"quant_algo": "MXFP8"},
                                                            "layers.1.attn": {"quant_algo": "FP8"}}}},
                               ["stored:nvfp4", "stored:fp8", "stored:mxfp8"]),
                              ({"quantization_config": {"quant_method": "modelopt", "quant_algo": "MIXED_PRECISION",
                                                        "quantized_layers": {}}}, ["stored:modelopt"])):
            with self.subTest(config=config):
                self.assertEqual(stored_facts(config), facts)

    def test_a_device_says_its_compute_capability_and_tensor_core_formats(self):
        from crit_core.engine import machine_facts
        self.assertEqual(machine_facts("cuda", {"compute_capability": "12.0"}),
                         ["device:cuda", "cuda:sm_120", "cuda:bf16", "cuda:fp8", "cuda:fp4"])
        self.assertEqual(machine_facts("cuda", {"devices": [{"compute_capability": "8.6"}]}),
                         ["device:cuda", "cuda:sm_86", "cuda:bf16"])
        self.assertEqual(machine_facts("cuda", {"compute_capability": None}), ["device:cuda"])   # unknown: none
        self.assertEqual(machine_facts("cpu", {"compute_capability": "12.0"}), ["device:cpu"])

    def test_without_a_probe_the_gpu_cuda_runs_on_is_asked(self):
        from unittest.mock import patch
        import os
        asked = []

        def run(command, **_):
            asked.append(command[command.index("-i") + 1])
            return types.SimpleNamespace(returncode=0, stdout="8.9\n")
        for visible, gpu in (("2,0", "2"), ("GPU-1234abcd", "GPU-1234abcd"), (None, "0")):
            environment = {k: v for k, v in os.environ.items() if k != "CUDA_VISIBLE_DEVICES"}
            environment.update({"CUDA_VISIBLE_DEVICES": visible} if visible else {})
            with patch.dict(os.environ, environment, clear=True), patch("subprocess.run", run), \
                    patch.dict(core_engine._PROBED, clear=True):
                self.assertEqual(core_engine.cuda_capability(), 89)
            self.assertEqual(asked[-1], gpu)


class LoadTests(PackageFixture, unittest.TestCase):
    def open(self, **options):
        engine = load(self.package, self.model, device="cpu", **options)
        self.addCleanup(engine.close)
        return engine

    def test_a_probed_engine_reports_its_slowest_prompt_and_decode_steps(self):
        # Without STEP_PROBE nothing is probed; with it, every step counts under its kind (a prompt step prefills a
        # row), timed with its device work finished.
        import os
        from unittest import mock
        from crit_core import native
        self.assertNotIn("probe", self.open().telemetry()["core"])
        with mock.patch.dict(os.environ, {native.STEP_PROBE: "1"}):
            engine = self.open()
        engine.generate(list(range(5, 45)), 4, stop_tokens=[])
        probe = engine.telemetry()["core"]["probe"]
        self.assertEqual(set(probe), {"prompt", "decode"})
        for kind in ("prompt", "decode"):
            self.assertGreater(probe[kind]["steps"], 0)
            self.assertGreater(probe[kind]["slowest_seconds"], 0)

    def test_the_engine_runs_its_model_plugin_and_reports_its_plugins(self):
        engine = self.open()
        info = engine.info()
        self.assertEqual((info["implementation"], info["plugins"]), ("toy step model", USE))
        self.assertGreater(info["max_concurrent_requests"], 1)
        tokens, result = engine.generate([5, 6, 7], 6, stop_tokens=[])
        self.assertEqual(tokens, oracle([5, 6, 7], 6, stops=()))
        self.assertEqual(result["finish"], "length")

    def test_requests_batch_and_decode_what_they_decode_alone(self):
        engine = self.open()
        prompts = [[(r * 37 + i * 11) % VOCAB for i in range(1 + r * 5)] for r in range(9)]
        with ThreadPoolExecutor(9) as pool:
            together = list(pool.map(lambda p: engine.generate(p, 12, stop_tokens=[])[0], prompts))
        self.assertEqual(together, [oracle(p, 12, stops=()) for p in prompts])
        model = engine.model
        self.assertTrue(any(rows > 1 for rows, _ in model.steps), "no step held two requests")

    def test_every_step_a_step_engine_runs_is_kept_or_ends_its_request(self):
        # Core never discards a step engine's step (crit/step_engine.h, StepDecision): it runs one step at a time, so no
        # step is a discard-only descendant of another in flight; a prompt chunk it ran (its state written in place, as a
        # replayed one's is) is kept, or its request ends. Here under stops, output limits, cancellations and prompts
        # of several chunks, many requests at once.
        engine = self.open()
        model, order, decisions = engine.model, [], []
        step, commit = model.step, model.commit

        def stepped(rows):
            order.append("step")
            return step(rows)

        def committed(found):
            order.append("commit")
            decisions.extend(found)
            return commit(found)
        model.step, model.commit = stepped, committed
        prompts = [[(r * 41 + i * 13) % VOCAB for i in range(3 + r * 7)] for r in range(12)]

        def run(r):
            seen = []
            cancelled = (lambda: len(seen) > 3 or bool(seen.append(1))) if r % 4 == 3 else None
            return engine.generate(prompts[r], 3 + r % 5, stop_tokens=[1, 4] if r % 2 else [],
                                   **({"cancelled": cancelled} if cancelled else {}))
        with ThreadPoolExecutor(12) as pool:
            found = list(pool.map(run, range(12)))
        self.assertTrue(decisions)
        self.assertFalse(any(d.discard for d in decisions))
        self.assertEqual(order, ["step", "commit"] * (len(order) // 2), "one step at a time, each committed")
        for r, (tokens, result) in enumerate(found):
            if r % 4 != 3:
                self.assertEqual(tokens, oracle(prompts[r], 3 + r % 5, stops=(1, 4) if r % 2 else ()))

    def test_a_model_that_cannot_batch_is_refused_and_its_plugins_closed(self):
        configuration = self.configuration()
        configuration["plugins"]["options"] = {"toy-model": {"concurrency": 1}}
        with self.assertRaisesRegex(PluginError, "toy-model: Core cannot serve it: info\\(\\) declares max_concurrency 1"):
            load(self.package, self.model, configuration, device="cpu")

    def test_a_model_plugin_that_leaves_a_slot_idle_is_refused(self):
        # A model plugin that does a slot's work itself fills the slot with a plugin it never calls: the engine is
        # not its plugins, and Core's loader refuses it after one request.
        for skipped, message in (("attention", "never called the attention plugin \\(toy-attention\\) during a request"),
                                 ("projections", "never called the projections plugin"),
                                 ("placement", "never asked the placement plugin where its tensors live")):
            configuration = self.configuration()
            configuration["plugins"]["options"] = {"toy-model": {"skip": skipped}}
            with self.subTest(skipped=skipped), self.assertRaisesRegex(PluginError, message):
                load(self.package, self.model, configuration, device="cpu")
        self.open().close()   # every slot called: it loads

    def test_close_closes_every_plugin(self):
        engine = load(self.package, self.model, device="cpu")
        model = engine.model
        engine.close()
        self.assertTrue(model.closed)
        with self.assertRaisesRegex(NativeError, "engine is closed"):
            engine.generate([5], 1)

    def test_close_frees_a_model_whose_parts_refer_to_each_other(self):
        import gc
        import weakref
        engine = load(self.package, self.model, device="cpu")
        model = engine.model
        model.parts_cycle = types.SimpleNamespace(model=model)   # a reference cycle, as a drafter and its model make
        freed = weakref.ref(model)
        del model
        gc.disable()   # only close() may collect
        try:
            engine.close()
            self.assertIsNone(freed())
        finally:
            gc.enable()

    def test_reserving_more_device_memory_than_the_device_holds_warns_once_under_wddm(self):
        import contextlib
        import io
        from unittest import mock
        from crit_core import native
        reserved, free = [10 << 30], [8 << 30]
        cuda = types.SimpleNamespace(is_initialized=lambda: True, device_count=lambda: 1,
                                     max_memory_reserved=lambda device: reserved[0],
                                     get_device_properties=lambda device: types.SimpleNamespace(total_memory=32 << 30),
                                     mem_get_info=lambda device: (free[0], 32 << 30), empty_cache=lambda: None)
        import torch

        class Fake(types.ModuleType):   # torch, with the CUDA memory figures above: the toy still computes in torch
            def __getattr__(self, name):
                return getattr(torch, name)
        fake = Fake("torch")
        fake.cuda = cuda
        for wddm, warned in ((False, 0), (True, 1)):
            err = io.StringIO()
            reserved[0] = 10 << 30
            engine = self.open()   # the toy computes in torch: only the memory check reads the fake
            with mock.patch.dict(sys.modules, {"torch": fake}), mock.patch.object(native, "WDDM", wddm), \
                    contextlib.redirect_stderr(err):
                engine.generate([5], 1, stop_tokens=[])
                self.assertEqual(err.getvalue(), "")
                reserved[0] = 39 << 30   # past the device's 32 GiB: the driver spills the excess to system memory
                engine.generate([5], 1, stop_tokens=[])
                engine.score([5, 6], 1)
            self.assertEqual(err.getvalue().count("which holds 32.0 GiB"), warned)
        # Below the device's size, but other processes (a desktop) hold the rest: the device is full all the same.
        err = io.StringIO()
        reserved[0], free[0] = 30 << 30, 16 << 20
        engine = self.open()
        with mock.patch.dict(sys.modules, {"torch": fake}), mock.patch.object(native, "WDDM", True), \
                contextlib.redirect_stderr(err):
            engine.generate([5], 1, stop_tokens=[])
            engine.generate([5], 1, stop_tokens=[])
        self.assertEqual(err.getvalue().count("left 16 MiB free on cuda:0"), 1)

    def test_eval_v1_serves_the_package(self):
        evaluated = eval_v1({"model": str(self.model), "config": {"package": str(self.package), "label": "x"},
                             "context_tokens": 2048})
        self.addCleanup(evaluated.close)
        info = evaluated.info()
        self.assertEqual((info["name"], info["max_context"], info["plugins"], info["speculative"]),
                         ("toy step model", 2048, USE, False))
        self.assertEqual(info["configuration"]["label"], "x")
        self.assertNotIn("package", info["configuration"])
        out = evaluated.generate([5, 6], 4, lambda tokens: None)
        self.assertEqual(out["tokens"], oracle([5, 6], 4, stops=()))
        # Core's facts of the request's rounds: one a decode row, nothing drafted.
        self.assertEqual(out["rounds"], 3)
        self.assertNotIn("accepted", out)
        self.assertGreater(evaluated.telemetry()["core"]["steps"], 0)
        self.assertEqual(evaluated.replay([5, 6], [9, 10], lambda tokens: None)["tokens"], [9, 10])
        sampled = evaluated.generate_sampled([5], 2, lambda tokens: None, {"temperature": 0.7, "top_p": 0.9},
                                             logprobs=2)
        self.assertEqual(sampled["logprobs"][0]["top"][0][1], -0.5)
        rows = evaluated.score([5, 6], [7], 2)
        self.assertEqual((len(rows["prompt"]), len(rows["continuation"])), (1, 1))
        self.assertFalse(hasattr(evaluated, "generate_plain"))

    def test_eval_v1_runs_crit_evals_concurrency_cohort_in_process(self):
        # Clients start together and send their prompts one after another; every reply is the serial greedy one.
        evaluated = eval_v1({"model": str(self.model), "config": {"package": str(self.package)}, "context_tokens": 2048})
        self.addCleanup(evaluated.close)
        self.assertGreater(evaluated.capacity(), 1)
        clients = [[[5, 6], [7]], [[8], [9, 10, 11]], [[12, 13]]]
        found = evaluated.cohort(clients, 4)
        self.assertEqual([(r["client"], r["index"]) for r in found["requests"]], [(0, 0), (0, 1), (1, 0), (1, 1), (2, 0)])
        for row in found["requests"]:
            prompt = clients[row["client"]][row["index"]]
            self.assertEqual(row["tokens"], oracle(prompt, 4, stops=()))
            self.assertEqual((row["prompt_tokens"], row["completion_tokens"], row["errors"]), (len(prompt), 4, []))
            self.assertTrue(row["started"] <= row["first"] <= row["last"] <= row["finished"] <= found["seconds"])

    def test_a_drafting_engine_decodes_plainly_on_request_and_eval_s_plain_engine_drafts_nothing(self):
        drafting = write_package(self.root / "drafting", drafting=True)
        evaluated = eval_v1({"model": str(self.model), "config": {"package": str(drafting)}})
        self.addCleanup(evaluated.close)
        self.assertTrue(evaluated.info()["speculative"])
        out = evaluated.generate([5, 6], 6, lambda tokens: None)
        plain = evaluated.generate_plain([5, 6], 6, lambda tokens: None)
        self.assertEqual(out["tokens"], plain["tokens"])
        self.assertEqual(out["tokens"], oracle([5, 6], 6, stops=()))
        self.assertLess(out["rounds"], plain["rounds"])
        # Each round's accepted drafts in the result, Eval's acceptance by depth; plain decoding reports none.
        self.assertEqual((len(out["accepted"]), sum(out["accepted"]) + out["rounds"]), (out["rounds"], 5))
        self.assertNotIn("accepted", plain)
        second = eval_v1({"model": str(self.model), "config": {"package": str(drafting)}, "speculative": False})
        self.addCleanup(second.close)
        self.assertEqual((second.info()["speculative"], second.info()["plugins"]["drafter"]), (False, None))

    def test_resident_serves_chat_requests_concurrently(self):
        context = types.SimpleNamespace(configuration={"label": "run"}, limits={"kv_tokens": 1024},
                                        model_file=lambda name: str(self.model / name))
        engine = resident(context, self.package)
        self.addCleanup(engine.close)
        self.assertEqual((engine.capacity, engine.logprobs, engine.scores, engine.plain), (1024, True, True, False))
        self.assertGreater(engine.max_concurrent_requests, 1)
        self.assertEqual(engine.effective_configuration()["plugins"]["use"], USE)
        request = {"request_id": "r", "candidate_id": "c", "package_sha256": "p", "limits": {"max_output_tokens": 4},
                   "sampling": {"temperature": 0.0}, "messages": [{"role": "user", "content": "hi"}]}
        replies = []
        threads = [threading.Thread(target=lambda: replies.append(
            engine.create_request(request).generate(request, threading.Event()))) for _ in range(3)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        self.assertEqual(len({reply["assistant"] for reply in replies}), 1)
        self.assertEqual(engine.facts["plugins"], USE)


class ConstraintTests(PackageFixture, unittest.TestCase):
    """Core keeps a reply to a constraint for every engine: its matcher masks each row of a constrained request, the
    model plugin hands the mask to its sampler, and Core checks the token the row produced."""
    SCHEMA = {"type": "object", "properties": {"ok": {"type": "boolean"}, "n": {"enum": [3, 7]}},
              "required": ["ok", "n"], "additionalProperties": False}

    def open(self, configuration=None):
        from crit_core.constraints import byte_vocabulary
        from crit_core.model import open_model
        from crit_core.text import load_model_text
        engine = load(self.package, self.model, configuration, device="cpu")
        self.addCleanup(engine.close)
        self.text = load_model_text(open_model(self.model).files, clock=None)
        self.eos = core_engine.reply_end(self.text.tokenizer, [1, 4])
        engine.constrain(byte_vocabulary(self.text.tokenizer, self.eos))
        return engine

    def test_a_constrained_reply_keeps_to_its_language_and_ends_where_it_is_complete(self):
        from crit_core.constraints import compile_language, matches
        engine = self.open()
        for kind, source in (("schema", json.dumps(self.SCHEMA)), ("regex", "(yes|no), [a-c]{2}")):
            grammar = compile_language(kind, source)
            with self.subTest(kind=kind):
                tokens, result = engine.generate([5, 6, 7], 48, sampling={"temperature": 0.0},
                                                 constraint={"type": "json_schema", "grammar": grammar})
                self.assertEqual((result["finish"], tokens[-1]), ("stop", self.eos))
                self.assertTrue(matches(grammar, self.text.tokenizer.decode(tokens[:-1]).encode()))
        tokens, _ = engine.generate([5, 6, 7], 6, sampling={"temperature": 0.0}, constraint={"type": "json_object"})
        text = self.text.tokenizer.decode(tokens)
        self.assertTrue(text.lstrip().startswith("{") or not text.strip(), text)

    def test_constrained_requests_batch_with_free_ones(self):
        from crit_core.constraints import compile_language
        engine = self.open()
        grammar = {"type": "json_schema", "grammar": compile_language("schema", json.dumps(self.SCHEMA))}
        prompts = [[(r * 37 + i * 11) % VOCAB for i in range(1 + r * 3)] for r in range(8)]
        kept = lambda p, r: r % 2 and engine.generate(p, 24, sampling={"temperature": 0.0}, constraint=grammar)[0]
        alone = [kept(p, r) for r, p in enumerate(prompts)]
        with ThreadPoolExecutor(8) as pool:
            together = list(pool.map(lambda r: kept(prompts[r], r) or engine.generate(prompts[r], 12,
                                                                                      stop_tokens=[])[0], range(8)))
        for r, prompt in enumerate(prompts):
            self.assertEqual(together[r], alone[r] if r % 2 else oracle(prompt, 12, stops=()))

    def test_a_model_plugin_that_keeps_masks_from_its_sampler_is_refused(self):
        configuration = self.configuration()
        configuration["plugins"]["options"] = {"toy-model": {"skip": "masks"}}
        with self.assertRaisesRegex(PluginError, "its rows do not keep to Core's masks: a request allowed only token"):
            load(self.package, self.model, configuration, device="cpu")

    def test_a_token_outside_the_constraint_fails_its_request_alone(self):
        from unittest import mock
        configuration = self.configuration()
        configuration["plugins"]["options"] = {"toy-model": {"skip": "masks"}}
        with mock.patch.object(core_engine, "slot_problems", return_value=[]):
            engine = self.open(configuration)
        with self.assertRaisesRegex(NativeError, "the model plugin produced token .*StepRow.allowed"):
            engine.generate([5, 6, 7], 8, sampling={"temperature": 0.0}, constraint={"type": "json_object"})
        self.assertEqual(engine.generate([5, 6, 7], 6, stop_tokens=[])[0], oracle([5, 6, 7], 6, stops=()))

    def test_resident_keeps_a_reply_to_its_response_format(self):
        from crit_core.constraints import compile_language
        caps = capabilities(self.package, self.model)
        self.assertTrue({"chat-constraints-v1", "chat-constraints-grammar-v1"} <= set(caps), caps)
        context = types.SimpleNamespace(configuration={}, limits={"kv_tokens": 1024},
                                        model_file=lambda name: str(self.model / name))
        engine = resident(context, self.package)
        self.addCleanup(engine.close)
        self.assertTrue(engine.constraints and engine.grammars)
        constraint = {"type": "json_schema", "name": "r", "schema": self.SCHEMA,
                      "grammar": compile_language("schema", json.dumps(self.SCHEMA))}
        request = {"request_id": "r", "candidate_id": "c", "package_sha256": "p", "limits": {"max_output_tokens": 32},
                   "sampling": {"temperature": 0.0}, "messages": [{"role": "user", "content": "hi"}],
                   "response_format": constraint}
        reply = engine.create_request(request).generate(request, threading.Event())
        value = json.loads(reply["assistant"])
        self.assertEqual((set(value), value["n"] in (3, 7), type(value["ok"])), ({"ok", "n"}, True, bool))

    def test_resident_forces_the_calls_a_request_asks_for(self):
        from test_core_serving import TEMPLATE   # a chat template that teaches a call syntax crit writes
        model = write_files(self.root / "tools", llama_files(**{"chat_template.jinja": TEMPLATE, "config.json": {
            "model_type": "llama", "eos_token_id": 1, "text_config": {"eos_token_id": 4, "vocab_size": VOCAB}}}))
        self.assertIn("chat-tool-choice-v1", capabilities(self.package, model))
        context = types.SimpleNamespace(configuration={}, limits={"kv_tokens": 1024},
                                        model_file=lambda name: str(model / name))
        engine = resident(context, self.package)
        self.addCleanup(engine.close)
        tools = [{"type": "function", "function": {"name": "ping", "parameters": {"type": "object", "properties": {}}}}]
        request = {"request_id": "r", "candidate_id": "c", "package_sha256": "p", "limits": {"max_output_tokens": 64},
                   "sampling": {"temperature": 0.0}, "messages": [{"role": "user", "content": "hi"}], "tools": tools,
                   "tool_choice": "required"}
        reply = engine.create_request(request).generate(request, threading.Event())
        self.assertEqual((reply["assistant"], [c["function"] for c in reply["tool_calls"]]),
                         ("", [{"name": "ping", "arguments": "{}"}]))


# A chat template that writes <|im_start|> (the toy-vision placeholder, token 5) where a message holds an image.
VISION_TEMPLATE = ("{%- for m in messages %}{{ m.role }}:{% if m.content is string %}{{ m.content }}{% else %}"
                   "{% for p in m.content %}{% if p.type == 'image' %}<|im_start|>{% else %}{{ p.text }}{% endif %}"
                   "{% endfor %}{% endif %};{% endfor %}{% if add_generation_prompt %}assistant:{% endif %}")


class ImageTests(PackageFixture, unittest.TestCase):
    """Images reach a model plugin that takes them (crit_plugins.IMAGES): Core decodes the request's data: URLs, the
    model plugin lays them out in the rendered prompt (prepare), and its inputs reach open() with the request."""

    def setUp(self):
        super().setUp()
        self.vision = write_package(self.root / "vision", model="toy-vision")
        self.seeing = write_files(self.root / "seeing", llama_files(**{"chat_template.jinja": VISION_TEMPLATE,
                                                                      "config.json": {
            "model_type": "llama", "eos_token_id": 1, "text_config": {"eos_token_id": 4, "vocab_size": VOCAB}}}))
        context = types.SimpleNamespace(configuration={}, limits={"kv_tokens": 1024},
                                        model_file=lambda name: str(self.seeing / name))
        self.engine = resident(context, self.vision)
        self.addCleanup(self.engine.close)

    def ask(self, *images, url=None):
        import base64
        parts = [{"type": "text", "text": "look"}] + [
            {"type": "image_url", "image_url": {"url": url or "data:image/png;base64," + base64.b64encode(data).decode(),
                                                "detail": "auto"}} for data in images]
        request = {"request_id": "r", "candidate_id": "c", "package_sha256": "p", "limits": {"max_output_tokens": 6},
                   "sampling": {"temperature": 0.0}, "messages": [{"role": "user", "content": parts}]}
        return self.engine.create_request(request).generate(request, threading.Event())

    def test_only_a_model_plugin_that_takes_images_serves_them(self):
        self.assertIn("chat-images-v1", capabilities(self.vision, self.seeing))
        self.assertNotIn("chat-images-v1", capabilities(self.package, self.seeing))
        self.assertTrue(self.engine.images)

    def test_what_the_engine_writes_depends_on_the_image(self):
        from crit_core.model import open_model
        from crit_core.text import load_model_text
        first, again, other = self.ask(bytes([16, 32, 48])), self.ask(bytes([16, 32, 48])), self.ask(bytes([153, 1]))
        self.assertEqual(first["assistant"], again["assistant"])
        self.assertNotEqual(first["assistant"], other["assistant"])
        # The image's positions read its bytes: the reply is the toy's continuation of the prompt with them in place
        # of the placeholder.
        text = load_model_text(open_model(self.seeing).files, clock=None)
        ids = text.encode_chat([{"role": "user", "content": [{"type": "text", "text": "look"},
                                                             {"type": "image", "image": "x"}]}],
                               add_generation_prompt=True)
        at = ids.index(5)
        expected = oracle(ids[:at] + [1 + b % (VOCAB - 1) for b in (0x10, 0x20, 0x30)] + ids[at + 1:], 6)
        reply = text.tokenizer.decode([t for t in expected if t not in (1, 4)], skip_special_tokens=True)
        self.assertEqual(first["assistant"], reply)

    def test_an_image_url_core_would_have_to_fetch_is_refused(self):
        from crit_core.serving import RequestRejected
        with self.assertRaisesRegex(RequestRejected, "data: URLs: it fetches no URL"):
            self.ask(b"cat", url="https://example.com/cat.png")

    def test_a_model_plugin_that_provides_images_without_prepare_is_refused(self):
        blind = write_package(self.root / "blind", model="toy-vision")
        model = blind / "plugins" / "toy-vision" / "plugin.py"
        source = model.read_bytes().decode().replace("    def prepare(", "    def _unused_prepare(")
        model.write_bytes(source.encode())
        with self.assertRaisesRegex(PluginError, "provides inputs:images but has no prepare"):
            load(blind, self.seeing, device="cpu")


if __name__ == "__main__":
    unittest.main()


class PackageTuningTests(unittest.TestCase):
    def test_a_package_carries_its_kernel_tuning(self):
        from crit_plugins import kernels
        with tempfile.TemporaryDirectory() as temp:
            package = Path(temp)
            (package / "tuning").mkdir()
            (package / "tuning" / "mma-rows.json").write_text("{}", encoding="utf-8")
            self.addCleanup(kernels.use_tuning, None)
            self.assertEqual(core_engine.package_tuning(package, {"kernel_tuning": "tuning"}),
                             (package / "tuning").resolve())
            self.assertEqual(kernels.tuning_file("mma-rows"), (package / "tuning" / "mma-rows.json").resolve())
            for outside in ("..", "missing", 3):
                with self.subTest(outside=outside), self.assertRaisesRegex(PluginError, "inside the package"):
                    core_engine.package_tuning(package, {"kernel_tuning": outside})
            self.assertIsNone(core_engine.package_tuning(package, {}))


class ReleaseThresholdTests(unittest.TestCase):
    def test_the_release_threshold_is_the_engines_or_the_runs_headroom_never_a_constant(self):
        self.assertEqual(core_engine.release_threshold({}, {"vram_headroom_bytes": 3 << 30}), 3 << 30)
        self.assertEqual(core_engine.release_threshold({"release_below_mib": 640}, {"vram_headroom_bytes": 1}), 640 << 20)
        self.assertIsNone(core_engine.release_threshold({"release_below_mib": 0}, {"vram_headroom_bytes": 1 << 30}))
        self.assertIsNone(core_engine.release_threshold({}, {}))
