"""Core's engine loader (crit_core.engine): a package is its plugins and the configuration that selects them, and an
engine is Core running what the selection assembles. The loader refuses a package that is not a whole engine and a
model plugin that cannot batch or serve the contract, runs the toy package (inference/starters/plugin-engine/plugins) on Core's concurrent
runtime, and Core's eval-v1 and chat-resident-v1 adapters serve it with its plugins in their facts."""
import json
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import shutil
import sys
import tempfile
import threading
import types
import unittest

ROOT = Path(__file__).resolve().parents[1]
for _folder in ("inference/core/python", "tests"):
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

    def test_a_configuration_asks_for_exactness_by_level(self):
        sys.path.insert(0, str(ROOT / "kernels"))   # crit_kernels.exactness, the levels
        from crit_core.engine import exactness_levels
        self.assertIsNone(exactness_levels({}))
        self.assertEqual(exactness_levels({"exactness": "exact"}), {"decode": "row-exact", "prompt": "exact"})
        self.assertEqual(exactness_levels({"exactness": {"decode": "declared", "prompt": "repeatable"}}),
                         {"decode": "declared", "prompt": "repeatable"})
        for wrong in ("bitwise", {"prompt": "exact"}, {"decode": "exact", "prompt": "exact", "verify": "exact"}, 3,
                      ["exact"]):
            with self.subTest(wrong=wrong), self.assertRaisesRegex(PluginError, 'configuration "exactness"'):
                check_package(self.package, self.model, {**self.configuration(), "exactness": wrong}, device="cpu")
        check_package(self.package, self.model, {**self.configuration(), "exactness": "row-exact"}, device="cpu")

    def test_a_quality_mode_asks_its_levels_wherever_the_package_loads(self):
        # One resolution of the levels, crit run's and the loader's: a configuration that states a quality mode and no
        # exactness asks the mode's levels, and one that names a phase takes the mode's for the other.
        sys.path.insert(0, str(ROOT / "kernels"))
        from crit_core.engine import exactness_levels
        self.assertEqual(exactness_levels({"quality_mode": "speed"}), {"decode": "row-exact", "prompt": "row-exact"})
        self.assertEqual(exactness_levels({"quality_mode": "preserve"}), {"decode": "row-exact", "prompt": "exact"})
        self.assertEqual(exactness_levels({"quality_mode": "speed", "exactness": {"prompt": "repeatable"}}),
                         {"decode": "row-exact", "prompt": "repeatable"})
        with self.assertRaisesRegex(PluginError, 'configuration "quality_mode"'):
            exactness_levels({"quality_mode": "fast"})

    def test_the_run_s_vram_headroom_is_the_device_s_to_keep(self):
        # crit run writes the run's headroom into every candidate (vram_headroom_bytes); the loader gives it to the
        # device, which keeps it free at every moment of the engine's steps, and no plugin option sizes it.
        from crit_core.engine import headroom_of
        from crit_plugins import memory
        self.assertEqual(headroom_of({}), memory.HEADROOM)
        self.assertEqual(headroom_of({"vram_headroom_bytes": 1 << 30}), 1 << 30)
        for wrong in (-1, 0.5, "1GiB", True):
            with self.subTest(wrong=wrong), self.assertRaisesRegex(PluginError, 'configuration "vram_headroom_bytes"'):
                check_package(self.package, self.model, {**self.configuration(), "vram_headroom_bytes": wrong},
                              device="cpu")

    def test_a_configuration_asks_prompt_activations_in_the_form_eval_reads(self):
        from crit_core.engine import activation_problems, activations_of
        self.assertIsNone(activations_of({}))
        self.assertEqual(activations_of({"prompt_activations": "stored"}), "stored")
        self.assertEqual(activations_of({"prompt_activations": {"nvfp4": "stored", "fp8": "fp8"}}),
                         {"nvfp4": "stored", "fp8": "fp8"})
        for wrong in ("int4", {"nvfp4": "int4"}, {}, 3):
            with self.subTest(wrong=wrong), self.assertRaisesRegex(PluginError, 'configuration "prompt_activations"'):
                check_package(self.package, self.model, {**self.configuration(), "prompt_activations": wrong},
                              device="cpu")
        # What the engine's facts serve against what is asked: one activation for every format, or one each.
        served = {"prompt_activations": {"nvfp4": "stored", "mxfp8": "bf16"},
                  "chosen": {"mxfp8_activations": "mxfp8_activations auto by quality mode preserve"}}
        self.assertEqual(activation_problems(None, served), [])
        self.assertEqual(activation_problems({"nvfp4": "stored", "mxfp8": "bf16"}, served), [])
        self.assertEqual(activation_problems("stored", {}), [])   # no quantized format's prompt rows
        problem, = activation_problems("stored", served)
        self.assertIn('multiply {"mxfp8": "bf16", "nvfp4": "stored"}', problem)
        self.assertIn("mxfp8_activations auto by quality mode preserve", problem)
        self.assertIn('asks "stored"', problem)

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
        before = engine.telemetry()["core"]["prompt_tokens"]   # the load's own steps prefill too
        engine.generate(list(range(5, 45)), 4, stop_tokens=[])
        probe = engine.telemetry()["core"]["probe"]
        self.assertEqual(set(probe), {"prompt", "decode"})
        # Probed or not, Core counts the prompt tokens its steps prefilled: a prefill's rate while its request runs.
        self.assertEqual(engine.telemetry()["core"]["prompt_tokens"] - before, 40)
        for kind in ("prompt", "decode"):
            self.assertGreater(probe[kind]["steps"], 0)
            self.assertGreater(probe[kind]["slowest_seconds"], 0)

    def test_an_engine_told_to_allocate_nothing_fails_at_the_step_that_allocates_while_it_serves(self):
        # STEP_ALLOCATIONS=0: a step during which the device's allocator made an allocation ends the engine with
        # that step's kind and count, once the engine serves (its load's steps may allocate).
        import os
        from unittest import mock
        from crit_core import native

        device, count = native.memory_of("cuda"), [0, 0]   # the allocations so far, and what each reading adds

        def made():
            count[0] += count[1]
            return count[0]
        patches = (mock.patch.dict(os.environ, {native.STEP_ALLOCATIONS: "0"}),
                   mock.patch.object(device, "made", made), mock.patch.object(device, "serving", True))
        with patches[0], patches[1], patches[2]:
            engine = self.open()
            self.assertEqual(engine.generate([5, 6, 7], 4, stop_tokens=[])[0], oracle([5, 6, 7], 4, stops=()))
            device.serving, count[1] = False, 3      # a load's steps may allocate
            self.assertEqual(engine.generate([5, 6, 7], 4, stop_tokens=[])[0], oracle([5, 6, 7], 4, stops=()))
            device.serving = True
            with self.assertRaisesRegex(Exception, "a prompt step made 3 device allocations while the engine serves"):
                engine.generate([5, 6, 7], 4, stop_tokens=[])

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
        # Core runs one step at a time (crit/step_engine.h, StepDecision): a prompt chunk it ran (its state written in
        # place, as a replayed one's is) is kept, or its request ends. Here under stops, output limits, cancellations and prompts
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
        # what Core's reading of the device (crit_plugins.memory) asks torch for
        cuda = types.SimpleNamespace(is_initialized=lambda: True, is_available=lambda: True, device_count=lambda: 1,
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
        # Eval's result carries the engine's spec, each value with its source (crit_core.spec)
        self.assertEqual((info["spec"]["context_tokens"], info["spec"]["cancel_grace_seconds"]["source"]),
                         ({"value": 2048, "source": "user"}, "core"))
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
        # A sampled cohort (crit-eval's concurrency speed run): each request at its own sampling, as generate_sampled
        # decodes it alone.
        sampling = [[{"temperature": 1.0, "seed": 10 * c + r, "presence_penalty": 0.0} for r in range(len(row))]
                    for c, row in enumerate(clients)]
        drawn = evaluated.cohort(clients, 4, sampling)
        for row in drawn["requests"]:
            prompt, chosen = clients[row["client"]][row["index"]], sampling[row["client"]][row["index"]]
            alone = evaluated.generate_sampled(prompt, 4, lambda tokens: None, chosen)
            self.assertEqual(row["tokens"], alone["tokens"])
        # Each request at its own length (crit-eval's case requests, max_new_tokens[c][r]).
        lengths = [[3, 5], [2, 4], [6]]
        sized = evaluated.cohort(clients, lengths)
        for row in sized["requests"]:
            prompt, length = clients[row["client"]][row["index"]], lengths[row["client"]][row["index"]]
            self.assertEqual(row["tokens"], oracle(prompt, length, stops=()))
            self.assertEqual(row["completion_tokens"], length)

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

    def test_core_counts_the_drafts_of_the_requests_it_retires(self):
        # Core's own count of drafting (telemetry core.drafts), which crit run serve --check judges whatever the model
        # plugin is: the rounds that drafted, the drafts proposed where every round's count is known, those accepted.
        engine = load(write_package(self.root / "drafting-counts", drafting=True), self.model, device="cpu")
        self.addCleanup(engine.close)
        before = engine.telemetry()["core"].get("drafts") or {}   # the warm requests drafted at load
        _, result = engine.generate([5, 6], 6, stop_tokens=[])
        after = engine.telemetry()["core"]["drafts"]
        self.assertEqual(after["requests"] - before.get("requests", 0), 1)
        self.assertEqual(after["rounds"] - before.get("rounds", 0), len(result["accepted"]))
        self.assertEqual(after["accepted"] - before.get("accepted", 0), sum(result["accepted"]))
        self.assertGreater(after["accepted"], 0)
        engine.generate([5, 6], 4, stop_tokens=[], plain=True)   # a plain request drafts nothing
        self.assertEqual(engine.telemetry()["core"]["drafts"]["requests"], after["requests"])

    def test_the_loader_gives_the_device_the_configured_headroom_and_frames_every_step_with_it(self):
        from unittest import mock
        from crit_plugins import memory
        device = memory.of("cpu")
        self.addCleanup(setattr, device, "headroom", device.headroom)
        seen = []
        begin, end = device.begin_step, device.end_step
        with mock.patch.object(device, "begin_step", lambda tokens, score=False: (seen.append(("begin", tokens)),
                                                                                  begin(tokens, score))), \
                mock.patch.object(device, "end_step", lambda open_=False: (seen.append(("end", open_)), end(open_))):
            engine = load(self.package, self.model, {**self.configuration(), "vram_headroom_bytes": 3 << 28},
                          device="cpu")
            try:
                self.assertEqual(device.headroom, 3 << 28)
                del seen[:]
                engine.generate([5, 6, 7], 2, stop_tokens=[])
                self.assertEqual(seen[0], ("begin", 3))   # the prompt step, then its decode steps
                self.assertIn(("end", False), seen)
            finally:
                engine.close()

    def test_the_warm_requests_reach_every_length_the_model_says_its_paths_switch_at(self):
        # A sparse attention selects its blocks past a budget no short warm prompt reaches: its kernels compiled on
        # the first long request. A model states such lengths (info's warm_prompts) and Core serves each at load.
        from crit_core.engine import WARM_PROMPTS, warm_requests

        class Stub:
            def __init__(self, info):
                self._info, self.lengths = info, []

            def info(self):
                return self._info

            def generate(self, prompt, output_tokens, **options):
                self.lengths.append(len(prompt))
                return [1] * output_tokens, {}

            def score(self, *args, **kwargs):
                return None
        stub = Stub({"vocabulary": 100, "context": 8192, "warm_prompts": [3109, 3109, 0, "x"]})
        ran = warm_requests(stub)
        self.assertIn("prompt:3109", ran)
        self.assertEqual(stub.lengths.count(3109), 1)
        for length in WARM_PROMPTS:
            self.assertIn(f"prompt:{length}", ran)
        short = Stub({"vocabulary": 100, "context": 2048, "warm_prompts": [3109]})   # bounded by the context
        self.assertIn(f"prompt:{2048 - 4 - 1}", warm_requests(short))

    def test_the_warm_requests_run_the_largest_prompt_and_score_steps_the_arena_is_measured_by(self):
        # The step arena is what the load's steps took: an 8-token score left a long score's logits and their ranking
        # at a full slice of rows unmeasured. Core serves a prompt and a scored text of the longest chunk a budget gives.
        from crit_core.engine import TOP_LOGPROBS, WARM_PROMPTS, warm_requests

        class Stub:
            def __init__(self, context):
                self.context, self.prompts, self.scores = context, [], []

            def info(self):
                return {"vocabulary": 100, "context": self.context}

            def generate(self, prompt, output_tokens, **options):
                self.prompts.append(len(prompt))
                return [1] * output_tokens, {}

            def score(self, ids, first_target, path="", top_k=0, **options):
                self.scores.append((len(ids), top_k))
        stub = Stub(8192)
        ran = warm_requests(stub, largest=2048)
        self.assertIn((2048, TOP_LOGPROBS), stub.scores)
        self.assertIn(2048, stub.prompts)
        self.assertIn("score:2048", ran)
        stub = Stub(8192)
        warm_requests(stub)   # no budget: a score as long as the longest warm prompt
        self.assertIn((WARM_PROMPTS[-1], TOP_LOGPROBS), stub.scores)
        stub = Stub(1024)
        warm_requests(stub, largest=2048)   # within the context
        self.assertIn((1023, TOP_LOGPROBS), stub.scores)

    def test_core_watches_what_an_engine_compiles_once_its_load_is_over(self):
        try:
            import torch.utils.cpp_extension as extension
        except ImportError:
            self.skipTest("torch is not installed")
        original = extension._jit_compile
        extension._jit_compile = lambda name, *args, **kwargs: f"built {name}"   # stands in for a real build
        self.addCleanup(setattr, extension, "_jit_compile", original)
        engine = load(self.package, self.model, device="cpu")
        try:
            self.assertEqual(engine.telemetry()["core"]["compiles_after_load"], {"count": 0, "first": []})
            self.assertEqual(extension._jit_compile("crit_toy_kernels"), "built crit_toy_kernels")
            self.assertEqual(engine.telemetry()["core"]["compiles_after_load"],
                             {"count": 1, "first": ["torch extension build or load of crit_toy_kernels"]})
        finally:
            engine.close()
        self.assertEqual(extension._jit_compile("again"), "built again", "the builder is put back at close")

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


class PackageExactnessTests(PackageFixture, unittest.TestCase):
    """The exactness a package asks reaches its plugins before any is made, the engine's facts carry what is asked and
    what the plugins keep, and an engine whose plugin keeps less than it asks is refused."""

    def setUp(self):
        super().setUp()
        sys.path.insert(0, str(ROOT / "kernels"))   # crit_kernels.exactness, the levels
        from crit_plugins import kernels
        self.kernels = kernels
        self.addCleanup(kernels.use_exactness, None)

    def test_the_quality_mode_is_told_to_the_plugins(self):
        self.addCleanup(self.kernels.use_quality_mode, None)
        self.assertEqual(core_engine.package_quality_mode({"quality_mode": "speed"}), "speed")
        self.assertEqual(self.kernels.quality_mode(), "speed")
        self.assertIsNone(core_engine.package_quality_mode({}))   # a package that states none: every option decides
        self.assertIsNone(self.kernels.quality_mode())
        with self.assertRaisesRegex(PluginError, 'configuration "quality_mode" is one of preserve, speed, '
                                                 "not 'fast'"):
            core_engine.package_quality_mode({"quality_mode": "fast"})

    def test_the_levels_are_told_to_the_plugins(self):
        self.assertEqual(core_engine.package_exactness({"exactness": {"decode": "row-exact", "prompt": "repeatable"}}),
                         {"decode": "row-exact", "prompt": "repeatable"})
        self.assertEqual(self.kernels.asks("prompt"), "repeatable")
        # a quality mode alone asks its levels (row-exact in speed, exact in preserve), as crit run completes a candidate
        self.assertEqual(core_engine.package_exactness({"quality_mode": "speed"}),
                         {"decode": "row-exact", "prompt": "row-exact"})
        self.assertEqual(self.kernels.asks("prompt"), "row-exact")
        self.assertEqual(core_engine.package_exactness({"quality_mode": "preserve"}),
                         {"decode": "row-exact", "prompt": "exact"})
        self.assertIsNone(core_engine.package_exactness({}))
        self.assertIsNone(self.kernels.exactness())

    def test_the_asked_prompt_activations_are_told_to_the_plugins(self):
        self.addCleanup(self.kernels.use_activations, None)
        self.assertEqual(core_engine.package_activations({"prompt_activations": {"nvfp4": "stored", "mxfp8": "bf16",
                                                                                 "fp8": {"fp8": 1.0}}}),
                         {"nvfp4": "stored", "mxfp8": "bf16", "fp8": {"fp8": 1.0}})
        self.assertEqual(self.kernels.activation("nvfp4"), "stored")
        self.assertEqual(self.kernels.activation("mxfp8"), "bf16")
        self.assertIsNone(self.kernels.activation("fp8"))     # a mix asks nothing of the format
        self.assertIsNone(self.kernels.activation("int4_zp"))
        core_engine.package_activations({"prompt_activations": "fp8"})
        self.assertEqual(self.kernels.activation("nvfp4"), "fp8")
        self.assertIsNone(core_engine.package_activations({}))
        self.assertIsNone(self.kernels.activation("nvfp4"))

    @unittest.skipUnless(STEP_LIBRARY, "Core's runtime crit_core_steps is not built: `crit core build` builds it")
    def test_an_engine_says_what_it_asks_and_one_that_keeps_less_is_refused(self):
        from unittest import mock
        from crit_plugins.assemble import Assembly
        asked = {"decode": "row-exact", "prompt": "exact"}
        configuration = {**self.configuration(), "exactness": "exact"}
        engine = load(self.package, self.model, configuration, device="cpu")
        try:   # the toy's plugins state no level: the engine says what it asks, in its own policy
            self.assertEqual(engine.facts["exactness"], {"asked": asked})
            self.assertEqual(engine.policy.exactness, asked)
            self.assertIsNone(self.kernels.exactness())   # outside the engine the process asks nothing
        finally:
            engine.close()
        kept = {"asked": asked, "decode": "row-exact", "prompt": "repeatable",
                "plugins": {"toy-projections": {"decode": "row-exact", "prompt": "repeatable"}}}
        with mock.patch.object(Assembly, "exactness", lambda assembly: kept):
            with self.assertRaisesRegex(PluginError, "keeps less than it asks: toy-projections keeps its prompt rows "
                                                     "repeatable where the configuration's exactness asks exact"):
                load(self.package, self.model, configuration, device="cpu")
        engine = load(self.package, self.model, device="cpu")   # a package that asks nothing
        try:
            self.assertNotIn("exactness", engine.facts)
            self.assertIsNone(engine.policy.exactness)
        finally:
            engine.close()

    @unittest.skipUnless(STEP_LIBRARY, "Core's runtime crit_core_steps is not built: `crit core build` builds it")
    def test_an_engine_that_serves_other_prompt_activations_than_asked_is_refused_at_load(self):
        from unittest import mock
        from crit_plugins.assemble import Assembly
        self.addCleanup(self.kernels.use_activations, None)
        facts = Assembly.facts
        served = {"nvfp4": "stored", "mxfp8": "bf16"}

        def serving(assembly):
            return {**facts(assembly), "prompt_activations": dict(served)}
        configuration = {**self.configuration(), "prompt_activations": "stored"}
        with mock.patch.object(Assembly, "facts", serving):
            with self.assertRaisesRegex(PluginError, 'the engine is not what it is asked: its prompt rows multiply '
                                                     r'\{"mxfp8": "bf16", "nvfp4": "stored"\}'):
                load(self.package, self.model, configuration, device="cpu")
            engine = load(self.package, self.model, {**configuration, "prompt_activations": served}, device="cpu")
            engine.close()

    @unittest.skipUnless(STEP_LIBRARY, "Core's runtime crit_core_steps is not built: `crit core build` builds it")
    def test_an_engine_that_is_not_what_its_plugins_resolved_before_the_load_is_refused(self):
        # The plugins' resolve() runs before any plugin is made (no weights, no device): a load that makes another
        # engine than it resolved is refused, so the answer crit run checks a plan against is what every load gives.
        from unittest import mock
        self.assertEqual(core_engine.resolve_package(self.package, self.model, device="cpu"),
                         {"facts": {}, "by": {}})   # the toy's plugins resolve nothing
        wrong = {"facts": {"kv_bytes_per_token": 64}, "by": {"kv_bytes_per_token": "toy-model"}}
        with mock.patch.object(core_engine, "resolve_selection", lambda *args: wrong):
            with self.assertRaisesRegex(PluginError, "not what its plugins resolved it to be before the load: "
                                                     "toy-model's resolve\\(\\) states kv_bytes_per_token 64, and "
                                                     "the loaded engine None"):
                load(self.package, self.model, device="cpu")
        right = {"facts": {"plugins": self.configuration()["plugins"]["use"]}, "by": {"plugins": "toy-model"}}
        with mock.patch.object(core_engine, "resolve_selection", lambda *args: right):
            engine = load(self.package, self.model, device="cpu")
        try:
            self.assertEqual(engine.telemetry()["load"]["resolved"], right["facts"])
        finally:
            engine.close()

    @unittest.skipUnless(STEP_LIBRARY, "Core's runtime crit_core_steps is not built: `crit core build` builds it")
    def test_the_engine_spec_gives_every_value_its_source_and_the_load_records_it(self):
        # One description of the engine before any weight (crit_core.spec): a value crit run wrote from the user's
        # request says user (the configuration's provenance), one the package names says package, a default says
        # default, Core's bounds say core; the load records the spec it was held to.
        from crit_core.spec import differences, engine_spec, plain
        from crit_plugins.memory import HEADROOM, HOST_HEADROOM
        from crit_plugins.steps import GRACE
        configuration = {**self.configuration(), "vram_headroom_bytes": 1 << 30, "quality_mode": "speed",
                         "provenance": {"vram_headroom_bytes": "user"}}
        spec = engine_spec(self.package, self.model, configuration, context_tokens=512, device="cpu")
        sources = {name: item["source"] for name, item in spec["values"].items()}
        self.assertEqual((sources["vram_headroom_bytes"], sources["quality_mode"], sources["ram_headroom_bytes"],
                          sources["cancel_grace_seconds"], sources["context_tokens"], sources["exactness"]),
                         ("user", "package", "default", "core", "user", "mode"))
        values = plain(spec)
        self.assertEqual((values["vram_headroom_bytes"], values["ram_headroom_bytes"], values["cancel_grace_seconds"],
                          values["context_tokens"]), (1 << 30, HOST_HEADROOM, GRACE, 512))
        self.assertEqual(plain(engine_spec(self.package, self.model, device="cpu"))["vram_headroom_bytes"], HEADROOM)
        self.assertEqual({item["source"] for options in spec["options"].values() for item in options.values()} -
                         {"default", "mode", "package"}, set())
        with self.assertRaisesRegex(ValueError, "has the source 'guessed'"):
            engine_spec(self.package, self.model, {**configuration, "provenance": {"x": "guessed"}}, device="cpu")
        engine = load(self.package, self.model, configuration, context_tokens=512, device="cpu")
        try:
            self.assertEqual(engine.telemetry()["load"]["spec"]["values"], spec["values"])
        finally:
            engine.close()
        resolved = {**spec, "values": {**spec["values"], "kv_bytes_per_token": {"value": 64, "source": "resolved",
                                                                                 "by": "toy-model"}}}
        self.assertEqual(differences(resolved, {"kv_bytes_per_token": 64}), [])
        self.assertIn("kv_bytes_per_token", differences(resolved, {"kv_bytes_per_token": 80})[0])
        # a price sizes the engine before it exists and holds no load; the engine serves the spec's context at least
        priced = {**resolved, "values": {**resolved["values"], "held_matrix_bytes": {"value": 10, "source": "resolved",
                                                                                    "by": "projections"}}}
        self.assertEqual(differences(priced, {"kv_bytes_per_token": 64}), [])
        self.assertEqual(differences(resolved, {"kv_bytes_per_token": 64, "context": 512}), [])
        self.assertIn("the engine serves 256 tokens of context, its spec asks 512",
                      differences(resolved, {"kv_bytes_per_token": 64, "context": 256})[0])
        from unittest import mock
        with mock.patch("crit_core.spec.differences", lambda spec, facts: ["context_tokens: fixture"]):
            with self.assertRaisesRegex(Exception, "the engine is not its spec .*context_tokens: fixture"):
                load(self.package, self.model, configuration, context_tokens=512, device="cpu")

    def test_a_kept_file_is_named_only_where_the_server_names_its_folder(self):
        # crit serve --one-process names the folder (CRIT_KEPT_FILES); elsewhere a kept store stays in the process
        from unittest import mock
        context = core_engine.LoadContext.__new__(core_engine.LoadContext)
        with mock.patch.dict(os.environ, {core_engine.KEPT_FILES: ""}):
            self.assertIsNone(context.kept_file("expert-tensors:store", {"a": 1}))
        with mock.patch.dict(os.environ, {core_engine.KEPT_FILES: "/srv/kept"}):
            found = context.kept_file("expert-tensors:store", {"a": 1})
            self.assertEqual((found.parent.as_posix(), found.suffix), ("/srv/kept", ".store"))
            self.assertTrue(found.name.startswith("expert-tensors-store-"))
            self.assertNotEqual(found, context.kept_file("expert-tensors:store", {"a": 2}))

    def test_crit_plugins_resolve_prints_what_the_plugins_answer_before_a_load(self):
        import contextlib
        import io
        from crit.plugins import main
        from crit.resolve import render
        printed = io.StringIO()
        with contextlib.redirect_stdout(printed):
            self.assertEqual(main(["resolve", str(self.package), "--model", str(self.model), "--json"]), 0)
        self.assertEqual(json.loads(printed.getvalue()), {"facts": {}, "by": {}, "context": None})
        self.assertIn("no plugin of the selection resolves anything", render({"facts": {}, "by": {}}))
        self.assertEqual(render({"facts": {"kv_bytes_per_token": 1 << 20}, "by": {"kv_bytes_per_token": "a + b"}},
                                2048),
                         "kv_bytes_per_token: 1048576  (a + b)\nKV at 2,048 tokens: 2.00 GiB (sliding windows' "
                         "rings apart)")


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
    def test_the_release_threshold_is_the_configurations_or_what_the_next_step_needs_never_a_constant(self):
        import types
        from crit_plugins.steps import Steps
        memory = types.SimpleNamespace(headroom=3 << 30, arena=1 << 30)
        self.assertEqual(Steps(memory=memory).threshold(), 4 << 30)   # the run's headroom and the step arena
        override = core_engine.release_override({"release_below_mib": 640})
        self.assertEqual(Steps(release_below=override, memory=memory).threshold(), 640 << 20)
        self.assertIsNone(Steps(release_below=core_engine.release_override({"release_below_mib": 0}),
                                memory=memory).threshold())
        self.assertIsNone(Steps().threshold())   # no device memory: nothing to hand back
        self.assertIsNone(core_engine.release_override({}))
        with self.assertRaises(PluginError):
            core_engine.release_override({"release_below_mib": -1})


class KeptStoreTests(PackageFixture, unittest.TestCase):
    """A host-side store a plugin builds from the checkpoint (LoadContext.keep) is kept for the next engine built for
    the model: by its keeper, one store a name, replaced (and closed first) when asked under another key."""

    def test_a_keeper_keeps_one_store_a_name_until_it_closes(self):
        from crit_core.kept import Kept
        kept, closed = Kept(), []
        first = kept.keep("store", {"layout": 1}, lambda: ["a"], closed.append, nbytes=64)
        kept.release("store")
        self.assertIs(kept.keep("store", {"layout": 1}, lambda: ["b"], closed.append), first)
        self.assertEqual(kept.report(), {"store": {"nbytes": 64, "users": 1, "reused": 1}})
        with self.assertRaisesRegex(ValueError, "held by an engine under another key"):
            kept.keep("store", {"layout": 2}, lambda: ["c"], closed.append)
        with self.assertRaisesRegex(ValueError, "an engine still holds: store"):
            kept.close()
        kept.release("store")
        other = kept.keep("store", {"layout": 2}, lambda: ["c"], closed.append)
        self.assertEqual((other, closed), (["c"], [["a"]]))   # the old one closed before the new one was built
        self.assertEqual(kept.nbytes("store"), 0)   # held: no engine to come has its memory
        kept.release("store")
        kept.keep("other", 1, lambda: ["d"], closed.append, nbytes=32)
        kept.release("other")
        self.assertEqual(kept.nbytes("other"), 32)
        kept.drop("other")   # an engine that will not ask for it frees it first
        self.assertEqual((kept.nbytes("other"), closed[-1]), (0, ["d"]))
        kept.close()
        self.assertEqual((closed, kept.report()), ([["a"], ["d"], ["c"]], {}))

    def test_a_load_context_keeps_its_stores_where_its_keeper_is(self):
        from crit_core.engine import LoadContext
        from crit_core.kept import Kept
        builds, closed = [], []

        def build():
            builds.append(1)
            return {"store": len(builds)}
        keeper = Kept()
        for _ in range(2):   # two engines in one process: the second gets the first one's store
            context = LoadContext(self.package, self.model, {}, device="cpu", kept=keeper)
            self.assertEqual(context.keep("toy:store", ["ckpt", 1], build, closed.append), {"store": 1})
            context.release_kept()
        self.assertEqual((len(builds), closed), (1, []))
        keeper.close()
        self.assertEqual(closed, [{"store": 1}])
        # Without a keeper the store is the engine's, closed with it.
        alone = LoadContext(self.package, self.model, {}, device="cpu")
        alone.keep("toy:store", ["ckpt", 1], build, closed.append)
        alone.release_kept()
        self.assertEqual(closed, [{"store": 1}, {"store": 2}])

    def test_crit_evals_engines_keep_stores_for_one_model_at_a_time(self):
        from crit_core.engine import eval_kept, eval_release
        self.addCleanup(eval_release)
        closed = []
        kept = eval_kept(self.model)
        self.assertIs(eval_kept(self.model), kept)
        kept.keep("toy:store", ["ckpt"], lambda: "a", closed.append)
        kept.release("toy:store")
        other = eval_kept(self.root / "another-model")   # another model: the first one's stores go
        self.assertIsNot(other, kept)
        self.assertEqual(closed, ["a"])
        other.keep("toy:store", ["ckpt"], lambda: "b", closed.append)
        other.release("toy:store")
        eval_release()
        self.assertEqual(closed, ["a", "b"])

    @unittest.skipUnless(STEP_LIBRARY, "needs Core's crit_core_steps library")
    def test_evals_next_engine_finds_the_store_its_plugin_kept(self):
        # The toy projections plugin keeps a store; crit-eval opens two engines of the package one after the other
        # (crit eval ab's rounds) and the second one's plugin gets the first one's store without building it.
        plugin = self.package / "plugins" / "toy-projections" / "plugin.py"
        builds = self.root / "builds.txt"
        plugin.write_text(plugin.read_text(encoding="utf-8").replace(
            "def create(context, options, parts):\n    return Projections()",
            "def create(context, options, parts):\n"
            "    def build():\n"
            f"        with open({str(builds)!r}, 'a') as out:\n"
            "            out.write('built' + chr(10))\n"
            "        return ['store']\n"
            "    context.keep('toy:store', ['ckpt'], build)\n"
            "    return Projections()"), encoding="utf-8")
        from crit_core.engine import eval_release
        self.addCleanup(eval_release)
        for _ in range(2):
            engine = eval_v1({"model": str(self.model), "config": {"package": str(self.package)}})
            engine.close()
        self.assertEqual(builds.read_text(encoding="utf-8"), "built\n")
