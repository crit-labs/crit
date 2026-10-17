"""crit eval harness: metrics, runner and comparison with the mock adapter (no GPU)."""
import hashlib
import json
import math
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
for _folder in ("core/python", "eval"): sys.path.insert(0, str(ROOT / _folder))
from crit_core.model import inspect_safetensors  # noqa: E402
from crit_eval import ab, ceiling, context, headroom, metrics, pin, prepare, probe, report, runner, status  # noqa: E402

MOCK = ROOT / "eval/examples/mock_adapter.py"


def softmax_logprobs(logits):
    top = max(logits)
    total = sum(math.exp(v - top) for v in logits)
    return [v - top - math.log(total) for v in logits]


def fixture(directory):
    prepared = {
        "protocol": "crit-eval-v3", "suite_sha256": "suite", "model": {"sha256": "model"}, "stop_ids": [999],
        "valid_vocab": 1000,
        "speed": [{"id": "short", "kind": "prose", "prompt_ids": [1, 2, 3], "output_tokens": 12},
                  {"id": "long", "kind": "long-context", "prompt_ids": list(range(60)), "output_tokens": 12}],
        "quality": [{"id": "c", "ids": [5, 6, 7, 8, 9], "first_target": 2}],
    }
    targets = []
    for position in range(2, 5):
        lp = softmax_logprobs([3.0, 1.0, 0.5, 0.0] + [-4.0] * 20)
        ids = [prepared["quality"][0]["ids"][position], 100 + position, 200 + position, 300 + position]
        targets.append({"target": ids[0], "logprob": lp[0], "top_ids": ids, "top_logprobs": lp[:4]})
    reference = {"prepared_model_sha256": "model", "suite_sha256": "suite", "top_k": 4,
                 "conversations": [{"id": "c", "targets": targets}]}
    (directory / "prepared.json").write_text(json.dumps(prepared))
    (directory / "reference.json").write_text(json.dumps(reference))
    return prepared, reference


class MetricTests(unittest.TestCase):
    def test_lumped_kl_is_zero_for_equal_and_a_lower_bound_of_full_kl(self):
        p = softmax_logprobs([2.0, 1.0, 0.0, -1.0, -2.0])
        q = softmax_logprobs([1.5, 1.2, 0.1, -0.5, -3.0])
        self.assertAlmostEqual(metrics.lumped_kl(p[:3], p[:3]), 0.0, places=12)
        full = sum(math.exp(a) * (a - b) for a, b in zip(p, q))
        lumped = metrics.lumped_kl(p[:3], q[:3])
        self.assertGreater(lumped, 0.0)
        self.assertLessEqual(lumped, full + 1e-12)

    def test_speed_sample_counts_decode_after_the_first_token(self):
        sample = metrics.speed_sample(100, list(range(11)), 0.0, 0.5, 1.5, 4)
        self.assertAlmostEqual(sample["decode_tps"], 10.0)
        self.assertAlmostEqual(sample["prefill_tps"], 200.0)
        self.assertAlmostEqual(sample["tokens_per_round"], 11 / 4)

    def test_round_ms_covers_decode_time_per_target_pass(self):
        self.assertAlmostEqual(metrics.speed_sample(100, list(range(11)), 0.0, 0.5, 1.5, 4)["round_ms"], 250.0)
        self.assertAlmostEqual(metrics.speed_sample(100, list(range(11)), 0.0, 0.5, 1.5, None)["round_ms"], 100.0)

    def test_headroom_reports_pass_efficiency_and_draft_saturation(self):
        thresholds = {}
        speed = {"cases": {"code": {"status": "measured", "tokens_per_round": 3.6, "round_ms": 25.0},
                           "chat": {"status": "measured", "tokens_per_round": 2.0, "round_ms": 24.0}}}
        info = {"draft_width": 3, "bytes_per_pass": 18e9}
        room = headroom.summarize(info, speed, 20.0, 1800, thresholds)
        self.assertAlmostEqual(room["pass_floor_ms"], 10.0)
        self.assertAlmostEqual(room["pass_efficiency"], 0.5)
        self.assertAlmostEqual(room["cases"]["code"]["saturation"], 0.9)
        self.assertIn("model pass 20.00 ms is 50% of its 10.00 ms bandwidth floor", room["warnings"])
        self.assertNotIn("pass_polished", room)  # Eval reports the numbers, not a judgment
        unmeasured = headroom.summarize({"draft_width": 3}, speed, 20.0, None, thresholds)
        self.assertIsNone(unmeasured["pass_efficiency"])
        self.assertTrue(any("bytes_per_pass" in w for w in unmeasured["warnings"]))

    def test_acceptance_projects_tokens_and_speed_by_width(self):
        # Rounds accepted 0, 1, 2 and 4 drafts at width 4.
        curve = headroom.tokens_by_width([0, 1, 2, 4], 4)
        self.assertEqual(curve, [1.0, 1.75, 2.25, 2.5, 2.75])
        # Pass 10 ms, draft 3 ms: width 2 gives 2.25 / 16 ms = 140.6 tok/s, the best.
        width, score = headroom.best_width([curve], 10.0, 3.0)
        self.assertEqual(width, 2)
        self.assertAlmostEqual(score, 140.625)
        self.assertEqual(headroom.best_width([curve], 10.0, 1.0)[0], 4)
        thresholds = {}
        samples = [{"warmup": False, "accepted": [0, 1, 2, 4]}]
        speed = {"decode_score": 100.0, "cases": {"code": {"status": "measured", "tokens_per_round": 2.75,
                                                            "round_ms": 14.0, "samples": samples}}}
        info = {"draft_width": 4, "bytes_per_pass": 10e9, "draft_bytes_per_step": 0.5e9}
        room = headroom.summarize(info, speed, 20.0, 1000, thresholds)
        projection = room["projection"]
        self.assertAlmostEqual(projection["draft_ms"], 0.0)  # round faster than the step: clamped
        self.assertEqual(projection["best_width_at_floor"], 4)  # 2.75 / 12 ms beats 2.5 / 11.5 ms
        self.assertAlmostEqual(projection["speed_of_light_score"], 1000 * 2.75 / 12.0)
        self.assertAlmostEqual(projection["fraction_of_speed_of_light"], 100.0 / (1000 * 2.75 / 12.0))
        # A drafter accepting every depth at the first depth's rate (0.75) is the decode ceiling:
        # (1 - 0.75^8) / 0.25 tokens over 10 + 7 * 0.5 ms at width 7.
        self.assertEqual(projection["acceptance_bound_width"], 7)
        self.assertAlmostEqual(projection["acceptance_bound_score"], 1000 * (1 - 0.75 ** 8) / 0.25 / 13.5)
        self.assertEqual(projection["ceiling_score"], projection["acceptance_bound_score"])
        self.assertTrue(any("of its ceiling" in w for w in room["warnings"]))
        self.assertTrue(any("widest measured" in w for w in room["warnings"]))

    def test_conditional_acceptance_and_a_uniform_drafter(self):
        self.assertEqual(headroom.conditional_acceptance([0, 1, 2, 4], 4), [0.75, 2 / 3, 0.5, 1.0])
        self.assertEqual(headroom.tokens_at_rate(0.5, 3), [1.0, 1.5, 1.75, 1.875])
        self.assertEqual(headroom.conditional_acceptance([0, 0], 2), [0.0, 0.0])

    def test_judge_reports_threshold_failures(self):
        self.assertEqual(metrics.judge({"ppl_ratio": 1.0, "kld_mean": 0.0}, {"ppl_ratio_max": 1.01, "kld_mean_max": 0.01}), [])
        self.assertEqual(len(metrics.judge({"ppl_ratio": 1.2, "kld_mean": 0.0}, {"ppl_ratio_max": 1.01})), 1)

    def test_budget_used_is_the_largest_share_of_any_limit(self):
        limits = {"ppl_ratio_max": 1.05, "kld_mean_max": 0.05, "kld_p99_max": 0.5}
        self.assertAlmostEqual(metrics.budget_used({"ppl_ratio": 1.01, "kld_mean": 0.005, "kld_p99": 0.1}, limits), 0.2)
        self.assertAlmostEqual(metrics.budget_used({"ppl_ratio": 0.99, "kld_mean": 0.0, "kld_p99": 0.0}, limits), 0.0)
        self.assertEqual(metrics.budget_used({"ppl_ratio": 1.0}, {"ppl_ratio_max": 1.0}), 0.0)
        self.assertEqual(metrics.budget_used({"ppl_ratio": 1.001}, {"ppl_ratio_max": 1.0}), float("inf"))


class RunnerTests(unittest.TestCase):
    def test_mock_engine_end_to_end(self):
        with tempfile.TemporaryDirectory() as directory:
            data = Path(directory)
            fixture(data)
            config = {"reference": str(data / "reference.json"), "noise": 0.2, "max_context": 64, "bytes_per_pass": 1e6,
                      "draft_bytes_per_step": 1e5}
            result = runner.run(MOCK, "unused-model", data, data / "result.json", config,
                                warmups=0, repetitions=2, log=lambda message: None, bandwidth_gbs=1.0)
            speed = result["speed"]
            self.assertEqual(speed["cases"]["short"]["status"], "measured")
            self.assertEqual(speed["cases"]["long"]["status"], "skipped")  # 60 + 12 > 64
            self.assertAlmostEqual(speed["cases"]["short"]["tokens_per_round"], 12 / 3)
            self.assertTrue(speed["cases"]["short"]["repeatable"])
            self.assertFalse(speed["complete"])
            prefill, decode = result["quality"]["paths"]["prefill"], result["quality"]["paths"]["decode"]
            self.assertAlmostEqual(prefill["kld_mean"], 0.0, places=12)
            self.assertTrue(prefill["passed"])
            self.assertFalse(decode["passed"])  # every target shifted by 0.2 nats
            self.assertEqual(result["quality"]["unsupported"], ["verify"])
            self.assertTrue(result["exactness"]["passed"])
            room = result["headroom"]
            self.assertEqual(room["draft_width"], 4)
            self.assertAlmostEqual(room["pass_floor_ms"], 1.0)
            self.assertGreater(room["step_ms"], 0)
            self.assertAlmostEqual(room["cases"]["short"]["saturation"], 12 / 3 / 5)
            self.assertEqual(room["cases"]["short"]["tokens_by_width"], [1.0, 2.0, 3.0, 4.0, 5.0])
            self.assertIsNotNone(room["projection"]["speed_of_light_score"])
            table = report.compare([data / "result.json", data / "result.json"])
            self.assertIn("Decode tok/s: short", table)
            self.assertIn("Quality: decode", table)
            self.assertIn("Model pass vs bandwidth floor", table)
            self.assertIn("Decode ceiling", table)

    def test_loading_an_adapter_writes_no_bytecode_into_its_package(self):
        with tempfile.TemporaryDirectory() as directory:
            package = Path(directory)
            (package / "helper_for_adapter.py").write_text("VALUE = 3\n")
            (package / "adapter.py").write_text("import helper_for_adapter\n"
                                               "def create_eval_engine(context):\n    return helper_for_adapter.VALUE\n")
            previous = sys.dont_write_bytecode
            sys.dont_write_bytecode = False
            try:
                module = runner.load_adapter(package / "adapter.py")
            finally:
                sys.dont_write_bytecode = previous
                sys.modules.pop("helper_for_adapter", None)
            self.assertEqual(module.create_eval_engine({}), 3)
            self.assertFalse(any(package.rglob("__pycache__")))

    def test_wrong_token_count_is_rejected(self):
        class Short:
            def generate(self, prompt, n, on_tokens):
                on_tokens([1])
                return {"tokens": [1]}
        with self.assertRaises(RuntimeError):
            runner.timed_generate(Short(), [1, 2], 4)


class CharacterText:
    """A stand-in for crit_core.text.ModelText whose tokens are characters."""
    class tokenizer:
        @staticmethod
        def encode(text, *, add_special_tokens, split_special_tokens=None):
            return [ord(c) for c in text]


class PrepareTests(unittest.TestCase):
    def test_model_identity_counts_index_shards_in_a_subfolder(self):
        with tempfile.TemporaryDirectory() as folder:
            model = Path(folder)
            (model / "tensors").mkdir()
            (model / "tensors" / "a.safetensors").write_bytes(b"x" * 5)
            (model / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {"w": "tensors/a.safetensors"}}))
            first = prepare.model_identity(model)
            self.assertEqual(first["files"]["weight_files"], {"tensors/a.safetensors": 5})
            (model / "tensors" / "a.safetensors").write_bytes(b"x" * 6)
            self.assertNotEqual(prepare.model_identity(model)["sha256"], first["sha256"])

    def test_long_context_prompt_has_the_exact_length(self):
        class Model(CharacterText):
            def render(self, messages, tools=None, **variables):
                return "<u>" + messages[-1]["content"] + "</u><a>"
        case = {"id": "long", "prompt_tokens": 400, "document": "long_document.txt",
                "messages": [{"role": "user", "content": "Summarize.\n\n{document}"}]}
        ids = prepare.speed_case(Model(), case)
        self.assertEqual(len(ids), 400)
        self.assertEqual(ids[-7:], [ord(c) for c in "</u><a>"])

    def test_agent_case_renders_tools_and_file_contents_and_fills_the_inline_document(self):
        seen = {}

        class Model(CharacterText):
            def render(self, messages, tools=None, **variables):
                seen["tools"] = tools
                return "".join(f"<{m['role']}>{m['content']}" for m in messages) + "<a>"
        case = {"id": "agent", "prompt_tokens": 9000, "tools": "agent/tools.json", "document": "agent/reference.py.txt",
                "messages": [{"role": "user", "content": "Task"},
                             {"role": "tool", "content_file": "agent/prepare.py.txt"},  # contains "{document}"
                             {"role": "tool", "content": "{document}"}]}
        ids = prepare.speed_case(Model(), case)
        self.assertEqual(len(ids), 9000)
        self.assertEqual([t["function"]["name"] for t in seen["tools"]], ["read_file", "grep", "edit_file", "run_command"])
        text = "".join(chr(i) for i in ids)
        self.assertIn('replace("{document}"', text)  # the file read is left as written
        self.assertTrue(text.endswith("<a>"))

    def test_suite_cases_and_protocol_prefill_cases_agree(self):
        suite = json.loads((ROOT / "eval/suite/speed.json").read_text(encoding="utf-8"))
        protocol = json.loads((ROOT / "eval/protocol.json").read_text(encoding="utf-8"))
        ids = [c["id"] for c in suite["cases"]]
        self.assertEqual(ids, ["chat", "code", "math", "tool-call", "long-summary", "agent-long"])
        self.assertTrue(set(protocol["speed"]["prefill_cases"]) <= set(ids))
        for case in suite["cases"]:
            for message in case["messages"]:
                if "content_file" in message:
                    self.assertTrue((ROOT / "eval/suite" / message["content_file"]).is_file())


def header_fixture():
    tensors = {
        "model.language_model.embed_tokens.weight": {"dtype": "BF16", "shape": [1000, 64]},
        "model.language_model.layers.0.mlp.up_proj.weight_packed": {"dtype": "U8", "shape": [128, 32]},
        "model.language_model.layers.0.mlp.up_proj.weight_scale": {"dtype": "F8_E4M3", "shape": [128, 4]},
        "model.language_model.layers.0.self_attn.q_proj.weight": {"dtype": "F8_E4M3", "shape": [64, 64]},
        "model.language_model.layers.0.self_attn.q_proj.weight_scale": {"dtype": "BF16", "shape": [64, 1]},
        "model.language_model.layers.0.linear_attn.in_proj_a.weight": {"dtype": "BF16", "shape": [4, 64]},
        "model.language_model.layers.0.input_layernorm.weight": {"dtype": "BF16", "shape": [64]},
        "model.language_model.layers.0.linear_attn.conv1d.weight": {"dtype": "BF16", "shape": [64, 1, 4]},
        "model.visual.blocks.0.mlp.fc1.weight": {"dtype": "BF16", "shape": [256, 64]},
        "lm_head.weight": {"dtype": "F8_E4M3", "shape": [1000, 64]},
    }
    size = {"BF16": 2, "F8_E4M3": 1, "U8": 1}
    offset = 0
    for tensor in tensors.values():
        count = 1
        for dim in tensor["shape"]:
            count *= dim
        tensor["data_offsets"] = [offset, offset + count * size[tensor["dtype"]]]
        offset += count * size[tensor["dtype"]]
    return tensors


def write_safetensors(path, tensors):
    """A safetensors file of zero bytes for each tensor of a header."""
    blob = json.dumps(tensors).encode()
    payload = max((t["data_offsets"][1] for t in tensors.values()), default=0)
    Path(path).write_bytes(len(blob).to_bytes(8, "little") + blob + bytes(payload))
    return Path(path)


class CeilingTests(unittest.TestCase):
    def test_matmul_flops_by_stored_format(self):
        with tempfile.TemporaryDirectory() as directory:
            shard = inspect_safetensors(write_safetensors(Path(directory) / "model.safetensors", header_fixture()))
        flops, matrices = ceiling.matmul_flops({tensor.name: tensor for tensor in shard.tensors})
        self.assertEqual(flops, {"nvfp4": 2 * 128 * 64, "fp8": 2 * 64 * 64, "bf16": 2 * 4 * 64})
        self.assertEqual(matrices, {"nvfp4": 1, "fp8": 1, "bf16": 1})

    def test_model_ceiling_reads_headers_and_config(self):
        with tempfile.TemporaryDirectory() as directory:
            model = Path(directory)
            write_safetensors(model / "model.safetensors", header_fixture())
            write_safetensors(model / "model_mtp.safetensors", header_fixture())  # draft: ignored
            config = {"text_config": {"num_hidden_layers": 4, "layer_types": ["linear_attention"] * 3 + ["full_attention"],
                                      "num_attention_heads": 2, "num_key_value_heads": 1, "head_dim": 8,
                                      "linear_num_value_heads": 2, "linear_key_head_dim": 4, "linear_value_head_dim": 4}}
            (model / "config.json").write_text(json.dumps(config))
            out = ceiling.model_ceiling(model)
        self.assertEqual(out["matmul_flops_per_token"]["nvfp4"], 2 * 128 * 64)
        self.assertEqual(out["attention_layers"], 1)
        self.assertEqual(out["attention_flops_per_pair"], 4 * 2 * 8)
        self.assertEqual(out["kv_bytes_per_token"], 2 * 1 * 8 * 2)
        self.assertEqual(out["recurrent_flops_per_token"], 3 * 2 * (4 * 4 * 4 + 2 * 64 * 8))
        # Everything but the embedding table and the vision tower is read once per pass.
        read = 128 * 32 + 128 * 4 + 64 * 64 + 64 * 2 + 4 * 64 * 2 + 64 * 2 + 64 * 4 * 2 + 1000 * 64
        self.assertEqual(out["weight_bytes"], read)
        self.assertIsNone(ceiling.model_ceiling(Path(directory) / "missing"))

    def test_prefill_floor_adds_matmul_attention_and_recurrent_time(self):
        model = {"matmul_flops_per_token": {"nvfp4": 4e9, "fp8": 2e9}, "attention_flops_per_pair": 1e5,
                 "recurrent_flops_per_token": 1e6}
        floor, missing = ceiling.prefill_floor_ms(model, {"nvfp4": 1000, "fp8": 500, "bf16": 200}, 1000)
        self.assertEqual(missing, [])
        self.assertAlmostEqual(floor["matmul_ms"], 1000 * (4e9 / 1e15 + 2e9 / 5e14) * 1000)
        # Attention runs at the faster FP8 rate (500 TFLOPS); the recurrence stays at BF16.
        self.assertAlmostEqual(floor["attention_ms"], 1000 * 1001 / 2 * 1e5 / 5e14 * 1000)
        self.assertAlmostEqual(floor["recurrent_ms"], 1000 * 1e6 / 2e14 * 1000)
        self.assertAlmostEqual(floor["total_ms"], floor["matmul_ms"] + floor["attention_ms"] + floor["recurrent_ms"])
        self.assertEqual(ceiling.prefill_floor_ms(model, {"nvfp4": 1000, "bf16": 200}, 10)[1], ["fp8 GEMM rate"])
        bf16_only = {"matmul_flops_per_token": {"bf16": 1e9}, "attention_flops_per_pair": 1e5}
        self.assertAlmostEqual(ceiling.prefill_floor_ms(bf16_only, {"bf16": 200}, 10)[0]["attention_ms"],
                               10 * 11 / 2 * 1e5 / 2e14 * 1000)
        self.assertIsNone(ceiling.prefill_floor_ms(model, None, 10)[0])

    def test_ttft_floor_is_the_larger_of_compute_and_one_weight_read(self):
        model = {"matmul_flops_per_token": {"bf16": 1e10}, "attention_flops_per_pair": 0, "recurrent_flops_per_token": 0,
                 "weight_bytes": 10e9, "kv_bytes_per_token": 1e5}
        short, missing = ceiling.ttft_floor_ms(model, {"bf16": 100}, 1000, 100)
        self.assertEqual(missing, [])
        self.assertEqual(short["bound"], "memory")  # 10 ms of reading weights beats 10 ms x 100 / 1,000 of math
        self.assertAlmostEqual(short["total_ms"], (10e9 + 1e5 * 100) / 1e12 * 1000)
        long, _ = ceiling.ttft_floor_ms(model, {"bf16": 100}, 1000, 10000)
        self.assertEqual(long["bound"], "compute")
        self.assertAlmostEqual(long["total_ms"], 10000 * 1e10 / 1e14 * 1000)
        declared, _ = ceiling.ttft_floor_ms(model, {"bf16": 100}, 1000, 100, bytes_per_pass=20e9)
        self.assertAlmostEqual(declared["memory_ms"], (20e9 + 1e5 * 100) / 1e12 * 1000)
        self.assertIn("device bandwidth (--device)", ceiling.ttft_floor_ms(model, {"bf16": 100}, None, 100)[1])

    def test_probe_keeps_the_median_of_each_value(self):
        samples = [{"device": "gpu", "bandwidth_gbs": b, "tflops": {"fp8": f}} for b, f in ((100, 5), (300, 7), (200, 6))]
        combined = probe.combine(samples)
        self.assertEqual(combined["bandwidth_gbs"], 200)
        self.assertEqual(combined["tflops"], {"fp8": 6})


THRESHOLDS = {"acceptance_ceiling_width": 8}


class GapTests(unittest.TestCase):
    def speed(self):
        return {"decode_score": 100.0, "cases": {
            "short": {"status": "measured", "prompt_tokens": 100, "output_tokens": 200, "decode_tps": 100.0,
                      "tokens_per_round": 2.0, "round_ms": 20.0, "ttft_ms": 50.0, "prefill_tps": 2000.0},
            "long": {"status": "measured", "prompt_tokens": 9900, "output_tokens": 200, "decode_tps": 100.0,
                     "tokens_per_round": 2.0, "round_ms": 20.0, "ttft_ms": 2000.0, "prefill_tps": 4950.0}}}

    def test_kv_bytes_raise_the_floor_at_long_context(self):
        info = {"draft_width": 3, "bytes_per_pass": 10e9, "draft_bytes_per_step": 1e9}
        room = headroom.summarize(info, self.speed(), 16.0, 1000, THRESHOLDS, ceiling={"kv_bytes_per_token": 1e5})
        self.assertAlmostEqual(room["cases"]["short"]["pass_floor_ms"], (10e9 + 1e5 * 200) / 1e12 * 1000)
        self.assertAlmostEqual(room["cases"]["long"]["pass_floor_ms"], (10e9 + 1e5 * 10000) / 1e12 * 1000)

    def test_decode_gaps_project_single_terms_in_a_fixed_order(self):
        info = {"draft_width": 3, "bytes_per_pass": 10e9, "draft_bytes_per_step": 1e9}
        room = headroom.summarize(info, self.speed(), 16.0, 1000, THRESHOLDS)
        gaps = {gap["name"]: gap for gap in room["gaps"]}
        # Rounds of 20 ms: pass 16 -> 10 ms gives 14 ms rounds; drafts (20 - 16) / 3 -> 1 ms gives 19 ms rounds.
        # Without engine-measured parts the round beyond one pass is one term: draft steps and verify columns.
        self.assertAlmostEqual(gaps["model pass at its bandwidth floor"]["projected_gain"], 20 / 14 - 1)
        self.assertAlmostEqual(gaps["draft steps and verify columns at the draft floor"]["projected_gain"], 20 / 19 - 1)
        self.assertEqual([gap["name"] for gap in room["gaps"]],
                         ["model pass at its bandwidth floor", "draft steps and verify columns at the draft floor"])
        self.assertFalse(any({"polished_kpi", "below_noise"} & set(gap) for gap in room["gaps"]))
        self.assertIn("verify_ms and draft_ms from generate() (separates the draft steps from the verify pass)",
                      room["missing"])

    def test_engine_measured_parts_separate_draft_steps_from_the_verify_pass(self):
        info = {"draft_width": 3, "bytes_per_pass": 10e9, "draft_bytes_per_step": 1e9}
        speed = self.speed()
        for case in speed["cases"].values():
            # 20 ms rounds: an 18.5 ms verify pass (vs a 16 ms ordinary step) and three 0.5 ms draft steps.
            case["samples"] = [{"warmup": False, "rounds": 10, "verify_ms": 185.0, "draft_ms": 15.0, "accepted": [1] * 10}]
        room = headroom.summarize(info, speed, 16.0, 1000, THRESHOLDS)
        gaps = {gap["name"]: gap for gap in room["gaps"]}
        self.assertAlmostEqual(room["cases"]["short"]["verify_ms"], 18.5)
        self.assertAlmostEqual(room["cases"]["short"]["draft_step_ms"], 0.5)
        # Drafts at 0.5 ms are already below the 1.0 ms floor: no gap. Verify 18.5 -> 16 ms: 20 -> 17.5 ms rounds.
        self.assertNotIn("draft steps at their bandwidth floor", gaps)
        self.assertAlmostEqual(gaps["verify pass of 4 columns at the ordinary pass cost"]["projected_gain"], 20 / 17.5 - 1)
        self.assertNotIn("draft steps and verify columns at the draft floor", gaps)
        self.assertFalse(any("verify_ms" in m for m in room["missing"]))
        slow = headroom.summarize({**info, "draft_bytes_per_step": 0.25e9}, speed, 16.0, 1000, THRESHOLDS)
        gaps = {gap["name"]: gap for gap in slow["gaps"]}
        # Draft floor 0.25 ms: three steps 0.5 -> 0.25 ms take 0.75 ms off each 20 ms round.
        self.assertAlmostEqual(gaps["draft steps at their bandwidth floor"]["projected_gain"], 20 / 19.25 - 1)

    def test_prefill_score_against_its_compute_ceiling(self):
        model = {"matmul_flops_per_token": {"bf16": 1e10}, "attention_flops_per_pair": 0, "recurrent_flops_per_token": 0}
        room = headroom.summarize({}, self.speed(), None, None, THRESHOLDS, ceiling=model, tflops={"bf16": 100},
                                  prefill_cases=["long"])
        prefill = room["prefill"]
        # 9,900 tokens x 1e10 FLOPs at 100 TFLOPS: 990 ms against 2,000 ms measured.
        self.assertAlmostEqual(prefill["cases"]["long"]["floor"]["total_ms"], 990.0)
        self.assertAlmostEqual(prefill["fraction_of_ceiling"], 990 / 2000)
        self.assertNotIn("polished", prefill)
        gap = next(gap for gap in room["gaps"] if gap["kpi"] == "prefill")
        self.assertAlmostEqual(gap["projected_gain"], 2000 / 990 - 1)
        self.assertIsNone(room["ttft"]["fraction_of_floor"])  # no bandwidth for the memory floor
        unmeasured = headroom.summarize({}, self.speed(), None, None, THRESHOLDS, ceiling=model, prefill_cases=["long"])
        self.assertIsNone(unmeasured["prefill"]["fraction_of_ceiling"])
        self.assertIn("device GEMM rates (--device)", unmeasured["missing"])


    def test_ttft_score_against_its_floor_over_every_case(self):
        model = {"matmul_flops_per_token": {"bf16": 1e10}, "attention_flops_per_pair": 0, "recurrent_flops_per_token": 0,
                 "weight_bytes": 20e9, "kv_bytes_per_token": 0}
        room = headroom.summarize({}, self.speed(), None, 1000, THRESHOLDS, ceiling=model, tflops={"bf16": 100})
        ttft = room["ttft"]
        # Floors: short is memory bound at 20 ms, long compute bound at 990 ms; measured 50 and 2,000 ms.
        self.assertAlmostEqual(ttft["cases"]["short"]["floor"]["total_ms"], 20.0)
        self.assertAlmostEqual(ttft["score_ms"], (50 * 2000) ** 0.5)
        self.assertAlmostEqual(ttft["floor_ms"], (20 * 990) ** 0.5)
        self.assertAlmostEqual(ttft["fraction_of_floor"], ((20 * 990) / (50 * 2000)) ** 0.5)
        gap = next(gap for gap in room["gaps"] if gap["kpi"] == "ttft")
        self.assertAlmostEqual(gap["projected_gain"], 1 / ttft["fraction_of_floor"] - 1)
        self.assertIn("short 50 ms vs 20 ms (memory bound)", gap["detail"])


class HeldoutTests(unittest.TestCase):
    """Held-out prompts and the generality gate: suite gains must show on prompts nobody tuned for."""

    @staticmethod
    def result(suite, heldout, prompts=None):
        prompts = prompts or {"zh": heldout, "de": heldout}
        return {"model_sha256": "m", "suite_sha256": "s", "label": "x",
                "speed": {"decode_score": suite, "decode_text": {"source": "pinned", "sha256": "t"}, "cases": {}},
                "heldout": {"status": "measured", "decode_score": heldout,
                            "prompts": {pid: {"status": "measured", "language": pid, "decode_tps": tps,
                                              "tokens_per_round": 2.0} for pid, tps in prompts.items()}}}

    def gate(self, result, baseline):
        protocol = json.loads((ROOT / "eval/protocol.json").read_text(encoding="utf-8"))
        return next(g for g in status.build(result, protocol, baseline)["gates"] if g["name"] == "generality")

    def test_held_out_prompts_are_prepared_from_the_suite(self):
        held = json.loads((ROOT / "eval/suite/heldout.json").read_text(encoding="utf-8"))
        protocol = json.loads((ROOT / "eval/protocol.json").read_text(encoding="utf-8"))
        speed = json.loads((ROOT / "eval/suite/speed.json").read_text(encoding="utf-8"))
        texts = {m.get("content") for case in speed["cases"] for m in case["messages"]}
        self.assertGreaterEqual(len({p["language"] for p in held["prompts"]}), 8)
        self.assertFalse(texts & {p["content"] for p in held["prompts"]})
        self.assertIn("heldout", protocol)

    def test_mock_engine_decodes_the_held_out_prompts(self):
        with tempfile.TemporaryDirectory() as directory:
            data = Path(directory)
            prepared, _ = fixture(data)
            prepared["heldout"] = [{"id": "zh", "language": "Chinese", "kind": "prose", "prompt_ids": [7, 8, 9],
                                    "output_tokens": 12}]
            (data / "prepared.json").write_text(json.dumps(prepared))
            result = runner.run(MOCK, "unused-model", data, data / "result.json",
                                {"reference": str(data / "reference.json"), "max_context": 128},
                                parts=("speed", "heldout"), warmups=0, repetitions=1, log=lambda message: None)
            held = result["heldout"]
            self.assertEqual(held["status"], "measured")
            self.assertEqual(held["prompts"]["zh"]["output_tokens"], 12)
            self.assertGreater(held["decode_score"], 0)
            gate = next(g for g in result["status"]["gates"] if g["name"] == "generality")
            self.assertEqual(gate["state"], "unmeasured")  # no baseline to judge against

    def test_a_suite_gain_with_slower_held_out_prompts_fails(self):
        baseline = self.result(100.0, 50.0)
        self.assertEqual(self.gate(self.result(160.0, 45.0, {"zh": 44.0, "de": 46.0}), baseline)["state"], "fail")
        # Trailing the suite's gain is reported, not failed: speculation gains less on text that drafts poorly.
        gate = self.gate(self.result(144.0, 58.0), baseline)
        self.assertEqual(gate["state"], "pass")
        self.assertIn("suite +44.0%", gate["detail"])

    def test_one_held_out_prompt_slowing_down_fails_even_when_the_mean_keeps_up(self):
        baseline = self.result(100.0, 50.0)
        gate = self.gate(self.result(100.0, 52.0, {"zh": 40.0, "de": 67.6}), baseline)
        self.assertEqual(gate["state"], "fail")
        self.assertIn("zh", gate["detail"])

    def test_without_a_measured_baseline_the_gate_is_unmeasured(self):
        self.assertEqual(self.gate(self.result(100.0, 50.0), None)["state"], "unmeasured")
        old = self.result(100.0, 50.0)
        del old["heldout"]
        self.assertEqual(self.gate(self.result(100.0, 50.0), old)["state"], "unmeasured")


class StatusTests(unittest.TestCase):
    def run_mock(self, directory, device, **config):
        data = Path(directory)
        prepared, _ = fixture(data)
        prepared["ceiling"] = {"matmul_flops_per_token": {"fp8": 1e8}, "attention_flops_per_pair": 0,
                               "recurrent_flops_per_token": 0, "kv_bytes_per_token": 0}
        (data / "prepared.json").write_text(json.dumps(prepared))
        config = {"reference": str(data / "reference.json"), "max_context": 128, "bytes_per_pass": 1e6,
                  "draft_bytes_per_step": 1e5, **config}
        return runner.run(MOCK, "unused-model", data, data / "result.json", config, warmups=0, repetitions=2,
                          log=lambda message: None, device=device)

    def test_status_reports_gates_kpis_and_missing_inputs_without_a_verdict(self):
        with tempfile.TemporaryDirectory() as directory:
            result = self.run_mock(directory, {"bandwidth_gbs": 1.0, "tflops": {"fp8": 1.0, "bf16": 1.0}}, noise=0.2)
            current = result["status"]
            self.assertFalse({"verdict", "next"} & set(current))
            gates = {g["name"]: g["state"] for g in current["gates"]}
            # Exactness is reported by default, so it is a note rather than a gate.
            # The fixture's prepared data predates the held-out prompts: generality is unmeasured, not failed.
            self.assertEqual(gates, {"coverage": "pass", "repeatable": "pass", "quality": "fail",
                                 "capacity": "unmeasured", "lower_context": "unmeasured", "generality": "unmeasured"})
            self.assertIn("exactness reported: 2/2 cases equal plain greedy", current["notes"])
            self.assertIn("scoring on paths verify (adapter score_paths)", current["missing"])
            self.assertIsNotNone(current["kpis"]["decode"]["speed_of_light"])
            self.assertIsNotNone(current["kpis"]["ttft"]["fraction_of_floor"])
            self.assertIn("ttft", status.render(current))
            self.assertEqual(current["kpis"]["quality"]["worst_kld_mean_path"], "decode")
            text = status.render(current)
            self.assertIn("quality ✗", text)  # decode-path quality fails with 0.2 nats noise
            self.assertNotIn("polished", text)
            # The stop-record shape is what `crit run ... stop` validates.
            for _folder in ("core/python", "api", "eval", "kernels", "cli"): sys.path.insert(0, str(ROOT / _folder))
            from crit.run.records import headroom_record
            account = current["stop_record_headroom"]
            self.assertEqual([row["kpi"] for row in account["kpis"]], ["decode", "prefill", "ttft"])
            headroom_record({"kpis": [{**row, "built": "fixture", "cause": "fixture"} for row in account["kpis"]]},
                            "search_exhausted")
            self.assertEqual([goal["kpi"] for goal in current["goals"]], ["decode", "prefill", "ttft"])
            self.assertIn("goals for the next candidate", text)
            self.assertIsNotNone(current["kpis"]["end_to_end"]["score"])
            self.assertIn("end-to-end", text)
            again = status.status_of(Path(directory) / "result.json", runner.load_json(runner.EVAL_DIR / "protocol.json"),
                                     Path(directory) / "result.json")
            self.assertEqual(again["baseline"]["decode_score"], 0.0)
            self.assertEqual(again["stop_record_headroom"]["kpis"][0]["start"],
                             again["stop_record_headroom"]["kpis"][0]["reached"])  # the baseline is the start

    def test_run_sets_its_own_drift_budget_and_can_require_exactness(self):
        with tempfile.TemporaryDirectory() as directory:
            data = Path(directory)
            fixture(data)
            config = {"reference": str(data / "reference.json"), "noise": 0.2, "max_context": 128}
            budget = {"ppl_ratio_max": 1.5, "kld_mean_max": 1.0, "kld_p99_max": 10.0}
            result = runner.run(MOCK, "unused-model", data, data / "result.json", config, warmups=0, repetitions=2,
                                log=lambda message: None, quality_budget=budget, exactness="required")
            self.assertEqual(result["requirements"], {"quality_budget": budget, "exactness": "required"})
            decode = result["quality"]["paths"]["decode"]
            self.assertTrue(decode["passed"])  # 0.2 nats of drift fits this run's budget
            self.assertAlmostEqual(decode["budget_used"], (decode["ppl_ratio"] - 1) / 0.5)
            gates = {g["name"]: g["state"] for g in result["status"]["gates"]}
            self.assertEqual(gates["exactness"], "pass")
            self.assertIn("of the drift budget used (decode)", status.render(result["status"]))
            with self.assertRaises(SystemExit):
                runner.run(MOCK, "unused-model", data, data / "bad.json", config, warmups=0, repetitions=1,
                           log=lambda message: None, quality_budget={"kld_max": 1.0})

    def test_exactness_runs_on_the_loaded_engine_when_it_decodes_plain_greedy(self):
        opened = []
        original = runner.open_engine

        def counting(module, model, config, speculative, context_tokens=None):
            opened.append(speculative)
            return original(module, model, config, speculative, context_tokens)
        runner.open_engine = counting
        try:
            with tempfile.TemporaryDirectory() as directory:
                data = Path(directory)
                fixture(data)
                for plain, engine, opens in ((True, "shared", [True]), (False, "separate", [True, False])):
                    opened.clear()
                    result = runner.run(MOCK, "unused-model", data, data / "result.json",
                                        {"max_context": 128, "plain": plain}, parts=("speed", "exactness"),
                                        warmups=0, repetitions=2, log=lambda message: None, exactness="required")
                    self.assertEqual(opened, opens)
                    self.assertEqual(result["exactness"]["engine"], engine)
                    self.assertTrue(result["exactness"]["identical"])
                    # Plain decoding is one target pass per token, whichever engine ran it.
                    self.assertGreater(result["exactness"]["step_ms"], 0)
        finally:
            runner.open_engine = original

    def test_a_disturbed_device_stops_the_round(self):
        with tempfile.TemporaryDirectory() as directory:
            data = Path(directory)
            fixture(data)
            config = {"reference": str(data / "reference.json"), "max_context": 128, "disturbed": True}
            result = runner.run(MOCK, "unused-model", data, data / "result.json", config, warmups=0, repetitions=2,
                                log=lambda message: None)
            self.assertEqual(result["aborted"]["reason"], "disturbed")
            self.assertIn("speed short: decode varied", result["aborted"]["detail"])
            self.assertNotIn("status", result)
            self.assertNotIn("quality", result)  # nothing after the disturbed case is measured
            self.assertEqual(json.loads((data / "result.json").read_text())["aborted"], result["aborted"])
            protocol = runner.load_json(runner.EVAL_DIR / "protocol.json")
            with self.assertRaises(SystemExit):
                status.status_of(data / "result.json", protocol)
            kept = runner.run(MOCK, "unused-model", data, data / "kept.json", config, warmups=0, repetitions=2,
                              log=lambda message: None, abort_disturbed=False)
            self.assertNotIn("aborted", kept)
            self.assertIn("status", kept)

    def test_differences_from_plain_greedy_fail_only_when_exactness_is_required(self):
        protocol = runner.load_json(runner.EVAL_DIR / "protocol.json")
        exact = {"cases": {"a": {"identical": False, "first_difference": 7}}}
        base = {"engine": {"speculative": True}, "speed": {"cases": {"a": {"status": "measured", "repeatable": True}}}}
        reported = status.build({**base, "exactness": {**exact, "mode": "reported"}}, protocol)
        self.assertNotIn("exactness", [g["name"] for g in reported["gates"]])
        self.assertIn("a differs at token 7; the quality budget judges the output", reported["notes"][0])
        required = status.build({**base, "exactness": {**exact, "mode": "required"}}, protocol)
        self.assertEqual({g["name"]: g["state"] for g in required["gates"]}["exactness"], "fail")
        legacy = status.build({**base, "exactness": exact}, protocol)  # results from before modes were required
        self.assertEqual({g["name"]: g["state"] for g in legacy["gates"]}["exactness"], "fail")

    def test_end_to_end_score_is_a_fact_and_gaps_carry_no_end_to_end_gain(self):
        speed = {"cases": {
            "short": {"status": "measured", "output_tokens": 101, "ttft_ms": 10.0, "decode_tps": 200.0},
            "long": {"status": "measured", "output_tokens": 101, "ttft_ms": 3000.0, "decode_tps": 200.0}}}
        ttft = {"cases": {"short": {"floor": {"total_ms": 5.0}}, "long": {"floor": {"total_ms": 1000.0}}}}
        e2e = headroom.end_to_end_summary(speed, ttft, 200.0, 250.0)
        self.assertAlmostEqual(e2e["score"], ((101000 / 510) * (101000 / 3500)) ** 0.5)
        self.assertAlmostEqual(e2e["ceiling_score"], ((101000 / 405) * (101000 / 1400)) ** 0.5)
        self.assertFalse(hasattr(headroom, "end_to_end_gains"))

    def test_goals_take_every_kpi_to_its_limit_at_once(self):
        protocol = runner.load_json(runner.EVAL_DIR / "protocol.json")
        room = {"projection": {"ceiling_score": 250.0, "fraction_of_ceiling": 0.88},
                "prefill": {"score": 5000.0, "ceiling_score": 14000.0, "fraction_of_ceiling": 5 / 14, "cases": {}},
                "ttft": {"score_ms": 300.0, "floor_ms": 200.0, "fraction_of_floor": 2 / 3, "cases": {}},
                "gaps": [{"kpi": "prefill", "name": "prefill", "projected_gain": 1.8, "end_to_end_gain": 0.14}],
                "missing": []}
        result = {"engine": {"speculative": False}, "headroom": room,
                  "speed": {"decode_score": 220.0, "complete": True, "cases": {"a": {"status": "measured", "repeatable": True}}}}
        current = status.build(result, protocol)
        changes = {goal["kpi"]: round(goal["change"], 3) for goal in current["goals"]}
        self.assertEqual(changes, {"decode": round(250 / 220 - 1, 3), "prefill": 1.8, "ttft": round(200 / 300 - 1, 3)})
        text = status.render(current)
        self.assertIn("prefill  5,000 -> 14,000 tok/s (+180%)", text)
        self.assertIn("ttft     300 -> 200 ms (-33%)", text)
        self.assertNotIn("end to end", text)  # older results' end-to-end gains are not shown
        self.assertNotIn("end_to_end_gain", current["gaps"][0])

    def test_status_reports_each_kpi_against_its_limit_and_no_classification(self):
        protocol = runner.load_json(runner.EVAL_DIR / "protocol.json")
        room = {"step_ms": 10.5, "pass_floor_ms": 10.0, "pass_efficiency": 10 / 10.5,
                "projection": {"ceiling_score": 100.0, "fraction_of_ceiling": 0.9, "speed_of_light_score": 95.0},
                "prefill": {"score": 800.0, "ceiling_score": 1000.0, "fraction_of_ceiling": 0.8, "cases": {}},
                "ttft": {"score_ms": 100.0, "floor_ms": 80.0, "fraction_of_floor": 0.8, "cases": {}},
                "levers": [{"kpi": "decode", "name": "old", "projected_gain": 0.1, "polished_kpi": True}], "missing": []}
        quality = {"passed": True, "unsupported": [], "paths": {"prefill": {
            "passed": True, "failures": [], "kld_mean": 1e-5, "kld_p99": 1e-4, "ppl_ratio": 1.0, "top1_agreement": 1.0}}}
        result = {"engine": {"speculative": False}, "headroom": room, "quality": quality,
                  "speed": {"decode_score": 90.0, "complete": True, "cases": {"a": {"status": "measured", "repeatable": True}}}}
        current = status.build(result, protocol)
        self.assertFalse({"verdict", "meaning", "next", "levers"} & set(current))
        self.assertEqual(current["gaps"], [{"kpi": "decode", "name": "old", "projected_gain": 0.1}])  # older results
        text = status.render(current)
        self.assertIn("decode   90.0 tok/s, 90% of its ceiling 100.0", text)
        self.assertIn("prefill  800 tok/s, 80% of its compute ceiling 1,000", text)
        self.assertNotIn("polished", text)
        self.assertNotIn("next step", text)
        self.assertEqual(current["stop_record_headroom"]["kpis"][0], {"kpi": "decode", "start": None, "reached": 90.0,
                                                                      "limit": 100.0, "built": "", "cause": ""})

    def test_noise_is_the_spread_of_per_repetition_scores(self):
        def sample(tps, warmup=False):
            return {"decode_tps": tps, "ttft_ms": 10.0, "request_ms": 1000.0 / tps * 100 + 10, "output_tokens": 101,
                    "warmup": warmup}
        speed = {"cases": {"a": {"status": "measured", "samples": [sample(50, True), sample(95), sample(100), sample(105)]}}}
        self.assertAlmostEqual(headroom.noise(speed)["decode"], 0.1)
        self.assertIsNone(headroom.noise({"cases": {"a": {"status": "measured", "samples": [sample(100)]}}})["decode"])

    def test_pinned_continuation_makes_every_engine_decode_the_same_text(self):
        with tempfile.TemporaryDirectory() as directory:
            data = Path(directory)
            prepared, _ = fixture(data)
            config = {"reference": str(data / "reference.json"), "max_context": 128}
            pinned = pin.pin(MOCK, "unused-model", data, config, log=lambda message: None)
            self.assertEqual(set(pinned["cases"]), {"short", "long"})
            self.assertEqual(pinned["sha256"], pin.continuation_sha256(pinned["cases"]))
            with self.assertRaises(SystemExit):  # one pinned text per data directory
                pin.pin(MOCK, "unused-model", data, config, log=lambda message: None)
            result = runner.run(MOCK, "unused-model", data, data / "result.json", config, warmups=0, repetitions=2,
                                log=lambda message: None, parts=("speed", "exactness"))
            speed = result["speed"]
            self.assertEqual(speed["decode_text"], {"source": "pinned", "sha256": pinned["sha256"],
                                                    "pinned_by": pinned["engine"]})
            self.assertEqual(len(speed["cases"]["short"]["samples"]), 2)
            self.assertTrue(speed["cases"]["short"]["repeatable"])
            self.assertIn(f"decode measured on the pinned continuation {pinned['sha256'][:12]}", result["status"]["notes"])
            free = runner.run(MOCK, "unused-model", data, data / "free.json", {**config, "replay": False}, warmups=0,
                              repetitions=1, log=lambda message: None, parts=("speed",))
            self.assertEqual(free["speed"]["decode_text"]["source"], "generated")
            compared = status.changes(free, result)
            self.assertTrue(compared["decode_text_differs"])
            self.assertIsNone(compared["decode_score"])  # different texts are not compared
            self.assertIn("decoded different texts", report.compare([data / "result.json", data / "free.json"]))
            stored = json.loads((data / pin.CONTINUATIONS).read_text())
            stored["cases"]["short"][0] += 1
            (data / pin.CONTINUATIONS).write_text(json.dumps(stored))
            with self.assertRaises(SystemExit):  # tampered text no longer matches its identity
                pin.load_continuations(data, prepared)

    def test_replay_must_emit_the_pinned_continuation(self):
        class Wrong:
            def replay(self, prompt, continuation, on_tokens):
                tokens = list(continuation)
                tokens[3] += 1
                on_tokens(tokens)
                return {"tokens": tokens, "rounds": 1}
        with self.assertRaises(RuntimeError):
            runner.timed_replay(Wrong(), [1, 2], [5, 6, 7, 8, 9])

    def test_paired_runs_alternate_the_order_and_report_ratios(self):
        with tempfile.TemporaryDirectory() as directory:
            data = Path(directory)
            fixture(data)
            config = {"reference": str(data / "reference.json"), "max_context": 128}
            result = ab.run_ab(MOCK, config, MOCK, {**config, "draft_tokens": 2}, "unused-model", data, data / "ab.json",
                               rounds=2, warmups=0, repetitions=1, log=lambda message: None)
            self.assertEqual([row["order"] for row in result["rounds"]], [["a", "b"], ["b", "a"]])
            self.assertEqual(len(result["paired"]["decode_score"]["ratios"]), 2)
            self.assertIn("B vs A decode", ab.render(result))
            self.assertTrue((data / "ab.json").is_file())

    def test_declared_native_libraries_are_hashed_into_the_result(self):
        with tempfile.TemporaryDirectory() as directory:
            library = Path(directory) / "libengine.so"
            library.write_bytes(b"native bytes")
            identity = runner.native_identity({"native_libraries": [str(library), str(Path(directory) / "missing.so")]})
            self.assertEqual(identity[0]["sha256"], hashlib.sha256(b"native bytes").hexdigest())
            self.assertEqual(identity[0]["bytes"], 12)
            self.assertIn("error", identity[1])

    def test_missing_device_ceilings_are_listed_as_missing_inputs(self):
        with tempfile.TemporaryDirectory() as directory:
            result = self.run_mock(directory, None)
            current = result["status"]
            self.assertTrue(any("device bandwidth" in m for m in current["missing"]))


def safetensors_model(directory, shapes, config):
    """A model directory with config.json and one safetensors file of zero-filled BF16 tensors."""
    offset, tensors = 0, {}
    for name, shape in shapes.items():
        size = 2 * math.prod(shape)
        tensors[name] = {"dtype": "BF16", "shape": list(shape), "data_offsets": [offset, offset + size]}
        offset += size
    model = Path(directory)
    write_safetensors(model / "model.safetensors", tensors)
    (model / "config.json").write_text(json.dumps(config))
    return model


# 8,192 + 128,000 bytes read on every pass, 16,384 of experts, 128,000 of embedding rows, 32,768 of vision tower.
SHAPES = {"model.embed_tokens.weight": (1000, 64), "model.layers.0.self_attn.q_proj.weight": (64, 64),
          "model.layers.0.mlp.experts.0.up_proj.weight": (128, 64), "model.visual.blocks.0.mlp.fc1.weight": (256, 64),
          "lm_head.weight": (1000, 64)}
RESIDENT = 64 * 64 * 2 + 1000 * 64 * 2
# Two full-attention layers of one KV head of 8: 32 elements, 32 bytes per token at FP8.
DENSE = {"num_hidden_layers": 2, "num_attention_heads": 2, "num_key_value_heads": 1, "head_dim": 8,
         "max_position_embeddings": 32768}


class ContextTests(unittest.TestCase):
    def setUp(self):
        self.settings = runner.load_json(runner.EVAL_DIR / "protocol.json")["context"]
        self.overhead = context.GIB + int(self.settings["reserve_gib"] * context.GIB)  # headroom and reserve

    def test_max_context_is_the_smaller_of_the_model_and_device_limits(self):
        with tempfile.TemporaryDirectory() as directory:
            model = safetensors_model(directory, SHAPES, DENSE)
            device = {"device": "fixture gpu", "memory_free_bytes": self.overhead + RESIDENT + 32 * 20000}
            value = context.max_context(model, device, context.GIB, self.settings)
            self.assertEqual(value["kv_bytes_per_token"], 32)
            self.assertEqual(value["device_limit"], 20000 // 1024 * 1024)
            self.assertEqual((value["tokens"], value["bound"]), (19456, "device"))
            basis = value["basis"]
            self.assertEqual(basis["resident_weight_bytes"], RESIDENT)  # embedding rows, experts and vision stay out
            self.assertEqual((basis["expert_bytes"], basis["embedding_bytes"]), (128 * 64 * 2, 1000 * 64 * 2))
            self.assertEqual(basis["expert_bytes_in_device_memory_at_max"], 128 * 64 * 2)
            self.assertIn("19,456 tokens (device bound)", context.render(value))
            (model / "config.json").write_text(json.dumps({**DENSE, "max_position_embeddings": 8192}))
            value = context.max_context(model, device, context.GIB, self.settings)
            self.assertEqual((value["tokens"], value["bound"]), (8192, "model"))
            tight = {"memory_free_bytes": self.overhead + RESIDENT - 1}
            value = context.max_context(model, tight, context.GIB, self.settings)
            self.assertIsNone(value["tokens"])
            self.assertEqual(value["device_limit"], 0)
            self.assertIn("do not fit", value["reason"])
            value = context.max_context(model, {}, context.GIB, self.settings)
            self.assertIsNone(value["tokens"])
            self.assertIn("free device memory (re-run crit eval probe)", value["missing"])

    def test_a_tied_output_head_keeps_the_embedding_in_device_memory(self):
        with tempfile.TemporaryDirectory() as directory:
            shapes = {name: shape for name, shape in SHAPES.items() if name != "lm_head.weight"}
            model = safetensors_model(directory, shapes, {**DENSE, "tie_word_embeddings": True})
            split = context.weight_split(model, ceiling.model_config(model))
        self.assertEqual(split["embedding"], 0)
        self.assertEqual(split["resident"], 64 * 64 * 2 + 1000 * 64 * 2)
        self.assertEqual(split["vision"], 256 * 64 * 2)

    def test_curve_points_lower_context_and_filled_prompts(self):
        self.assertEqual(context.curve_points(262144, self.settings), [32768, 65536, 131072, 262144])
        self.assertEqual(context.curve_points(200000, self.settings), [32768, 65536, 131072, 200000])
        self.assertEqual(context.curve_points(8192, self.settings), [8192])
        self.assertEqual(context.lower_context(262144, 14000, self.settings), 65536)
        self.assertEqual(context.lower_context(8192, 72, self.settings), 2048)
        self.assertIsNone(context.lower_context(4096, 3500, self.settings))  # the longest case needs all of it
        self.assertEqual(context.filled_prompt([1, 2, 3], 7), [3, 1, 2, 3, 1, 2, 3])
        self.assertEqual(context.filled_prompt([1, 2, 3], 3), [1, 2, 3])

    def test_a_case_is_slower_at_a_lower_context_beyond_its_own_spread(self):
        def case(decode, ttft, decodes=None):
            return {"status": "measured", "decode_tps": decode, "ttft_ms": ttft,
                    "samples": [{"warmup": False, "decode_tps": d, "ttft_ms": ttft} for d in decodes or (decode, decode)]}
        at_max = {"decode_score": 100.0, "ttft_score_ms": 50.0, "cases": {cid: case(100, 50) for cid in "abcd"}}
        at_lower = {"decode_score": 94.0, "ttft_score_ms": 52.0, "cases": {
            "a": case(98, 51),  # within the protocol's 5%
            "b": case(90, 50),  # decode 10% slower
            "c": case(90, 50, decodes=(80, 100)),  # 10% slower, within its 22% run-to-run spread
            "d": case(100, 60)}}  # first token 20% later
        out = runner.compare_lower(at_max, at_lower, 2048, self.settings)
        self.assertEqual(out["slower"], {"b": ["decode"], "d": ["ttft"]})
        self.assertFalse(out["passed"])
        at_lower["cases"].update(b=case(100, 50), d=case(101, 49))
        self.assertTrue(runner.compare_lower(at_max, at_lower, 2048, self.settings)["passed"])
        del at_lower["cases"]["a"]
        out = runner.compare_lower(at_max, at_lower, 2048, self.settings)
        self.assertEqual((out["unmeasured"], out["passed"]), (["a"], False))

    def run_mock(self, directory, context_tokens, model="unused-model", device=None, **config):
        data = Path(directory)
        fixture(data)
        config = {"reference": str(data / "reference.json"), "bytes_per_pass": 1e6, "draft_bytes_per_step": 1e5, **config}
        device = {"bandwidth_gbs": 1.0, "tflops": {"fp8": 1.0, "bf16": 1.0}, **(device or {})}
        return runner.run(MOCK, model, data, data / "result.json", config, warmups=0, repetitions=2,
                          log=lambda message: None, device=device, context_tokens=context_tokens)

    def test_engines_are_built_for_the_required_context_and_checked_below_it(self):
        with tempfile.TemporaryDirectory() as directory:
            # The mock gets ten times slower per round when it is opened below 8,192 tokens.
            result = self.run_mock(directory, 8192, kv_bytes_per_token=64, slow_below_context=8192)
        required = result["context"]
        self.assertEqual((required["tokens"], required["source"], required["engine_max_context"]), (8192, "given", 8192))
        point = required["points"]["8192"]
        self.assertEqual((point["status"], point["prompt_tokens"], point["output_tokens"]), ("measured", 8192 - 256, 256))
        lower = required["lower"]
        self.assertEqual((lower["context"], lower["engine_max_context"]), (2048, 2048))
        self.assertEqual(set(lower["slower"]), {"short", "long"})
        self.assertTrue(all("decode" in kinds for kinds in lower["slower"].values()))
        current = result["status"]
        gates = {g["name"]: g for g in current["gates"]}
        self.assertEqual((gates["capacity"]["state"], gates["lower_context"]["state"]), ("pass", "fail"))
        self.assertIn("slower at 2,048 tokens than at 8,192", gates["lower_context"]["detail"])
        self.assertEqual([g["kpi"] for g in current["goals"]], ["decode", "prefill", "ttft", "context_decode", "context_ttft"])
        self.assertIsNotNone(result["headroom"]["context"]["decode"]["ceiling_score"])
        rows = status.stop_record_headroom(result)["kpis"]
        self.assertEqual([row["kpi"] for row in rows][-2:], ["context_decode", "context_ttft"])
        self.assertIn("context  8,192 tokens required (given); engine holds 8,192", status.render(current))

    def test_context_gates_fail_when_the_engine_cannot_serve_a_context(self):
        with tempfile.TemporaryDirectory() as directory:
            failing = self.run_mock(directory, 8192, fails_above_context=4096)
        point = failing["context"]["points"]["8192"]
        self.assertEqual(point["status"], "failed")
        gate = next(g for g in failing["status"]["gates"] if g["name"] == "capacity")
        self.assertEqual(gate["state"], "fail")
        self.assertIn("out of device memory", gate["detail"])
        with tempfile.TemporaryDirectory() as directory:
            unbuilt = self.run_mock(directory, 8192, fails_below_context=4096)
        gate = next(g for g in unbuilt["status"]["gates"] if g["name"] == "lower_context")
        self.assertEqual(gate["state"], "fail")
        self.assertIn("the engine opened at 2,048 tokens failed: RuntimeError", gate["detail"])
        with tempfile.TemporaryDirectory() as directory:
            short = self.run_mock(directory, 8192, max_context=4096)
        self.assertEqual(short["context"]["engine_max_context"], 4096)
        self.assertEqual(short["context"]["points"]["8192"]["status"], "skipped")
        gate = next(g for g in short["status"]["gates"] if g["name"] == "capacity")
        self.assertEqual(gate["state"], "fail")
        self.assertIn("max_context 4,096 is below 8,192", gate["detail"])

    def test_without_a_given_context_eval_computes_it_from_the_model_and_device(self):
        with tempfile.TemporaryDirectory() as directory, tempfile.TemporaryDirectory() as folder:
            model = safetensors_model(folder, SHAPES, {**DENSE, "max_position_embeddings": 4096})
            device = {"device": "fixture gpu", "memory_free_bytes": self.overhead + RESIDENT + 32 * 100000}
            result = self.run_mock(directory, None, model=model, device=device)
        required = result["context"]
        self.assertEqual((required["tokens"], required["source"], required["bound"]), (4096, "computed", "model"))
        self.assertEqual(required["engine_max_context"], 4096)
        self.assertEqual(list(required["points"]), ["4096"])
        self.assertEqual(required["lower"]["context"], 1024)

    def test_probe_keeps_the_least_free_device_memory(self):
        samples = [{"device": "gpu", "bandwidth_gbs": 100, "tflops": {}, "memory_total_bytes": 32 << 30,
                    "memory_free_bytes": free << 30} for free in (30, 28, 29)]
        combined = probe.combine(samples)
        self.assertEqual((combined["memory_free_bytes"], combined["memory_total_bytes"]), (28 << 30, 32 << 30))

    def test_long_context_scores_compare_only_at_the_same_required_context(self):
        def result(tokens, score):
            return {"model_sha256": "m", "suite_sha256": "s", "context": {"tokens": tokens},
                    "headroom": {"context": {"decode": {"score": score}, "ttft": {"score_ms": 1000.0}}}}
        self.assertAlmostEqual(status.changes(result(8192, 110.0), result(8192, 100.0))["context_decode"], 0.1)
        self.assertIsNone(status.changes(result(8192, 110.0), result(4096, 100.0))["context_decode"])



class ContextTextTests(unittest.TestCase):
    """Long-context points replay the longest speed case's pinned continuation, like the speed cases."""

    class Engine:
        def __init__(self):
            self.calls = []

        def generate(self, prompt, n, on_tokens):
            self.calls.append(("generate", len(prompt)))
            tokens = [7] * n
            on_tokens(tokens)
            return {"tokens": tokens, "rounds": n}

        def replay(self, prompt, continuation, on_tokens):
            self.calls.append(("replay", len(prompt)))
            on_tokens(list(continuation))
            return {"tokens": list(continuation), "rounds": 2}

    prepared = {"speed": [{"id": "short", "prompt_ids": [1, 2]}, {"id": "long", "prompt_ids": [3, 4, 5, 6]}]}
    settings = {"output_tokens": 4, "curve_from": 16, "repetitions": 1}
    pinned = {"sha256": "ab" * 32, "engine": "fixture", "cases": {"short": [9] * 6, "long": [5, 6, 7, 8, 9, 10]}}

    def run_points(self, info, continuations):
        engine = self.Engine()
        points, text = runner.run_context(engine, {"max_context": 64, **info}, self.prepared, self.settings, 32,
                                          lambda message: None, continuations)
        return engine, points, text

    def test_points_replay_the_longest_cases_pinned_text(self):
        engine, points, text = self.run_points({"replay": True, "speculative": True}, self.pinned)
        self.assertEqual(text["source"], "pinned")
        self.assertEqual(text["case"], "long")
        self.assertTrue(engine.calls and all(kind == "replay" for kind, _ in engine.calls))
        self.assertTrue(all(p["status"] == "measured" for p in points.values()))

    def test_points_fall_back_to_own_output_without_replay_or_enough_text(self):
        engine, _, text = self.run_points({"replay": False, "speculative": True}, self.pinned)
        self.assertEqual(text["source"], "generated")
        self.assertTrue(all(kind == "generate" for kind, _ in engine.calls))
        short = {**self.pinned, "cases": {"long": [5, 6]}}
        _, _, text = self.run_points({"replay": True, "speculative": True}, short)
        self.assertEqual(text["source"], "generated")
        self.assertIn("fewer than the 4", text["reason"])

    def test_context_decode_compares_only_on_the_same_text(self):
        pinned = {"context": {"tokens": 32, "decode_text": {"source": "pinned", "sha256": "ab" * 32}}}
        own = {"context": {"tokens": 32}}
        self.assertEqual(status.context_text_identity(pinned), "ab" * 32)
        self.assertEqual(status.context_text_identity(own), "generated")


if __name__ == "__main__":
    unittest.main()
