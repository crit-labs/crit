"""crit eval harness: metrics, runner and comparison with the mock adapter (no GPU)."""
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, mock_open, patch

ROOT = Path(__file__).resolve().parents[1]
for _folder in ("core/python", "eval"): sys.path.insert(0, str(ROOT / _folder))
from crit_core.model import inspect_safetensors  # noqa: E402
from crit_eval import ab, ceiling, context, headroom, host, hostload, metrics, pin, prepare, probe, report, runner, status  # noqa: E402
import crit_eval  # noqa: E402
from crit_eval import protocol_version  # noqa: E402

MOCK = ROOT / "eval/examples/mock_adapter.py"
SUITE = prepare.suite_identity()  # prepared data must come from the suite Eval runs
# What every result holds before its parts run (crit_eval.runner): the run's requirements and the required context.
MEASURED = {"requirements": {"exactness": "reported"}, "context": {"tokens": None, "reason": "not given", "missing": []}}


def softmax_logprobs(logits):
    top = max(logits)
    total = sum(math.exp(v - top) for v in logits)
    return [v - top - math.log(total) for v in logits]


FACTS = 2000  # retrieval fact ids in the fixture start here; the mock's facts_from knob repeats them


def fixture(directory):
    prepared = {
        "protocol": "crit-eval-v7", "suite_sha256": SUITE, "model": {"sha256": "model"}, "stop_ids": [999],
        "sampling": {"temperature": 1.0, "top_p": 0.95, "top_k": 8}, "sampling_source": "generation_config",
        "valid_vocab": 1000,
        "speed": [{"id": "short", "kind": "prose", "prompt_ids": [1, 2, 3], "output_tokens": 12},
                  {"id": "long", "kind": "long-context", "prompt_ids": list(range(60)), "output_tokens": 12}],
        "quality": [{"id": "c", "set": "conversations", "prompt_ids": [5, 6], "continuation_ids": [7, 8, 9],
                     "messages": [{"role": "user", "content": "c"}]}],
        # The long-context curve asks for the facts and a summary ([5, 6]) after the same archive.
        "retrieval": {"head_ids": [1, 2], "tail_ids": [3, 4], "curve_tail_ids": [5, 6], "filler_ids": list(range(10, 50)),
                      "facts": [{"id": "start", "answer": "ab", "ids": [FACTS, FACTS + 1]},
                                {"id": "middle", "answer": "cd", "ids": [FACTS + 2, FACTS + 3]},
                                {"id": "end", "answer": "ef", "ids": [FACTS + 4, FACTS + 5]}],
                      "token_text": [None] * FACTS + list("abcdef")},
    }
    (directory / "prepared.json").write_text(json.dumps(prepared))
    return prepared


class MetricTests(unittest.TestCase):
    def test_partial_kl_is_zero_for_equal_and_a_lower_bound_of_full_kl(self):
        p = softmax_logprobs([2.0, 1.0, 0.0, -1.0, -2.0])
        q = softmax_logprobs([1.5, 1.2, 0.1, -0.5, -3.0])
        self.assertAlmostEqual(metrics.partial_kl([0, 1, 2], p[:3], dict(enumerate(p[:3]))), 0.0, places=12)
        full = sum(math.exp(a) * (a - b) for a, b in zip(p, q))
        both = metrics.partial_kl([0, 1, 2], p[:3], dict(enumerate(q[:3])))
        self.assertGreater(both, 0.0)
        self.assertLessEqual(both, full + 1e-12)
        # A reference token outside the engine's top-k goes into the lumped rest: still a lower bound, and lower.
        missing = metrics.partial_kl([0, 1, 2], p[:3], {0: q[0], 2: q[2], 4: q[4]})
        self.assertLessEqual(missing, both + 1e-12)

    def test_text_quality_by_position_buckets_shows_drift_where_it_starts(self):
        # Neither side chose the fixed text: top-1 agreement compares the two first-ranked tokens. Drift that starts
        # at position 1024 shows in the 1024-2047 bucket only.
        lp = softmax_logprobs([3.0, 1.0, 0.5, 0.0])
        bent = softmax_logprobs([1.0, 3.0, 0.5, 0.0])
        reference = [{"logprob": lp[1], "top_ids": [7, 8, 9], "top_logprobs": lp[:3]}] * 2000
        engine = ([{"logprob": lp[1], "top": [[7, lp[0]], [8, lp[1]], [9, lp[2]]]}] * 1023
                  + [{"logprob": bent[1], "top": [[8, bent[1]], [7, bent[0]], [9, bent[2]]]}] * 977)
        labels = [metrics.log2_bucket(p, 512) for p in range(1, 2001)]
        self.assertEqual(sorted(set(labels), key=labels.index), ["1-511", "512-1023", "1024-2047"])
        buckets = metrics.bucketed(reference, engine, labels)
        self.assertEqual([round(b["kld_mean"], 9) for b in list(buckets.values())[:2]], [0.0, 0.0])
        self.assertEqual((buckets["1-511"]["top1_agreement"], buckets["1024-2047"]["top1_agreement"]), (1.0, 0.0))
        self.assertGreater(buckets["1024-2047"]["kld_mean"], 0.1)
        self.assertEqual((metrics.log2_bucket(31, 32), metrics.log2_bucket(32, 32), metrics.log2_bucket(255, 32)),
                         ("1-31", "32-63", "128-255"))

    def test_speed_sample_counts_decode_after_the_first_token(self):
        sample = metrics.speed_sample(100, 11, 0.0, 0.5, 1.5, 4)
        self.assertAlmostEqual(sample["decode_tps"], 10.0)
        self.assertAlmostEqual(sample["prefill_tps"], 200.0)
        self.assertAlmostEqual(sample["tokens_per_round"], 11 / 4)

    def test_round_ms_covers_decode_time_per_target_pass(self):
        self.assertAlmostEqual(metrics.speed_sample(100, 11, 0.0, 0.5, 1.5, 4)["round_ms"], 250.0)
        self.assertAlmostEqual(metrics.speed_sample(100, 11, 0.0, 0.5, 1.5, None)["round_ms"], 100.0)

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
            config = {"max_context": 64, "bytes_per_pass": 1e6, "draft_bytes_per_step": 1e5}
            result = runner.run(MOCK, "unused-model", data, data / "result.json", config,
                                warmups=0, repetitions=2, log=lambda message: None, bandwidth_gbs=1.0)
            speed = result["speed"]
            # The adapter by its checkout-relative path and its bytes: no user's home directory in a result.
            self.assertEqual(result["adapter"], "eval/examples/mock_adapter.py")
            self.assertEqual(result["adapter_sha256"], hashlib.sha256(MOCK.read_bytes()).hexdigest())
            self.assertEqual(speed["cases"]["short"]["status"], "measured")
            self.assertEqual(speed["cases"]["long"]["status"], "skipped")  # 60 + 12 > 64
            self.assertAlmostEqual(speed["cases"]["short"]["tokens_per_round"], 12 / 3)
            self.assertTrue(speed["cases"]["short"]["repeatable"])
            self.assertFalse(speed["complete"])
            # An adapter's engine replies with no log-probabilities: quality is measured over the chat API.
            self.assertEqual(result["quality"]["status"], "unsupported")
            self.assertTrue(result["exactness"]["passed"])
            room = result["headroom"]
            self.assertEqual(room["draft_width"], 4)
            self.assertAlmostEqual(room["pass_floor_ms"], 1.0)
            self.assertGreater(room["step_ms"], 0)
            # From crit-eval-v7 the decode terms describe the sampled suite; the mock drafts nothing when it samples.
            self.assertEqual(room["decode_suite"], "sampled")
            self.assertAlmostEqual(room["cases"]["short"]["saturation"], 1 / 5)
            self.assertEqual(room["cases"]["short"]["tokens_by_width"], [1.0] * 5)
            self.assertIsNotNone(room["projection"]["speed_of_light_score"])
            table = report.compare([data / "result.json", data / "result.json"])
            self.assertIn("Decode tok/s: short", table)
            self.assertNotIn("Quality:", table)
            self.assertIn("Model pass vs bandwidth floor", table)
            self.assertIn("Decode ceiling", table)
            minor = {**json.loads((data / "result.json").read_text()), "protocol": "crit-eval-v7.1"}
            (data / "minor.json").write_text(json.dumps(minor))
            (data / "v4.json").write_text(json.dumps({**minor, "protocol": "crit-eval-v4"}))
            self.assertNotIn("major versions", report.compare([data / "result.json", data / "minor.json"]))
            self.assertIn("major versions", report.compare([data / "result.json", data / "v4.json"]))

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

        @staticmethod
        def decode(ids):
            return "".join(map(chr, ids))


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

    def test_a_speed_prompts_messages_render_to_its_token_ids(self):
        class Model(CharacterText):
            def render(self, messages, tools=None, **variables):
                return "".join(f"<{m['role']}>{m['content']}" for m in messages) + "<a>"
        case = {"id": "long", "prompt_tokens": 400, "document": "long_document.txt",
                "messages": [{"role": "user", "content": "Summarize.\n\n{document}"}]}
        ids, messages, tools = prepare.speed_prompt(Model(), case)
        self.assertEqual(len(ids), 400)
        self.assertEqual(prepare.token_ids(Model(), Model().render(messages, tools)), ids)  # the document cut to fit

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


# Every protocol id with the protocol and suite it names. Append only: a change to eval/protocol.json or eval/suite
# gets a new id, the next major if it can move a score and the next minor otherwise (docs/eval/README.md).
PROTOCOLS = {
    "crit-eval-v4": "345143e9b1a3c33f08efd20512d66afe438b2e30a07b6c4e67cf7ad3a58bc461",
    "crit-eval-v4.1": "90b938a9641178a19c14e3b505491757c844a442bcbcb86ccad1d7415a5cc0af",  # served engines
    "crit-eval-v5": "24a1ca7b64b191246c7e4a964bd3a6d413207a6c2cddf9d8ba4a747493822bd1",  # chat messages, reply quality
    "crit-eval-v6": "5efdc396d65f37edd3c19c10ca91c34d1be262b86f522c25b9b2e4f4685e611e",  # fixed-text quality, honesty
    "crit-eval-v7": "2b752699bd3895c0f14809ddd7ee16ed78a26e1faedc83042e936b25ef68d21e",  # sampled decode, sampling check
}


def protocol_fingerprint():
    protocol = json.loads((ROOT / "eval/protocol.json").read_text(encoding="utf-8"))
    protocol.pop("protocol")
    return hashlib.sha256((json.dumps(protocol, sort_keys=True) + prepare.suite_identity()).encode()).hexdigest()


class ProtocolVersionTests(unittest.TestCase):
    def test_the_protocol_id_names_one_protocol_and_suite(self):
        current = json.loads((ROOT / "eval/protocol.json").read_text(encoding="utf-8"))["protocol"]
        fingerprint = protocol_fingerprint()
        self.assertEqual(PROTOCOLS.get(current), fingerprint,
                         f"eval/protocol.json or eval/suite changed under {current}: give the protocol a new id "
                         f"(the next major if a score can move, else the next minor) and add it to PROTOCOLS "
                         f"with {fingerprint}")
        self.assertEqual(max(PROTOCOLS, key=protocol_version), current)  # versions only go up

    def test_the_suite_has_one_identity_on_every_platform(self):
        class WindowsOrder(type(Path())):  # sorts as a Windows path does, without case
            def __lt__(self, other):
                return [part.lower() for part in self.parts] < [part.lower() for part in other.parts]
        names = ["agent/README.md.txt", "agent/cli.py.txt", "long.json", "long_document.txt"]
        with tempfile.TemporaryDirectory() as temp:
            for name in names:
                (Path(temp) / name).parent.mkdir(exist_ok=True)
                (Path(temp) / name).write_bytes(name.encode())
            with patch.object(prepare, "SUITE_DIR", WindowsOrder(temp)):
                found = prepare.suite_identity()
        self.assertEqual(found, hashlib.sha256(b"".join(n.encode() + b"\0" + n.encode() for n in names)).hexdigest())

        self.assertEqual(protocol_version("crit-eval-v4"), (4, 0))
        self.assertEqual(protocol_version("crit-eval-v4.1"), (4, 1))
        self.assertEqual(protocol_version("crit-eval-v12.3"), (12, 3))
        for wrong in ("crit-eval-v", "crit-eval-4", "crit-eval-v4.1.2", "eval-v1", None):
            with self.assertRaises(ValueError):
                protocol_version(wrong)

    def test_prepared_data_runs_under_every_minor_version_of_its_suite(self):
        prepared = {"protocol": "crit-eval-v4", "suite_sha256": SUITE}
        prepare.check_prepared(prepared, {"protocol": "crit-eval-v4"}, "data")
        prepare.check_prepared(prepared, {"protocol": "crit-eval-v4.2"}, "data")
        for other in ({"protocol": "crit-eval-v3", "suite_sha256": SUITE},
                      {"protocol": "crit-eval-v4", "suite_sha256": "another suite"},
                      {"suite_sha256": SUITE}):
            with self.assertRaises(SystemExit):
                prepare.check_prepared(other, {"protocol": "crit-eval-v4"}, "data")


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

    def test_exllama_checkpoints_are_refused_not_counted_without_their_matrices(self):
        trellis = {"model.layers.0.mlp.up_proj.trellis": {"dtype": "I16", "shape": [4, 4, 256],
                                                          "data_offsets": [0, 8192]},
                   "model.norm.weight": {"dtype": "BF16", "shape": [64], "data_offsets": [8192, 8320]}}
        q_weight = {"model.layers.0.mlp.up_proj.q_weight": {"dtype": "I32", "shape": [16, 64],
                                                            "data_offsets": [0, 4096]}}
        cases = {"exl3 by its config": ({"quant_method": "exl3", "bits": 4}, trellis, "EXL3"),
                 "exl3 by its tensors": (None, trellis, "EXL3"),
                 "exl2 by its tensors": (None, q_weight, "EXL2")}
        for label, (quantization, tensors, method) in cases.items():
            with self.subTest(label), tempfile.TemporaryDirectory() as directory:
                model = Path(directory)
                write_safetensors(model / "model.safetensors", tensors)
                config = {"num_hidden_layers": 1, **({"quantization_config": quantization} if quantization else {})}
                (model / "config.json").write_text(json.dumps(config))
                with self.assertRaisesRegex(ValueError, f"{method} weights are not read: exllama decodes them only"):
                    ceiling.model_ceiling(model)

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

    def test_a_plain_engine_has_one_token_per_pass_at_its_floor_as_its_decode_ceiling(self):
        speed = self.speed()
        speed["decode_score"] = 50.0
        for case in speed["cases"].values():
            case.update(decode_tps=50.0, tokens_per_round=1.0)
        room = headroom.summarize({"bytes_per_pass": 10e9}, speed, 20.0, 1000, THRESHOLDS,
                                  ceiling={"kv_bytes_per_token": 1e5})
        # Pass floors with the KV read at each case's context (200 and 10,000 tokens): 10.02 and 11 ms.
        ceiling = (1000 / 10.02 * 1000 / 11.0) ** 0.5
        projection = room["projection"]
        self.assertEqual(projection["basis"], "one token per pass")
        self.assertAlmostEqual(projection["ceiling_score"], ceiling)
        self.assertAlmostEqual(projection["fraction_of_ceiling"], 50.0 / ceiling)
        self.assertIn(f"greedy decode score 50.0 is {50 / ceiling:.0%} of its ceiling {ceiling:.1f} "
                      "(one token per model pass at its bandwidth floor)", room["warnings"])
        # An engine that drafts without stating its width has no ceiling to derive; nor has one without bytes per pass.
        self.assertIsNone(headroom.summarize({"bytes_per_pass": 10e9, "speculative": True}, speed, 20.0, 1000,
                                             THRESHOLDS)["projection"])
        self.assertIsNone(headroom.summarize({}, speed, 20.0, 1000, THRESHOLDS)["projection"])

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
        return {**MEASURED, "model_sha256": "m", "suite_sha256": "s", "label": "x",
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
            prepared = fixture(data)
            prepared["heldout"] = [{"id": "zh", "language": "Chinese", "kind": "prose", "prompt_ids": [7, 8, 9],
                                    "output_tokens": 12}]
            (data / "prepared.json").write_text(json.dumps(prepared))
            result = runner.run(MOCK, "unused-model", data, data / "result.json",
                                {"max_context": 128},
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


class RepetitionTests(unittest.TestCase):
    """protocol speed.extra_repetition: two measured repetitions, a third only when they disagree."""

    def sample(self, decode, ttft):
        return {"decode_tps": decode, "ttft_ms": ttft, "warmup": False}

    def test_repetitions_disagree_by_their_decode_ttft_or_outputs(self):
        agree = [self.sample(100.0, 50.0), self.sample(103.0, 51.0)]
        self.assertFalse(runner.disagree(agree, [[1, 2], [1, 2]], 0.05))
        self.assertTrue(runner.disagree([self.sample(100.0, 50.0), self.sample(110.0, 50.0)], [[1], [1]], 0.05))
        self.assertTrue(runner.disagree([self.sample(100.0, 50.0), self.sample(100.0, 60.0)], [[1], [1]], 0.05))
        self.assertTrue(runner.disagree(agree, [[1, 2], [1, 3]], 0.05))
        # A warmup is no measured repetition.
        self.assertFalse(runner.disagree(agree + [{"decode_tps": 10.0, "ttft_ms": 900.0, "warmup": True}],
                                         [[1], [1]], 0.05))

    def test_a_third_repetition_runs_only_after_two_that_disagree(self):
        protocol = runner.load_json(runner.EVAL_DIR / "protocol.json")
        prepared = {"stop_ids": [], "speed": [{"id": "a", "kind": "chat", "prompt_ids": [1, 2], "output_tokens": 4}]}
        for speeds, runs in (([100.0, 101.0, 0.0], 3), ([100.0, 120.0, 110.0], 4)):
            with self.subTest(speeds=speeds):
                queue = [self.sample(100.0, 50.0)] + [self.sample(s, 50.0) for s in speeds]
                calls = []

                def timed(engine, prompt, n):
                    calls.append(n)
                    return [7] * n, dict(queue.pop(0), request_ms=200.0, prefill_tps=None, tokens_per_round=None,
                                         round_ms=None)
                with patch.object(runner, "timed_generate", timed):
                    speed = runner.run_speed(None, {"max_context": 64}, prepared, protocol, 1, 2, lambda m: None)
                self.assertEqual(len(calls), runs)   # the warmup, two repetitions and a third only on disagreement
                self.assertEqual(len([s for s in speed["cases"]["a"]["samples"] if not s["warmup"]]), runs - 1)


class SampledSpeedTests(unittest.TestCase):
    """crit-eval-v7: the speed cases at the model's own sampling, and whether sampled tokens follow it."""

    def run_mock(self, directory, min_tokens=1, **config):
        """The mock through Eval; min_tokens: the sampling check's least judged tokens (the fixture samples 12)."""
        data = Path(directory)
        fixture(data)
        config = {"max_context": 64, "bytes_per_pass": 1e6, "draft_bytes_per_step": 1e5, **config}
        load = runner.load_json

        def protocol(path):
            value = load(path)
            if Path(path).name == "protocol.json" and min_tokens is not None:
                value["sampling_check"]["min_tokens"] = min_tokens
            return value
        with patch.object(runner, "load_json", protocol):
            return runner.run(MOCK, "unused-model", data, data / "result.json", config, warmups=0, repetitions=2,
                              log=lambda message: None, bandwidth_gbs=1.0)

    def test_sampled_decode_is_measured_at_the_frozen_sampling_and_reported_next_to_greedy(self):
        with tempfile.TemporaryDirectory() as directory:
            result = self.run_mock(directory)
            sampled = result["sampled_speed"]
            self.assertEqual(sampled["status"], "measured")
            self.assertEqual(sampled["sampling"], {"temperature": 1.0, "top_p": 0.95, "top_k": 8})
            self.assertGreater(sampled["decode_score"], 0)
            self.assertEqual(sampled["cases"]["short"]["status"], "measured")
            self.assertEqual(sampled["cases"]["long"]["status"], "skipped")
            # The mock does not speculate when it samples: one token per pass.
            self.assertAlmostEqual(sampled["cases"]["short"]["tokens_per_round"], 1.0)
            kpis = result["status"]["kpis"]
            self.assertEqual((kpis["decode"]["score"], kpis["decode"]["suite"]), (sampled["decode_score"], "sampled"))
            self.assertEqual(kpis["greedy_decode"]["score"], result["speed"]["decode_score"])
            text = status.render(result["status"])
            self.assertIn("at the model's sampling", text)
            self.assertIn("greedy", text)

    def test_goals_headroom_and_stop_records_follow_the_sampled_decode(self):
        with tempfile.TemporaryDirectory() as directory:
            result = self.run_mock(directory)
            sampled, room = result["sampled_speed"], result["headroom"]
            goal = next(g for g in result["status"]["goals"] if g["kpi"] == "decode")
            self.assertEqual(goal["score"], sampled["decode_score"])
            stop = next(k for k in result["status"]["stop_record_headroom"]["kpis"] if k["kpi"] == "decode")
            self.assertEqual(stop["reached"], round(sampled["decode_score"], 3))
            projection = room["projection"]
            # Drafting nothing under sampling, this drafter at floor costs is one token per pass; the ceiling keeps
            # the greedy suite's first-draft acceptance, reachable by rejection sampling.
            self.assertGreater(projection["acceptance_bound_score"], projection["speed_of_light_score"])
            self.assertEqual(goal["limit"], projection["ceiling_score"])
            gap = next(g for g in room["gaps"] if g["name"].startswith("sampled requests decoded as fast as greedy"))
            self.assertAlmostEqual(gap["projected_gain"] + 1, metrics.geometric_mean(
                [max(result["speed"]["cases"][c]["decode_tps"], s["decode_tps"]) for c, s in sampled["cases"].items()
                 if s["status"] == "measured"]) / sampled["decode_score"])

    def test_without_a_sampled_suite_the_decode_stays_greedy(self):
        with tempfile.TemporaryDirectory() as directory:
            data = Path(directory)
            fixture(data)
            prepared = json.loads((data / "prepared.json").read_text())
            del prepared["sampling"]
            (data / "prepared.json").write_text(json.dumps(prepared))
            result = runner.run(MOCK, "unused-model", data, data / "result.json",
                                {"max_context": 64, "bytes_per_pass": 1e6, "draft_bytes_per_step": 1e5}, warmups=0,
                                repetitions=2, log=lambda message: None, bandwidth_gbs=1.0)
            room, kpis = result["headroom"], result["status"]["kpis"]
            self.assertEqual(result["sampled_speed"]["status"], "unsupported")
            self.assertEqual(room["decode_suite"], "greedy")
            self.assertEqual(room["cases"]["short"]["tokens_by_width"], [1.0, 2.0, 3.0, 4.0, 5.0])
            self.assertEqual((kpis["decode"]["score"], kpis["decode"]["suite"]), (result["speed"]["decode_score"], "greedy"))
            self.assertIsNone(kpis["greedy_decode"])
            goal = next(g for g in result["status"]["goals"] if g["kpi"] == "decode")
            self.assertEqual(goal["score"], result["speed"]["decode_score"])

    def test_a_sampler_that_favours_likely_tokens_fails_the_sampling_gate(self):
        with tempfile.TemporaryDirectory() as directory:
            good = self.run_mock(directory)["sampling_check"]
            self.assertTrue(good["passed"])
            self.assertEqual(good["outside"], 0)
            self.assertLessEqual(abs(good["z"]), good["z_max"])
        with tempfile.TemporaryDirectory() as directory:
            result = self.run_mock(directory, greedy_sampling=True)
            bad = result["sampling_check"]
            self.assertFalse(bad["passed"])
            self.assertLess(bad["z"], -bad["z_max"])
            gates = {g["name"]: g["state"] for g in result["status"]["gates"]}
            self.assertEqual(gates["sampling"], "fail")
        with tempfile.TemporaryDirectory() as directory:
            # The protocol's least judged tokens: the fixture's 12 sampled tokens leave the gate unmeasured.
            result = self.run_mock(directory, min_tokens=None)
            self.assertIsNone(result["sampling_check"]["passed"])
            gate = next(g for g in result["status"]["gates"] if g["name"] == "sampling")
            self.assertEqual(gate["state"], "unmeasured")
            self.assertIn("fewer than 256 tokens judged", gate["detail"])

    def test_an_engine_that_cannot_sample_has_no_sampled_decode(self):
        class Greedy:
            def __init__(self, inner):
                self.inner = inner

            def __getattr__(self, name):
                if name == "generate_sampled":
                    raise AttributeError(name)
                return getattr(self.inner, name)
        found = runner.run_sampled_speed(Greedy(object()), {"max_context": 64}, {"speed": [], "sampling": {}},
                                         runner.load_json(runner.EVAL_DIR / "protocol.json"), 0, 2, lambda m: None)
        self.assertEqual(found["status"], "unsupported")
        gate = status._sampling_gate({"sampled_speed": found})
        self.assertEqual(gate["state"], "unmeasured")

    def test_processed_distribution_applies_temperature_top_k_top_p_and_min_p(self):
        logprobs = [math.log(p) for p in (0.5, 0.3, 0.15, 0.05)]
        self.assertEqual([round(p, 6) for p in metrics.processed_distribution(logprobs, {"temperature": 1.0})],
                         [0.5, 0.3, 0.15, 0.05])
        top_k = metrics.processed_distribution(logprobs, {"temperature": 1.0, "top_k": 2})
        self.assertAlmostEqual(top_k[0], 0.5 / 0.8)
        self.assertEqual(top_k[2:], [0.0, 0.0])
        top_p = metrics.processed_distribution(logprobs, {"temperature": 1.0, "top_p": 0.75})
        self.assertEqual(sum(1 for p in top_p if p > 0), 2)  # 0.5 + 0.3 reaches 0.75
        cold = metrics.processed_distribution(logprobs, {"temperature": 0.5})
        self.assertGreater(cold[0], 0.5)
        min_p = metrics.processed_distribution(logprobs, {"temperature": 1.0, "min_p": 0.2})
        self.assertEqual(min_p[3], 0.0)  # 0.05 < 0.2 * 0.5

    def test_calibration_is_uniform_for_true_samples_and_skewed_for_greedy_ones(self):
        import random
        rng = random.Random(3)
        settings = {"temperature": 1.0, "top_k": 8}
        logprobs = [math.log(p) for p in (0.35, 0.2, 0.14, 0.1, 0.08, 0.06, 0.04, 0.03)]
        probs = metrics.processed_distribution(logprobs, settings)
        true_rows = [{"top": logprobs, "chosen": rng.choices(range(8), weights=probs)[0]} for _ in range(2000)]
        greedy_rows = [{"top": logprobs, "chosen": 0} for _ in range(2000)]
        fair = metrics.sampling_calibration(true_rows, settings, seed=7)
        self.assertLess(abs(fair["z"]), 4)
        self.assertLess(fair["ks"], 0.05)
        skewed = metrics.sampling_calibration(greedy_rows, settings, seed=7)
        self.assertLess(skewed["z"], -20)
        # Two of eight reported under top_k 8: the six kept unreported tokens' share is open, so it is unresolved.
        open_tail = metrics.sampling_calibration([{"top": logprobs[:2], "chosen": None}], settings)
        self.assertEqual((open_tail["tokens"], open_tail["outside"], open_tail["unresolved"]), (0, 0, 1))
        # top_k 2 keeps only reported tokens: one missing from them is outside.
        outside = metrics.sampling_calibration([{"top": logprobs[:4], "chosen": None}], {"temperature": 1.0, "top_k": 2})
        self.assertEqual((outside["tokens"], outside["outside"]), (0, 1))

    def test_the_unreported_rest_of_a_large_vocabulary_is_judged_where_the_alternatives_determine_it(self):
        import random
        rng = random.Random(5)
        vocab = 4000

        def position():
            # Zipf-like rows, some peaked and some flat, sorted likeliest first.
            exponent = rng.choice([2.5, 2.5, 1.6, 1.2])
            weights = [r ** -exponent * math.exp(rng.gauss(0, 0.3)) for r in range(1, vocab + 1)]
            total = sum(weights)
            return sorted((math.log(w / total) for w in weights), reverse=True)

        def full(row, settings):
            # The sampler's own processed distribution over the whole vocabulary.
            probs = metrics.processed_distribution(row, settings)
            return probs

        def check(settings, bias=0.0, n=600):
            rows = []
            for _ in range(n):
                row = position()
                probs = full(row, settings)
                k = 0 if rng.random() < bias else rng.choices(range(vocab), weights=probs)[0]
                rows.append({"top": row[:20], "chosen": k if k < 20 else None})
            return metrics.sampling_calibration(rows, settings, 7, vocab)
        for settings in ({"temperature": 1.0}, {"temperature": 1.0, "top_p": 0.95}, {"temperature": 1.0, "top_k": 64,
                         "top_p": 0.95}, {"temperature": 0.6, "top_p": 0.9}, {"temperature": 1.0, "min_p": 0.05}):
            with self.subTest(settings=settings):
                fair = check(settings)
                self.assertEqual(fair["outside"], 0)   # tokens past the top 20 are not counted outside
                self.assertGreater(fair["tokens"], 150)
                self.assertLess(abs(fair["z"]), 4)
        # Without filters the rest is exact: every position is judged, and a sampler favouring the likeliest fails.
        self.assertEqual(check({"temperature": 1.0})["unresolved"], 0)
        self.assertLess(check({"temperature": 1.0}, bias=0.3, n=1500)["z"], -4)

    def test_a_resolved_distribution_follows_the_filters_or_stays_open(self):
        top = [math.log(p) for p in (0.5, 0.3, 0.1)]   # 0.1 of the mass unreported
        # Temperature 1 without filters: the rest is one last alternative holding 0.1.
        self.assertEqual([round(p, 6) for p in metrics.resolved_distribution(top, {"temperature": 1.0})],
                         [0.5, 0.3, 0.1, 0.1])
        # top_p 0.75 ends at the second token whatever the rest holds.
        self.assertEqual([round(p, 6) for p in metrics.resolved_distribution(top, {"temperature": 1.0, "top_p": 0.75})],
                         [0.625, 0.375, 0.0, 0.0])
        # top_p 0.95 reaches into the unreported rest: open.
        self.assertIsNone(metrics.resolved_distribution(top, {"temperature": 1.0, "top_p": 0.95}))
        # min_p 0.3 drops the third token and every unreported one (each at most as likely as it).
        self.assertEqual([round(p, 6) for p in metrics.resolved_distribution(top, {"temperature": 1.0, "min_p": 0.3})],
                         [0.625, 0.375, 0.0, 0.0])
        # A top_p edge within rounding of a cumulative mass is left open.
        self.assertIsNone(metrics.resolved_distribution(top, {"temperature": 1.0, "top_p": 0.8}))

    def test_a_sampled_request_names_every_field_and_an_engine_need_apply_only_the_others(self):
        request = prepare.sampled_request({"temperature": 0.7, "top_k": 20}, 3)
        self.assertEqual(request, {"temperature": 0.7, "top_p": 1.0, "top_k": 20, "min_p": 0.0, "presence_penalty": 0.0,
                                   "frequency_penalty": 0.0, "repetition_penalty": 1.0, "seed": 3})
        self.assertEqual(prepare.unapplied(request, {"temperature", "top_k"}), [])   # the rest are neutral
        self.assertEqual(prepare.unapplied(request, {"temperature"}), ["top_k"])
        self.assertEqual(prepare.unapplied(request, set()), ["temperature", "top_k"])

    def test_the_sampled_suite_sends_neutral_penalties_and_refuses_an_engine_short_of_the_models_fields(self):
        protocol = runner.load_json(runner.EVAL_DIR / "protocol.json")
        prepared = {"speed": [{"id": "a", "kind": "prose", "prompt_ids": [1, 2], "output_tokens": 4}],
                    "sampling": {"temperature": 1.0, "top_p": 0.95}, "sampling_source": "generation_config"}
        sent = []

        class Engine:
            sampling_fields = {"temperature", "top_p", "seed"}

            def generate_sampled(self, prompt, n, on_tokens, sampling, logprobs=None):
                sent.append(dict(sampling))
                on_tokens([5] * n)
                return {"tokens": [5] * n, "rounds": n - 1}
        found = runner.run_sampled_speed(Engine(), {"max_context": 64}, prepared, protocol, 0, 2, lambda m: None)
        self.assertEqual(found["status"], "measured")
        self.assertEqual(sent[0], {"temperature": 1.0, "top_p": 0.95, "top_k": 0, "min_p": 0.0, "presence_penalty": 0.0,
                                   "frequency_penalty": 0.0, "repetition_penalty": 1.0, "seed": 1})
        Engine.sampling_fields = {"temperature", "seed"}
        found = runner.run_sampled_speed(Engine(), {"max_context": 64}, prepared, protocol, 0, 2, lambda m: None)
        self.assertEqual(found, {"status": "unsupported", "reason": "the engine does not apply the sampling's top_p"})

    def test_sampled_repetitions_run_every_seed_and_a_disturbance_shows_in_the_time_per_pass(self):
        protocol = runner.load_json(runner.EVAL_DIR / "protocol.json")
        prepared = {"speed": [{"id": "a", "kind": "prose", "prompt_ids": [1, 2], "output_tokens": 17}],
                    "sampling": {"temperature": 1.0}, "sampling_source": "protocol_default"}

        class Engine:
            """Each seed's text drafts differently: seed 2 accepts twice as many tokens per round. With slow, the
            third repetition's passes take three times as long."""
            def __init__(self, slow=False):
                self.slow, self.seeds = slow, []

            def generate_sampled(self, prompt, n, on_tokens, sampling, logprobs=None):
                self.seeds.append(sampling["seed"])
                per_round = 4 if sampling["seed"] == 2 else 2
                pause = 0.06 if self.slow and sampling["seed"] == 3 else 0.02   # above Windows' timer step
                for i in range(0, n, per_round):
                    time.sleep(pause)
                    on_tokens([5] * min(per_round, n - i))
                return {"tokens": [5] * n, "rounds": -(-(n - 1) // per_round)}
        quiet = Engine()
        found = runner.run_sampled_speed(quiet, {"max_context": 1000}, prepared, protocol, 1, None, lambda m: None,
                                         disturbed_spread=0.25)
        self.assertEqual(quiet.seeds, [0, 1, 2, 3])   # the warmup, then every seed, and no extra repetition
        decode = [s["decode_tps"] for s in found["cases"]["a"]["samples"] if not s["warmup"]]
        self.assertGreater(max(decode) / min(decode), 1.5)   # far past the disturbed spread, from the text alone
        with self.assertRaisesRegex(runner.Disturbed, "time per target pass"):
            runner.run_sampled_speed(Engine(slow=True), {"max_context": 1000}, prepared, protocol, 1, None,
                                     lambda m: None, disturbed_spread=0.25)

        class Served(Engine):
            """Behind a server: seed 3's reply ends early at a tool call, its few passes streamed slowly."""
            def generate_sampled(self, prompt, n, on_tokens, sampling, logprobs=None):
                if sampling["seed"] != 3:
                    out = Engine.generate_sampled(self, prompt, n, on_tokens, sampling)
                    return {**out, "served": {"prompt_tokens": 2, "output_tokens": n, "finish_reason": "length"}}
                for _ in range(3):
                    time.sleep(0.06)
                    on_tokens([5, 5])
                return {"tokens": [5] * 6, "rounds": 3,
                        "served": {"prompt_tokens": 2, "output_tokens": 6, "finish_reason": "tool_calls"}}
        found = runner.run_sampled_speed(Served(), {"max_context": 1000}, prepared, protocol, 1, None, lambda m: None,
                                         disturbed_spread=0.25)
        self.assertEqual(found["status"], "measured")   # judged on the two full-length replies alone

    def test_the_decode_score_carries_its_standard_error_and_small_changes_read_as_noise(self):
        # Two cases of three repetitions: log spreads give each case's variance of its mean.
        found = metrics.score_error([[100.0, 110.0, 90.0], [200.0, 200.0, 200.0]])
        logs = [math.log(v) for v in (100.0, 110.0, 90.0)]
        mean = sum(logs) / 3
        self.assertAlmostEqual(found, math.sqrt(sum((x - mean) ** 2 for x in logs) / 2 / 3) / 2)
        self.assertIsNone(metrics.score_error([[100.0]]))
        with tempfile.TemporaryDirectory() as directory, tempfile.TemporaryDirectory() as other:
            result, before = self.run_mock(directory), self.run_mock(other)
            sampled = result["sampled_speed"]
            self.assertIsNotNone(sampled["decode_se"])
            self.assertEqual(result["status"]["kpis"]["decode"]["standard_error"], sampled["decode_se"])
            # A change smaller than two standard errors of the difference is called noise.
            result["sampled_speed"]["decode_se"] = before["sampled_speed"]["decode_se"] = 0.03
            result["sampled_speed"]["decode_score"] = before["sampled_speed"]["decode_score"] * 1.02
            built = status.build(result, runner.load_json(runner.EVAL_DIR / "protocol.json"), before)
            self.assertAlmostEqual(built["baseline"]["decode_score_se"], math.sqrt(2) * 0.03)
            self.assertIn("within the noise: two standard errors are 8.5%", status.render(built))

    def test_prepare_freezes_the_models_own_sampling(self):
        from crit_core.text.generation import GenerationConfig
        self.assertEqual(prepare.sampled_settings(GenerationConfig(do_sample=True, temperature=0.7, top_p=0.8, top_k=20,
                                                                   repetition_penalty=1.05)),
                         {"temperature": 0.7, "top_p": 0.8, "top_k": 20})
        self.assertEqual(prepare.sampled_settings(GenerationConfig(do_sample=True)), {"temperature": 1.0})
        self.assertEqual(prepare.sampled_settings(GenerationConfig(temperature=0.6)), {"temperature": 1.0})
        self.assertEqual(prepare.sampling_source(GenerationConfig(do_sample=True, temperature=0.7)), "generation_config")
        self.assertEqual(prepare.sampling_source(GenerationConfig(temperature=0.6)), "protocol_default")
        self.assertEqual(prepare.sampling_source(GenerationConfig(do_sample=True, temperature=0.0)), "protocol_default")

    def test_only_a_well_formed_user_override_replaces_the_models_sampling(self):
        self.assertEqual(prepare.sampling_override("top_k=20, temperature=0.6,top_p=0.95"),
                         {"temperature": 0.6, "top_p": 0.95, "top_k": 20})
        self.assertEqual(prepare.sampling_override({"temperature": 1, "min_p": 0.05}), {"temperature": 1.0, "min_p": 0.05})
        for bad in ("top_p=0.9", "temperature=0", "temperature=0.6,top_k=2.5", "temperature=0.6,repetition_penalty=1.1",
                    "temperature=0.6,top_p=1.5", "temperature=0.6,min_p=1", "temperature", {"temperature": True}):
            with self.assertRaises(ValueError, msg=bad):
                prepare.sampling_override(bad)

    def test_a_user_override_is_recorded_and_labelled_and_not_compared_with_the_models_sampling(self):
        with tempfile.TemporaryDirectory() as directory, tempfile.TemporaryDirectory() as other:
            data = Path(directory)
            prepared = fixture(data)
            prepared.update(sampling={"temperature": 0.6, "top_k": 8}, sampling_source="override")
            (data / "prepared.json").write_text(json.dumps(prepared))
            config = {"max_context": 64, "bytes_per_pass": 1e6, "draft_bytes_per_step": 1e5}
            result = runner.run(MOCK, "unused-model", data, data / "result.json", config, warmups=0, repetitions=2,
                                log=lambda message: None, bandwidth_gbs=1.0)
            self.assertEqual((result["sampled_speed"]["sampling"], result["sampled_speed"]["sampling_source"]),
                             ({"temperature": 0.6, "top_k": 8}, "override"))
            self.assertEqual(result["sampling_check"]["sampling_source"], "override")
            self.assertEqual(result["status"]["kpis"]["decode"]["sampling_source"], "override")
            self.assertIn("at the user's sampling override (temperature 0.6, top_k 8)", status.render(result["status"]))
            self.assertIn("temperature 0.6, top_k 8 (user override)", report.compare([data / "result.json"]))
            default = SampledSpeedTests.run_mock(self, other)
            self.assertIsNone(status.changes(result, default)["decode_score"])


class StatusTests(unittest.TestCase):
    def run_mock(self, directory, device, **config):
        data = Path(directory)
        prepared = fixture(data)
        prepared["ceiling"] = {"matmul_flops_per_token": {"fp8": 1e8}, "attention_flops_per_pair": 0,
                               "recurrent_flops_per_token": 0, "kv_bytes_per_token": 0}
        (data / "prepared.json").write_text(json.dumps(prepared))
        config = {"max_context": 128, "bytes_per_pass": 1e6,
                  "draft_bytes_per_step": 1e5, **config}
        return runner.run(MOCK, "unused-model", data, data / "result.json", config, warmups=0, repetitions=2,
                          log=lambda message: None, device=device)

    def test_status_reports_gates_kpis_and_missing_inputs_without_a_verdict(self):
        with tempfile.TemporaryDirectory() as directory:
            result = self.run_mock(directory, {"bandwidth_gbs": 1.0, "tflops": {"fp8": 1.0, "bf16": 1.0}})
            current = result["status"]
            self.assertFalse({"verdict", "next"} & set(current))
            gates = {g["name"]: g["state"] for g in current["gates"]}
            # Exactness is reported by default, so it is a note rather than a gate.
            # The fixture's prepared data predates the held-out prompts: generality is unmeasured, not failed.
            # The fixture samples 12 tokens, fewer than the sampling check judges (SampledSpeedTests run it).
            self.assertEqual(gates, {"coverage": "pass", "repeatable": "pass", "request_history": "pass",
                                     "prompt_cache": "unmeasured", "quality": "unmeasured", "sampling": "unmeasured",
                                 "capacity": "unmeasured", "retrieval": "unmeasured", "lower_context": "unmeasured",
                                 "generality": "unmeasured"})
            self.assertIn("exactness reported: 2/2 cases equal plain greedy", current["notes"])
            self.assertIsNotNone(current["kpis"]["decode"]["speed_of_light"])
            self.assertIsNotNone(current["kpis"]["ttft"]["fraction_of_floor"])
            self.assertIn("ttft", status.render(current))
            self.assertIsNone(current["kpis"]["quality"])
            text = status.render(current)
            self.assertIn("quality: quality is measured through crit serve's score call", text)
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
            config = {"max_context": 128}
            budget = {"ppl_ratio_max": 1.5, "kld_mean_max": 1.0, "kld_p99_max": 10.0}
            result = runner.run(MOCK, "unused-model", data, data / "result.json", config, warmups=0, repetitions=2,
                                log=lambda message: None, quality_budget=budget, exactness="required")
            # crit eval reference judges the quality sets by the run's budget (QualityTests).
            self.assertEqual(result["requirements"], {"quality_budget": budget, "exactness": "required"})
            gates = {g["name"]: g["state"] for g in result["status"]["gates"]}
            self.assertEqual(gates["exactness"], "pass")
            with self.assertRaises(SystemExit):
                runner.run(MOCK, "unused-model", data, data / "bad.json", config, warmups=0, repetitions=1,
                           log=lambda message: None, quality_budget={"kld_max": 1.0})

    def test_exactness_runs_on_the_loaded_engine_when_it_decodes_plain_greedy(self):
        opened = []
        original = runner.open_engine

        def counting(module, model, config, speculative, context_tokens=None, **options):
            opened.append(speculative)
            return original(module, model, config, speculative, context_tokens, **options)
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
            config = {"max_context": 128, "disturbed": True}
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

    def test_a_result_of_another_major_version_keeps_its_stored_status(self):
        protocol = runner.load_json(runner.EVAL_DIR / "protocol.json")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "result.json"
            path.write_text(json.dumps({**MEASURED, "protocol": "crit-eval-v5"}), encoding="utf-8")
            with self.assertRaisesRegex(SystemExit, "measured with crit-eval-v5; its status is the one stored in it"):
                status.status_of(path, protocol)

    def test_differences_from_plain_greedy_fail_only_when_exactness_is_required(self):
        protocol = runner.load_json(runner.EVAL_DIR / "protocol.json")
        exact = {"cases": {"a": {"identical": False, "first_difference": 7}}}
        base = {**MEASURED, "engine": {"speculative": True},
                "speed": {"cases": {"a": {"status": "measured", "repeatable": True}}}}
        reported = status.build({**base, "exactness": {**exact, "mode": "reported"}}, protocol)
        self.assertNotIn("exactness", [g["name"] for g in reported["gates"]])
        self.assertIn("a differs at token 7; the quality budget judges the output", reported["notes"][0])
        required = status.build({**base, "requirements": {"exactness": "required"},
                                 "exactness": {**exact, "mode": "required"}}, protocol)
        self.assertEqual({g["name"]: g["state"] for g in required["gates"]}["exactness"], "fail")

    def test_end_to_end_score_is_a_fact_and_gaps_carry_no_end_to_end_gain(self):
        speed = {"cases": {
            "short": {"status": "measured", "output_tokens": 101, "ttft_ms": 10.0, "decode_tps": 200.0},
            "long": {"status": "measured", "output_tokens": 101, "ttft_ms": 3000.0, "decode_tps": 200.0}}}
        ttft = {"cases": {"short": {"floor": {"total_ms": 5.0}}, "long": {"floor": {"total_ms": 1000.0}}}}
        e2e = headroom.end_to_end_summary(speed, ttft, 200.0, 250.0)
        self.assertAlmostEqual(e2e["score"], ((101000 / 510) * (101000 / 3500)) ** 0.5)
        self.assertAlmostEqual(e2e["ceiling_score"], ((101000 / 405) * (101000 / 1400)) ** 0.5)
        self.assertFalse(hasattr(headroom, "end_to_end_gains"))

    def test_goals_take_every_kpi_to_its_limit_with_the_share_reached(self):
        protocol = runner.load_json(runner.EVAL_DIR / "protocol.json")
        room = {"projection": {"ceiling_score": 250.0, "fraction_of_ceiling": 0.88},
                "prefill": {"score": 5000.0, "ceiling_score": 14000.0, "fraction_of_ceiling": 5 / 14, "cases": {}},
                "ttft": {"score_ms": 300.0, "floor_ms": 200.0, "fraction_of_floor": 2 / 3, "cases": {}},
                "gaps": [{"kpi": "prefill", "name": "prefill", "projected_gain": 1.8, "end_to_end_gain": 0.14}],
                "missing": []}
        result = {**MEASURED, "engine": {"speculative": False}, "headroom": room,
                  "speed": {"decode_score": 220.0, "complete": True, "cases": {"a": {"status": "measured", "repeatable": True}}}}
        current = status.build(result, protocol)
        changes = {goal["kpi"]: round(goal["change"], 3) for goal in current["goals"]}
        self.assertEqual(changes, {"decode": round(250 / 220 - 1, 3), "prefill": 1.8, "ttft": round(200 / 300 - 1, 3)})
        shares = {goal["kpi"]: round(goal["fraction_of_limit"], 3) for goal in current["goals"]}
        self.assertEqual(shares, {"decode": 0.88, "prefill": round(5 / 14, 3), "ttft": round(2 / 3, 3)})
        text = status.render(current)
        # Eval states the goals; the run's priority rule orders them.
        self.assertIn("goals for the next candidate: each KPI to its physical limit (Eval states them; the run orders "
                      "them)", text)
        self.assertIn("prefill  5,000 -> 14,000 tok/s (+180%; 36% of its limit)", text)
        self.assertIn("ttft     300 -> 200 ms (-33%; 67% of its limit)", text)
        self.assertNotIn("end to end", text)  # older results' end-to-end gains are not shown
        self.assertNotIn("end_to_end_gain", current["gaps"][0])

    def test_every_kpi_has_its_noise_and_a_baseline_change_inside_it_is_marked(self):
        sample = lambda decode, prefill: {"decode_tps": decode, "prefill_tps": prefill, "ttft_ms": 10.0,  # noqa: E731
                                          "request_ms": 100.0, "output_tokens": 8}
        speed = {"cases": {"long": {"status": "measured", "samples": [sample(100, 1000), sample(104, 1040)]},
                           "chat": {"status": "measured", "samples": [sample(50, 200), sample(50, 200)]}}}
        found = headroom.noise(speed, ["long"])
        self.assertAlmostEqual(found["prefill"], 40 / 1020)   # the prefill cases alone
        self.assertIsNone(headroom.noise(speed, ["long", "missing"])["prefill"])
        protocol = runner.load_json(runner.EVAL_DIR / "protocol.json")

        def result(prefill, context_ttft):
            noise = {"decode": 0.01, "prefill": 0.05, "ttft": 0.02, "context_decode": 0.01, "context_ttft": 0.05}
            room = {"prefill": {"score": prefill, "ceiling_score": 2000.0, "fraction_of_ceiling": prefill / 2000,
                                "cases": {}},
                    "gaps": [{"kpi": "prefill", "name": "prefill at its compute ceiling", "projected_gain": 1.0}],
                    "noise": noise, "missing": [],
                    "context": {"decode": {"score": 100.0}, "ttft": {"score_ms": context_ttft}}}
            return {**MEASURED, "engine": {"speculative": False}, "headroom": room,
                    "context": {"tokens": 4096, "decode_text": {"source": "generated"}},
                    "speed": {"decode_score": 90.0, "ttft_score_ms": 10.0, "complete": True,
                              "cases": {"a": {"status": "measured", "repeatable": True}}}}
        current = status.build(result(1020.0, 1000.0), protocol, result(1000.0, 1300.0))
        self.assertEqual(current["kpis"]["prefill"]["noise"], 0.05)
        self.assertEqual(current["gaps"][0]["noise"], 0.05)
        self.assertEqual(current["baseline"]["noise"]["context_ttft"], 0.05)
        text = status.render(current)
        self.assertIn("+100% prefill score (noise 5.0%)", text)
        self.assertIn("prefill +2.0% (within the noise: spread 5.0%)", text)   # a 2% change inside a 5% spread
        self.assertIn("context ttft +30.0%,", text)   # 30% faster first tokens: outside it

    def test_status_reports_each_kpi_against_its_limit_and_no_classification(self):
        protocol = runner.load_json(runner.EVAL_DIR / "protocol.json")
        room = {"step_ms": 10.5, "pass_floor_ms": 10.0, "pass_efficiency": 10 / 10.5,
                "projection": {"ceiling_score": 100.0, "fraction_of_ceiling": 0.9, "speed_of_light_score": 95.0},
                "prefill": {"score": 800.0, "ceiling_score": 1000.0, "fraction_of_ceiling": 0.8, "cases": {}},
                "ttft": {"score_ms": 100.0, "floor_ms": 80.0, "fraction_of_floor": 0.8, "cases": {}},
                "levers": [{"kpi": "decode", "name": "old", "projected_gain": 0.1, "polished_kpi": True}], "missing": []}
        quality = {"status": "scored", "passed": True, "sets": {"conversations": {
            "passed": True, "failures": [], "kld_mean": 1e-5, "kld_p99": 1e-4, "ppl_ratio": 1.0, "top1_agreement": 1.0}}}
        result = {**MEASURED, "engine": {"speculative": False}, "headroom": room, "quality": quality,
                  "speed": {"decode_score": 90.0, "complete": True, "cases": {"a": {"status": "measured", "repeatable": True}}}}
        current = status.build(result, protocol)
        self.assertFalse({"verdict", "meaning", "next", "levers"} & set(current))
        self.assertEqual(current["gaps"], [{"kpi": "decode", "name": "old", "projected_gain": 0.1,
                                            "noise": None}])  # older results
        text = status.render(current)
        self.assertIn("decode   90.0 tok/s, 90% of its ceiling 100.0", text)
        self.assertIn("prefill  800 tok/s, 80% of its compute ceiling 1,000", text)
        self.assertNotIn("polished", text)
        self.assertNotIn("next step", text)
        self.assertEqual(current["stop_record_headroom"]["kpis"][0], {"kpi": "decode", "start": None, "reached": 90.0,
                                                                      "limit": 100.0, "built": "", "cause": ""})
        # A plain engine's ceiling is one token per pass at its floor, not a drafter's.
        room["projection"] = {"basis": "one token per pass", "ceiling_score": 100.0, "fraction_of_ceiling": 0.9}
        plain = status.build(result, protocol)
        self.assertEqual(plain["goals"][0]["limit"], 100.0)
        self.assertIn("decode   90.0 tok/s, 90% of its ceiling 100.0 (one token per model pass at its bandwidth floor)",
                      status.render(plain))

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
            prepared = fixture(data)
            config = {"max_context": 128}
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
            # Two repetitions, a third when the mock's timings happened to disagree (protocol speed.extra_repetition).
            self.assertIn(len(speed["cases"]["short"]["samples"]), (2, 3))
            self.assertTrue(speed["cases"]["short"]["repeatable"])
            self.assertIn(f"decode measured on the pinned continuation {pinned['sha256'][:12]}", result["status"]["notes"])
            free = runner.run(MOCK, "unused-model", data, data / "free.json", {**config, "replay": False}, warmups=0,
                              repetitions=1, log=lambda message: None, parts=("speed",))
            self.assertEqual(free["speed"]["decode_text"]["source"], "generated")
            compared = status.changes(free, result)
            self.assertTrue(compared["decode_text_differs"])
            self.assertIsNone(compared["greedy_decode_score"])  # different greedy texts are not compared
            # The decode KPI is the sampled one from v7: same sampling, so it compares.
            self.assertIsNotNone(compared["decode_score"])
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
            config = {"max_context": 128}
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
            # Neither the stored paths nor the error message name the directory the libraries were in.
            self.assertEqual([e["path"] for e in identity], [crit_eval.portable_path(library),
                                                             crit_eval.portable_path(Path(directory) / "missing.so")])
            self.assertNotIn(Path(directory).name, identity[1]["error"])

    def test_portable_paths_name_no_user(self):
        anchor = Path(ROOT.anchor)
        home = anchor / "home" / "tester"  # outside the checkout, which may hold the temporary directory
        with patch.object(crit_eval.Path, "home", return_value=home):
            for path, stored in ((ROOT / "eval" / "examples" / "mock_adapter.py", "eval/examples/mock_adapter.py"),
                                 (home / "engines" / "adapter.py", "~/engines/adapter.py"),
                                 (anchor / "Users" / "someone" / "build" / "libengine.so", "libengine.so"),
                                 (anchor / "mnt" / "c" / "Users" / "someone" / "adapter.py", "adapter.py"),
                                 (anchor / "home" / "someone" / "cuda" / "bin" / "nvcc", "nvcc"),
                                 (anchor / "usr" / "local" / "cuda-13.1" / "bin" / "nvcc",
                                  (anchor / "usr" / "local" / "cuda-13.1" / "bin" / "nvcc").as_posix()),
                                 (None, None)):
                with self.subTest(path=path):
                    self.assertEqual(crit_eval.portable_path(path), stored)

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

    def test_the_requested_context_caps_the_max_context_and_says_when_it_cannot_be_served(self):
        import crit_eval
        self.assertEqual(crit_eval.DEFAULT_CONTEXT_TOKENS, 262144)
        with tempfile.TemporaryDirectory() as directory:
            model = safetensors_model(directory, SHAPES, DENSE)
            device = {"device": "fixture gpu", "memory_free_bytes": self.overhead + RESIDENT + 32 * 20000}
            value = context.max_context(model, device, context.GIB, self.settings, requested=16384)
            self.assertEqual((value["tokens"], value["bound"], value["requested"], value["note"]),
                             (16384, "requested", 16384, None))
            self.assertIn("requested    16,384 tokens", context.render(value))
            value = context.max_context(model, device, context.GIB, self.settings, requested=262144)
            self.assertEqual((value["tokens"], value["bound"]), (19456, "device"))
            self.assertIn("the requested 262,144 tokens exceed the 19,456 tokens whose KV cache fits the device",
                          value["note"])
            (model / "config.json").write_text(json.dumps({**DENSE, "max_position_embeddings": 8192}))
            value = context.max_context(model, device, context.GIB, self.settings, requested=262144)
            self.assertEqual((value["tokens"], value["bound"]), (8192, "model"))
            self.assertIn("exceed the model's 8,192 positions", value["note"])
            requirement = runner.context_requirement(model, device, None, context.GIB, self.settings, 4096)
            self.assertEqual((requirement["tokens"], requirement["bound"], requirement["source"]), (4096, "requested", "computed"))
            given = runner.context_requirement(model, device, 8192, context.GIB, self.settings,
                                               basis={"requested": 262144, "bound": "model", "note": "clamped"})
            self.assertEqual((given["tokens"], given["source"], given["requested"], given["bound"]),
                             (8192, "given", 262144, "model"))

    def test_a_pass_reads_the_resident_weights_and_the_routed_experts_share(self):
        with tempfile.TemporaryDirectory() as directory:
            shapes = {**SHAPES, "model.layers.0.mlp.experts.1.up_proj.weight": (128, 64),
                      "model.layers.0.mlp.experts.2.up_proj.weight": (128, 64),
                      "model.layers.0.mlp.experts.3.up_proj.weight": (128, 64)}
            model = safetensors_model(directory, shapes, {**DENSE, "num_experts": 4, "num_experts_per_tok": 1})
            # Resident weights, then one of four experts; no vision tower or embedding rows.
            self.assertEqual(context.pass_bytes(model, 1.0), RESIDENT + 128 * 64 * 2)
            (model / "config.json").write_text(json.dumps(DENSE))   # no routing known: experts are not counted
            self.assertEqual(context.pass_bytes(model, 1.0), RESIDENT)
            self.assertIsNone(context.pass_bytes(Path(directory) / "missing", 1.0))

    def test_a_draft_step_reads_the_draft_layers_their_experts_share_and_the_head(self):
        mtp = {"mtp.fc.weight": (64, 128), "mtp.layers.0.self_attn.q_proj.weight": (64, 64),
               "mtp.layers.0.mlp.experts.0.up_proj.weight": (128, 64), "mtp.layers.0.mlp.experts.1.up_proj.weight": (128, 64),
               "mtp.embed_tokens.weight": (1000, 64)}
        draft_layers = (64 * 128 + 64 * 64) * 2
        with tempfile.TemporaryDirectory() as directory:
            model = safetensors_model(directory, {**SHAPES, **mtp}, {**DENSE, "num_experts": 2, "num_experts_per_tok": 1})
            # The draft layers, one of their two experts, and the model's head; their own embedding rows are gathered.
            self.assertEqual(context.draft_step_bytes(model), draft_layers + 128 * 64 * 2 + 1000 * 64 * 2)
        with tempfile.TemporaryDirectory() as directory:
            own = {**SHAPES, **mtp, "mtp.shared_head.head.weight": (500, 64)}
            model = safetensors_model(directory, own, {**DENSE, "num_experts": 2, "num_experts_per_tok": 1})
            self.assertEqual(context.draft_step_bytes(model), draft_layers + 128 * 64 * 2 + 500 * 64 * 2)
        with tempfile.TemporaryDirectory() as directory:
            self.assertIsNone(context.draft_step_bytes(safetensors_model(directory, SHAPES, DENSE)))   # no draft layers

    def test_a_tied_output_head_keeps_the_embedding_in_device_memory(self):
        with tempfile.TemporaryDirectory() as directory:
            shapes = {name: shape for name, shape in SHAPES.items() if name != "lm_head.weight"}
            model = safetensors_model(directory, shapes, {**DENSE, "tie_word_embeddings": True})
            split = context.weight_split(model, ceiling.model_config(model))
        self.assertEqual(split["embedding"], 0)
        self.assertEqual(split["resident"], 64 * 64 * 2 + 1000 * 64 * 2)
        self.assertEqual(split["vision"], 256 * 64 * 2)

    def test_curve_points_lower_context_and_curve_prompts(self):
        self.assertEqual(context.curve_points(262144, self.settings), [32768, 65536, 131072, 262144])
        self.assertEqual(context.curve_points(200000, self.settings), [32768, 65536, 131072, 200000])
        self.assertEqual(context.curve_points(8192, self.settings), [8192])
        self.assertEqual(context.lower_context(262144, 14000, self.settings), 65536)
        self.assertEqual(context.lower_context(8192, 72, self.settings), 2048)
        self.assertIsNone(context.lower_context(4096, 3500, self.settings))  # the longest case needs all of it

    def test_a_curve_prompt_is_the_retrieval_archive_with_its_own_request(self):
        retrieval = {"head_ids": [1], "tail_ids": [9], "curve_tail_ids": [7, 8, 8], "filler_ids": [5, 6],
                     "facts": [{"ids": [40]}, {"ids": [41]}]}
        prompt = context.curve_prompt(retrieval, 14, [0.0, 0.5])
        # The archive's head, filler with a fact at 0% and 50% of it, then the request for the facts and a summary.
        self.assertEqual(prompt, [1, 40, 5, 6, 5, 6, 41, 5, 6, 5, 6, 7, 8, 8])
        self.assertEqual(prompt[:-3], context.retrieval_prompt(retrieval, 12, [0.0, 0.5])[:-1])
        self.assertIsNone(context.curve_prompt(retrieval, 6, [0.0, 0.5]))   # no room for the frame and facts

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
        config = {"bytes_per_pass": 1e6, "draft_bytes_per_step": 1e5, **config}
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
    """Each long-context point asks the retrieval question, decodes the engine's own reply and judges retrieval on
    it: one prefill of each long prompt serves both."""

    class Engine:
        def __init__(self, answer):
            self.calls, self.answer = [], answer

        def generate(self, prompt, n, on_tokens):
            self.calls.append(len(prompt))
            tokens = (self.answer + [7] * n)[:n]
            on_tokens(tokens)
            return {"tokens": tokens, "rounds": n}

    prepared = {"speed": [{"id": "short", "prompt_ids": [1, 2]}, {"id": "long", "prompt_ids": [3, 4, 5, 6]}],
                "retrieval": {"head_ids": [30], "tail_ids": [31], "curve_tail_ids": [7], "filler_ids": [8, 9],
                              "facts": [{"id": "only", "ids": [40], "answer": "ab"}],
                              "token_text": ["", "a", "b"] + [""] * 50}, "stop_ids": []}
    settings = {"output_tokens": 4, "curve_from": 16, "repetitions": 1,
                "retrieval": {"positions": [0.5], "output_tokens": 4, "points": [12]}}

    def test_every_point_times_its_reply_and_judges_retrieval_on_it(self):
        engine = self.Engine([1, 2])   # "ab": the fact's answer
        points, text, asked = runner.run_context(engine, {"max_context": 64}, self.prepared, self.settings, 32,
                                                 lambda message: None)
        self.assertEqual(engine.calls, [12, 28])   # the curve's 16 and 32, one request each, prompt plus output
        self.assertEqual(text["source"], "generated")
        self.assertTrue(all(p["status"] == "measured" for p in points.values()))
        self.assertEqual({point: verdict["passed"] for point, verdict in asked.items()}, {16: True, 32: True})
        # The point below the curve gets a request of its own; the curve's points take their verdicts.
        retrieval = runner.run_retrieval(engine, {"max_context": 64}, self.prepared, self.settings, 32,
                                         lambda message: None, asked)
        self.assertEqual(list(retrieval["points"]), ["12", "16", "32"])
        self.assertEqual(engine.calls[2:], [8])
        self.assertTrue(retrieval["passed"])
        missing = runner.run_context(self.Engine([2, 1]), {"max_context": 64}, self.prepared, self.settings, 32,
                                     lambda message: None)[2]
        self.assertFalse(missing[32]["passed"])

    def test_context_decode_compares_only_on_the_same_text(self):
        pinned = {"context": {"tokens": 32, "decode_text": {"source": "pinned", "sha256": "ab" * 32}}}
        own = {"context": {"tokens": 32}}
        self.assertEqual(status.context_text_identity(pinned), "ab" * 32)
        self.assertEqual(status.context_text_identity(own), "generated")


class QualityTests(unittest.TestCase):
    """The engine's scores of the fixed quality texts against the reference's (protocol quality), judged per set and
    path, and the honesty check of its served replies."""

    LP = softmax_logprobs([3.0, 1.0, 0.5, 0.0])
    TOP = [7, 8, 9]

    def test_the_honesty_check_replies_to_the_conversations_and_one_long_text(self):
        protocol = runner.load_json(runner.EVAL_DIR / "protocol.json")
        texts = [{"id": text_id, "set": set_name, "prompt_ids": [1, 2, 3]} for text_id, set_name in (
            ("tides", "conversations"), ("letter", "conversations"), ("explain-tides@32768", "long-32768"),
            ("german-letter@32768", "long-32768"), ("explain-tides@4096", "long-4096"))]
        replied = []

        class Engine:
            def reply(self, prompt, n, top_k):
                return {"tokens": [5], "logprobs": [-0.5], "top": [[[5, -0.5]]]}

            def score(self, prompt, continuation, top_k, prompt_rows=True):
                replied.append(len(replied))
                return {"continuation": [{"logprob": -0.5, "top": [[5, -0.5]]}]}
        found = runner.run_honesty(Engine(), {}, {"quality": texts}, protocol, lambda m: None, {})
        # Both conversations and explain-tides at 32K; not the other 32K text, nor a set outside honesty's.
        self.assertEqual([r["id"] for r in found["replies"]], ["tides", "letter", "explain-tides@32768"])

    def lp(self, token):
        return self.LP[self.TOP.index(token)] if token in self.TOP else self.LP[3]

    def texts(self):
        """Two conversations and a long text whose prompt crosses position 512."""
        return [{"id": "a", "set": "conversations", "prompt_ids": [1, 7, 8], "continuation_ids": [7, 9, 8, 7]},
                {"id": "b", "set": "conversations", "prompt_ids": [2, 8], "continuation_ids": [9, 9]},
                {"id": "a@600", "set": "long-600", "prompt_ids": [1] + [7, 8] * 300, "continuation_ids": [8] * 40}]

    def reference(self, texts):
        return {t["id"]: [{"logprob": self.lp(token), "top_ids": list(self.TOP), "top_logprobs": self.LP[:3]}
                          for token in (t["prompt_ids"] + t["continuation_ids"])[1:]] for t in texts}

    class Engine:
        """Scores as the reference does, but `shift` nats off its top token at positions from `drift_from` (absolute)
        or, with `decode_drift`, at the continuation tokens after the first; its served replies `served_shift` off
        what it scores."""

        def __init__(self, test, drift_from=None, shift=0.0, decode_drift=0.0, served_shift=0.0, prompt_only=False):
            self.test, self.drift_from, self.shift = test, drift_from, shift
            self.decode_drift, self.served_shift, self.scored = decode_drift, served_shift, []
            self.prompt_only = prompt_only

        def row(self, token, position, decode, extra=0.0, prompt=False):
            drifts = self.drift_from is not None and position >= self.drift_from and (prompt or not self.prompt_only)
            off = self.shift if drifts else 0.0
            off += self.decode_drift if decode else 0.0
            top = [[7, self.test.LP[0] - off + extra], [8, self.test.LP[1] + extra], [9, self.test.LP[2] + extra]]
            return {"logprob": self.test.lp(token) - (off if token == 7 else 0.0) + extra, "top": top}

        def score(self, prompt, continuation, top_k, prompt_rows=True):
            self.scored.append((list(prompt), list(continuation)))
            return {"prompt": [self.row(t, i + 1, False, prompt=True) for i, t in enumerate(prompt[1:])]
                    if prompt_rows else [],
                    "continuation": [self.row(t, len(prompt) + j, j > 0) for j, t in enumerate(continuation)]}

        def reply(self, prompt, max_new_tokens, top_k):
            tokens = [7, 8, 7][:max_new_tokens]
            rows = [self.row(t, len(prompt) + j, j > 0, self.served_shift) for j, t in enumerate(tokens)]
            return {"tokens": tokens, "logprobs": [r["logprob"] for r in rows], "top": [r["top"] for r in rows]}

    def measure(self, engine, thresholds=None, max_context=10_000, honesty_sets=("conversations", "long-600")):
        protocol = runner.load_json(runner.EVAL_DIR / "protocol.json")
        protocol["quality"]["honesty"] = dict(protocol["quality"]["honesty"], sets=list(honesty_sets))
        thresholds = thresholds or protocol["quality"]["thresholds"]
        texts = self.texts()
        quality = runner.run_quality(engine, {"max_context": max_context}, {"quality": texts}, protocol,
                                     lambda message: None, self.reference(texts), thresholds)
        result = {**MEASURED, "engine": {"speculative": False}, "quality": quality, "model_sha256": "m", "suite_sha256": "s",
                  "protocol": protocol["protocol"], "speed": {"cases": {"a": {"status": "measured", "repeatable": True}}}}
        return result, status.build(result, protocol)

    def gate(self, current):
        return next(g for g in current["gates"] if g["name"] == "quality")

    def test_an_engine_scoring_as_the_reference_passes_every_set(self):
        engine = self.Engine(self)
        result, current = self.measure(engine)
        quality = result["quality"]
        # Judged: the prompt positions from the prefill and the served continuation (its first token from the
        # prefill's last position, the rest from decode).
        self.assertEqual(list(quality["sets"]), ["conversations/prompt", "conversations/served", "long-600/prompt",
                                                 "long-600/served"])
        self.assertNotIn("prompt", quality)
        self.assertEqual((quality["sets"]["conversations/served"]["targets"], quality["sets"]["conversations/prompt"]["targets"]),
                         (4 + 2, 2 + 1))
        self.assertEqual(quality["sets"]["conversations/served"]["texts"], ["a", "b"])
        self.assertTrue(all(s["passed"] and s["kld_mean"] == 0.0 and s["top1_agreement"] == 1.0
                            for s in quality["sets"].values()))
        self.assertEqual(list(quality["buckets"]["long-600/prompt"]), ["1-511", "512-1023"])
        self.assertEqual(list(quality["buckets"]["long-600/served"]), ["1-31", "32-63"])
        # The engine was given exactly the frozen ids: the forced continuation is the text's own.
        self.assertEqual(engine.scored[0], ([1, 7, 8], [7, 9, 8, 7]))
        self.assertEqual((quality["honesty"]["passed"], quality["honesty"]["max_abs_logprob_diff"]), (True, 0.0))
        self.assertTrue(quality["passed"])
        self.assertEqual(self.gate(current)["state"], "pass")
        self.assertIn("served replies scored alike", self.gate(current)["detail"])

    def test_drift_past_a_position_shows_in_its_bucket_and_fails_the_set(self):
        result, current = self.measure(self.Engine(self, drift_from=520, shift=2.0))
        buckets = result["quality"]["buckets"]["long-600/prompt"]
        self.assertEqual(buckets["1-511"]["kld_mean"], 0.0)
        self.assertGreater(buckets["512-1023"]["kld_mean"], 0.05)
        sets = result["quality"]["sets"]
        self.assertFalse(sets["long-600/served"]["passed"])   # every served position is past 520
        self.assertTrue(sets["conversations/served"]["passed"])
        gate = self.gate(current)
        self.assertEqual(gate["state"], "fail")
        self.assertIn("long-600/served: ", gate["detail"])
        self.assertEqual(current["kpis"]["quality"]["worst_kld_mean_set"], "long-600/served")

    def test_drift_in_the_prefill_alone_fails_the_prompt_positions(self):
        # A prefill that drifts while decode does not: every reply's first token comes from that arithmetic.
        result, current = self.measure(self.Engine(self, drift_from=520, shift=2.0, prompt_only=True))
        self.assertGreater(result["quality"]["buckets"]["long-600/prompt"]["512-1023"]["kld_mean"], 0.05)
        sets = result["quality"]["sets"]
        self.assertFalse(sets["long-600/prompt"]["passed"])
        self.assertTrue(sets["long-600/served"]["passed"] and sets["conversations/prompt"]["passed"])
        gate = self.gate(current)
        self.assertEqual(gate["state"], "fail")
        self.assertIn("long-600/prompt: ", gate["detail"])

    def test_drift_in_the_decode_path_fails_the_served_positions(self):
        result, _ = self.measure(self.Engine(self, decode_drift=0.4))
        sets = result["quality"]["sets"]
        self.assertFalse(sets["conversations/served"]["passed"] or sets["long-600/served"]["passed"])
        self.assertEqual(sets["conversations/prompt"]["kld_mean"], 0.0)

    def test_a_scoring_path_other_than_the_serving_path_fails_the_honesty_check(self):
        result, current = self.measure(self.Engine(self, served_shift=-0.01))
        honesty = result["quality"]["honesty"]
        self.assertFalse(honesty["passed"])
        self.assertAlmostEqual(honesty["max_abs_logprob_diff"], 0.01)
        self.assertTrue(all(s["passed"] for s in result["quality"]["sets"].values()))   # the fixed texts score fine
        self.assertFalse(result["quality"]["passed"])
        self.assertIn("differ from the served ones", self.gate(current)["detail"])
        self.assertEqual(self.gate(current)["state"], "fail")

    def test_the_runs_budget_judges_the_sets(self):
        budget = {"ppl_ratio_max": 2.0, "kld_mean_max": 1.0, "kld_p99_max": 10.0}
        result, current = self.measure(self.Engine(self, drift_from=520, shift=0.4), budget)
        row = result["quality"]["sets"]["long-600/served"]
        self.assertTrue(row["passed"])
        self.assertAlmostEqual(row["budget_used"], max((row["ppl_ratio"] - 1) / 1.0, row["kld_mean"], row["kld_p99"] / 10))
        self.assertEqual(self.gate(current)["state"], "pass")

    def test_a_set_beyond_the_context_is_skipped_and_its_replies_not_checked(self):
        engine = self.Engine(self)
        result, current = self.measure(engine, max_context=100)
        self.assertEqual(list(result["quality"]["sets"]), ["conversations/prompt", "conversations/served"])
        self.assertEqual(result["quality"]["skipped"], {"long-600": "needs 641 tokens of context"})
        self.assertEqual(len(result["quality"]["honesty"]["replies"]), 2)
        self.assertIn("skipped: long-600, needs 641 tokens of context", self.gate(current)["detail"])

    def test_quality_without_reference_scores_or_a_score_call_is_unmeasured(self):
        protocol = runner.load_json(runner.EVAL_DIR / "protocol.json")
        texts = self.texts()
        quiet = lambda message: None  # noqa: E731
        missing = runner.run_quality(self.Engine(self), {"max_context": 1000}, {"quality": texts}, protocol, quiet,
                                     None, {})
        self.assertIn("crit eval reference --model --data", missing["reason"])
        adapter = runner.run_quality(object(), {"max_context": 1000}, {"quality": texts}, protocol, quiet,
                                     self.reference(texts), {})
        self.assertEqual(adapter["status"], "unsupported")
        base = {**MEASURED, "engine": {"speculative": False}}
        for quality in (missing, adapter):
            gate = self.gate(status.build({**base, "quality": quality}, protocol))
            self.assertEqual(gate["state"], "unmeasured")
        tiny = status.build({**base, "quality": {"status": "skipped", "reason": "no quality text fits"}}, protocol)
        self.assertEqual(self.gate(tiny)["state"], "fail")

    def test_a_baseline_compares_within_its_data_suite_and_major_version_only(self):
        now, _ = self.measure(self.Engine(self))
        now["speed"]["decode_score"] = 110.0
        before = {"model_sha256": "m", "suite_sha256": "s", "protocol": now["protocol"], "speed": {"decode_score": 100.0},
                  "quality": {"sets": {"conversations/served": {"kld_mean": 0.01}}}}
        compared = status.changes(now, before)
        self.assertTrue(compared["comparable"])
        self.assertAlmostEqual(compared["decode_score"], 0.1)
        self.assertEqual(status.changes(now, dict(before, protocol="crit-eval-v5")),
                         {"comparable": False, "reason": "measured with crit-eval-v5"})
        for other in (dict(before, suite_sha256="older suite"), dict(before, model_sha256="other")):
            self.assertFalse(status.changes(now, other)["comparable"])

    def test_compare_shows_every_set_and_path(self):
        result, current = self.measure(self.Engine(self))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "result.json"
            row = {**result, "status": current, "engine": {"name": "e", "speculative": False}, "host": {"platform": "p"}}
            path.write_text(json.dumps({k: v for k, v in row.items() if k != "speed"}))
            table = report.compare([path, path])
        for name in ("conversations/served", "long-600/served"):
            self.assertIn(f"Quality: {name}", table)


class LongPromptTests(unittest.TestCase):
    """Retrieval up to the required context and request history."""

    def run_mock(self, directory, context_tokens=None, parts=runner.PARTS, **config):
        data = Path(directory)
        fixture(data)
        config = {"facts_from": FACTS, **config}
        return runner.run(MOCK, "unused-model", data, data / "result.json", config, parts=parts, warmups=0,
                          repetitions=2, log=lambda message: None, context_tokens=context_tokens)

    def gates(self, result):
        return {g["name"]: g for g in result["status"]["gates"]}

    def test_retrieval_prompts_hold_every_fact_at_its_position(self):
        prepared = fixture(Path(tempfile.mkdtemp()))
        pieces = prepared["retrieval"]
        rule = runner.load_json(runner.EVAL_DIR / "protocol.json")["context"]["retrieval"]
        prompt = context.retrieval_prompt(pieces, 1000, rule["positions"])
        self.assertEqual(len(prompt), 1000)
        self.assertEqual((prompt[:2], prompt[-2:]), ([1, 2], [3, 4]))
        where = [prompt.index(fact["ids"][0]) for fact in pieces["facts"]]
        self.assertEqual(where, sorted(where))
        self.assertLess(where[0], 50)
        self.assertTrue(450 < where[1] < 550)
        self.assertGreater(where[2], 940)
        self.assertIsNone(context.retrieval_prompt(pieces, 12, rule["positions"]))
        self.assertEqual(context.retrieval_points(262144, rule), [4096, 32768, 131072, 262144])
        self.assertEqual(context.retrieval_points(8192, rule), [4096, 8192])
        text = context.token_text([FACTS, FACTS + 1, 5, FACTS + 2, 999, FACTS + 3], pieces["token_text"], [999])
        self.assertEqual(text, "ab�c")  # an id without text cannot complete an answer; output stops at 999

    def test_retrieval_finds_every_fact_up_to_the_required_context(self):
        with tempfile.TemporaryDirectory() as directory:
            result = self.run_mock(directory, 8192, parts=("context",))
        retrieval = result["context"]["retrieval"]
        self.assertEqual(list(retrieval["points"]), ["4096", "8192"])
        point = retrieval["points"]["8192"]
        # It shares the curve's prompt at 8,192 up to the curve case's tail, which is as long as the question's.
        self.assertEqual((point["prompt_tokens"], point["found"]), (8192 - 256, {"start": True, "middle": True, "end": True}))
        self.assertTrue(retrieval["passed"])
        self.assertEqual(self.gates(result)["retrieval"]["state"], "pass")

    def test_retrieval_fails_when_a_long_prompt_loses_its_facts_or_is_not_served(self):
        with tempfile.TemporaryDirectory() as directory:
            forgetful = self.run_mock(directory, 8192, parts=("context",), forgets_above=6000)
        points = forgetful["context"]["retrieval"]["points"]
        self.assertTrue(points["4096"]["passed"])
        self.assertFalse(points["8192"]["passed"])
        gate = self.gates(forgetful)["retrieval"]
        self.assertEqual(gate["state"], "fail")
        self.assertIn("8,192: start, middle, end fact not found", gate["detail"])
        # Capacity alone passes: the engine served the request, it just answered wrongly.
        self.assertEqual(self.gates(forgetful)["capacity"]["state"], "pass")
        with tempfile.TemporaryDirectory() as directory:
            failing = self.run_mock(directory, 8192, parts=("context",), fails_above_context=6000)
        self.assertEqual(failing["context"]["retrieval"]["points"]["8192"]["status"], "failed")
        self.assertIn("8,192: RuntimeError: out of device memory", self.gates(failing)["retrieval"]["detail"])

    def test_request_history_catches_state_left_by_a_long_request(self):
        with tempfile.TemporaryDirectory() as directory:
            clean = self.run_mock(directory, parts=("speed", "history"), max_context=128)
        history = clean["history"]
        self.assertEqual(set(history["cases"]), {"short", "long"})  # the shortest and the longest prompt
        self.assertEqual(history["served_between"], ["speed"])
        self.assertEqual(self.gates(clean)["request_history"]["state"], "pass")
        with tempfile.TemporaryDirectory() as directory:
            # Outputs change once a 50-token prompt has been served: every speed repetition already sees the
            # changed state, so only the fresh engine's output tells.
            drifting = self.run_mock(directory, parts=("speed", "history"), max_context=128, drifts_after_prompt=50)
        gates = self.gates(drifting)
        self.assertEqual(gates["repeatable"]["state"], "pass")
        self.assertEqual(gates["request_history"]["state"], "fail")
        self.assertIn("short differs at token 0", gates["request_history"]["detail"])
        # A gate in every exactness mode.
        self.assertEqual(drifting["requirements"]["exactness"], "reported")

    def test_gates_of_parts_not_run_are_unmeasured(self):
        protocol = runner.load_json(runner.EVAL_DIR / "protocol.json")
        result = {"protocol": protocol["protocol"], "engine": {"speculative": False},
                  "requirements": {"exactness": "reported"},
                  "speed": {"cases": {"a": {"status": "measured", "repeatable": True}}},
                  "context": {"tokens": 8192, "points": {"8192": {"status": "measured"}}}}
        gates = {g["name"]: g["state"] for g in status.build(result, protocol)["gates"]}
        self.assertEqual((gates["request_history"], gates["retrieval"]), ("unmeasured", "unmeasured"))

    def test_a_drop_in_plain_greedy_equality_is_flagged_against_the_baseline(self):
        protocol = runner.load_json(runner.EVAL_DIR / "protocol.json")

        def result(*identical):
            return {**MEASURED, "engine": {"speculative": True}, "label": "parent",
                    "exactness": {"mode": "reported", "cases": {cid: {"identical": same, "first_difference": None if same else 3}
                                                                for cid, same in zip("abc", identical)}}}
        notes = status.build(result(True, False, False), protocol, result(True, True, True))["notes"]
        self.assertTrue(any("plain-greedy equality fell from 3/3 (parent) to 1/3 (b, c now differ)" in n
                            and "bug until a root cause is demonstrated" in n for n in notes))
        same = status.build(result(True, False, False), protocol, result(True, False, False))["notes"]
        self.assertFalse(any("equality fell" in n for n in same))

    def test_cli_parts_are_the_runners(self):
        from crit_eval import cli
        self.assertEqual(cli.PARTS, runner.PARTS)


class LongPrepareTests(unittest.TestCase):
    class Model(CharacterText):
        class tokenizer(CharacterText.tokenizer):
            vocab_size = 128

            @staticmethod
            def decode(ids):
                return "".join(chr(i) for i in ids)

        def render(self, messages, tools=None, add_generation_prompt=False, **variables):
            return "".join(f"<{m['role']}>{m['content']}" for m in messages) + ("<assistant>" if add_generation_prompt else "")

        def encode_chat(self, messages, tools=None, add_generation_prompt=False, **variables):
            return [ord(c) for c in self.render(messages, tools, add_generation_prompt)]

    def test_long_quality_prompts_render_to_their_length_from_their_messages(self):
        long = json.loads((ROOT / "eval/suite/long.json").read_text(encoding="utf-8"))
        conversation = {"id": "c", "messages": [{"role": "user", "content": "Question?"}]}
        model = self.Model()
        messages = prepare.long_quality_messages(model, conversation, long["quality"]["preamble"],
                                                 prepare.filler_ids(model, long), 3000)
        prompt = prepare.quality_prompt(model, "c@3000", "long-3000", messages, "The answer.", 5)
        self.assertEqual(len(prompt["prompt_ids"]), 3000)
        self.assertEqual(prompt["continuation_ids"], [ord(c) for c in "The a"])   # the written answer, cut
        self.assertEqual(prompt["prompt_ids"], prepare.prompt_ids(model, prompt["messages"]))
        text = "".join(chr(i) for i in prompt["prompt_ids"])
        self.assertTrue(text.startswith("<user>The following background material"))
        self.assertTrue(text.endswith("---\n\nQuestion?<assistant>"))

    def test_quality_sets_are_the_conversations_and_each_long_length(self):
        long = json.loads((ROOT / "eval/suite/long.json").read_text(encoding="utf-8"))
        quality = {"conversations": [{"id": c, "messages": [{"role": "user", "content": c}], "continuation": c * 3}
                                     for c in "ab"]}
        rule = {"long": {"prompt_tokens": [1000, 2000], "conversations": ["b"]}, "continuation_tokens": 2}
        model = self.Model()
        prompts = prepare.quality_prompts(model, quality, long, prepare.filler_ids(model, long), rule)
        self.assertEqual([(p["id"], p["set"]) for p in prompts],
                         [("a", "conversations"), ("b", "conversations"), ("b@1000", "long-1000"), ("b@2000", "long-2000")])
        self.assertEqual(prompts[0]["messages"], [{"role": "user", "content": "a"}])
        self.assertEqual([p["continuation_ids"] for p in prompts], [[97, 97]] + [[98, 98]] * 3)

    def test_retrieval_pieces_spell_every_answer(self):
        long = json.loads((ROOT / "eval/suite/long.json").read_text(encoding="utf-8"))
        model = self.Model()
        pieces = prepare.retrieval_pieces(model, long, prepare.filler_ids(model, long))
        prompt = context.retrieval_prompt(pieces, 20000, [0.02, 0.5, 0.97])
        self.assertEqual(len(prompt), 20000)
        text = context.token_text(prompt, pieces["token_text"])
        for fact in pieces["facts"]:
            self.assertEqual(text.count(fact["answer"]), 1)
        self.assertTrue(text.endswith("as STATION: identifier.<assistant>"))

    def test_suite_long_prompts_agree_with_the_protocol(self):
        protocol = json.loads((ROOT / "eval/protocol.json").read_text(encoding="utf-8"))
        long = json.loads((ROOT / "eval/suite/long.json").read_text(encoding="utf-8"))
        quality = json.loads((ROOT / "eval/suite/quality.json").read_text(encoding="utf-8"))
        self.assertTrue(set(protocol["quality"]["long"]["conversations"]) <= {c["id"] for c in quality["conversations"]})
        # Every quality prompt is the engine's to answer: no conversation ends with a reply of its own.
        self.assertTrue(all(c["messages"][-1]["role"] == "user" for c in quality["conversations"]))
        self.assertIn("{document}", long["quality"]["preamble"])
        filler = "".join((ROOT / "eval/suite" / name).read_text(encoding="utf-8") for name in long["filler"])
        for fact in long["retrieval"]["facts"]:
            self.assertIn(fact["answer"], fact["text"])
            self.assertTrue(fact["answer"].isascii())
            self.assertNotIn(fact["answer"], filler)
        self.assertEqual(len(long["retrieval"]["facts"]), len(protocol["context"]["retrieval"]["positions"]))


HAS_REFERENCE = all(importlib.util.find_spec(name) for name in ("torch", "transformers", "safetensors"))


def scored_text(ref, ids, top_k, valid_vocab=None, long=False):
    """(target log-probabilities of ids[1:], their top_k ids) from the reference's score_texts of one text."""
    with tempfile.TemporaryDirectory() as scratch:
        targets, tops, _ = ref.score_texts([ids], top_k, valid_vocab, scratch, log=lambda m: None,
                                           long={0} if long else ())[0]
    return targets, tops


@unittest.skipUnless(HAS_REFERENCE, "needs torch, transformers and safetensors")
class ReferenceTests(unittest.TestCase):
    def test_long_prompts_score_like_eager_attention_without_its_mask(self):
        import torch
        from transformers import Qwen3Config, Qwen3ForCausalLM
        from crit_eval import reference
        torch.manual_seed(0)
        config = Qwen3Config(vocab_size=128, hidden_size=64, intermediate_size=128, num_hidden_layers=2,
                             num_attention_heads=4, num_key_value_heads=2, head_dim=16, max_position_embeddings=4096,
                             tie_word_embeddings=False)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        with tempfile.TemporaryDirectory() as folder:
            Qwen3ForCausalLM(config).save_pretrained(folder)
            ids = torch.randint(0, 128, (2 * reference.QUERY_BLOCK + 100,)).tolist()
            eager = scored_text(reference.StreamingReference(folder, device), ids, 4, 128)
            chunked = scored_text(reference.StreamingReference(folder, device), ids, 4, 128, long=True)
        self.assertTrue(torch.equal(eager[1], chunked[1]))
        self.assertLess(float((eager[0] - chunked[0]).abs().max()), 1e-5)

    def tiny_models(self):
        """(name, model) for a Llama-style model and, where transformers has it, a hybrid one whose cache keeps
        linear-attention conv and recurrent states beside the full-attention layer's keys and values."""
        import torch
        from transformers import Qwen3Config, Qwen3ForCausalLM
        torch.manual_seed(0)
        yield "llama-style", Qwen3ForCausalLM(Qwen3Config(
            vocab_size=128, hidden_size=64, intermediate_size=128, num_hidden_layers=2, num_attention_heads=4,
            num_key_value_heads=2, head_dim=16, max_position_embeddings=4096, tie_word_embeddings=False))
        if importlib.util.find_spec("transformers.models.qwen3_next"):
            from transformers import Qwen3NextConfig, Qwen3NextForCausalLM
            model = Qwen3NextForCausalLM(Qwen3NextConfig(
                vocab_size=128, hidden_size=64, intermediate_size=128, num_hidden_layers=4, num_attention_heads=4,
                num_key_value_heads=2, head_dim=16, linear_num_key_heads=2, linear_num_value_heads=4,
                linear_key_head_dim=16, linear_value_head_dim=16, moe_intermediate_size=32,
                shared_expert_intermediate_size=32, num_experts=4, num_experts_per_tok=2,
                max_position_embeddings=4096, tie_word_embeddings=False,
                layer_types=["linear_attention"] * 3 + ["full_attention"]))
            with torch.no_grad():
                for param in model.parameters():
                    param.add_(torch.randn_like(param) * 0.05)   # off the initialisation (zero norms)
            yield "hybrid", model

    def layer_major_models(self):
        """tiny_models, and where transformers has them a Gemma 4 text model (sliding and full layers, so the forward
        reads two layers' caches) and a qwen4_exp one (n-gram context, full positions and an indexer in the cache)."""
        import torch
        yield from self.tiny_models()
        from transformers import Gemma4ForCausalLM, Gemma4TextConfig
        torch.manual_seed(0)
        yield "gemma4", Gemma4ForCausalLM(Gemma4TextConfig(
            vocab_size=128, hidden_size=64, intermediate_size=128, num_hidden_layers=4, num_attention_heads=4,
            num_key_value_heads=2, num_global_key_value_heads=1, head_dim=16, global_head_dim=32, sliding_window=40,
            max_position_embeddings=1024, layer_types=["sliding_attention"] * 3 + ["full_attention"],
            hidden_size_per_layer_input=0, tie_word_embeddings=True))
        if importlib.util.find_spec("transformers.models.qwen4_exp"):
            from transformers import Qwen4ExpForCausalLM, Qwen4ExpTextConfig
            model = Qwen4ExpForCausalLM(Qwen4ExpTextConfig(
                vocab_size=128, hidden_size=64, num_hidden_layers=4, num_attention_heads=4, num_key_value_heads=2,
                head_dim=16, linear_num_key_heads=2, linear_num_value_heads=4, linear_key_head_dim=16,
                linear_value_head_dim=16, moe_intermediate_size=32, shared_expert_intermediate_size=32,
                num_experts=4, num_experts_per_tok=2, hc_lowrank=8, ple_layer_ids=[2], ngram_vocab_size_base=50,
                split_ngram_parts=3, indexer_n_heads=2, indexer_kv_heads=1, indexer_head_dim=16, indexer_budget=8,
                indexer_compress_ratio=4, eos_token_id=1, max_position_embeddings=4096, tie_word_embeddings=False,
                rope_parameters={"rope_type": "default", "rope_theta": 10000.0, "partial_rotary_factor": 0.5,
                                 "mrope_section": [2, 1, 1], "mrope_interleaved": True}))
            with torch.no_grad():
                for param in model.parameters():
                    param.add_(torch.randn_like(param) * 0.05)
            yield "qwen4_exp", model

    def score_texts(self, folder, texts, **patches):
        """(rows, log lines, layer loads) of StreamingReference.score_texts on the saved model, with `patches`
        ({attribute: replacement}) set on the reference first."""
        from crit_eval import reference
        ref = reference.StreamingReference(folder, "cpu")
        for name, value in patches.items():
            setattr(ref, name, value)
        loads, layer, logged = [], ref.layer, []

        def counted(index, upcoming=None):
            loads.append(index)
            return layer(index, upcoming)
        ref.layer = counted
        with tempfile.TemporaryDirectory() as scratch:
            rows = ref.score_texts(texts, 4, None, Path(scratch) / "states", log=logged.append, long={len(texts) - 1})
        return rows, logged, loads

    def test_layer_major_texts_load_each_layer_once_where_the_forward_allows(self):
        # All chunks of every text run on a layer while it is loaded, but for the layers up to the one whose cache
        # state the forward reads between chunks; the rows equal the chunk-major order's exactly.
        import torch
        from crit_eval import reference
        torch.manual_seed(2)
        texts = [torch.randint(2, 128, (n,)).tolist() for n in (150, 37, 260)]
        for name, model in self.layer_major_models():
            with tempfile.TemporaryDirectory() as folder, patch.object(reference, "PROMPT_CHUNK", 64):
                model.float().eval().save_pretrained(folder)
                count = reference.StreamingReference(folder, "cpu")._layer_count()
                rows, _, loads = self.score_texts(folder, texts)
                chunked, _, rounds = self.score_texts(
                    folder, texts, _layer_major=unittest.mock.Mock(side_effect=reference._OrderUnsafe("forced")))
                for a, b in zip(rows, chunked):
                    self.assertTrue(all(torch.equal(x, y) for x, y in zip(a, b)), name)
                self.assertEqual(len(rounds), 5 * count, name)   # five chunks of the longest text, each a round
                # The llama-style forward reads layer 0, which runs every chunk while it is loaded: each layer loads
                # once. The others read their only full-attention layer, the last, which leaves nothing to save.
                self.assertLessEqual(len(loads), len(rounds), name)
                if name == "llama-style":
                    self.assertEqual(sorted(loads), list(range(count)))

    def test_a_layer_reading_another_layers_cache_runs_the_texts_chunk_major(self):
        # A decoder layer that reads another layer's cache would see it further along in the layer-major order: the
        # reference notices the read and runs the texts chunk by chunk, with the same rows.
        import torch
        from transformers.models.qwen3 import modeling_qwen3
        from crit_eval import reference
        torch.manual_seed(3)
        texts = [torch.randint(2, 128, (n,)).tolist() for n in (150, 40)]
        name, model = next(self.tiny_models())
        forward = modeling_qwen3.Qwen3DecoderLayer.forward

        def reading(self, *args, **kwargs):
            kwargs["past_key_values"].get_seq_length()   # layer 0's keys
            return forward(self, *args, **kwargs)
        with tempfile.TemporaryDirectory() as folder, patch.object(reference, "PROMPT_CHUNK", 64):
            model.float().eval().save_pretrained(folder)
            plain, _, _ = self.score_texts(folder, texts)
            with patch.object(modeling_qwen3.Qwen3DecoderLayer, "forward", reading):
                rows, logged, _ = self.score_texts(folder, texts)
        for a, b in zip(plain, rows):
            self.assertTrue(all(torch.equal(x, y) for x, y in zip(a, b)))
        self.assertTrue(any("reads the cache of layer 0" in line and "chunk by chunk" in line for line in logged), logged)

    def test_fixed_texts_score_every_position_like_the_model(self):
        # Every position of each text, run in chunks layer-major (or chunk-major, or whole when the cache cannot
        # be resumed), scores as the model's own FP32 forward over the whole text; the scratch states are removed.
        import torch
        from crit_eval import reference
        torch.manual_seed(4)
        texts = [torch.randint(2, 128, (n,)).tolist() for n in (150, 37, 200)]
        for name, model in self.layer_major_models():
            with tempfile.TemporaryDirectory() as folder, tempfile.TemporaryDirectory() as scratch, \
                    patch.object(reference, "PROMPT_CHUNK", 64), patch.object(reference, "SCORE_ROWS", 50):
                model.float().eval().save_pretrained(folder)
                with torch.no_grad():
                    expected = [torch.log_softmax(model.float()(torch.tensor([ids]), use_cache=False).logits[0, :-1]
                                                  .double(), dim=-1) for ids in texts]
                runs = {}
                for order in ("layer-major", "chunk-major", "whole"):
                    ref = reference.StreamingReference(folder, "cpu")
                    if order == "chunk-major":
                        ref._layer_major = unittest.mock.Mock(side_effect=reference._OrderUnsafe("forced"))
                    if order == "whole":
                        ref._layer_major = unittest.mock.Mock(side_effect=reference.CacheUnsupported("forced"))
                    runs[order] = ref.score_texts(texts, 4, None, Path(scratch) / "states", log=lambda m: None,
                                                  long={2})
                    self.assertFalse((Path(scratch) / "states").exists(), (name, order))
                for order, rows in runs.items():
                    for ids, want, (target, top, values) in zip(texts, expected, rows):
                        lp = want.gather(1, torch.tensor(ids[1:])[:, None])[:, 0]
                        self.assertEqual((target.shape, top.shape), ((len(ids) - 1,), (len(ids) - 1, 4)))
                        self.assertLess(float((target - lp).abs().max()), 1e-4, (name, order))
                        self.assertEqual(top.tolist(), want.topk(4, dim=-1).indices.tolist(), (name, order))
                for (a, b, c), (x, y, z) in zip(runs["layer-major"], runs["chunk-major"]):
                    self.assertTrue(torch.equal(a, x) and torch.equal(b, y) and torch.equal(c, z), name)

    def test_fixed_text_scores_are_stored_once_per_identity(self):
        # Scored once, stored with each file's sha256 and read in pure Python as scored; read thereafter, another
        # identity scores again, and a file whose bytes changed is refused.
        import torch
        from crit_eval import fixed, reference
        torch.manual_seed(5)
        texts = {"a": (torch.randint(2, 128, (90,)).tolist(), False),
                 "b@32768": (torch.randint(2, 128, (40,)).tolist(), True)}
        name, model = next(self.tiny_models())
        with tempfile.TemporaryDirectory() as folder, tempfile.TemporaryDirectory() as data:
            model.float().eval().save_pretrained(folder)
            ref = reference.StreamingReference(folder, "cpu")
            how = reference.store_texts(ref, texts, 4, None, data, "sha", log=lambda m: None)
            self.assertEqual((how["method"], how["positions"]), ("scored", 89 + 39))
            store = Path(data) / fixed.STORE
            self.assertEqual(sorted(p.name for p in store.iterdir())[0], "index.json")
            self.assertEqual(len(list(store.glob("text-*.safetensors"))), 2)
            expected = {"model_sha256": "sha", "reference": reference.REFERENCE_VERSION,
                        **fixed.identity(texts, 4, None)}
            read = fixed.load(data, expected)
            with tempfile.TemporaryDirectory() as scratch:
                direct = reference.StreamingReference(folder, "cpu").score_texts(
                    [ids for ids, _ in texts.values()], 4, None, scratch, log=lambda m: None, long={1})
            for text_id, (target, top, values) in zip(texts, direct):
                rows = read[text_id]
                self.assertEqual([r["logprob"] for r in rows], target.tolist())
                self.assertEqual([r["top_ids"] for r in rows], top.tolist())
                self.assertEqual([r["top_logprobs"] for r in rows], values.tolist())
            self.assertIsNone(fixed.load(data, dict(expected, top_k=8)))
            with patch.object(reference.StreamingReference, "score_texts", side_effect=AssertionError("scored")):
                how = reference.store_texts(reference.StreamingReference(folder, "cpu"), texts, 4, None, data, "sha",
                                            log=lambda m: None)
            self.assertEqual(how["method"], "stored")
            how = reference.store_texts(reference.StreamingReference(folder, "cpu"), texts, 4, None, data,
                                        "another sha", log=lambda m: None)
            self.assertEqual(how["method"], "scored")
            victim = next(store.glob("text-*.safetensors"))
            victim.write_bytes(victim.read_bytes()[:-1] + b"\0")
            with self.assertRaises(SystemExit):
                reference.store_texts(reference.StreamingReference(folder, "cpu"), texts, 4, None, data,
                                      "another sha", log=lambda m: None)

    def test_a_cache_that_cannot_be_stored_runs_each_text_whole(self):
        import torch
        from crit_eval import reference
        torch.manual_seed(1)
        texts = [torch.randint(2, 128, (n,)).tolist() for n in (90, 40)]
        name, model = next(self.tiny_models())
        with tempfile.TemporaryDirectory() as folder, patch.object(reference, "PROMPT_CHUNK", 64):
            model.float().eval().save_pretrained(folder)
            plain, _, _ = self.score_texts(folder, texts)

            def unstorable(value, path, tensors):
                raise reference.CacheUnsupported(f"cache state {path} is a Widget")
            with patch.object(reference, "_encode", unstorable):
                rows, logged, _ = self.score_texts(folder, texts)
        for a, b in zip(plain, rows):
            self.assertLess(float((a[0] - b[0]).abs().max()), 1e-5)
            self.assertEqual(a[1].tolist(), b[1].tolist())
        self.assertTrue(any("is a Widget" in line and "each text runs whole" in line for line in logged), logged)

    def test_the_store_format_round_trips_in_pure_python(self):
        from crit_eval import fixed
        with tempfile.TemporaryDirectory() as data:
            scores = {"a": ([-0.5, -1.25], [[3, 4], [5, 6]], [[-0.5, -2.0], [-1.25, -3.5]]), "b": ([-0.75], [[1, 2]],
                                                                                                    [[-0.75, -4.0]])}
            fixed.write(data, {"top_k": 2, "x": 1}, scores)
            read = fixed.load(data, {"top_k": 2})
            self.assertEqual(read["a"], [{"logprob": -0.5, "top_ids": [3, 4], "top_logprobs": [-0.5, -2.0]},
                                         {"logprob": -1.25, "top_ids": [5, 6], "top_logprobs": [-1.25, -3.5]}])
            self.assertEqual(read["b"][0]["top_ids"], [1, 2])
            self.assertIsNone(fixed.load(data, {"top_k": 3}))
            texts = fixed.texts({"quality": [{"id": "a", "set": "conversations", "prompt_ids": [3, 4],
                                              "continuation_ids": [5]},
                                             {"id": "a@600", "set": "long-600", "prompt_ids": [3], "continuation_ids": [9]}]})
            self.assertEqual(texts, {"a": ([3, 4, 5], False), "a@600": ([3, 9], True)})

    def test_gathered_rows_are_the_stored_rows(self):
        # Rows read by parallel reads of exactly their bytes are the stored rows, in lookup order with repeats,
        # whatever their dtype, file and run of consecutive rows.
        import torch
        from safetensors.torch import save_file
        from crit_eval import reference
        torch.manual_seed(0)
        tables = {"a.weight": torch.randn(300, 24).to(torch.bfloat16),
                  "b.weight": torch.randn(97, 16).to(torch.float8_e4m3fn),
                  "c.weight": torch.randint(0, 256, (500, 5), dtype=torch.uint8),
                  "d.weight_scale": torch.rand(500, 1)}
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            save_file({k: tables[k] for k in ("a.weight", "b.weight")}, folder / "model-00001-of-00002.safetensors")
            save_file({k: tables[k] for k in ("c.weight", "d.weight_scale")}, folder / "model-00002-of-00002.safetensors")
            weight_map = {"a.weight": "model-00001-of-00002.safetensors", "b.weight": "model-00001-of-00002.safetensors",
                          "c.weight": "model-00002-of-00002.safetensors",
                          "d.weight_scale": "model-00002-of-00002.safetensors"}
            (folder / "model.safetensors.index.json").write_text(json.dumps({"metadata": {}, "weight_map": weight_map}))
            (folder / "config.json").write_text(json.dumps({"model_type": "llama"}))
            checkpoint = reference.Checkpoint(folder)
            requests = [(key, torch.cat([torch.randint(0, len(table), (200,)), torch.arange(5, 40),
                                         torch.tensor([len(table) - 1, 0, 0])]))
                        for key, table in tables.items()]
            requests.append(("a.weight", torch.tensor([], dtype=torch.int64)))
            for threads in (1, 3):
                with patch.object(reference, "GATHER_THREADS", threads):
                    found = checkpoint.rows_many(requests)
                for (key, index), rows in zip(requests, found):
                    expected = tables[key][index]
                    self.assertEqual((rows.dtype, rows.shape), (expected.dtype, expected.shape), key)
                    self.assertTrue(torch.equal(rows.view(torch.uint8), expected.view(torch.uint8)), key)
            self.assertTrue(torch.equal(checkpoint.rows("c.weight", torch.tensor([7, 7, 3])), tables["c.weight"][[7, 7, 3]]))

    def test_staged_tensors_are_the_stored_tensors(self):
        # A layer's tensors read in large reads into one buffer and placed in one transfer are the stored tensors:
        # misaligned ones (after an odd-sized tensor), ones read through a gap, in pieces and over several files.
        import torch
        from safetensors.torch import save_file
        from crit_eval import reference
        torch.manual_seed(0)
        first = {"l.a": torch.randint(0, 256, (7,), dtype=torch.uint8), "l.b": torch.randn(5, 3),
                 "l.c": torch.randn(4, 6).to(torch.bfloat16), "skip": torch.randn(300), "l.d": torch.tensor(2.5),
                 "l.e": torch.randn(64, 8).to(torch.float8_e4m3fn)}
        second = {"l.f": torch.randn(33), "l.g": torch.randint(-5, 5, (3, 3), dtype=torch.int32)}
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            save_file(first, folder / "model-00001-of-00002.safetensors")
            save_file(second, folder / "model-00002-of-00002.safetensors")
            weight_map = {**{k: "model-00001-of-00002.safetensors" for k in first},
                          **{k: "model-00002-of-00002.safetensors" for k in second}}
            (folder / "model.safetensors.index.json").write_text(json.dumps({"metadata": {}, "weight_map": weight_map}))
            (folder / "config.json").write_text(json.dumps({"model_type": "llama"}))
            checkpoint = reference.Checkpoint(folder)
            keys = [k for k in {**first, **second} if k.startswith("l.")]
            for gap, piece in ((0, 64 << 20), (1 << 20, 16)):
                with patch.object(reference, "STAGE_GAP", gap), patch.object(reference, "STAGE_READ", piece):
                    buffer, layout = checkpoint.stage(keys, torch.zeros(3, dtype=torch.uint8))
                    checkpoint.place(buffer, layout, "cpu")
                staged = dict(checkpoint.placed)
                checkpoint.placed = {}
                self.assertEqual(set(staged), set(keys))
                for key in keys:
                    stored = {**first, **second}[key]
                    self.assertEqual((staged[key].dtype, staged[key].shape), (stored.dtype, stored.shape), key)
                    self.assertTrue(torch.equal(staged[key].reshape(-1).view(torch.uint8), stored.reshape(-1).view(torch.uint8)),
                                    key)
                    self.assertTrue(torch.equal(checkpoint.raw(key).reshape(-1).view(torch.uint8),
                                                stored.reshape(-1).view(torch.uint8)))

    def test_layers_load_alike_staged_or_read_tensor_by_tensor(self):
        # Every decoder layer's parameters and buffers are bit-identical whether its stored tensors were read ahead
        # into one buffer or read one by one, and no parameter keeps the staged copy alive.
        import torch
        from crit_eval import reference
        for name, model in self.tiny_models():
            with tempfile.TemporaryDirectory() as folder:
                model.float().eval().save_pretrained(folder)
                staged, plain = (reference.StreamingReference(folder, "cpu") for _ in range(2))
                count = staged._layer_count()
                for index in range(count):
                    a = staged.layer(index, index + 1 if index + 1 < count else None)
                    with patch.object(reference, "STAGE_LIMIT", 0):
                        b = plain.layer(index)
                    self.assertEqual(plain.ahead, None)
                    for (key, x), (_, y) in zip(a.state_dict().items(), b.state_dict().items()):
                        self.assertTrue(x.is_meta or torch.equal(x, y), (name, index, key))
                        self.assertTrue(x.is_meta or x.untyped_storage().nbytes() == y.untyped_storage().nbytes(),
                                        (name, index, key))
                    self.assertTrue(staged.checkpoint.placed == {})
                self.assertIsNone(staged.ahead)

    def test_chunked_attention_honours_a_supplied_mask(self):
        # A boolean or additive mask further restricts each block of query rows, as eager attention applies it.
        import torch
        from crit_eval import reference
        torch.manual_seed(0)
        length, heads, kv_heads, dim = reference.QUERY_BLOCK + 70, 4, 2, 8
        query = torch.randn(1, heads, length, dim)
        key, value = torch.randn(1, kv_heads, length, dim), torch.randn(1, kv_heads, length, dim)
        causal = torch.ones(length, length, dtype=torch.bool).tril()
        mask = (causal & (torch.rand(length, length) < 0.3) | torch.eye(length, dtype=torch.bool))[None, None]
        scores = query @ key.repeat_interleave(2, 1).transpose(-2, -1) * dim ** -0.5
        expected = (torch.softmax(scores.masked_fill(~mask, float("-inf")), -1) @ value.repeat_interleave(2, 1))
        expected = expected.transpose(1, 2)
        additive = torch.zeros(mask.shape).masked_fill(~mask, torch.finfo(torch.float32).min)
        for supplied in (mask, additive):
            got, _ = reference.chunked_attention(None, query, key, value, supplied)
            self.assertLess(float((got - expected).abs().max()), 1e-5)

    def test_nvfp4_layouts_dequantize_alike(self):
        # compressed-tensors divides the group scales by weight_global_scale; ModelOpt multiplies by weight_scale_2.
        import torch
        from crit_eval import reference, schemes
        torch.manual_seed(0)
        packed = torch.randint(0, 256, (4, 16), dtype=torch.uint8)
        scale = (torch.rand(4, 2) * 4 + .5).to(torch.float8_e4m3fn)

        def checkpoint(tensors):
            ckpt = object.__new__(reference.StateDictCheckpoint)
            ckpt.where, ckpt.quantization = tensors, {}
            return ckpt
        ct = checkpoint({"w.weight_packed": packed, "w.weight_scale": scale, "w.weight_global_scale": torch.tensor(4.0)})
        mo = checkpoint({"w.weight": packed, "w.weight_scale": scale, "w.weight_scale_2": torch.tensor(0.25)})
        a, b = ct.dense("w", "cpu"), mo.dense("w", "cpu")
        self.assertEqual(a.shape, (4, 32))
        self.assertTrue(torch.equal(a, b))
        first = float(schemes.E2M1[int(packed[0, 0]) & 7] * (-1 if int(packed[0, 0]) & 8 else 1))
        self.assertEqual(float(a[0, 0]), first * float(scale[0, 0].float()) / 4)

    def assert_scores_like_the_model(self, model, folder, ids, first, top_k=4, save=None):
        """The layer-streamed reference, eager and chunked, scores like the model's own FP32 forward. save(folder)
        writes the checkpoint (default: save_pretrained)."""
        import torch
        from crit_eval import reference
        (save or model.save_pretrained)(folder)
        with torch.no_grad():
            expected = torch.log_softmax(model.float()(torch.tensor([ids]), use_cache=False).logits[0].double(),
                                         dim=-1)
        expected = expected[first - 1:len(ids) - 1]
        for long in (False, True):
            got, tops = scored_text(reference.StreamingReference(folder, "cpu"), ids, top_k, long=long)
            want = expected.gather(1, torch.tensor(ids[first:])[:, None])[:, 0]
            self.assertLess(float((got[first - 1:] - want).abs().max()), 1e-4, long)
            self.assertEqual(tops[first - 1:].tolist(), expected.topk(top_k, dim=-1).indices.tolist(), long)

    def test_gemma4_layer_kinds_score_like_the_model(self):
        # Per-layer-type RoPE, a sliding window shorter than the prompt, keys equal to values on full layers, a
        # scaled tied embedding, a stored layer scalar and final logit soft-capping.
        import torch
        from transformers import Gemma4TextConfig, Gemma4ForCausalLM
        torch.manual_seed(0)
        config = Gemma4TextConfig(vocab_size=128, hidden_size=64, intermediate_size=128, num_hidden_layers=4,
                                  num_attention_heads=4, num_key_value_heads=2, num_global_key_value_heads=1,
                                  head_dim=16, global_head_dim=32, sliding_window=8, max_position_embeddings=256,
                                  layer_types=["sliding_attention", "sliding_attention", "sliding_attention",
                                               "full_attention"], attention_k_eq_v=True,
                                  final_logit_softcapping=3.0, hidden_size_per_layer_input=0,
                                  tie_word_embeddings=True)
        model = Gemma4ForCausalLM(config)
        with torch.no_grad():
            for i, layer in enumerate(model.model.layers):
                layer.layer_scalar.fill_(0.5 + 0.25 * i)
        from crit_eval import reference
        ids = torch.randint(0, 128, (2 * reference.QUERY_BLOCK + 40,)).tolist()   # chunked blocks past the window
        with tempfile.TemporaryDirectory() as folder:
            self.assert_scores_like_the_model(model, folder, ids, reference.QUERY_BLOCK - 20)

    @unittest.skipUnless(HAS_REFERENCE and importlib.util.find_spec("transformers.models.qwen4_exp"),
                         "needs transformers with qwen4_exp")
    def test_converted_checkpoint_layouts_score_like_the_model(self):
        # A text model whose forward does its own work around the decoder layers (repeated residual streams, 3-D
        # rotary positions, per-layer n-gram inputs, a final mixer instead of a norm, a sparse indexer that needs a
        # mask), stored as published: under model.language_model., experts per expert (one layer NVFP4) where the
        # modeling code fuses them, an FP8 n-gram table in unequal shards with one scale for the whole table, and
        # vision and draft tensors the reference must not read.
        import torch
        from safetensors.torch import save_file
        from transformers import Qwen4ExpForCausalLM, Qwen4ExpTextConfig
        from crit_eval import reference, schemes
        torch.manual_seed(0)
        config = Qwen4ExpTextConfig(
            vocab_size=128, hidden_size=64, num_hidden_layers=4, num_attention_heads=4, num_key_value_heads=2,
            head_dim=16, linear_num_key_heads=2, linear_num_value_heads=4, linear_key_head_dim=16,
            linear_value_head_dim=16, moe_intermediate_size=32, shared_expert_intermediate_size=32, num_experts=4,
            num_experts_per_tok=2, hc_lowrank=8, ple_layer_ids=[2], ngram_vocab_size_base=50, split_ngram_parts=3,
            indexer_n_heads=2, indexer_kv_heads=1, indexer_head_dim=16, indexer_budget=8, indexer_compress_ratio=4,
            eos_token_id=1, max_position_embeddings=4096, tie_word_embeddings=False,
            rope_parameters={"rope_type": "default", "rope_theta": 10000.0, "partial_rotary_factor": 0.5,
                             "mrope_section": [2, 1, 1], "mrope_interleaved": True})
        config._attn_implementation = "eager"
        config._experts_implementation = "eager"
        model = Qwen4ExpForCausalLM(config).float().eval()
        tensors = {}
        with torch.no_grad():
            for name, param in model.named_parameters():
                param.add_(torch.randn_like(param) * 0.05)   # off the initialisation (zero norms, zero PLE conv)
            # Layer 0's experts in ModelOpt NVFP4, the model given exactly the values they decode to.
            experts = model.model.layers[0].mlp.experts
            for e in range(config.num_experts):
                width = config.moe_intermediate_size
                decoded = {}
                for proj, (rows, cols) in {"gate_proj": (width, 64), "up_proj": (width, 64),
                                           "down_proj": (64, width)}.items():
                    base = f"model.language_model.layers.0.mlp.experts.{e}.{proj}"
                    stored = {base + ".weight": torch.randint(0, 256, (rows, cols // 2), dtype=torch.uint8),
                              base + ".weight_scale": (torch.rand(rows, cols // 16) + .5).to(torch.float8_e4m3fn),
                              base + ".weight_scale_2": torch.tensor(0.01), base + ".input_scale": torch.tensor(1.0)}
                    view = object.__new__(reference.StateDictCheckpoint)
                    view.where, view.quantization = stored, {}
                    decoded[proj] = schemes.dense(view, base, "cpu")
                    tensors.update(stored)
                experts.gate_up_proj[e] = torch.cat([decoded["gate_proj"], decoded["up_proj"]])
                experts.down_proj[e] = decoded["down_proj"]
            # The n-gram table in FP8 E4M3 with one per-tensor scale.
            table = model.model.layers[1].ple.ple_embedding.ngram_embedding.weight
            scale = torch.tensor([2.0 ** -6], dtype=torch.bfloat16)
            codes = (table / scale.float()).to(torch.float8_e4m3fn)
            table.copy_(codes.float() * scale.float())
        ngram = "model.language_model.layers.1.ple.ple_embedding.ngram_embedding"
        shards = {f"{ngram}.shard_{k}.weight": part.contiguous() for k, part in enumerate(codes.tensor_split([7, 300]))}
        shards[f"{ngram}.weight_scale"] = scale
        for name, value in model.state_dict().items():
            if name.startswith("model.layers.0.mlp.experts.") or name == ngram.replace(".language_model", "") + ".weight":
                continue
            if name.endswith(("mlp.experts.gate_up_proj", "mlp.experts.down_proj")):
                layer = name.split(".")[2]
                for e in range(config.num_experts):
                    stem = f"model.language_model.layers.{layer}.mlp.experts.{e}."
                    if name.endswith("gate_up_proj"):
                        gate, up = value[e].chunk(2)
                        tensors[stem + "gate_proj.weight"], tensors[stem + "up_proj.weight"] = gate.clone(), up.clone()
                    else:
                        tensors[stem + "down_proj.weight"] = value[e].clone()
                continue
            tensors[name.replace("model.", "model.language_model.", 1)] = value.clone()
        tensors["mtp.layers.0.mlp.experts.0.gate_proj.weight"] = torch.full((32, 64), float("nan"))
        tensors["model.visual.blocks.0.attn.qkv.weight"] = torch.full((8, 8), float("nan"))

        def save(folder):
            folder = Path(folder)
            config.save_pretrained(folder)
            save_file(tensors, folder / "model-00001-of-00002.safetensors", metadata={"format": "pt"})
            save_file(shards, folder / "model-00002-of-00002.safetensors", metadata={"format": "pt"})
            weight_map = {**{k: "model-00001-of-00002.safetensors" for k in tensors},
                          **{k: "model-00002-of-00002.safetensors" for k in shards}}
            (folder / "model.safetensors.index.json").write_text(json.dumps({"metadata": {}, "weight_map": weight_map}))
        ids = torch.randint(0, 128, (2 * reference.QUERY_BLOCK + 40,)).tolist()
        ids[100] = ids[300] = config.eos_token_id   # n-gram segments restart after an end of sequence
        with tempfile.TemporaryDirectory() as folder:
            self.assert_scores_like_the_model(model, folder, ids, reference.QUERY_BLOCK - 20, save=save)
            # Only the n-gram rows a pass looks up are read (the table is never built), and nothing outside the
            # text model is.
            ref = reference.StreamingReference(folder, "cpu")
            raw = ref.checkpoint.raw

            def guarded(key):
                if ".shard_" in key or key.startswith(("mtp.", "model.visual.")):
                    raise AssertionError(f"read {key}")
                return raw(key)
            gathered, rows_many = [], ref.checkpoint.rows_many

            def gather(requests):
                gathered.extend(key for key, _ in requests)
                return rows_many(requests)
            ref.checkpoint.raw, ref.checkpoint.rows_many = guarded, gather
            scored_text(ref, ids[:40], 4)
            self.assertTrue(any(".shard_" in key for key in gathered))
            self.assertTrue(ref.skeleton.layers[1].ple.ple_embedding.ngram_embedding.weight.is_meta)


class ReferenceDtLimitTest(unittest.TestCase):
    """Mamba-2 mixers in the reference take config.json's time_step_limit, not transformers' (time_step_min, inf)."""

    def test_config_limit_replaces_the_mixer_clamp(self):
        from types import SimpleNamespace
        from crit_eval.reference import StreamingReference
        mixer = SimpleNamespace(time_step_limit=(0.001, float("inf")))
        attention = SimpleNamespace()
        layer = SimpleNamespace(modules=lambda: [layer, mixer, attention])
        owner = SimpleNamespace(config=SimpleNamespace(time_step_limit=[0.0, float("inf")]))
        StreamingReference._config_dt_limit(owner, layer)
        self.assertEqual(mixer.time_step_limit, (0.0, float("inf")))
        self.assertFalse(hasattr(attention, "time_step_limit"))
        mixer.time_step_limit = (0.001, float("inf"))
        StreamingReference._config_dt_limit(SimpleNamespace(config=SimpleNamespace()), layer)
        self.assertEqual(mixer.time_step_limit, (0.001, float("inf")), "no config limit: unchanged")


class HostLoadTest(unittest.TestCase):
    """Exact CPU accounting, independent of unrelated programs running on the test host."""

    def measure(self, host, process, wall):
        # Control the counter inputs, keeping HostLoad and its own-process reader real.
        with patch.object(hostload, "_host_busy_seconds", side_effect=host), \
                patch.object(hostload.os, "times", side_effect=[os.times_result(t) for t in process]), \
                patch.object(hostload.time, "perf_counter", side_effect=wall):
            return hostload.HostLoad().start().cores()

    def test_counts_other_processes_not_this_one(self):
        # os.times fields: own user/system, reaped children's user/system, elapsed.
        start = (10, 4, 7, 3, 1000)
        cases = [
            # Two CPU seconds of our own work over two wall seconds: no other load.
            ("own work", 102, (11.5, 4.5, 7, 3, 1002), 0.0),
            # A waited-for child's two CPU seconds must still count as other work.
            ("child work", 102, (10, 4, 8.5, 3.5, 1002), 1.0),
            # Two own, three child and four unrelated CPU seconds: 3.5 other cores.
            ("mixed work", 109, (11.5, 4.5, 9, 4, 1002), 3.5),
        ]
        # The old live test compared different windows and assumed background load
        # stayed constant. Each scenario now has its own exact accounting oracle.
        for name, host_end, process_end, expected in cases:
            with self.subTest(work=name):
                self.assertEqual(self.measure((100, host_end), (start, process_end), (20, 22)), expected)

    def test_normalizes_by_elapsed_wall_time(self):
        process = ((10, 4, 7, 3, 1000), (11.5, 4.5, 7, 3, 1004))
        for elapsed, expected in ((0.5, 8.0), (2, 2.0), (4, 1.0)):
            with self.subTest(elapsed=elapsed):
                self.assertEqual(self.measure((100, 106), process, (20, 20 + elapsed)), expected)

    def test_unavailable_host_totals_are_not_reported_as_zero(self):
        process = ((10, 4, 7, 3, 1000), (10, 4, 7, 3, 1002))
        for host in ((None, 102), (100, None), (None, None)):
            with self.subTest(host=host):
                self.assertIsNone(self.measure(host, process, (20, 22)))

    def test_nonpositive_wall_interval_is_unavailable(self):
        process = ((10, 4, 7, 3, 1000), (10, 4, 7, 3, 1002))
        for wall in ((20, 20), (20, 19)):
            with self.subTest(wall=wall):
                self.assertIsNone(self.measure((100, 102), process, wall))

    def test_counter_sampling_skew_cannot_report_negative_load(self):
        process = ((10, 4, 7, 3, 1000), (10.25, 4, 7, 3, 1002))
        self.assertEqual(self.measure((100, 100.125), process, (20, 22)), 0.0)


class HostLoadCountersTest(unittest.TestCase):
    """Exercise both platform readers on either OS, plus the real local counter APIs."""

    def test_windows_excludes_idle_and_converts_filetime(self):
        import ctypes

        def system_times(idle, kernel, user):
            # 1000 idle seconds, 1600 kernel seconds (including idle), 250.25 user
            # seconds. Totals exceed 32 bits; FILETIME ticks are 100 ns.
            for pointer, ticks in ((idle, 10_000_000_000), (kernel, 16_000_000_000), (user, 2_502_500_000)):
                ctypes.cast(pointer, ctypes.POINTER(ctypes.c_ulonglong)).contents.value = ticks
            return success

        for success, expected in ((1, 850.25), (0, None)):
            with self.subTest(api_success=success):
                windows = SimpleNamespace(kernel32=SimpleNamespace(GetSystemTimes=system_times))
                with patch.object(hostload, "os", SimpleNamespace(name="nt")), \
                        patch.object(ctypes, "windll", windows, create=True):
                    self.assertEqual(hostload._host_busy_seconds(), expected)

    def test_linux_excludes_idle_iowait_and_duplicate_guest_time(self):
        # Busy ticks: 1000 user + 200 nice + 300 system + 60 irq + 70 softirq
        # + 80 steal = 1710. Guest and guest_nice are already in user and nice.
        stat = "cpu 1000 200 300 4000 500 60 70 80 900 150\ncpu0 1 2 3 4 5 6 7 8 9 10\n"
        for ticks_per_second, expected in ((100, 17.1), (250, 6.84)):
            with self.subTest(ticks_per_second=ticks_per_second):
                platform = SimpleNamespace(name="posix", sysconf=Mock(return_value=ticks_per_second))
                with patch.object(hostload, "os", platform), \
                        patch.object(hostload, "open", mock_open(read_data=stat), create=True):
                    self.assertEqual(hostload._host_busy_seconds(), expected)
                platform.sysconf.assert_called_once_with("SC_CLK_TCK")

    def test_linux_accepts_totals_without_optional_fields(self):
        platform = SimpleNamespace(name="posix", sysconf=lambda name: 100)
        with patch.object(hostload, "os", platform), \
                patch.object(hostload, "open", mock_open(read_data="cpu 1000 200 300 4000\n"), create=True):
            self.assertEqual(hostload._host_busy_seconds(), 15.0)

    def test_linux_unreadable_or_invalid_totals_are_unavailable(self):
        platform = SimpleNamespace(name="posix", sysconf=lambda name: 100)
        for data in ("", "cpu0 1 2 3 4\n", "cpu 1 2\n", "cpu 1 two 3 4\n", OSError("unreadable")):
            with self.subTest(data=data):
                source = Mock(side_effect=data) if isinstance(data, OSError) else mock_open(read_data=data)
                with patch.object(hostload, "os", platform), \
                        patch.object(hostload, "open", source, create=True):
                    self.assertIsNone(hostload._host_busy_seconds())

    @unittest.skipUnless(os.name == "nt" or sys.platform.startswith("linux"), "Windows/Linux counter APIs")
    def test_native_counters_are_available_and_monotonic(self):
        # A supported platform losing its reader must fail, not silently skip the
        # accounting test. No assumption about how much background work occurs.
        for name, read in (("host", hostload._host_busy_seconds), ("own", hostload._own_seconds)):
            with self.subTest(counter=name):
                before, after = read(), read()
                self.assertIsNotNone(before)
                self.assertIsNotNone(after)
                self.assertTrue(math.isfinite(before) and math.isfinite(after))
                self.assertGreaterEqual(before, 0.0)
                self.assertGreaterEqual(after, before)


class HostTest(unittest.TestCase):
    """The result's host record: nvidia-smi, /proc/cpuinfo and every probe failing to None."""

    def smi(self, *answers):
        """Patch nvidia-smi to give answers in order: (stdout, returncode) or an exception."""
        def run(command, **kwargs):
            answer = replies.pop(0)
            if isinstance(answer, BaseException):
                raise answer
            return SimpleNamespace(stdout=answer[0], returncode=answer[1])
        replies = list(answers)
        return patch.object(host, "_nvidia_smi", return_value="nvidia-smi"), patch.object(host.subprocess, "run", run)

    def gpus(self, *answers):
        found, run = self.smi(*answers)
        with found, run:
            return host.gpus()

    def test_gpu_link_vram_and_driver(self):
        gpus = self.gpus(("0, NVIDIA GeForce RTX 5090, 32607, 596.49, 1, 4, 16, 16\n"
                          "1, Other, 24576, 596.49, [N/A], [Not Supported], 8, 16\n", 0), ("5, 4\n5, [N/A]\n", 0))
        self.assertEqual(gpus[0], {"index": 0, "name": "NVIDIA GeForce RTX 5090", "memory_total_bytes": 32607 * 1024 ** 2,
                                   "driver": "596.49",
                                   "pcie": {"gen_current": 1, "width_current": 16, "current_sampled": "idle",
                                            "gen_max": 4, "width_max": 16, "gen_gpu_max": 5, "gen_host_max": 4}})
        self.assertEqual(gpus[1]["pcie"], {"gen_current": None, "width_current": 8, "current_sampled": "idle",
                                           "gen_max": None, "width_max": 16, "gen_gpu_max": 5, "gen_host_max": None})

    def test_a_driver_without_side_maxima_keeps_the_link(self):
        for side in (("Field \"pcie.link.gen.gpumax\" is not a valid field", 2), FileNotFoundError(), ("5, 4\n5, 4\n", 0)):
            with self.subTest(side=side):
                gpus = self.gpus(("0, G, 1024, 1.0, 4, 4, 16, 16\n", 0), side)
                self.assertEqual((gpus[0]["pcie"]["gen_max"], gpus[0]["pcie"]["gen_gpu_max"]), (4, None))

    def test_nvidia_smi_failures_record_none(self):
        for answer in (("NVIDIA-SMI has failed", 9), OSError("denied"), host.subprocess.TimeoutExpired("nvidia-smi", 15),
                       ("", 0), ("0, short\n", 0)):
            with self.subTest(answer=answer):
                self.assertIsNone(self.gpus(answer))
        with patch.object(host, "_nvidia_smi", return_value=None):
            self.assertIsNone(host.gpus())

    def test_cpuinfo_counts_cores_not_threads(self):
        block = "processor\t: {0}\nmodel name\t: Ryzen\nphysical id\t: 0\ncore id\t\t: {1}\n\n"
        text = "".join(block.format(i, i % 6) for i in range(12))
        with patch("builtins.open", mock_open(read_data=text)):
            self.assertEqual(host._cpuinfo(), ("Ryzen", 6))
        with patch("builtins.open", mock_open(read_data="processor\t: 0\nmodel name\t: Arm\n")):
            self.assertEqual(host._cpuinfo(), ("Arm", None))

    def test_a_failing_probe_records_none_and_never_raises(self):
        with patch.object(host, "cpu", side_effect=RuntimeError("probe")), \
                patch.object(host, "gpus", side_effect=ValueError("probe")), \
                patch.object(host, "host_memory", side_effect=OSError("probe")):
            described = host.describe()
        self.assertEqual((described["cpu"], described["gpus"], described["memory_total_bytes"]), (None, None, None))
        self.assertIsInstance(described["platform"], str)
        with patch("builtins.open", side_effect=OSError("no proc")), \
                patch.object(host, "_windows_cpu_model", side_effect=OSError("no registry")):
            self.assertEqual(host.cpu()["logical_cores"], os.cpu_count())

    def test_summary_names_the_link_the_host_allows(self):
        record = {"platform": "Windows-11", "os": {"system": "Windows", "release": "11", "wsl": False},
                  "cpu": {"model": "Ryzen", "physical_cores": 6, "logical_cores": 12}, "memory_total_bytes": 96 * 1024 ** 3,
                  "gpus": [{"name": "RTX", "memory_total_bytes": 32 * 1024 ** 3,
                            "pcie": {"gen_max": 4, "width_max": 16, "gen_gpu_max": 5}}]}
        self.assertEqual(host.summary(record),
                         "Windows 11; Ryzen (6/12 cores); 96 GiB RAM; RTX 32 GiB, PCIe 4.0 x16 (GPU supports 5.0)")
        record["os"] = {"system": "Linux", "distribution": "Ubuntu", "wsl": True}
        self.assertTrue(host.summary(record).startswith("Ubuntu (WSL); "))
        self.assertEqual(host.summary({"platform": "Linux-6.1"}), "Linux-6.1")   # a result from before the probe

    def test_this_machine_describes_itself(self):
        described = host.describe()
        self.assertEqual(set(described), {"platform", "python", "os", "cpu", "memory_total_bytes", "gpus"})
        self.assertEqual(described["cpu"]["logical_cores"], os.cpu_count())
        json.dumps(described)


if __name__ == "__main__":
    unittest.main()
