"""Model recon: facts gathered from tiny fixture model folders and frozen into candidates; no network, no GPU."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
for _folder in ("core/python", "api", "../crit-eval", "../crit-eval/tests", "kernels", "cli", "tests"): sys.path.insert(0, str(ROOT / _folder))
from core_text_fixtures import llama_files, write_files
from crit.run import recon
from crit.run.cli import main
from crit.run.records import file_sha, read_json
from crit.run.store import RunStore

THINKING = """{%- for message in messages %}<|im_start|>{{ message.role }}
{{ message.content }}<|im_end|>
{% endfor %}{%- if add_generation_prompt %}<|im_start|>assistant
{%- if enable_thinking is defined and not enable_thinking %}<think>

</think>{% endif %}{% endif %}"""


class ReconTests(unittest.TestCase):
    def setUp(self):
        # A run as created before plans: recon freezing does not depend on them.
        plans = patch("crit.run.records.NEW_RUN_PLANS", None)
        plans.start()
        self.addCleanup(plans.stop)
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name).resolve()
        self.model = write_files(self.root / "model", llama_files(**{
            "README.md": "# Fixture model\nUse temperature 0.7 without thinking.\n",
            "config.json": {"architectures": ["FixtureForCausalLM"], "model_type": "fixture", "eos_token_id": 1,
                            "text_config": {"eos_token_id": 4, "vocab_size": 420, "num_hidden_layers": 2,
                                            "rope_scaling": {"type": "yarn", "factor": 4.0}}}}))
        self.store = RunStore(self.root / "runs")
        self.run_id = self.store.create({"model_path": str(self.model), "correctness_criteria": "Fixture gates",
                                         "budget_minutes": 1, "kv_tokens": 8192})["run"]["run_id"]
        self.directory = self.store.directory(self.run_id)

    def candidate(self, configuration):
        captured = self.directory / "captured/input.fixture"
        if not (self.directory / "inputs.json").exists():
            captured.write_text("fixed workload", encoding="utf-8")
            self.store.freeze_inputs(self.run_id, {"hardware_manifest": {"device": "host"},
                "environment_manifest": {"benchmark": "synthetic"}, "case_manifests": {"case-0001": [
                    {"path": "captured/input.fixture", "bytes": captured.stat().st_size, "sha256": file_sha(captured)}]}})
        source = self.root / "source"
        source.mkdir(exist_ok=True)
        (source / "engine.fixture").write_text("source", encoding="utf-8")
        return self.store.candidate(self.run_id, source, {"label": "Fixture", "rationale": "Recon freezing",
            "strategy": "Fixture", "configuration": configuration, "store_decisions": {}})

    def test_gathers_sampling_stops_template_and_architecture_with_sources(self):
        value = self.store.recon(self.run_id)
        gathered, frozen = value["gathered"], value["frozen"]
        self.assertEqual(gathered["generation"]["sampling"], {"do_sample": True, "temperature": 0.6, "top_p": 0.9})
        self.assertEqual(frozen["sampling"], {"do_sample": True, "temperature": 0.6, "top_p": 0.9})
        self.assertEqual(gathered["eos_token_ids"], [1, 4])
        self.assertEqual(frozen["stop_token_ids"], gathered["stop_token_ids"])
        self.assertIn(4, frozen["stop_token_ids"])
        self.assertEqual(gathered["stop_token_text"]["4"], "<|eot_id|>")
        self.assertEqual(frozen["chat_template"]["name"], "default")
        self.assertEqual(frozen["template_variables"], {})   # a template without a thinking switch
        self.assertEqual(gathered["architecture"]["architectures"], ["FixtureForCausalLM"])
        self.assertEqual(gathered["architecture"]["rope_scaling"], {"type": "yarn", "factor": 4.0})
        self.assertEqual(gathered["architecture"]["num_hidden_layers"], 2)
        self.assertEqual(gathered["model_card"], "README.md")
        sources = {row["path"]: row for row in value["sources"]}
        for name in ("generation_config.json", "tokenizer_config.json", "tokenizer.json", "chat_template.jinja",
                     "config.json", "README.md"):
            self.assertEqual(sources[name]["sha256"], file_sha(self.model / name), name)
        notes = (self.directory / recon.RECON_MD).read_text(encoding="utf-8")
        for heading in ("## Sources", "## Model card", "## Reference implementation", "## Decisions"):
            self.assertIn(heading, notes)
        self.assertIn("<|eot_id|>", notes)

    def test_companion_drafters_are_recorded_with_their_taps(self):
        eagle = self.root / "eagle"
        eagle.mkdir()
        (eagle / "config.json").write_text(json.dumps({
            "speculators_model_type": "eagle3", "eagle_aux_hidden_state_layer_ids": [0, 1, 2], "draft_vocab_size": 64,
            "transformer_layer_config": {"num_hidden_layers": 1},
            "speculators_config": {"proposal_methods": [{"speculative_tokens": 3}], "verifier": {"name_or_path": "fixture"}}}),
            encoding="utf-8")
        mtp = self.root / "mtp"
        mtp.mkdir()
        (mtp / "config.json").write_text(json.dumps({"num_hidden_layers": 1, "crit_drafter": {
            "format": "crit-drafter-v1", "kind": "mtp-head", "target_layers": 2, "target": {"path": str(self.model)}}}),
            encoding="utf-8")
        value = self.store.recon(self.run_id, drafters=[eagle, mtp])
        first, second = value["gathered"]["drafters"]
        self.assertEqual((first["kind"], first["taps"], first["draft_vocab_size"], first["speculative_tokens"]),
                         ("eagle3", [0, 1, 2], 64, 3))
        self.assertEqual(first["config_sha256"], file_sha(eagle / "config.json"))
        self.assertEqual((second["kind"], second["taps"], second["drafts_per_pass"]), ("mtp-head", [2], 1))
        notes = (self.directory / recon.RECON_MD).read_text(encoding="utf-8")
        self.assertIn(str(eagle), notes)
        self.assertIn("map through d2t", notes)
        deep = self.root / "deep"
        deep.mkdir()
        (deep / "config.json").write_text(json.dumps({"speculators_model_type": "eagle3",
                                                      "eagle_aux_hidden_state_layer_ids": [2, 9]}), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "reach past"):
            self.store.recon(self.run_id, force=True, drafters=[deep])
        with self.assertRaisesRegex(ValueError, "not a hidden-state drafter"):
            self.store.recon(self.run_id, force=True, drafters=[self.model])

    def test_template_switches_are_listed_and_rerun_needs_force_but_keeps_notes(self):
        (self.model / "chat_template.jinja").write_text(THINKING, encoding="utf-8")
        value = self.store.recon(self.run_id)
        self.assertEqual(value["gathered"]["chat_templates"]["default"]["variables"], ["enable_thinking"])
        # Served with thinking off by default, as crit Eval renders every prompt; requests turn it on.
        self.assertEqual(value["frozen"]["template_variables"], {"enable_thinking": False})
        self.assertIn("Thinking is off by default", (self.directory / recon.RECON_MD).read_text(encoding="utf-8"))
        notes = self.directory / recon.RECON_MD
        notes.write_text("my findings", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "--force"):
            self.store.recon(self.run_id)
        self.store.recon(self.run_id, force=True)
        self.assertEqual(notes.read_text(encoding="utf-8"), "my findings")

    def test_inputs_are_refused_when_a_case_renders_otherwise_than_candidates_serve_it(self):
        from crit_core.model import open_model
        from crit_core.text import load_model_text
        (self.model / "chat_template.jinja").write_text(THINKING, encoding="utf-8")
        messages = [{"role": "user", "content": "Summarize the maintenance log in three lines."}]
        text = load_model_text(open_model(self.model).files, clock=None)
        thinking = len(text.encode_chat(messages, add_generation_prompt=True, enable_thinking=True))
        self.assertNotEqual(thinking, len(text.encode_chat(messages, add_generation_prompt=True, enable_thinking=False)))
        workload = {"prompt_tokens": thinking, "output_tokens": 16, "concurrency": 1, "repetitions": 1, "warmup_runs": 0,
                    "mode": "resident"}
        run_id = self.store.create({"model_path": str(self.model), "correctness_criteria": "Fixture gates",
                                    "budget_minutes": 1, "kv_tokens": 8192, "workload": workload})["run"]["run_id"]
        directory = self.store.directory(run_id)
        self.store.recon(run_id)   # served with thinking off
        captured = directory / "captured/case-0001/request.json"
        captured.parent.mkdir(parents=True)

        def freeze(request):
            captured.write_text(json.dumps(request), encoding="utf-8")
            return self.store.freeze_inputs(run_id, {"hardware_manifest": {"device": "host"},
                "environment_manifest": {"benchmark": "synthetic"}, "case_manifests": {"case-0001": [
                    {"path": "captured/case-0001/request.json", "bytes": captured.stat().st_size,
                     "sha256": file_sha(captured)}]}})
        with self.assertRaisesRegex(ValueError, rf"case-0001: its captured messages render to \d+ prompt tokens with "
                                                f"template variables .*enable_thinking.*, the workload has {thinking}"):
            freeze({"messages": messages})
        self.assertFalse((directory / "inputs.json").exists())
        self.assertIn("case_manifests", freeze({"messages": messages, "chat_template_kwargs": {"enable_thinking": True}}))

    def test_candidates_freeze_the_recon_block(self):
        self.store.recon(self.run_id)
        path = self.directory / recon.RECON_JSON
        value = read_json(path)
        value["frozen"]["template_variables"] = {"enable_thinking": False}
        path.write_text(json.dumps(value), encoding="utf-8")
        candidate = self.candidate({"batch": 1})
        self.assertEqual(candidate["configuration"]["batch"], 1)
        self.assertEqual(candidate["configuration"]["recon"], value["frozen"])
        self.assertEqual(candidate["extensions"]["recon"], {"path": recon.RECON_JSON, "sha256": file_sha(path),
                                                            "fingerprint_sha256": value["gathered"]["fingerprint"]["sha256"]})
        with self.assertRaisesRegex(ValueError, "differs"):
            self.candidate({"recon": {**value["frozen"], "sampling": {}}})
        self.assertEqual(self.candidate({"recon": value["frozen"]})["configuration"]["recon"], value["frozen"])

    def test_the_sampling_is_the_models_and_recon_says_the_agent_cannot_change_it(self):
        value = self.store.recon(self.run_id)
        self.assertEqual(value["gathered"]["eval_sampling"],
                         {"settings": {"temperature": 0.6, "top_p": 0.9}, "source": "generation_config"})
        notes = (self.directory / recon.RECON_MD).read_text(encoding="utf-8")
        self.assertIn("## Eval sampling", notes)
        self.assertIn("run at `{'temperature': 0.6, 'top_p': 0.9}`, from the model's generation_config.json. "
                      "You cannot change them", notes)
        self.assertIn("## Drafting plan for both decode modes", notes)
        self.assertIn("## Required context\n\nEvery candidate must serve 8,192 tokens of prompt plus output, given "
                      "explicitly as kv_tokens at run create.", notes)
        self.assertIn("You cannot change it: only the user does", notes)
        self.assertIn("Plan the KV cache", notes)
        self.assertIn("token-exact against plain greedy", notes)
        self.assertIn("rejection sampling", notes)
        path = self.directory / recon.RECON_JSON
        edited = read_json(path)
        edited["frozen"]["sampling"] = {"do_sample": True, "temperature": 0.3, "top_p": 0.9}
        path.write_text(json.dumps(edited), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "not the engine builder's to change"):
            self.candidate({"batch": 1})

    def test_a_users_eval_sampling_override_is_recorded_served_and_required_of_the_data(self):
        from crit.run import evaluation
        run = self.store.create({"model_path": str(self.model), "correctness_criteria": "Fixture gates",
                                 "budget_minutes": 1, "kv_tokens": 8192,
                                 "eval_sampling": {"settings": "temperature=0.3,top_k=20", "approved_by": "the user"}})["run"]
        self.assertEqual(run["spec"]["eval_sampling"], {"settings": {"temperature": 0.3, "top_k": 20}, "approved_by": "the user"})
        value = self.store.recon(run["run_id"])
        self.assertEqual(value["gathered"]["eval_sampling"]["source"], "override")
        self.assertEqual(value["frozen"]["sampling"], {"do_sample": True, "temperature": 0.3, "top_k": 20})
        self.assertIn("the user's eval_sampling override (approved by the user)",
                      (self.store.directory(run["run_id"]) / recon.RECON_MD).read_text(encoding="utf-8"))
        override = {"sampling": {"temperature": 0.3, "top_k": 20}, "sampling_source": "override"}
        own = {"sampling": {"temperature": 0.6, "top_p": 0.9}, "sampling_source": "generation_config"}
        evaluation.check_sampling(run, override, "DATA", "MODEL")
        with self.assertRaisesRegex(ValueError, "--sampling temperature=0.3,top_k=20"):
            evaluation.check_sampling(run, own, "DATA", "MODEL")
        plain = self.store.get(self.run_id)["run"]
        evaluation.check_sampling(plain, own, "DATA", "MODEL")
        with self.assertRaisesRegex(ValueError, "only the user"):
            evaluation.check_sampling(plain, override, "DATA", "MODEL")
        for bad in ({"settings": "temperature=0.3"}, {"settings": "top_k=20", "approved_by": "the user"}):
            with self.assertRaises(ValueError):
                self.store.create({"model_path": str(self.model), "correctness_criteria": "Fixture gates",
                                   "budget_minutes": 1, "kv_tokens": 8192, "eval_sampling": bad})

    def test_model_folder_requires_recon_and_frozen_block_is_validated(self):
        with self.assertRaisesRegex(ValueError, "Run recon before the first candidate"):
            self.candidate({"batch": 1})
        self.assertEqual(list((self.directory / "candidates").iterdir()), [])
        self.store.recon(self.run_id)
        path = self.directory / recon.RECON_JSON
        value = read_json(path)
        value["frozen"]["stop_token_ids"] = ["<|eot_id|>"]
        path.write_text(json.dumps(value), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "stop_token_ids"):
            self.candidate({"batch": 1})

    def test_a_lone_gguf_gathers_the_files_its_metadata_derives(self):
        from test_eval_gguf import gguf_folder
        (self.root / "gguf").mkdir()
        model = gguf_folder(self.root / "gguf", config=None) / "model-IQ3_S.gguf"
        rows, gathered, frozen = recon.gather(model)
        self.assertEqual(gathered["derived_files"], ["chat_template.jinja", "config.json", "generation_config.json",
                                                     "tokenizer.json", "tokenizer_config.json"])
        self.assertEqual((gathered["architecture"]["model_type"], gathered["eos_token_ids"]), ("qwen3_5_text", [2, 0]))
        self.assertEqual((frozen["chat_template"]["name"], frozen["stop_token_ids"]), ("default", [2, 0]))
        self.assertIn("came from the weights' own metadata", recon.scaffold(
            {"model_path": str(model), "created_at": "now", "sources": rows, "gathered": gathered, "frozen": frozen}))

    def test_cli_and_single_file_models(self):
        self.assertEqual(main(["--project", str(ROOT), "--root", str(self.root / "runs"), "--id", self.run_id, "recon"]), 0)
        self.assertTrue((self.directory / recon.RECON_JSON).is_file())
        single = self.root / "model.bin"
        single.write_bytes(b"not a model folder")
        with self.assertRaisesRegex(ValueError, "description files; .* has none"):
            recon.gather(single)
        # A single-file model needs no recon; its candidates are unchanged.
        store = RunStore(self.root / "single-runs")
        run_id = store.create({"model_path": str(single), "correctness_criteria": "Fixture gates",
                               "budget_minutes": 1, "kv_tokens": 8192})["run"]["run_id"]
        self.store, self.run_id, self.directory = store, run_id, store.directory(run_id)
        self.assertEqual(self.candidate({"batch": 1})["configuration"], {"batch": 1})


if __name__ == "__main__":
    unittest.main()


def header_only_safetensors(path, tensors):
    """A safetensors file with real headers and zeroed data for {name: (dtype, shape)}."""
    import struct
    width = {"BF16": 2, "F32": 4, "F8_E4M3": 1, "U8": 1}
    header, offset = {}, 0
    for name, (dtype, shape) in tensors.items():
        size = width[dtype]
        for dim in shape:
            size *= dim
        header[name] = {"dtype": dtype, "shape": list(shape), "data_offsets": [offset, offset + size]}
        offset += size
    data = json.dumps(header).encode()
    data += b" " * (-len(data) % 8)
    path.write_bytes(struct.pack("<Q", len(data)) + data + bytes(offset))


class FingerprintTests(unittest.TestCase):
    """Two checkpoints compare by what an engine is written against: architecture, weight layout, tokenizer, template."""

    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name).resolve()
        self.store = RunStore(self.root / "runs")

    def model(self, name, layers=2, dtype="BF16", **extra):
        folder = write_files(self.root / name, llama_files(**{
            "config.json": {"architectures": ["FixtureForCausalLM"], "model_type": "fixture", "hidden_size": 8,
                            "num_hidden_layers": layers, "torch_dtype": "bfloat16", "eos_token_id": 1,
                            "text_config": {"eos_token_id": 4, "vocab_size": 420}}, **extra}))
        tensors = {"model.embed_tokens.weight": ("BF16", (420, 8)), "lm_head.weight": ("BF16", (420, 8))}
        for index in range(layers):
            tensors[f"model.layers.{index}.mlp.down_proj.weight"] = (dtype if index == layers - 1 else "BF16", (8, 16))
        header_only_safetensors(folder / "model.safetensors", tensors)
        return folder

    def test_fingerprint_sections_and_the_weight_layout(self):
        _, gathered, _ = recon.gather(self.model("a"))
        print_ = gathered["fingerprint"]
        self.assertEqual(print_["version"], recon.FINGERPRINT_VERSION)
        self.assertEqual(sorted(print_["hashes"]), sorted(recon.FINGERPRINT_SECTIONS))
        self.assertEqual(print_["architecture"]["num_hidden_layers"], 2)
        self.assertNotIn("torch_dtype", print_["architecture"])
        self.assertEqual(print_["weights"]["format"], "safetensors")
        self.assertEqual(print_["weights"]["dtype"], "bfloat16")
        self.assertEqual(print_["weights"]["layout"]["model.layers.*.mlp.down_proj.weight"],
                         [{"dtype": "BF16", "shape": [8, 16], "count": 2}])
        self.assertIn("tokenizer.json", print_["tokenizer"]["files"])
        self.assertEqual(print_["tokenizer"]["eos_token_ids"], [1, 4])
        self.assertEqual(print_["template"], {"default": gathered["chat_templates"]["default"]["sha256"]})
        _, again, _ = recon.gather(self.model("a-copy"))
        self.assertEqual(again["fingerprint"]["sha256"], print_["sha256"], "the fingerprint is the checkpoint's shape, not its path")

    def test_compare_names_what_changes_for_the_engine(self):
        a = self.model("a")
        same = recon.compare_models(a, self.model("finetune"))["comparison"]
        self.assertTrue(same["same"])
        self.assertEqual(same["differences"], [])
        self.assertIn("weight values only", same["assessment"][0])
        layers = recon.compare_models(a, self.model("deeper", layers=3))["comparison"]
        self.assertFalse(layers["same"])
        self.assertEqual(layers["sections"]["architecture"], "differs")
        self.assertIn({"section": "architecture", "key": "num_hidden_layers", "a": 2, "b": 3}, layers["differences"])
        self.assertEqual(layers["sections"]["weights"], "differs", "a third layer changes the layout's counts")
        self.assertTrue(any(note.startswith("Architecture differs in num_hidden_layers") for note in layers["assessment"]))
        quant = recon.compare_models(a, self.model("fp8", dtype="F8_E4M3"))["comparison"]
        self.assertEqual(quant["sections"], {"architecture": "same", "weights": "differs", "tokenizer": "same", "template": "same"})
        self.assertTrue(any("1 tensor patterns differ" in note and "down_proj" in note for note in quant["assessment"]))
        other_template = self.model("template", **{"chat_template.jinja": THINKING})
        template = recon.compare_models(a, other_template)["comparison"]
        self.assertEqual(template["sections"]["template"], "differs")
        self.assertTrue(any(note.startswith("Chat template differs") for note in template["assessment"]))
        (other_template / "tokenizer_config.json").write_text(json.dumps({"bos_token": "<s>", "added_tokens_decoder": {}}), encoding="utf-8")
        tokenizer = recon.compare_models(a, other_template)["comparison"]
        self.assertEqual(tokenizer["sections"]["tokenizer"], "differs")
        self.assertIn("files.tokenizer_config.json", {row["key"] for row in tokenizer["differences"]})

    def test_a_run_compares_its_model_through_the_cli(self):
        a = self.model("a")
        run_id = self.store.create({"model_path": str(a), "correctness_criteria": "Fixture gates", "budget_minutes": 1,
                                    "kv_tokens": 8192})["run"]["run_id"]
        with self.assertRaisesRegex(ValueError, "Run recon first"):
            self.store.recon_compare(run_id, self.model("b"))
        self.store.recon(run_id)
        notes = (self.store.directory(run_id) / recon.RECON_MD).read_text(encoding="utf-8")
        self.assertIn("## Fingerprint", notes)
        result = self.store.recon_compare(run_id, self.model("b", layers=1))
        self.assertEqual(result["run_model"]["model"], str(a))
        self.assertFalse(result["comparison"]["same"])
        code = main(["--project", str(ROOT), "--root", str(self.store.root), "--id", run_id, "recon", "--compare", str(self.root / "b")])
        self.assertEqual(code, 0)
