"""Model recon: facts gathered from tiny fixture model folders and frozen into candidates; no network, no GPU."""
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
for _folder in ("core/python", "api", "eval", "kernels", "cli", "tests"): sys.path.insert(0, str(ROOT / _folder))
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

    def test_template_switches_are_listed_and_rerun_needs_force_but_keeps_notes(self):
        (self.model / "chat_template.jinja").write_text(THINKING, encoding="utf-8")
        value = self.store.recon(self.run_id)
        self.assertEqual(value["gathered"]["chat_templates"]["default"]["variables"], ["enable_thinking"])
        notes = self.directory / recon.RECON_MD
        notes.write_text("my findings", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "--force"):
            self.store.recon(self.run_id)
        self.store.recon(self.run_id, force=True)
        self.assertEqual(notes.read_text(encoding="utf-8"), "my findings")

    def test_candidates_freeze_the_recon_block(self):
        self.store.recon(self.run_id)
        path = self.directory / recon.RECON_JSON
        value = read_json(path)
        value["frozen"]["template_variables"] = {"enable_thinking": False}
        path.write_text(json.dumps(value), encoding="utf-8")
        candidate = self.candidate({"batch": 1})
        self.assertEqual(candidate["configuration"]["batch"], 1)
        self.assertEqual(candidate["configuration"]["recon"], value["frozen"])
        self.assertEqual(candidate["extensions"]["recon"], {"path": recon.RECON_JSON, "sha256": file_sha(path)})
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
