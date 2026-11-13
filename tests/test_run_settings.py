"""Machine settings from PROJECT/settings.toml for crit run."""
import contextlib
import io
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
for _folder in ("core/python", "api", "../crit-eval", "kernels", "cli"): sys.path.insert(0, str(ROOT / _folder))
from crit import cli
from crit.cli import PROJECT_VARIABLE
from crit.pull.bench import main as bench_main
from crit.run.cli import load_settings, main
from crit.run.records import GIB


class RunSettingsTest(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.project = Path(temp.name).resolve()
        marker = self.project / "core/include/crit/core.h"
        marker.parent.mkdir(parents=True)
        marker.write_text("// project marker\n", encoding="utf-8")
        model = self.project / "model.fixture"
        model.write_text("synthetic input, not a model", encoding="utf-8")
        self.request = self.project / "request.json"
        self.request.write_text(json.dumps({"model_path": str(model), "correctness_criteria": "Fixed fixture gates",
                                            "vram_headroom_gib": 3, "kv_tokens": 8192}),
                                encoding="utf-8")

    def create(self):
        out = io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(io.StringIO()):
            code = main(["--project", str(self.project), "create", "--record", str(self.request)])
        return code, out.getvalue()

    def limits(self, output):
        return json.loads(output)["run"]["spec"]["hardware_constraints"]

    def test_missing_file_keeps_defaults(self):
        self.assertEqual(load_settings(self.project), {})
        code, output = self.create()
        self.assertEqual(code, 0)
        self.assertEqual(self.limits(output)["ram_headroom_bytes"], 2 * GIB)

    def test_settings_fill_omitted_headroom_and_request_wins(self):
        (self.project / "settings.toml").write_text('# This machine\nwsl_run_root = "/srv/crit/runs"\n'
                                                    'ram_headroom_gib = 8\nvram_headroom_gib = 0.5\n', encoding="utf-8")
        code, output = self.create()
        self.assertEqual(code, 0)
        limits = self.limits(output)
        self.assertEqual(limits["ram_headroom_bytes"], 8 * GIB)
        self.assertEqual(limits["vram_headroom_bytes"], 3 * GIB)

    def test_create_says_when_eval_cannot_score_the_model(self):
        for refusal, says in (("GGUF architecture 'x' has no layout rules", True), (None, False)):
            err, out = io.StringIO(), io.StringIO()
            with patch("crit_eval.quality.checkpoints.scoring_refusal", return_value=refusal) as check,                     contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
                code = main(["--project", str(self.project), "create", "--record", str(self.request)])
            self.assertEqual(code, 0)   # the run is created either way
            self.assertEqual(check.call_args.args, (str(self.project / "model.fixture"),))
            self.assertEqual("Eval cannot score this model: GGUF architecture 'x' has no layout rules" in err.getvalue(),
                             says)

    def test_settings_quality_mode_fills_the_request_and_names_a_mode_crit_knows(self):
        for text, expected in (("", "speed"), ('quality_mode = "preserve"\n', "preserve")):
            (self.project / "settings.toml").write_text(text, encoding="utf-8")
            code, output = self.create()
            self.assertEqual(code, 0)
            self.assertEqual(json.loads(output)["run"]["spec"]["correctness"]["quality_mode"], expected)
        # A mode no longer named is refused like any unknown one.
        (self.project / "settings.toml").write_text('quality_mode = "as_quantized"\n', encoding="utf-8")
        err = io.StringIO()
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(err):
            code = main(["--project", str(self.project), "create", "--record", str(self.request)])
        self.assertNotEqual(code, 0)
        self.assertIn("quality_mode requires preserve or speed", err.getvalue())

    def test_the_example_file_lists_every_setting_with_the_codes_defaults(self):
        import re
        import shutil
        import crit_eval
        from crit.run.cli import SETTINGS
        example = ROOT / "settings.example.toml"
        shutil.copy(example, self.project / "settings.toml")
        active = load_settings(self.project)  # parses, with known keys of the right types
        text = example.read_text(encoding="utf-8")
        commented = set(re.findall(r"^# ([a-z_]+) = ", text, flags=re.MULTILINE))
        self.assertEqual(set(active) | commented, set(SETTINGS))
        self.assertFalse(set(active) & commented)
        from crit_eval import EVAL_DIR
        protocol = json.loads((EVAL_DIR / "protocol.json").read_text(encoding="utf-8"))
        self.assertEqual(active, {"ram_headroom_gib": 2, "vram_headroom_gib": 1,
                                  "context_reserve_gib": protocol["context"]["reserve_gib"],
                                  "context_tokens": crit_eval.DEFAULT_CONTEXT_TOKENS})
        # crit eval reads a subset of the same keys.
        self.assertLessEqual({"vram_headroom_gib", "context_reserve_gib", "context_tokens"}, set(SETTINGS))
        self.assertIn("!/settings.example.toml", (ROOT / ".gitignore").read_text(encoding="utf-8").splitlines())

    def test_settings_device_sizes_the_run_for_eval_max_context(self):
        model = self.project / "model"
        model.mkdir()
        header = json.dumps({"lm_head.weight": {"dtype": "BF16", "shape": [64, 64], "data_offsets": [0, 8192]}}).encode()
        (model / "model.safetensors").write_bytes(len(header).to_bytes(8, "little") + header + bytes(8192))
        (model / "config.json").write_text(json.dumps({"num_hidden_layers": 2, "num_attention_heads": 2, "num_key_value_heads": 1,
                                                       "head_dim": 8, "max_position_embeddings": 65536}), encoding="utf-8")
        (self.project / "device.json").write_text(json.dumps({"device": "fixture gpu", "memory_free_bytes": 8 * GIB}),
                                                  encoding="utf-8")
        (self.project / "settings.toml").write_text('device = "device.json"\n', encoding="utf-8")
        request = {key: value for key, value in json.loads(self.request.read_text(encoding="utf-8")).items() if key != "kv_tokens"}
        self.request.write_text(json.dumps(request | {"model_path": str(model)}), encoding="utf-8")
        code, output = self.create()
        self.assertEqual(code, 0)
        limits = self.limits(output)
        self.assertEqual((limits["kv_tokens"], limits["context"]["bound"]), (65536, "model"))

    def test_a_companion_drafter_is_kept_out_of_the_kv_cache(self):
        model, drafter = self.project / "model", self.project / "drafter"
        for folder, name, size in ((model, "lm_head.weight", 8192), (drafter, "layers.0.fc.weight", 2**20)):
            folder.mkdir()
            header = json.dumps({name: {"dtype": "BF16", "shape": [size // 2], "data_offsets": [0, size]}}).encode()
            (folder / "model.safetensors").write_bytes(len(header).to_bytes(8, "little") + header + bytes(size))
        (model / "config.json").write_text(json.dumps({"num_hidden_layers": 2, "num_attention_heads": 2, "num_key_value_heads": 1,
                                                       "head_dim": 8, "max_position_embeddings": 1_048_576}), encoding="utf-8")
        # 4 MiB for a 32-byte-per-token cache after a 1 GiB headroom and a 3 GiB reserve: 131,072 tokens
        free = 4 * GIB + 4 * 2**20 + 8192
        (self.project / "device.json").write_text(json.dumps({"device": "fixture gpu", "memory_free_bytes": free}),
                                                  encoding="utf-8")
        (self.project / "settings.toml").write_text('device = "device.json"\n', encoding="utf-8")
        request = {key: value for key, value in json.loads(self.request.read_text(encoding="utf-8")).items() if key != "kv_tokens"}
        found = {}
        for drafters in ([], [str(drafter)]):
            self.request.write_text(json.dumps(request | {"model_path": str(model), "drafters": drafters, "vram_headroom_gib": 1,
                                                       "context_reserve_gib": 3}), encoding="utf-8")
            code, output = self.create()
            self.assertEqual(code, 0)
            found[len(drafters)] = self.limits(output)
        self.assertEqual(found[0]["context"]["bound"], "device")
        self.assertEqual((found[0]["kv_tokens"], found[1]["kv_tokens"]), (131072, 98304))
        self.assertEqual(found[1]["context"]["companion_drafter_bytes"], 2**20)
        self.assertNotIn("companion_drafter_bytes", found[0]["context"])

    def test_the_wanted_context_defaults_to_200000_and_settings_context_tokens_changes_it(self):
        model = self.project / "model"
        model.mkdir()
        header = json.dumps({"lm_head.weight": {"dtype": "BF16", "shape": [64, 64], "data_offsets": [0, 8192]}}).encode()
        (model / "model.safetensors").write_bytes(len(header).to_bytes(8, "little") + header + bytes(8192))
        (model / "config.json").write_text(json.dumps({"num_hidden_layers": 2, "num_attention_heads": 2, "num_key_value_heads": 1,
                                                       "head_dim": 8, "max_position_embeddings": 1_048_576}), encoding="utf-8")
        (self.project / "device.json").write_text(json.dumps({"device": "fixture gpu", "memory_free_bytes": 8 * GIB}),
                                                  encoding="utf-8")
        request = {key: value for key, value in json.loads(self.request.read_text(encoding="utf-8")).items() if key != "kv_tokens"}
        self.request.write_text(json.dumps(request | {"model_path": str(model)}), encoding="utf-8")
        for text, tokens in (('device = "device.json"\n', 200000), ('device = "device.json"\ncontext_tokens = 32768\n', 32768)):
            (self.project / "settings.toml").write_text(text, encoding="utf-8")
            code, output = self.create()
            self.assertEqual(code, 0)
            limits = self.limits(output)
            self.assertEqual((limits["kv_tokens"], limits["context"]["bound"], limits["context"]["requested"]),
                             (tokens, "requested", tokens))
        (self.project / "settings.toml").write_text('device = "device.json"\ncontext_tokens = 2000000\n', encoding="utf-8")
        code, output = self.create()
        context = self.limits(output)["context"]
        self.assertEqual((self.limits(output)["kv_tokens"], context["bound"]), (1_048_576, "model"))
        self.assertIn("the requested 2,000,000 tokens exceed the model's 1,048,576 positions", context["note"])
        (self.project / "settings.toml").write_text('context_tokens = "big"\n', encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "context_tokens has the wrong type"):
            load_settings(self.project)

    def test_context_reserve_setting_keeps_device_memory_out_of_the_kv_cache(self):
        model = self.project / "model"
        model.mkdir()
        header = json.dumps({"lm_head.weight": {"dtype": "BF16", "shape": [64, 64], "data_offsets": [0, 8192]}}).encode()
        (model / "model.safetensors").write_bytes(len(header).to_bytes(8, "little") + header + bytes(8192))
        (model / "config.json").write_text(json.dumps({"num_hidden_layers": 32, "num_attention_heads": 32, "num_key_value_heads": 8,
                                                       "head_dim": 128, "max_position_embeddings": 10_000_000}), encoding="utf-8")
        (self.project / "device.json").write_text(json.dumps({"device": "fixture gpu", "memory_free_bytes": 8 * GIB}),
                                                  encoding="utf-8")
        request = {key: value for key, value in json.loads(self.request.read_text(encoding="utf-8")).items()
                   if key not in ("kv_tokens", "vram_headroom_gib")}
        self.request.write_text(json.dumps(request | {"model_path": str(model)}), encoding="utf-8")
        capacities = {}
        for reserve in (3, 5):
            text = 'device = "device.json"' + chr(10) + 'vram_headroom_gib = 0.3' + chr(10) + f'context_reserve_gib = {reserve}' + chr(10)
            (self.project / "settings.toml").write_text(text, encoding="utf-8")
            code, output = self.create()
            self.assertEqual(code, 0)
            limits = self.limits(output)
            self.assertEqual(limits["context"]["reserve_bytes"], reserve * GIB)
            self.assertEqual(limits["vram_headroom_bytes"], round(0.3 * GIB))
            capacities[reserve] = limits["kv_tokens"]
        # 2 GiB more reserve at 65,536 bytes of FP8 KV per token: 32,768 fewer tokens
        self.assertEqual(capacities[3] - capacities[5], 2 * GIB // 65536)

    def test_unknown_or_mistyped_setting_is_an_error(self):
        for value in ("ram_headroom = 8", 'ram_headroom_gib = "8"', "wsl_run_root = true", "device = 1", "not toml"):
            (self.project / "settings.toml").write_text(value, encoding="utf-8")
            with self.assertRaises(ValueError):
                load_settings(self.project)
            self.assertEqual(self.create()[0], 1)

    def test_crit_eval_reads_the_settings_of_the_project_a_command_uses(self):
        """crit run and crit bench name their --project to crit-eval (CRIT_PROJECT, which crit_eval.machine_settings
        reads) while they run, and crit eval names crit's project unless the environment names one."""
        seen = []

        def look(*arguments, **options):
            seen.append(os.environ.get(PROJECT_VARIABLE))
            raise ValueError("looked")

        other = self.project / "other"
        with patch.dict(os.environ), patch("crit.run.cli.RunStore", side_effect=look), \
                patch("crit.pull.bench.bench", side_effect=look), \
                patch("crit.cli.family", return_value=SimpleNamespace(main=look)), \
                contextlib.redirect_stderr(io.StringIO()):
            os.environ.pop(PROJECT_VARIABLE, None)
            self.assertEqual(main(["--project", str(self.project), "--id", "run-x", "show"]), 1)
            self.assertEqual(bench_main(["owner/engine", "--data", str(self.project), "--project", str(other)]), 1)
            for named in (None, str(other)):
                if named:
                    os.environ[PROJECT_VARIABLE] = named
                with self.assertRaisesRegex(ValueError, "looked"):
                    cli.main(["eval", "status", "result.json"])
            self.assertEqual(os.environ[PROJECT_VARIABLE], str(other))   # as the environment named it
            os.environ.pop(PROJECT_VARIABLE)
            self.assertEqual(seen, [str(self.project), str(other), str(cli.default_project().resolve()), str(other)])
            self.assertNotIn(PROJECT_VARIABLE, os.environ)   # named only while the command ran


if __name__ == "__main__":
    unittest.main()
