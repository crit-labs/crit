"""crit bench and crit hub report: an installed engine measured with crit Eval through crit serve, its card's stats
beside the measured ones, and the result added to its card in a crit-hub checkout."""
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
for _folder in ("core/python", "api", "../crit-eval", "kernels", "cli", "tests"): sys.path.insert(0, str(ROOT / _folder))
from crit.pull import bench, report  # noqa: E402
from crit.run.records import read_json, write_json  # noqa: E402
from test_pull import ENGINE, PullTests, yaml  # noqa: E402


class Installed(PullTests):
    """test_pull's fixture (a crit-engines checkout, a hub card, a checkpoint, a store) without its tests."""
for _name in [n for n in dir(PullTests) if n.startswith("test_")]:
    setattr(Installed, _name, None)


@unittest.skipIf(yaml is None, "PyYAML is not installed")
class BenchTests(Installed):
    def prepared_data(self):
        data = self.root / "data"
        data.mkdir()
        write_json(data / "prepared.json", {"protocol": "crit-eval-fixture", "speed": []})
        return data

    def test_bench_serves_the_installed_engine_to_crit_eval_at_its_context_and_compares_with_the_card(self):
        record = self.install()
        data = self.prepared_data()
        seen = {}

        def run(adapter, model, data_folder, out, config, parts, **options):
            seen.update(model=model, parts=parts, options=options, server=adapter.server)

            class Engine:
                close = None

            def served(server, context):   # crit Eval's chat client over the installed engine's API
                with urllib.request.urlopen(server.url + "/models", timeout=30) as response:
                    seen["models"] = json.load(response)
                return Engine()
            with patch("crit_eval.run.chat.Server.create_eval_engine", served):
                adapter.create_eval_engine({"context_tokens": options["context_tokens"]})
                with self.assertRaisesRegex(ValueError, "serves at the 4,096 tokens"):
                    adapter.create_eval_engine({"context_tokens": 1024})
            result = {"protocol": "crit-eval-fixture", "server": adapter.server, "label": options["label"],
                      "speed": {"decode_score": 80.0, "prefill_score": 900.0, "ttft_score_ms": 120.0},
                      "sampled_speed": {"decode_score": 99.0}, "context": {"tokens": options["context_tokens"]}}
            write_json(out, result)
            return result
        # The fixture package streams no replies; crit serve's chat API is all this test reaches.
        with patch("crit_eval.run.runner.run", run), patch("sys.stderr"),                 patch.object(bench, "SERVED", ("chat-json-v1", "chat-resident-v1")):
            result, card, label, out = bench.bench(ENGINE, self.store, data, device=self.device, project=ROOT)
        self.assertEqual(seen["model"], str(self.store.root / record["checkpoint"]["folder"]))
        self.assertEqual(seen["parts"], ("speed",), "no reference scores in the data: speed only")
        self.assertEqual(seen["options"]["context_tokens"], 4096)
        self.assertEqual(seen["options"]["label"], f"{ENGINE} @ {self.revision[:12]}")
        self.assertEqual(seen["server"], f"crit serve --engine {ENGINE}")
        self.assertEqual([m["id"] for m in seen["models"]["data"]], [ENGINE])
        self.assertEqual(out.parent, self.store.folder(ENGINE) / "bench")
        self.assertEqual((label, card["decode_tok_s"]), ("card (rtx5090)", 100.0))
        lines = bench.comparison(card, bench.measured_stats(result), label).splitlines()
        self.assertIn("100.0", next(l for l in lines if l.startswith("decode")))
        self.assertIn("99.0", next(l for l in lines if l.startswith("decode")))
        self.assertTrue(next(l for l in lines if l.startswith("context")).endswith("4,096"))

    def test_quality_needs_the_reference_scores_and_the_packages_score_call(self):
        self.install()
        data = self.prepared_data()
        (data / "reference").mkdir()
        (data / "reference" / "index.json").write_text("{}", encoding="utf-8")
        seen = {}

        def run(adapter, model, data_folder, out, config, parts, **options):
            seen["parts"] = parts
            write_json(out, {"speed": {}})
            return {"speed": {}}
        with patch("crit_eval.run.runner.run", run), patch("sys.stderr"),                 patch.object(bench, "SERVED", ("chat-json-v1", "chat-resident-v1")):
            bench.bench(ENGINE, self.store, data, device=self.device, project=ROOT)
        # The fixture package declares no chat-score-v1: quality cannot be measured whatever the data holds.
        self.assertEqual(seen["parts"], ("speed",))

    def test_the_stats_are_what_crit_hub_reads_from_a_result(self):
        result = {"speed": {"decode_score": 80.0, "prefill_score": 900.0, "ttft_score_ms": 120.0},
                  "quality": {"sets": {"a/served": {"ppl_ratio": 1.01, "kld_mean": 0.02},
                                       "a/prompt": {"ppl_ratio": 1.5, "kld_mean": 0.03}}},
                  "context": {"tokens": 4096}}
        self.assertEqual(bench.measured_stats(result), {"decode_tok_s": 80.0, "prefill_tok_s": 900.0, "ttft_ms": 120.0,
                                                        "ppl": {"vs_reference": 1.01}, "kld_mean": 0.03,
                                                        "context_tokens": 4096})
        card = {"results": [{"gpu": "rtx5090", "model": "m"}, {"gpu": "rtx4090", "model": "m"}]}
        self.assertEqual(bench.gpu_of(card, "NVIDIA GeForce RTX 4090"), "rtx4090")
        self.assertEqual(bench.gpu_of(card, "NVIDIA GeForce RTX 3090"), "rtx3090")
        self.assertEqual(bench.card_result(card, "rtx3090", "m")["gpu"], "rtx5090", "the model on another GPU")


@unittest.skipIf(yaml is None, "PyYAML is not installed")
class ReportTests(Installed):
    def checkout(self):
        """A crit-hub checkout whose tools/add_result.py records how it was run."""
        hub = self.root / "checkout"
        (hub / "engines" / ENGINE).mkdir(parents=True)
        (hub / "engines" / ENGINE / "card.yaml").write_text(yaml.safe_dump(self.card), encoding="utf-8")
        (hub / "tools").mkdir()
        (hub / "tools" / "add_result.py").write_text(
            "import json, sys\nfrom pathlib import Path\n"
            "Path('ran.json').write_text(json.dumps(sys.argv[1:]))\n", encoding="utf-8")
        return hub

    def test_a_bench_result_of_the_cards_commit_goes_to_crit_hubs_tool(self):
        self.install()
        hub = self.checkout()
        result = self.root / "result.json"
        write_json(result, {"server": f"crit serve --engine {ENGINE}", "label": f"{ENGINE} @ {self.revision[:12]}"})
        with patch("sys.stderr"):
            self.assertEqual(report.report(ENGINE, result, hub, "someone", self.store, gpu="rtx5090"), 0)
        ran = read_json(hub / "ran.json")
        self.assertEqual(ran[0], str((hub / "engines" / ENGINE).resolve()))
        self.assertEqual(ran[1:], [str(result.resolve()), "--by", "someone", "--gpu", "rtx5090"])

    def test_a_result_of_another_engine_or_commit_is_refused(self):
        self.install()
        hub = self.checkout()
        result = self.root / "result.json"
        write_json(result, {"server": "crit serve run-1/candidate-0001", "label": "x"})
        with self.assertRaisesRegex(ValueError, "not a crit bench result of someone/fast-engine"):
            report.report(ENGINE, result, hub, "someone", self.store)
        write_json(result, {"server": f"crit serve --engine {ENGINE}", "label": f"{ENGINE} @ {self.revision[:12]}"})
        card = dict(self.card, source={**self.card["source"], "revision": "e" * 40})
        (hub / "engines" / ENGINE / "card.yaml").write_text(yaml.safe_dump(card), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "a result describes the commit it was measured on"):
            report.report(ENGINE, result, hub, "someone", self.store)
        self.assertFalse((hub / "ran.json").exists())


del PullTests   # its tests run in tests/test_pull.py


if __name__ == "__main__":
    unittest.main()
