"""crit run ... ablate end to end: a toy engine built from plugins, packaged as a run's candidate and served through
crit serve, built again in the same resident host with each variant of its plugins, benched and smoke-tested; the
report says which plugin pays and which one slows the engine. The package is a whole engine (the toy engine's
plugins, inference/examples/plugin-engine/plugins, with the attention and drafter plugins below in their slots); its resident host runs
the scripted ENGINE over the assembled plugins (support/serve.py's scripted serving)."""
import json
from pathlib import Path
import sys
import unittest

import test_run_measure as run_measure
from crit.run.ablate import ablate
from crit.run.records import read_json
from support.engines import USE
from support.serve import RECIPE, scripted_package, scripted_serving

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "core" / "python"))
from crit_plugins.testing import run_tests  # noqa: E402

# The toy engine: the plugin in its attention slot adds seconds per token, the one in its drafter slot makes a decode
# round give several tokens. Every round costs ROUND seconds, so drafting three tokens a round decodes faster and a slow
# attention plugin decodes slower, which the ablation must tell apart. The drafter reads the target's hidden states,
# a fact the engine provides itself, as an MTP drafter needs the model's own head.
ENGINE = r'''
import time
from pathlib import Path

from crit_plugins import Registry, assemble

ROUND = 0.006


class Engine:
    def __init__(self, context):
        self.weights = context.host('model.fixture')
        self.parts = assemble(context.configuration['plugins'], Registry.discover(Path(__file__).parent / 'plugins'),
                              context, facts=['taps:hidden'])
        self.facts = self.parts.facts()

    def generate(self, request, cancellation):
        return self.generate_stream(request, cancellation, lambda piece: None)

    def generate_stream(self, request, cancellation, emit):
        n = request['limits']['max_output_tokens']
        drafter, attention = self.parts.get('drafter'), self.parts['attention']
        per_round = drafter.tokens_per_round if drafter is not None else 1
        time.sleep(0.1 + attention.seconds_per_token * 4)   # the prefill, long enough that serving jitter stays under 3%
        emit('x')
        done, rounds = 1, 0
        while done < n:
            cancellation.check()
            k = min(per_round, n - done)
            time.sleep(ROUND + attention.seconds_per_token * k)
            for _ in range(k):
                emit('x')
            done, rounds = done + k, rounds + 1
        prompt = sum(len(m['content']) for m in request['messages'])
        return dict(schema_version=1, record_kind='chat_response',
                    **{k: request[k] for k in ('request_id', 'candidate_id', 'package_sha256')}, assistant='x' * n,
                    telemetry=dict(prompt_tokens=prompt, output_tokens=n, prefill_tokens=prompt, prefill_ms=5,
                                   decode_tokens=n - 1, decode_ms=1, ttft_ms=5, missing={}, decode_rounds=rounds))

    def reset(self):
        pass

    def close(self):
        self.parts.close()


def create_engine(context):
    return Engine(context)
'''

PLUGIN = '''
class Plugin:
    def __init__(self, options):
        self.seconds_per_token = {seconds}
        self.tokens_per_round = options.get("width", 1)


def create(context, options, parts):
    return Plugin(options)
'''

TEST = '''from crit_plugins.testing import PluginTestCase


class Tests(PluginTestCase):
    def test_it_assembles(self):
        self.assertIsNotNone(self.assemble(facts=["taps:hidden"]))
'''

PLUGINS = {"attn-plain": (["attention"], 0.0, {}, [], ["cpu"]), "attn-slow": (["attention"], 0.004, {}, [], ["cpu"]),
           "draft-three": (["drafter"], 0.0, {"width": {"type": "int", "default": 3, "min": 1, "max": 8}},
                           ["taps:hidden"], ["host"])}
# The toy engine's plugins in every other slot an engine fills: their base is themselves, so the ablation switches
# only the attention and drafter plugins.
SELECTION = {"use": {**USE, "attention": "attn-slow", "drafter": "draft-three"},
             "base": {"attention": "attn-plain", "drafter": None}}


class AblateTests(run_measure.MeasureFixture, unittest.TestCase):
    def setUp(self):
        super().setUp()
        # A candidate-budget run: its safety window outlasts the ablation's reservation.
        data = {key: value for key, value in self.data.items() if key != "budget_minutes"}
        data["workload"] = {"repetitions": 3, "output_tokens": 48}
        self.run_id = self.store.create(data)["run"]["run_id"]
        self.directory = self.store.directory(self.run_id)

    @staticmethod
    def write_plugins(package):
        """The attention and drafter plugins in package/plugins, beside the toy engine's."""
        for plugin_id, (fills, seconds, options, needs, devices) in PLUGINS.items():
            folder = package / "plugins" / plugin_id
            (folder / "tests").mkdir(parents=True, exist_ok=True)
            (folder / "plugin.json").write_text(json.dumps({
                "schema": "crit-plugin-v1", "id": plugin_id, "title": plugin_id, "fills": fills,
                "entry": "plugin.py:create", "options": options, "needs": needs, "devices": devices}), encoding="utf-8")
            (folder / "plugin.py").write_text(PLUGIN.format(seconds=seconds), encoding="utf-8")
            (folder / "tests" / "test_plugin.py").write_text(TEST, encoding="utf-8")

    def plugin_package(self, engine=ENGINE, selection=SELECTION):
        self.write_plugins(self.root / "source-input")
        candidate = self.candidate(configuration={"plugins": selection})["candidate_id"]
        package = self.root / ("package-" + candidate)
        self.write_plugins(package)
        scripted_serving(self, run_measure.SERVED)
        self.store.package(self.run_id, candidate, scripted_package(package, engine), RECIPE)
        self.folder = self.directory / "candidates" / candidate
        return candidate

    def test_each_variant_is_served_and_the_report_says_which_plugin_pays_and_which_slows(self):
        candidate = self.plugin_package()
        items = run_tests(self.folder / "package" / "plugins", log=lambda line: None)
        self.assertEqual({i["state"] for i in items}, {"passed"}, items)
        # Wider than ablation.NOISE (3%): the toy's effects are 14% and more (attn-slow's prefill), while a loaded
        # machine's serving jitter reaches about 6% (a prefill of the same fixed sleep measured 5.8% apart).
        result = ablate(self.store, self.run_id, candidate, repetitions=2, noise=0.10)
        self.assertEqual(result["attempt"]["outcome"], "completed")
        variants = {row["name"]: row for row in result["variants"]}
        # With two toggles, base plus one plugin is the engine without the other one.
        self.assertEqual(list(variants), ["full", "base", "without:attn-slow", "without:draft-three"])
        self.assertEqual({row["state"] for row in variants.values()}, {"measured"}, variants)
        decode = {name: row["kpis"]["decode_tps"] for name, row in variants.items()}
        self.assertGreater(decode["without:attn-slow"], decode["full"])
        self.assertGreater(decode["full"], decode["without:draft-three"])
        report = read_json(self.directory / result["report"])
        rows = {row["plugin"]: row for row in report["blame"]}
        self.assertEqual(rows["draft-three"]["verdict"], "pays", rows)
        self.assertEqual(rows["attn-slow"]["verdict"], "slows", rows)
        self.assertEqual(report["variants"][2]["also"], ["only:draft-three"])
        full = report["variants"][0]
        self.assertEqual(full["use"], SELECTION["use"])
        self.assertAlmostEqual(full["cases"]["case-0001"]["accepted_tokens_per_round"], 3.0, delta=0.1)
        # crit smoke ran on every variant; the toy offers none of what it reads, so it skipped.
        self.assertEqual(full["smoke"]["failed"], [])
        self.assertGreater(full["smoke"]["skipped"], 0)
        self.assertEqual(result["attempt"]["logs"][1]["path"], result["report"])
        self.assertEqual(self.store.get(self.run_id)["errors"], [])

    def test_option_variants_on_evals_cases_say_what_an_option_does(self):
        # An in-process A/B: the drafter at width 1 against the engine as built, on the run's case and on crit-eval's
        # speed cases and held-out prompts, one request per seed.
        import tempfile
        candidate = self.plugin_package()
        with tempfile.TemporaryDirectory() as data:
            prompt = [{"role": "user", "content": "tell me about rivers"}]
            (Path(data) / "prepared.json").write_text(json.dumps({
                "sampling": {"temperature": 1.0, "top_p": 0.95},
                "speed": [{"id": "chat", "messages": prompt, "output_tokens": 32}],
                "heldout": [{"id": "de", "messages": prompt, "output_tokens": 16}]}), encoding="utf-8")
            result = ablate(self.store, self.run_id, candidate, repetitions=2, smoke=False, noise=0.10, data=data,
                            seeds=2, sets=["draft-three.width=1"])
        self.assertEqual([row["name"] for row in result["variants"]], ["full", "set:draft-three.width=1"])
        report = read_json(self.directory / result["report"])
        full = report["variants"][0]
        self.assertEqual(sorted(full["cases"]), ["case-0001", "eval:chat", "eval:heldout:de"])
        self.assertEqual((report["eval_data"], report["seeds"]), (data, 2))
        (row,) = report["blame"]
        self.assertEqual((row["plugin"], row["verdict"]), ("set:draft-three.width=1", "slows"), row)
        self.assertLess(row["in_engine"]["decode_tps"], -0.10)

    def test_a_variant_is_warmed_until_its_decode_rounds_take_a_steady_time(self):
        # A rebuilt engine whose first three requests run cold, each less than the one before, as a real engine's
        # lazy captures do: one warmup would leave the bench's repetitions cold.
        cold = ENGINE.replace("        self.facts = self.parts.facts()\n",
                              "        self.facts = self.parts.facts()\n        self.cold = 3\n")
        cold = cold.replace("            time.sleep(ROUND + attention.seconds_per_token * k)\n",
                            "            time.sleep(ROUND + attention.seconds_per_token * k + 0.01 * self.cold)\n")
        cold = cold.replace("        prompt = sum(", "        self.cold = max(0, self.cold - 1)\n        prompt = sum(")
        self.assertEqual(cold.count("self.cold"), 4)
        candidate = self.plugin_package(engine=cold)
        result = ablate(self.store, self.run_id, candidate, names=["full"], repetitions=2, smoke=False)
        (full,) = read_json(self.directory / result["report"])["variants"]
        case = full["cases"]["case-0001"]
        # Three cold requests, then two that agree; a round costs 18 ms warm and 28 to 48 ms cold.
        self.assertGreaterEqual(case["warmups"], 5, case)
        self.assertGreater(case["decode_tps"], 3 / 0.024, case)

    def test_an_engine_that_does_not_report_its_plugins_fails_every_variant(self):
        candidate = self.plugin_package(engine=run_measure.ADAPTER)
        result = ablate(self.store, self.run_id, candidate, names=["full", "base"], smoke=False)
        self.assertEqual([row["state"] for row in result["variants"]], ["failed", "failed"])
        self.assertIn("the engine reports the plugins None", result["variants"][0]["detail"])
        self.assertEqual([row["verdict"] for row in result["blame"]], ["unmeasured", "unmeasured"])

    def test_a_candidate_whose_plugins_are_all_plain_has_nothing_to_switch(self):
        candidate = self.served()
        result = ablate(self.store, self.run_id, candidate, smoke=False)
        self.assertEqual([row["name"] for row in result["variants"]], ["full"])
        self.assertEqual(result["blame"], [])


if __name__ == "__main__":
    unittest.main()
