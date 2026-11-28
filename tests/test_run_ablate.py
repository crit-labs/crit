"""crit run ... ablate end to end: a toy engine built from plugins, packaged as a run's candidate and served through
crit serve, built again in the same resident host with each variant of its plugins, benched and smoke-tested; the
report says which plugin pays and which one slows the engine. The package is a whole engine (the toy engine's
plugins, inference/starters/plugin-engine/plugins, with the attention and drafter plugins below in their slots); its resident host runs
the scripted ENGINE over the assembled plugins (support/serve.py's scripted serving)."""
import json
import os
from pathlib import Path
import sys
import unittest

import test_run_measure as run_measure
from crit.run.ablate import ablate
from crit.run.records import read_json
from support.engines import USE
from support.serve import RECIPE, scripted_package, scripted_serving

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "inference" / "core" / "python"))
from crit_plugins.ablation import describe as ablation_describe  # noqa: E402
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
# The prefill: long against the serving jitter of a loaded machine. crit serve's request path took up to 30 ms more
# while other suites ran beside it, which at a 0.1 s prefill moved a variant's unchanged first token by 12-14% and its
# verdict from pays to trades; at PREFILL it stays near 6%, inside the tests' noise.
PREFILL = 0.5
SPIN = False   # wait by spinning, not sleeping: a sub-millisecond schedule


class Clock:
    """The toy's time: a request's phases on one absolute schedule from its start, so a wake that comes late (a
    sleep on a loaded Windows host overshoots by tens of milliseconds) delays no phase after it, and a request takes
    the time its phases add up to instead of that plus every overshoot."""

    def __init__(self):
        self.at = time.perf_counter()

    def wait(self, seconds):
        self.at += seconds
        while True:
            left = self.at - time.perf_counter()
            if left <= 0:
                return
            if not SPIN:
                time.sleep(min(left, 0.002))


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
        clock = Clock()
        clock.wait(PREFILL + attention.seconds_per_token * 4)   # the prefill
        emit('x')
        done, rounds = 1, 0
        while done < n:
            cancellation.check()
            k = min(per_round, n - done)
            clock.wait(ROUND + attention.seconds_per_token * k)
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

# An attention plugin that follows the level the engine asks of its prompt rows: its row-exact and exact ways are slower.
LEVEL_PLUGIN = '''
from crit_plugins.kernels import asks


class Plugin:
    def __init__(self, options):
        self.level = asks("prompt") if asks("prompt") in ("exact", "row-exact") else "repeatable"
        self.seconds_per_token = 0.004 if self.level != "repeatable" else 0.0
        self.tokens_per_round = 1

    def exactness(self):
        return {"prompt": self.level}


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
    evaluated = False   # an ablation checks a packaged candidate
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

    def test_a_level_variant_serves_the_engine_asking_that_level_of_its_rows(self):
        # An in-process A/B of the engine's exactness: the attention plugin takes its slow, exact way where the engine
        # asks exact prompt rows and its fast one otherwise, as a plugin whose option is auto does, and says what its
        # rows keep. The candidate as built asks exact prompt rows, as its run's quality mode gives them.
        engine = ENGINE.replace("        self.parts = assemble(",
                                "        from crit_plugins.kernels import use_exactness\n"
                                "        use_exactness(context.configuration.get('exactness'))\n"
                                "        self.parts = assemble(")
        self.assertIn("use_exactness", engine)

        def write_plugins(package):
            AblateTests.write_plugins(package)
            folder = package / "plugins" / "attn-level"
            (folder / "tests").mkdir(parents=True, exist_ok=True)
            (folder / "plugin.json").write_text(json.dumps({
                "schema": "crit-plugin-v1", "id": "attn-level", "title": "attn-level", "fills": ["attention"],
                "entry": "plugin.py:create", "options": {}, "needs": [], "devices": ["cpu"]}), encoding="utf-8")
            (folder / "plugin.py").write_text(LEVEL_PLUGIN, encoding="utf-8")
            (folder / "tests" / "test_plugin.py").write_text(TEST, encoding="utf-8")
        self.write_plugins = write_plugins
        selection = {"use": {**USE, "attention": "attn-level", "drafter": "draft-three"},
                     "base": {"attention": "attn-plain", "drafter": None}}
        candidate = self.plugin_package(engine=engine, selection=selection)
        exact = {"decode": "row-exact", "prompt": "row-exact"}   # the speed run's levels
        self.assertEqual(read_json(self.folder / "candidate.json")["configuration"]["exactness"], exact)
        result = ablate(self.store, self.run_id, candidate, repetitions=2, smoke=False, noise=0.10,
                        sets=["exactness.prompt=repeatable"])
        self.assertEqual([row["name"] for row in result["variants"]], ["full", "set:exactness.prompt=repeatable"])
        report = read_json(self.directory / result["report"])
        full, fast = report["variants"]
        self.assertEqual((full["state"], fast["state"]), ("measured", "measured"), report["variants"])
        self.assertEqual(full["exactness"], {"asked": exact, "prompt": "row-exact",
                                             "plugins": {"attn-level": {"prompt": "row-exact"}}})
        self.assertEqual(fast["exactness"], {"asked": {"decode": "row-exact", "prompt": "repeatable"},
                                             "prompt": "repeatable",
                                             "plugins": {"attn-level": {"prompt": "repeatable"}}})
        (row,) = report["blame"]
        self.assertEqual((row["plugin"], row["verdict"]), ("set:exactness.prompt=repeatable", "pays"), row)
        with self.assertRaisesRegex(Exception, "exactness is one of declared, exact, row-exact, repeatable"):
            ablate(self.store, self.run_id, candidate, smoke=False, sets=["exactness=bitwise"])

    @unittest.skipIf(os.environ.get("GITHUB_ACTIONS") == "true",
                     "resolves a 1% change in wall time: a shared runner's jitter between rounds (up to 12% on "
                     "Windows, a steady 2-3% prefill skew on Linux) is larger than the change")
    def test_alternating_rounds_resolve_a_one_percent_change(self):
        # The engine's decode rounds take 1% longer with attn-spin.extra=0.01, a change well inside the 3% a single
        # pass is judged against. Served in four alternating rounds and compared round by round, it slows the engine
        # in every round, while the same options served again (extra=0, the engine as built) change nothing every
        # round agrees on.
        engine = ENGINE.replace("ROUND = 0.006\n", "ROUND = 0.008\n").replace("SPIN = False", "SPIN = True")
        engine = engine.replace("clock.wait(PREFILL + attention.seconds_per_token * 4)", "clock.wait(0.05)")
        engine = engine.replace("clock.wait(ROUND + attention.seconds_per_token * k)",
                                "clock.wait(ROUND * (1 + attention.extra))")
        self.assertEqual((engine.count("SPIN = True"), engine.count("clock.wait(0.05)"),
                          engine.count("clock.wait(ROUND * (1 + attention.extra))")), (1, 1, 1))

        def write_plugins(package):
            AblateTests.write_plugins(package)
            folder = package / "plugins" / "attn-spin"
            (folder / "tests").mkdir(parents=True, exist_ok=True)
            (folder / "plugin.json").write_text(json.dumps({
                "schema": "crit-plugin-v1", "id": "attn-spin", "title": "attn-spin", "fills": ["attention"],
                "entry": "plugin.py:create", "needs": [], "devices": ["cpu"],
                "options": {"extra": {"type": "float", "default": 0.0, "min": 0.0, "max": 1.0}}}), encoding="utf-8")
            (folder / "plugin.py").write_text(PLUGIN.format(seconds=0.0).replace(
                '        self.tokens_per_round = options.get("width", 1)\n',
                '        self.tokens_per_round = 1\n        self.extra = options["extra"]\n'), encoding="utf-8")
            (folder / "tests" / "test_plugin.py").write_text(TEST, encoding="utf-8")
        self.write_plugins = write_plugins
        selection = {"use": {**USE, "attention": "attn-spin", "drafter": None},
                     "base": {"attention": "attn-plain", "drafter": None}}
        candidate = self.plugin_package(engine=engine, selection=selection)
        result = ablate(self.store, self.run_id, candidate, repetitions=3, smoke=False, rounds=4,
                        sets=["attn-spin.extra=0.01", "attn-spin.extra=0"])
        report = read_json(self.directory / result["report"])
        self.assertEqual((report["rounds"], report["noise"]), (4, 0.005))
        self.assertEqual([len(row["rounds"]) for row in report["variants"]], [4, 4, 4])
        self.assertEqual({row["state"] for row in report["variants"]}, {"measured"}, report["variants"])
        log = (self.directory / result["attempt"]["logs"][0]["path"]).read_text(encoding="utf-8")
        self.assertIn("round 2 of 4: set:attn-spin.extra=0, set:attn-spin.extra=0.01, full", log)
        rows = {row["plugin"]: row for row in report["blame"]}
        slower, same = rows["set:attn-spin.extra=0.01"], rows["set:attn-spin.extra=0"]
        changes = slower["paired"]["in_engine"]["decode_tps"]
        self.assertEqual(len(changes), 4)
        self.assertTrue(all(-0.03 < change < 0 for change in changes), changes)
        self.assertAlmostEqual(slower["in_engine"]["decode_tps"], -0.01, delta=0.005)
        self.assertEqual(slower["verdict"], "slows", slower)
        self.assertEqual(same["verdict"], "neutral", same)
        self.assertIn("(rounds ", ablation_describe(slower))

    def test_a_variant_is_warmed_until_its_decode_rounds_take_a_steady_time(self):
        # A rebuilt engine whose first three requests run cold, each less than the one before, as a real engine's
        # lazy captures do: one warmup would leave the bench's repetitions cold.
        cold = ENGINE.replace("        self.facts = self.parts.facts()\n",
                              "        self.facts = self.parts.facts()\n        self.cold = 3\n")
        cold = cold.replace("            clock.wait(ROUND + attention.seconds_per_token * k)\n",
                            "            clock.wait(ROUND + attention.seconds_per_token * k + 0.01 * self.cold)\n")
        cold = cold.replace("        prompt = sum(", "        self.cold = max(0, self.cold - 1)\n        prompt = sum(")
        self.assertEqual(cold.count("self.cold"), 4)
        candidate = self.plugin_package(engine=cold)
        result = ablate(self.store, self.run_id, candidate, names=["full"], repetitions=2, smoke=False)
        (full,) = read_json(self.directory / result["report"])["variants"]
        case = full["cases"]["case-0001"]
        # Three cold requests, then two that agree; a round costs 18 ms warm and 28 to 48 ms cold.
        self.assertGreaterEqual(case["warmups"], 5, case)
        self.assertGreater(case["decode_tps"], 3 / 0.024, case)

    def test_evals_cases_are_warmed_before_their_seeds_are_measured(self):
        # An engine whose first three requests of each prompt run cold, as an expert cache's residency follows the
        # content: an Eval case measured unwarmed decodes its first seeds slow, and later variants faster.
        import tempfile
        cold = ENGINE.replace("        self.facts = self.parts.facts()\n",
                              "        self.facts = self.parts.facts()\n        self.cold = {}\n")
        cold = cold.replace("        per_round = drafter", "        text = str(request['messages'])\n"
                            "        self.cold[text] = self.cold.get(text, 4) - 1\n        per_round = drafter")
        cold = cold.replace("            clock.wait(ROUND + attention.seconds_per_token * k)\n",
                            "            clock.wait(ROUND + attention.seconds_per_token * k + 0.01 * max(0, self.cold[text]))\n")
        self.assertEqual(cold.count("self.cold"), 4)
        candidate = self.plugin_package(engine=cold)
        with tempfile.TemporaryDirectory() as data:
            (Path(data) / "prepared.json").write_text(json.dumps({
                "sampling": {"temperature": 0},
                "speed": [{"id": "chat", "messages": [{"role": "user", "content": "rivers"}], "output_tokens": 24}]}),
                encoding="utf-8")
            result = ablate(self.store, self.run_id, candidate, names=["full"], repetitions=2, smoke=False, data=data,
                            seeds=2)
        (full,) = read_json(self.directory / result["report"])["variants"]
        case = full["cases"]["eval:chat"]
        # Three cold requests, then two that agree; a round costs 18 ms warm and 28 to 48 ms cold.
        self.assertGreaterEqual(case["warmups"], 4, case)
        self.assertGreater(case["decode_tps"], 3 / 0.024, case)

    def test_an_engine_that_does_not_report_its_plugins_fails_every_variant(self):
        candidate = self.plugin_package(engine=run_measure.ADAPTER)
        result = ablate(self.store, self.run_id, candidate, names=["full", "base"], smoke=False)
        self.assertEqual([row["state"] for row in result["variants"]], ["failed", "failed"])
        self.assertIn("the engine reports the plugins None", result["variants"][0]["detail"])
        self.assertEqual([row["verdict"] for row in result["blame"]], ["unmeasured", "unmeasured"])

    def test_an_option_is_not_blamed_for_a_serve_the_engine_as_built_failed_too(self):
        # Every variant failing alike (here an engine that does not report its plugins; in a build, a serve refused
        # for another attempt's unsettled cleanup) says nothing about the option: it is unmeasured, not "breaks".
        candidate = self.plugin_package(engine=run_measure.ADAPTER)
        result = ablate(self.store, self.run_id, candidate, smoke=False, sets=["draft-three.width=1"])
        self.assertEqual([row["state"] for row in result["variants"]], ["failed", "failed"])
        (row,) = result["blame"]
        self.assertEqual((row["plugin"], row["verdict"]), ("set:draft-three.width=1", "unmeasured"), row)
        self.assertIn("as built", row["detail"])

    def test_a_failed_variant_is_checked_against_the_engine_as_built_served_right_after_it(self):
        import crit.run.ablate as ablate_module

        class Served:
            expired, opened = False, 0

            def __init__(self, fails):
                self.fails = fails

            def open(self):
                self.opened += 1
                if self.fails:
                    raise RuntimeError("CUDA error: device lost")
        lines = []
        row = ablate_module._recheck(Served(True), {"name": "set:a.x=1", "state": "failed"}, lines.append)
        self.assertIn("the engine as built failed right after it too (RuntimeError: CUDA error: device lost)",
                      row["infrastructure"])
        self.assertIn("is not blamed on it", lines[-1])
        served = Served(False)
        self.assertNotIn("infrastructure", ablate_module._recheck(served, {"name": "set:b", "state": "failed"},
                                                                  lines.append))
        self.assertEqual(served.opened, 1)

    def test_an_interrupted_ablation_settles_its_attempt_and_keeps_the_variants_it_served(self):
        # Ctrl+C or SIGTERM while a variant is served: the variant leaves no empty entry the report cannot combine,
        # the attempt settles as interrupted (an unsettled one refuses every later serve of the run) and the interrupt
        # goes on.
        import crit.run.ablate as ablate_module
        candidate = self.plugin_package()
        served = ablate_module._variant
        calls = []

        def second_interrupts(*args, **kwargs):
            calls.append(1)
            if len(calls) == 2:
                raise KeyboardInterrupt
            return served(*args, **kwargs)

        ablate_module._variant = second_interrupts
        try:
            with self.assertRaises(KeyboardInterrupt):
                ablate(self.store, self.run_id, candidate, names=["full", "base"], repetitions=1, smoke=False)
        finally:
            ablate_module._variant = served
        (report,) = (self.folder / "logs").glob("ablation-*.json")
        self.assertEqual([row["name"] for row in read_json(report)["variants"]], ["full"])
        attempt = report.stem.removeprefix("ablation-")
        result = read_json(self.folder / "attempts" / attempt / "result.json")
        self.assertEqual((result["outcome"], result["cleanup_settled"]), ("interrupted", True))

    def test_quality_scores_each_variant_on_evals_quality_texts_and_needs_the_data(self):
        candidate = self.plugin_package()
        with self.assertRaisesRegex(ValueError, "--quality scores Eval's quality texts: give --data"):
            ablate(self.store, self.run_id, candidate, quality=True)
        import crit.run.ablate as ablate_module
        from unittest import mock
        seen = {}

        def item(engine, info, prepared, protocol, data, mode, budget, log):
            seen.update(mode=mode, budget=budget, data=data)
            return {"name": "precheck_quality", "state": "passed", "detail": "within", "budget_used": 0.38,
                    "worst_set": "long-32768/prompt",
                    "baseline": {"budget_used": 0.21, "set": "long-32768/prompt", "dtype": "bfloat16"}}
        judge = {"prepared": {"speed": []}, "model": "m", "kv_tokens": 4096, "protocol": {}, "data": "d", "mode": "speed",
                 "budget": {"kld_mean_max": 0.4}}
        lines = []
        with mock.patch("crit_eval.run.adapters.open_engine", lambda *a, **k: (None, {})), \
                mock.patch("crit_eval.run.precheck.quality_item", item):
            found = ablate_module._quality(judge, 1, "run/candidate", lines.append, "set:x.keep=8")
        self.assertEqual(found, {"state": "passed", "detail": "within", "budget_used": 0.38,
                                 "worst_set": "long-32768/prompt",
                                 "baseline": {"budget_used": 0.21, "set": "long-32768/prompt", "dtype": "bfloat16"}})
        self.assertEqual(seen, {"mode": "speed", "budget": {"kld_mean_max": 0.4}, "data": "d"})
        self.assertIn("set:x.keep=8: quality passed, 38% of the drift budget", lines[-1])
        self.assertIn("the 16-bit baseline 21% of it there", lines[-1])   # beside it, never judged

    def test_a_single_pass_that_judged_a_change_asks_for_rounds_before_it_is_acted_on(self):
        import contextlib
        import io
        from crit.run.ablate import _one_pass
        rows = [{"plugin": "attn-slow", "verdict": "slows"}, {"plugin": "draft-three", "verdict": "neutral"}]
        for rounds, said in ((1, True), (2, False)):
            err = io.StringIO()
            with contextlib.redirect_stderr(err):
                _one_pass(rounds, rows)
            self.assertEqual("one pass judged attn-slow;" in err.getvalue(), said, err.getvalue())
            self.assertNotIn("draft-three", err.getvalue())

    def test_a_candidate_whose_plugins_are_all_plain_has_nothing_to_switch(self):
        candidate = self.served()
        result = ablate(self.store, self.run_id, candidate, smoke=False)
        self.assertEqual([row["name"] for row in result["variants"]], ["full"])
        self.assertEqual(result["blame"], [])



class PreviousCloseTests(unittest.TestCase):
    def test_a_variant_row_says_how_the_engine_served_before_it_closed(self):
        # Served.open reads each reload's close (crit serve's last_close): the variant's row carries the one its
        # reload made, and a row whose reload closed nothing carries none.
        from types import SimpleNamespace
        from crit.run.ablate import _closed
        served = SimpleNamespace(closes=[{"returned": True, "judged": True, "message": "returned", "close": 1}])
        self.assertNotIn("previous_close", _closed(served, 1, {"name": "base"}))
        served.closes.append({"returned": False, "judged": True, "message": "elastic range a/b is open", "close": 2,
                              "reopened_at": None})
        self.assertEqual(_closed(served, 1, {"name": "full"})["previous_close"],
                         {"returned": False, "judged": True, "message": "elastic range a/b is open"})


if __name__ == "__main__":
    unittest.main()
