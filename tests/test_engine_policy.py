"""An engine's policy is its own (crit_plugins.policy): two engines loaded in one process, each with its own
exactness, quality mode, kernel tuning and decisions record, each see their own, at load and in the steps Core's
worker runs; the process's own policy, which answers outside any engine, is left as it was."""
import json
from pathlib import Path
import sys
import tempfile
import threading
import unittest

ROOT = Path(__file__).resolve().parents[1]
for _folder in ("core/python", "tests"):
    sys.path.insert(0, str(ROOT / _folder))

from core_text_fixtures import llama_files, write_files  # noqa: E402
from crit_core.engine import load, resolve_package  # noqa: E402
from crit_plugins import decisions, kernels, policy  # noqa: E402
from support.engines import STEP_LIBRARY, write_package  # noqa: E402

VOCAB = 420


def observe():
    """What a plugin sees of its engine's policy, through the functions plugins call."""
    return {"prompt": kernels.asks("prompt"), "decode": kernels.asks("decode"), "mode": kernels.quality_mode(),
            "activation": kernels.activation("nvfp4"),
            "tuning": None if kernels.tuning_file("toy-kernel") is None else kernels.tuning_file("toy-kernel").name,
            "choice": decisions.settle("probe", "choice", key={"shape": 1}, measure=lambda: {"us": 1.0},
                                       decide=lambda found: "measured")}


class PolicyTests(unittest.TestCase):
    def test_a_policy_is_active_only_inside_its_block_and_carried_to_another_thread(self):
        engine = policy.Policy(quality_mode="speed", load=7)
        self.assertIsNone(policy.active())
        with engine.active():
            self.assertIs(policy.current(), engine)
            self.assertEqual(kernels.quality_mode(), "speed")
            seen = []
            carried = policy.carried(lambda: seen.append(policy.active()))
            plain = lambda: seen.append(policy.active())   # noqa: E731
            for function in (carried, plain):
                thread = threading.Thread(target=function)
                thread.start()
                thread.join()
        self.assertEqual(seen, [engine, None])   # a new thread starts outside any engine unless carried
        self.assertIsNone(policy.active())
        self.assertIs(policy.current(), policy.process())

    def test_the_use_functions_set_the_active_engines_policy_and_outside_it_the_processs(self):
        engine = policy.Policy()
        before = policy.process().quality_mode
        with engine.active():
            kernels.use_quality_mode("preserve")
            kernels.use_exactness("repeatable")
        self.assertEqual((engine.quality_mode, engine.exactness), ("preserve", {"decode": "repeatable",
                                                                                "prompt": "repeatable"}))
        self.assertEqual(policy.process().quality_mode, before)

    def test_each_engine_settles_its_own_decisions(self):
        first, second = policy.Policy(load=2), policy.Policy(load=3)
        with first.active():
            self.assertEqual(decisions.settle("p", "d", key={}, measure=lambda: {"a": 1}, decide=lambda m: "one"), "one")
        with second.active():
            self.assertEqual(decisions.settle("p", "d", key={}, measure=lambda: {"a": 2}, decide=lambda m: "two"), "two")
            self.assertEqual(decisions.report()["decisions"][0]["measurement"], {"a": 2})
        self.assertEqual(first.decisions.settled("p", "d")["choice"], "one")
        self.assertEqual(first.decisions.report()["decisions"][0]["conditions"], {"load": 2, "first_load": False})


@unittest.skipUnless(STEP_LIBRARY, "Core's runtime (crit_core_steps) is not built")
class TwoEnginesTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.model = write_files(self.root / "model", llama_files(**{"config.json": {
            "model_type": "llama", "eos_token_id": 1, "text_config": {"eos_token_id": 4, "vocab_size": VOCAB}}}))

    def package(self, name, configuration, tuning=False, recorded=None):
        folder = write_package(self.root / name, configuration=configuration)
        if tuning:
            (folder / "tuning").mkdir()
            (folder / "tuning" / "toy-kernel.json").write_text("{}", encoding="utf-8")
        if recorded is not None:
            row = {"plugin": "probe", "name": "choice", "key": {"shape": 1}, "choice": recorded,
                   "measurement": {"us": 0.5}, "conditions": {"load": 2, "first_load": False}}
            (folder / "decisions.json").write_text(json.dumps({"version": 1, "decisions": [row]}), encoding="utf-8")
        return folder

    def watched(self, engine):
        """engine with its model's step() recording what it sees of the policy, on Core's worker."""
        seen, step = [], engine.model.step

        def watching(rows):
            seen.append(observe())
            return step(rows)
        engine.model.step = watching
        return seen

    def test_two_engines_in_one_process_each_see_their_own_policy(self):
        first = self.package("first", {"exactness": "repeatable", "quality_mode": "speed",
                                        "prompt_activations": "fp8", "kernel_tuning": "tuning",
                                        "decisions": "decisions.json"}, tuning=True, recorded="recorded")
        second = self.package("second", {"quality_mode": "preserve"})
        process = (kernels.quality_mode(), kernels.exactness(), policy.process().tuning)
        a = load(first, self.model, device="cpu")
        try:
            b = load(second, self.model, device="cpu")   # loaded while the first one serves
            try:
                self.assertEqual(a.policy.report()["exactness"], {"decode": "repeatable", "prompt": "repeatable"})
                self.assertEqual(b.policy.report()["exactness"], {"decode": "row-exact", "prompt": "exact"})
                seen_a, seen_b = self.watched(a), self.watched(b)
                a.generate([5, 6, 7], 2, stop_tokens=[])
                b.generate([5, 6, 7], 2, stop_tokens=[])
                a.generate([8, 9], 2, stop_tokens=[])
                self.assertTrue(seen_a and seen_b)
                self.assertEqual({json.dumps(s, sort_keys=True) for s in seen_a}, {json.dumps(
                    {"prompt": "repeatable", "decode": "repeatable", "mode": "speed", "activation": "fp8",
                     "tuning": "toy-kernel.json", "choice": "recorded"}, sort_keys=True)})
                self.assertEqual({json.dumps(s, sort_keys=True) for s in seen_b}, {json.dumps(
                    {"prompt": "exact", "decode": "row-exact", "mode": "preserve", "activation": None,
                     "tuning": None, "choice": "measured"}, sort_keys=True)})
                # Each engine's telemetry reports its own decisions, whichever thread asks.
                self.assertEqual([r["source"] for r in a.telemetry()["decisions"]["decisions"]], ["recorded"])
                self.assertEqual([r["source"] for r in b.telemetry()["decisions"]["decisions"]], ["measured"])
                self.assertTrue(a.telemetry()["decisions"]["record"].endswith("decisions.json"))
                self.assertIsNone(b.telemetry()["decisions"]["record"])
                self.assertEqual(a.info()["exactness"]["asked"], {"decode": "repeatable", "prompt": "repeatable"})
                self.assertEqual(b.info()["exactness"]["asked"], {"decode": "row-exact", "prompt": "exact"})
            finally:
                b.close()
        finally:
            a.close()
        self.assertIsNone(policy.active())
        self.assertEqual((kernels.quality_mode(), kernels.exactness(), policy.process().tuning), process)

    def test_a_dry_resolve_answers_under_the_packages_policy(self):
        package = self.package("dry", {"quality_mode": "preserve"})
        self.assertEqual(resolve_package(package, self.model, device="cpu")["facts"],
                         resolve_package(package, self.model, device="cpu")["facts"])
        self.assertIsNone(policy.active())


if __name__ == "__main__":
    unittest.main()
