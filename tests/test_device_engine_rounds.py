"""Core's captured rounds over the library's device plugins on a GPU. `crit plugins add` copies attention-state,
softmax-attention, projections-dense, token-sampling, fused-norms and device-placement into a package beside the
Llama-layout model plugin (support/device_plugins/llama-model) and two test stand-ins for the sampler's and drafter's
round forms (round-sampler over token-sampling, greedy only, and chain-drafter, drafting from the model's stub draft
head), which token-sampling and greedy-head-drafter also provide; the KV pool's and attention's round forms are the
library's own (attention-state, softmax-attention). With the model's
`rounds`, its greedy requests decode through crit_plugins.rounds.Rounds, captured per row count (in both of the
driver's forms) and launched by Core natively: plain rounds (no drafts) and drafting rounds decode token for token what
the same plugins decode through step(), batched rows what each decodes alone, a request asking for log-probabilities
gets in rounds what step() scores, each result counts its rounds and accepted drafts, a selection whose attention has
no round form decodes through step() instead, and loading leaves the package's bytes as they were. Needs CUDA, transformers, safetensors and Core's
crit_core_steps (CRIT_CORE_STEP_LIBRARY)."""
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
for _folder in ("core/python", "cli"):
    sys.path.insert(0, str(ROOT / _folder))

from crit_core.native import step_library  # noqa: E402

try:
    import torch
    import transformers
    import safetensors  # noqa: F401
    READY = torch.cuda.is_available() and step_library() is not None
except ImportError:
    READY = False

USE = {"model": "llama-model", "placement": "device-placement", "projections": "projections-dense",
       "attention": "softmax-attention", "kv_cache": "attention-state", "sampler": "round-sampler",
       "norms": "fused-norms"}
LIBRARY = ["attention-state", "softmax-attention", "projections-dense", "token-sampling", "device-placement",
           "fused-norms"]
STAND_INS = ["llama-model", "round-sampler", "chain-drafter"]
# A decode row through step() reads the KV as a round's column does (softmax-attention's split-KV reader, or its
# torch fallback without crit's kernel), so rounds decode token for token what step() decodes.
ATTENTION = {"softmax-attention": {"decode": "split"}}
GREEDY = {"temperature": 0.0}


def configuration(rounds, drafter=False, **options):
    use = {**USE, **({"drafter": "chain-drafter"} if drafter else {})}
    return {"plugins": {"use": use, "options": {"llama-model": {"rounds": rounds, **options}, **ATTENTION}}}


@unittest.skipUnless(READY, "a GPU engine: CUDA, transformers, safetensors and CRIT_CORE_STEP_LIBRARY")
class DeviceRoundTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from crit.plugins import add
        from crit.records import inventory
        from crit_core.engine import load
        cls.temp = tempfile.TemporaryDirectory()
        root = Path(cls.temp.name)
        torch.manual_seed(0)
        config = transformers.LlamaConfig(vocab_size=256, hidden_size=128, intermediate_size=256, num_hidden_layers=2,
                                          num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=1024,
                                          rms_norm_eps=1e-6, tie_word_embeddings=False)
        model = transformers.LlamaForCausalLM(config)
        with torch.no_grad():
            model.lm_head.weight.mul_(30)
        cls.model = root / "model"
        model.save_pretrained(cls.model, safe_serialization=True)
        cls.package = root / "package"
        add(cls.package / "plugins", LIBRARY)
        for plugin in STAND_INS:
            shutil.copytree(ROOT / "tests" / "support" / "device_plugins" / plugin, cls.package / "plugins" / plugin)
        # An attention plugin without a round form: softmax-attention's copy that does not provide rounds:attention.
        plain = cls.package / "plugins" / "plain-attention"
        shutil.copytree(cls.package / "plugins" / "base" / "softmax-attention", plain)
        manifest = json.loads((plain / "plugin.json").read_text(encoding="utf-8"))
        manifest.update(id="plain-attention", provides=[f for f in manifest["provides"] if f != "rounds:attention"],
                        sources={"state": "../attention-state/code"})
        (plain / "plugin.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        (cls.package / "configuration.json").write_text(json.dumps(configuration(False)), encoding="utf-8")
        cls.sealed = inventory(cls.package)
        written = sys.dont_write_bytecode
        sys.dont_write_bytecode = False   # loading must not write bytecode into the package even so
        try:
            cls.steps = load(cls.package, cls.model)                                    # every row through step()
            cls.plain = load(cls.package, cls.model, configuration(True))               # rounds of one token
            cls.drafting = load(cls.package, cls.model, configuration(True, drafter=True, width=3))
        finally:
            sys.dont_write_bytecode = written
        # The first requests after load, before any test serves one (a later test would find what earlier ones
        # compiled): prompt lengths no load request used, through step(), plain rounds and drafting rounds.
        from crit_plugins.testing import compiles
        prompts = [cls.prompts(cls, 1, length)[0] for length in (3, 31, 130, 300)]
        with compiles() as found:
            for engine in (cls.steps, cls.plain, cls.drafting):
                cls.alone(cls, engine, prompts, 5)
                cls.together(cls, engine, prompts, 5)
        cls.compiled_after_load = found

    @classmethod
    def tearDownClass(cls):
        for engine in (cls.steps, cls.plain, cls.drafting):
            engine.close()
        cls.temp.cleanup()

    def prompts(self, count, length=9):
        g = torch.Generator().manual_seed(count * 7 + length)
        return [torch.randint(3, 256, (length + 4 * i,), generator=g).tolist() for i in range(count)]

    def alone(self, engine, prompts, count, **options):
        return [engine.generate(p, count, stop_tokens=[], sampling=GREEDY, **options) for p in prompts]

    def together(self, engine, prompts, count):
        with ThreadPoolExecutor(len(prompts)) as pool:
            return list(pool.map(lambda p: engine.generate(p, count, stop_tokens=[], sampling=GREEDY), prompts))

    def test_requests_after_load_compile_nothing_at_new_prompt_lengths_and_round_widths(self):
        # A plugin compiles at load: the first requests after it (setUpClass, before any test), at prompt lengths no
        # load request used, through step(), plain rounds and drafting rounds, compiled no kernel and built no
        # extension.
        self.assertEqual(self.compiled_after_load, [])

    def test_a_round_launched_without_a_graph_compiles_nothing_after_the_drivers_warm(self):
        # warm() runs one eager round, every row held, of each (rows, width) the driver has no graph of, and puts the
        # driver's state back: a model's step() that launches such a round itself compiles nothing on a request, and
        # the engine decodes as before.
        from crit_plugins.testing import compiles
        rounds = self.drafting.model.rounds
        before = {name: tensor.clone() for name, tensor in rounds._state().items()}
        self.assertEqual(rounds.warm("greedy", widths=[1, 2]), 2 * rounds.rows * len(rounds.forms))
        for name, tensor in rounds._state().items():
            self.assertTrue(torch.equal(tensor, before[name]), name)
        with compiles() as found:
            rounds.launch(rounds.rows, 2, "greedy", hold=range(rounds.rows))
            rounds.read()
        self.assertEqual(found, [])
        prompt = self.prompts(1)[0]
        reference, _ = self.steps.generate(prompt, 12, stop_tokens=[], sampling=GREEDY)
        tokens, _ = self.drafting.generate(prompt, 12, stop_tokens=[], sampling=GREEDY)
        self.assertEqual(tokens, reference)

    def test_the_rounds_are_captured_and_handed_to_core(self):
        for engine, width in ((self.plain, 0), (self.drafting, 3)):
            rounds = engine.model.rounds
            self.assertEqual(rounds.problems, [])
            self.assertIsNotNone(rounds.native)
            self.assertEqual(sorted(rounds.graphs), [(rows, width, "greedy", form) for rows in range(1, 5)
                                                     for form in (False, True)])
        self.assertTrue(self.drafting.info()["speculative"])
        self.assertIsNone(self.steps.model.rounds)

    def test_plain_rounds_decode_what_step_decodes_batched_as_alone(self):
        prompts = self.prompts(4)
        reference = [tokens for tokens, _ in self.alone(self.steps, prompts, 24)]
        def python_steps():   # Core's steps less its native rounds: the steps the model ran in Python
            core = self.plain.telemetry()["core"]
            return core["steps"] - core["native_rounds"]
        before = python_steps()
        found = self.alone(self.plain, prompts, 24)
        self.assertEqual([tokens for tokens, _ in found], reference)
        # Core ran the decode rounds: Python stepped each request's prompt, not its 23 decode rows.
        self.assertLess(python_steps() - before, 4 * 4)
        for tokens, result in found:
            self.assertEqual(result["rounds"], len(tokens) - 1)
            self.assertNotIn("accepted", result)   # one token a round: nothing drafted
        self.assertEqual([tokens for tokens, _ in self.together(self.plain, prompts, 24)], reference)

    def test_drafting_rounds_decode_what_step_decodes_and_count_their_accepted_drafts(self):
        prompts = self.prompts(4, 11)
        reference = [tokens for tokens, _ in self.alone(self.steps, prompts, 30)]
        found = self.alone(self.drafting, prompts, 30)
        self.assertEqual([tokens for tokens, _ in found], reference)
        accepted = []
        for tokens, result in found:
            self.assertEqual(len(result["accepted"]), result["rounds"])
            self.assertEqual(sum(result["accepted"]) + result["rounds"], len(tokens) - 1)
            self.assertEqual(result["drafted"], [3] * result["rounds"])
            accepted += result["accepted"]
        # The stub head drafts wrong at one position in five: rounds keep some drafts and lose others.
        self.assertGreater(max(accepted), 1)
        self.assertLess(min(accepted), 3)
        self.assertLess(sum(r["rounds"] for _, r in found), sum(len(t) - 1 for t, _ in found) * 0.7)
        self.assertEqual([tokens for tokens, _ in self.together(self.drafting, prompts, 30)], reference)
        # The driver's counters reach the engine's telemetry through the model plugin's telemetry().
        drafts = self.drafting.telemetry()["drafts"]
        self.assertEqual(len(drafts["accepted_by_depth"]), 3)
        self.assertGreater(drafts["rounds"], 0)
        self.assertTrue(0 < drafts["accepted_by_depth"][0] <= drafts["offered_by_depth"][0])

    def test_a_stop_token_ends_a_round_where_step_decoding_ends(self):
        (prompt,) = self.prompts(1, 13)
        reference, _ = self.steps.generate(prompt, 30, stop_tokens=[], sampling=GREEDY)
        stop = reference[11]
        expected, _ = self.steps.generate(prompt, 30, stop_tokens=[stop], sampling=GREEDY)
        for engine in (self.plain, self.drafting):
            tokens, result = engine.generate(prompt, 30, stop_tokens=[stop], sampling=GREEDY)
            self.assertEqual(tokens, expected)
            self.assertEqual(result["finish"], "stop")

    def test_log_probability_requests_decode_in_rounds_scoring_what_step_scores(self):
        # Each round with such a row runs the driver's log-probability form: every emitted token's log-probability and
        # the alternatives asked for, as step() scores them, the drafting engine's rounds drafting as before.
        (prompt,) = self.prompts(1, 10)
        def scored(engine):
            rows = []
            tokens, result = engine.generate(prompt, 12, stop_tokens=[], sampling=GREEDY, logprobs=2,
                                             on_tokens=lambda found, values: rows.extend(values))
            return tokens, rows, result
        tokens, reference, _ = scored(self.steps)
        self.assertEqual(len(reference), 12)
        for engine in (self.plain, self.drafting):
            before = engine.model.rounds.telemetry()["rounds"]["by_form"]["logprobs"]
            found, rows, result = scored(engine)
            self.assertEqual(found, tokens)
            self.assertEqual([[t for t, _ in top] for _, top in rows], [[t for t, _ in top] for _, top in reference])
            for (value, top), (expected, expected_top) in zip(rows, reference):
                self.assertAlmostEqual(value, expected, delta=1e-3)
                for (_, a), (_, b) in zip(top, expected_top):
                    self.assertAlmostEqual(a, b, delta=1e-3)
            self.assertGreater(engine.model.rounds.telemetry()["rounds"]["by_form"]["logprobs"], before)
        self.assertLess(result["rounds"], len(tokens) - 1)   # the drafting engine's rounds kept drafts

    def test_requests_rounds_cannot_take_decode_through_step(self):
        # A replay or plain decoding on request: the same plugins, one token a step.
        (prompt,) = self.prompts(1, 10)
        reference, _ = self.steps.generate(prompt, 12, stop_tokens=[], sampling=GREEDY)
        tokens, _ = self.drafting.generate(prompt, 12, stop_tokens=[], sampling=GREEDY, plain=True)
        self.assertEqual(tokens, reference)
        tokens, _ = self.drafting.generate(prompt, 12, stop_tokens=[], sampling=GREEDY, replay=reference)
        self.assertEqual(tokens, reference)

    def test_no_proxy_stays_on_the_hot_path(self):
        from crit_plugins.observe import Counted, Observed
        model = self.drafting.model
        rounds = model.rounds
        for part in (model.proj, model.attention, model.sampler, model.pool, model.drafter, rounds.sampler,
                     rounds.drafter, rounds.pools[0][1], rounds.model):
            self.assertNotIsInstance(part, (Observed, Counted))

    def test_a_selection_whose_attention_has_no_round_form_decodes_through_step(self):
        from crit_core.engine import load
        chosen = configuration(True, drafter=True)
        chosen["plugins"]["use"]["attention"] = "plain-attention"
        chosen["plugins"]["options"]["plain-attention"] = chosen["plugins"]["options"].pop("softmax-attention")
        engine = load(self.package, self.model, chosen)
        try:
            rounds = engine.model.rounds
            self.assertIsNone(rounds.native)
            self.assertEqual(rounds.graphs, {})
            self.assertIn("plain-attention in attention does not provide rounds:attention", rounds.problems[0])
            (prompt,) = self.prompts(1)
            tokens, result = engine.generate(prompt, 10, stop_tokens=[], sampling=GREEDY)
            self.assertEqual(len(tokens), 10)
            self.assertEqual(result["rounds"], 9)
        finally:
            engine.close()

    def test_loading_left_the_packages_bytes_as_they_were(self):
        from crit.records import verify_inventory
        self.assertEqual(list(self.package.rglob("__pycache__")), [])
        verify_inventory(self.package, self.sealed)


if __name__ == "__main__":
    unittest.main()
