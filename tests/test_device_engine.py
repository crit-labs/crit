"""A whole engine of the library's device plugins on a GPU. `crit plugins add` copies attention-state (the KV pool),
softmax-attention, projections-dense, token-sampling, fused-norms (with its base, torch-norms) and device-placement
into a package beside a Llama-layout model plugin (support/device_plugins/llama-model); Core's loader runs it on Core's
concurrent runtime, and with torch-norms in fused-norms' place it decodes the same tokens. Its fixed-text
scores (a prefill chunk at a time) equal transformers' log-probabilities of the same checkpoint, each token of its greedy continuations is
transformers' choice at that position up to BF16 rounding, a reply kept to a grammar takes transformers' best allowed
token, and requests in flight together decode what each decodes alone. Needs CUDA, transformers, safetensors and
Core's crit_core_steps (CRIT_CORE_STEP_LIBRARY)."""
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

SELECTION = {"use": {"model": "llama-model", "placement": "device-placement", "projections": "projections-dense",
                     "attention": "softmax-attention", "kv_cache": "attention-state", "sampler": "token-sampling",
                     "norms": "fused-norms"},
             "base": {"norms": "torch-norms"}}


@unittest.skipUnless(READY, "a GPU engine: CUDA, transformers, safetensors and CRIT_CORE_STEP_LIBRARY")
class DeviceEngineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from crit.plugins import add
        from crit_core.engine import load
        cls.temp = tempfile.TemporaryDirectory()
        root = Path(cls.temp.name)
        torch.manual_seed(0)
        config = transformers.LlamaConfig(vocab_size=256, hidden_size=128, intermediate_size=256, num_hidden_layers=2,
                                          num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=1024,
                                          rms_norm_eps=1e-6, tie_word_embeddings=False)
        model = transformers.LlamaForCausalLM(config)
        with torch.no_grad():
            model.lm_head.weight.mul_(30)   # sharp logits: a greedy choice no rounding flips
        cls.reference = model.to(torch.bfloat16).cuda().eval()
        cls.model = root / "model"
        model.save_pretrained(cls.model, safe_serialization=True)
        package = root / "package"
        add(package / "plugins", ["attention-state", "softmax-attention", "projections-dense", "token-sampling",
                                  "device-placement", "fused-norms"])
        shutil.copytree(ROOT / "tests" / "support" / "device_plugins" / "llama-model", package / "plugins" / "llama-model")
        (package / "configuration.json").write_text(json.dumps({"plugins": SELECTION}), encoding="utf-8")
        cls.engine = load(package, cls.model)

    @classmethod
    def tearDownClass(cls):
        cls.engine.close()
        cls.temp.cleanup()

    def prompts(self, count, length=17):
        g = torch.Generator().manual_seed(count * 101 + length)
        return [torch.randint(3, 256, (length + 4 * i,), generator=g).tolist() for i in range(count)]

    def test_the_engine_runs_the_librarys_device_plugins(self):
        info = self.engine.info()
        self.assertEqual(info["plugins"], SELECTION["use"])
        self.assertGreater(info["max_concurrent_requests"], 1)

    def test_scores_equal_transformers_log_probabilities(self):
        (ids,) = self.prompts(1, 150)   # three of the model's prefill chunks, the KV carried from one to the next
        scored = self.engine.score(ids, 1, "prefill")
        with torch.no_grad():
            logp = self.reference(torch.tensor([ids]).cuda()).logits[0].float().log_softmax(-1)
        expected = logp[torch.arange(len(ids) - 1), torch.tensor(ids[1:])].tolist()
        worst = max(abs(a - b) for a, b in zip(scored.target, expected))
        self.assertLess(worst, 0.25, worst)

    def test_greedy_continuations_are_transformers_choices(self):
        # Each token, given the ones before it, is transformers' argmax unless two logits tie within BF16's step
        # (0.125 at these logits' size), where the two engines' rounding may pick either.
        for prompt in self.prompts(3):
            tokens, _ = self.engine.generate(prompt, 12, stop_tokens=[], sampling={"temperature": 0.0})
            with torch.no_grad():
                logits = self.reference(torch.tensor([prompt + tokens]).cuda()).logits[0].float()
            for i, token in enumerate(tokens):
                row = logits[len(prompt) - 1 + i]
                self.assertGreaterEqual(float(row[token]), float(row.max()) - 0.25, (i, token, int(row.argmax())))

    def test_a_constrained_reply_takes_the_best_token_its_grammar_allows(self):
        # Token t > 0 writes byte t and token 0 ends the reply: Core's grammar matcher masks each row, the sampler
        # applies the mask on the device, and each token is transformers' best among the allowed ones.
        from crit_core.constraints import ByteGrammar, GrammarState, compile_language, from_pieces, matches
        self.engine.constrain(from_pieces([b""] + [bytes([t]) for t in range(1, 256)], 0))
        grammar = compile_language("regex", "[a-h]{3}(-[a-h]{2})*")
        for prompt in self.prompts(2, 11):
            tokens, result = self.engine.generate(prompt, 14, stop_tokens=[0], sampling={"temperature": 0.0},
                                                  constraint={"type": "grammar", "grammar": grammar})
            with torch.no_grad():
                logits = self.reference(torch.tensor([prompt + tokens]).cuda()).logits[0].float()
            state = GrammarState(ByteGrammar.load(grammar))
            for i, token in enumerate(tokens):
                allowed = [t for t in range(256) if (state.complete if t == 0 else state.advance(bytes([t])) is not None)]
                row = logits[len(prompt) - 1 + i]
                self.assertIn(token, allowed)
                self.assertGreaterEqual(float(row[token]), max(float(row[t]) for t in allowed) - 0.25, (i, token))
                state = state.advance(bytes([token])) if token else state
            if result["finish"] == "stop":
                self.assertTrue(matches(grammar, bytes(tokens[:-1])))

    def test_an_engine_holding_more_device_memory_than_its_plan_is_refused(self):
        # The model plugin's factory keeps 512 MiB that no placement plan put there and its memory() does not declare:
        # Core's loader measures each factory and names the plugin.
        from crit_core.engine import load
        from crit_plugins import PluginError
        package = Path(self.temp.name) / "package"
        configuration = {"plugins": {**SELECTION, "options": {"llama-model": {"unplanned_mib": 512}}}}
        with self.assertRaisesRegex(PluginError, r"llama-model: its factory took 512\.\d MiB of device memory beyond "
                                                 r"the placement plans made while it ran, and its memory\(\) declares "
                                                 r"0\.0 MiB"):
            load(package, self.model, configuration)

    def test_the_loader_hands_back_a_loads_cached_temporaries_before_the_engine_opens(self):
        from crit_core.engine import load
        package = Path(self.temp.name) / "package"
        configuration = {"plugins": {**SELECTION, "options": {"llama-model": {"temporary_mib": 256}}}}
        torch.cuda.empty_cache()
        engine = load(package, self.model, configuration)
        try:
            found = engine.telemetry()["load"]
        finally:
            engine.close()
        made, opened = found["made"], found["opened"]
        self.assertGreaterEqual(made["reserved"] - made["allocated"], 256 << 20, "the factory left its temporaries")
        self.assertGreaterEqual(found["released_cached"], 256 << 20)
        self.assertLess(opened["reserved"] - opened["allocated"], 256 << 20, "handed back before the engine opened")
        self.assertEqual(set(found["checked"]), {"allocated", "reserved"})

    def test_the_base_norms_plugin_serves_the_same_greedy_tokens(self):
        # torch-norms, the norms slot's base (what an ablation falls back to), in fused-norms' place. Their sums round
        # differently, so where the two engines part (if they do), the tokens they chose are a near-tie under
        # transformers' logits (BF16's step at these logits, as the greedy test allows); the rest follows from that.
        from crit_core.engine import load
        package = Path(self.temp.name) / "package"
        base = {"plugins": {**SELECTION, "use": {**SELECTION["use"], "norms": "torch-norms"}}}
        engine = load(package, self.model, base)
        try:
            self.assertEqual(engine.info()["plugins"]["norms"], "torch-norms")
            for prompt in self.prompts(2):
                expected, _ = self.engine.generate(prompt, 8, stop_tokens=[], sampling={"temperature": 0.0})
                found, _ = engine.generate(prompt, 8, stop_tokens=[], sampling={"temperature": 0.0})
                self.assertEqual(len(found), len(expected))
                for i, (a, b) in enumerate(zip(found, expected)):
                    if a != b:
                        with torch.no_grad():
                            row = self.reference(torch.tensor([prompt + expected[:i]]).cuda()).logits[0, -1].float()
                        self.assertLess(abs(float(row[a]) - float(row[b])), 0.25, (i, a, b))
                        break
        finally:
            engine.close()

    def test_requests_in_flight_together_decode_what_each_decodes_alone(self):
        prompts = self.prompts(6, 9)
        alone = [self.engine.generate(p, 10, stop_tokens=[], sampling={"temperature": 0.0})[0] for p in prompts]
        with ThreadPoolExecutor(6) as pool:
            together = list(pool.map(lambda p: self.engine.generate(p, 10, stop_tokens=[],
                                                                    sampling={"temperature": 0.0})[0], prompts))
        self.assertEqual(together, alone)


if __name__ == "__main__":
    unittest.main()
