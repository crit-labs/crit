"""Core's prompt cache (crit_plugins.prompt_cache) over the library's paged KV pool on a GPU: the Llama-layout model
plugin (support/device_plugins/llama-model) keeps its ended requests' sequences, and a prompt that begins as one of
them did prefills only the rest and decodes what the same prompt decodes fresh, through step(), in Core's captured
rounds and in drafting rounds. Kept sequences give their pages back before the pool refuses a request one. Needs CUDA,
transformers, safetensors and Core's crit_core_steps (CRIT_CORE_STEP_LIBRARY)."""
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
GREEDY = {"temperature": 0.0}


def configuration(rounds=False, drafter=False, **options):
    use = {**USE, **({"drafter": "chain-drafter"} if drafter else {})}
    return {"plugins": {"use": use, "options": {"llama-model": {"rounds": rounds, "prompt_cache": 4, **options},
                                                "softmax-attention": {"decode": "split"}}}}


@unittest.skipUnless(READY, "a GPU engine: CUDA, transformers, safetensors and CRIT_CORE_STEP_LIBRARY")
class DevicePromptCacheTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from crit.plugins import add
        cls.temp = tempfile.TemporaryDirectory()
        root = Path(cls.temp.name)
        torch.manual_seed(0)
        config = transformers.LlamaConfig(vocab_size=256, hidden_size=128, intermediate_size=256, num_hidden_layers=2,
                                          num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=1024,
                                          rms_norm_eps=1e-6, tie_word_embeddings=False)
        model = transformers.LlamaForCausalLM(config)
        with torch.no_grad():
            model.lm_head.weight.mul_(30)   # sharp logits: a greedy choice no rounding flips
        cls.model = root / "model"
        model.save_pretrained(cls.model, safe_serialization=True)
        cls.package = root / "package"
        add(cls.package / "plugins", LIBRARY)
        for plugin in STAND_INS:
            shutil.copytree(ROOT / "tests" / "support" / "device_plugins" / plugin, cls.package / "plugins" / plugin)
        (cls.package / "configuration.json").write_text(json.dumps(configuration()), encoding="utf-8")
        cls.engines = []

    @classmethod
    def tearDownClass(cls):
        for engine in cls.engines:
            engine.close()
        cls.temp.cleanup()

    def engine(self, **options):
        from crit_core.engine import load
        engine = load(self.package, self.model, configuration(**options))
        self.engines.append(engine)
        return engine

    def prompt(self, length, seed=0):
        g = torch.Generator().manual_seed(1000 * seed + length)
        return torch.randint(3, 256, (length,), generator=g).tolist()

    def ask(self, engine, prompt, output=8, **options):
        tokens, result = engine.generate(prompt, output, stop_tokens=[], sampling=GREEDY, **options)
        return tokens, result["timing"]["reused_prompt_tokens"]

    def conversation(self, engine, kept_reply):
        """Two turns: the second begins as the first did and goes on. Returns what the second reused."""
        first = self.prompt(150)
        reply, reused = self.ask(engine, first, prompt_cache=True)
        self.assertEqual(reused, 0)
        turn = first + reply + self.prompt(21, seed=1)
        fresh, none = self.ask(engine, turn)                       # the same prompt, every token prefilled
        cached, reused = self.ask(engine, turn, prompt_cache=True)
        self.assertEqual(none, 0)
        self.assertEqual(cached, fresh)
        # Another reply in the first one's place shares the first prompt alone, and still answers as fresh.
        other = first + self.prompt(8, seed=2) + self.prompt(21, seed=1)
        fresh, _ = self.ask(engine, other)
        branched, shared = self.ask(engine, other, prompt_cache=True)
        self.assertEqual((branched, shared), (fresh, len(first)))
        self.assertTrue(engine.info()["prompt_cache"])
        return reused, len(first), len(reply)

    def test_a_follow_up_turn_through_step_reuses_the_kept_turn_and_answers_as_fresh(self):
        reused, prompt, reply = self.conversation(self.engine(), True)
        # The pool held the prompt and every output but the last, still pending when the request ended.
        self.assertEqual(reused, prompt + reply - 1)

    def test_a_follow_up_turn_after_captured_rounds_reuses_the_prompt(self):
        engine = self.engine(rounds=True)
        reused, prompt, _ = self.conversation(engine, False)
        # Core's native rounds publish the reply without the model seeing it: the request is kept to its prompt.
        self.assertEqual(reused, prompt)
        self.assertGreater(engine.rounds.runs[0], 0)

    def test_a_follow_up_turn_after_drafting_rounds_reuses_the_prompt(self):
        reused, prompt, _ = self.conversation(self.engine(rounds=True, drafter=True, width=3), False)
        self.assertEqual(reused, prompt)

    def test_requests_at_once_fork_the_kept_sequence_and_decode_what_each_decodes_alone(self):
        engine = self.engine(rounds=True)
        first = self.prompt(120)
        self.ask(engine, first, prompt_cache=True)
        turns = [first + self.prompt(9 + i, seed=3 + i) for i in range(4)]
        alone = [self.ask(engine, turn)[0] for turn in turns]
        with ThreadPoolExecutor(4) as pool:
            together = list(pool.map(lambda turn: self.ask(engine, turn, prompt_cache=True), turns))
        self.assertEqual([tokens for tokens, _ in together], alone)
        self.assertEqual([reused for _, reused in together], [len(first)] * 4)

    def test_kept_sequences_give_their_pages_back_before_a_request_is_refused_one(self):
        # A pool of 2 x 96 positions: each kept sequence holds about 70 of them, so the third request needs pages a
        # kept one has. The pool asks the cache for room, the cache closes the one kept longest ago.
        engine = self.engine(context=96, concurrency=2, prompt_cache=8)
        for seed in range(6):
            prompt = self.prompt(60, seed=10 + seed)
            fresh, _ = self.ask(engine, prompt)
            cached, reused = self.ask(engine, prompt + fresh[:2], prompt_cache=True)
            self.assertEqual(reused, 0 if seed == 0 else reused)   # whatever it reused, it was served
            self.assertEqual(len(cached), 8)
        counted = engine.telemetry()["prompt_cache"]
        self.assertGreater(counted["evicted"], 0)
        self.assertLessEqual(counted["tokens"], 2 * 96)


HYBRID = {"use": {"model": "next-model", "placement": "device-placement", "projections": "projections-dense",
                  "attention": "softmax-attention", "kv_cache": "attention-state", "sampler": "token-sampling",
                  "recurrence": "sequential-recurrence", "router": "topk-router", "experts": "experts-reference",
                  "expert_source": "expert-tensors", "norms": "fused-norms"},
          "base": {"norms": "torch-norms"},
          "options": {"topk-router": {"score_before_selection": True, "renormalize": True},
                      "next-model": {"prompt_cache": 4}}}
HYBRID_LIBRARY = ["attention-state", "softmax-attention", "projections-dense", "token-sampling", "device-placement",
                  "sequential-recurrence", "topk-router", "experts-reference", "expert-tensors", "fused-norms"]


class HybridEngine:
    """A hybrid model (support/device_plugins/next-model) on a small random checkpoint, with OPTIONS over the
    selection's own."""
    OPTIONS = {}

    @classmethod
    def setUpClass(cls):
        from crit.plugins import add
        from crit_core.engine import load
        cls.temp = tempfile.TemporaryDirectory()
        root = Path(cls.temp.name)
        torch.manual_seed(0)
        config = transformers.Qwen3NextConfig(
            vocab_size=256, hidden_size=128, intermediate_size=256, num_hidden_layers=4,
            layer_types=["linear_attention", "linear_attention", "full_attention", "linear_attention"],
            num_attention_heads=4, num_key_value_heads=2, head_dim=32, linear_num_key_heads=2,
            linear_num_value_heads=4, linear_key_head_dim=32, linear_value_head_dim=32, linear_conv_kernel_dim=4,
            num_experts=8, num_experts_per_tok=2, moe_intermediate_size=64, shared_expert_intermediate_size=64,
            decoder_sparse_step=1, norm_topk_prob=True, max_position_embeddings=1024, rms_norm_eps=1e-6,
            tie_word_embeddings=False)
        model = transformers.Qwen3NextForCausalLM(config)
        with torch.no_grad():
            for name, parameter in model.named_parameters():
                if name.endswith("norm.weight") and "linear_attn" not in name:
                    parameter.normal_(0.0, 0.1)
                if name.endswith("A_log"):
                    parameter.copy_(torch.empty_like(parameter).uniform_(0.05, 1.0).log())   # milder decays
            model.lm_head.weight.mul_(30)   # sharp logits: a greedy choice no rounding flips
        model.to(torch.bfloat16)
        cls.model = root / "model"
        model.save_pretrained(cls.model, safe_serialization=True)
        cls.reference = model.float().cuda().eval()   # the stored BF16 weights, computed in FP32
        package = root / "package"
        add(package / "plugins", HYBRID_LIBRARY)
        shutil.copytree(ROOT / "tests" / "support" / "device_plugins" / "next-model", package / "plugins" / "next-model")
        selection = {**HYBRID, "options": {**HYBRID["options"],
                                            "next-model": {**HYBRID["options"]["next-model"], **cls.OPTIONS}}}
        (package / "configuration.json").write_text(json.dumps({"plugins": selection}), encoding="utf-8")
        cls.engine = load(package, cls.model)

    @classmethod
    def tearDownClass(cls):
        cls.engine.close()
        cls.temp.cleanup()

    def prompt(self, length, seed=0):
        g = torch.Generator().manual_seed(1000 * seed + length)
        return torch.randint(3, 256, (length,), generator=g).tolist()

    def ask(self, prompt, output=8, **options):
        tokens, result = self.engine.generate(prompt, output, stop_tokens=[], sampling=GREEDY, **options)
        return tokens, result["timing"]["reused_prompt_tokens"]

    def assertReferenceChoices(self, prompt, tokens):
        """Each token is the FP32 reference's choice at its position, up to BF16 rounding of near-ties."""
        with torch.no_grad():
            logits = self.reference(torch.tensor([prompt + tokens]).cuda()).logits[0].float()
        for i, token in enumerate(tokens):
            row = logits[len(prompt) - 1 + i]
            self.assertGreaterEqual(float(row[token]), float(row.max()) - 0.25, (i, token, int(row.argmax())))


HYBRID_READY = unittest.skipUnless(
    READY and hasattr(transformers, "Qwen3NextForCausalLM"),
    "a GPU engine: CUDA, transformers with Qwen3-Next, safetensors and CRIT_CORE_STEP_LIBRARY")


@HYBRID_READY
class HybridPromptCacheTests(HybridEngine, unittest.TestCase):
    """Its KV is cut at any shared prefix, its recurrent state resumes where it stood, at a kept sequence's end or
    at a prompt chunk's end the model marked."""

    def test_a_follow_up_turn_resumes_the_recurrent_state_where_the_kept_turn_ended(self):
        first = self.prompt(150)   # chunks of 64: the state is marked at 64, 128 and 150, the newest two kept
        reply, reused = self.ask(first, prompt_cache=True)
        self.assertEqual(reused, 0)
        turn = first + reply + self.prompt(21, seed=1)
        cached, reused = self.ask(turn, prompt_cache=True)
        self.assertEqual(reused, len(first) + len(reply) - 1)   # the kept sequence's own end
        self.assertReferenceChoices(turn, cached)
        fresh, none = self.ask(turn)
        self.assertEqual((none, cached), (0, fresh))

    def test_a_turn_that_parts_from_the_kept_one_resumes_at_the_last_mark_it_shares(self):
        first = self.prompt(150, seed=5)
        self.ask(first, prompt_cache=True)
        # Shares the whole first prompt, another reply after it: the mark at the prompt's end.
        other = first + self.prompt(30, seed=6)
        tokens, reused = self.ask(other, prompt_cache=True)
        self.assertEqual(reused, 150)
        self.assertReferenceChoices(other, tokens)
        self.assertEqual(tokens, self.ask(other)[0])
        # Shares 140 tokens: the recurrent state last stood at the chunk boundary before them.
        early = first[:140] + self.prompt(30, seed=7)
        tokens, reused = self.ask(early, prompt_cache=True)
        self.assertEqual(reused, 128)
        self.assertReferenceChoices(early, tokens)
        self.assertEqual(tokens, self.ask(early)[0])
        # Shares 50: no state stood that early, so every token is prefilled.
        self.assertEqual(self.ask(first[:50] + self.prompt(20, seed=8), prompt_cache=True)[1], 0)

    def test_requests_at_once_resume_from_the_same_kept_state(self):
        first = self.prompt(128, seed=9)
        self.ask(first, prompt_cache=True)
        turns = [first + self.prompt(9 + i, seed=20 + i) for i in range(4)]
        alone = [self.ask(turn)[0] for turn in turns]
        with ThreadPoolExecutor(4) as pool:
            together = list(pool.map(lambda turn: self.ask(turn, prompt_cache=True), turns))
        self.assertEqual([tokens for tokens, _ in together], alone)
        self.assertEqual([reused for _, reused in together], [128] * 4)


@HYBRID_READY
class HybridTailTests(HybridEngine, unittest.TestCase):
    """With a tail a prompt chunk ends eight tokens before the prompt's end, through Core's scheduler, and is
    marked: prompts that do not repeat a kept prompt to its last token resume there."""
    OPTIONS = {"prompt_cache_tail": 8}

    def test_the_same_prompt_again_resumes_before_its_tail(self):
        first = self.prompt(150)   # chunks of 64: the state is marked at 64, 128, 142 and 150, the newest two kept
        reply, reused = self.ask(first, prompt_cache=True)
        self.assertEqual(reused, 0)
        self.assertEqual(reply, self.ask(first)[0])           # the split chunk decodes what the whole one does
        self.assertReferenceChoices(first, reply)
        again, reused = self.ask(first, prompt_cache=True)
        self.assertEqual((again, reused), (reply, 142))

    def test_a_turn_that_parts_at_the_kept_prompts_last_token_resumes_before_the_tail(self):
        first = self.prompt(150, seed=3)
        self.ask(first, prompt_cache=True)
        turn = first[:149] + self.prompt(40, seed=4)          # as a template that ends a prompt otherwise
        cached, reused = self.ask(turn, prompt_cache=True)
        self.assertEqual(reused, 142)
        self.assertReferenceChoices(turn, cached)
        self.assertEqual(cached, self.ask(turn)[0])
        # The turn after it extends it, and a branch off the first prompt still finds the place both resumed from.
        later = turn + cached + self.prompt(12, seed=5)
        tokens, reused = self.ask(later, prompt_cache=True)
        self.assertEqual(reused, len(turn) + len(cached) - 1)
        self.assertEqual(tokens, self.ask(later)[0])
        branch = first[:145] + self.prompt(10, seed=6)
        tokens, reused = self.ask(branch, prompt_cache=True)
        self.assertEqual(reused, 142)
        self.assertEqual(tokens, self.ask(branch)[0])

    def test_a_prompt_no_longer_than_the_tail_is_one_chunk(self):
        short = self.prompt(6, seed=7)
        tokens, reused = self.ask(short, prompt_cache=True)
        self.assertEqual((tokens, reused), (self.ask(short)[0], 0))


if __name__ == "__main__":
    unittest.main()
