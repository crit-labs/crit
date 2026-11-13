"""Captured rounds of a hybrid model: the Qwen3-Next-layout model plugin (support/device_plugins/next-model) over the
library's device plugins, its gated-delta layers' states and convolution windows in the recurrence pool
(sequential-recurrence) and its attention layers' KV in the KV pool (attention-state), decoding through
crit_plugins.rounds.Rounds with both pools following the round slots, at 1, 2 and 4 rows. Plain rounds (one token) and
drafting rounds (3 drafts from an oracle that drafts the greedy continuation, wrong at one position in five) decode,
captured and eager, token for token what the model decodes through its step path one sequence at a time; a round
truncated to fewer tokens than it emitted leaves each row's recurrent state and window bit for bit those of decoding
the kept tokens one by one, and decoding continues from there as the step path does. The router and experts are
test stand-ins over the library's (round-router, round-experts), whose round forms topk-router and experts-reference
also provide. Needs CUDA,
transformers with Qwen3-Next, safetensors and Core's crit_core_steps (CRIT_CORE_STEP_LIBRARY)."""
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
    READY = (torch.cuda.is_available() and step_library() is not None
             and hasattr(transformers, "Qwen3NextForCausalLM"))
except ImportError:
    READY = False

SELECTION = {"use": {"model": "next-model", "placement": "device-placement", "projections": "projections-dense",
                     "attention": "softmax-attention", "kv_cache": "attention-state", "sampler": "token-sampling",
                     "recurrence": "sequential-recurrence", "router": "round-router", "experts": "round-experts",
                     "expert_source": "expert-tensors", "norms": "fused-norms"},
             "base": {"norms": "torch-norms"},
             # The recurrence pool keeps verify rows of 3 drafts; Qwen3-Next's router: a softmax over every expert,
             # the top ones renormalized; a decode row through the step path reads the KV as a round's column does
             # (softmax-attention's split-KV reader).
             "options": {"next-model": {"widest": 3},
                         "round-router": {"score_before_selection": True, "renormalize": True},
                         "softmax-attention": {"decode": "split"}}}
LIBRARY = ["attention-state", "softmax-attention", "projections-dense", "token-sampling", "device-placement",
           "sequential-recurrence", "topk-router", "experts-reference", "expert-tensors", "fused-norms"]
STAND_INS = ["next-model", "round-router", "round-experts"]
WIDTH = 3
MISS = 5   # the oracle drafts wrong where the draft's position p has p % MISS == 2


class Oracle:
    """A drafter (draft:device) that drafts each row's known greedy continuation from a device table, wrong at the
    positions p with p % MISS == 2, inside a captured round."""

    def __init__(self, rows, length, vocabulary, device):
        self.plan = torch.zeros(rows, length, dtype=torch.int64, device=device)
        self.depths = torch.arange(1, WIDTH + 1, dtype=torch.int64, device=device)
        self.vocabulary, self.length = vocabulary, length

    def draft(self, view, head, sampler):
        at = (view.frontier[:, None] + self.depths[None, :view.width]).clamp(max=self.length - 1)
        tokens = self.plan[:view.rows].gather(1, at)
        view.drafts[:, :view.width] = torch.where(at % MISS == 2, (tokens + 1) % self.vocabulary, tokens)


class Head:
    """The draft head the driver asks of a drafting round; the oracle reads none."""


@unittest.skipUnless(READY, "a GPU engine: CUDA, transformers with Qwen3-Next, safetensors and CRIT_CORE_STEP_LIBRARY")
class HybridRoundTests(unittest.TestCase):
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
                    parameter.normal_(0.0, 0.1)   # Qwen3-Next's norms scale by 1 + weight: not all ones
                if name.endswith("A_log"):
                    parameter.copy_(torch.empty_like(parameter).uniform_(0.05, 1.0).log())   # milder decays
            model.lm_head.weight.mul_(30)   # sharp logits: a greedy choice no rounding flips
        model.to(torch.bfloat16)
        cls.model = root / "model"
        model.save_pretrained(cls.model, safe_serialization=True)
        package = root / "package"
        add(package / "plugins", LIBRARY)
        for plugin in STAND_INS:
            shutil.copytree(ROOT / "tests" / "support" / "device_plugins" / plugin, package / "plugins" / plugin)
        (package / "configuration.json").write_text(json.dumps({"plugins": SELECTION}), encoding="utf-8")
        cls.engine = load(package, cls.model)
        cls.m = cls.engine.model
        cls.sequences = 0

    @classmethod
    def tearDownClass(cls):
        cls.engine.close()
        cls.temp.cleanup()

    # -- the step path, one sequence at a time -------------------------------------------------------------------------
    def open(self):
        """A new sequence in both pools (the state pool holds the model's concurrency + 9 at once: close() each)."""
        type(self).sequences += 1
        sequence = ("test", self.sequences)
        self.m._open(sequence)
        return sequence

    def close(self, *sequences):
        for sequence in sequences:
            self.m._close(sequence)

    def step(self, sequence, frontier, tokens):
        """The step path's row of `tokens` at `frontier`, accepted: the greedy token after it."""
        from crit_plugins import StepBatch
        batch = StepBatch.build([(sequence, frontier, len(tokens))], self.m.device)
        logits = self.m._forward(torch.tensor(tokens, dtype=torch.int64, device=self.m.device), batch,
                                 [len(tokens) - 1])
        self.m.states.accept(sequence, len(tokens))
        self.m.pool.keep(sequence, frontier + len(tokens))
        return int(logits[0].argmax())

    def eager(self, prompt, count, keep=False):
        """The first `count` greedy tokens through the step path (the prompt, then a row a token), and with `keep` the
        sequence too, open: its state covers the prompt and every token but the last."""
        sequence = self.open()
        tokens = [self.step(sequence, 0, prompt)]
        while len(tokens) < count:
            tokens.append(self.step(sequence, len(prompt) + len(tokens) - 1, [tokens[-1]]))
        if keep:
            return sequence, tokens
        self.close(sequence)
        return tokens

    def prompts(self, count):
        g = torch.Generator().manual_seed(count * 31)
        return [torch.randint(3, 256, (9 + 5 * i,), generator=g).tolist() for i in range(count)]

    # -- rounds --------------------------------------------------------------------------------------------------------
    def rounds(self, rows, width, captured, plans=()):
        from crit_plugins.rounds import Rounds
        m = self.m
        oracle = Oracle(rows, 256, m.vocabulary, m.device) if width else None
        rounds = Rounds(m, m.sampler, rows=rows, widest=width, device=m.device, drafter=oracle,
                        head=Head() if width else None, pools=[m.pool, m.states], vocabulary=m.vocabulary)
        for slot, plan in enumerate(plans):
            if oracle is not None:
                oracle.plan[slot, :len(plan)] = torch.tensor(plan, device=m.device)
        if captured:
            for count in range(1, rows + 1):
                rounds.capture(count, width, "greedy")
        return rounds

    def bind(self, rounds, slot, prompt, count, width):
        """A sequence prefilled through the step path and bound to round slot `slot`: (sequence, its tokens so far)."""
        sequence = self.open()
        token = self.step(sequence, 0, prompt)
        rounds.bind(slot, sequence, len(prompt), token, count - 1, len(prompt) + count + width + 2,
                    params={"temperature": 0.0}, tokens=prompt + [token])
        return sequence, [token]

    def assertSameStates(self, sequence, prompt, tokens, slot):
        """The sequence's recurrent state and convolution window, every gated-delta layer, bit for bit those of the
        step path after the prompt and `tokens` greedy tokens (the state covers all but the pending last one)."""
        expected, _ = self.eager(prompt, tokens, keep=True)
        try:
            for layer in self.m.linear:
                self.assertTrue(torch.equal(self.m.states.state(layer, sequence), self.m.states.state(layer, expected)),
                                (slot, layer, "state"))
                self.assertTrue(torch.equal(self.m.states.window(layer, sequence),
                                            self.m.states.window(layer, expected)), (slot, layer, "window"))
        finally:
            self.close(expected)

    def release(self, rounds, bound):
        """Unbind the round slots and close their sequences (once: a cleanup after a failure, or the test's end)."""
        while bound:
            slot = len(bound) - 1
            sequence, _ = bound.pop()
            rounds.unbind(slot)
            self.close(sequence)

    def decode(self, rounds, outputs, count, width):
        """Rounds of every bound row until each has `count` tokens."""
        rows = len(outputs)
        for _ in range(4 * count):
            if all(len(found) >= count for found in outputs):
                return
            rounds.launch(rows, width)
            for slot, found in enumerate(rounds.read()):
                outputs[slot] += found["tokens"]
        self.fail(f"the rounds stopped short of {count} tokens: {[len(found) for found in outputs]}")

    def test_rounds_decode_what_the_step_path_decodes_batched_as_alone_captured_and_eager(self):
        count = 16
        prompts = self.prompts(4)
        reference = [self.eager(p, count) for p in prompts]
        for rows in (1, 2, 4):
            for width in (0, WIDTH):
                for captured in (False, True):
                    with self.subTest(rows=rows, width=width, captured=captured):
                        plans = [prompts[s] + reference[s] for s in range(rows)]
                        rounds = self.rounds(rows, width, captured, plans)
                        bound = [self.bind(rounds, s, prompts[s], count, width) for s in range(rows)]
                        try:
                            outputs = [tokens for _, tokens in bound]
                            self.decode(rounds, outputs, count, width)
                            for slot, (sequence, _) in enumerate(bound):
                                self.assertEqual(outputs[slot][:count], reference[slot], (slot, rows, width, captured))
                                self.assertSameStates(sequence, prompts[slot], len(outputs[slot]), slot)
                        finally:
                            self.release(rounds, bound)

    def test_a_truncated_round_leaves_the_states_of_the_kept_tokens(self):
        rows, count = 4, 24
        prompts = self.prompts(rows)
        reference = [self.eager(p, count) for p in prompts]
        for captured in (False, True):
            with self.subTest(captured=captured):
                rounds = self.rounds(rows, WIDTH, captured, [prompts[s] + reference[s] for s in range(rows)])
                bound = [self.bind(rounds, s, prompts[s], count, WIDTH) for s in range(rows)]
                try:
                    self.truncating(rounds, bound, prompts, reference, count)
                finally:
                    self.release(rounds, bound)

    def truncating(self, rounds, bound, prompts, reference, count):
        """Rounds of every bound row, three of them truncated: none, one or all but one of a row's tokens kept."""
        rows = len(bound)
        outputs = [tokens for _, tokens in bound]
        truncated = 0
        for _ in range(3 * count):
            if min(len(found) for found in outputs) >= count - WIDTH - 1:
                break
            rounds.launch(rows, WIDTH)
            found = rounds.read()
            emitted = [len(f["tokens"]) for f in found]
            if max(emitted) > 1 and truncated < 3:
                # Keep fewer than emitted on every row that emitted two or more: none, one, or all but one.
                keep = [[0, 1, n - 1][(slot + truncated) % 3] if n > 1 else n for slot, n in enumerate(emitted)]
                rounds.truncate(keep)
                found = [{**f, "tokens": f["tokens"][:k]} for f, k in zip(found, keep)]
                truncated += 1
                for slot, (sequence, _) in enumerate(bound):
                    outputs[slot] += found[slot]["tokens"]
                    self.assertSameStates(sequence, prompts[slot], len(outputs[slot]), slot)
                continue
            for slot in range(rows):
                outputs[slot] += found[slot]["tokens"]
        self.assertEqual(truncated, 3)
        for slot, (sequence, _) in enumerate(bound):   # decoding went on as the step path does
            self.assertEqual(outputs[slot], reference[slot][:len(outputs[slot])], slot)
            self.assertSameStates(sequence, prompts[slot], len(outputs[slot]), slot)


if __name__ == "__main__":
    unittest.main()
