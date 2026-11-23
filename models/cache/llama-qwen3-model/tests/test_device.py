"""llama-qwen3-model as whole engines (crit_models.testing): a Llama checkpoint, a Qwen3 checkpoint with a
sliding-window layer, a Qwen3-MoE checkpoint through the router and experts slots, and the Llama checkpoint
written as GGUF, each against transformers' log-probabilities and greedy choices of the same checkpoint."""
from pathlib import Path
import unittest

from crit_models import testing as t

HERE = Path(__file__).resolve().parents[1]
torch, transformers, GREEDY = t.torch, t.transformers, t.GREEDY


@unittest.skipIf(torch is None, "torch")
class DecoderTests(t.Layout):
    """The decoder layout: a Llama checkpoint (grouped-query attention, rotary positions, SwiGLU)."""
    plugin = HERE

    @classmethod
    def checkpoint(cls, root):
        config = transformers.LlamaConfig(vocab_size=256, hidden_size=128, intermediate_size=256, num_hidden_layers=2,
                                          num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=2048,
                                          rms_norm_eps=1e-6, tie_word_embeddings=False)
        return cls.save(transformers.LlamaForCausalLM(config), root)

    def test_short_spans_score_as_whole_chunks_and_prefill_in_spans(self):
        # A prefill chunk of QUANTUM tokens: every prompt step is that long, and a text scored in such spans gets
        # the whole chunks' scores within prompt-row rounding (other chunk boundaries, the same arithmetic).
        short = self.load(prefill_chunk=16)
        spans = short.model.spans
        self.assertEqual([spans.rows(p) for p in (0, 100, 900)], [16, 16, 16])
        (ids,) = self.prompts(1, 150)
        self.assertScores(short, ids)
        before = short.telemetry()["core"]["steps"]
        short.generate(ids, 2, stop_tokens=[], sampling=GREEDY)
        self.assertGreaterEqual(short.telemetry()["core"]["steps"] - before, -(-len(ids) // 16))
        short.close()
        self.engines.remove(short)

    def test_a_score_s_slices_of_logits_live_in_the_step_arena_it_states(self):
        # A score's logits are its rows' FP32 [rows][vocabulary] beside their log-softmax and, for top_k, a mask of
        # the entries at or past each row's k-th value (crit_plugins.ranking: no sorted vocabulary row): on a large
        # vocabulary hundreds of megabytes a slice, which no prompt step of its size takes. Core's warm requests
        # score a text of a full chunk with 20 alternatives at load, its arena holds what that took, and a score's
        # segments come out of it: no slice passes the arena.
        from crit_plugins import memory
        model = self.engine.model
        if model.device.type != "cuda":
            self.skipTest("Core allocates a CUDA device's step memory; this engine runs on the CPU")
        device = memory.of(model.device)
        overflows = device.steps()["overflows"]
        (ids,) = self.prompts(1, 300)
        self.engine.score(ids, 1, "prefill", top_k=4)
        steps = device.steps()
        if steps.get("unaccounted"):
            self.skipTest(steps["unaccounted"])
        self.assertGreater(steps["most"].get("score", 0), 0)
        self.assertEqual(steps["overflows"], overflows)
        self.assertGreaterEqual(device.arena, steps["most"]["score"])

    def test_a_request_asking_decode_via_prompt_decodes_as_one_token_prompt_chunks(self):
        # The debug hook crit smoke tells a decode gap's rounding from a fault by (StepRequest.decode_via_prompt): a
        # replay's decode rows multiply as one-token prompt chunks where they would multiply as decode rows, and
        # score what the prompt path scores. (The head multiplies every logits row as a decode row on both paths.)
        model = self.engine.model
        self.assertIs(model.info()["decode_via_prompt"], True)
        groups, matmul = [], model.rows.matmul

        def spy(x, matrix, found, out_dtype=None):
            if matrix is not model.head:
                groups.append(list(found))
            return matmul(x, matrix, found, out_dtype)
        model.rows.matmul = spy
        prompt, continuation = self.prompts(2, 21)
        continuation = continuation[:12]

        def replay(via_prompt):
            rows = []
            groups.clear()
            tokens, _ = self.engine.generate(prompt, len(continuation), stop_tokens=[], replay=continuation,
                                             sampling=GREEDY, logprobs=1, decode_via_prompt=via_prompt,
                                             on_tokens=lambda _tokens, found: rows.extend(found))
            self.assertEqual(tokens, continuation)
            return [logprob for logprob, _ in rows], [g for found in groups for g in found if g[1] - g[0] == 1]
        def rounds():   # decode rounds Core ran in its captured rounds (none where the model has no rounds)
            return ((model.telemetry() or {}).get("drafts") or {}).get("rounds") or 0
        try:
            began = rounds()
            plain, plain_rows = replay(False)
            plain_rounds = rounds() - began
            via, via_rows = replay(True)
            via_rounds = rounds() - began - plain_rounds
        finally:
            model.rows.matmul = matmul
        # Plain decode rows multiply as decode rows: eagerly (the spy sees them), or inside Core's captured rounds,
        # whose replays no Python spy sees (a replayed request decodes there too since rounds mask), never as prompt
        # chunks. Through the hook they are one-token prompt chunks, run eagerly, in no round.
        self.assertFalse(any(prompt for _, _, prompt in plain_rows), plain_rows)
        self.assertTrue(plain_rows or plain_rounds >= len(continuation) - 1, (plain_rows, plain_rounds))
        self.assertTrue(via_rows and all(prompt for _, _, prompt in via_rows), via_rows)
        self.assertEqual(via_rounds, 0)
        scored = self.engine.score(prompt + continuation, len(prompt), "prefill").target
        self.assertLess(max(abs(a - b) for a, b in zip(via, scored)), 0.25, (via, scored))
        self.assertLess(max(abs(a - b) for a, b in zip(via, plain)), 0.25, (via, plain))

    def test_a_selection_without_a_slot_the_layout_calls_is_refused_before_load(self):
        from crit_plugins import PluginError
        from crit_core.engine import load
        selection = self.selection()
        del selection["use"]["embeddings"]
        with self.assertRaisesRegex(PluginError, "llama-qwen3-model requires embeddings"):
            load(self.package, self.model, {"plugins": selection}, device=self.device)


@unittest.skipIf(torch is None, "torch")
class SlidingTests(t.Layout):
    """The decoder layout with sliding-window layers: a Qwen3 checkpoint (query/key norms) whose first layer sees its
    last 24 positions, its prompts longer than the window."""
    plugin = HERE

    @classmethod
    def checkpoint(cls, root):
        if not hasattr(transformers, "Qwen3ForCausalLM"):
            return None
        config = transformers.Qwen3Config(vocab_size=256, hidden_size=128, intermediate_size=256, num_hidden_layers=2,
                                          num_attention_heads=4, num_key_value_heads=2, head_dim=32,
                                          max_position_embeddings=2048, rms_norm_eps=1e-6, tie_word_embeddings=False,
                                          use_sliding_window=True, sliding_window=24, max_window_layers=1,
                                          layer_types=["sliding_attention", "full_attention"])
        model = transformers.Qwen3ForCausalLM(config)
        if list(getattr(model.config, "layer_types", [])) != ["sliding_attention", "full_attention"]:
            return None
        return cls.save(model, root)


@unittest.skipIf(torch is None, "torch")
class GgufDecoderTests(t.GgufEngines, DecoderTests):
    """The decoder layout over a Llama checkpoint written as GGUF: its query and key rows as the converter permutes
    them, every matrix Q8_0, the token embeddings kept as the file's Q8_0 rows (lookup-table-rows decodes the rows a
    lookup reads)."""
    arch = "llama"
    library = t.BASE + ["gguf-projections", "lookup-table-rows"]
    use = {**t.GGUF_USE, "embeddings": "lookup-table-rows"}

    def test_the_matrices_load_as_the_file_stores_them(self):
        found = self.engine.facts
        self.assertIn("Q8_0", found["gguf_formats"], found)
        self.assertEqual(self.engine.model.embed.format, "gguf:Q8_0")   # stored rows, never decoded whole


@unittest.skipIf(torch is None, "torch")
class GgufDenseDecoderTests(t.GgufEngines, DecoderTests):
    """The decoder layout over a Llama checkpoint written as GGUF with some matrices kept F32 (the attention's output
    and the MLP's down projection), as GGUF files keep routers and small gates: gguf-projections loads them dense
    beside the Q8_0 ones, and an engine of the run's preserve mode, which asks exact prompt rows, is taken by Core's
    loader and serves."""
    arch = "llama"
    dense = ("self_attn.o_proj.weight", "mlp.down_proj.weight")
    library = t.BASE + ["gguf-projections", "lookup-table-rows"]
    use = {**t.GGUF_USE, "embeddings": "lookup-table-rows"}

    def test_a_preserve_mode_engine_keeps_the_exact_rows_it_asks(self):
        from crit_core.engine import load
        found = self.engine.facts
        self.assertIn("Q8_0", found["gguf_formats"], found)
        self.assertIn("bf16", found["gguf_formats"], found)   # the F32 matrices, dense
        engine = load(self.package, self.model, {"plugins": self.selection(), "quality_mode": "preserve"},
                      device=self.device)
        self.engines.append(engine)
        facts = engine.facts
        self.assertEqual(facts["row_invariance"]["prompt_exact"], True, facts)
        # Its prompt rows have its decode rows' bits: each greedy reply is the top token its own prefill gives at
        # every position. The engine without a mode multiplies other bits on CUDA (prompt rows cuBLAS's GEMM over
        # the decoded matrices, the exact path's decode rows int8 activations), so its reply may part from this one
        # only where the reference has a near-tie; its scores are the reference's.
        prompts = self.prompts(2, 40)
        for p in prompts:
            reply = engine.generate(p, 6, stop_tokens=[], sampling=GREEDY)[0]
            scored = engine.score(p + reply, 1, "prefill", top_k=1)
            self.assertEqual([scored.top[len(p) - 1 + k][0][0] for k in range(len(reply))], reply)
            other = self.engine.generate(p, 6, stop_tokens=[], sampling=GREEDY)[0]
            if other != reply:
                at = next(k for k, (a, b) in enumerate(zip(reply, other)) if a != b)
                logp = self.logits(p + reply[:at])[len(p) - 1 + at].log_softmax(-1)
                self.assertLess(abs(float(logp[reply[at]]) - float(logp[other[at]])), 0.5, (reply, other))
        self.assertScores(engine, prompts[0])


@unittest.skipIf(torch is None, "torch")
class MoeTests(t.Layout):
    """The decoder layout with routed experts: a Qwen3-MoE checkpoint through the router and experts slots."""
    plugin = HERE
    library = t.BASE + ["topk-router", "experts-reference", "expert-tensors"]
    use = {**t.USE, **t.MOE}
    plugin_options = {"topk-router": {"score_before_selection": True, "renormalize": True}}
    sharp_routes = True
    # Judged on average: on CI's CPU torch the BF16 rounding of one route moves a position 0.37 nats from the
    # reference, beyond the per-position limit, where CUDA's arithmetic keeps every position within it.
    on_average = True

    @classmethod
    def checkpoint(cls, root):
        if not hasattr(transformers, "Qwen3MoeForCausalLM"):
            return None
        config = transformers.Qwen3MoeConfig(vocab_size=256, hidden_size=128, intermediate_size=256,
                                             moe_intermediate_size=64, num_hidden_layers=2, num_attention_heads=4,
                                             num_key_value_heads=2, head_dim=32, num_experts=8, num_experts_per_tok=2,
                                             norm_topk_prob=True, decoder_sparse_step=1, max_position_embeddings=2048,
                                             rms_norm_eps=1e-6, tie_word_embeddings=False)
        return cls.save(transformers.Qwen3MoeForCausalLM(config), root, fp32=True)

    def test_a_router_whose_rule_is_not_the_checkpoints_is_refused(self):
        from crit_core.engine import load
        selection = self.selection()
        selection["options"]["topk-router"] = {"score_before_selection": True, "renormalize": False}
        with self.assertRaisesRegex(Exception, "routes its experts by .*renormalize=True"):
            load(self.package, self.model, {"plugins": selection}, device=self.device)


# A release's own spelling of the Qwen3-MoE checkpoint (another decoder prefix, its layers numbered from 1, other part
# names), and the names rules a layout serving it would carry to map it back (the test's layout: Respelled).
RELEASE_SPELLING = [["^model\\.layers\\.(\\d+)\\.", "decoder.block_\\1.", 1], ["^model\\.", "decoder."],
                    ["^lm_head\\.", "output_head."], ["\\.self_attn\\.", ".attention."],
                    ["\\.mlp\\.gate\\.", ".ffn.router."], ["\\.mlp\\.experts\\.", ".ffn.routed."],
                    ["layernorm", "ln"], ["_proj\\.weight$", "_w"]]
RELEASE_NAMES = [["_w$", "_proj.weight"], ["(?<=_)ln(?=[._])", "layernorm"], ["\\.ffn\\.router\\.", ".mlp.gate."],
                 ["\\.ffn\\.routed\\.", ".mlp.experts."], ["\\.attention\\.", ".self_attn."],
                 ["^output_head\\.", "lm_head."], ["^decoder\\.block_(\\d+)\\.", "model.layers.\\1.", -1],
                 ["^decoder\\.", "model."]]


@unittest.skipIf(torch is None, "torch")
class RespelledMoeTests(t.Respelled, MoeTests):
    """The Qwen3-MoE checkpoint under a release's own tensor names (RELEASE_SPELLING), which names rules map back
    (RELEASE_NAMES): every technique reads its tensors through the layout, so the engine keeps the reference."""
    spelling, names = RELEASE_SPELLING, RELEASE_NAMES


if __name__ == "__main__":
    unittest.main()
