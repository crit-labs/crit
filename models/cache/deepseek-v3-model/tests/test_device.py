"""deepseek-v3-model as a whole engine (crit_models.testing): a DeepSeek-V3 checkpoint against transformers'
log-probabilities and greedy choices of the same checkpoint."""
from pathlib import Path
import unittest

from crit_models import testing as t

HERE = Path(__file__).resolve().parents[1]
torch, transformers, GREEDY = t.torch, t.transformers, t.GREEDY


@unittest.skipIf(torch is None, "torch")
class LatentTests(t.Layout):
    """The latent layout: a DeepSeek V3 checkpoint (multi-head latent attention with low-rank queries and interleaved
    rotary pairs, a dense MLP on its first layer, then experts routed by sigmoid scores within the best group, with a
    shared expert), its BF16 weights computed by transformers in FP32."""
    plugin = HERE
    library = t.BASE + ["topk-router", "experts-reference", "expert-tensors"]
    use = {**t.USE, **t.MOE}
    plugin_options = {"topk-router": {"scoring": "sigmoid", "score_before_selection": True, "renormalize": True,
                                      "scaling": 2.5, "groups": 2, "selected_groups": 1, "group_score_top": 2}}
    reference_fp32 = on_average = sharp_routes = True
    flips = 0.02   # two groups' scores, or two sigmoid scores near 1, may tie within BF16's step: a route flips

    @classmethod
    def checkpoint(cls, root):
        if not hasattr(transformers, "DeepseekV3ForCausalLM"):
            return None
        config = transformers.DeepseekV3Config(
            vocab_size=256, hidden_size=128, intermediate_size=256, moe_intermediate_size=64, num_hidden_layers=3,
            num_attention_heads=4, num_key_value_heads=4, n_shared_experts=1, n_routed_experts=8,
            routed_scaling_factor=2.5, kv_lora_rank=32, q_lora_rank=48, qk_rope_head_dim=16, v_head_dim=32,
            qk_nope_head_dim=32, n_group=2, topk_group=1, num_experts_per_tok=2, first_k_dense_replace=1,
            norm_topk_prob=True, max_position_embeddings=2048, rope_interleave=True, rms_norm_eps=1e-6,
            tie_word_embeddings=False)
        model = transformers.DeepseekV3ForCausalLM(config)
        with torch.no_grad():
            for name, buffer in model.named_buffers():
                if name.endswith("e_score_correction_bias"):
                    buffer.normal_(0.0, 0.05)
        return cls.save(model, root)

    def test_the_layers_keep_one_latent_row_a_position_and_no_kv(self):
        model = self.engine.model
        self.assertEqual(list(model.pools), ["latent"])
        pool = model.pools["latent"]
        self.assertEqual((pool.kv_heads, pool.head_dim, pool.value_dim), (1, 32 + 16, 0))


# A release's own spelling of the DeepSeek V3 checkpoint (another decoder prefix, its layers numbered from 1, other
# part names), and the names rules a layout serving it would carry to map it back (the test's layout: Respelled).
RELEASE_SPELLING = [["^model\\.layers\\.(\\d+)\\.", "transformer.h.\\1.", 1], ["^model\\.", "transformer."],
                    ["^lm_head\\.", "transformer.out."], ["\\.self_attn\\.", ".mla."],
                    ["kv_a_proj_with_mqa", "kv_down_proj"], ["\\.mlp\\.shared_experts\\.", ".mlp.common."],
                    ["e_score_correction_bias$", "route_bias"], ["\\.mlp\\.", ".ffn."],
                    ["(q_a|kv_a|input|post_attention)_layernorm", "\\1_norm"]]
RELEASE_NAMES = [["(q_a|kv_a|input|post_attention)_norm\\.", "\\1_layernorm."], ["\\.ffn\\.", ".mlp."],
                 ["route_bias$", "e_score_correction_bias"], ["\\.mlp\\.common\\.", ".mlp.shared_experts."],
                 ["kv_down_proj", "kv_a_proj_with_mqa"], ["\\.mla\\.", ".self_attn."],
                 ["^transformer\\.out\\.", "lm_head."], ["^transformer\\.h\\.(\\d+)\\.", "model.layers.\\1.", -1],
                 ["^transformer\\.", "model."]]


@unittest.skipIf(torch is None, "torch")
class RespelledLatentTests(t.Respelled, LatentTests):
    """The DeepSeek V3 checkpoint under a release's own tensor names (RELEASE_SPELLING), which names rules map back
    (RELEASE_NAMES): every technique reads its tensors through the layout, so the engine keeps the reference."""
    spelling, names = RELEASE_SPELLING, RELEASE_NAMES


if __name__ == "__main__":
    unittest.main()
