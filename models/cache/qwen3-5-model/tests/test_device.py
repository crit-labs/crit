"""qwen3-5-model as whole engines (crit_models.testing): a Qwen3-Next checkpoint and a Qwen3.5-MoE text
checkpoint (gated delta-rule layers, gated attention, routed experts with a gated shared expert), each against
transformers' log-probabilities and greedy choices of the same checkpoint."""
from pathlib import Path
import unittest

from crit_models import testing as t

HERE = Path(__file__).resolve().parents[1]
torch, transformers, GREEDY = t.torch, t.transformers, t.GREEDY


def hybrid_weights(model):
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if name.endswith("norm.weight") and "linear_attn" not in name:
                parameter.normal_(0.0, 0.1)   # zero-centred norms scale by 1 + weight: not all ones
            if name.endswith("A_log"):
                parameter.copy_(torch.empty_like(parameter).uniform_(0.05, 1.0).log())   # milder decays


@unittest.skipIf(torch is None, "torch")
class HybridMoeTests(t.Layout):
    """The gated-hybrid layout with routed experts: a Qwen3-Next checkpoint (gated delta-rule layers, gated attention,
    a gated shared expert), its BF16 weights computed by transformers in FP32."""
    plugin = HERE
    library = t.BASE + ["sequential-recurrence", "topk-router", "experts-reference", "expert-tensors"]
    use = {**t.USE, **t.MOE, "recurrence": "sequential-recurrence"}
    plugin_options = {"topk-router": {"score_before_selection": True, "renormalize": True}}
    reference_fp32 = on_average = True

    @classmethod
    def checkpoint(cls, root):
        if not hasattr(transformers, "Qwen3NextForCausalLM"):
            return None
        config = transformers.Qwen3NextConfig(
            vocab_size=256, hidden_size=128, intermediate_size=256, num_hidden_layers=4,
            layer_types=["linear_attention", "linear_attention", "full_attention", "linear_attention"],
            num_attention_heads=4, num_key_value_heads=2, head_dim=32, linear_num_key_heads=2,
            linear_num_value_heads=4, linear_key_head_dim=32, linear_value_head_dim=32, linear_conv_kernel_dim=4,
            num_experts=8, num_experts_per_tok=2, moe_intermediate_size=64, shared_expert_intermediate_size=64,
            decoder_sparse_step=1, norm_topk_prob=True, max_position_embeddings=2048, rms_norm_eps=1e-6,
            tie_word_embeddings=False)
        model = transformers.Qwen3NextForCausalLM(config)
        hybrid_weights(model)
        return cls.save(model, root)


def hybrid_moe(root, cls):
    """A Qwen3.5-MoE text checkpoint (gated delta-rule layers, a gated attention layer, routed experts with a gated
    shared expert; its router always renormalizes the kept probabilities, which its config has no key for)."""
    if not hasattr(transformers, "Qwen3_5MoeForCausalLM"):
        return None
    config = transformers.Qwen3_5MoeTextConfig(
        vocab_size=256, hidden_size=128, num_hidden_layers=4,
        layer_types=["linear_attention", "linear_attention", "linear_attention", "full_attention"],
        num_attention_heads=4, num_key_value_heads=2, head_dim=32, linear_num_key_heads=2, linear_num_value_heads=4,
        linear_key_head_dim=32, linear_value_head_dim=32, linear_conv_kernel_dim=4, num_experts=8,
        num_experts_per_tok=2, moe_intermediate_size=64, shared_expert_intermediate_size=64,
        max_position_embeddings=2048, rms_norm_eps=1e-6, tie_word_embeddings=False)
    model = transformers.Qwen3_5MoeForCausalLM(config)
    hybrid_weights(model)
    return cls.save(model, root)


@unittest.skipIf(torch is None, "torch")
class HybridRoutedTests(t.Layout):
    """The gated-hybrid layout on Qwen3.5-MoE, whose config does not say that its router renormalizes: the layout's
    defaults do, and the router's rule is checked against them."""
    plugin = HERE
    library = t.BASE + ["sequential-recurrence", "topk-router", "experts-reference", "expert-tensors"]
    use = {**t.USE, **t.MOE, "recurrence": "sequential-recurrence"}
    plugin_options = {"topk-router": {"score_before_selection": True, "renormalize": True}}
    reference_fp32 = on_average = sharp_routes = True

    @classmethod
    def checkpoint(cls, root):
        return hybrid_moe(root, cls)


# A release's own spelling of the Qwen3.5-MoE checkpoint (another decoder prefix, its layers numbered from 1, other
# part names), and the names rules a layout serving it would carry to map it back (the test's layout: Respelled).
RELEASE_SPELLING = [["^model\\.layers\\.(\\d+)\\.", "body.layer\\1.", 1], ["^model\\.", "body."],
                    ["^lm_head\\.", "body.unembed."], ["\\.linear_attn\\.", ".delta."], ["\\.self_attn\\.", ".attn."],
                    ["\\.mlp\\.shared_expert", ".moe.common_expert"], ["\\.mlp\\.", ".moe."],
                    ["\\.A_log$", ".log_decay"], ["(input|post_attention)_layernorm", "\\1_norm"]]
RELEASE_NAMES = [["(input|post_attention)_norm", "\\1_layernorm"], ["\\.log_decay$", ".A_log"],
                 ["\\.moe\\.common_expert", ".mlp.shared_expert"], ["\\.moe\\.", ".mlp."],
                 ["\\.attn\\.", ".self_attn."],
                 ["\\.delta\\.", ".linear_attn."], ["^body\\.unembed\\.", "lm_head."],
                 ["^body\\.layer(\\d+)\\.", "model.layers.\\1.", -1], ["^body\\.", "model."]]


@unittest.skipIf(torch is None, "torch")
class RespelledHybridRoutedTests(t.Respelled, HybridRoutedTests):
    """The Qwen3.5-MoE checkpoint under a release's own tensor names (RELEASE_SPELLING), which names rules map back
    (RELEASE_NAMES): every technique reads its tensors through the layout, so the engine keeps the reference."""
    spelling, names = RELEASE_SPELLING, RELEASE_NAMES


if __name__ == "__main__":
    unittest.main()
