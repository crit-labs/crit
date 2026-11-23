"""gemma4-model as a whole engine (crit_models.testing): a Gemma 4 checkpoint against transformers'
log-probabilities and greedy choices of the same checkpoint."""
from pathlib import Path
import unittest

from crit_models import testing as t

HERE = Path(__file__).resolve().parents[1]
torch, transformers, GREEDY = t.torch, t.transformers, t.GREEDY


@unittest.skipIf(torch is None, "torch")
class SandwichTests(t.Layout):
    """The sandwich layout: a Gemma 4 checkpoint with its mixture-of-experts block (each block between two plain
    RMSNorms and a scalar a layer, sliding-window layers beside a full layer of another head shape whose values are
    its key projection's rows and whose rotary table turns a quarter of each head, value norms, a GELU-gated MLP
    beside GELU-gated experts routed from the residual stream, scaled embeddings, capped logits), its BF16 weights
    computed by transformers in FP32. The window is shorter than the prompts."""
    plugin = HERE
    library = t.BASE + ["topk-router", "experts-reference", "expert-tensors"]
    use = {**t.USE, **t.MOE}
    plugin_options = {"topk-router": {"score_before_selection": True, "renormalize": True}}
    reference_fp32 = on_average = True
    flips = 0.02   # two experts' scores may tie within BF16's step: a route flips (transformers' own BF16 does it)

    @classmethod
    def checkpoint(cls, root):
        if not hasattr(transformers, "Gemma4ForCausalLM"):
            return None
        config = transformers.Gemma4TextConfig(
            vocab_size=256, vocab_size_per_layer_input=256, hidden_size=128, intermediate_size=256,
            num_hidden_layers=4, num_attention_heads=4, num_key_value_heads=2, head_dim=32, global_head_dim=64,
            num_global_key_value_heads=1, attention_k_eq_v=True, sliding_window=24,
            layer_types=["sliding_attention", "full_attention", "sliding_attention", "full_attention"],
            enable_moe_block=True, num_experts=8, top_k_experts=2, moe_intermediate_size=64,
            hidden_size_per_layer_input=0, max_position_embeddings=2048, rms_norm_eps=1e-6,
            tie_word_embeddings=False, final_logit_softcapping=30.0, pad_token_id=0, eos_token_id=1, bos_token_id=2,
            rope_parameters={"full_attention": {"partial_rotary_factor": 0.25, "rope_theta": 1000000.0,
                                                "rope_type": "proportional"},
                             "sliding_attention": {"rope_theta": 10000.0, "rope_type": "default"}})
        model = transformers.Gemma4ForCausalLM(config)
        g = torch.Generator().manual_seed(5)
        with torch.no_grad():
            for name, parameter in model.named_parameters():
                if name.endswith(("experts.gate_up_proj", "experts.down_proj")):
                    parameter.copy_(torch.randn(parameter.shape, generator=g) * 0.08)
                elif name.endswith("router.proj.weight"):
                    parameter.copy_(torch.randn(parameter.shape, generator=g) * 2.0)   # routes no rounding flips
                elif name.endswith(("q_norm.weight", "k_norm.weight")):
                    # Normed queries and keys at scale 1: weights of 1 would make scores as large as the head is
                    # wide, which BF16 rounds by more than a softmax forgives.
                    parameter.copy_(0.35 + torch.randn(parameter.shape, generator=g) * 0.03)
                elif "post_" in name and name.endswith(".weight"):
                    # A block's output leaves its norm at the norm's weight whatever the block computed: weights
                    # of 1 would add a stream's own size a block, and BF16's rounding of one layer would be the
                    # next layer's input error in full (transformers' own BF16 then leaves its FP32 as far).
                    parameter.copy_(0.1 + torch.randn(parameter.shape, generator=g) * 0.02)
                elif name.endswith(("router.scale", "router.per_expert_scale")) or (
                        "norm" in name and name.endswith(".weight")):
                    parameter.copy_(1.0 + torch.randn(parameter.shape, generator=g) * 0.1)   # not all ones
            for name, buffer in model.named_buffers():
                if name.endswith("layer_scalar"):
                    buffer.fill_(0.5 + 0.25 * int(name.split(".")[-2]))   # a scalar a layer
        return cls.save(model, root)

    def test_the_layout_composes_two_head_shapes_and_experts_beside_the_mlp(self):
        model = self.engine.model
        self.assertEqual((model.activation, model.residual.name), ("gelu_tanh", "sandwich"))
        shapes = [(layer.mixer.kv_heads, layer.mixer.head_dim, layer.mixer.window) for layer in model.decoder]
        self.assertEqual(shapes, [(2, 32, 24), (1, 64, 0), (2, 32, 24), (1, 64, 0)])
        self.assertEqual([layer.mixer.shared_values for layer in model.decoder], [False, True, False, True])
        self.assertEqual(sorted(str(key) for key in model.kv_keys), ["('kv', 24)", "kv"])
        self.assertEqual([type(layer.ffn).__name__ for layer in model.decoder], ["ParallelMoE"] * 4)
        self.assertEqual([round(layer.res[4], 2) for layer in model.decoder], [0.5, 0.75, 1.0, 1.25])
        # the full layer's table is as wide as its head and turns its first quarter
        freqs = model.decoder[1].mixer.rope.freqs.tolist()
        self.assertEqual((len(freqs), sum(1 for f in freqs if f > 0)), (32, 8))


# A release's own spelling of the Gemma 4 checkpoint (another decoder prefix, its layers numbered from 1, other part
# names), and the names rules a layout serving it would carry to map it back (the test's layout: Respelled).
RELEASE_SPELLING = [["^model\\.layers\\.(\\d+)\\.", "lm.layer.\\1.", 1], ["^model\\.", "lm."],
                    ["^lm_head\\.", "lm.logits."], ["\\.self_attn\\.", ".attention."], ["\\.router\\.", ".gating."],
                    ["\\.experts\\.", ".moe."], ["_feedforward_layernorm", "_ffw_norm"],
                    ["\\.layer_scalar$", ".output_scale"]]
RELEASE_NAMES = [["\\.output_scale$", ".layer_scalar"], ["_ffw_norm", "_feedforward_layernorm"],
                 ["\\.moe\\.", ".experts."], ["\\.gating\\.", ".router."], ["\\.attention\\.", ".self_attn."],
                 ["^lm\\.logits\\.", "lm_head."], ["^lm\\.layer\\.(\\d+)\\.", "model.layers.\\1.", -1],
                 ["^lm\\.", "model."]]


@unittest.skipIf(torch is None, "torch")
class RespelledSandwichTests(t.Respelled, SandwichTests):
    """The Gemma 4 checkpoint under a release's own tensor names (RELEASE_SPELLING), which names rules map back
    (RELEASE_NAMES): every technique reads its tensors through the layout, so the engine keeps the reference."""
    spelling, names = RELEASE_SPELLING, RELEASE_NAMES


if __name__ == "__main__":
    unittest.main()
