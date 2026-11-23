"""inkling2-loop-model as a whole engine (crit_models.testing): an Inkling 2 loop checkpoint against
transformers' log-probabilities and greedy choices of the same checkpoint."""
import json
from pathlib import Path
import unittest

from crit_models import testing as t

HERE = Path(__file__).resolve().parents[1]
torch, transformers, GREEDY = t.torch, t.transformers, t.GREEDY


LOOPED = dict(vocab_size=288, unpadded_vocab_size=272, hidden_size=64, num_hidden_layers=6, num_attention_heads=4,
              num_key_value_heads=2, head_dim=16, swa_num_attention_heads=4, swa_num_key_value_heads=2, swa_head_dim=16,
              sliding_window_size=8, d_rel=4, rel_extent=8, log_scaling_n_floor=4, qk_l2_clip_threshold=2.0,
              rel_l2_clip_threshold=1.0, n_layers_per_attn_res_block=2, moe_intermediate_size=64, n_routed_experts=8,
              num_experts_per_tok=2, loop_layer_index_start=2, loop_layer_index_end=4, loop_num_passes=3,
              loop_gate_rank=8, eos_token_id=2, pad_token_id=0,
              layer_types=["hybrid_sliding", "hybrid", "hybrid_sliding", "hybrid", "hybrid_sliding", "hybrid"])


@unittest.skipIf(torch is None, "torch")
class LoopedTests(t.Layout):
    """The looped-blocks layout: an Inkling 2 loop checkpoint (its middle span of layers run three times, the later
    passes taking messages from the pass before; attention residuals over blocks of two layers; sliding and full layers
    of relative attention with learned sinks a token, short convolutions over keys and values, dynamic ones in the full
    layers, and after the attention and the FFN; experts routed by sigmoid scores with a correction bias, gated by a
    capped SwiGLU; a head of fewer rows than the vocabulary), its BF16 weights computed by transformers in FP32. The
    window and the bias's reach are 8 positions and the log scaling starts at position 4, so the prompts here cross
    all three."""
    plugin = HERE
    library = t.BASE + ["sequential-recurrence", "topk-router", "experts-reference", "expert-tensors"]
    use = {**t.USE, **t.MOE, "recurrence": "sequential-recurrence"}
    plugin_options = {"topk-router": {"scoring": "sigmoid", "score_before_selection": True, "renormalize": True,
                                      "scaling": 8.0}}
    reference_fp32 = on_average = sharp_routes = True
    flips = 0.02   # sigmoid scores near 1 under the sharpened router may tie within BF16's step: a route flips

    @classmethod
    def checkpoint(cls, root):
        try:
            from transformers.models.inkling2_loop.configuration_inkling2_loop import Inkling2LoopTextConfig
            from transformers.models.inkling2_loop.modeling_inkling2_loop import Inkling2LoopForCausalLM
        except ImportError:
            return None
        model = Inkling2LoopForCausalLM(Inkling2LoopTextConfig(**LOOPED))
        with torch.no_grad():   # the parts the reference starts at zero or one, random so each one counts
            for name, parameter in model.named_parameters():
                if name.endswith((".query", "static_weight", "e_score_correction_bias")):
                    parameter.normal_(0.0, 0.5)
                elif name.endswith(".scale"):
                    parameter.copy_(1.0 + 0.3 * torch.randn_like(parameter))
                elif "conv" in name and name.endswith("conv1d.weight"):
                    parameter.normal_(0.0, 0.3)
            for name, buffer in model.named_buffers():
                if name.endswith("e_score_correction_bias"):
                    buffer.normal_(0.0, 0.05)
        return cls.save(model, root)

    def test_each_pass_of_a_looped_layer_keeps_its_own_kv_and_windows_and_shares_its_weights(self):
        model = self.engine.model
        self.assertEqual(model.sources, (0, 1, 2, 3, 2, 3, 2, 3, 4, 5))
        self.assertEqual(model.layers, 10)
        first, again = model.decoder[2].mixer, model.decoder[4].mixer
        self.assertIs(first.query_rows[0], again.query_rows[0])   # one checkpoint layer's matrices, loaded once
        self.assertEqual(sorted(model.pools[("kv", 8)].layers), [0, 2, 4, 6, 8])
        self.assertEqual(sorted(model.pools["kv"].layers), [1, 3, 5, 7, 9])
        self.assertEqual(model.expert_ids[2], model.expert_ids[4])


# transformers saves the looped checkpoint under the release's own names (its conversion mapping, reversed): the
# layout's names rules map those to the names the techniques read. The same rules, applied to the saved names, store
# it under transformers' own module names (a wrapper's prefix, model.language_model.), which the rules leave as they are.
TRANSFORMERS_NAMES = json.loads((HERE / "layout.json").read_text(encoding="utf-8"))["names"]


@unittest.skipIf(torch is None, "torch")
class TransformersSpellingTests(t.Respelled, LoopedTests):
    """The same checkpoint under transformers' own module names (TRANSFORMERS_NAMES): the techniques read every tensor
    through the layout, whichever of the two spellings the checkpoint stores, so the engine keeps the reference."""
    spelling, files, every_rule = TRANSFORMERS_NAMES, "model*.safetensors", False


if __name__ == "__main__":
    unittest.main()
