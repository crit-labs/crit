"""crit_eval.tensors: one classification of stored tensors for the ceiling and the context limit."""
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
for _folder in ("core/python", "eval"): sys.path.insert(0, str(ROOT / _folder))
from crit_eval.tensors import tensor_role  # noqa: E402


class TensorRoleTests(unittest.TestCase):
    def test_roles_across_checkpoint_layouts(self):
        cases = {
            ("model.layers.0.self_attn.q_proj.weight", ""): ("layer", False),
            ("model.embed_tokens.weight", ""): ("embedding", False),
            ("backbone.embeddings.weight", ""): ("embedding", False),
            ("lm_head.weight", ""): ("head", False),
            ("model.layers.3.mlp.experts.7.down_proj.weight", ""): ("expert", False),
            ("mtp.layers.0.mlp.experts.1.up_proj.weight", ""): ("expert", True),
            ("mtp.fc.weight", ""): ("layer", True),
            ("model.layers.0.input_layernorm.weight", "model-mtp.safetensors"): ("layer", True),
            ("model.visual.blocks.0.attn.qkv.weight", ""): ("other_modality", False),
            ("vision_model.embeddings.patch_embedding.weight", ""): ("other_modality", False),
            ("mlp1.1.weight", ""): ("other_modality", False),
            ("language_model.layers.0.mixer.mlp1.weight", ""): ("layer", False),
        }
        for (name, file_name), expected in cases.items():
            self.assertEqual(tuple(tensor_role(name, file_name)), expected, name)

    def test_gguf_names_and_draft_blocks(self):
        cases = {
            "token_embd.weight": ("embedding", False),
            "per_layer_token_embd.weight": ("embedding", False),
            "output.weight": ("head", False),
            "output_norm.weight": ("layer", False),
            "blk.0.attn_output.weight": ("layer", False),
            "blk.1.ffn_gate_exps.weight": ("expert", False),
            "blk.1.ffn_gate_up_exps.weight": ("expert", False),
            "blk.1.ffn_up_shexp.weight": ("layer", False),
            "blk.2.attn_q.weight": ("layer", True),
            "blk.2.nextn.eh_proj.weight": ("layer", True),
            "blk.12.attn_q.weight": ("layer", False),
        }
        for name, expected in cases.items():
            self.assertEqual(tuple(tensor_role(name, "model.gguf", frozenset({2}))), expected, name)


if __name__ == "__main__":
    unittest.main()
