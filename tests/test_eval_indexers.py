"""crit_eval.indexers: the reference's faster forms of modeling code select what transformers' own loops select."""
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

ROOT = Path(__file__).resolve().parents[1]
for _folder in ("core/python", "eval"): sys.path.insert(0, str(ROOT / _folder))
from crit_eval import indexers  # noqa: E402

try:
    import torch
    from transformers.models.qwen4_exp import modeling_qwen4_exp as qwen4
except ImportError:
    torch = qwen4 = None


class Keys:
    """The indexer's part of a cache: the keys of the chunks before, then this chunk's."""

    def __init__(self, past):
        self.past = past

    def update_indexer(self, keys, layer):
        return torch.cat([self.past, keys], dim=1)


@unittest.skipIf(qwen4 is None, "needs torch and transformers with qwen4_exp")
class QsaIndexerTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        # Sixteen heads: a block scores zero only when every head's ReLU is zero, so exact ties at the top-k edge,
        # which both forms leave to the device's top-k, do not come up.
        config = SimpleNamespace(hidden_size=64, indexer_n_heads=16, indexer_kv_heads=1, indexer_head_dim=16,
                                 indexer_budget=24, indexer_compress_ratio=4, rms_norm_eps=1e-6)
        self.module = qwen4.Qwen4ExpTextQSAIndexer(config, 0)   # six blocks of four tokens per query
        with torch.no_grad():
            for parameter in self.module.parameters():
                parameter.normal_(0, 0.5)

    def inputs(self, length, past=0, batch=2):
        """Hidden states of `length` queries after `past` keys, RoPE over 8 of the 16 dims, a causal bool mask."""
        hidden = torch.randn(batch, length, 64)
        kv = past + length
        angle = torch.arange(kv, dtype=torch.float32)[:, None] * 10000 ** (-torch.arange(4) / 4)[None]
        cos = torch.cat([angle.cos()] * 2, -1)[None].expand(batch, kv, 8).contiguous()
        sin = torch.cat([angle.sin()] * 2, -1)[None].expand(batch, kv, 8).contiguous()
        mask = (torch.arange(kv)[None] <= (past + torch.arange(length))[:, None])[None, None].expand(batch, 1, length, kv)
        cache = Keys(torch.randn(batch, past, 16)) if past else None
        return hidden, (cos, sin), mask.contiguous(), cache

    def same(self, *args):
        with torch.no_grad():
            loop = type(self.module).forward(self.module, *args)
            fast = indexers.qsa_indexer_forward(self.module, *args)
        self.assertEqual(fast.dtype, loop.dtype)
        self.assertTrue(torch.equal(fast, loop))
        return fast

    def test_the_selection_matches_the_loop_for_causal_prompts_and_resumed_chunks(self):
        for length, past in ((3, 0), (23, 0), (200, 0), (64, 150), (131, 37)):
            with self.subTest(length=length, past=past):
                mask = self.same(*self.inputs(length, past))
                # Far enough in, a query keeps its budget of blocks plus its tail, not everything it sees.
                if past + length > 40:
                    self.assertLess(int(mask[0, 0, -1].sum()), past + length)

    def test_a_float_mask_gives_the_loops_float_mask(self):
        hidden, rope, mask, cache = self.inputs(90, 20)
        additive = torch.where(mask, torch.zeros(()), torch.finfo(torch.float32).min)
        self.same(hidden, rope, additive, cache)

    def test_a_mask_that_is_no_prefix_runs_transformers_own_loop(self):
        hidden, rope, mask, cache = self.inputs(40)
        padded = mask.clone()
        padded[1, :, :, :5] = False   # left padding in the second row
        self.same(hidden, rope, padded, cache)

    def test_the_form_binds_only_to_the_source_it_was_checked_against(self):
        self.assertEqual(indexers.source_hash(qwen4.Qwen4ExpTextQSAIndexer.forward),
                         indexers.CHECKED["Qwen4ExpTextQSAIndexer"],
                         "transformers changed the indexer: check the form against it again and update CHECKED")
        holder = torch.nn.ModuleList([self.module])
        self.assertEqual(indexers.exact_forms(holder), ["Qwen4ExpTextQSAIndexer"])
        self.assertEqual(self.module.forward.__func__, indexers.qsa_indexer_forward)


if __name__ == "__main__":
    unittest.main()
