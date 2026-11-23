"""qwen4-exp-model as a whole engine (crit_models.testing): a Qwen4Exp checkpoint against transformers'
log-probabilities and greedy choices of the same checkpoint."""
from pathlib import Path
import unittest

from crit_models import testing as t

HERE = Path(__file__).resolve().parents[1]
torch, transformers, GREEDY = t.torch, t.transformers, t.GREEDY


QWEN4 = dict(vocab_size=256, hidden_size=128, num_hidden_layers=4, num_attention_heads=4, num_key_value_heads=2,
             head_dim=32, linear_num_key_heads=2, linear_num_value_heads=4, linear_key_head_dim=32,
             linear_value_head_dim=32, linear_conv_kernel_dim=4, num_experts=8, num_experts_per_tok=2,
             moe_intermediate_size=64, shared_expert_intermediate_size=64, hc_count=4, hc_lowrank=16,
             ple_layer_ids=[2], ngram_vocab_size_base=48, seed=4321, split_ngram_parts=3, indexer_n_heads=2,
             indexer_kv_heads=1, indexer_head_dim=64, indexer_compress_ratio=4, output_gate_type="sigmoid",
             max_position_embeddings=4096, eos_token_id=2, rms_norm_eps=1e-6, tie_word_embeddings=False,
             rope_parameters={"rope_type": "default", "rope_theta": 10000.0, "partial_rotary_factor": 0.5,
                              "mrope_section": [4, 2, 2], "mrope_interleaved": True})


@unittest.skipIf(torch is None, "torch")
class StreamsTests(t.Layout):
    """The gated-streams layout: a Qwen4Exp checkpoint (four residual streams mixed for every block, gated delta-rule
    layers whose norm gate is a sigmoid, the second with hashed n-gram table rows ahead of it, a gated attention layer
    with an indexer, routed experts with a gated shared expert), its BF16 weights computed by transformers in FP32.
    The indexer's budget is past the engine's context here, so its attention reads every position."""
    plugin = HERE
    library = t.BASE + ["sequential-recurrence", "topk-router", "experts-reference", "expert-tensors"]
    use = {**t.USE, **t.MOE, "recurrence": "sequential-recurrence"}
    plugin_options = {"topk-router": {"score_before_selection": True, "renormalize": True}}
    reference_fp32 = on_average = sharp_routes = True
    budget = 8192

    @classmethod
    def checkpoint(cls, root):
        try:
            config = transformers.AutoConfig.for_model("qwen4_exp_text", indexer_budget=cls.budget, **QWEN4)
        except (KeyError, ValueError):
            return None
        model = transformers.AutoModelForCausalLM.from_config(config)
        with torch.no_grad():
            for name, parameter in model.named_parameters():
                if "norm" in name and name.endswith(".weight") and "linear_attn" not in name:
                    parameter.normal_(0.0, 0.1)   # zero-centred norms scale by 1 + weight: not all ones
                if name.endswith("A_log"):
                    parameter.copy_(torch.empty_like(parameter).uniform_(0.05, 1.0).log())
                if "ple.conv1d" in name:
                    parameter.normal_(0.0, 0.2)   # the reference starts the n-gram rows' convolution at zero
                if "ngram_embedding" in name:
                    parameter.normal_(0.0, 0.5)
        path = cls.save(model, root)
        cls.draft_layer(path, config)
        return path

    @classmethod
    def draft_layer(cls, path, config):
        """A draft layer in the checkpoint's own layout (transformers runs none for this architecture): its decoder
        layer and last mix copies of the attention layer's and the model's, its input maps random."""
        from safetensors.torch import load_file, save_file
        (shard,) = sorted(path.glob("*.safetensors"))
        tensors = load_file(str(shard))
        g = torch.Generator().manual_seed(3)
        last, hidden, streams = config.num_hidden_layers - 1, config.hidden_size, config.hc_count
        for name in [n for n in tensors if n.startswith(f"model.layers.{last}.")]:
            tensors["mtp.layers.0." + name[len(f"model.layers.{last}."):]] = tensors[name].clone()
        for name in [n for n in tensors if n.startswith("model.hyper_connection_mixer.")]:
            tensors["mtp." + name[len("model."):]] = tensors[name].clone()
        for name in ("fc_embedding", "fc_hidden"):
            tensors[f"mtp.{name}.weight"] = (torch.randn(hidden, hidden, generator=g) * 0.05).to(torch.bfloat16)
        tensors["mtp.pre_fc_norm_embedding.weight"] = (torch.randn(hidden, generator=g) * 0.1).to(torch.bfloat16)
        tensors["mtp.pre_fc_norm_hidden.weight"] = (torch.randn(streams * hidden, generator=g) * 0.1).to(torch.bfloat16)
        save_file(tensors, str(shard), metadata={"format": "pt"})

    def test_the_layout_composes_streams_ngram_rows_and_an_indexed_attention(self):
        model = self.engine.model
        self.assertEqual(model.residual.count, 4)
        self.assertEqual([len(layer.before) for layer in model.decoder], [0, 1, 0, 0])
        self.assertEqual(model.decoder[3].mixer.sparse, self.budget // 4 < model.kv_tokens // 4)
        self.assertEqual("index" in model.pools, model.decoder[3].mixer.sparse)
        self.assertIn("ngram", model.state_keys)
        self.assertEqual(model.facts()["block_selection"] is not None, model.decoder[3].mixer.sparse)
        # its chunks' windows of ids come from the request on the host: captured prompt chunks are refused by name
        bucketed = self.load(prompt_buckets="16")
        self.assertIn("NgramRows", bucketed.model.facts()["prompt_graphs"]["problems"][0])
        bucketed.close()
        self.engines.remove(bucketed)

    def test_a_prompt_in_several_steps_and_one_with_end_tokens_decode_as_whole(self):
        # The n-gram rows' ids before a chunk and the block keys' earlier positions come from the sequence: chunks of
        # 16 tokens decode what chunks of 64 decode; an end token inside a prompt stops the n-grams after it.
        (prompt,) = self.prompts(1, 90)
        prompt[40], prompt[41], prompt[77] = 2, 2, 2
        expected, _ = self.engine.generate(prompt, 10, stop_tokens=[], sampling=GREEDY)
        short = self.load(prefill_chunk=16)
        found, _ = short.generate(prompt, 10, stop_tokens=[], sampling=GREEDY)
        self.assertEqual(found, expected)
        logits = self.logits(prompt + expected)
        for i, token in enumerate(expected):
            row = logits[len(prompt) - 1 + i]
            self.assertGreaterEqual(float(row[token]), float(row.max()) - 0.25, (i, token, int(row.argmax())))
        short.close()
        self.engines.remove(short)


# A release's own spelling of the Qwen4Exp checkpoint and its draft layer (other prefixes, its layers numbered from 1,
# other part names), and the names rules a layout serving it would carry to map it back (the test's layout: Respelled).
RELEASE_SPELLING = [["^model\\.layers\\.(\\d+)\\.", "trunk.blocks.\\1.", 1], ["^model\\.", "trunk."],
                    ["^lm_head\\.", "trunk.head."], ["^mtp\\.", "draft."], ["_hyper_connection\\.", "_streams."],
                    ["hyper_connection_mixer\\.", "stream_mix."], ["\\.ple\\.", ".ngram."],
                    ["\\.linear_attn\\.", ".delta."], ["\\.self_attn\\.indexer\\.", ".self_attn.selector."],
                    ["\\.mlp\\.", ".moe."]]
RELEASE_NAMES = [["\\.moe\\.", ".mlp."], ["\\.self_attn\\.selector\\.", ".self_attn.indexer."],
                 ["\\.delta\\.", ".linear_attn."], ["\\.ngram\\.", ".ple."],
                 ["stream_mix\\.", "hyper_connection_mixer."],
                 ["_streams\\.", "_hyper_connection."], ["^draft\\.", "mtp."], ["^trunk\\.head\\.", "lm_head."],
                 ["^trunk\\.blocks\\.(\\d+)\\.", "model.layers.\\1.", -1], ["^trunk\\.", "model."]]


@unittest.skipIf(torch is None, "torch")
class RespelledStreamsTests(t.Respelled, StreamsTests):
    """The Qwen4Exp checkpoint with its draft layer under a release's own tensor names (RELEASE_SPELLING), which names
    rules map back (RELEASE_NAMES): every technique reads its tensors through the layout, so the engine keeps the
    reference."""
    spelling, names = RELEASE_SPELLING, RELEASE_NAMES


if __name__ == "__main__":
    unittest.main()
