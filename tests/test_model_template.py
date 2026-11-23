"""The finished model plugins (models/cache) as whole engines beyond their own tests, on the harness those use
(crit_models.testing: a small random checkpoint of a transformers architecture, the model plugin beside library
plugins in a package, run by Core's loader, on CUDA or where there is none on the CPU): each plugin's own class
varied over other library plugins. Qwen3-MoE over expert-offload in its device, host-memory and file tiers; Qwen3.5
with an MTP layer whose drafted greedy replies are the plain ones, and its experts in the host tier; Qwen4Exp's
sparse attention, off-device tables, published shapes and echo drafts; Gemma 4 and the looped decoder over the
expert tiers; every architecture written as GGUF; and captured prompt chunks over quantized-projections decoding as
eager ones. They live here, not in the plugins' tests, so that `crit plugins add` of a model plugin copies only the
plain plugins its own tests use."""
from concurrent.futures import ThreadPoolExecutor
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
for _folder in ("inference/core/python", "cli", "kernels"):   # crit_kernels: the exactness levels Core holds an engine to
    sys.path.insert(0, str(ROOT / _folder))

from crit_models import testing as t  # noqa: E402


def _plugin_tests(plugin):
    """A finished model plugin's own engine tests (models/cache/ID/tests/test_device.py), whose classes these vary."""
    spec = importlib.util.spec_from_file_location(f"crit_model_tests_{plugin.replace('-', '_')}",
                                                  ROOT / "models" / "cache" / plugin / "tests" / "test_device.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


llama, q35, deepseek = (_plugin_tests(name) for name in ("llama-qwen3-model", "qwen3-5-model", "deepseek-v3-model"))
q4, gemma, inkling = (_plugin_tests(name) for name in ("qwen4-exp-model", "gemma4-model", "inkling2-loop-model"))
torch, transformers, GREEDY = t.torch, t.transformers, t.GREEDY


@unittest.skipIf(torch is None, "torch")
class PromptBucketTests(t.Engines):
    """The decoder layout over quantized-projections, whose prompt form runs inside a captured graph."""
    plugin = llama.HERE
    library = t.BASE + ["quantized-projections"]
    devices = ("cuda",)

    @classmethod
    def checkpoint(cls, root):
        return llama.DecoderTests.checkpoint.__func__(cls, root)

    def test_captured_prompt_chunks_decode_what_eager_chunks_decode(self):
        # Prompt buckets over a projections plugin whose prompt form captures (prompts:projections): a chunk that fits
        # a bucket replays its captured graph (padding changes no state), and greedy replies are the eager engine's.
        # projections-dense's prompt form does not capture: its engine names why and runs every chunk eager.
        self.assertIn("prompts:projections", " ".join(self.load(prompt_buckets="16").model.prompts.problems))
        use = {"projections": "quantized-projections"}
        eager, graphed = self.load(use), self.load(use, prompt_buckets="16,64")
        self.assertEqual(graphed.model.prompts.problems, [])
        self.assertEqual(sorted(graphed.model.prompts.graphs), [16, 64])
        for prompt in self.prompts(3, 23):
            expected, _ = eager.generate(prompt, 8, stop_tokens=[], sampling=GREEDY)
            found, _ = graphed.generate(prompt, 8, stop_tokens=[], sampling=GREEDY)
            self.assertEqual(found, expected)
        self.assertGreater(graphed.model.prompts.counts["graph_chunks"], 0)


@unittest.skipIf(torch is None, "torch")
class MoeOffloadTests(llama.MoeTests):
    """The same engine with expert-offload in the experts slot: the model plans no device bytes for the routed experts,
    begins the plugin before the loads, opens it after its pools and calls it with each step's decode rows and prompt
    chunks apart. The tier is the device's here (the experts fit), so the test takes no more of a shared GPU."""
    library = t.BASE + ["topk-router", "expert-offload", "expert-tensors"]
    use = {**t.USE, **t.MOE, "experts": "expert-offload"}
    devices = ("cuda",)
    plugin_options = {**llama.MoeTests.plugin_options, "expert-offload": {"tier": "device", "misses": "link"}}
    on_average = True   # its device rows are grouped-experts' BF16 tensor arithmetic, not the reference's FP64 sums

    def test_the_experts_plugin_placed_the_experts(self):
        facts = self.engine.facts
        self.assertEqual(facts["expert_offload"]["tier"], "device")
        self.assertTrue(self.engine.model.offload)


@unittest.skipIf(torch is None, "torch")
class MoeHostTierTests(MoeOffloadTests):
    """The same engine with the experts in host memory (expert-offload's ram tier) behind a device cache that holds
    half of them: decode rows plan through the cache in captured rounds, each round counting its expert reads and
    hits, and prompt chunks stage their misses through the cache's ring."""
    plugin_options = {**llama.MoeTests.plugin_options,
                      "expert-offload": {"tier": "ram", "misses": "link", "settings": "offload.json"}}

    @classmethod
    def checkpoint(cls, root):
        (root / "package").mkdir(parents=True, exist_ok=True)
        (root / "package" / "offload.json").write_text('{"options": {"expert-offload": {"slots": 16}}}',
                                                       encoding="utf-8")
        return super().checkpoint(root)

    def test_the_experts_plugin_placed_the_experts(self):
        facts = self.engine.facts["expert_offload"]
        self.assertEqual((facts["tier"], facts["misses"], facts["slots"]), ("ram", "link", 16))
        self.assertEqual(self.engine.model.facts()["rounds"]["problems"], [])

    def test_rounds_count_each_requests_expert_reads_and_hits(self):
        (prompt,) = self.prompts(1, 21)
        tokens, facts = self.engine.generate(prompt, 12, stop_tokens=[], sampling=GREEDY)
        self.assertGreater(facts["expert_reads"], 0, facts)
        self.assertLessEqual(facts["expert_hits"], facts["expert_reads"])

    def test_a_prompts_spans_bring_each_layers_experts_to_the_device_once(self):
        # The experts plugin says a prompt chunk brings its layers' experts to the device (stages_prompts), so the
        # model runs a prompt's spans layer by layer without being told to: a layer's experts cross the link once
        # for the span, where chunk by chunk every chunk brings every layer's again, and the reply is the same.
        passes = self.engine.model.facts()["passes"]
        self.assertTrue(passes["on"], passes)
        self.assertIn("experts plugin", passes["reason"])
        plain = self.load(passes="off")
        (prompt,) = self.prompts(1, 300)   # five chunks of 64 tokens before the place the prompt cache marks

        def served(engine):
            copies = engine.telemetry()["expert_offload"]["expert_cache"]["prompt_copies"]
            tokens, _ = engine.generate(prompt, 6, stop_tokens=[], sampling=GREEDY)
            return tokens, engine.telemetry()["expert_offload"]["expert_cache"]["prompt_copies"] - copies
        (tokens, copies), (want, most) = served(self.engine), served(plain)
        self.assertEqual(tokens, want)
        self.assertGreater(copies, 0)
        self.assertLess(2 * copies, most, (copies, most))
        plain.close()
        self.engines.remove(plain)


@unittest.skipIf(torch is None, "torch")
class GgufMoeHostTierTests(t.GgufEngines, MoeHostTierTests):
    """The host-memory engine on the Qwen3-MoE checkpoint written as GGUF (architecture qwen3moe): its experts reach
    expert-offload as the file stores them (experts:gguf), are cached and staged as their stored bytes and computed by
    gguf-rows in their slots, every miss over the link; the other matrices through gguf-projections."""
    arch = "qwen3moe"
    library = MoeHostTierTests.library + ["gguf-projections"]
    use = {**MoeHostTierTests.use, "projections": "gguf-projections"}
    # gguf-rows multiplies the experts' stored rows by int8 activations (a scale a 32 values): a near-tie greedy choice
    # may round the other way than the reference's FP32 activations
    flips = 0.05

    def test_the_experts_reach_the_plugin_as_stored(self):
        facts = self.engine.facts["expert_offload"]
        self.assertEqual(facts["format"], "gguf")
        self.assertTrue(self.engine.model.matrices.stored_experts)


def _nvfp4_experts(path, input_scale=0.004, weight_only=()):
    """The checkpoint's routed experts stored again as NVFP4 in ModelOpt's layout (a matrix per expert: packed E2M1
    codes, E4M3 scales a block of 16, one weight scale) with a static input scale each, so their layers are W4A4: the
    format expert-offload's CPU tier computes. weight_only: the name prefixes whose experts get no input scale
    (W4A16), as published checkpoints store a draft layer's."""
    from safetensors.torch import load_file, save_file
    sys.path.insert(0, str(ROOT / "tests"))
    from nvfp4_checkpoint import nvfp4
    (shard,) = sorted(path.glob("*.safetensors"))
    tensors = load_file(str(shard))
    stored = [n for n in tensors if ".mlp.experts." in n and n.endswith("_proj.weight")]
    if not stored:
        raise unittest.SkipTest("the checkpoint stores its experts in another layout than a matrix per expert")
    for name in stored:
        found, at = nvfp4(tensors.pop(name)), name[:-len("weight")]
        tensors[at + "weight"] = found["weight_packed"]
        tensors[at + "weight_scale"] = found["weight_scale"]
        tensors[at + "weight_scale_2"] = (1.0 / found["weight_global_scale"]).reshape(())
        if not name.startswith(tuple(weight_only)):
            tensors[at + "input_scale"] = torch.tensor(input_scale)
    save_file(tensors, str(shard), metadata={"format": "pt"})


@unittest.skipIf(torch is None, "torch")
class OffloadTierTests(t.Engines):
    """The decoder layout over expert-offload on a checkpoint whose experts are NVFP4 W4A4: every expert on the device
    (the class's engine), then the same package with the experts in host memory behind a device cache that holds half
    of them, their misses over the link or on the CPU tier, and in a file. Each engine loads through Core's loader
    (the loads, the tier's store, the cache, the CPU tier over the store, captured rounds over the cache), decodes in
    rounds that count each request's expert reads and hits, and answers as the engine with every expert on the device
    does: bit for bit where the device computes every route in the same arithmetic (misses over the link, a file),
    within rounding where the CPU's order serves decode rows."""
    plugin = llama.HERE
    library = t.BASE + ["topk-router", "expert-offload", "expert-tensors"]
    use = {**t.USE, **t.MOE, "experts": "expert-offload"}
    devices = ("cuda",)
    plugin_options = {"topk-router": {"score_before_selection": True, "renormalize": True},
                      "expert-offload": {"tier": "device"}}
    sharp_routes = True

    @classmethod
    def checkpoint(cls, root):
        path = llama.MoeTests.checkpoint.__func__(cls, root)
        if path is not None:
            _nvfp4_experts(path)
        return path

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        (cls.package / "offload.json").write_text('{"options": {"expert-offload": {"slots": 16}}}', encoding="utf-8")
        cls.stores = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)

    @classmethod
    def tearDownClass(cls):
        super().tearDownClass()
        if hasattr(cls, "stores"):
            cls.stores.cleanup()

    def tier(self, kv=None, **options):
        return self.load(plugins={"expert-offload": {"settings": "offload.json", **options},
                                  "attention-state": dict(kv or {})})

    def replies(self, engine, count=3):
        return [engine.generate(p, 12, stop_tokens=[], sampling=GREEDY) for p in self.prompts(count, 19)]

    def test_host_memory_with_misses_over_the_link_decodes_the_device_tiers_bits(self):
        # The KV pool elastic here (its pages backed as sequences grow) beside a cache that keeps its size.
        offloaded = self.tier(tier="ram", misses="link", kv={"elastic": True})
        facts = offloaded.facts["expert_offload"]
        self.assertEqual((facts["tier"], facts["misses"], facts["slots"]), ("ram", "link", 16))
        self.assertEqual(offloaded.model.facts()["rounds"]["problems"], [])
        self.assertEqual(offloaded.model.info()["round_counters"], ["expert_reads", "expert_hits"])
        want = [tokens for tokens, _ in self.replies(self.engine)]
        found = self.replies(offloaded)
        self.assertEqual([tokens for tokens, _ in found], want)
        for _, result in found:
            self.assertGreater(result["expert_reads"], 0, result)
            self.assertLessEqual(result["expert_hits"], result["expert_reads"])
        (ids,) = self.prompts(1, 150)   # prompt chunks stage their misses through the cache's ring
        self.assertEqual(offloaded.score(ids, 1, "prefill").target, self.engine.score(ids, 1, "prefill").target)

    def test_host_memory_with_the_cpu_tier_loads_and_decodes_within_rounding(self):
        offloaded = self.tier(tier="ram", misses="cpu")
        facts = offloaded.facts["expert_offload"]
        if facts["misses"] != "cpu":
            self.skipTest(f"no CPU tier here: {facts['reasons'].get('misses')}")
        self.assertEqual((facts["tier"], facts["policy"], facts["decode_arithmetic"]), ("ram", "recent", "declared"))
        self.assertEqual(offloaded.model.facts()["rounds"]["problems"], [])
        (ids,) = self.prompts(1, 150)
        scored, want = offloaded.score(ids, 1, "prefill").target, self.engine.score(ids, 1, "prefill").target
        self.assertLess(sum(abs(a - b) for a, b in zip(scored, want)) / len(want), 0.1)
        for tokens, result in self.replies(offloaded):
            self.assertEqual(len(tokens), 12)
            self.assertGreater(result["expert_reads"], 0, result)
        with ThreadPoolExecutor(3) as pool:   # requests together as alone: a miss is the CPU's bits, a hit the same
            alone = [tokens for tokens, _ in self.replies(offloaded)]
            together = list(pool.map(lambda p: offloaded.generate(p, 12, stop_tokens=[], sampling=GREEDY)[0],
                                     self.prompts(3, 19)))
        self.assertEqual(together, alone)
        seen = offloaded.telemetry()["expert_offload"]
        self.assertGreater(seen["cpu_routes"]["routes"], 0, seen)

    def test_a_file_tier_loads_and_decodes_the_device_tiers_bits_through_steps(self):
        offloaded = self.tier(tier="disk", misses="link", store_dir=self.stores.name, host_tier_mib=0)
        facts = offloaded.facts["expert_offload"]
        self.assertEqual((facts["tier"], facts["misses"]), ("disk", "link"))
        # its misses are read on the host, which no captured round does: the model decodes through step()
        self.assertTrue(offloaded.model.facts()["rounds"]["problems"])
        self.assertEqual([tokens for tokens, _ in self.replies(offloaded)],
                         [tokens for tokens, _ in self.replies(self.engine)])
        (ids,) = self.prompts(1, 150)
        self.assertEqual(offloaded.score(ids, 1, "prefill").target, self.engine.score(ids, 1, "prefill").target)
        self.assertGreater(offloaded.telemetry()["expert_offload"]["file_store"]["file_reads"], 0)

    def test_a_store_file_reopened_whole_serves_without_the_loads(self):
        first = self.tier(tier="ram", misses="link", store_dir=self.stores.name)
        want = [tokens for tokens, _ in self.replies(first)]
        first.close()
        self.engines.remove(first)
        again = self.tier(tier="ram", misses="link", store_dir=self.stores.name)
        self.assertTrue(again.model.experts.complete)
        self.assertEqual([tokens for tokens, _ in self.replies(again)], want)


def _draft_layer(path, hidden):
    """A draft layer added to a pre-norm checkpoint: its decoder layer a copy of the last layer's tensors (its
    experts too), fc and its norms random."""
    from safetensors.torch import load_file, save_file
    (shard,) = sorted(path.glob("*.safetensors"))
    tensors = load_file(str(shard))
    last = max(int(n.split("layers.")[1].split(".")[0]) for n in tensors if ".layers." in n)
    prefix = next(n[:-len(f"layers.{last}.input_layernorm.weight")] for n in tensors
                  if n.endswith(f"layers.{last}.input_layernorm.weight"))
    g = torch.Generator().manual_seed(3)
    for name in [n for n in tensors if n.startswith(f"{prefix}layers.{last}.")]:
        tensors["mtp.layers.0." + name[len(f"{prefix}layers.{last}."):]] = tensors[name].clone()
    tensors["mtp.fc.weight"] = (torch.randn(hidden, 2 * hidden, generator=g) * 0.05).to(torch.bfloat16)
    for name in ("pre_fc_norm_embedding", "pre_fc_norm_hidden", "norm"):
        tensors[f"mtp.{name}.weight"] = (torch.randn(hidden, generator=g) * 0.1).to(torch.bfloat16)
    save_file(tensors, str(shard), metadata={"format": "pt"})


@unittest.skipIf(torch is None, "torch")
class GgufHybridRoutedTests(t.GgufEngines, q35.HybridRoutedTests):
    """The same Qwen3.5-MoE checkpoint written as GGUF (architecture qwen35moe): its zero-centred norms stored 1 + w,
    A_log as -exp(A_log), the delta net's value heads tiled (two key heads, four value heads), convolution kernels
    squeezed, the experts stacked, every matrix Q8_0; the experts reach experts-reference as their stored values."""
    arch = "qwen35moe"
    library = q35.HybridRoutedTests.library + ["gguf-projections"]
    use = {**q35.HybridRoutedTests.use, "projections": "gguf-projections"}


@unittest.skipIf(torch is None, "torch")
class DraftExpertTierTests(t.Engines):
    """A checkpoint as published ones store their experts: the decoder's NVFP4 with input scales (W4A4), the draft
    layer's weight-only (W4A16). The class's engine has every expert on the device and drafts from the draft layer;
    the same package then serves with the experts in host memory. Where the CPU tier computes misses the draft
    layer is the cache's link layer (the tier computes W4A4 experts only): its misses admitted at once from a reserve
    of the round's victims, as many as a decode round routes in it, and copied over the link before it reads them;
    the engine loads, drafts in captured rounds and answers within rounding of the device tier's engine. Over the
    link every layer is cached and the replies are the device tier's, token for token."""
    plugin = q35.HERE
    library = t.BASE + ["sequential-recurrence", "topk-router", "expert-offload", "expert-tensors",
                        "greedy-head-drafter"]
    use = {**t.USE, **t.MOE, "experts": "expert-offload", "recurrence": "sequential-recurrence",
           "drafter": "greedy-head-drafter"}
    devices = ("cuda",)
    options = {**t.OPTIONS, "mtp": True, "width": 2}
    plugin_options = {"topk-router": {"score_before_selection": True, "renormalize": True},
                      "expert-offload": {"tier": "device"}}
    sharp_routes = True

    @classmethod
    def checkpoint(cls, root):
        path = q35.hybrid_moe(root, cls)
        if path is not None:
            _draft_layer(path, 128)
            _nvfp4_experts(path, weight_only=("mtp.",))
        return path

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        (cls.package / "offload.json").write_text('{"options": {"expert-offload": {"slots": 32}}}', encoding="utf-8")

    def tier(self, **options):
        return self.load(plugins={"expert-offload": {"settings": "offload.json", "elastic": False, **options}})

    def replies(self, engine):
        return [engine.generate(p, 16, stop_tokens=[], sampling=GREEDY) for p in self.prompts(3, 15)]

    def test_the_cpu_tier_serves_the_decoders_misses_and_the_link_the_draft_layers(self):
        offloaded = self.tier(tier="ram", misses="cpu")
        facts = offloaded.facts["expert_offload"]
        if facts["misses"] != "cpu":
            self.skipTest(f"no CPU tier here: {facts['reasons'].get('misses')}")
        experts, model = offloaded.model.experts, offloaded.model
        draft = model.expert_ids[model.layers]
        self.assertEqual((experts.resident, experts.link_layers), ({}, [draft]))
        # what a captured decode round routes in it: every request's catch-up of width + 1 columns and its chain's
        # width - 1 steps, top routes each
        rows = model.concurrency * (model.width + 1 + model.width - 1)
        reserve = min(experts.shape[1], rows * model.config.moe.experts_per_token)
        self.assertEqual((sorted(experts.cache.link_layers), experts.cache.link_slots), ([draft], reserve))
        self.assertIn("link_layers", facts["reasons"])
        self.assertEqual(offloaded.model.facts()["rounds"]["problems"], [])
        (ids,) = self.prompts(1, 150)
        scored, want = offloaded.score(ids, 1, "prefill").target, self.engine.score(ids, 1, "prefill").target
        self.assertLess(sum(abs(a - b) for a, b in zip(scored, want)) / len(want), 0.1)
        for tokens, result in self.replies(offloaded):
            self.assertEqual(len(tokens), 16)
            self.assertEqual(result["drafted"], [2] * result["rounds"])
            self.assertGreater(result["expert_reads"], 0, result)

    def test_the_draft_layer_over_the_link_replies_as_the_engine_that_holds_it_whole(self):
        # The ranked policy keeps the draft layer whole on the device, as every policy did before link layers; under
        # recent residency it is a link layer in the cache. Its rows are the kernel's tensor arithmetic either way
        # and the CPU tier computes the same W4A4 misses, so the drafted replies are the same, token for token.
        whole, link = self.tier(tier="ram", misses="cpu", policy="ranked"), self.tier(tier="ram", misses="cpu")
        if link.facts["expert_offload"]["misses"] != "cpu":
            self.skipTest(f"no CPU tier here: {link.facts['expert_offload']['reasons'].get('misses')}")
        draft = link.model.expert_ids[link.model.layers]
        self.assertEqual((sorted(whole.model.experts.resident), whole.model.experts.link_layers), ([draft], []))
        self.assertEqual((link.model.experts.resident, link.model.experts.link_layers), ({}, [draft]))
        self.assertEqual([tokens for tokens, _ in self.replies(link)], [tokens for tokens, _ in self.replies(whole)])
        self.assertEqual(link.model.experts.cache.telemetry(sync=True)["failure"], 0)   # no route lost

    def test_over_the_link_every_layer_is_cached_and_replies_are_the_device_tiers(self):
        offloaded = self.tier(tier="ram", misses="link")
        self.assertEqual(offloaded.model.experts.resident, {})
        self.assertEqual([tokens for tokens, _ in self.replies(offloaded)],
                         [tokens for tokens, _ in self.replies(self.engine)])


@unittest.skipIf(torch is None, "torch")
class HybridDraftTests(t.Layout):
    """The gated-hybrid layout, dense, with an MTP layer: a Qwen3.5 text checkpoint (separate in_proj_qkv / z / b / a)
    with an MTP layer added (its decoder layer a copy of the full-attention layer's tensors, fc and norms random), whose
    engine drafts in captured rounds: greedy drafted replies are the plain engine's, each round counting its drafts."""
    plugin = q35.HERE
    library = t.BASE + ["sequential-recurrence", "greedy-head-drafter"]
    use = {**t.USE, "recurrence": "sequential-recurrence"}
    reference_fp32 = True
    devices = ("cuda",)   # drafting runs in captured rounds

    @classmethod
    def checkpoint(cls, root):
        if not hasattr(transformers, "Qwen3_5ForCausalLM"):
            return None
        config = transformers.Qwen3_5TextConfig(
            vocab_size=256, hidden_size=128, intermediate_size=256, num_hidden_layers=4,
            layer_types=["linear_attention", "linear_attention", "linear_attention", "full_attention"],
            num_attention_heads=4, num_key_value_heads=2, head_dim=32, linear_num_key_heads=2,
            linear_num_value_heads=4, linear_key_head_dim=32, linear_value_head_dim=32, linear_conv_kernel_dim=4,
            max_position_embeddings=2048, rms_norm_eps=1e-6, tie_word_embeddings=False)
        model = transformers.Qwen3_5ForCausalLM(config)
        q35.hybrid_weights(model)
        path = cls.save(model, root)
        from safetensors.torch import load_file, save_file
        (shard,) = sorted(path.glob("*.safetensors"))
        tensors = load_file(str(shard))
        prefix = next(n[:-len("layers.3.input_layernorm.weight")] for n in tensors
                      if n.endswith("layers.3.input_layernorm.weight"))
        g = torch.Generator().manual_seed(3)
        for name in [n for n in tensors if n.startswith(prefix + "layers.3.")]:
            tensors["mtp.layers.0." + name[len(prefix + "layers.3."):]] = tensors[name].clone()
        hidden = config.hidden_size
        tensors["mtp.fc.weight"] = (torch.randn(hidden, 2 * hidden, generator=g) * 0.05).to(torch.bfloat16)
        for name in ("pre_fc_norm_embedding", "pre_fc_norm_hidden", "norm"):
            tensors[f"mtp.{name}.weight"] = (torch.randn(hidden, generator=g) * 0.1).to(torch.bfloat16)
        save_file(tensors, str(shard), metadata={"format": "pt"})
        return path

    @classmethod
    def setUpClass(cls):
        super().setUpClass()   # the class's own selection has no drafter: the checks every layout passes load it
        cls.drafting = cls.load(use={"drafter": "greedy-head-drafter"}, mtp=True, width=2)

    def test_drafted_greedy_replies_are_the_plain_ones(self):
        info = self.drafting.info()
        self.assertTrue(info["speculative"])
        self.assertEqual(info["draft_width"], 2)   # an engine fact crit-eval's drafted ceilings read
        self.assertGreater(self.drafting.model.info()["draft_bytes_per_step"], 0)
        prompts = self.prompts(3, 13)
        plain = [self.engine.generate(p, 20, stop_tokens=[], sampling=GREEDY)[0] for p in prompts]
        found = [self.drafting.generate(p, 20, stop_tokens=[], sampling=GREEDY) for p in prompts]
        self.assertEqual([tokens for tokens, _ in found], plain)
        for tokens, result in found:
            self.assertEqual(result["drafted"], [2] * result["rounds"])
            self.assertEqual(sum(result["accepted"]) + result["rounds"], len(tokens) - 1)
        with ThreadPoolExecutor(3) as pool:
            together = list(pool.map(lambda p: self.drafting.generate(p, 20, stop_tokens=[], sampling=GREEDY)[0],
                                     prompts))
        self.assertEqual(together, plain)

    def test_every_commit_of_the_recurrence_loads_in_rounds_and_drafts_the_plain_replies(self):
        # The recurrence's commits that keep a state per round slot (checkpoints, recompute) size themselves by the
        # rounds' rows, which the template gives the pool.
        prompts = self.prompts(2, 13)
        plain = [self.engine.generate(p, 20, stop_tokens=[], sampling=GREEDY)[0] for p in prompts]
        for commit in ("checkpoints", "recompute"):
            with self.subTest(commit=commit):
                engine = self.load(use={"drafter": "greedy-head-drafter"}, mtp=True, width=2,
                                   plugins={"sequential-recurrence": {"commit": commit}})
                self.assertEqual(engine.model.facts()["rounds"]["problems"], [])
                self.assertEqual([engine.generate(p, 20, stop_tokens=[], sampling=GREEDY)[0] for p in prompts], plain)
                engine.close()
                self.engines.remove(engine)


@unittest.skipIf(torch is None, "torch")
class GgufHybridDraftTests(t.GgufEngines, HybridDraftTests):
    """The Qwen3.5 checkpoint with its MTP layer written as GGUF (architecture qwen35): the draft layer as the block
    after the text layers, its fc and norms as the block's nextn parts; drafted greedy replies are the plain ones."""
    arch = "qwen35"
    library = HybridDraftTests.library + ["gguf-projections"]
    use = {**HybridDraftTests.use, "projections": "gguf-projections"}


@unittest.skipIf(torch is None, "torch")
class GgufLatentTests(t.GgufEngines, deepseek.LatentTests):
    """The DeepSeek V3 checkpoint written as GGUF (architecture deepseek2): kv_b_proj split by heads into attn_k_b
    (transposed) and attn_v_b, read joined back; the router's score correction as exp_probs_b.bias, the shared experts
    as *_shexp, the experts stacked, every matrix whose inputs fill Q8_0's blocks Q8_0."""
    arch = "deepseek2"
    library = deepseek.LatentTests.library + ["gguf-projections"]
    use = {**deepseek.LatentTests.use, "projections": "gguf-projections"}


@unittest.skipIf(torch is None, "torch")
class GgufStreamsTests(t.GgufEngines, q4.StreamsTests):
    """The Qwen4Exp checkpoint written as GGUF (architecture qwen4exp, as Qwen3.8-Flash-Next's GGUFs are): the
    hyper-connections renamed, their norms and the n-gram rows' norms stored 1 + w, the indexer's query and key rows
    split, the n-gram table renamed and its hash constants moved to metadata, beside the qwen35moe steps."""
    arch = "qwen4exp"
    library = q4.StreamsTests.library + ["gguf-projections"]
    use = {**q4.StreamsTests.use, "projections": "gguf-projections"}

    def test_the_draft_layer_reads_from_the_draft_block(self):
        # The converter writes the MTP layer as the block after the text layers, its fc_embedding and fc_hidden
        # joined by inputs into one eh_proj: the template reads its halves as the stored blocks of their columns.
        model = self.engine.model
        self.assertEqual(model.matrices.format("mtp.fc_embedding"), "gguf:Q8_0")
        self.assertTrue(model.matrices.has("mtp.fc_hidden"))


@unittest.skipIf(torch is None, "torch")
class SparseStreamsTests(q4.StreamsTests):
    """The same model with a budget of two blocks of four positions, far below the engine's context: every token past
    eight positions attends the two blocks its indexer scores highest and its own block's tail. A cut between two
    blocks that score alike may pick the other block than transformers' FP32 scores do (and blocks that score exactly
    zero tie: crit keeps the lower block), so a few positions may differ. On CUDA the indexer's shape is one crit's
    kernel sparse-indexer takes."""
    budget = 8
    flips = 0.03
    sharp_attention = True   # which blocks the indexer picks decides the logits

    def test_greedy_continuations_are_transformers_choices(self):
        # A token whose block list differs from transformers' at a near-tie may be another: all but a few agree.
        missed = total = 0
        for prompt in self.prompts(3):
            tokens, _ = self.engine.generate(prompt, 12, stop_tokens=[], sampling=GREEDY)
            logits = self.logits(prompt + tokens)
            for i, token in enumerate(tokens):
                row = logits[len(prompt) - 1 + i]
                missed += float(row[token]) < float(row.max()) - 0.25
                total += 1
        self.assertLessEqual(missed, 2, total)

    def test_the_indexer_scores_a_steps_tokens_in_bounded_groups_with_the_same_choices(self):
        # A step's scores are a row of every earlier block a token: held for a whole step at a long context they
        # are gigabytes. The block scores its tokens in groups within score_bytes; a token's choice is its own, so
        # a bound of one token a group scores a text exactly as the default does, with no more than a group's scores
        # alive at a time.
        (ids,) = self.prompts(1, 300)
        mixer = self.engine.model.decoder[3].mixer
        want = self.engine.score(ids, 1, "prefill").target
        seen = []
        scores = mixer._scores
        mixer._scores = lambda q, keys: seen.append(tuple(q.shape[:1]) + tuple(keys.shape[:1])) or scores(q, keys)
        mixer.score_bytes = 1
        try:
            found = self.engine.score(ids, 1, "prefill").target
        finally:
            del mixer.score_bytes, mixer._scores
        self.assertEqual(found, want)
        if mixer.kernel is None:   # the torch form: a token a group, never a chunk's tokens at once
            self.assertTrue(seen and all(tokens == 1 for tokens, _ in seen), seen[:4])

    def test_a_draft_layer_over_streams_drafts_the_plain_greedy_replies(self):
        # The checkpoint's draft layer (its own streams from the target's, a block-sparse attention with index keys of
        # its own, its own last mix) drafts in captured rounds; every draft is verified, so replies are the plain ones.
        if self.device != "cuda":
            self.skipTest("drafting runs in captured rounds")
        type(self).library = self.library + ["greedy-head-drafter"]
        from crit.plugins import add
        add(self.package / "plugins", ["greedy-head-drafter"])
        drafting = self.load({"drafter": "greedy-head-drafter"}, mtp=True, width=2)
        self.assertTrue(drafting.info()["speculative"])
        self.assertIn("mtp-index", drafting.model.pools)
        prompts = self.prompts(3, 13)
        plain = [self.engine.generate(p, 20, stop_tokens=[], sampling=GREEDY)[0] for p in prompts]
        found = [drafting.generate(p, 20, stop_tokens=[], sampling=GREEDY) for p in prompts]
        self.assertEqual([tokens for tokens, _ in found], plain)
        for tokens, result in found:
            self.assertEqual(result["drafted"], [2] * result["rounds"])
        with ThreadPoolExecutor(3) as pool:
            together = list(pool.map(lambda p: drafting.generate(p, 20, stop_tokens=[], sampling=GREEDY)[0], prompts))
        self.assertEqual(together, plain)


@unittest.skipIf(torch is None, "torch")
class OffDeviceTablesTests(q4.StreamsTests):
    """The same engine with its lookup tables placed off the device (device-placement's off_device "tables.", as an
    engine for a checkpoint whose n-gram table is larger than the device places them): the token embeddings and the
    n-gram table are read where the checkpoint stores them through lookup-table-rows (embeddings:host), the plan's
    device bytes leave them out, and every comparison of q4.StreamsTests holds."""
    library = q4.StreamsTests.library + ["lookup-table-rows"]
    use = {**q4.StreamsTests.use, "embeddings": "lookup-table-rows"}
    plugin_options = {**q4.StreamsTests.plugin_options, "device-placement": {"off_device": "tables."}}

    def test_the_tables_the_plan_puts_off_the_device_are_read_where_they_are_stored(self):
        model = self.engine.model
        self.assertEqual({name: place for name, place in model.placed.items() if name.startswith("tables.")},
                         {name: "host" for name in model.placed if name.startswith("tables.")})
        self.assertNotEqual(model.embed.mode, "device")
        self.assertNotEqual(model.decoder[1].before[0].table.mode, "device")


@unittest.skipIf(torch is None, "torch")
@unittest.skipIf(sys.platform == "win32", "host memory regions need libc's mmap")
class OffDeviceMemoryTablesTests(q4.StreamsTests):
    """The same off-device tables through memory-table-rows (embeddings:host): the plain embeddings plugin reads
    each table the plan put off the device from the checkpoint's files once into registered host memory (the
    Flash-Next build's candidate that refused at load with "does not provide embeddings:host"), and every comparison
    of q4.StreamsTests holds."""
    plugin_options = {**q4.StreamsTests.plugin_options, "device-placement": {"off_device": "tables."}}

    def test_the_tables_the_plan_puts_off_the_device_are_held_in_host_memory(self):
        model = self.engine.model
        self.assertTrue(any(name.startswith("tables.") for name in model.placed))
        self.assertEqual({name: place for name, place in model.placed.items() if name.startswith("tables.")},
                         {name: "host" for name in model.placed if name.startswith("tables.")})
        self.assertEqual(model.embed.mode, "mapped")
        self.assertEqual(model.decoder[1].before[0].table.mode, "mapped")


@unittest.skipIf(torch is None, "torch")
class PublishedStreamsTests(q4.StreamsTests):
    """The Qwen4Exp checkpoint stored as NVIDIA's NVFP4 Flash-Next stores its own: the hashed n-gram table as E4M3
    codes with one BF16 multiplier for the table (weight_scale), the draft layer's routed experts a matrix each in
    FP8 with a multiplier a 32 x 32 block (weight_scale_inv). The table's values are E4M3 values times a power of two,
    so the BF16 model transformers runs holds them exactly, and every comparison of q4.StreamsTests holds as it does for
    the BF16 table, on the device and off it. The draft layer's experts are re-encoded to the format the experts slot
    holds the decoder's in (BF16 here) and draft the plain greedy replies."""
    TABLE_SCALE = 0.125

    @classmethod
    def save(cls, model, root, fp32=False):
        with torch.no_grad():
            for name, parameter in model.named_parameters():
                if "ngram_embedding" in name:
                    codes = (parameter.float() / cls.TABLE_SCALE).to(torch.float8_e4m3fn)
                    parameter.copy_(codes.float() * cls.TABLE_SCALE)
        return super().save(model, root, fp32)

    @classmethod
    def draft_layer(cls, path, config):
        from safetensors.torch import load_file, save_file
        super().draft_layer(path, config)
        (shard,) = sorted(path.glob("*.safetensors"))
        tensors = load_file(str(shard))
        tables = [n for n in tensors if "ngram_embedding" in n and n.endswith(".weight")]
        for name in tables:
            tensors[name] = (tensors[name].float() / cls.TABLE_SCALE).to(torch.float8_e4m3fn)
        at = tables[0][:tables[0].index("ngram_embedding.") + len("ngram_embedding.")]
        tensors[at + "weight_scale"] = torch.tensor([cls.TABLE_SCALE], dtype=torch.bfloat16)
        g = torch.Generator().manual_seed(6)
        for name in sorted(n for n in tensors if n.startswith("mtp.layers.0.mlp.experts.") and n.endswith("_proj.weight")):
            matrix = tensors[name].float()
            rows, columns = matrix.shape
            # a power of two a block, so each code times its multiplier is the BF16 value it reads as
            scale = 2.0 ** -torch.randint(4, 8, (rows // 32, columns // 32), generator=g).float()
            factors = scale.repeat_interleave(32, 0).repeat_interleave(32, 1)
            tensors[name] = (matrix / factors).to(torch.float8_e4m3fn)
            tensors[name + "_scale_inv"] = scale.to(torch.bfloat16)
        save_file(tensors, str(shard), metadata={"format": "pt"})

    def test_the_table_reads_as_fp8_and_the_draft_layers_experts_are_re_encoded(self):
        from crit.plugins import add
        self.assertEqual(self.engine.model.decoder[1].before[0].table.format, "fp8")
        if "greedy-head-drafter" not in self.library:
            type(self).library = self.library + ["greedy-head-drafter"]
            add(self.package / "plugins", ["greedy-head-drafter"])
        drafting = self.load({"drafter": "greedy-head-drafter"}, mtp=True, width=2)
        self.assertEqual(drafting.model.facts()["mtp"]["experts"], "stored fp8:block, re-encoded to bf16")
        if self.device != "cuda":
            return   # drafting runs in captured rounds
        prompts = self.prompts(2, 13)
        plain = [self.engine.generate(p, 16, stop_tokens=[], sampling=GREEDY)[0] for p in prompts]
        self.assertEqual([drafting.generate(p, 16, stop_tokens=[], sampling=GREEDY)[0] for p in prompts], plain)


@unittest.skipIf(torch is None, "torch")
class EchoDraftTests(q4.StreamsTests):
    """A draft layer whose drafts are known. transformers runs no draft layer for this architecture, so a random one
    only shows that drafts are verified. This one passes the target's streams on: fc_hidden is the identity,
    fc_embedding zero, the feature's norm weight zero, its attention's and experts' output maps zero, its last mix
    the model's own. Its output rows are then the target's at the position its feature came from, so every draft of
    a greedy round is the token that position predicted: the round's pending token, at every depth. A draft head
    that reads another position's feature, or another column's, proposes other tokens."""
    devices = ("cuda",)

    @classmethod
    def draft_layer(cls, path, config):
        from safetensors.torch import load_file, save_file
        super().draft_layer(path, config)
        (shard,) = sorted(path.glob("*.safetensors"))
        tensors = load_file(str(shard))
        hidden, streams = config.hidden_size, config.hc_count
        tensors["mtp.fc_hidden.weight"] = torch.eye(hidden).to(torch.bfloat16)
        tensors["mtp.fc_embedding.weight"] = torch.zeros(hidden, hidden).to(torch.bfloat16)
        tensors["mtp.pre_fc_norm_hidden.weight"] = torch.zeros(streams * hidden).to(torch.bfloat16)
        silent = [name for name in tensors if name.startswith("mtp.layers.0.") and (
            name.endswith(("self_attn.o_proj.weight", "down_proj.weight")) or name.endswith("mlp.experts.down_proj"))]
        if len(silent) < 3:
            raise AssertionError(f"the draft layer's output maps: {silent}")
        for name in silent:
            tensors[name] = torch.zeros_like(tensors[name])
        save_file(tensors, str(shard), metadata={"format": "pt"})

    def test_a_draft_layer_that_passes_the_targets_streams_on_drafts_the_pending_token(self):
        from crit.plugins import add
        for drafter, options in (("greedy-head-drafter", {}), ("mtp-decode", {"override": 1})):
            with self.subTest(drafter=drafter):
                if drafter not in self.library:
                    type(self).library = self.library + [drafter]
                    add(self.package / "plugins", [drafter])
                drafting = self.load({"drafter": drafter}, {drafter: options}, mtp=True, width=2)
                rounds = drafting.model.rounds
                seen = []
                for prompt in self.prompts(3, 13):
                    for count in (2, 3, 5, 9):
                        tokens, result = drafting.generate(prompt, count, stop_tokens=[], sampling=GREEDY)
                        # the last round's start: its pending token is the reply's token before the round's own
                        drafts = rounds.drafts[0, :2].tolist()
                        pending = int(rounds.start["pending"][0])
                        seen.append((pending, drafts, tokens, result["rounds"]))
                echoed = sum(drafts == [pending, pending] for pending, drafts, _, _ in seen)
                self.assertEqual(echoed, len(seen), seen)
                drafting.close()
                self.engines.remove(drafting)
        # mtp-decode at its defaults proposes prompt copies alone and never asks the draft head: the draft layer
        # would be loaded and kept up for nothing, with every draft cut, so the selection is refused by name.
        with self.assertRaisesRegex(Exception, "never asks the model's draft head"):
            self.load({"drafter": "mtp-decode"}, mtp=True, width=2)


@unittest.skipIf(torch is None, "torch")
class GgufSandwichTests(t.GgufEngines, gemma.SandwichTests):
    """The Gemma 4 checkpoint written as GGUF (architecture gemma4): its norms renamed in pairs around each block, the
    scalar as layer_output_scale, the router's scale and each expert's scale as .scale tensors, the experts one gate-up
    stack beside the down stack, every matrix Q8_0."""
    arch = "gemma4"
    library = gemma.SandwichTests.library + ["gguf-projections"]
    use = {**gemma.SandwichTests.use, "projections": "gguf-projections"}
    # A prompt row multiplies the Q8_0 matrices decoded to BF16 (gguf-projections' dequantize prompts), which rounds
    # each stored value d q once, where the reference reads it exact; Gemma 4's attention at scale 1 carries that
    # rounding further than the other layouts do: 0.21 nats a position on average here, where the same file written
    # F32 scores what the safetensors engine scores (0.090), bit for bit. The same rounding moves a few more near-tie
    # tokens (3 of 149 positions here).
    mean_limit, flips = 0.25, 0.03


@unittest.skipIf(torch is None, "torch")
class SandwichHostTierTests(gemma.SandwichTests):
    """The same engine with its GELU-gated experts in host memory behind a device cache that holds half of them
    (expert-offload's ram tier: BF16 bundles through the grouped-experts kernel's GELU gate), its prompts run layer
    by layer because the experts plugin stages them."""
    library = t.BASE + ["topk-router", "expert-offload", "expert-tensors"]
    use = {**t.USE, **t.MOE, "experts": "expert-offload"}
    devices = ("cuda",)
    plugin_options = {**gemma.SandwichTests.plugin_options,
                      "expert-offload": {"tier": "ram", "misses": "link", "settings": "offload.json"}}

    @classmethod
    def checkpoint(cls, root):
        (root / "package").mkdir(parents=True, exist_ok=True)
        (root / "package" / "offload.json").write_text('{"options": {"expert-offload": {"slots": 16}}}',
                                                       encoding="utf-8")
        return super().checkpoint(root)

    def test_the_experts_are_gated_by_gelu_off_the_device(self):
        facts = self.engine.facts["expert_offload"]
        self.assertEqual((facts["tier"], facts["misses"], facts["slots"]), ("ram", "link", 16))
        self.assertEqual(self.engine.model.experts.activation, "gelu_tanh")
        self.assertTrue(self.engine.model.facts()["passes"]["on"])


@unittest.skipIf(torch is None, "torch")
class LoopedGroupedTests(inkling.LoopedTests):
    """The looped engine with grouped-experts in the experts slot: its kernel's capped SwiGLU over the BF16 experts."""
    library = t.BASE + ["sequential-recurrence", "topk-router", "grouped-experts", "expert-tensors"]
    use = {**inkling.LoopedTests.use, "experts": "grouped-experts"}
    devices = ("cuda",)


@unittest.skipIf(torch is None, "torch")
class LoopedDraftTests(inkling.LoopedTests):
    """The looped engine drafting from a block drafter stored beside the checkpoint (mtp.safetensors, the config's
    mtp_config: three bidirectional layers, its own cache, a recurrent head; random weights): greedy drafted replies are
    the plain engine's at every width, one block pass a round."""
    library = LoopedGroupedTests.library + ["greedy-head-drafter", "quantized-projections"]
    use = LoopedGroupedTests.use
    devices = ("cuda",)   # drafting runs in captured rounds

    @classmethod
    def checkpoint(cls, root):
        path = super().checkpoint(root)
        if path is None:
            return None
        from safetensors.torch import save_file
        h, heads, kv, d, f, r, block = 64, 4, 2, 16, 96, 8, 4
        cfg = dict(hidden_size=h, num_hidden_layers=3, num_attention_heads=heads, num_key_value_heads=kv, head_dim=d,
                   intermediate_size=f, rms_norm_eps=1e-6, rope_theta=1e6, local_layer_ids=[0, 2],
                   sliding_window_size=6, block_size=block, markov_rank=r, markov_head_type="rnn_hidden",
                   target_layer_ids=[1, 4, 7, 10])   # entries of layer applications 1, 4, 7, then the final row
        g = torch.Generator().manual_seed(11)

        def rnd(*shape, scale=None):
            return (torch.randn(*shape, generator=g) * (scale or shape[-1] ** -0.5)).to(torch.bfloat16)

        def norm(n):
            return (1.0 + 0.2 * torch.randn(n, generator=g)).to(torch.bfloat16)

        tensors = {"fuse_weight": rnd(4, 4, scale=1.0), "mask_embed": rnd(h, scale=1.0), "norm.weight": norm(h)}
        for i in range(3):
            p = f"layers.{i}."
            for n in ("dflare_norm", "input_layernorm", "post_attention_layernorm"):
                tensors[p + n + ".weight"] = norm(h)
            for n, rows in (("q_proj", heads * d), ("k_proj", kv * d), ("v_proj", kv * d), ("k_proj_ctx", kv * d),
                            ("v_proj_ctx", kv * d)):
                tensors[p + f"self_attn.{n}.weight"] = rnd(rows, h)
            tensors[p + "self_attn.o_proj.weight"] = rnd(h, heads * d)
            for n in ("q_norm", "k_norm", "k_norm_ctx"):
                tensors[p + f"self_attn.{n}.weight"] = norm(d)
            tensors[p + "mlp.gate_proj.weight"], tensors[p + "mlp.up_proj.weight"] = rnd(f, h), rnd(f, h)
            tensors[p + "mlp.down_proj.weight"] = rnd(h, f)
        m = "markov_head."
        tensors.update({m + "dflare_norm.weight": norm(h),
                        m + "markov_w1.weight": rnd(inkling.LOOPED["vocab_size"], r, scale=1.0),
                        m + "rnn_proj.weight": rnd(3 * r, 2 * r + 2 * h), m + "rnn_proj.bias": rnd(3 * r, scale=0.5),
                        m + "hidden_out.weight": rnd(h, r), m + "hidden_out.bias": rnd(h, scale=0.1)})
        save_file(tensors, str(path / "mtp.safetensors"), metadata={"format": "pt"})
        config = json.loads((path / "config.json").read_text(encoding="utf-8"))
        config["mtp_config"] = cfg
        (path / "config.json").write_text(json.dumps(config), encoding="utf-8")
        return path

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.drafting = cls.load(use={"drafter": "greedy-head-drafter"}, mtp=True, width=3, widths="1,2,3")

    def test_drafted_greedy_replies_are_the_plain_ones(self):
        model = self.drafting.model
        self.assertIsNotNone(model.feature_head)   # the block-drafter technique's head
        self.assertEqual(model.info().get("speculative"), True)
        self.assertEqual(model.taps, (1, 4, 7))
        self.assertEqual(model.facts()["rounds"]["problems"], [])
        prompts = self.prompts(3, 13)
        plain = [self.engine.generate(p, 24, stop_tokens=[], sampling=GREEDY)[0] for p in prompts]
        found = [self.drafting.generate(p, 24, stop_tokens=[], sampling=GREEDY) for p in prompts]
        self.assertEqual([tokens for tokens, _ in found], plain)
        for tokens, result in found:
            self.assertEqual(sum(result["accepted"]) + result["rounds"], len(tokens) - 1)
        with ThreadPoolExecutor(2) as pool:   # two round slots: each row's drafter cache its own
            together = list(pool.map(lambda p: self.drafting.generate(p, 24, stop_tokens=[], sampling=GREEDY)[0],
                                     prompts[:2]))
        self.assertEqual(together, plain[:2])

    def test_drafting_over_captured_prompt_chunks_gives_the_plain_replies(self):
        # a captured chunk gives the drafter's feature rows (its taps, then the head's rows), which its prompt rows read;
        # over quantized-projections, whose prompt form captures (projections-dense's runs every chunk eager)
        engine = self.load(use={"drafter": "greedy-head-drafter", "projections": "quantized-projections"}, mtp=True,
                           width=3, widths="1,2,3", prompt_buckets="16,64")
        prompts = self.prompts(3, 13)
        plain = [self.engine.generate(p, 24, stop_tokens=[], sampling=GREEDY)[0] for p in prompts]
        self.assertEqual([engine.generate(p, 24, stop_tokens=[], sampling=GREEDY)[0] for p in prompts], plain)
        graphs = engine.model.prompts
        self.assertEqual((engine.model.prompts_refused, graphs.problems, sorted(graphs.graphs)), (None, [], [16, 64]))
        self.assertGreater(graphs.counts["graph_chunks"], 0)
        engine.close()
        self.engines.remove(engine)


@unittest.skipIf(torch is None, "torch")
class TransformersSpellingDraftTests(t.Respelled, LoopedDraftTests):
    """The drafting engine over the target's tensors under transformers' own module names (the looped plugin's
    TRANSFORMERS_NAMES, its drafter beside them as stored): drafted greedy replies are the plain engine's."""
    spelling, files, every_rule = inkling.TRANSFORMERS_NAMES, "model*.safetensors", False


class LoopedOffloadTests(inkling.LoopedTests):
    """The looped engine with expert-offload in the experts slot, every expert on the device: each pass of a looped
    layer reads the same layer's experts (one experts slot a checkpoint layer)."""
    library = t.BASE + ["sequential-recurrence", "topk-router", "expert-offload", "expert-tensors"]
    use = {**inkling.LoopedTests.use, "experts": "expert-offload"}
    devices = ("cuda",)
    plugin_options = {**inkling.LoopedTests.plugin_options, "expert-offload": {"tier": "device", "misses": "link"}}


@unittest.skipIf(torch is None, "torch")
class LoopedHostTierTests(LoopedOffloadTests):
    """The looped engine with its experts in host memory behind a device cache of half of them: a looped layer's
    passes come back to experts the cache may have let go (24 slots of the 48 experts)."""
    plugin_options = {**inkling.LoopedTests.plugin_options,
                      "expert-offload": {"tier": "ram", "misses": "link", "settings": "offload.json"}}

    @classmethod
    def checkpoint(cls, root):
        (root / "package").mkdir(parents=True, exist_ok=True)
        (root / "package" / "offload.json").write_text('{"options": {"expert-offload": {"slots": 24}}}',
                                                       encoding="utf-8")
        return super().checkpoint(root)


if __name__ == "__main__":
    unittest.main()
