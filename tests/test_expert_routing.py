"""Routing descriptions: the host reference against hand-worked cases and, when installed, the
Hugging Face transformers router code each parameterization reproduces; config mapping from
Core's parsed fields. The CUDA kernel is compared with this reference in tests/inference/cuda."""
from dataclasses import replace
import importlib.util
import math
import sys
from pathlib import Path
from types import SimpleNamespace
import unittest

from test_techniques import load

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "core" / "python"))
from crit_core.model.config import parse_model_config  # noqa: E402

r = load("heterogeneous-expert-dispatch", "routing.py")
HAS_TRANSFORMERS = all(importlib.util.find_spec(name) for name in ("torch", "transformers"))


def softmax(values):
    exponents = [math.exp(v) for v in values]
    return [e / sum(exponents) for e in exponents]


def sigmoid(value):
    return 1 / (1 + math.exp(-value))


class HandWorkedRoutingTests(unittest.TestCase):
    logits = (1.0, 3.0, -2.0, 2.0, 0.5, 2.5)

    def check(self, result, ids, weights):
        self.assertEqual(result[0], ids)
        for got, expected in zip(result[1], weights, strict=True):
            self.assertAlmostEqual(got, expected, places=12)

    def test_default_is_softmax_over_top_logits(self):
        self.check(r.route(self.logits, 3), (1, 5, 3), softmax([3.0, 2.5, 2.0]))

    def test_softmax_before_selection_with_and_without_renormalization(self):
        p = softmax(self.logits)
        before = r.Routing(score_before_selection=True)
        self.check(r.route(self.logits, 2, before), (1, 5), (p[1], p[5]))
        self.check(r.route(self.logits, 2, replace(before, renormalize=True, scaling=2.0)),
                   (1, 5), (2 * p[1] / (p[1] + p[5]), 2 * p[5] / (p[1] + p[5])))

    def test_bias_changes_selection_but_not_weights(self):
        bias = (0.0, -1.0, 0.0, 0.0, 0.0, 0.0)
        routing = r.Routing(scoring="sigmoid", score_before_selection=True, correction_bias=True)
        # Selection values: sigmoid + bias = .731, -.047, .119, .881, .622, .924.
        self.check(r.route(self.logits, 2, routing, bias), (5, 3), (sigmoid(2.5), sigmoid(2.0)))

    def test_groups_rank_by_their_best_values(self):
        # Groups of two: (1, 3), (2.5, 2.4), (-2, 2). Best one ranks 3 > 2.5 > 2; best two 4.9 > 4 > 0.
        logits = (1.0, 3.0, 2.5, 2.4, -2.0, 2.0)
        one = r.Routing(groups=3, selected_groups=1, group_score_top=1)
        self.assertEqual(r.route(logits, 2, one)[0], (1, 0))
        two = replace(one, group_score_top=2)
        self.assertEqual(r.route(logits, 2, two)[0], (2, 3))
        self.assertEqual(r.route(logits, 3, replace(two, selected_groups=2))[0], (1, 2, 3))

    def test_sigmoid_after_selection_and_floor(self):
        self.check(r.route(self.logits, 2, r.Routing(scoring="sigmoid")), (1, 5), (sigmoid(3.0), sigmoid(2.5)))
        floor = r.Routing(score_before_selection=True, renormalize=True, renormalize_floor=10.0)
        p = softmax(self.logits)
        self.check(r.route(self.logits, 1, floor), (1,), (p[1] / 10,))

    def test_ties_choose_lower_expert_and_group(self):
        self.assertEqual(r.route((0.0,) * 8, 3), ((0, 1, 2), (1 / 3,) * 3))
        grouped = r.Routing(groups=4, selected_groups=2, group_score_top=2)
        self.assertEqual(r.route((1.0, 0.0) * 4, 3, grouped)[0], (0, 2, 1))

    def test_invalid_descriptions_fail(self):
        for routing, top in ((r.Routing(scoring="sqrtsoftplus"), 2), (r.Routing(groups=4), 2),
                             (r.Routing(groups=3, selected_groups=1), 3),
                             (r.Routing(groups=3, selected_groups=1, group_score_top=3), 1),
                             (r.Routing(renormalize=True), 2), (r.Routing(scaling=0.0), 2),
                             (r.Routing(score_before_selection=True, renormalize_epsilon=-1.0), 2)):
            with self.assertRaises(ValueError):
                r.route(self.logits, top, routing)
        with self.assertRaises(ValueError):
            r.route(self.logits, 2, r.Routing(correction_bias=True))
        with self.assertRaises(ValueError):
            r.route(self.logits, 2, r.Routing(), bias=(0.0,) * 6)
        with self.assertRaises(ValueError):
            r.route((1.0, math.nan), 1)


@unittest.skipUnless(HAS_TRANSFORMERS, "torch and transformers provide the reference router code")
class TransformersRoutingTests(unittest.TestCase):
    """Each Routing reproduces the cited transformers router; IDs exact, FP32 weights near."""

    def compare(self, routing, reference, experts=64, top=6, tokens=16, bias=False, seed=0):
        import torch
        generator = torch.Generator().manual_seed(seed)
        logits = torch.randn(tokens, experts, generator=generator, dtype=torch.float32) * 2
        offsets = torch.randn(experts, generator=generator, dtype=torch.float32) * 0.1 if bias else None
        with torch.no_grad():
            ids, weights = reference(logits.clone(), offsets)
        for t in range(tokens):
            expected = sorted(zip(ids[t].tolist(), weights[t].tolist()))
            got_ids, got_weights = r.route(logits[t].tolist(), top, routing,
                                           offsets.tolist() if bias else None)
            got = sorted(zip(got_ids, got_weights))
            self.assertEqual([e for e, _ in got], [e for e, _ in expected])
            for (_, a), (_, b) in zip(got, expected):
                self.assertAlmostEqual(a, b, delta=2e-6 * max(1.0, abs(b)))

    @staticmethod
    def identity(router, experts):
        import torch
        router.weight.data = torch.eye(experts)
        return router

    def test_mixtral_softmax_then_renormalized_top_k(self):
        from transformers.models.mixtral.modeling_mixtral import MixtralTopKRouter
        router = self.identity(MixtralTopKRouter(SimpleNamespace(num_experts_per_tok=6, num_local_experts=64,
                                                                 hidden_size=64)), 64)
        self.compare(r.Routing(score_before_selection=True, renormalize=True), lambda x, _: router(x)[2:0:-1])
        # The default (softmax over the selected logits) equals it up to rounding.
        self.compare(r.Routing(), lambda x, _: router(x)[2:0:-1])

    def test_qwen_moe_norm_topk_prob(self):
        from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeTopKRouter
        from transformers.models.qwen2_moe.modeling_qwen2_moe import Qwen2MoeTopKRouter
        for cls, norm in ((Qwen3MoeTopKRouter, True), (Qwen2MoeTopKRouter, False)):
            router = self.identity(cls(SimpleNamespace(num_experts_per_tok=8, num_experts=128, hidden_size=128,
                                                       norm_topk_prob=norm)), 128)
            self.compare(r.Routing(score_before_selection=True, renormalize=norm), lambda x, _: router(x)[2:0:-1],
                         experts=128, top=8)

    def test_gpt_oss_and_llama4_score_after_selection(self):
        import torch
        from transformers.models.gpt_oss.modeling_gpt_oss import GptOssTopKRouter
        from transformers.models.llama4.modeling_llama4 import Llama4Router
        gpt = self.identity(GptOssTopKRouter(SimpleNamespace(num_experts_per_tok=4, num_local_experts=32,
                                                             hidden_size=32)), 32)
        self.compare(r.Routing(), lambda x, _: gpt(x)[2:0:-1], experts=32, top=4)
        llama = self.identity(Llama4Router(SimpleNamespace(hidden_size=16, num_local_experts=16, num_experts_per_tok=1)), 16)

        def selected(x, _):
            scores, _ = llama(x)
            ids = torch.argmax(scores, dim=-1, keepdim=True)
            return ids, scores.gather(1, ids)
        self.compare(r.Routing(scoring="sigmoid"), selected, experts=16, top=1)

    def test_deepseek_v3_family_sigmoid_bias_groups(self):
        from transformers.models.deepseek_v3.modeling_deepseek_v3 import DeepseekV3MoE
        from transformers.models.glm4_moe.modeling_glm4_moe import Glm4MoeMoE
        # DeepSeek-V3 (8 groups, best 4, 2.5x), and the single-group GLM-4.5 / Kimi K2 form.
        for cls, groups, selected, scale in ((DeepseekV3MoE, 8, 4, 2.5), (Glm4MoeMoE, 1, 1, 2.5),
                                             (DeepseekV3MoE, 1, 1, 2.827)):
            def reference(x, bias, cls=cls, groups=groups, selected=selected, scale=scale):
                owner = SimpleNamespace(gate=SimpleNamespace(e_score_correction_bias=bias), n_group=groups,
                                        n_routed_experts=64, topk_group=selected, top_k=8, norm_topk_prob=True,
                                        routed_scaling_factor=scale)
                return cls.route_tokens_to_experts(owner, x)
            routing = r.Routing(scoring="sigmoid", score_before_selection=True, correction_bias=True, groups=groups,
                                selected_groups=selected, group_score_top=2 if groups > 1 else 1, renormalize=True,
                                renormalize_epsilon=1e-20, scaling=scale)
            self.compare(routing, reference, top=8, bias=True)

    def test_deepseek_v2_group_limited_greedy(self):
        from transformers.models.deepseek_v2.modeling_deepseek_v2 import DeepseekV2Moe
        for method, groups in (("group_limited_greedy", 8), ("greedy", 1)):
            def reference(x, _, method=method, groups=groups):
                owner = SimpleNamespace(topk_method=method, top_k=6, num_group=groups, topk_group=3,
                                        num_experts=64, routed_scaling_factor=16.0)
                return DeepseekV2Moe.route_tokens_to_experts(owner, x.view(1, *x.shape))
            routing = r.Routing(score_before_selection=True, groups=groups, selected_groups=min(3, groups),
                                scaling=16.0)
            self.compare(routing, reference)

    def test_minimax_ernie_longcat_bias_forms(self):
        from transformers.models.minimax_m2.modeling_minimax_m2 import MiniMaxM2TopKRouter
        from transformers.models.ernie4_5_moe.modeling_ernie4_5_moe import Ernie4_5_MoeTopKRouter
        from transformers.models.longcat_flash.modeling_longcat_flash import LongcatFlashTopkRouter
        minimax = self.identity(MiniMaxM2TopKRouter(SimpleNamespace(num_experts_per_tok=8, num_local_experts=64,
                                                                     hidden_size=64)), 64)
        self.compare(r.Routing(scoring="sigmoid", score_before_selection=True, correction_bias=True, renormalize=True),
                     lambda x, bias: minimax(x, bias)[2:0:-1], top=8, bias=True)
        ernie = self.identity(Ernie4_5_MoeTopKRouter(SimpleNamespace(moe_num_experts=64, hidden_size=64, moe_k=6,
                                                                       moe_norm_min=1e-12)), 64)

        def ernie_reference(x, bias):
            ernie.moe_statics.e_score_correction_bias.data = bias.view(1, -1)
            return ernie(x)[1:]
        self.compare(r.Routing(score_before_selection=True, correction_bias=True, renormalize=True,
                               renormalize_floor=1e-12), ernie_reference, bias=True)
        longcat = LongcatFlashTopkRouter(SimpleNamespace(n_routed_experts=64, zero_expert_num=0, moe_topk=6,
                                                         routed_scaling_factor=6.0, router_bias=False, hidden_size=64))
        self.identity(longcat.classifier, 64)

        def longcat_reference(x, bias):
            longcat.e_score_correction_bias = bias
            return longcat(x)[::-1]
        self.compare(r.Routing(score_before_selection=True, correction_bias=True, scaling=6.0), longcat_reference,
                     bias=True)


class ConfigRoutingTests(unittest.TestCase):
    base = {"hidden_size": 64, "num_hidden_layers": 2, "num_attention_heads": 4, "vocab_size": 32}

    def moe(self, **fields):
        return parse_model_config({**self.base, **fields}).moe

    def test_deepseek_v3_config(self):
        moe = self.moe(n_routed_experts=256, num_experts_per_tok=8, n_group=8, topk_group=4, topk_method="noaux_tc",
                       scoring_func="sigmoid", norm_topk_prob=True, routed_scaling_factor=2.5)
        routing = r.routing_from_config(moe, correction_bias=True,
                                        defaults={"score_before_selection": True, "renormalize_epsilon": 1e-20})
        self.assertEqual(routing, r.Routing("sigmoid", True, True, 8, 4, 2, True, 1e-20, 0.0, 2.5))
        with self.assertRaises(ValueError):  # noaux_tc selects with the bias
            r.routing_from_config(moe, defaults={"score_before_selection": True})

    def test_deepseek_v2_and_single_group_configs(self):
        v2 = self.moe(n_routed_experts=160, num_experts_per_tok=6, n_group=8, topk_group=3,
                      topk_method="group_limited_greedy", norm_topk_prob=False, routed_scaling_factor=16.0)
        routing = r.routing_from_config(v2, defaults={"scoring_func": "softmax", "score_before_selection": True})
        self.assertEqual((routing.groups, routing.selected_groups, routing.group_score_top, routing.scaling), (8, 3, 1, 16.0))
        greedy = self.moe(n_routed_experts=64, num_experts_per_tok=6, n_group=8, topk_group=3, topk_method="greedy",
                          norm_topk_prob=False, routed_scaling_factor=1.0)
        routing = r.routing_from_config(greedy, defaults={"scoring_func": "softmax", "score_before_selection": True})
        self.assertEqual((routing.groups, routing.selected_groups), (1, 1))
        kimi = self.moe(n_routed_experts=384, num_experts_per_tok=8, n_group=1, topk_group=1, scoring_func="sigmoid",
                        norm_topk_prob=True, routed_scaling_factor=2.827)
        routing = r.routing_from_config(kimi, correction_bias=True, defaults={"score_before_selection": True})
        self.assertEqual((routing.groups, routing.correction_bias, routing.scaling), (1, True, 2.827))

    def test_omitted_fields_come_from_defaults_or_fail(self):
        mixtral = self.moe(num_local_experts=8, num_experts_per_tok=2)
        defaults = {"scoring_func": "softmax", "norm_topk_prob": True, "routed_scaling_factor": 1.0,
                    "n_group": 1, "topk_group": 1, "score_before_selection": True}
        self.assertEqual(r.routing_from_config(mixtral, defaults=defaults),
                         r.Routing(score_before_selection=True, renormalize=True))
        for missing in defaults:
            with self.assertRaises(ValueError):
                r.routing_from_config(mixtral, defaults={k: v for k, v in defaults.items() if k != missing})
        with self.assertRaises(ValueError):
            r.routing_from_config(mixtral, defaults={**defaults, "n_gruop": 1})
        grouped = self.moe(n_routed_experts=64, num_experts_per_tok=4, n_group=8, topk_group=2, scoring_func="softmax",
                           norm_topk_prob=False, routed_scaling_factor=1.0)
        with self.assertRaises(ValueError):  # a group limit needs its ranking rule
            r.routing_from_config(grouped, defaults={"score_before_selection": True})


class CudaCaseTests(unittest.TestCase):
    def test_cuda_cases_are_written_from_the_reference(self):
        spec = importlib.util.spec_from_file_location(
            "crit_test_routing_cases", Path(__file__).resolve().parent / "inference" / "cuda" / "routing_cases.py")
        cases = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cases)
        lines = cases.render().splitlines()
        self.assertEqual(int(lines[0]), len(cases.CASES))


if __name__ == "__main__":
    unittest.main()
