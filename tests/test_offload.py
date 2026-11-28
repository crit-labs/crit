"""crit plugins offload: the library plugins' options for experts larger than the device, derived from config.json and
crit Eval's placement, and a routing trace replayed through the expert cache's residency offline."""
import json
import random
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
for _folder in ("inference/core/python", "models/core/python", "cli"): sys.path.insert(0, str(ROOT / _folder))
from crit import offload  # noqa: E402

CONFIG = {"architectures": ["Qwen4ExpForConditionalGeneration"],
          "text_config": {"num_experts": 64, "num_experts_per_tok": 4, "moe_intermediate_size": 128,
                          "hidden_size": 256, "num_hidden_layers": 6},
          "quantization_config": {"quant_method": "modelopt", "quant_algo": "MIXED_PRECISION",
                                  "quantized_layers": {"layers.0.mlp.experts": {"quant_algo": "NVFP4"}}}}


def placement(name="sparse_device_host", cache_bytes=None, floors=None, tables=None):
    return {"scenario": name, "expert_cache_bytes": cache_bytes, "lookup_tables": tables,
            "plain_pass_floor_ms": floors or {}, "plain_pass_ceiling_tps": {}}


class OffloadTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.model = Path(temp.name)
        (self.model / "config.json").write_text(json.dumps(CONFIG), encoding="utf-8")
        self.size = offload.bundle_bytes("nvfp4", 256, 128)

    def plan(self, found, **kwargs):
        context = types.SimpleNamespace(placement=lambda *a, **k: found)
        kwargs.setdefault("vram_headroom_bytes", 1 << 30)   # not this machine's setting
        with mock.patch.dict(sys.modules, {"crit_eval.limits.context": context}):
            return offload.plan(self.model, {}, 4096, settings={}, **kwargs)

    def test_it_reads_the_experts_shape_and_format(self):
        self.assertEqual(offload.expert_shape(CONFIG), (6, 64, 4, 256, 128))
        self.assertEqual(offload.expert_format(CONFIG), "nvfp4")
        self.assertEqual(offload.expert_format({"quantization_config": {"quant_method": "awq", "bits": 4}}), "int4")
        self.assertEqual(offload.expert_format({}), "bf16")
        self.assertIsNone(offload.expert_shape({"hidden_size": 8}))

    def test_the_plugin_times_the_miss_path_where_the_cpus_floor_wins(self):
        value = self.plan(placement(cache_bytes=100 * self.size + 5, floors={"link": 30.0, "cpu": 17.0, "split": 15.0}),
                          cores=6)
        self.assertEqual(value["miss_path"], "cpu")
        self.assertEqual(value["selection"], {"experts": "expert-offload", "expert_source": "expert-tensors"})
        self.assertEqual(value["options"]["expert-offload"],
                         {"misses": "auto", "workers": 5})   # timed in the loaded engine
        self.assertEqual(value["cache"]["slots"], 100)
        preserve = self.plan(placement(cache_bytes=self.size, floors={"cpu": 1.0}), quality_mode="preserve",
                             vram_headroom_bytes=3 << 29)
        self.assertNotIn("prompt_arithmetic", preserve["options"]["expert-offload"])   # the plugin's own choice
        # The run's VRAM headroom is the device's to keep (Core's, from the candidate's configuration): no option.
        self.assertNotIn("margin_mib", preserve["options"]["expert-offload"])

    def test_the_link_alone_copies_misses_where_it_wins_or_the_cpu_cannot_compute_them(self):
        value = self.plan(placement(cache_bytes=10 * self.size, floors={"link": 10.0, "cpu": 17.0}))
        self.assertEqual(value["miss_path"], "link")
        self.assertEqual(value["options"]["expert-offload"]["misses"], "link")
        self.assertEqual(value["options"]["device-placement"]["off_device"], "")

    def test_lookup_tables_stay_off_the_device_beside_a_cache(self):
        value = self.plan(placement(cache_bytes=10 * self.size, floors={"link": 10.0},
                                    tables={"bytes": 3 << 30, "tier": "host"}))
        self.assertEqual(value["options"]["device-placement"]["off_device"], "tables.")
        self.assertTrue(any("token embeddings" in note and "embeddings slot" in note for note in value["notes"]))

    def test_recurrent_states_are_elastic_beside_the_cache_as_the_kv_pools_are(self):
        value = self.plan(placement(cache_bytes=10 * self.size, floors={"link": 10.0}))
        self.assertNotIn("sequential-recurrence", value["options"])
        text = {**CONFIG["text_config"], "full_attention_interval": 2}
        (self.model / "config.json").write_text(json.dumps({**CONFIG, "text_config": text}), encoding="utf-8")
        value = self.plan(placement(cache_bytes=10 * self.size, floors={"link": 10.0}))
        self.assertEqual(value["options"]["attention-state"], {"elastic": True})
        self.assertEqual(value["options"]["sequential-recurrence"], {"elastic": True})

    def test_experts_that_fit_need_no_cache(self):
        value = self.plan(placement("fits_device"))
        self.assertEqual(value["options"], {})

    def test_a_trace_is_replayed_at_several_sizes_and_copy_budgets(self):
        import numpy  # noqa: F401  (before plan() patches sys.modules: the replay's import is not taken back)
        rng = random.Random(3)
        hot = [rng.sample(range(64), 8) for _ in range(6)]   # skewed routing: most routes on 8 experts a layer
        trace = [(layer, rng.sample(hot[layer], 3) + rng.sample(range(64), 1)) for _ in range(200) for layer in range(6)]
        value = self.plan(placement(cache_bytes=60 * self.size, floors={"cpu": 1.0}), trace=trace)
        rows = value["trace"]["replay"]
        self.assertEqual({r["slots"] for r in rows}, {45, 60, 75})
        self.assertEqual({r["policy"] for r in rows}, {"recent", "static"})
        self.assertEqual({r["swaps"] for r in rows if r["policy"] == "recent"}, set(offload.SWAPS))
        best = value["trace"]["best"]
        self.assertGreater(best["hit_rate"], 0.5)
        self.assertEqual(value["options"]["expert-offload"]["swaps"], best["swaps"])
        more = {r["slots"]: r["hit_rate"] for r in rows if r["policy"] == "recent" and r["swaps"] == best["swaps"]}
        self.assertGreaterEqual(more[75], more[45])           # more slots never hit less on this routing
        self.assertIn("copies a round", offload.render(value))
        with self.assertRaisesRegex(ValueError, "outside"):
            offload.replay_trace([(9, [0])], 6, 64, 10)
        # With a profile the trace is also replayed with a pinned core: the hot experts pinned hit as the plain
        # rule does here, and a core of experts the trace never routes costs the hits their slots would have had.
        profile = [layer * 64 + e for layer in range(6) for e in hot[layer]]
        value = self.plan(placement(cache_bytes=60 * self.size, floors={"cpu": 1.0}), trace=trace, profile=profile)
        pinned = value["trace"]["pinned"]
        self.assertEqual([row["pinned"] for row in pinned], [0, 15, 30, 45])
        self.assertGreaterEqual(pinned[1]["hit_rate"], pinned[0]["hit_rate"] - 0.02)
        self.assertIn("15 pinned", offload.render(value))
        cold = [k for k in range(6 * 64) if k not in set(profile)]
        wrong = offload.replay_pinned(trace, 6, 64, 60, cold)
        self.assertLess(wrong[3]["hit_rate"], wrong[0]["hit_rate"] - 0.2)

    def test_a_trace_reads_from_json_or_json_lines(self):
        path = self.model / "trace.json"
        path.write_text(json.dumps([[0, [1, 2]], [1, [3]]]), encoding="utf-8")
        self.assertEqual(offload.read_trace(path), [(0, [1, 2]), (1, [3])])
        path.write_text("[0, [1, 2]]\n[1, [3]]\n", encoding="utf-8")
        self.assertEqual(offload.read_trace(path), [(0, [1, 2]), (1, [3])])


if __name__ == "__main__":
    unittest.main()
