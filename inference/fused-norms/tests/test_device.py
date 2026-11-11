"""fused-norms' device form on CUDA, through crit's kernel fused-norms where it builds here (with the engine's tuning,
CRIT_KERNEL_TUNING, or its default knobs) and through its torch fallback otherwise: every method against
the FP64 host form, each row's bits the same whatever rows share the call, strided views as their copies, out= and
in-place rope, a captured graph replaying to the eager bits, the fused forms (norms:fused) with the separate calls'
bits, and torch-norms' device form within one rounding."""
import importlib

from crit_plugins import kernels
from crit_plugins.assemble import load_module
from crit_plugins.testing import PluginTestCase

KERNEL = "fused-norms"


def conformance(test):
    module = load_module(test.registry()[test.plugin_id()])
    return importlib.import_module(module.__name__.rsplit(".", 1)[0] + ".norms.conformance")


class FusedNormsDeviceTests(PluginTestCase):
    def setUp(self):
        if self.device() != "cuda":
            self.skipTest("fused-norms runs on CUDA")
        self.norms = self.assemble()["norms"]
        self.conformance = conformance(self)

    def test_it_runs_the_kernel_where_it_builds_and_falls_back_otherwise(self):
        record = kernels.loaded()[KERNEL]
        self.assertIn(record["source"], kernels.SOURCES)
        if record["source"] == "fallback":
            self.assertIsNone(self.norms.kernel)
            self.assertTrue(record["reason"])
        else:   # tuned (the engine's tuning table) or built (the default knobs)
            self.assertEqual(record["source"] == "tuned", kernels.tuning_file(KERNEL) is not None, record)
            self.assertIsNotNone(self.norms.kernel)
            self.assertIs(self.norms.rope, self.norms.kernel.rope)
        self.assertEqual(record["plugins"], ["fused-norms"])

    def test_no_library_serves_a_call(self):
        # FlashInfer is the kernel's yardstick only: every call, the activation products and the norms without a
        # residual included, is the kernel's (or torch-norms' where the kernel does not build)
        self.assertEqual(self.norms.facts()["norms"], {"kernel": KERNEL if self.norms.kernel is not None
                                                       else "torch fallback"})
        for name in ("rms_norm", "rms_norm_heads", "silu_mul", "gelu_mul", "stream_add_norm"):
            self.assertIs(getattr(self.norms, name), getattr(self.norms.source, name), name)

    def test_every_method_follows_the_host_form(self):
        self.conformance.against_rows(self, self.norms, "cuda")

    def test_a_row_has_its_bits_alone_and_beside_others(self):
        self.conformance.rows_alone(self, self.norms, "cuda")
        self.conformance.odd_layouts(self, self.norms, "cuda")

    def test_views_outputs_and_dtypes(self):
        self.conformance.strided(self, self.norms, "cuda")
        self.conformance.outputs(self, self.norms, "cuda")

    def test_prompt_chunks_in_padded_buckets_keep_the_real_rows_bits(self):
        self.conformance.prompt_chunks(self, self.norms, "cuda")

    def test_a_captured_graph_replays_the_eager_bits(self):
        self.conformance.captured(self, self.norms)

    def test_the_fused_forms_give_the_separate_calls_bits(self):
        self.conformance.fused_forms(self, self.norms, "cuda")

    def test_it_agrees_with_torch_norms(self):
        self.conformance.compare(self, self.norms, self.reference()["norms"], "cuda")
