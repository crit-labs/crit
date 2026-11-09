"""fused-norms' device form on CUDA, through the registry kernel when a build of crit-labs/fused-norms is there (a
published one, or a local build named in LOCAL_KERNELS) and through its torch fallback otherwise: every method against
the FP64 host form, each row's bits the same whatever rows share the call, strided views as their copies, out= and
in-place rope, a captured graph replaying to the eager bits, the fused forms (norms:fused) with the separate calls'
bits, and torch-norms' device form within one rounding."""
import importlib
import os

from crit_plugins import kernels
from crit_plugins.assemble import load_module
from crit_plugins.testing import PluginTestCase

KERNEL = "crit-labs/fused-norms"


def conformance(test):
    module = load_module(test.registry()[test.plugin_id()])
    return importlib.import_module(module.__name__.rsplit(".", 1)[0] + ".norms.conformance")


class FusedNormsDeviceTests(PluginTestCase):
    def setUp(self):
        if self.device() != "cuda":
            self.skipTest("fused-norms runs on CUDA")
        self.norms = self.assemble()["norms"]
        self.conformance = conformance(self)

    def test_it_runs_the_registry_kernel_where_a_build_is_named_and_falls_back_otherwise(self):
        record = kernels.loaded()[KERNEL]
        local = any(entry.startswith(KERNEL + "=") for entry in os.environ.get("LOCAL_KERNELS", "").split(":"))
        if local:
            self.assertEqual(record["source"], "local", record)
            self.assertIsNotNone(self.norms.kernel)
            self.assertIs(self.norms.rope, self.norms.kernel.rope)
        elif record["source"] == "fallback":
            self.assertIsNone(self.norms.kernel)
            self.assertTrue(record["reason"])
        self.assertEqual(record["plugins"], ["fused-norms"])

    def test_a_provider_takes_the_calls_whose_function_it_computes(self):
        torch = __import__("torch")
        try:
            import flashinfer  # noqa: F401
        except ImportError:
            self.skipTest("FlashInfer is not installed: the registry kernel serves every call")
        self.assertEqual(self.norms.facts()["norms"]["provider"], "flashinfer")
        x = torch.randn(4, 512, device="cuda", dtype=torch.bfloat16)
        w16, w32 = torch.randn(512, device="cuda", dtype=torch.bfloat16), torch.randn(512, device="cuda")
        # the provider for a 16-bit weight, the kernel (or torch) for an FP32 one: both the slot's function
        found = self.norms.rms_norm(x, w16, 1e-6, out_dtype=torch.float32)
        self.assertEqual(found.dtype, torch.float32)
        torch.testing.assert_close(self.norms.rms_norm(x, w16, 1e-6).float(),
                                   self.norms.rms_norm(x, w16.float(), 1e-6).float(), atol=0.02, rtol=0.01)
        plain = self.assemble(options={"flashinfer": False})["norms"]
        self.assertIsNone(plain.facts()["norms"]["provider"])
        self.assertIs(plain.rms_norm, plain.source.rms_norm)
        torch.testing.assert_close(self.norms.silu_mul(torch.cat([x, x], -1)).float(),
                                   plain.silu_mul(torch.cat([x, x], -1)).float(), atol=0.02, rtol=0.01)
        del w32

    def test_without_the_provider_too(self):
        plain = self.assemble(options={"flashinfer": False})["norms"]
        self.conformance.rows_alone(self, plain, "cuda")
        self.conformance.strided(self, plain, "cuda")
        self.conformance.fused_forms(self, plain, "cuda")

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
