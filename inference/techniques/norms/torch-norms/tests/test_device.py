"""torch-norms' device form against its FP64 host form: every method within FP32 rounding, each row's bits the same
whatever rows share the call, strided views as their copies, out= and in-place rope, and on CUDA a captured graph
replaying to the eager bits; on CUDA and on the CPU."""
import importlib

from crit_plugins.assemble import load_module
from crit_plugins.testing import PluginTestCase


def conformance(test):
    """fused-norms' code/conformance.py, imported through the plugin's sources (its subpackage .norms)."""
    module = load_module(test.registry()[test.plugin_id()])
    return importlib.import_module(module.__name__.rsplit(".", 1)[0] + ".norms.conformance")


class TorchNormsDeviceTests(PluginTestCase):
    def setUp(self):
        self.device()   # skips without torch
        self.norms = self.assemble()["norms"]
        self.conformance = conformance(self)

    def devices(self):
        return ["cpu"] + (["cuda"] if self.device() == "cuda" else [])

    def test_every_method_follows_the_host_form(self):
        for device in self.devices():
            with self.subTest(device=device):
                self.conformance.against_rows(self, self.norms, device)

    def test_a_row_has_its_bits_alone_and_beside_others(self):
        for device in self.devices():
            with self.subTest(device=device):
                self.conformance.rows_alone(self, self.norms, device)
                self.conformance.odd_layouts(self, self.norms, device)

    def test_views_outputs_and_dtypes(self):
        for device in self.devices():
            with self.subTest(device=device):
                self.conformance.strided(self, self.norms, device)
                self.conformance.outputs(self, self.norms, device)

    def test_prompt_chunks_in_padded_buckets_keep_the_real_rows_bits(self):
        for device in self.devices():
            with self.subTest(device=device):
                self.conformance.prompt_chunks(self, self.norms, device)

    def test_the_fused_forms_give_the_separate_calls_bits(self):
        for device in self.devices():
            with self.subTest(device=device):
                self.conformance.fused_forms(self, self.norms, device)

    def test_a_captured_graph_replays_the_eager_bits(self):
        if self.device() != "cuda":
            self.skipTest("CUDA graphs")
        self.conformance.captured(self, self.norms)
