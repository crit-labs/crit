"""Routing conformance (crit_plugins.testing): every step of the phase x length grid multiplied as a model multiplies
it, a decode row's kernel never follows its length or the rows beside it, and a prompt row's follows its chunk's length
only where the facts do not say row_invariance prompt_rows. On CUDA: the kernels it chooses among run there."""
from contextlib import contextmanager
from unittest import mock

from crit_plugins.testing import PluginTestCase

PATHS = ("_decode_rows", "_tile", "_prompt_tile", "_prompt_rows")


class RoutingTests(PluginTestCase):
    def setUp(self):
        if self.device() != "cuda":
            self.skipTest("quantized-projections' kernels run on CUDA")
        import torch
        from crit_plugins.assemble import load_module
        self.torch = torch
        self.matrices = load_module(self.registry()["quantized-projections"]).device_matrices

    def record(self):
        """The outermost of the kernel paths each call takes (a path that calls another is its own), with its rows."""
        matrices = self.matrices

        @contextmanager
        def recording():
            calls, depth = [], [0]

            def wrap(name, real):
                def recorded(x, m, *args, **kwargs):
                    if not depth[0]:
                        mode = next((a for a in args if a in ("decode", "prompt")), "")
                        calls.append((f"{name} {mode}".strip(), x.shape[0]))
                    depth[0] += 1
                    try:
                        return real(x, m, *args, **kwargs)
                    finally:
                        depth[0] -= 1
                return recorded
            patches = [mock.patch.object(matrices, name, wrap(name, getattr(matrices, name))) for name in PATHS]
            for patch in patches:
                patch.start()
            try:
                yield calls
            finally:
                for patch in patches:
                    patch.stop()
        return recording

    def test_a_rows_kernel_follows_its_phase(self):
        torch = self.torch
        for options in ({}, {"prompt_gemm": "fixed_tiles"}, {"gemv": "mma"}):
            with self.subTest(options=options):
                plugin = self.assemble(options=options)["projections"]
                weight = torch.randn(256, 512, generator=torch.Generator().manual_seed(3)) * 0.05
                matrix = plugin.load({"weight": weight.cuda()}, "bf16")
                found = self.assertProjectionsRouteByPhase(plugin, matrix, 512, self.record(), device="cuda")
                self.assertEqual(len(found["decode"]), 1, (options, found["decode"]))
