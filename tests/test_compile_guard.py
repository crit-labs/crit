"""crit_plugins.testing.compiles and PluginTestCase.assertNoCompileAfter: a block that JIT-compiles a Triton kernel
(a new specialization) or builds a torch extension is caught, one that only runs what is compiled is not, and the
hooks are put back afterwards."""
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "core" / "python"))

from crit_plugins.testing import compiles  # noqa: E402

try:
    import torch
    import triton
    import triton.language as tl
    CUDA = torch.cuda.is_available()
except ImportError:  # pragma: no cover - torch and triton are in crit's test environment
    torch = triton = None
    CUDA = False

if triton is not None:
    @triton.jit
    def _scaled(x, out, n, FACTOR: tl.constexpr, BLOCK: tl.constexpr):
        i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        tl.store(out + i, tl.load(x + i, mask=i < n) * FACTOR, mask=i < n)


class CompileGuardTests(unittest.TestCase):
    def test_a_torch_extension_build_is_caught_and_the_builder_put_back(self):
        if torch is None:
            self.skipTest("torch is not installed")
        import torch.utils.cpp_extension as extension
        original = extension._jit_compile
        extension._jit_compile = lambda name, *args, **kwargs: f"built {name}"   # stands in for a real build
        try:
            with compiles() as found:
                self.assertEqual(extension._jit_compile("crit_toy_kernels"), "built crit_toy_kernels")
            self.assertEqual(found, ["torch extension build or load of crit_toy_kernels"])
            self.assertEqual(extension._jit_compile("again"), "built again", "the builder is put back")
        finally:
            extension._jit_compile = original

    @unittest.skipUnless(CUDA, "no CUDA device for a Triton kernel")
    def test_a_new_triton_specialization_is_caught_and_a_compiled_one_is_not(self):
        x = torch.arange(64, device="cuda", dtype=torch.float32)
        out = torch.empty_like(x)
        _scaled[(1,)](x, out, 64, FACTOR=2, BLOCK=64)   # compiled before the block: a load's warm-up
        torch.cuda.synchronize()
        with compiles() as found:
            _scaled[(1,)](x, out, 64, FACTOR=2, BLOCK=64)
        self.assertEqual(found, [])
        with compiles() as found:
            _scaled[(1,)](x, out, 64, FACTOR=3, BLOCK=64)   # a new constexpr: a new specialization, compiled now
        self.assertEqual(len(found), 1, found)
        self.assertIn("_scaled", found[0])
        self.assertIn("FACTOR", found[0])
        self.assertIsNone(triton.knobs.runtime.jit_cache_hook, "the hook is put back")

        from crit_plugins.testing import PluginTestCase

        class Case(PluginTestCase):
            def runTest(self):
                pass
        case = Case()
        self.assertEqual(case.assertNoCompileAfter(lambda: 7), 7)
        with self.assertRaisesRegex(AssertionError, "compiled after load: Triton compile of .*_scaled"):
            case.assertNoCompileAfter(_scaled[(1,)], x, out, 64, FACTOR=4, BLOCK=64)


if __name__ == "__main__":
    unittest.main()
