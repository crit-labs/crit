"""crit_plugins.testing.compiles and PluginTestCase.assertNoCompileAfter: a block that JIT-compiles a Triton kernel
(a new specialization) or builds a torch extension is caught, one that only runs what is compiled is not, and the
hooks are put back afterwards."""
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "inference" / "core" / "python"))

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

    @triton.jit
    def _counted(x, out, n, BLOCK: tl.constexpr):   # n follows the request and is specialized: a bug of its own
        i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        tl.store(out + i, tl.load(x + i, mask=i < n) + 1, mask=i < n)

    @triton.jit(do_not_specialize=["n"])
    def _counted_once(x, out, n, BLOCK: tl.constexpr):
        i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        tl.store(out + i, tl.load(x + i, mask=i < n) + 1, mask=i < n)


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

    def test_watches_nest_and_end_in_any_order(self):
        # Core's loader keeps a watch for an engine's life while a test opens its own: both see what compiles while
        # they run, and the hook goes back only when the last ends.
        if torch is None:
            self.skipTest("torch is not installed")
        import torch.utils.cpp_extension as extension
        from crit_plugins.compiles import Watch
        original = extension._jit_compile
        stand_in = extension._jit_compile = lambda name, *args, **kwargs: f"built {name}"
        try:
            engine = Watch()
            with compiles() as found:
                extension._jit_compile("inner")
            extension._jit_compile("outer")
            self.assertEqual(found, ["torch extension build or load of inner"])
            self.assertEqual(engine.report(), {"count": 2, "first": ["torch extension build or load of inner",
                                                                     "torch extension build or load of outer"]})
            self.assertIsNot(extension._jit_compile, stand_in, "a watch still runs")
            first = Watch()
            engine.stop()
            extension._jit_compile("late")
            self.assertEqual(first.stop(), ["torch extension build or load of late"])
            self.assertIs(extension._jit_compile, stand_in, "the builder is put back once no watch runs")
            self.assertEqual(engine.stop(), engine.found, "a second stop returns the same")
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


    def test_a_kernel_compiled_again_for_a_request_s_count_names_the_argument(self):
        # A token count that is divisible by 16 at load and 1 on a request: the same constexprs, another kernel.
        from crit_plugins import compiles as watched
        if torch is None or not CUDA:
            self.skipTest("Triton compiles on a CUDA device")
        watched.record()                        # as Core's loader does: every compile from the load's start
        self.addCleanup(watched.record, False)
        x = torch.arange(64, device="cuda", dtype=torch.float32)
        out = torch.empty_like(x)
        _counted[(1,)](x, out, 64, BLOCK=64)    # the load's
        with compiles() as found:
            _counted[(1,)](x, out, 1, BLOCK=64)
            _counted[(1,)](x, out, 37, BLOCK=64)
        self.assertEqual(len(found), 2, found)
        self.assertIn("n is now 1, was divisible by 16", found[0])
        self.assertIn("do_not_specialize", found[0])
        self.assertIn("n is now any", found[1])
        _counted_once[(1,)](x, out, 64, BLOCK=64)
        with compiles() as found:                # left alone: one kernel for every count
            _counted_once[(1,)](x, out, 1, BLOCK=64)
            _counted_once[(1,)](x, out, 37, BLOCK=64)
        self.assertEqual(found, [])
        with compiles() as found:
            _counted_once[(1,)](x, out, 37, BLOCK=32)
        self.assertIn("constexprs this process never compiled it with", found[0])
        self.assertIn("BLOCK=32", found[0])
        self.assertTrue(watched._compiled)

    def test_recording_imports_no_jit_and_hooks_it_when_imported(self):
        # Core depends on no Triton: record() imports nothing, and the hooks go in when whoever brings Triton or torch's
        # extension builder imports it (a kernel package, a plugin), before any of the load's kernels compile.
        if triton is None:
            self.skipTest("no triton here")
        import os
        import subprocess
        code = ("import sys\n"
                "from crit_plugins import compiles\n"
                "compiles.record()\n"
                "assert 'triton' not in sys.modules, 'record() imported triton'\n"
                "assert 'triton' not in compiles._installed\n"
                "import triton\n"
                "assert 'triton' in compiles._installed, compiles._installed\n"
                "import torch.utils.cpp_extension\n"
                "assert 'extension' in compiles._installed, compiles._installed\n"
                "print('hooked')\n")
        path = os.pathsep.join([str(Path(__file__).resolve().parents[1] / "inference" / "core" / "python"),
                                os.environ.get("PYTHONPATH", "")])
        found = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=600,
                               env={**os.environ, "PYTHONPATH": path})
        self.assertEqual(found.stdout.strip().splitlines()[-1:], ["hooked"], found.stderr[-3000:])


if __name__ == "__main__":
    unittest.main()
