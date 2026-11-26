"""Each kernel.json's `launches` against its sources, and the ranking of a kernel trace by them: every CUDA kernel of a
kernel's sources outside crit's namespaces and every Triton kernel (a jit function named *_kernel) is declared by
name, every crit namespace that holds a CUDA kernel by a prefix, so a trace's time reaches the kernel that launched
it."""
from pathlib import Path
import re
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from crit_kernels.build import manifests  # noqa: E402
from crit_kernels.launches import declared, owners, plain, rank  # noqa: E402

GLOBAL = re.compile(r"__global__\s+(?:static\s+)?void\s+(?:__launch_bounds__\([^)]*\)\s*)?(\w+)\s*\(")
NAMESPACE = re.compile(r"namespace\s+(crit[:\w]*)\s*\{")
TRITON = re.compile(r"@triton\.(?:jit|autotune)[^\n]*\n(?:@[^\n]*\n)*def\s+(\w+_kernel)\b")


def launched(folder):
    """(names, namespaces) of the kernels a kernel's sources define (its tests and bench left out)."""
    names, namespaces = set(), set()
    for path in folder.rglob("*"):
        if "tests" in path.relative_to(folder).parts or path.name == "bench.py" or not path.is_file():
            continue
        if path.suffix in (".cu", ".cuh", ".inc"):
            text = path.read_text(encoding="utf-8", errors="ignore")
            for found in GLOBAL.finditer(text):
                inside = list(NAMESPACE.finditer(text[:found.start()]))
                (namespaces.add(inside[-1].group(1) + "::") if inside else names.add(found.group(1)))
        elif path.suffix == ".py":
            names.update(TRITON.findall(path.read_text(encoding="utf-8", errors="ignore")))
    return names, namespaces


class LaunchesTests(unittest.TestCase):
    def test_every_kernel_declares_the_kernels_its_sources_launch(self):
        launches = declared()
        for kernel, (folder, _) in manifests().items():
            with self.subTest(kernel=kernel):
                self.assertIn(kernel, launches)
                names, namespaces = launched(Path(folder))
                entries = set(launches[kernel])
                self.assertEqual(sorted(names - entries), [], "kernels its launches do not name")
                self.assertEqual(sorted(namespaces - entries), [], "namespaces its launches do not name")

    def test_a_traced_name_is_read_without_templates_and_anonymous_namespaces(self):
        self.assertEqual(plain("crit::kernels::mma_rows::(anonymous namespace)::mma_kernel<2, 0, 1, 1>"),
                         "crit::kernels::mma_rows::mma_kernel")
        self.assertEqual(plain("(anonymous namespace)::fill_kernel<uint4>"), "fill_kernel")
        launches = {"a": ["crit::x::"], "b": ["fill_kernel", "_merge_kernel"], "c": ["_merge_kernel"]}
        self.assertEqual(owners("crit::x::(anonymous namespace)::k<1>", launches), ["a"])
        self.assertEqual(owners("_merge_kernel", launches), ["b", "c"])
        self.assertEqual(owners("cutlass::Kernel<x>", launches), [])

    def test_a_trace_ranks_the_kernels_by_their_share_and_names_the_rest(self):
        launches = {"mma-rows": ["crit::kernels::mma_rows::"], "expert-directory": ["fill_kernel"],
                    "decode-gemv": ["crit::kernels::decode_gemv::"], "lat": ["_merge_kernel"],
                    "paged": ["_merge_kernel", "_sparse_kernel"]}
        trace = {"phases": {
            "decode": {"kernels": [{"kernel": "(anonymous namespace)::fill_kernel<uint4>", "ms": 55.0},
                                   {"kernel": "crit::kernels::mma_rows::(anonymous namespace)::mma_kernel<2>", "ms": 30.0},
                                   {"kernel": "_sparse_kernel", "ms": 5.0}, {"kernel": "_merge_kernel", "ms": 2.0},
                                   {"kernel": "at::native::elementwise_kernel<128>", "ms": 8.0}]},
            "prefill": {"kernels": [{"kernel": "crit::kernels::mma_rows::prompt_kernel<1>", "ms": 100.0}]}}}
        found = rank(trace, launches)
        self.assertEqual([row["kernel"] for row in found["kernels"]], ["mma-rows", "expert-directory", "paged"])
        self.assertEqual(found["kernels"][0]["phases"], {"decode": 0.3, "prefill": 1.0})
        self.assertEqual(found["absent"], ["decode-gemv", "lat"])
        self.assertEqual(found["ambiguous"], [{"name": "_merge_kernel", "kernels": ["lat", "paged"], "ms": 2.0}])
        self.assertEqual(found["other"][0]["name"], "at::native::elementwise_kernel")


if __name__ == "__main__":
    unittest.main()
