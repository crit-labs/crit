"""The kernel workbench's scaffolding on the host: folders, shapes from a checkpoint, kernel-builder projects to start
from (kernel-builder's own template, here a stand-in for `kernel-builder init`, or --from), the starter gate. crit-eval
measures the folders (crit eval workbench)."""
import json
import os
from pathlib import Path
import struct
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
for _folder in ("core/python", "kernels", "cli"):
    sys.path.insert(0, str(ROOT / _folder))
from crit_kernels import main
from crit_kernels import workbench


def safetensors(path, tensors):
    """A safetensors file of zeros: {name: (dtype, shape)}."""
    sizes = {"BF16": 2, "F8_E4M3": 1, "F32": 4, "U8": 1}
    header, offset = {}, 0
    for name, (dtype, shape) in tensors.items():
        size = sizes[dtype]
        for dim in shape:
            size *= dim
        header[name] = {"dtype": dtype, "shape": list(shape), "data_offsets": [offset, offset + size]}
        offset += size
    raw = json.dumps(header).encode()
    raw += b" " * (-len(raw) % 8)
    path.write_bytes(struct.pack("<Q", len(raw)) + raw + bytes(offset))


# A stand-in for `kernel-builder init --name ORG/NAME PATH`: the files of its template project the workbench relies on.
# In Python, so it runs on Windows too (through a command file there, by its first line elsewhere).
FAKE_INIT = """import sys
from pathlib import Path
args = sys.argv[1:]
if len(args) != 4 or args[:2] != ["init", "--name"]:
    sys.exit(3)
name, out = args[2], Path(args[3])
for folder in ("torch-ext", "kernel_cuda", ".git"):
    (out / folder).mkdir(parents=True, exist_ok=True)
(out / "build.toml").write_text(f'[general]\\nname = "{name}"\\n', encoding="utf-8")
(out / "torch-ext" / "torch_binding.cpp").write_text(
    '#include "registration.h"\\nTORCH_LIBRARY_EXPAND(TORCH_EXTENSION_NAME, ops) {}\\n', encoding="utf-8")
(out / "kernel_cuda" / "kernel.cu").write_text(f"// the template kernel of {name}\\n", encoding="utf-8")
"""


class WorkbenchTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        if os.name == "nt":   # Windows runs no script by its first line: a command file runs the stand-in
            script = self.root / "kernel_builder_init.py"
            script.write_text(FAKE_INIT, encoding="utf-8")
            tool = self.root / "kernel-builder.cmd"
            tool.write_text(f'@"{sys.executable}" "{script}" %*\r\n', encoding="utf-8", newline="")
        else:
            tool = self.root / "kernel-builder"
            tool.write_text(f"#!{sys.executable}\n" + FAKE_INIT, encoding="utf-8", newline="\n")
            tool.chmod(0o755)
        environment = patch.dict(os.environ, {"CRIT_KERNEL_BUILDER": str(tool)})
        environment.start()
        self.addCleanup(environment.stop)

    def starter(self):
        """A starter project as crit's workbench-starter holds it (a kernel project with its license)."""
        source = self.root / "workbench-starter"
        (source / "torch-ext").mkdir(parents=True)
        (source / "LICENSE").write_text("Apache-2.0\n", encoding="utf-8")
        (source / "build.toml").write_text('[general]\nname = "workbench-starter"\n', encoding="utf-8")
        (source / "starter.cu").write_text("// A correct, simple starting point\n", encoding="utf-8")
        (source / "torch-ext" / "torch_binding.cpp").write_text(
            '#include "registration.h"\nTORCH_LIBRARY_EXPAND(TORCH_EXTENSION_NAME, ops) {}\n', encoding="utf-8")
        return source

    def test_a_matmul_starts_with_its_binding_reference_and_yardstick(self):
        spec = workbench.new("down_gemv", "gemv", self.root, "fp8", ["5120,17408", "10240,5120"])
        folder = self.root / "down_gemv"
        self.assertEqual(sorted(p.name for p in folder.iterdir()), ["harness.py", "kernel.json", "kernel.py", "src"])
        self.assertEqual([s["shape"] for s in spec["shapes"]], [[5120, 17408], [10240, 5120]])
        self.assertEqual((spec["columns"], spec["column_invariant"], spec["rate"]), ([1, 2, 4, 8], True, "fp8"))
        self.assertIn("cuBLASLt", spec["yardstick"])
        kernel = (folder / "kernel.py").read_text(encoding="utf-8")
        self.assertIn('FORMAT = "fp8"', kernel)
        self.assertIn('name="crit_wb_down_gemv"', kernel)
        self.assertNotIn("{name}", kernel)
        self.assertIn("ext().run_fp8(", kernel)
        # src/ is kernel-builder's own template project, ready for the workbench's build (registration.h), no repo
        self.assertIn('"crit-labs/down-gemv"', (folder / "src" / "build.toml").read_text(encoding="utf-8"))
        self.assertTrue((folder / "src" / "registration.h").is_file())
        self.assertFalse((folder / "src" / ".git").exists())
        with self.assertRaisesRegex(ValueError, "exists"):
            workbench.new("down_gemv", "gemv", self.root, "fp8", ["16,32"])
        with self.assertRaisesRegex(ValueError, "--format bf16 or fp8 or nvfp4"):
            workbench.new("other", "gemm", self.root, "mxfp4", ["16,32"])
        with self.assertRaisesRegex(ValueError, "Give the shapes"):
            workbench.new("bare", "gemm", self.root, "bf16")

    def test_an_nvfp4_matmul_reads_stored_weights_against_16_bit_activations(self):
        model = self.root / "model"
        model.mkdir()
        (model / "config.json").write_text(json.dumps({"model_type": "llama"}), encoding="utf-8")
        safetensors(model / "model.safetensors", {
            "model.layers.0.mlp.down_proj.weight_packed": ("U8", (32, 32)),
            "model.layers.0.mlp.down_proj.weight_scale": ("F8_E4M3", (32, 4)),
            "model.layers.0.mlp.down_proj.weight_global_scale": ("F32", (1,)),
            "model.layers.0.mlp.up_proj.weight": ("BF16", (64, 32))})
        spec = workbench.new("down_gemv", "gemv", self.root, "nvfp4", model=model)
        self.assertEqual(spec["shapes"], [{"shape": [32, 64], "matrices": 1}])
        self.assertEqual(spec["rate"], "bf16")
        self.assertIn("widened to BF16", spec["yardstick"])
        kernel = (self.root / "down_gemv" / "kernel.py").read_text(encoding="utf-8")
        self.assertIn('FORMAT = "nvfp4"', kernel)
        self.assertIn("ext().run_nvfp4(", kernel)

    def test_any_other_operation_gets_the_harness_and_the_functions_to_fill_in(self):
        spec = workbench.new("decode_attention", "attention", self.root, shapes=["32768,128"], columns=[1, 4],
                             column_invariant=True)
        kernel = (self.root / "decode_attention" / "kernel.py").read_text(encoding="utf-8")
        for function in ("inputs", "select", "reference", "candidate", "yardstick", "work"):
            self.assertIn(f"def {function}(", kernel)
        self.assertEqual((spec["columns"], spec["column_invariant"], spec["rate"]), ([1, 4], True, None))
        self.assertTrue(spec["repeatable"])
        scatter = workbench.new("expert_scatter", "custom", self.root, shapes=["64,32"], repeatable=False)
        self.assertFalse(scatter["repeatable"])   # atomics: the check judges repeats within the tolerance

    def test_shapes_come_from_the_checkpoints_matrices_in_the_format(self):
        model = self.root / "model"
        model.mkdir()
        (model / "config.json").write_text(json.dumps({"model_type": "llama"}), encoding="utf-8")
        safetensors(model / "model.safetensors", {
            "model.layers.0.mlp.up_proj.weight": ("BF16", (64, 32)),
            "model.layers.1.mlp.up_proj.weight": ("BF16", (64, 32)),
            "model.layers.0.mlp.down_proj.weight": ("BF16", (32, 64)),
            "model.norm.weight": ("BF16", (32,))})
        spec = workbench.new("mlp", "gemm", self.root, "bf16", model=model)
        self.assertEqual(spec["shapes"], [{"shape": [64, 32], "matrices": 2}, {"shape": [32, 64], "matrices": 1}])

    def test_a_kernel_to_start_from_comes_with_its_license_and_a_hub_binding_builds(self):
        source = self.root / "hub-kernel"
        (source / "torch-ext").mkdir(parents=True)
        (source / "kernel.cu").write_text("// kernel\n", encoding="utf-8")
        (source / "torch-ext" / "torch_binding.cpp").write_text(
            '#include "registration.h"\nTORCH_LIBRARY_EXPAND(TORCH_EXTENSION_NAME, ops) {}\n', encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "holds no license file"):
            workbench.new("scan", "scan", self.root, shapes=["2048,128"], source=source)
        (source / "LICENSE").write_text("Apache-2.0\n", encoding="utf-8")
        spec = workbench.new("scan2", "scan", self.root, shapes=["2048,128"], source=source,
                             yardstick="lib:fla (chunk_gated_delta_rule)")
        folder = self.root / "scan2" / "src"
        self.assertTrue((folder / "registration.h").is_file())
        self.assertEqual(sorted(f["path"] for f in spec["from"]["files"]), ["LICENSE", "kernel.cu", "torch-ext/torch_binding.cpp"])
        self.assertEqual(spec["yardstick"], "lib:fla (chunk_gated_delta_rule)")

    def test_without_kernel_builder_a_project_comes_from_from(self):
        with patch.dict(os.environ, {"CRIT_KERNEL_BUILDER": ""}), patch("shutil.which", return_value=None):
            with self.assertRaisesRegex(ValueError, "install kernel-builder"):
                workbench.new("bare_gemv", "gemv", self.root, "bf16", ["64,128"])
            spec = workbench.new("from_gemv", "gemv", self.root, "bf16", ["64,128"], source=self.starter())
        self.assertIn("starter.cu", [f["path"] for f in spec["from"]["files"]])

    def test_a_starter_kernel_comes_only_from_a_runs_fourth_candidate_on(self):
        workbench.new("plain", "gemm", self.root, "bf16", ["64,128"])
        self.assertIn("the template kernel of crit-labs/plain",
                      (self.root / "plain" / "src" / "kernel_cuda" / "kernel.cu").read_text(encoding="utf-8"))
        run = self.root / "run"
        starter = self.starter()
        with self.assertRaisesRegex(ValueError, "--starter needs --run"):   # without --from: crit's starter
            workbench.new("s0", "gemm", self.root, "bf16", ["64,128"], starter=True)
        with self.assertRaisesRegex(ValueError, "--starter needs --run"):
            workbench.new("s1", "gemm", self.root, "bf16", ["64,128"], source=starter, starter=True)
        for number in (1, 2):
            (run / "candidates" / f"candidate-000{number}").mkdir(parents=True)
            (run / "candidates" / f"candidate-000{number}" / "candidate.json").write_text("{}", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "allowed from candidate 4 on"):
            workbench.new("s2", "gemm", self.root, "bf16", ["64,128"], source=starter, starter=True, run=run)
        (run / "candidates" / "candidate-0003").mkdir()
        (run / "candidates" / "candidate-0003" / "candidate.json").write_text("{}", encoding="utf-8")
        spec = workbench.new("s3", "gemm", self.root, "bf16", ["64,128"], source=starter, starter=True, run=run)
        self.assertTrue(spec["starter"])
        self.assertIn("A correct, simple starting point",
                      (self.root / "s3" / "src" / "starter.cu").read_text(encoding="utf-8"))
        with self.assertRaisesRegex(ValueError, "gemv and gemm in bf16 or fp8 only"):
            workbench.new("s4", "attention", self.root, shapes=["64,128"], source=starter, starter=True, run=run)
        with self.assertRaisesRegex(ValueError, "gemv and gemm in bf16 or fp8 only"):
            workbench.new("s5", "gemv", self.root, "nvfp4", ["64,128"], source=starter, starter=True, run=run)

    def test_the_cli_scaffolds(self):
        code = main(["new", "q_gemv", "--op", "gemv", "--format", "bf16", "--shape", "128,256", "--columns", "1,3",
                     "--root", str(self.root)])
        self.assertEqual(code, 0)
        spec = json.loads((self.root / "q_gemv" / "kernel.json").read_text(encoding="utf-8"))
        self.assertEqual(spec["columns"], [1, 3])
        with self.assertRaises(SystemExit):
            main(["new", "x"])   # --op is required


if __name__ == "__main__":
    unittest.main()
