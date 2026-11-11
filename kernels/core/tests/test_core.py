"""The kernel core's device tests: each tests/<area>_tests.cu checks one header's blocks against host references on
this GPU. Each is built with nvcc three ways and run: for this device's architecture-specific target (sm_120a on an
RTX 5090: every instruction it has), for its plain target (sm_120: the paths without the specific instructions) and as
PTX for compute_80 (run through the driver's JIT with __CUDA_ARCH__ 800: the oldest paths), so one GPU checks every
path a block has. test_every_arch_builds compiles every test for each architecture the core supports."""
import os
import shutil
import subprocess
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
INCLUDE = HERE.parent / "include"
TESTS = sorted(p.stem for p in HERE.glob("*_tests.cu"))
ARCHS = ["80", "86", "89", "90a", "100a", "120a"]   # the core's architectures (nvcc names)
# torch's extension builder's flags, which every kernel build has: the blocks must compile under them
TORCH_FLAGS = ["-std=c++20", "--expt-relaxed-constexpr", "-D__CUDA_NO_HALF_OPERATORS__",
               "-D__CUDA_NO_HALF_CONVERSIONS__", "-D__CUDA_NO_BFLOAT16_CONVERSIONS__", "-D__CUDA_NO_HALF2_OPERATORS__"]

nvcc = shutil.which("nvcc") or (os.path.join(os.environ["CUDA_HOME"], "bin", "nvcc") if "CUDA_HOME" in os.environ
                                else None)
needs_nvcc = pytest.mark.skipif(not nvcc or not Path(nvcc).exists(), reason="needs nvcc")


def _device_targets():
    import torch
    if not torch.cuda.is_available():
        return []
    major, minor = torch.cuda.get_device_capability()
    plain = f"{major}{minor}"
    targets = [f"{plain}a", plain] if major >= 9 else [plain]
    return targets + ["80ptx"]


def _gencode(target):
    if target == "80ptx":
        return "-gencode=arch=compute_80,code=compute_80"
    return f"-gencode=arch=compute_{target},code=sm_{target}"


def _build(name, flags, out):
    subprocess.run([nvcc, "-O3", *TORCH_FLAGS, *flags, "-I", str(INCLUDE), "-I", str(HERE), str(HERE / f"{name}.cu"),
                    "-o", str(out)], check=True)


@needs_nvcc
@pytest.mark.parametrize("target", _device_targets() or ["none"])
@pytest.mark.parametrize("name", TESTS)
def test_blocks_on_device(name, target, tmp_path):
    if target == "none":
        pytest.skip("needs a CUDA device")
    binary = tmp_path / name
    _build(name, [_gencode(target)], binary)
    run = subprocess.run([str(binary)], capture_output=True, text=True)
    print(run.stdout)
    assert run.returncode == 0, run.stdout + run.stderr


@needs_nvcc
@pytest.mark.parametrize("name", TESTS)
def test_every_arch_builds(name, tmp_path):
    subprocess.run([nvcc, "-O3", *TORCH_FLAGS, *[_gencode(a) for a in ARCHS], "-I", str(INCLUDE), "-I", str(HERE),
                    "-c", str(HERE / f"{name}.cu"), "-o", str(tmp_path / f"{name}.o")], check=True)
