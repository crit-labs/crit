"""The CUDA crit builds outside kernels/ (a plugin's or Core's own native code, built by crit_plugins.native at load;
token-sampling's law, the expert cache's directory and the expert routes' CPU share are crit's kernels
token-sampling, expert-directory and cpu-routes now, checked as the kernels are),
compiled for every architecture crit's kernels claim, with no GPU: what `crit kernels gates --limits-only` does for
the kernels, so these sources stop being checked on one GPU only (the assessment's section 7). Each source compiles
to a cubin per architecture (nvcc -cubin), its functions' registers and spills read from ptxas, and a function past
255 registers or with local-memory spills over SPILL_BYTES fails its architecture.

python -m crit_kernels.outside [--arch 120a ...]: prints a line a source and architecture, exits 1 on a failure."""
import argparse
import os
import re
import subprocess
import sys
import sysconfig
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ARCHS = ("80", "86", "89", "90a", "100a", "120a")
SPILL_BYTES = 1024
# (source, its include folders, whether it includes torch's extension headers)
SOURCES = (
    ("inference/core/python/crit_plugins/host/handoff.cu", ("inference/core/python/crit_plugins/host",), False),
    ("inference/core/python/crit_plugins/host/handoff_module.cu", ("inference/core/python/crit_plugins/host",), True),
)


def _torch_includes():
    from torch.utils import cpp_extension
    return [*cpp_extension.include_paths(), sysconfig.get_paths()["include"]]


def compile_one(source, includes, torch_headers, arch, nvcc="nvcc"):
    """(ok, detail) of one source at one architecture."""
    flags = [nvcc, "-cubin", "-o", os.devnull, "-O3", "-std=c++20", "--expt-relaxed-constexpr", "-Xptxas", "-v",
             f"-gencode=arch=compute_{arch},code=sm_{arch}", "-DTORCH_EXTENSION_NAME=crit_outside_check"]
    flags += [f"-I{ROOT / i}" for i in includes]
    if torch_headers:
        flags += [x for i in _torch_includes() for x in ("-isystem", i)]
    run = subprocess.run(flags + [str(ROOT / source)], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    if run.returncode:
        errors = [line for line in run.stdout.splitlines() if "error" in line][:3]
        return False, "; ".join(errors) or run.stdout[-300:]
    worst_regs, worst_spill = 0, 0
    for line in run.stdout.splitlines():
        found = re.search(r"Used (\d+) registers", line)
        if found:
            worst_regs = max(worst_regs, int(found.group(1)))
        spill = re.search(r"(\d+) bytes spill stores", line)
        if spill:
            worst_spill = max(worst_spill, int(spill.group(1)))
    ok = worst_regs <= 255 and worst_spill <= SPILL_BYTES
    return ok, f"most registers {worst_regs}, most spill stores {worst_spill} bytes"


def check(archs=ARCHS, log=print, nvcc="nvcc"):
    """Every source at every architecture: True when all compile within the limits."""
    passed = True
    for source, includes, torch_headers in SOURCES:
        for arch in archs:
            ok, detail = compile_one(source, includes, torch_headers, arch, nvcc)
            passed = passed and ok
            log(f"{'pass' if ok else 'FAIL'} {source} sm_{arch}: {detail}")
    return passed


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--arch", action="append", help="architectures (nvcc names; default: " + ", ".join(ARCHS) + ")")
    parser.add_argument("--nvcc", default=os.environ.get("NVCC", "nvcc"))
    args = parser.parse_args(argv)
    return 0 if check(tuple(args.arch or ARCHS), nvcc=args.nvcc) else 1


if __name__ == "__main__":
    sys.exit(main())
