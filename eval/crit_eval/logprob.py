"""Device reference for full-vocabulary distribution comparison: probe/logprob_compare.cu.

`enqueue_row_compare` computes each row's log-softmax over the valid outputs of a strided FP32 logits
matrix and, as requested, writes the target NLL, stores the log-probabilities as a reference, or writes
KL(reference || row) from stored reference log-probabilities. One block per row with fixed-order
reductions: deterministic, not tuned. An engine compiles it next to its own forward path (or copies it
into its log-softmax) and summarizes the rows on the host with `metrics.Summary`. `build` compiles the
kernel with nvcc together with a caller's host program, as the probes are built; no CMake project needed.
"""
import os
from pathlib import Path
import subprocess

from . import EVAL_DIR
from .probe import host_compiler

SOURCE = EVAL_DIR / "probe" / "logprob_compare.cu"
HEADER = SOURCE.with_suffix(".h")


def build(nvcc, directory, main, name="logprob_compare"):
    """Compile SOURCE with the host program `main` (a .cu or .cpp file including logprob_compare.h)."""
    binary = Path(directory) / (name + (".exe" if os.name == "nt" else ""))
    command = [nvcc, *host_compiler(), "-O2", "-std=c++20", "-I", str(SOURCE.parent), "-o", str(binary),
               str(SOURCE), str(main)]
    done = subprocess.run(command, capture_output=True, text=True)
    if done.returncode != 0:   # nvcc reports a missing host compiler on stdout
        raise SystemExit("building the log-probability comparison failed:\n" + (done.stdout + done.stderr)[-4000:])
    return binary
