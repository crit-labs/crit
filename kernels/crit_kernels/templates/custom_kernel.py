"""{name}: a {op} kernel on the workbench (see harness.py). Fill in the five functions below; the harness checks
candidate() against reference() on every shape in kernel.json and times it against its floor and yardstick().

Start from an existing kernel where one exists (kernels/SOURCES.md: the Hub, cuBLASLt, CUTLASS, FlashInfer,
flash-linear-attention, Triton, ...): put its source in src/ (`crit kernels new --from` copies a local one there with
its license), then change it.
"""
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent


def inputs(shape, columns, seed):
    """The operation's inputs for one entry of kernel.json's shapes and one column (token) count, as a dict."""
    raise NotImplementedError("{name}: inputs()")


def select(data, column):
    """The inputs of one column alone, when kernel.json declares column_invariant."""
    raise NotImplementedError("{name}: select()")


def reference(data):
    """The exact result in float64, from the same (already quantized) inputs the kernel reads."""
    raise NotImplementedError("{name}: reference()")


def candidate(data):
    """The kernel under work (load it from src/ with torch.utils.cpp_extension.load, Triton, or a library)."""
    raise NotImplementedError("{name}: candidate()")


def yardstick(data):
    """{yardstick}: the best existing provider for the same arithmetic, or None when none exists (say why in
    kernel.json's yardstick)."""
    return None


def work(shape, columns):
    """What one call must move ({{"bytes"}}) and compute ({{"flops"}}), for the floor."""
    raise NotImplementedError("{name}: work()")
