"""Host model of the batched engine's step layout, projection split and admission.

`arrange` lays out one mixed step: decode items first (one row each, read by the decode attention kernel), then
prompt chunks (read by the varlen prefill kernel); a row's kernel follows its phase, never the step's row counts.
Rows that end a prompt or decode produce logits. `plan_gemm` splits a projection's K dimension by the multiprocessor
count and the weight shape only, so every row is reduced in the same order whatever the step's row count.
`pages_needed` is what admission reserves for a request.
"""
import math

TILE = 64


def plan_gemm(n, k, multiprocessors):
    """(splits, tiles_per_split) for a projection with n outputs and k inputs (both multiples of 64)."""
    if n % TILE or k % TILE or n <= 0 or k <= 0:
        raise ValueError("projection dimensions must be positive multiples of 64")
    tiles_n, ktiles = n // TILE, k // TILE
    want = max(1, min(ktiles, multiprocessors // max(1, tiles_n)))
    per = math.ceil(ktiles / want)
    return math.ceil(ktiles / per), per


def split_dot(x, w, tiles_per_split):
    """A row's dot product reduced as the kernel does: within each split in K order, then splits in order."""
    span = tiles_per_split * TILE
    parts = [sum(a * b for a, b in zip(x[i:i + span], w[i:i + span])) for i in range(0, len(x), span)]
    total = 0.0
    for part in parts:
        total += part
    return total


def arrange(items):
    """items: [(request, phase, rows, ends_prompt)] with phase "decode" or "prefill". Returns the row layout
    [(request, kernel)], the decode row count and the rows that produce logits."""
    ordered = [i for i in items if i[1] == "decode"] + [i for i in items if i[1] == "prefill"]
    layout, logits = [], []
    for request, phase, rows, ends_prompt in ordered:
        if phase == "decode" and rows != 1:
            raise ValueError("a decode item stages one row")
        layout += [(request, phase)] * rows
        if phase == "decode" or ends_prompt:
            logits.append(len(layout) - 1)
    decode_rows = sum(1 for _, phase in layout if phase == "decode")
    return layout, decode_rows, logits


def pages_needed(prompt, output, page_tokens):
    """Pages admission reserves: the whole prompt and output, with no credit for shared prefix pages."""
    return math.ceil((prompt + output) / page_tokens)
