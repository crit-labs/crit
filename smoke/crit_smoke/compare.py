"""Comparing rows from two paths of one engine, and the tolerances a comparison is judged by.

Prefill against prefill (TOLERANCES): a prompt's rows never depend on how long the prompt is, so a prefix computes
the whole prompt's first rows to within rounding on any engine. An engine whose GEMM kernels change with the row
count (cuBLAS's heuristics, padded buckets) moves them by tenths of a nat on a W4A4 engine and fails: it pins its
prompt kernels instead, since a limit above that floor would also pass a bug that moves a few rows. Decode against
prefill (DECODE): decoding may run other kernels than prefilling (GEMV against GEMM), and `loose`, for
quality mode speed, allows the rounding floor that leaves, which NVFP4 activations raise; `strict` (preserve) keeps one
arithmetic on every path. A stale cache, a position offset or a mask a decode step gets wrong moves rows by nats."""
import math

TOLERANCES = {
    # tie: top-1 margin below which a different choice counts as rounding; mean: the rows' mean delta (in nats); big: a delta that counts as large; big_share and disagree_share: the share of rows allowed to have a
    # large delta, or a different top-1 outside a tie, beyond `free` rows of each.
    "loose": {"tie": 0.25, "mean": 0.15, "big": 1.0, "big_share": 0.02, "disagree_share": 0.05, "free": 1},
    "strict": {"tie": 0.02, "mean": 0.02, "big": 0.25, "big_share": 0.0, "disagree_share": 0.0, "free": 0},
}
# Decode against prefill. On a Qwen3.8-27B NVFP4 engine (W4A4, FP8 KV, RTX 5090) that is right, decoding scored the
# same tokens 0.12 to 0.43 nats apart on average, with no row over 1 nat and at most 1 top-1 disagreement; one bf16
# ulp moved at random in one of its GEMMs moved 4% of its prompt rows over 1 nat. A Qwen3.8-Flash-Next engine whose
# prompt ring overwrote a staged layer measured 0.54 nats with 6 of 48 rows over 1 nat, and a mask leaking into decode
# moves the toy engine's rows (tests/test_smoke.py) by 2.8 to 3.5 nats.
DECODE = {
    "loose": {"tie": 0.25, "mean": 0.5, "big": 1.0, "big_share": 0.06, "disagree_share": 0.06, "free": 1},
    "strict": TOLERANCES["strict"],
}


def margin(row):
    """The row's top-1 log-probability less its top-2's (inf with one alternative)."""
    alternatives = row[2]
    return alternatives[0][1] - alternatives[1][1] if len(alternatives) > 1 else math.inf


def is_choice(row, token, tie):
    """Whether greedy decoding at this row may pick `token`: its top-1, or within `tie` of it."""
    if not row[2]:
        return False
    best = row[2][0][1]
    return any(t == token and best - p <= tie for t, p in row[2])


# Alternatives less likely than this (in nats) are left out of a row's delta: the tail of the ranking reorders
# under rounding without meaning anything.
LIKELY = -5.0


def delta(x, y):
    """How far apart two rows of one position are: the largest |delta logprob| of the text's token and of every
    likely alternative either row lists. An alternative one row lists and the other does not differs by at least
    its log-probability less the other row's last listed one."""
    largest = abs(x[1] - y[1])
    for a, b in ((x, y), (y, x)):
        listed = dict(b[2])
        floor = b[2][-1][1] if b[2] else None
        for token, logprob in a[2]:
            if logprob < LIKELY:
                continue
            if token in listed:
                largest = max(largest, abs(logprob - listed[token]))
            elif floor is not None and logprob > floor:
                largest = max(largest, logprob - floor)
    return largest


def compare(a, b, tolerance):
    """Compare rows of the same positions from two paths. Returns (ok, summary) with the mean and largest delta
    (`delta`) of the rows, the share of large deltas and of top-1 disagreements outside a tie, and
    the first row outside the tolerance."""
    if len(a) != len(b):
        return False, f"{len(a)} rows against {len(b)}"
    if not a:
        return True, "no rows"
    deltas, big, disagree, first = [], 0, 0, None
    for index, (x, y) in enumerate(zip(a, b)):
        if x[0] != y[0]:
            return False, f"row {index} scores token {x[0]} on one path and {y[0]} on the other"
        apart = delta(x, y)
        deltas.append(apart)
        large = apart > tolerance["big"]
        differs = bool(x[2] and y[2]) and x[2][0][0] != y[2][0][0] and not (
            is_choice(x, y[2][0][0], tolerance["tie"]) or is_choice(y, x[2][0][0], tolerance["tie"]))
        big += large
        disagree += differs
        if first is None and (large or differs):
            first = index
    mean = sum(deltas) / len(deltas)
    free = tolerance["free"]
    ok = (mean <= tolerance["mean"] and big <= tolerance["big_share"] * len(a) + free
          and disagree <= tolerance["disagree_share"] * len(a) + free)
    summary = (f"{len(a)} rows, mean delta {mean:.4f}, max {max(deltas):.3f}, {big} large, "
               f"{disagree} top-1 disagreements")
    if first is not None:
        summary += f", first at row {first}"
    return ok, summary


def same_tokens(first, rows, second, tie):
    """Whether two greedy replies agree: equal, or parting only where the first reply's row is a near-tie (`tie`;
    0 asks for equal tokens). Returns (ok, summary)."""
    for index, (x, y) in enumerate(zip(first, second)):
        if x != y:
            if tie > 0 and index < len(rows) and is_choice(rows[index], y, tie):
                return True, f"parted at token {index} of {len(first)} on a near-tie (margin {margin(rows[index]):.3f})"
            return False, f"parted at token {index} of {len(first)}: {x} against {y}" + (
                f" (margin {margin(rows[index]):.3f})" if index < len(rows) else "")
    if len(first) != len(second):
        return False, f"{len(first)} tokens against {len(second)}"
    return True, f"{len(first)} equal tokens"


def nll(rows):
    """Mean negative log-likelihood of the rows' own tokens, in nats."""
    return -sum(r[1] for r in rows) / len(rows) if rows else math.nan
