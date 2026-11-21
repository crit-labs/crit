"""Choosing a kernel structure from what kernel.json declares and what the sweep measured (the assessment's section 3):
a plugin says what it computes (an operation, the level its rows must keep, the row count, the shape), and gets the
structure to run, instead of a hand-written tree that repeats the sweep's measurements in code.

- fitting(operation, asked, rows): the structures of crit's kernels that compute the operation (kernel.json's
  structure or kernel "operations"), keep the asked exactness (crit_kernels.exactness.keeps) and take the row count
  (their "rows" range), in the kernels' and structures' order.
- times(report, structure, key): what a sweep measured for a structure at a shape, {rows: us}, from the report `crit
  kernels sweep` writes beside its tuning (<id>.report.json: each table row's tuned_us, and its yardstick's
  library_us with library=True: the library the structure is measured against, a plugin's provider path).
- fastest(options, rows, default): the option with the least measured time at a row count (the time at the smallest
  timed count at or above it, else at the largest timed count), or over every count all options timed (rows None:
  their sums), ties to the earlier option; `default` (today's choice) where the options' times do not cover it.

The selector never measures: a plugin with no sweep report on this device keeps its default, and one with reports
takes what they measured on the model's own shapes."""
from . import build as kbuild
from . import exactness


def fitting(operation, asked=None, rows=None, kernels=None):
    """[(kernel id, structure id)] that compute `operation`, keep `asked` (an exactness level, None: any) and take
    `rows` rows (None: any count)."""
    found = []
    for kid in kernels or sorted(kbuild.manifests()):
        _, manifest = kbuild.find(kid, None)
        whole = manifest.get("operations") or []
        for sid, spec in manifest.get("structures", {}).items():
            if operation not in (spec.get("operations") or whole):
                continue
            if asked is not None and not exactness.keeps(spec, asked):
                continue
            lo, hi = (spec.get("rows") or [1, 1 << 62])[:2]
            if rows is not None and not lo <= int(rows) <= hi:
                continue
            found.append((kid, sid))
    return found


def times(report, structure, key, library=False):
    """{rows: microseconds} a sweep report measured for `structure` at the shape `key` (every key field equal):
    tuned_us, or library_us (its yardstick) with library. {} where the report has none."""
    if not report:
        return {}
    field = "library_us" if library else "tuned_us"
    out = {}
    for entry in (report.get("structures") or {}).get(structure, []):
        shape = entry.get("shape") or {}
        if field in entry and entry[field] and all(int(shape.get(f, -1)) == int(v) for f, v in key.items()):
            out[int(entry["rows"])] = float(entry[field])
    return out


def _at(measured, rows):
    above = [r for r in measured if r >= rows]
    return measured[min(above)] if above else measured[max(measured)]


def fastest(options, rows=None, default=None):
    """The name of the fastest of `options` ([(name, {rows: us})], in preference order) at `rows`, or over the row
    counts every option timed when rows is None; `default` where an option has no time there (or none share a
    count)."""
    if not options or any(not measured for _, measured in options):
        return default
    if rows is not None:
        scored = [(_at(measured, int(rows)), i, name) for i, (name, measured) in enumerate(options)]
    else:
        common = set.intersection(*(set(measured) for _, measured in options))
        if not common:
            return default
        scored = [(sum(measured[r] for r in common), i, name) for i, (name, measured) in enumerate(options)]
    return min(scored)[2]


def crossover(first, second, most):
    """The most rows (up to `most`) for which `first` ({rows: us}) is measured no slower than `second` at every row
    count both timed up to it: the calls `first` takes before `second` does; 0 where it is not."""
    best = 0
    for rows in sorted(set(first) & set(second)):
        if rows > most or fastest([("first", first), ("second", second)], rows) != "first":
            break
        best = rows
    return best


__all__ = ["crossover", "fastest", "fitting", "times"]
