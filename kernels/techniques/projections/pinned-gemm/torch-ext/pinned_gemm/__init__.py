"""Prompt GEMMs whose rows do not depend on the row count: one cuBLASLt algorithm per GEMM shape and format, chosen
once (pin) and run for every row count (Pinned.mm), from crit's recorded-selection technique (pinned GEMMs).

cuBLAS's heuristics pick a GEMM's algorithm by its row count, and algorithms split a row's sum differently (split-K,
stream-K, other K tiles), so a prompt row computes other bits when the prompt is longer, is cut into other chunks or
is padded to a captured bucket. pin() takes, of the split-K-free algorithms cuBLASLt's heuristic lists for the shape
at REP_ROWS rows, the first that is row exact (a row's bits the same at every count of LENGTHS and at the offsets of
OFFSETS, on random operands of the format), and every call of any row count runs it. The choice depends on the
shape, the format, the device and the cuBLASLt version only (no timing), so every process makes the same one. Short
calls take the same algorithm too: another algorithm that agrees with it on test operands may still sum in another
order (outputs rounded to 16 bits hide most of the difference), so none is swapped in.

Formats are INPUTS[-SCALE]:OUTPUT (parse): INPUTS bf16, fp16, fp32 (no TF32), fp8e4m3, fp8e5m2, mxfp8 (E4M3 with one
UE8M0 scale per 32 along K), mxfp4 (E2M1, UE8M0 per 32) or nvfp4 (E2M1, UE4M3 per 16, an FP32 global scale alpha);
SCALE, for FP8 only, tensor (one FP32 element an operand) or row (FP32 vectors of a's M rows and b's N rows); OUTPUT
bf16, fp16 or fp32. a [M, K] row major (the rows) times b [N, K] row major (a weight) gives out [M, N] = alpha
(scale_a a) (scale_b b)^T with FP32 accumulation; out may be N columns of a wider row-major matrix. FP4 is packed two
a byte (uint8 or float4_e2m1fn_x2, the first element in the low nibble); block scales go in cuBLASLt's blocked layout
(blocked_scales: 128-row by 4-column tiles, rows padded to 128 and columns to 4).

A Pinned call allocates nothing when given `out` and runs on the current stream without host synchronization, so a
CUDA graph captures it; a plan checks a row count with cuBLASLt (host only) the first time it meets it.

Knobs (kernel.json, tuned per shape by crit kernels sweep; tuned()): REP_ROWS, the row count whose heuristic list is
searched, and CHOICE, which of that list's row-exact split-K-free algorithms is pinned (0: the first, as without a
tuning). Both pick the algorithm, and with it a row's summation order, so they are order knobs: a shape keeps one value
of each for every row count, and every call of the shape runs the one algorithm."""
import torch

from ._crit_tuning import knobs as _knobs
from ._ops import ops

INPUTS = {"fp16": "fp16", "bf16": "bf16", "fp32": "fp32", "fp8e4m3": "e4m3", "fp8e5m2": "e5m2", "mxfp8": "e4m3",
          "mxfp4": "e2m1", "nvfp4": "e2m1"}
BLOCKS = {"mxfp8": ("vec32", 32), "mxfp4": ("vec32", 32), "nvfp4": ("vec16", 16)}
SCALES = ("tensor", "row")
OUTPUTS = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
WORKSPACE = 32 << 20     # bytes of workspace a pinned algorithm may take
REP_ROWS = 2048          # the row count whose heuristic list gives the pinned algorithm (the REP_ROWS knob's default)
LENGTHS = (1, 2, 7, 16, 17, 63, 64, 65, 127, 128, 129, 255, 256, 257, 511, 512, 513, 1025, 2048, 2049, 3001, 4096)
OFFSETS = ((1, 17), (7, 40), (100, 200), (129, 513), (1000, 1025))   # (first row, rows): slices of the longest call
BUDGET = 512 << 20       # bytes of output the longest check call may take

STRUCTURES = {"e2m1": "lt-block", "e4m3": "lt-fp8", "e5m2": "lt-fp8", "bf16": "lt-dense", "fp16": "lt-dense",
              "fp32": "lt-dense"}   # kernel.json's structure for a format, by a's element type (block scales: lt-block)
BITS = {"e2m1": 4, "e4m3": 8, "e5m2": 8, "bf16": 16, "fp16": 16, "fp32": 32}

__all__ = ["Pinned", "UnsupportedFormat", "blocked_scales", "candidates", "describe", "operands", "parse", "pin",
           "structure", "tuned", "version"]


class UnsupportedFormat(RuntimeError):
    """cuBLASLt's heuristic lists no split-K-free algorithm for the shape and format on this device."""


def parse(fmt):
    """The format's parts: name, a and b (element types), out, scale (none, tensor, row, vec32, vec16) and block
    (elements a block scale covers, 0 for none)."""
    if not isinstance(fmt, str):
        raise ValueError("a pinned GEMM format is a string such as nvfp4:bf16")
    inputs, _, out = fmt.partition(":")
    inputs, _, suffix = inputs.partition("-")
    pair = inputs.split("x") if inputs.count("x") == 1 and not inputs.startswith("mx") else [inputs, inputs]
    if out not in OUTPUTS or any(p not in INPUTS for p in pair):
        raise ValueError(f"pinned GEMM formats are INPUTS[-SCALE]:OUTPUT with inputs from {', '.join(sorted(INPUTS))} "
                         f"and outputs from {', '.join(OUTPUTS)}, not {fmt!r}")
    fp8 = all(p.startswith("fp8") for p in pair)
    if pair[0] != pair[1] and not fp8:
        raise ValueError(f"only FP8 inputs mix element types, not {fmt!r}")
    if suffix and (not fp8 or suffix not in SCALES):
        raise ValueError(f"scale suffixes ({', '.join(SCALES)}) are for FP8 inputs, not {fmt!r}")
    scale, block = BLOCKS.get(pair[0], (suffix or "none", 0))
    return {"name": fmt, "a": INPUTS[pair[0]], "b": INPUTS[pair[1]], "out": out, "scale": scale, "block": block}


def _native(f):
    return [f["a"], f["b"], f["out"], f["scale"]]


def structure(fmt, n, k):
    """The format's structure in kernel.json (lt-dense: 16- and 32-bit inputs; lt-fp8: FP8 unscaled, tensor or row
    scales; lt-block: MXFP8, MXFP4 and NVFP4 block scales) and the shape key its tuning is keyed by: N, K, IN (a's
    element bits) and OUT (the output's bits)."""
    f = parse(fmt)
    sid = "lt-block" if f["block"] else STRUCTURES[f["a"]]
    return sid, {"N": int(n), "K": int(k), "IN": BITS[f["a"]], "OUT": OUTPUTS[f["out"]].itemsize * 8}


def tuned(fmt, n, k):
    """The knobs pin() uses for an [N, K] GEMM of the format: {"REP_ROWS", "CHOICE"} from the build's tuning table
    (the defaults, 2048 and 0, without one). Both are order knobs, the same for every row count of a shape."""
    sid, key = structure(fmt, n, k)
    return _knobs(sid, key, 1)


def version():
    """The cuBLASLt version (cublasLtGetVersion) of the library this process loaded."""
    return int(ops.version())


def candidates(fmt, n, k, rows, workspace=WORKSPACE):
    """How many split-K-free algorithms the heuristic lists for an [N, K] GEMM of the format at `rows` rows."""
    return int(ops.candidates(_native(parse(fmt)), n, k, rows, workspace))


def describe(plan):
    """A plan's algorithm configuration: algorithm id, tile, stages, split-K count, reduction scheme, swizzle,
    custom option, inner shape, cluster shape."""
    return [int(v) for v in ops.describe(plan)]


def blocked_scales(scales):
    """Row-major [rows, K / block] one-byte scales in cuBLASLt's blocked layout, flat and padded with zeros (the
    dtype kept): 128 x 4 tiles in row-major tile order, row r and column c of a tile at (r % 32) * 16 + (r // 32) * 4
    + c."""
    rows, cols = scales.shape
    pr, pc = -(-rows // 128) * 128, -(-cols // 4) * 4
    padded = torch.zeros(pr, pc, dtype=torch.uint8, device=scales.device)
    padded[:rows, :cols] = scales.contiguous().view(torch.uint8)
    tiles = padded.view(pr // 128, 128, pc // 4, 4).permute(0, 2, 1, 3)
    return tiles.reshape(-1, 4, 32, 4).transpose(1, 2).reshape(-1).view(scales.dtype)


def _values(element, rows, k, g, device):
    if element == "e2m1":
        codes = torch.randint(0, 16, (rows, k), device=device, generator=g, dtype=torch.uint8)
        return codes[:, 0::2] | (codes[:, 1::2] << 4)
    x = torch.randn(rows, k, device=device, generator=g)
    if element in ("e4m3", "e5m2"):
        x = (x * 60).clamp(-448, 448)
    return x.to({"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32, "e4m3": torch.float8_e4m3fn,
                 "e5m2": torch.float8_e5m2}[element])


def operands(fmt, n, k, rows, device=None, seed=0):
    """Random a [rows, K] and b [N, K] of the format and their scales {"a", "b", "alpha"}: row-major block scales
    ([rows, K / block] and [N, K / block] bytes), FP32 row vectors, one-element FP32 tensors, or None."""
    f = parse(fmt)
    device = torch.device(device or "cuda")
    g = torch.Generator(device=device).manual_seed(seed)
    a, b = _values(f["a"], rows, k, g, device), _values(f["b"], n, k, g, device)
    scales = {"a": None, "b": None, "alpha": None}
    if f["scale"] == "tensor":
        scales.update(a=torch.tensor([0.02], device=device), b=torch.tensor([0.005], device=device))
    elif f["scale"] == "row":
        scales.update(a=torch.rand(rows, device=device, generator=g) * 0.02 + 0.01,
                      b=torch.rand(n, device=device, generator=g) * 0.005 + 0.0025)
    elif f["scale"] == "vec32":
        if f["a"] == "e4m3":
            a, b = (x.float().div(15).to(torch.float8_e4m3fn) for x in (a, b))
        scales.update({s: torch.randint(124, 129, (r, k // 32), device=device, generator=g, dtype=torch.uint8)
                       for s, r in (("a", rows), ("b", n))})
    elif f["scale"] == "vec16":
        scales.update({s: (torch.rand(r, k // 16, device=device, generator=g) * 0.75 + 0.25).to(
            torch.float8_e4m3fn).view(torch.uint8) for s, r in (("a", rows), ("b", n))})
        scales["alpha"] = torch.tensor([0.5], device=device)
    return a, b, scales


def _call_scales(f, scales, first, rows):
    """Pinned.mm's scale_a for rows [first, first + rows) of operands()' a."""
    sa = scales["a"]
    if f["scale"] == "row":
        return sa[first:first + rows].contiguous()
    if f["block"]:
        return blocked_scales(sa[first:first + rows])
    return sa


def _weight_scale(f, scales):
    """operands()' b scale as a call takes it."""
    return blocked_scales(scales["b"]) if f["block"] and scales["b"] is not None else scales["b"]


class Pinned:
    """An [N, K] GEMM of a format pinned to one cuBLASLt algorithm (plan; algo: describe())."""

    def __init__(self, fmt, n, k, plan, workspace, device):
        self.format, self.n, self.k, self.plan = fmt, n, k, plan
        self._f = parse(fmt)
        self.dtype = OUTPUTS[self._f["out"]]
        self.workspace = workspace
        self.algo = describe(plan)
        # alpha and beta of a block-scaled plan (read from device memory) and the workspace, made here so no call
        # allocates (a captured call must not).
        self._one = torch.ones(1, dtype=torch.float32, device=device)
        self._zero = torch.zeros(1, dtype=torch.float32, device=device)
        self._space = None

    def mm(self, a, b, scale_a=None, scale_b=None, alpha=None, out=None, workspace=None):
        """out [M, N] = alpha (scale_a a) (scale_b b)^T in the plan's output type (allocated unless given)."""
        if out is None:
            out = torch.empty(a.shape[0], self.n, dtype=self.dtype, device=a.device)
        beta = None
        if self._f["scale"] == "vec16":
            alpha = self._one if alpha is None else alpha
            beta = self._zero
        if workspace is None:
            if self._space is None:
                self._space = torch.empty(max(self.workspace, 1), dtype=torch.uint8, device=a.device)
            workspace = self._space
        ops.run(self.plan, a, b, out, scale_a, scale_b, alpha, beta, workspace)
        return out


def pin(fmt, n, k, device=None, workspace=WORKSPACE, rep_rows=None, lengths=LENGTHS, offsets=OFFSETS, seed=0,
        buffer=None, weight=None, budget=BUDGET, choice=None):
    """The Pinned GEMM for [N, K] in `fmt` on `device`: of the split-K-free algorithms the heuristic lists at
    rep_rows rows (in its order, duplicates once), the first whose rows of a call of every count in `lengths` equal
    the first rows of the longest call bit for bit, and whose slices at `offsets` equal the longest call's rows, on
    random operands (seed); with `choice` c, the (c + 1)-th such algorithm (the last row-exact one where fewer are
    listed). rep_rows and choice default to the build's tuning for the shape (tuned(): 2048 and 0 without one).
    weight: (b, scale_b) as calls will pass them, the weight the checks multiply instead of a random one (no copy of
    a large matrix). The longest check is cut to the rows whose output fits `budget` bytes (and the counts past it
    left out). buffer: the workspace calls use (at least `workspace` bytes; one of its own otherwise). Raises
    UnsupportedFormat where the heuristic lists none, RuntimeError where none is row exact."""
    f = parse(fmt)
    native = _native(f)
    if rep_rows is None or choice is None:
        knobs = tuned(fmt, n, k)
        rep_rows = knobs["REP_ROWS"] if rep_rows is None else rep_rows
        choice = knobs["CHOICE"] if choice is None else choice
    device = torch.device(device or "cuda")
    if device.index is None:
        device = torch.device("cuda", torch.cuda.current_device())
    with torch.cuda.device(device):
        listed = int(ops.candidates(native, n, k, rep_rows, workspace))
        if not listed:
            raise UnsupportedFormat(f"cuBLASLt {version()} lists no split-K-free algorithm for {fmt} ({n}x{k} at "
                                    f"{rep_rows} rows) on {torch.cuda.get_device_name(device)}")
        total = max(max(lengths), max((s + m for s, m in offsets), default=0))
        total = max(1, min(total, budget // (n * OUTPUTS[f["out"]].itemsize)))
        space = buffer if buffer is not None else torch.empty(max(workspace, 1), dtype=torch.uint8, device=device)
        one, zero = torch.ones(1, device=device), torch.zeros(1, device=device)

        def caller(plan, data):
            a, b, scales, scale_b = data
            cache = {}

            def fn(first, rows):
                if (first, rows) not in cache:
                    cache[first, rows] = _call_scales(f, scales, first, rows)
                out = torch.empty(rows, n, dtype=OUTPUTS[f["out"]], device=device)
                alpha = scales["alpha"] if scales["alpha"] is not None else one
                block = f["scale"] == "vec16"
                ops.run(plan, a[first:first + rows], b, out, cache[first, rows], scale_b, alpha if block else None,
                        zero if block else None, space)
                return out
            return fn

        a, b, scales = operands(fmt, n if weight is None else 1, k, total, device, seed)
        data = (a, weight[0], scales, weight[1]) if weight is not None else (a, b, scales, _weight_scale(f, scales))
        seen, chosen, found = set(), None, 0
        for index in range(listed):
            plan = int(ops.plan(native, n, k, rep_rows, workspace, index))
            algo = tuple(describe(plan))
            if algo in seen:
                continue
            seen.add(algo)
            fn = caller(plan, data)
            try:
                whole = fn(0, total)
                exact = all(torch.equal(fn(0, m), whole[:m]) for m in lengths if m <= total) and \
                    all(torch.equal(fn(s, m), whole[s:s + m]) for s, m in offsets if s + m <= total)
            except RuntimeError:
                continue
            if exact:
                chosen, found = plan, found + 1
                if found > choice:
                    break
        if chosen is None:
            raise RuntimeError(f"no row-exact split-K-free cuBLASLt algorithm for {fmt} ({n}x{k}) among {len(seen)}")
        torch.cuda.synchronize(device)
        found = Pinned(fmt, n, k, chosen, workspace, device)
        found._space = space
        return found
