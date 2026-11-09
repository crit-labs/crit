"""Pinned prompt GEMMs: one recorded cuBLASLt algorithm per GEMM shape and format, run for every row count.

cuBLAS's heuristics pick a GEMM's algorithm by its row count, and algorithms split a row's sum differently, so a
prompt row computes other bits when the prompt is longer. pinned_gemm.cpp runs one split-K-free algorithm of the
heuristic list for every row count instead. record() picks it once per shape, format, device and cuBLASLt version,
keeping only row-exact candidates (a row's bits are the same at every prompt length) and taking the fastest of those
by mean time per row, then short_buckets() records, per short row bucket, a faster algorithm that gives the same
bits where one exists; a GemmTable holds the choices for the frozen configuration and refuses to replay them on
another device or cuBLASLt version; pinned_mm() replays one.

a [M, K] row-major (the prompt's rows) times b [N, K] row-major (a linear layer's weight) gives out [M, N] = alpha
(scale_a a) @ (scale_b b).T with FP32 accumulation. A format is INPUTS[-SCALE]:OUTPUT (parse_format): INPUTS is
fp16, bf16, fp32 (no TF32), fp8e4m3, fp8e5m2, an FP8 pair such as fp8e4m3xfp8e5m2 (a's then b's), or a block-scaled
mxfp8 (e4m3, one UE8M0 scale per 32 elements along K), mxfp4 (e2m1, UE8M0 per 32) or nvfp4 (e2m1, UE4M3 per 16,
with an optional FP32 global scale alpha); FP4 is packed two per byte (uint8 or torch.float4_e2m1fn_x2, the first
element in the low nibble). SCALE, for FP8 only, is tensorscale (one FP32 element per operand, absent means 1) or
rowscale (FP32 vectors of a's M rows and b's N rows). OUTPUT is bf16, fp16 or fp32. Block scales go to pinned_mm
in cuBLASLt's blocked layout: blocked_scales() turns row-major [rows, K / block] scales into it, so a prefix a[:m]
takes blocked_scales(row_scales[:m]). The old names bf16 and fp8 still read as bf16:bf16 and
fp8e4m3-tensorscale:bf16 under their old keys. Importing needs no torch or CUDA; building (once per process,
torch.utils.cpp_extension), record() and pinned_mm() need both.
"""
import hashlib
import json
import os
import sys

try:
    from .selection import bucket, key, validate
except ImportError:
    from selection import bucket, key, validate

KIND = "pinned-gemm-table"
INPUTS = {"fp16": "fp16", "bf16": "bf16", "fp32": "fp32", "fp8e4m3": "e4m3", "fp8e5m2": "e5m2",
          "mxfp8": "e4m3", "mxfp4": "e2m1", "nvfp4": "e2m1"}     # input names and their element types
BLOCKS = {"mxfp8": ("vec32", 32), "mxfp4": ("vec32", 32), "nvfp4": ("vec16", 16)}   # scale mode, block size
SCALES = {"tensorscale": "tensor", "rowscale": "row"}            # an FP8 format's scale suffixes
OUTPUTS = ("bf16", "fp16", "fp32")
DTYPES = {"bf16": "bf16:bf16", "fp8": "fp8e4m3-tensorscale:bf16"}   # the old names, keyed as before
FORMATS = ("bf16:bf16", "bf16:fp32", "fp16:fp16", "fp16:fp32", "fp32:fp32", "fp8e4m3:bf16",
           "fp8e4m3-tensorscale:bf16", "fp8e4m3-tensorscale:fp16", "fp8e4m3-tensorscale:fp32",
           "fp8e4m3-rowscale:bf16", "fp8e5m2-tensorscale:bf16", "fp8e4m3xfp8e5m2-tensorscale:bf16",
           "fp8e5m2xfp8e4m3-tensorscale:bf16", "mxfp8:bf16", "mxfp4:bf16", "nvfp4:bf16", "nvfp4:fp32")
TILE_ROWS, TILE_COLS = 128, 4  # the blocked scale layout's tiles: rows padded to 128, scale columns to 4
EVERY_ROW_COUNT = 0            # the width bucket of a pinned GEMM's key: one choice for every row count
WORKSPACE = 32 << 20
ROWS = (128, 512, 2048)
LENGTHS = (1, 7, 17, 128, 513, 4096)
SHORT = 64                     # short prompts' row buckets: powers of two up to this many rows
SPEEDUP = 0.05                 # a short bucket's algorithm replaces the entry's when it is this much faster
HERE = os.path.dirname(os.path.abspath(__file__))
_ext, _plans, _workspaces, _constants = None, {}, {}, {}


class UnsupportedFormat(RuntimeError):
    """cuBLASLt's heuristic offers no algorithm for the format on this device and cuBLASLt version."""


def parse_format(fmt):
    """The format's parts: name (canonical), inputs (a's and b's names), a and b (element types), out, scale (none,
    tensor, row, vec32 or vec16) and block (elements per block scale, 0 for none). Old names read as before."""
    if not isinstance(fmt, str):
        raise ValueError("a pinned GEMM format is a string such as fp8e4m3-rowscale:bf16")
    name = DTYPES.get(fmt, fmt)
    inputs, _, out = name.partition(":")
    inputs, _, suffix = inputs.partition("-")
    pair = inputs.split("x") if inputs.count("x") == 1 and not inputs.startswith("mx") else [inputs, inputs]
    if out not in OUTPUTS or any(p not in INPUTS for p in pair):
        raise ValueError("pinned GEMM formats are INPUTS[-SCALE]:OUTPUT with inputs from %s and outputs from %s, "
                         "not %r" % (", ".join(sorted(INPUTS)), ", ".join(OUTPUTS), fmt))
    fp8 = all(p.startswith("fp8") for p in pair)
    if pair[0] != pair[1] and not fp8:
        raise ValueError("only FP8 inputs mix element types, not %r" % fmt)
    if suffix and (not fp8 or suffix not in SCALES):
        raise ValueError("scale suffixes (%s) are for FP8 inputs, not %r" % (", ".join(SCALES), fmt))
    scale, block = BLOCKS.get(pair[0], (SCALES.get(suffix, "none"), 0))
    return {"name": name, "inputs": tuple(pair), "a": INPUTS[pair[0]], "b": INPUTS[pair[1]], "out": out,
            "scale": scale, "block": block}


def gemm_key(n, k, fmt):
    """The table key of an [N, K] GEMM in `fmt`: key('pinned-gemm-INPUTS[-SCALE]-to-OUTPUT', (N, K), 0), or for the
    old names bf16 and fp8 their old key('pinned-gemm-DTYPE', (N, K), 0)."""
    parse_format(fmt)
    if not isinstance(n, int) or not isinstance(k, int) or n < 1 or k < 1:
        raise ValueError("N and K must be positive integers")
    return key("pinned-gemm-" + fmt.replace(":", "-to-"), (n, k), EVERY_ROW_COUNT)


def scale_layout(rows, k, block):
    """The (rows, columns) of `rows` rows of block scales over K in the blocked layout: K / block columns, rows
    padded to a multiple of 128 and columns to a multiple of 4."""
    if not all(isinstance(v, int) and v > 0 for v in (rows, k, block)) or k % block:
        raise ValueError("positive rows and a K that is a multiple of the block size required")
    return -(-rows // TILE_ROWS) * TILE_ROWS, -(-(k // block) // TILE_COLS) * TILE_COLS


def blocked_position(row, col, cols):
    """Where a row-major scale (row, col) goes in the blocked layout of `cols` (padded) columns: 128 x 4 tiles in
    row-major tile order, each storing row r, column c at (r % 32) * 16 + (r // 32) * 4 + c."""
    tile = (row // TILE_ROWS) * (cols // TILE_COLS) + col // TILE_COLS
    r, c = row % TILE_ROWS, col % TILE_COLS
    return tile * TILE_ROWS * TILE_COLS + (r % 32) * 16 + (r // 32) * 4 + c


def blocked_order(rows, cols):
    """For the blocked layout of a row-major [rows, cols] scale matrix, the row-major source index at each position,
    or -1 for padding."""
    padded_rows = -(-rows // TILE_ROWS) * TILE_ROWS
    padded_cols = -(-cols // TILE_COLS) * TILE_COLS
    order = [-1] * (padded_rows * padded_cols)
    for row in range(rows):
        for col in range(cols):
            order[blocked_position(row, col, padded_cols)] = row * cols + col
    return order


def blocked_scales(scales):
    """Row-major [rows, K / block] one-byte scales (any row count) in cuBLASLt's blocked layout, flat and padded
    with zeros; the dtype is kept."""
    import torch
    rows, cols = scales.shape
    pr, pc = -(-rows // TILE_ROWS) * TILE_ROWS, -(-cols // TILE_COLS) * TILE_COLS
    raw = scales.contiguous().view(torch.uint8)
    padded = torch.zeros(pr, pc, dtype=torch.uint8, device=scales.device)
    padded[:rows, :cols] = raw
    tiles = padded.view(pr // TILE_ROWS, TILE_ROWS, pc // TILE_COLS, TILE_COLS).permute(0, 2, 1, 3)
    out = tiles.reshape(-1, 4, 32, TILE_COLS).transpose(1, 2).reshape(-1)
    return out.view(scales.dtype)


class GemmTable:
    """Recorded entries by gemm_key, with the cuBLASLt version and device name they were recorded on."""

    def __init__(self, cublaslt, device, workspace_bytes=WORKSPACE, entries=None):
        if not isinstance(cublaslt, int) or isinstance(cublaslt, bool) or cublaslt < 1 or not device:
            raise ValueError("a positive cuBLASLt version and a device name required")
        if not isinstance(workspace_bytes, int) or workspace_bytes < 0:
            raise ValueError("workspace bytes must be a non-negative integer")
        self.cublaslt, self.device, self.workspace_bytes = cublaslt, str(device), workspace_bytes
        self.entries = {}
        for entry in (entries or {}).values():
            self.add(entry)

    def add(self, entry):
        """An entry runs its algorithm (rep_rows, index, algo) for every row count, except that `buckets`, when
        present, maps a short row bucket's width (a power of two, as a string) to the algorithm for its counts."""
        k = gemm_key(entry.get("n"), entry.get("k"), entry.get("dtype"))
        for choice in [entry] + list(entry.get("buckets", {}).values()):
            if not isinstance(choice, dict):
                raise ValueError("a bucket's choice must be an object")
            validate({k: choice.get("index")})
            rep, algo = choice.get("rep_rows"), choice.get("algo")
            if not isinstance(rep, int) or rep < 1 or not isinstance(algo, list) or not all(
                    isinstance(v, int) for v in algo):
                raise ValueError("an entry needs a positive rep_rows and its algorithm configuration")
        for width, choice in entry.get("buckets", {}).items():
            if not width.isdigit() or bucket(int(width)) != int(width) or choice["rep_rows"] != int(width):
                raise ValueError("a short bucket is a power-of-two width recorded at that width")
        self.entries[k] = dict(entry)
        return k

    def entry(self, n, k, dtype, cublaslt=None, device=None):
        """The recorded entry for the shape and format, refused when the current (or given) device or cuBLASLt
        version is not the table's; it carries both, so pinned_mm checks them again at its first call."""
        if cublaslt is None or device is None:
            current = environment()
            cublaslt = current[0] if cublaslt is None else cublaslt
            device = current[1] if device is None else device
        if cublaslt != self.cublaslt or device != self.device:
            raise ValueError("the pinned GEMM table was recorded with cuBLASLt %d on %s, not cuBLASLt %d on %s: "
                             "record it again" % (self.cublaslt, self.device, cublaslt, device))
        k_ = gemm_key(n, k, dtype)
        if k_ not in self.entries:
            raise KeyError("no recorded pinned GEMM for " + k_)
        return dict(self.entries[k_], cublaslt=self.cublaslt, device=self.device,
                    workspace_bytes=self.workspace_bytes)

    def to_json(self):
        return {"kind": KIND, "cublaslt": self.cublaslt, "device": self.device,
                "workspace_bytes": self.workspace_bytes,
                "entries": {k: self.entries[k] for k in sorted(self.entries)}}

    @classmethod
    def from_json(cls, data):
        if not isinstance(data, dict) or data.get("kind") != KIND or not isinstance(data.get("entries"), dict):
            raise ValueError("not a pinned GEMM table")
        table = cls(data.get("cublaslt"), data.get("device"), data.get("workspace_bytes"), data["entries"])
        if sorted(table.entries) != sorted(data["entries"]):
            raise ValueError("pinned GEMM table keys do not match their entries")
        return table

    def save(self, path):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_json(), f, indent=1)
            f.write("\n")

    @classmethod
    def load(cls, path):
        with open(path, encoding="utf-8") as f:
            return cls.from_json(json.load(f))


def _native(fmt):
    """The format as pinned_gemm.cpp takes it: [a type, b type, output type, scale mode]."""
    f = parse_format(fmt)
    return [f["a"], f["b"], f["out"], f["scale"]]


def _torch_dtype(element):
    import torch
    return {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32, "e4m3": torch.float8_e4m3fn,
            "e5m2": torch.float8_e5m2, "e2m1": torch.uint8}[element]


def extension():
    """pinned_gemm.cpp, built once per process (torch.utils.cpp_extension caches the build across processes)."""
    global _ext
    if _ext is None:
        from torch.utils.cpp_extension import load
        lib = "cublasLt.lib" if sys.platform == "win32" else "-lcublasLt"
        source = os.path.join(HERE, "pinned_gemm.cpp")
        # One build cache per copy of the source (each engine has its own copy of the plugin): under one name, every
        # switch between copies would build it again.
        with open(source, "rb") as f:
            copy = hashlib.sha256(source.encode() + f.read()).hexdigest()[:12]
        _ext = load(name=f"crit_pinned_gemm_{copy}", sources=[source], with_cuda=True, extra_cflags=["-O2"],
                    extra_ldflags=[lib], verbose=False)
    return _ext


def environment():
    """(cuBLASLt version, device name) of the current CUDA device."""
    import torch
    return int(extension().version()), torch.cuda.get_device_name(torch.cuda.current_device())


def _workspace(device, size):
    import torch
    buf = _workspaces.get(device)
    if buf is None or buf.numel() < size:
        buf = _workspaces[device] = torch.empty(max(size, 1), dtype=torch.uint8, device=device)
    return buf


def _constant(device, value):
    import torch
    if (device, value) not in _constants:
        _constants[device, value] = torch.full((1,), value, dtype=torch.float32, device=device)
    return _constants[device, value]


def _plan(entry, rows):
    """The plan for `rows` rows: the short bucket's algorithm when the entry has one for that bucket, else the
    entry's. Each is planned once and must still be the recorded algorithm."""
    choice = entry
    if rows <= SHORT and str(bucket(max(rows, 1))) in entry.get("buckets", {}):
        choice = entry["buckets"][str(bucket(max(rows, 1)))]
    ident = (entry["dtype"], entry["n"], entry["k"], choice["rep_rows"], choice["index"], entry["workspace_bytes"])
    if ident not in _plans:
        if (entry["cublaslt"], entry["device"]) != environment():
            raise ValueError("the pinned GEMM entry was recorded with cuBLASLt %d on %s: record it again here"
                             % (entry["cublaslt"], entry["device"]))
        pid = extension().plan(_native(entry["dtype"]), entry["n"], entry["k"], choice["rep_rows"],
                               entry["workspace_bytes"], choice["index"])
        if list(extension().describe(pid)) != list(choice["algo"]):
            raise ValueError("the heuristic's algorithm %d at %d rows for %s is %s, not the recorded %s: record it "
                             "again" % (choice["index"], choice["rep_rows"],
                                        gemm_key(entry["n"], entry["k"], entry["dtype"]),
                                        list(extension().describe(pid)), choice["algo"]))
        _plans[ident] = pid
    return _plans[ident]


def _run(pid, fmt, a, b, workspace_bytes, scale_a=None, scale_b=None, alpha=None):
    import torch
    f = parse_format(fmt)
    out = torch.empty(a.shape[0], b.shape[0], dtype=_torch_dtype(f["out"]), device=a.device)
    beta = None
    if f["scale"] == "vec16":
        alpha = _constant(a.device, 1.0) if alpha is None else alpha
        beta = _constant(a.device, 0.0)
    extension().run(pid, a, b, out, scale_a, scale_b, alpha, beta, _workspace(a.device, workspace_bytes))
    return out


def pinned_mm(a, b, entry, scale_a=None, scale_b=None, alpha=None):
    """a [M, K] times b [N, K]^T in the entry's format with its recorded algorithm, or its short bucket's
    bit-identical one, so a row's bits are the same for every M (see the module docstring for scales; block scales
    in the blocked layout of a's M rows and b's N rows, alpha an FP32 one-element global scale for nvfp4). An M the
    algorithm cannot run raises."""
    return _run(_plan(entry, a.shape[0]), entry["dtype"], a.contiguous(), b.contiguous(), entry["workspace_bytes"],
                scale_a, scale_b, alpha)


def _timed(fn, repeats=20):
    import torch
    for _ in range(3):
        fn()
    start, stop = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(repeats):
        fn()
    stop.record()
    stop.synchronize()
    return start.elapsed_time(stop) / repeats


FP4_VALUES = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0)


def _values(element, rows, k, g):
    """Random [rows, K] values of an element type: 16/32-bit normal, FP8 spread over its range, FP4 codes packed
    two per byte."""
    import torch
    if element == "e2m1":
        codes = torch.randint(0, 16, (rows, k), device="cuda", generator=g, dtype=torch.uint8)
        return codes[:, 0::2] | (codes[:, 1::2] << 4)
    x = torch.randn(rows, k, device="cuda", generator=g)
    if element in ("e4m3", "e5m2"):
        x = (x * 60).clamp(-448, 448)
    return x.to(_torch_dtype(element))


def operands(n, k, fmt, rows, seed=0):
    """Random a [rows, K] and b [N, K] of the format on the current device, and their scales: {"a", "b", "alpha"}
    with row-major block scales ([rows, K / block] and [N, K / block] bytes), per-row FP32 vectors, one-element
    FP32 tensors or None."""
    import torch
    f = parse_format(fmt)
    g = torch.Generator(device="cuda").manual_seed(seed)
    a, b = _values(f["a"], rows, k, g), _values(f["b"], n, k, g)
    scales = {"a": None, "b": None, "alpha": None}
    if f["scale"] == "tensor":
        scales.update(a=torch.tensor([0.02], device="cuda"), b=torch.tensor([0.005], device="cuda"))
    elif f["scale"] == "row":
        scales.update(a=torch.rand(rows, device="cuda", generator=g) * 0.02 + 0.01,
                      b=torch.rand(n, device="cuda", generator=g) * 0.005 + 0.0025)
    elif f["scale"] == "vec32":
        if f["a"] == "e4m3":
            a, b = (x.float().div(15).to(torch.float8_e4m3fn) for x in (a, b))
        scales.update({s: torch.randint(124, 129, (r, k // 32), device="cuda", generator=g, dtype=torch.uint8)
                       for s, r in (("a", rows), ("b", n))})
    elif f["scale"] == "vec16":
        scales.update({s: (torch.rand(r, k // 16, device="cuda", generator=g) * 0.75 + 0.25).to(
            torch.float8_e4m3fn).view(torch.uint8) for s, r in (("a", rows), ("b", n))})
        scales["alpha"] = torch.tensor([0.5], device="cuda")
    return a, b, scales


def prompt_scales(fmt, scales, m):
    """pinned_mm's scale_a, scale_b and alpha for the first m rows of operands()' a."""
    f = parse_format(fmt)
    sa, sb = scales["a"], scales["b"]
    if f["scale"] == "row":
        sa = sa[:m].contiguous()
    elif f["block"]:
        sa, sb = blocked_scales(sa[:m]), blocked_scales(sb)
    return {"scale_a": sa, "scale_b": sb, "alpha": scales["alpha"]}


def dequantized(fmt, x, scales, side):
    """Operand x (side "a" or "b") of operands() as FP32 values, scales applied (alpha excepted)."""
    import torch
    f = parse_format(fmt)
    if f[side] == "e2m1":
        table = torch.tensor(FP4_VALUES, device=x.device)
        v = torch.stack((table[(x & 15).long()], table[(x >> 4).long()]), dim=-1).reshape(x.shape[0], -1)
    else:
        v = x.float()
    s = scales[side]
    if f["scale"] == "tensor" and s is not None:
        v = v * s
    elif f["scale"] == "row":
        v = v * s[:, None]
    elif f["scale"] == "vec32":
        v = v * torch.pow(2.0, s.float() - 127).repeat_interleave(32, dim=1)
    elif f["scale"] == "vec16":
        v = v * s.view(torch.float8_e4m3fn).float().repeat_interleave(16, dim=1)
    return v


def reference(fmt, a, b, scales):
    """The FP32 dequantized product of operands() (no TF32), in FP32."""
    import torch
    allowed = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        out = dequantized(fmt, a, scales, "a") @ dequantized(fmt, b, scales, "b").t()
    finally:
        torch.backends.cuda.matmul.allow_tf32 = allowed
    return out * scales["alpha"] if scales["alpha"] is not None else out


def torch_mm(fmt, a, b, scales):
    """torch's own kernel for the format (torch.matmul, or torch._scaled_mm with its scales), or None when torch
    has none with the format's output. The nvfp4 global scale is applied after it."""
    import torch
    f = parse_format(fmt)
    out = _torch_dtype(f["out"])
    try:
        if f["scale"] == "none" and f["a"] in ("fp16", "bf16", "fp32"):
            return a @ b.t() if f["out"] == f["a"] else None
        one = torch.ones(1, device=a.device)
        if f["scale"] in ("none", "tensor"):
            sa, sb = (one, one) if f["scale"] == "none" else (scales["a"], scales["b"])
        elif f["scale"] == "row":
            sa, sb = scales["a"][:, None].contiguous(), scales["b"][None, :].contiguous()
        elif f["scale"] == "vec32":
            sa, sb = (blocked_scales(scales[s]).view(torch.float8_e8m0fnu) for s in ("a", "b"))
        else:
            sa, sb = (blocked_scales(scales[s]).view(torch.float8_e4m3fn) for s in ("a", "b"))
        if f["a"] == "e2m1":
            a, b = a.view(torch.float4_e2m1fn_x2), b.view(torch.float4_e2m1fn_x2)
        result = torch._scaled_mm(a, b.t(), sa, sb, out_dtype=out)
        return result * scales["alpha"].to(out) if scales["alpha"] is not None else result
    except (RuntimeError, TypeError, ValueError, AttributeError, NotImplementedError):
        return None


def row_exact(fn, rows, lengths):
    """Whether fn(m), the product of the first m rows, gives the first m rows of fn(rows) bit for bit at every
    length m."""
    import torch
    whole = fn(rows)
    return all(torch.equal(fn(m), whole[:m]) for m in lengths if m <= rows)


def _unsupported(n, k, fmt, rows):
    cublaslt, device = environment()
    return UnsupportedFormat("cuBLASLt %d offers no split-K-free algorithm for %s (%dx%d at %s rows) on %s"
                             % (cublaslt, parse_format(fmt)["name"], n, k, "/".join(map(str, rows)), device))


def _caller(pid, fmt, a, b, scales, workspace_bytes):
    """fn(m): the product of a's first m rows with plan pid, block scales laid out once per m."""
    cache = {}

    def fn(m):
        if m not in cache:
            cache[m] = prompt_scales(fmt, scales, m)
        return _run(pid, fmt, a[:m], b, workspace_bytes, **cache[m])
    return fn


def record(n, k, dtype, rows=ROWS, lengths=LENGTHS, workspace_bytes=WORKSPACE, seed=0, short=SHORT, log=None):
    """The entry for an [N, K] GEMM in format `dtype`: of the split-K-free algorithms the heuristic lists at each
    count of `rows` (duplicates once), the row-exact ones over `lengths` (and the longest product), the fastest by
    mean time per row over `rows`; ties keep the earlier candidate. The entry carries the cuBLASLt version, the device
    and the workspace it was recorded with, so pinned_mm replays it as it is, as GemmTable.entry gives it. Candidates that cannot run a length are skipped;
    raises UnsupportedFormat, naming the format, device and cuBLASLt version, when the heuristic lists none, and
    RuntimeError when none is row-exact. Then short_buckets() up to `short` rows (0: none). Never on the serving
    path."""
    import torch
    ext = extension()
    native = _native(dtype)
    total = max(max(rows), max(lengths))
    a, b, scales = operands(n, k, dtype, total, seed)
    seen, best, tried, listed = set(), None, 0, 0
    for rep in sorted(rows, reverse=True):
        count = ext.candidates(native, n, k, rep, workspace_bytes)
        listed += count
        for index in range(count):
            pid = ext.plan(native, n, k, rep, workspace_bytes, index)
            algo = list(ext.describe(pid))
            if tuple(algo) in seen:
                continue
            seen.add(tuple(algo))
            tried += 1
            fn = _caller(pid, dtype, a, b, scales, workspace_bytes)
            try:
                exact = row_exact(fn, total, lengths)
            except RuntimeError as exc:
                if log:
                    log("%s rep %d #%d %s: cannot run (%s)" % (gemm_key(n, k, dtype), rep, index, algo,
                                                              str(exc).splitlines()[0]))
                continue
            ms = {str(m): _timed(lambda m=m: fn(m)) for m in rows}
            cost = sum(ms[str(m)] / m for m in rows) / len(rows)
            if log:
                log("%s rep %d #%d %s exact %s ms %s" % (gemm_key(n, k, dtype), rep, index, algo, exact, ms))
            if exact and (best is None or cost < best[0]):
                best = (cost, {"n": n, "k": k, "dtype": dtype, "rep_rows": rep, "index": index, "algo": algo,
                               "ms": ms})
    torch.cuda.synchronize()
    if not listed:
        raise _unsupported(n, k, dtype, rows)
    if best is None:
        raise RuntimeError("no row-exact split-K-free cuBLASLt algorithm for " + gemm_key(n, k, dtype))
    entry = dict(best[1], candidates=tried)
    buckets = short_buckets(entry, short, (seed, seed + 1), workspace_bytes, log) if short else {}
    if buckets:
        entry["buckets"] = buckets
    # What pinned_mm checks at its first call, so the recorded entry replays as it is (GemmTable.entry adds the same).
    cublaslt, device = environment()
    entry.update(cublaslt=cublaslt, device=device, workspace_bytes=workspace_bytes)
    return entry


def short_buckets(entry, short=SHORT, seeds=(0, 1), workspace_bytes=WORKSPACE, log=None):
    """Faster algorithms for short prompts that give the entry's bits. A row's bits only have to be the same at
    every length, not to come from one algorithm: split-K-free algorithms that accumulate K alike and differ in
    their M/N tiles, stages or swizzle often agree bit for bit. For each power-of-two row bucket up to `short`
    (bucket() of a count: 1, 2, 3-4, 5-8, ...), of the split-K-free algorithms the heuristic lists at the bucket's
    width, keep those whose output equals the entry algorithm's bit for bit at both edges of the bucket and counts
    inside it, on operands of every seed; take the fastest at the bucket's width when it beats the entry's
    algorithm there by more than SPEEDUP. Returns {width: choice} for the buckets that have one."""
    import torch
    ext = extension()
    n, k, dtype = entry["n"], entry["k"], entry["dtype"]
    native = _native(dtype)
    base = ext.plan(native, n, k, entry["rep_rows"], workspace_bytes, entry["index"])
    data = [operands(n, k, dtype, short, s) for s in seeds]
    bases = [_caller(base, dtype, a, b, s, workspace_bytes) for a, b, s in data]
    found, width = {}, 1
    while width <= short:
        low = width // 2 + 1
        counts = sorted({low, min(low + 1, width), (low + width) // 2, width})
        base_ms = _timed(lambda: bases[0](width), 50)
        best = None
        for index in range(ext.candidates(native, n, k, width, workspace_bytes)):
            pid = ext.plan(native, n, k, width, workspace_bytes, index)
            algo = list(ext.describe(pid))
            if algo == list(entry["algo"]):
                continue
            fns = [_caller(pid, dtype, a, b, s, workspace_bytes) for a, b, s in data]
            try:
                same = all(torch.equal(fn(m), base_fn(m)) for fn, base_fn in zip(fns, bases) for m in counts)
            except RuntimeError:
                continue
            ms = _timed(lambda: fns[0](width), 50) if same else None
            if log:
                log("%s rows %d #%d %s same %s ms %s (entry %.4f)" % (gemm_key(n, k, dtype), width, index, algo, same,
                                                                     ms, base_ms))
            if same and ms < base_ms * (1 - SPEEDUP) and (best is None or ms < best["ms"]):
                best = {"rep_rows": width, "index": index, "algo": algo, "ms": ms, "entry_ms": base_ms}
        if best:
            found[str(width)] = best
        width *= 2
    torch.cuda.synchronize()
    return found
