"""Gates a kernel passes beyond its reference check, each proven by a planted control: a kernel built to fail it.

crit kernels gates ID runs them over a kernel's structures on the model's shapes (or given shapes), with the sweep's
benches (crit_kernels.sweep: inputs, run, reference, check, first_row), on the kernel's default build or a tuning:

    nonfinite  A NaN planted in row 0 of every row input reaches row 0's output (a kernel that drops a NaN, as a max
               by fmax does, fails) and leaves every other row's bytes as they were (a kernel whose shared reduction or
               tile lets it leak fails). A bench whose outputs keep a stated rule (routing that ranks a NaN below every
               number) says nan_rule = "contained": its row 0 is then not required to carry the NaN. A causal
               structure, whose rows are a sequence's tokens and each sees the ones before it (a recurrent scan, a
               prompt chunk), says nan_rule = "causal": the NaN goes into its last row instead, which must carry it,
               and every earlier row must keep its bytes (a scan that leaks backwards fails). A chunked causal
               form whose chunks multiply their tokens' triangular matrices densely (0 x NaN is NaN) also says
               nan_block = B: the gate then runs B + 3 rows, and the rows of every chunk before the NaN's must keep
               their bytes (its own chunk may carry it).
    repeat     Five calls on the same inputs give the same bytes, and the inputs are as they were after them: a sum
               by atomic adds or a write into an input fails. A structure declared "repeatable": false in kernel.json
               keeps only the second check. A bench's scratch buffers (bench.scratch, names in the state) are exempt.
    edges      The rows a structure serves are run at 1, 2, 3 and at every power of two from 4 on, less one, itself
               and plus one, up to the most rows it serves (capped by its tune_rows): each against the reference, so a
               tail a tile drops fails where whole tiles pass.
    sums       For a bench that offers integer_inputs(shape, rows, device) (None where a shape has none) and
               integer_reference(state): integer operands whose partial sums pass
               FP16's range and BF16's precision while every FP32 sum is exact in any order, and a result that is
               small. The output must equal the exact sum rounded to its type, so a sum kept in 16 bits fails.
    limits     The kernel built for every architecture its kernel.json claims, every function's registers and static
               shared memory read from the binaries (cuobjdump --dump-resource-usage) against each architecture's
               limits, with no GPU. Register spills (local memory) are reported, not failed. Triton kernels compile
               on first use and are not covered.

A gate reports pass, fail (with the reason) or na (the structure gives the gate nothing to check). controls() runs
the planted controls; a run where a control passes its gate is a failed run, because then the gate proves nothing.
"""
import json
import re
import shutil
import subprocess
from pathlib import Path

from . import build as kbuild

GATES = ("nonfinite", "repeat", "edges", "sums", "limits")

# Per architecture (nvcc's number): the most registers a thread, the most static shared bytes a block and the most
# shared bytes a block may opt into (CUDA C Programming Guide, compute capabilities table).
ARCH_LIMITS = {
    "80": (255, 48 * 1024, 163 * 1024), "86": (255, 48 * 1024, 99 * 1024), "87": (255, 48 * 1024, 163 * 1024),
    "89": (255, 48 * 1024, 99 * 1024), "90": (255, 48 * 1024, 227 * 1024), "100": (255, 48 * 1024, 227 * 1024),
    "101": (255, 48 * 1024, 227 * 1024), "103": (255, 48 * 1024, 227 * 1024), "110": (255, 48 * 1024, 227 * 1024),
    "120": (255, 48 * 1024, 99 * 1024), "121": (255, 48 * 1024, 99 * 1024),
}

_SMEM = {}   # the answer of shared_memory_optin(), once a process


def shared_memory_optin():
    """The most shared memory a block may opt into on this machine's GPU, in bytes, for a bench's valid(): the GPU's
    own figure, never one GPU's constant, so a sweep on a GPU with more (an A100's 163 KiB, an H100's or B200's 227
    KiB) tries the deeper rings that fit there. Read from nvidia-smi's compute capability and ARCH_LIMITS, so the
    sweep's process does not initialize CUDA before it forks its build workers (torch's device properties where CUDA
    is already initialized); 99 KiB, the least of crit's architectures, when neither answers."""
    if "bytes" not in _SMEM:
        found = None
        try:
            import torch
            if torch.cuda.is_initialized():
                found = int(getattr(torch.cuda.get_device_properties(torch.cuda.current_device()),
                                    "shared_memory_per_block_optin", 0)) or None
        except Exception:  # noqa: BLE001 - no torch or no device: ask nvidia-smi
            found = None
        if found is None:
            try:
                out = subprocess.run(["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"],
                                     capture_output=True, text=True, timeout=30).stdout.split()
                major, minor = out[0].strip().split(".")
                limits = ARCH_LIMITS.get(f"{int(major)}{int(minor)}")
                found = limits[2] if limits else None
            except (OSError, ValueError, IndexError, subprocess.SubprocessError):
                found = None
        _SMEM["bytes"] = found or 99 * 1024
    return _SMEM["bytes"]


def _tensors(obj, name=""):
    """(name, tensor) for every tensor a state holds, nested in dicts, lists and tuples."""
    import torch
    if isinstance(obj, torch.Tensor):
        yield name, obj
    elif isinstance(obj, dict):
        for k, v in obj.items():
            yield from _tensors(v, f"{name}.{k}" if name else str(k))
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            yield from _tensors(v, f"{name}[{i}]")


def _span(t):
    start = t.data_ptr()
    return start, start + t.untyped_storage().nbytes() - t.storage_offset() * t.element_size()


def _overlaps(a, b):
    (a0, a1), (b0, b1) = _span(a), _span(b)
    return a0 < b1 and b0 < a1


def _snapshot(out):
    return [(n, t.detach().clone()) for n, t in _tensors(out)]


def _equal_bits(a, b):
    import torch
    if a.shape != b.shape or a.dtype != b.dtype:
        return False
    if a.dtype.is_floating_point:
        width = {1: torch.uint8, 2: torch.int16, 4: torch.int32, 8: torch.int64}[a.element_size()]
        return torch.equal(a.contiguous().view(width), b.contiguous().view(width))
    return torch.equal(a, b)


def _scratch(bench):
    return set(getattr(bench, "scratch", ())) | set(getattr(bench, "restore", ()))


_RESTORE = {}


def _run(bench, module, state):
    """bench.run, with the state's in-place buffers (bench.restore: names in the state a call updates in place, as a
    residual stream it adds into) put back to their values before the state's first call, so every call sees the same
    inputs. Those buffers are neither inputs a call must keep nor rows a NaN goes into."""
    names = getattr(bench, "restore", ())
    if names:
        kept = _RESTORE.get(id(state))
        if kept is None or kept[0] is not state:
            kept = _RESTORE[id(state)] = (state, {n: state[n].clone() for n in names if state.get(n) is not None})
        for n, t in kept[1].items():
            state[n].copy_(t)
    return bench.run(module, state)


def _inputs(bench, state, outputs):
    """The state's tensors that are not scratch and share no memory with an output: what a call must leave as it was."""
    scratch = _scratch(bench)
    return [(n, t) for n, t in _tensors(state)
            if n.split(".")[0].split("[")[0] not in scratch and not any(_overlaps(t, o) for _, o in outputs)]


def _verdict(gate, ok, reason=None):
    return {"gate": gate, "verdict": "pass" if ok else "fail", **({} if ok else {"reason": reason})}


def _na(gate, reason):
    return {"gate": gate, "verdict": "na", "reason": reason}


def _rows_range(spec):
    lo, hi = (spec.get("rows") or [1, max(spec.get("tune_rows", [1]))])[:2]
    cap = max(spec.get("tune_rows", [hi]))
    return int(lo), int(min(hi, max(cap, lo)))


def edge_rows(lo, hi):
    """Row counts the edges gate runs: 1, 2, 3, every power of two t from 4 on as t - 1, t, t + 1, and the most."""
    found = {1, 2, 3, hi}
    t = 4
    while t <= hi + 1:
        found.update((t - 1, t, t + 1))
        t *= 2
    return sorted(r for r in found if lo <= r <= hi)


def gate_repeat(bench, module, shape, rows, repeatable=True):
    import torch
    state = bench.inputs(shape, rows, "cuda")
    raw = _run(bench, module, state)
    first = _snapshot(raw)
    torch.cuda.synchronize()
    inputs = [(n, t.detach().clone(), t) for n, t in _inputs(bench, state, list(_tensors(raw)))]
    for attempt in range(4):
        again = _snapshot(_run(bench, module, state))
        torch.cuda.synchronize()
        if repeatable and not all(_equal_bits(a, b) for (_, a), (_, b) in zip(first, again)):
            return _verdict("repeat", False, f"rows {rows}: call {attempt + 2} gave other bytes than call 1")
    changed = [n for n, before, now in inputs if not _equal_bits(before, now)]
    if changed:
        return _verdict("repeat", False, f"rows {rows}: the calls changed input {', '.join(changed)}")
    return _verdict("repeat", True)


def gate_edges(bench, module, shape, lo, hi):
    import torch
    for rows in edge_rows(lo, hi):
        state = bench.inputs(shape, rows, "cuda")
        got = _run(bench, module, state)
        torch.cuda.synchronize()
        reason = bench.check(got, bench.reference(state), state)
        if reason:
            return _verdict("edges", False, f"rows {rows}: {reason}")
    return _verdict("edges", True)


def gate_nonfinite(bench, module, shape, rows, repeatable=True, most=None):
    import torch
    rule, block = getattr(bench, "nan_rule", "propagate"), int(getattr(bench, "nan_block", 0) or 0)
    if rule == "causal" and block:
        rows = min(block + 3, most or block + 3)
    if rows < 2:
        return _na("nonfinite", "a structure of one row has no other row to keep")
    state = bench.inputs(shape, rows, "cuda")
    raw = _run(bench, module, state)
    clean = _snapshot(raw)
    twice = _snapshot(_run(bench, module, state))
    torch.cuda.synchronize()
    # A kernel whose clean calls differ (atomics) is judged by where its outputs are finite, not by their bytes.
    repeatable = repeatable and all(_equal_bits(a, b) for (_, a), (_, b) in zip(clean, twice))
    targets = [(n, t) for n, t in _inputs(bench, state, list(_tensors(raw)))   # the output buffers themselves, not copies
               if t.dtype.is_floating_point and t.dim() >= 1 and t.shape[0] == rows and t.numel()]
    if not targets:
        return _na("nonfinite", "no floating input with one row a call row")
    at = rows - 1 if rule == "causal" else 0           # the row the NaN goes into
    others = slice(0, rows - 1) if rule == "causal" else slice(1, rows)
    if rule == "causal" and block:
        others = slice(0, at // block * block)          # the chunks before the NaN's
    kept = [(t, t[at].clone()) for _, t in targets]
    try:
        for t, _ in kept:
            t[at].view(-1)[0] = float("nan")
        poisoned = _snapshot(_run(bench, module, state))
        torch.cuda.synchronize()
    finally:
        for t, row in kept:   # benches cache their inputs: put the row back
            t[at].copy_(row)
    per_row = [(n, a, b) for (n, a), (_, b) in zip(clean, poisoned) if a.dim() >= 1 and a.shape[0] == rows]
    if not per_row:
        return _na("nonfinite", "no output with one row a call row")
    for n, a, b in per_row:
        if repeatable and not _equal_bits(a[others], b[others]):
            return _verdict("nonfinite", False, f"rows {rows}: a NaN in row {at} changed other rows of output {n}")
        if not repeatable and b[others].is_floating_point() and not torch.equal(torch.isfinite(a[others]),
                                                                                torch.isfinite(b[others])):
            return _verdict("nonfinite", False, f"rows {rows}: a NaN in row {at} reached other rows of output {n}")
    if rule in ("propagate", "causal"):
        floating = [(n, b) for n, _, b in per_row if b.is_floating_point()]
        if floating and not any(bool((~torch.isfinite(b[at])).any()) for _, b in floating):
            return _verdict("nonfinite", False, f"rows {rows}: a NaN in row {at}'s inputs left its outputs finite")
    return _verdict("nonfinite", True)


def gate_sums(bench, module, shape, rows):
    import torch
    if not hasattr(bench, "integer_inputs"):
        return _na("sums", "the bench offers no integer inputs")
    state = bench.integer_inputs(shape, rows, "cuda")
    if state is None:
        return _na("sums", "the bench offers no integer inputs for this shape")
    got = _run(bench, module, state)
    torch.cuda.synchronize()
    exact = bench.integer_reference(state)
    got_t = got if isinstance(got, torch.Tensor) else next(t for _, t in _tensors(got))
    want = exact.to(got_t.dtype)
    if not _equal_bits(got_t.contiguous(), want.contiguous()):
        bad = int((got_t.double() != want.double()).sum())
        return _verdict("sums", False, f"rows {rows}: {bad} outputs differ from the exact sum: a sum not kept in FP32")
    return _verdict("sums", True)


_FUNCTION = re.compile(r"Function\s+(\S+):\s*\n\s*REG:(\d+)\s+STACK:(\d+)\s+SHARED:(\d+)\s+LOCAL:(\d+)")
_ARCH = re.compile(r"arch\s*=\s*sm_(\d+)")


def resource_usage(text):
    """[(arch, function, registers, static shared bytes, local bytes)] from cuobjdump --dump-resource-usage."""
    found, arch = [], None
    for part in re.split(r"(?=arch\s*=\s*sm_\d+)", text):
        m = _ARCH.search(part)
        if m:
            arch = m.group(1)
        for f in _FUNCTION.finditer(part):
            found.append((arch, f.group(1), int(f.group(2)), int(f.group(4)), int(f.group(5))))
    return found


def check_limits(records):
    """(failures, spills) of resource_usage() records against ARCH_LIMITS."""
    failures, spills = [], []
    for arch, function, registers, shared, local in records:
        regs, static, _ = ARCH_LIMITS.get(str(arch).rstrip("af"), ARCH_LIMITS["80"])
        if registers > regs:
            failures.append(f"sm_{arch} {function}: {registers} registers a thread, the most is {regs}")
        if shared > static:
            failures.append(f"sm_{arch} {function}: {shared} static shared bytes, the most is {static}")
        if local:
            spills.append(f"sm_{arch} {function}: {local} bytes of local memory (spills)")
    return failures, spills


def _cuobjdump():
    """cuobjdump from CUDA_HOME, CUDA_PATH, PATH or torch's CUDA home."""
    import os
    exe = "cuobjdump.exe" if os.name == "nt" else "cuobjdump"
    homes = [os.environ.get("CUDA_HOME"), os.environ.get("CUDA_PATH")]
    try:
        from torch.utils.cpp_extension import CUDA_HOME
        homes.append(CUDA_HOME)
    except ImportError:
        pass
    for home in homes:
        if home and (Path(home) / "bin" / exe).exists():
            return str(Path(home) / "bin" / exe)
    found = shutil.which(exe)
    if not found:
        raise RuntimeError("the limits gate reads the binaries with cuobjdump: none in CUDA_HOME or on PATH")
    return found


def gate_limits(kernel_id, archs=None, root=None):
    folder, manifest = kbuild.find(kernel_id, root)
    if manifest["language"] != "cuda" and not manifest.get("cuda"):
        return _na("limits", "a Triton kernel compiles on first use")
    tool = _cuobjdump()
    archs = archs or [c.replace(".", "") for c in manifest.get("capabilities", [])]   # "10.0a" -> nvcc's "100a"
    out, _, _ = kbuild.build(kernel_id, archs=archs, root=root)
    binaries = [p for p in out.rglob("*") if p.suffix in (".so", ".pyd", ".cubin", ".fatbin", ".o")]
    records = []
    for b in binaries:
        run = subprocess.run([tool, "--dump-resource-usage", str(b)], capture_output=True, text=True)
        if run.returncode == 0:
            records.extend(resource_usage(run.stdout))
    if not records:
        return _na("limits", "no device functions found in the build")
    failures, spills = check_limits(records)
    result = _verdict("limits", not failures, "; ".join(failures[:8]))
    result["functions"] = len(records)
    if spills:
        result["spills"] = spills[:16]
    return result


def run_structure(bench, module, spec, shape):
    """Gates nonfinite, repeat, edges and sums for one structure on one shape."""
    lo, hi = _rows_range(spec)
    repeatable = spec.get("repeatable", True)
    rows = min(max(3, lo), hi)
    results = []
    for gate, call in (("nonfinite", lambda: gate_nonfinite(bench, module, shape, rows, repeatable, hi)),
                       ("repeat", lambda: gate_repeat(bench, module, shape, rows, repeatable)),
                       ("edges", lambda: gate_edges(bench, module, shape, lo, hi)),
                       ("sums", lambda: gate_sums(bench, module, shape, rows))):
        try:
            results.append(call())
        except Exception as error:  # noqa: BLE001 - a gate's failure on one shape, reported with its reason
            results.append(_verdict(gate, False, f"{type(error).__name__}: {str(error).splitlines()[0][:300] if str(error) else ''}"))
    return results


def gates(kernel_id, *, model=None, shapes=None, structures=None, tuning=None, archs=None, root=None, limits=True,
          log=print):
    """Run the gates over kernel_id's structures. Returns {"kernel", "structures": {structure: [{"shape", "gates"}]},
    "limits", "passed"}."""
    from .model import Model
    from .sweep import _bench_class
    folder, manifest = kbuild.find(kernel_id, root)
    chosen = structures or [s for s, spec in manifest.get("structures", {}).items() if spec.get("bench")]
    loaded = Model(model) if model else None
    module = kbuild.load(kernel_id, tuning=tuning, root=root)
    # The untuned build, as the sweep hands it to its benches (bench.default): some compare bits with it.
    default = module if tuning is None else kbuild.load(kernel_id, root=root)
    report = {"kernel": kernel_id, "structures": {}}
    for s in chosen:
        spec = manifest["structures"][s]
        if not spec.get("bench"):
            continue
        bench = _bench_class(folder, spec["bench"])
        bench.default = default
        found = (shapes or {}).get(s) if shapes else (bench.shapes(loaded) if loaded else [])
        seen, rows = set(), []
        for shape in found or []:
            key = json.dumps({f: shape.get(f) for f in spec.get("key", [])}, sort_keys=True, default=str)
            if key in seen:
                continue
            seen.add(key)
            results = run_structure(bench, module, spec, shape)
            rows.append({"shape": {f: shape.get(f) for f in spec.get("key", [])}, "gates": results})
            log(f"{kernel_id}/{s} {key}: " + ", ".join(f"{r['gate']} {r['verdict']}" for r in results))
        report["structures"][s] = rows
        _RESTORE.clear()
    if limits:
        try:
            report["limits"] = gate_limits(kernel_id, archs, root)
        except Exception as error:  # noqa: BLE001
            report["limits"] = _verdict("limits", False, f"{type(error).__name__}: {str(error).splitlines()[0][:300]}")
        log(f"{kernel_id} limits: {report['limits']['verdict']}")
    verdicts = [g["verdict"] for rows in report["structures"].values() for r in rows for g in r["gates"]]
    if limits:
        verdicts.append(report["limits"]["verdict"])
    report["passed"] = "fail" not in verdicts
    return report


# Planted controls: each a bench whose kernel breaks one gate (torch ops stand in for the kernel). controls() runs every
# gate on each and reports which gates it failed; its control's gate must be among them, and only the control's gate
# for the clean control.

class _RowSum:
    """Row sums of a [rows, K] FP32 matrix: the clean control, which passes every gate."""
    structure = "control-clean"
    K = 4099   # not a multiple of the tile below

    def inputs(self, shape, rows, device):
        import torch
        g = torch.Generator(device="cpu").manual_seed(7)
        x = torch.randn(rows, self.K, generator=g).to(device)
        return {"x": x, "out": torch.empty(rows, device=device)}

    def run(self, module, s):
        import torch
        torch.sum(s["x"], dim=1, out=s["out"])
        return s["out"]

    def reference(self, s):
        return s["x"].double().sum(1)

    def check(self, got, want, s):
        bad = int(((got.double() - want).abs() > 1e-3 * (1 + want.abs())).sum())
        return None if bad == 0 else f"{bad} sums off"

    def first_row(self, out):
        return out[0]


class _DropsNan(_RowSum):
    """A row maximum that skips NaN, as fmax does: fails nonfinite."""
    structure = "control-drops-nan"

    def run(self, module, s):
        import torch
        s["out"].copy_(torch.nan_to_num(s["x"], nan=float("-inf")).amax(1))
        return s["out"]

    def reference(self, s):
        return s["x"].double().amax(1)


class _RunningSum(_RowSum):
    """A running sum over the rows (row t the sum of rows 0 .. t), a causal structure: clean under nan_rule causal."""
    structure = "control-causal"
    nan_rule = "causal"
    K = 64

    def run(self, module, s):
        import torch
        torch.cumsum(s["x"].sum(1), dim=0, out=s["out"])
        return s["out"]

    def reference(self, s):
        return s["x"].double().sum(1).cumsum(0)


class _LeaksBackwards(_RunningSum):
    """A running sum taken from the last row back, a causal structure that leaks: fails nonfinite."""
    structure = "control-leaks-backwards"

    def run(self, module, s):
        import torch
        s["out"].copy_(torch.flip(torch.cumsum(torch.flip(s["x"].sum(1), [0]), 0), [0]))
        return s["out"]

    def reference(self, s):
        return s["x"].double().sum(1).flip(0).cumsum(0).flip(0)


class _ChunkMixes(_RunningSum):
    """A running sum whose chunks of 4 rows mix each row with its whole chunk (a NaN reaches its chunk's earlier
    rows, never an earlier chunk's): clean under nan_block 4."""
    structure = "control-chunk-mixes"
    nan_block = 4

    def run(self, module, s):
        import torch
        rows = s["x"].shape[0]
        sums = s["x"].sum(1)
        chunk = torch.arange(rows, device=sums.device) // 4
        own = torch.zeros(int(chunk.max()) + 1, device=sums.device).index_add_(0, chunk, sums * 0.0)  # 0 x NaN
        torch.cumsum(sums, dim=0, out=s["out"])
        s["out"].add_(own[chunk])
        return s["out"]


class _Atomic(_RowSum):
    """Row sums by atomic adds in an order that changes from call to call: fails repeat."""
    structure = "control-atomic"
    K = 1 << 16

    def run(self, module, s):
        import torch
        rows = s["x"].shape[0]
        index = torch.arange(rows, device=s["x"].device).repeat_interleave(self.K)
        s["out"].zero_().index_add_(0, index, s["x"].reshape(-1))
        return s["out"]


class _WritesInput(_RowSum):
    """Row sums that write into their input: fails repeat."""
    structure = "control-writes-input"

    def run(self, module, s):
        s["x"][0, 1] += 1.0
        return _RowSum.run(self, module, s)


class _DropsTail(_RowSum):
    """Row sums that run rows in tiles of 16 and skip a partial last tile: fails edges at 17 rows (passes at 16)."""
    structure = "control-drops-tail"

    def run(self, module, s):
        import torch
        rows = s["x"].shape[0]
        whole = rows // 16 * 16 if rows >= 16 else rows
        s["out"].zero_()
        torch.sum(s["x"][:whole], dim=1, out=s["out"][:whole])
        return s["out"]


class _HalfDot:
    """A dot product of integer operands summed in FP16: fails sums (the clean twin sums in FP32)."""
    structure = "control-half-sum"
    K, N = 8192, 64
    half = True

    def _make(self, rows, device):
        import torch
        g = torch.Generator(device="cpu").manual_seed(3)
        x = torch.randint(0, 16, (rows, self.K), generator=g).float()
        w = torch.randint(0, 16, (self.N, self.K), generator=g).float()
        w[:, self.K // 2:] = -w[:, :self.K // 2]   # the second half mirrors the first: partial sums large, result 0
        w[:, -1] += torch.arange(self.N).float() % 7
        return x.to(device), w.to(device)

    def inputs(self, shape, rows, device):
        import torch
        x, w = self._make(rows, device)
        return {"x": x, "w": w, "out": torch.empty(rows, self.N, device=device)}

    integer_inputs = inputs

    def integer_reference(self, s):
        return s["x"].double() @ s["w"].double().t()

    def reference(self, s):
        return self.integer_reference(s)

    def run(self, module, s):
        import torch
        if not self.half:
            s["out"].copy_(s["x"] @ s["w"].t())
            return s["out"]
        acc = torch.zeros_like(s["out"], dtype=torch.float16)
        for lo in range(0, self.K, 128):   # each chunk's 128 terms in FP32, the running sum in FP16
            acc = acc + (s["x"][:, lo:lo + 128] @ s["w"][:, lo:lo + 128].t()).half()
        s["out"].copy_(acc.float())
        return s["out"]

    def check(self, got, want, s):
        return None

    def first_row(self, out):
        return out[0]


class _FloatDot(_HalfDot):
    structure = "control-fp32-sum"
    half = False


CONTROLS = {"control-clean": (_RowSum, None), "control-drops-nan": (_DropsNan, "nonfinite"),
            "control-atomic": (_Atomic, "repeat"), "control-writes-input": (_WritesInput, "repeat"),
            "control-drops-tail": (_DropsTail, "edges"), "control-half-sum": (_HalfDot, "sums"),
            "control-fp32-sum": (_FloatDot, None), "control-causal": (_RunningSum, None),
            "control-leaks-backwards": (_LeaksBackwards, "nonfinite"), "control-chunk-mixes": (_ChunkMixes, None)}

LIMITS_CONTROL = """
Fatbin elf code:
================
arch = sm_120
code version = [1,8]

Resource usage:
 Function control_too_many_registers:
  REG:300 STACK:0 SHARED:0 LOCAL:0 CONSTANT[0]:400 TEXTURE:0 SURFACE:0 SAMPLER:0
 Function control_too_much_shared:
  REG:32 STACK:0 SHARED:65536 LOCAL:0 CONSTANT[0]:400 TEXTURE:0 SURFACE:0 SAMPLER:0
 Function control_fits:
  REG:64 STACK:0 SHARED:1024 LOCAL:16 CONSTANT[0]:400 TEXTURE:0 SURFACE:0 SAMPLER:0
"""


def controls():
    """Every planted control through every gate: {control: {"failed": [gates], "expected": gate or None, "ok": bool}},
    and "limits": the limits control. A control is ok when it fails exactly the gate it was built for."""
    spec = {"rows": [1, 64], "tune_rows": [64]}
    report = {}
    for name, (cls, expected) in CONTROLS.items():
        bench = cls()
        failed = [r["gate"] for r in run_structure(bench, None, spec, {}) if r["verdict"] == "fail"]
        report[name] = {"failed": failed, "expected": expected,
                        "ok": failed == ([expected] if expected else [])}
    failures, spills = check_limits(resource_usage(LIMITS_CONTROL))
    report["control-limits"] = {"failed": failures, "spills": spills, "ok": len(failures) == 2 and len(spills) == 1}
    return report
