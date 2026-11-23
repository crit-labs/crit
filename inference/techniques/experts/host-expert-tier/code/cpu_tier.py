"""The CPU expert tier from Python: an expert table over bundles read in place (bundles.py), one layer's routes
computed by the native tier (host_tier.h, tier_capi.cpp, built with host-expert-arithmetic's rows by its native.py) or,
where it cannot be built, by the same declared order in torch (host-expert-arithmetic's declared.py), the same bits.

    tier = CpuTier(hidden, inter, "nvfp4", arithmetic_code, quantized_code)
    tier.add(key, bundle)            # a CPU uint8 bundle the tier reads in place (kept alive here)
    y = tier.compute(keys, token, x) # keys, token [routes] int32; x [tokens][hidden] FP32 -> y [routes][hidden]
    tier.handler()                   # (C function address, context) for one of the host's handoffs: the device's posts

Native work runs on the host's worker pool (Core's, crit_plugins.host: one a process, one worker per physical core
but the caller's, pinned, awake through a decode step's layers and asleep between steps). workers and pin size that
pool when this tier is the first to ask for it.
"""
import ctypes
from pathlib import Path

from . import bundles

c_size, c_ptr, c_int, c_float, c_char_p, c_u64 = (ctypes.c_size_t, ctypes.c_void_p, ctypes.c_int, ctypes.c_float,
                                                  ctypes.c_char_p, ctypes.c_uint64)
ERROR_BYTES = 2048
STATS = ("jobs", "routes", "expert_chunks", "ms_inputs", "ms_gate_up", "ms_down", "ms_total", "weight_bytes")
TIER_SOURCES = ("host_tier.cpp", "tier_capi.cpp")
HERE = Path(__file__).resolve().parent


def arithmetic_modules(arithmetic):
    """host-expert-arithmetic's native, rows and declared modules from its code package."""
    from importlib import import_module
    return tuple(import_module(f"{arithmetic.__name__}.{name}") for name in ("native", "rows", "declared"))


def build_library(arithmetic, quantized):
    """The native tier library: host-expert-arithmetic's rows and this tier's sources in one shared library (the
    pool it runs on is Core's: only its header is read here)."""
    from crit_plugins.host import INCLUDE
    native, _, _ = arithmetic_modules(arithmetic)
    units = list(native.arithmetic_units())
    units += [(HERE / name, ()) for name in TIER_SOURCES]
    return native.build("crit_host_tier", units, native.includes(quantized) + [HERE, INCLUDE])


def library_isa(lib):
    """The integer path a built tier library runs on this CPU."""
    lib.crit_hx_int_isa.restype = c_char_p
    lib.crit_hx_int_isa.argtypes = []
    return lib.crit_hx_int_isa().decode()


class CpuTier:
    """One expert table on the host's worker pool. native: True builds the native tier (raising when it cannot),
    False runs the torch fallback, None tries native and falls back."""

    def __init__(self, hidden, inter, format, arithmetic, quantized, native=None, workers=-1, pin=True,
                 rows_per_task=128, isa="best"):
        from crit_plugins.host import current
        self.hidden, self.inter, self.format = int(hidden), int(inter), format
        self.plan = bundles.layout(format, self.hidden, self.inter)
        self.math = bundles.arithmetic(format)
        native_module, rows_module, self.declared = arithmetic_modules(arithmetic)
        self.handle, self.lib, self.rows, self.reason, self.pool = None, None, None, None, None
        self.entries = {}
        if native is not False:
            try:
                self.lib = build_library(arithmetic, quantized)
                self.pool = current().pool(workers, pin)
            except native_module.NativeUnavailable as error:
                if native:
                    raise
                self.lib, self.reason = None, str(error).splitlines()[0]
        if self.lib is not None:
            self._declare()
            self.rows = rows_module.HostRows(self.lib)
            error = ctypes.create_string_buffer(ERROR_BYTES)
            entry, pool = self.pool.native
            self.handle = self.lib.crit_tier_create(self.hidden, self.inter, c_ptr(entry), c_ptr(pool),
                                                    int(rows_per_task), rows_module.ISA[isa], error, ERROR_BYTES)
            if not self.handle:
                raise ValueError(error.value.decode() or "the CPU tier did not start")

    @property
    def native(self):
        return self.handle is not None

    def _declare(self):
        lib = self.lib
        lib.crit_tier_create.restype = c_ptr
        lib.crit_tier_create.argtypes = [c_size, c_size, c_ptr, c_ptr, c_size, c_int, c_ptr, c_size]
        lib.crit_tier_destroy.argtypes = [c_ptr]
        lib.crit_tier_destroy.restype = None
        lib.crit_tier_isa.argtypes, lib.crit_tier_isa.restype = [c_ptr], c_char_p
        lib.crit_tier_set_expert.restype = c_int
        lib.crit_tier_set_expert.argtypes = [c_ptr, c_size, c_ptr, c_ptr, c_ptr, c_ptr, c_ptr, c_ptr, c_int, c_float,
                                             c_int, c_float, c_int, c_int, c_float, c_int, c_ptr, c_size]
        lib.crit_tier_set_layout.argtypes, lib.crit_tier_set_layout.restype = [c_ptr, c_int], c_int
        lib.crit_tier_interleaved_bytes.argtypes, lib.crit_tier_interleaved_bytes.restype = [c_size, c_size], c_size
        lib.crit_tier_interleave.restype = c_int
        lib.crit_tier_interleave.argtypes = [c_ptr, c_size, c_size, c_ptr, c_ptr, c_size]
        lib.crit_tier_compute.restype = c_int
        lib.crit_tier_compute.argtypes = [c_ptr, c_int, c_ptr, c_ptr, c_ptr, c_size, c_int, c_ptr, c_size, c_ptr, c_size]
        lib.crit_tier_stats.argtypes, lib.crit_tier_stats.restype = [c_ptr, c_ptr], None
        lib.crit_tier_reset_stats.argtypes, lib.crit_tier_reset_stats.restype = [c_ptr], None
        lib.crit_tier_payload_layout.argtypes = [c_size, c_size, c_size, c_ptr]
        lib.crit_tier_payload_layout.restype = None

    def _call(self, function, *args):
        error = ctypes.create_string_buffer(ERROR_BYTES)
        if function(*args, error, ERROR_BYTES):
            raise ValueError(error.value.decode() or "the CPU tier failed")

    def interleave(self, codes, rows, cols):
        """A [rows][cols / 2] rows-layout code matrix (CPU uint8) in the interleaved layout, a new CPU uint8 tensor."""
        import torch
        codes = codes.contiguous()
        out = torch.empty(int(self.lib.crit_tier_interleaved_bytes(int(rows), int(cols))), dtype=torch.uint8)
        self._call(self.lib.crit_tier_interleave, c_ptr(codes.data_ptr()), int(rows), int(cols), c_ptr(out.data_ptr()))
        return out

    def add(self, key, bundle, interleaved=None):
        """Expert `key` (a table index >= 0) from a CPU uint8 bundle of this tier's layout, read in place; with
        interleaved=(gate_up codes, down codes) in the interleaved layout (interleave()), its codes read from those
        (a CPU-side copy: the faster rows, the same bits) and the rest from the bundle."""
        import torch
        if bundle.device.type != "cpu" or bundle.dtype != torch.uint8 or not bundle.is_contiguous() or \
                bundle.numel() < self.plan["bundle_bytes"]:
            raise ValueError(f"a bundle is a contiguous CPU uint8 tensor of {self.plan['bundle_bytes']} bytes")
        gu_global, dn_global = bundles.globals_of(bundle, self.plan)
        if self.math["input_kind"] == "e2m1x2" and not (gu_global > 0 and dn_global > 0):
            raise ValueError("an nvfp4 expert computed W4A4 needs its input scales (the bundle's input[0] and input[1])")
        self.entries[int(key)] = (bundle, gu_global, dn_global, interleaved)
        if self.native:
            base = bundle.data_ptr()
            offsets = [base + self.plan["components"][name][0] for name in bundles.OFFSETS]
            if interleaved is not None:   # OFFSETS: gate_up, gate_up_scale, gate_up_inverse, down, down_scale, ...
                offsets[0], offsets[3] = interleaved[0].data_ptr(), interleaved[1].data_ptr()
            self._call_layout(interleaved is not None)
            m = self.math
            self._call(self.lib.crit_tier_set_expert, self.handle, int(key), *[c_ptr(o) for o in offsets],
                       1 if m["weights"] == "int4" else 0, float(m["factor"]), 1 if m["input_kind"] == "int8" else 0,
                       float(gu_global), int(m["block"]), 1 if m["inter_kind"] == "int8" else 0, float(dn_global),
                       int(m["block"]))

    def _call_layout(self, interleaved):
        if getattr(self, "_layout", False) != bool(interleaved):
            self.lib.crit_tier_set_layout(self.handle, int(bool(interleaved)))
            self._layout = bool(interleaved)

    def compute(self, keys, token, x, out=None):
        """Each route's unweighted row: keys and token [routes] int32 (table keys, rows of x), x [tokens][hidden]."""
        import torch
        keys = keys.to(torch.int32).contiguous()
        token = token.to(torch.int32).contiguous()
        x = x.float().contiguous()
        routes = keys.numel()
        out = torch.zeros(routes, self.hidden, dtype=torch.float32) if out is None else out
        if routes == 0:
            return out
        if self.native:
            self._call(self.lib.crit_tier_compute, self.handle, routes, c_ptr(keys.data_ptr()), c_ptr(token.data_ptr()),
                       c_ptr(x.data_ptr()), self.hidden, x.shape[0], c_ptr(out.data_ptr()), self.hidden)
            return out
        for key in sorted(set(keys.tolist())):
            picked = (keys == key).nonzero(as_tuple=True)[0]
            out[picked] = self.expert_rows(key, x[token[picked].long()])
        return out

    def expert_rows(self, key, x):
        """One expert's rows by the torch declared order (the fallback, and the oracle the native tier matches)."""
        bundle, gu_global, dn_global, _ = self.entries[int(key)]
        v = lambda name: bundles.view(bundle, self.plan, name)  # noqa: E731
        m = self.math
        return self.declared.expert_rows(x, (v("gate_up"), v("gate_up_scale"), v("gate_up_inverse")),
                                         (v("down"), v("down_scale"), v("down_inverse")), m["weights"], m["factor"],
                                         m["input_kind"], gu_global, m["inter_kind"], dn_global, 32)

    def combine(self, y, ids, weights):
        """The route-order FMA sum of each token's rows (y [tokens * top][hidden])."""
        if self.rows is not None:
            return self.rows.combine(y, ids, weights)
        return self.declared.combine(y, ids, weights)

    def handler(self):
        """(address of crit_tier_serve, the tier's context): the handler of a handoff (crit_plugins.host) that posts
        this tier's routes; None without the native tier."""
        if not self.native:
            return None
        return ctypes.cast(self.lib.crit_tier_serve, c_ptr).value, int(self.handle)

    def payload_layout(self, routes, tokens):
        """Offsets of a handoff payload of FP32 rows for these capacities: {experts, token, x, out, bytes}."""
        if not self.native:
            raise ValueError("the handoff payload is the native tier's")
        values = (ctypes.c_int64 * 5)()
        self.lib.crit_tier_payload_layout(self.hidden, int(routes), int(tokens), values)
        return dict(zip(("experts", "token", "x", "out", "bytes"), list(values)))

    def stats(self):
        """The tier's counters, and the host pool's it runs on (pool_batches, pool_wakes, workers: the process's)."""
        if not self.native:
            return {"native": False}
        values = (ctypes.c_double * len(STATS))()
        self.lib.crit_tier_stats(self.handle, values)
        pool = self.pool.stats()
        return {"native": True, "isa": self.lib.crit_tier_isa(self.handle).decode(), **dict(zip(STATS, list(values))),
                "pool_batches": pool["batches"], "pool_wakes": pool["wakes"], "workers": pool["workers"]}

    def reset_stats(self):
        if self.native:
            self.lib.crit_tier_reset_stats(self.handle)

    def close(self):
        if self.handle:
            self.lib.crit_tier_destroy(self.handle)
            self.handle = None
        self.entries = {}

    def __del__(self):
        try:
            self.close()
        except Exception:  # noqa: BLE001 - interpreter teardown
            pass
