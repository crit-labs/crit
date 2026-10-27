"""ctypes bindings of Inference's CUDA entries through the C ABI in crit_inference.h; standard library only.

The header is the one signature table: `signatures()` reads every CRIT_API declaration from it, and `Library`
gives each function its ctypes types from there, so the two cannot drift apart. Buffers are integer device
addresses (torch's data_ptr()), streams integer cudaStream_t handles (torch's cuda_stream), flags 0 or 1.
"""
import ctypes
import os
from pathlib import Path
import re
import sys

HEADER = Path(__file__).with_name("crit_inference.h")
ABI_VERSION = 1
PLAN_BYTES = 64
# One letter per C type: e a crit_status (raised as CudaError when nonzero), s a C string, p any pointer.
LETTERS = {"crit_status": "e", "int": "i", "int32_t": "i", "uint32_t": "u", "size_t": "z", "uint64_t": "q",
           "float": "f", "double": "d"}
CTYPES = {"e": ctypes.c_int, "i": ctypes.c_int, "u": ctypes.c_uint32, "z": ctypes.c_size_t, "q": ctypes.c_uint64,
          "f": ctypes.c_float, "d": ctypes.c_double, "p": ctypes.c_void_p, "s": ctypes.c_char_p}


class CudaError(RuntimeError):
    def __init__(self, function, code, message):
        super().__init__(f"{function}: {message} (cudaError {code})")
        self.function, self.code = function, code


def signatures(header=HEADER):
    """{name without crit_: (return letter, argument letters)} for every CRIT_API declaration."""
    text = re.sub(r"//[^\n]*", "", Path(header).read_text(encoding="utf-8"))
    table = {}
    for ret, name, args in re.findall(r"CRIT_API\s+([\w\s*]+?)\s*\b(crit_\w+)\((.*?)\);", text, re.S):
        letters = ""
        if args.strip() != "void":
            for arg in args.split(","):
                arg = " ".join(arg.split())
                letters += "p" if "*" in arg else LETTERS[arg.rsplit(" ", 1)[0].removeprefix("const ")]
        table[name.removeprefix("crit_")] = ("s" if "*" in ret else LETTERS[ret.strip()], letters)
    return table


def default_library():
    """CRIT_INFERENCE_LIBRARY, else the crit_inference_python shared library next to this file."""
    if os.environ.get("CRIT_INFERENCE_LIBRARY"):
        return Path(os.environ["CRIT_INFERENCE_LIBRARY"])
    name = {"win32": "crit_inference_python.dll", "darwin": "libcrit_inference_python.dylib"}.get(
        sys.platform, "libcrit_inference_python.so")
    return Path(__file__).with_name(name)


class Library:
    """The loaded C ABI: lib.st_gemv_fp8(...) calls crit_st_gemv_fp8 and raises CudaError on a nonzero status."""

    def __init__(self, path=None, header=HEADER):
        path = Path(path) if path else default_library()
        if not path.is_file():
            raise FileNotFoundError(f"crit Inference library not found: {path} (build the crit_inference_python "
                                    "target, or set CRIT_INFERENCE_LIBRARY)")
        self._dll = ctypes.CDLL(str(path))
        self.signatures = signatures(header)
        for name, (ret, args) in self.signatures.items():
            function = getattr(self._dll, "crit_" + name)
            function.restype = CTYPES[ret]
            function.argtypes = [CTYPES[a] for a in args]
            setattr(self, name, self._checked(name, function) if ret == "e" else function)
        if self.abi_version() != ABI_VERSION:
            raise RuntimeError(f"{path} has C ABI {self.abi_version()}, crit_inference.py expects {ABI_VERSION}")

    def _checked(self, name, function):
        def call(*args):
            code = function(*args)
            if code:
                raise CudaError("crit_" + name, code, self.error_string(code).decode())
        call.__name__ = name
        return call

    # Plans: opaque host buffers passed to launches as they are.
    def gemm_plan(self, fp4, n, k, tokens, wide_tokens=256, min_blocks=0, two_row_tiles=True):
        """(plan, scratch sizes {activation_bytes, token_scale_count, partial_floats}) of a W8A8 / W4A4 GEMM."""
        plan, sizes = plan_buffer(), [ctypes.c_size_t() for _ in range(3)]
        self.st_plan(plan, int(fp4), n, k, tokens, wide_tokens, min_blocks, int(two_row_tiles),
                     *(ctypes.byref(s) for s in sizes))
        return plan, dict(zip(("activation_bytes", "token_scale_count", "partial_floats"), (s.value for s in sizes)))

    def gemv_plan(self, fp4, n, k, groups=0, draft=False):
        plan = plan_buffer()
        self.st_plan_gemv(plan, int(fp4), n, k, groups, int(draft))
        return plan

    def a16_plan(self, fp4, n, k, tokens, min_blocks=0):
        """(plan, partial_floats) of a BF16-activation GEMM."""
        plan, partials = plan_buffer(), ctypes.c_size_t()
        self.st_plan_a16(plan, int(fp4), n, k, tokens, min_blocks, ctypes.byref(partials))
        return plan, partials.value

    # Sampling rows packed on the host, for the caller to copy into device memory.
    def request(self, temperature=0.0, top_p=1.0, min_p=0.0, top_k=0, repetition=1.0, frequency=0.0, presence=0.0,
                min_tokens=0, stream=0, seed=0, position=0, bias_ids=0, bias_values=0, bias_count=0, suppress_ids=0,
                suppress_count=0, counts=0, prompt_seen=0):
        row = ctypes.create_string_buffer(self.ts_request_bytes())
        self.ts_request_pack(row, temperature, top_p, min_p, top_k, repetition, frequency, presence, min_tokens, stream,
                             seed, position, bias_ids, bias_values, bias_count, suppress_ids, suppress_count, counts,
                             prompt_seen)
        return row.raw

    def params(self, temperature=0.0, top_p=1.0, top_k=0, stream=0, seed=0, position=0):
        row = ctypes.create_string_buffer(self.ts_params_bytes())
        self.ts_params_pack(row, temperature, top_p, top_k, stream, seed, position)
        return row.raw

    def packed_layout(self, draft_tokens, top=0, logprobs=False):
        offsets = (ctypes.c_uint32 * 6)()
        self.dr_packed_layout(draft_tokens, top, int(logprobs), offsets)
        return dict(zip(("matched", "frontier", "logprob", "top_ids", "top_logprobs", "words"), offsets))

    def host_alloc(self, size):
        """(host address, device address) of size bytes of pinned, mapped host memory; free with host_free(host)."""
        host, device = ctypes.c_void_p(), ctypes.c_void_p()
        self.host_alloc_mapped(size, ctypes.byref(host), ctypes.byref(device))
        return host.value, device.value


def plan_buffer():
    return (ctypes.c_uint64 * (PLAN_BYTES // 8))()


def read_into(host, path, offset, size):
    """Read size bytes of a file at offset straight into host memory (a host_alloc buffer)."""
    view = memoryview((ctypes.c_char * size).from_address(host)).cast("B")
    with open(path, "rb") as file:
        file.seek(offset)
        done = 0
        while done < size:
            got = file.readinto(view[done:])
            if not got:
                raise OSError(f"{path}: {size - done} bytes short at offset {offset + done}")
            done += got
