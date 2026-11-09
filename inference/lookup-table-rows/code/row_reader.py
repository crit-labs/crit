"""row_reader.cpp built at load through torch's extension builder (host C++ only: the CUDA runtime for host
functions on a stream), once a process and cached across processes under a name hashed from the source's path and
bytes; None where it does not build, and device_rows.py reads rows with Python threads instead."""
import hashlib
from pathlib import Path
import threading
import warnings

HERE = Path(__file__).resolve().parent
_BUILT = []
_LOCK = threading.Lock()
WHY = []


def native():
    with _LOCK:
        if not _BUILT:
            _BUILT.append(_build())
    return _BUILT[0]


def _build():
    """With CUDA (a device torch sees): host functions on a stream too; without: the reader alone."""
    import os
    before = os.environ.get("TORCH_CUDA_ARCH_LIST")
    try:
        import torch
        from torch.utils.cpp_extension import load
        cuda = torch.cuda.is_available()
        source = HERE / "row_reader.cpp"
        copy = hashlib.sha256(str(source).encode() + source.read_bytes() + bytes([cuda])).hexdigest()[:12]
        if cuda:   # host code only: any target, never the build's device query
            major, minor = torch.cuda.get_device_capability()
            os.environ["TORCH_CUDA_ARCH_LIST"] = f"{major}.{minor}"
        return load(name=f"crit_row_reader_{copy}", sources=[str(source)],
                    extra_cflags=["-O3", "-std=c++20"] + (["-DCRIT_ROWS_CUDA"] if cuda else []), with_cuda=cuda,
                    verbose=False)
    except Exception as error:   # no compiler: Python threads read the rows
        WHY.append(f"{type(error).__name__}: {error}".splitlines()[0][:300])
        warnings.warn(f"lookup-table-rows: the native row reader did not build ({WHY[-1]}); Python threads read rows")
        return None
    finally:
        if before is None:
            os.environ.pop("TORCH_CUDA_ARCH_LIST", None)
        else:
            os.environ["TORCH_CUDA_ARCH_LIST"] = before
