"""row_reader.cpp built at load through Core's native build (crit_plugins.native.extension; host C++ only: the CUDA
runtime for host functions on a stream), once a process and cached across processes under a name hashed from the
source's path and bytes; None where it does not build, and device_rows.py reads rows with Python threads instead."""
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
    try:
        import torch
        from crit_plugins.native import extension
        cuda = torch.cuda.is_available()
        flags = ("-O3", "-std=c++20") + (("-DCRIT_ROWS_CUDA",) if cuda else ())
        return extension("crit_row_reader", [HERE / "row_reader.cpp"], flags=flags, cuda=cuda)
    except Exception as error:   # no compiler: Python threads read the rows
        WHY.append(str(error).splitlines()[0][:300])
        warnings.warn(f"lookup-table-rows: the native row reader did not build ({WHY[-1]}); Python threads read rows")
        return None
