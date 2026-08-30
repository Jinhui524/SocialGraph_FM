"""Private resource-limited worker for Facebook100 MATLAB sources."""

from __future__ import annotations

import ctypes
import json
import os
import sys
from pathlib import Path
from typing import Any


def _apply_memory_limit(maximum: int) -> Any:
    if maximum <= 0:
        raise ValueError("memory limit must be positive")
    if os.name != "nt":
        import resource

        getattr(resource, "setrlimit")(getattr(resource, "RLIMIT_AS"), (maximum, maximum))
        return None

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.CreateJobObjectW.argtypes = [ctypes.c_void_p, ctypes.c_wchar_p]
    kernel32.CreateJobObjectW.restype = ctypes.c_void_p
    kernel32.SetInformationJobObject.argtypes = [
        ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_uint32
    ]
    kernel32.SetInformationJobObject.restype = ctypes.c_int
    kernel32.GetCurrentProcess.argtypes = []
    kernel32.GetCurrentProcess.restype = ctypes.c_void_p
    kernel32.AssignProcessToJobObject.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
    kernel32.AssignProcessToJobObject.restype = ctypes.c_int
    job = kernel32.CreateJobObjectW(None, None)
    if not job:
        raise OSError(ctypes.get_last_error(), "CreateJobObjectW failed")

    class _BasicLimit(ctypes.Structure):
        _fields_ = [
            ("PerProcessUserTimeLimit", ctypes.c_longlong),
            ("PerJobUserTimeLimit", ctypes.c_longlong),
            ("LimitFlags", ctypes.c_uint32),
            ("MinimumWorkingSetSize", ctypes.c_size_t),
            ("MaximumWorkingSetSize", ctypes.c_size_t),
            ("ActiveProcessLimit", ctypes.c_uint32),
            ("Affinity", ctypes.c_size_t),
            ("PriorityClass", ctypes.c_uint32),
            ("SchedulingClass", ctypes.c_uint32),
        ]

    class _IoCounters(ctypes.Structure):
        _fields_ = [(name, ctypes.c_ulonglong) for name in (
            "ReadOperationCount", "WriteOperationCount", "OtherOperationCount",
            "ReadTransferCount", "WriteTransferCount", "OtherTransferCount",
        )]

    class _ExtendedLimit(ctypes.Structure):
        _fields_ = [
            ("BasicLimitInformation", _BasicLimit),
            ("IoInfo", _IoCounters),
            ("ProcessMemoryLimit", ctypes.c_size_t),
            ("JobMemoryLimit", ctypes.c_size_t),
            ("PeakProcessMemoryUsed", ctypes.c_size_t),
            ("PeakJobMemoryUsed", ctypes.c_size_t),
        ]

    information = _ExtendedLimit()
    information.BasicLimitInformation.LimitFlags = 0x100 | 0x2000
    information.ProcessMemoryLimit = maximum
    if not kernel32.SetInformationJobObject(
        job, 9, ctypes.byref(information), ctypes.sizeof(information)
    ):
        raise OSError(ctypes.get_last_error(), "SetInformationJobObject failed")
    if not kernel32.AssignProcessToJobObject(job, kernel32.GetCurrentProcess()):
        raise OSError(ctypes.get_last_error(), "AssignProcessToJobObject failed")
    return job


def _main(contract_path: Path) -> None:
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    expected = {"expectedKeys", "input", "maxArrayElements", "maxMemoryBytes", "output"}
    if set(contract) != expected:
        raise ValueError("MAT worker contract inventory is invalid")
    _job = _apply_memory_limit(int(contract["maxMemoryBytes"]))

    import numpy as np
    from scipy import sparse
    from scipy.io import loadmat

    arrays = {
        key: value
        for key, value in loadmat(Path(contract["input"]), spmatrix=True).items()
        if not key.startswith("__")
    }
    if set(arrays) != set(contract["expectedKeys"]):
        raise ValueError("MAT inventory does not match expected keys")
    maximum = int(contract["maxArrayElements"])
    for value in arrays.values():
        dtype = getattr(value, "dtype", None)
        if isinstance(dtype, np.dtype) and dtype.hasobject:
            raise ValueError("MAT object arrays are forbidden")
        elements = int(value.nnz) if sparse.issparse(value) else int(value.size)
        if elements > maximum:
            raise ValueError("MAT array element limit exceeded")
    adjacency = sparse.coo_matrix(arrays["A"])
    np.savez(
        Path(contract["output"]),
        a_row=adjacency.row.astype(np.int64, copy=False),
        a_col=adjacency.col.astype(np.int64, copy=False),
        a_data=adjacency.data,
        a_shape=np.asarray(adjacency.shape, dtype=np.int64),
        local_info=np.asarray(arrays["local_info"]),
    )


if __name__ == "__main__":
    try:
        if len(sys.argv) != 2:
            raise ValueError("MAT worker requires exactly one contract path")
        _main(Path(sys.argv[1]))
    except Exception as error:
        print(str(error), file=sys.stderr)
        raise SystemExit(2) from error
