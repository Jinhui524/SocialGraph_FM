"""Canonical tensor identities shared by materialization and checkpoints."""

from __future__ import annotations

import hashlib
import sys
from typing import Any

from .errors import ContractViolation


def canonical_tensor_digest(tensor: Any) -> dict[str, Any]:
    """Hash contiguous CPU tensor bytes in explicit little-endian order.

    ``numpy().tobytes()`` otherwise follows native byte order, which would make the same
    logical tensor hash differently on a big-endian runner. One-byte dtypes are naturally
    byte-order independent.
    """

    canonical = tensor.detach().cpu().contiguous()
    if canonical.is_floating_point() and not bool(canonical.isfinite().all()):
        raise ContractViolation("Canonical tensor contains NaN or Infinity")
    array = canonical.numpy()
    if array.dtype.itemsize > 1:
        byteorder = array.dtype.byteorder
        if byteorder == ">" or (byteorder == "=" and sys.byteorder == "big"):
            array = array.byteswap().view(array.dtype.newbyteorder("<"))
        elif byteorder != "<":
            array = array.view(array.dtype.newbyteorder("<"))
    return {
        "dtype": str(canonical.dtype),
        "shape": list(canonical.shape),
        "byteOrder": "little" if array.dtype.itemsize > 1 else "not-applicable",
        "sha256": hashlib.sha256(array.tobytes(order="C")).hexdigest(),
    }
