import hashlib

import pytest

from socialgraph_gfm.runtime import runtime_report
from socialgraph_gfm.tensor_digest import canonical_tensor_digest

pytestmark = pytest.mark.skipif(
    not runtime_report("cuda")["runtimeReady"],
    reason="exact CUDA runtime is not installed",
)


def test_tensor_digest_is_explicit_little_endian_and_device_independent():
    import torch

    cpu = torch.tensor([1, 256, -2], dtype=torch.int32)
    expected = hashlib.sha256(
        b"\x01\x00\x00\x00\x00\x01\x00\x00\xfe\xff\xff\xff"
    ).hexdigest()
    cpu_digest = canonical_tensor_digest(cpu)
    cuda_digest = canonical_tensor_digest(cpu.cuda())
    assert cpu_digest["byteOrder"] == "little"
    assert cpu_digest["sha256"] == expected
    assert cpu_digest == cuda_digest
