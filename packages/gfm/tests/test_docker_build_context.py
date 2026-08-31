from __future__ import annotations

from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]


def test_public_cuda_component_dockerfile_is_retired() -> None:
    assert not (PROJECT / "Dockerfile").exists()
