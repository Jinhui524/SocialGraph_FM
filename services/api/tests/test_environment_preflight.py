from __future__ import annotations

from pathlib import Path

import pytest

from tools.environment_preflight import (
    PreflightConfigurationError,
    evaluate_environment,
    parse_exact_constraints,
)


def test_exact_constraints_reject_ranges_and_duplicates(tmp_path: Path) -> None:
    valid = tmp_path / "valid.txt"
    valid.write_text("FastAPI==1.2.3\nnumpy==1.26.4\n", encoding="utf-8")
    assert parse_exact_constraints(valid) == {
        "fastapi": "1.2.3",
        "numpy": "1.26.4",
    }

    ranged = tmp_path / "ranged.txt"
    ranged.write_text("fastapi>=1\n", encoding="utf-8")
    with pytest.raises(PreflightConfigurationError, match="exact"):
        parse_exact_constraints(ranged)

    duplicate = tmp_path / "duplicate.txt"
    duplicate.write_text("typing_extensions==1\ntyping-extensions==1\n", encoding="utf-8")
    with pytest.raises(PreflightConfigurationError, match="duplicates"):
        parse_exact_constraints(duplicate)


def test_environment_report_blocks_version_drift_without_environment_values(
    tmp_path: Path,
) -> None:
    constraints = tmp_path / "constraints.txt"
    constraints.write_text("example==2.0\n", encoding="utf-8")
    report = evaluate_environment(
        profile_name="test",
        profile={"pythonExact": "3.12.4", "requireIsolated": True},
        constraints_path=constraints,
        expected_packages={"example": "2.0"},
        probe={
            "python": {
                "version": "3.12.3",
                "implementation": "CPython",
                "executable": "X:/venv/python.exe",
                "prefix": "X:/venv",
                "basePrefix": "X:/base",
                "architectureBits": 64,
                "isolation": {
                    "prefixDiffersFromBase": True,
                    "pyvenvConfigPresent": False,
                    "condaEnvironmentPath": False,
                },
            },
            "platform": {"system": "Windows", "release": "test", "machine": "AMD64"},
            "packages": {"example": "1.0"},
            "imports": {},
        },
        pip_check={"ok": True, "returnCode": 0, "output": "No broken requirements found."},
    )
    assert report["status"] == "blocked"
    assert {issue["code"] for issue in report["issues"]} == {
        "PYTHON_VERSION_MISMATCH",
        "PACKAGE_VERSION_MISMATCH",
    }
    assert report["environmentValuesCaptured"] is False
    assert len(report["runtimeFingerprint"]) == 64


def test_environment_report_accepts_a_pinned_python_patch_series(tmp_path: Path) -> None:
    constraints = tmp_path / "constraints.txt"
    constraints.write_text("example==2.0\n", encoding="utf-8")
    report = evaluate_environment(
        profile_name="test",
        profile={"pythonSeries": "3.12", "requireIsolated": True},
        constraints_path=constraints,
        expected_packages={"example": "2.0"},
        probe={
            "python": {
                "version": "3.12.13",
                "implementation": "CPython",
                "executable": "X:/venv/python.exe",
                "prefix": "X:/venv",
                "basePrefix": "X:/base",
                "architectureBits": 64,
                "isolation": {
                    "prefixDiffersFromBase": True,
                    "pyvenvConfigPresent": True,
                    "condaEnvironmentPath": False,
                },
            },
            "platform": {"system": "Windows", "release": "test", "machine": "AMD64"},
            "packages": {"example": "2.0"},
            "imports": {"example": {"ok": True}},
        },
        pip_check={"ok": True, "returnCode": 0, "output": "No broken requirements found."},
    )
    assert report["status"] == "ready"
