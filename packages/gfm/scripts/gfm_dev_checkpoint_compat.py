"""Archived temporal-v1-only compatibility launcher.

This file lives outside ``src/socialgraph_gfm`` and is therefore not part of
``code_identity_hash``.  It works around the already diagnosed duplicate
checkpoint-manifest publication in frozen temporal-v1 dev code. The archived
record must exist before this script is retained; it must never be used for
SocialGraph-FM Core work.
"""

# ruff: noqa: E402 -- repository source must precede the installed wheel.

from __future__ import annotations

import argparse
import ctypes
import inspect
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from socialgraph_gfm.canonical import canonical_json
from socialgraph_gfm.errors import ContractViolation
from socialgraph_gfm.gfm.checkpoint import (
    load_gfm_checkpoint,
    read_gfm_checkpoint_manifest,
)
from socialgraph_gfm.gfm.contracts import GfmCheckpointManifest
from socialgraph_gfm.gfm.corpus.common import exclusive_file_lock
from socialgraph_gfm.identity import code_identity_hash
import socialgraph_gfm.gfm_workflow as workflow


EXPECTED_CODE_HASH = "151879a24de3b867277433e1ba2b427c3864ad2ca4965b56ea71c8023dfb6365"
EXPECTED_CONFIG = "socialgraph-core.json"
EXPECTED_CONFIG_HASH = "c1b6e3fdd0db7a8e273d4c1bb515a98bcc728695b8605812bd3bceb52ecfed1c"
EXPECTED_EXPERIMENT_ID = "socialgraph-core-dev-8c4e2bb307b7a729"
EXPECTED_SEED = 20260820
EXPECTED_RUN_VARIANTS = {
    f"{EXPECTED_EXPERIMENT_ID}-core-base-{EXPECTED_SEED}": "core-base",
    f"{EXPECTED_EXPERIMENT_ID}-core-moe-{EXPECTED_SEED}": "core-moe",
}


class _MemoryStatusEx(ctypes.Structure):
    _fields_ = (
        ("dwLength", ctypes.c_ulong),
        ("dwMemoryLoad", ctypes.c_ulong),
        ("ullTotalPhys", ctypes.c_ulonglong),
        ("ullAvailPhys", ctypes.c_ulonglong),
        ("ullTotalPageFile", ctypes.c_ulonglong),
        ("ullAvailPageFile", ctypes.c_ulonglong),
        ("ullTotalVirtual", ctypes.c_ulonglong),
        ("ullAvailVirtual", ctypes.c_ulonglong),
        ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
    )


def _free_physical_memory_gib() -> float:
    status = _MemoryStatusEx()
    status.dwLength = ctypes.sizeof(status)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
        raise OSError(ctypes.get_last_error(), "GlobalMemoryStatusEx failed")
    return float(status.ullAvailPhys) / float(1024**3)


def _install_compatibility_boundary() -> None:
    original_write_contract = workflow._write_contract

    def write_contract(path: Path, value: Any) -> None:
        if not isinstance(value, GfmCheckpointManifest):
            original_write_contract(path, value)
            return
        path = Path(path)
        artifact = Path(value.artifact_path)
        expected_path = (artifact.parent / f"{value.checkpoint_id}.manifest.json").resolve()
        if (
            path.resolve() != expected_path
            or path.name != f"{value.checkpoint_id}.manifest.json"
            or not artifact.is_absolute()
            or artifact.name != f"{value.checkpoint_id}.pt"
            or path.parent.resolve() != artifact.parent.resolve()
            or path.is_symlink()
            or artifact.is_symlink()
        ):
            raise ContractViolation(
                "Checkpoint compatibility boundary received an unexpected manifest path"
            )
        # save_gfm_checkpoint is the sole publisher.  The legacy workflow call
        # may only become a verified no-op; it may never repair or overwrite.
        if not path.is_file() or not artifact.is_file():
            raise ContractViolation(
                "Primary checkpoint publication is incomplete at compatibility boundary"
            )
        expected_bytes = canonical_json(value).encode("utf-8")
        if path.read_bytes() != expected_bytes:
            raise ContractViolation("Checkpoint manifest bytes differ after primary publication")
        persisted = read_gfm_checkpoint_manifest(path)
        if canonical_json(persisted) != canonical_json(value):
            raise ContractViolation("Checkpoint manifest differs after its primary publication")
        load_gfm_checkpoint(persisted, map_location="cpu")

    workflow._write_contract = write_contract


def _validate_resume_identity(root: Path, run_id: str, device: str) -> None:
    expected_variant = EXPECTED_RUN_VARIANTS.get(run_id)
    if expected_variant is None:
        raise RuntimeError("Compatibility resume is restricted to the frozen dev runs")
    state_path = root / "runs" / "gfm" / EXPECTED_EXPERIMENT_ID / run_id / "run-state.json"
    try:
        state = json.loads(state_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise RuntimeError(f"Cannot read the expected dev run state: {state_path}") from error
    expected = {
        "schemaVersion": "gfm.workflow-run-state/1.0",
        "runId": run_id,
        "experimentId": EXPECTED_EXPERIMENT_ID,
        "runKind": "pretrain",
        "phase": "dev",
        "variant": expected_variant,
        "seed": EXPECTED_SEED,
        "configHash": EXPECTED_CONFIG_HASH,
        "codeHash": EXPECTED_CODE_HASH,
        "device": device,
        "status": "running",
    }
    mismatches = {
        key: {"expected": value, "observed": state.get(key)}
        for key, value in expected.items()
        if state.get(key) != value
    }
    if mismatches:
        raise RuntimeError("Frozen dev run identity mismatch: " + canonical_json(mismatches))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--minimum-free-memory-gib", type=float, default=8.0)
    parser.add_argument("--memory-wait-timeout-minutes", type=int, default=1440)
    parser.add_argument("--poll-seconds", type=int, default=30)
    subparsers = parser.add_subparsers(dest="operation", required=True)
    resume = subparsers.add_parser("resume")
    resume.add_argument("--run-id", required=True)
    resume.add_argument("--device", default="cuda")
    pretrain = subparsers.add_parser("pretrain")
    pretrain.add_argument("--phase", choices=("dev",), default="dev")
    pretrain.add_argument("--variant", choices=("core-moe",), required=True)
    pretrain.add_argument("--seed", type=int, default=EXPECTED_SEED)
    pretrain.add_argument("--device", default="cuda")
    pretrain.add_argument("--config", default=EXPECTED_CONFIG)
    return parser


def main() -> int:
    args = _parser().parse_args()
    workflow_source = Path(inspect.getsourcefile(workflow) or "").resolve()
    if workflow_source != REPOSITORY_ROOT / "src" / "socialgraph_gfm" / "gfm_workflow.py":
        raise RuntimeError(f"Unexpected workflow source: {workflow_source}")
    if code_identity_hash() != EXPECTED_CODE_HASH:
        raise RuntimeError("Frozen dev source identity changed; refusing compatibility execution")
    root = Path(args.root).expanduser().resolve()
    owner_lock = (
        root / "reports" / "gfm" / "automation" / "dev-after-wikimedia-embedding.owner.lock"
    )
    with exclusive_file_lock(owner_lock):
        if args.device != "cuda":
            raise RuntimeError("Frozen dev compatibility execution requires CUDA")
        if not math.isfinite(args.minimum_free_memory_gib) or args.minimum_free_memory_gib < 8.0:
            raise RuntimeError("The dev memory gate cannot be lowered below 8 GiB")
        if args.poll_seconds < 10 or args.poll_seconds > 60:
            raise RuntimeError("poll-seconds must be between 10 and 60")
        if args.memory_wait_timeout_minutes < 1:
            raise RuntimeError("memory wait timeout must be positive")
        deadline = time.monotonic() + args.memory_wait_timeout_minutes * 60
        while True:
            if code_identity_hash() != EXPECTED_CODE_HASH:
                raise RuntimeError("Frozen dev source changed during memory wait")
            free_gib = _free_physical_memory_gib()
            if free_gib >= args.minimum_free_memory_gib:
                break
            if time.monotonic() >= deadline:
                raise RuntimeError("Timed out waiting for the 8 GiB dev memory safety gate")
            print(
                canonical_json(
                    {
                        "event": "gfm.dev-checkpoint-compat-memory-wait",
                        "freePhysicalMemoryGiB": round(free_gib, 3),
                        "minimumFreeMemoryGiB": args.minimum_free_memory_gib,
                    }
                ),
                file=sys.stderr,
                flush=True,
            )
            time.sleep(args.poll_seconds)
        if args.operation == "resume":
            _validate_resume_identity(root, args.run_id, args.device)
        elif (
            args.phase != "dev"
            or args.variant != "core-moe"
            or args.seed != EXPECTED_SEED
            or args.config != EXPECTED_CONFIG
        ):
            raise RuntimeError(
                "Compatibility pretrain is restricted to the frozen core-moe dev cell"
            )
        _install_compatibility_boundary()
        if args.operation == "resume":
            result = workflow.resume_gfm(root=root, run_id=args.run_id, device=args.device)
        else:
            result = workflow.pretrain_gfm(
                root=root,
                phase=args.phase,
                config=args.config,
                device=args.device,
                variant=args.variant,
                seed=args.seed,
            )
    if code_identity_hash() != EXPECTED_CODE_HASH:
        raise RuntimeError("Frozen dev source changed during compatibility execution")
    print(canonical_json(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
