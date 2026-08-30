"""One-step, synthetic-only autograd/checkpoint/resume smoke verification."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from .canonical import canonical_json, canonical_sha256
from .checkpoint import read_manifest, save_checkpoint
from .contracts import RunStatus, SmokeCorpusManifest, SmokeRunMetrics, SmokeTrainingRunManifest
from .errors import RunCancelled
from .fixtures import fixture_names, get_fixture, smoke_fit_node_ids
from .identity import DEFAULT_SMOKE_SEED, SMOKE_SCHEMA_VERSION, code_identity_hash, smoke_config
from .materialize import homogeneous_tensors, materialize
from .registry import LocalRegistry
from .runtime import RunContext, artifact_root, require_ml_runtime, runtime_report, set_seed
from .tensor_digest import canonical_tensor_digest


def _peak_memory_mb(torch, device: str) -> float:
    if device == "cuda":
        return float(torch.cuda.max_memory_allocated()) / (1024.0 * 1024.0)
    try:
        import resource

        usage = resource.getrusage(resource.RUSAGE_SELF)  # type: ignore[attr-defined]
        peak = float(usage.ru_maxrss)
        return peak / 1024.0 if sys.platform != "darwin" else peak / (1024.0 * 1024.0)
    except ImportError:
        # Windows PROCESS_MEMORY_COUNTERS.PeakWorkingSetSize.
        import ctypes
        from ctypes import wintypes

        class ProcessMemoryCounters(ctypes.Structure):
            _fields_ = [
                ("cb", wintypes.DWORD),
                ("PageFaultCount", wintypes.DWORD),
                ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t),
            ]

        counters = ProcessMemoryCounters()
        counters.cb = ctypes.sizeof(counters)
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        psapi = ctypes.WinDLL("psapi", use_last_error=True)
        kernel32.GetCurrentProcess.argtypes = ()
        kernel32.GetCurrentProcess.restype = wintypes.HANDLE
        psapi.GetProcessMemoryInfo.argtypes = (
            wintypes.HANDLE,
            ctypes.POINTER(ProcessMemoryCounters),
            wintypes.DWORD,
        )
        psapi.GetProcessMemoryInfo.restype = wintypes.BOOL
        handle = kernel32.GetCurrentProcess()
        if not psapi.GetProcessMemoryInfo(handle, ctypes.byref(counters), counters.cb):
            return 0.0
        return float(counters.PeakWorkingSetSize) / (1024.0 * 1024.0)


def _model(torch, input_dim: int, hidden_dim: int = 8):
    class TinyMessageEncoder(torch.nn.Module):
        """Test-only message passing; never registered or reported as a model baseline."""

        def __init__(self) -> None:
            super().__init__()
            self.input = torch.nn.Linear(input_dim, hidden_dim)
            self.output = torch.nn.Linear(hidden_dim, 1)

        def forward(self, x, edge_index):
            aggregates = []
            for target in range(x.shape[0]):
                sources = edge_index[0, edge_index[1] == target]
                aggregates.append(
                    x[sources].mean(dim=0) if sources.numel() else torch.zeros_like(x[target])
                )
            aggregate = torch.stack(aggregates, dim=0)
            hidden = torch.relu(self.input((x + aggregate) * 0.5))
            return self.output(hidden).mean()

    return TinyMessageEncoder()


def _output_digest(value) -> str:
    return canonical_tensor_digest(value)["sha256"]


def _logical_run_manifest_hash(manifest: SmokeTrainingRunManifest) -> str:
    """Hash experiment meaning, excluding execution identity, timestamps and paths."""

    return canonical_sha256(
        {
            "schemaVersion": manifest.schema_version,
            "runKind": manifest.run_kind,
            "status": manifest.status,
            "seed": manifest.seed,
            "codeHash": manifest.code_hash,
            "environmentHash": manifest.environment_hash,
            "corpus": manifest.corpus,
            "configHash": manifest.config_hash,
            "failureCode": manifest.failure_code,
        }
    )


def _logical_checkpoint_manifest_hash(checkpoint) -> str:
    """Hash checkpoint meaning, excluding ID, artifact bytes/path and creation time."""

    return canonical_sha256(
        {
            "schemaVersion": checkpoint.schema_version,
            "step": checkpoint.step,
            "smokeOnly": checkpoint.smoke_only,
            "stateHash": checkpoint.state_hash,
            "configHash": checkpoint.config_hash,
        }
    )


def run_smoke(
    *,
    fixture: str = "both",
    device: Literal["cpu", "cuda"] = "cpu",
    root: str | Path | None = None,
    seed: int = DEFAULT_SMOKE_SEED,
) -> dict[str, Any]:
    if device not in ("cpu", "cuda"):
        raise ValueError("device must be cpu or cuda")
    torch, _ = require_ml_runtime(device)
    started_clock = time.perf_counter()
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    selected_root = artifact_root(root)
    results = []
    for fixture_name in fixture_names(fixture):
        fixture_clock = time.perf_counter()
        snapshot = get_fixture(fixture_name)
        run_id = f"smoke-{fixture_name}-{uuid.uuid4().hex[:12]}"
        context = RunContext(run_id=run_id, root=selected_root)
        context.prepare()
        registry = LocalRegistry(selected_root / "registry" / "registry.sqlite3")
        started_at = datetime.now(UTC)
        corpus = SmokeCorpusManifest(
            corpusId=f"synthetic-{fixture_name}",
            version="1.0",
            purpose="synthetic_test_only",
            licenseId="INTERNAL-SYNTHETIC-NONDATA",
            adapter="socialgraph_gfm.fixtures",
            split="synthetic_smoke",
            sourceHash=snapshot.ref.content_hash,
            snapshotRefs=(snapshot.ref,),
        )
        runtime_hash = runtime_report(device)["environmentHash"]
        materialized = materialize(
            snapshot,
            purpose="training_smoke",
            fit_node_ids=smoke_fit_node_ids(fixture_name),
            device=device,
        )
        x, edge_index = homogeneous_tensors(materialized)
        config = smoke_config(
            fixture=fixture_name,
            seed=seed,
            device=device,
            hidden_dim=8,
            input_dim=int(x.shape[1]),
        )
        running = SmokeTrainingRunManifest(
            runId=run_id,
            runKind="smoke",
            status=RunStatus.RUNNING,
            seed=seed,
            codeHash=code_identity_hash(),
            environmentHash=runtime_hash,
            corpus=corpus,
            configHash=canonical_sha256(config),
            startedAt=started_at,
        )
        registry.record_run(running)
        context.log("run_started", fixture=fixture_name, device=device)
        try:
            context.check_cancelled()
            set_seed(seed, device)
            x = x.to(device)
            edge_index = edge_index.to(device)
            model = _model(torch, config["inputDim"], config["hiddenDim"]).to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
            before = [parameter.detach().clone() for parameter in model.parameters()]
            optimizer.zero_grad(set_to_none=True)
            prediction = model(x, edge_index)
            loss = torch.nn.functional.mse_loss(prediction, torch.tensor(0.25, device=device))
            if not bool(torch.isfinite(loss)):
                raise RuntimeError("smoke loss is not finite")
            loss.backward()
            optimizer.step()
            if not any(
                not torch.equal(previous, current.detach())
                for previous, current in zip(before, model.parameters(), strict=True)
            ):
                raise RuntimeError("optimizer step did not update any parameter")
            model.eval()
            with torch.no_grad():
                expected = model(x, edge_index)
            checkpoint_id = f"{run_id}-step1"
            checkpoint = save_checkpoint(
                context.directory / "checkpoints",
                checkpoint_id=checkpoint_id,
                run_id=run_id,
                step=1,
                model_state=model.state_dict(),
                optimizer_state=optimizer.state_dict(),
                config=config,
            )
            manifest_path = context.directory / "checkpoints" / f"{checkpoint_id}.manifest.json"
            registry.record_checkpoint(checkpoint)
            verification = verify_in_subprocess(
                manifest_path=manifest_path,
                fixture=fixture_name,
                device=device,
            )
            if verification["outputDigest"] != _output_digest(expected):
                raise RuntimeError("fresh-process checkpoint output differs from saved model")
            if verification.get("optimizerRestored") is not True:
                raise RuntimeError("fresh-process checkpoint did not restore optimizer state")
            finished = SmokeTrainingRunManifest(
                **{
                    **running.model_dump(mode="python", by_alias=True),
                    "status": RunStatus.SUCCEEDED,
                    "finishedAt": datetime.now(UTC),
                    "artifacts": (str(manifest_path.resolve()),),
                    "smokeMetrics": SmokeRunMetrics(
                        device=device,
                        elapsedSeconds=round(time.perf_counter() - fixture_clock, 6),
                        maxMemoryMb=round(_peak_memory_mb(torch, device), 3),
                        freshProcessVerified=True,
                        optimizerRestored=True,
                        checkpointStateHash=checkpoint.state_hash,
                        checkpointArtifactSha256=checkpoint.artifact_sha256,
                    ),
                }
            )
            registry.record_run(finished)
            (context.directory / "run.manifest.json").write_text(
                canonical_json(finished), encoding="utf-8", newline="\n"
            )
            context.log("run_succeeded", loss=float(loss.detach().cpu()))
            run_manifest_hash = canonical_sha256(finished)
            checkpoint_manifest_hash = canonical_sha256(checkpoint)
            logical_run_manifest_hash = _logical_run_manifest_hash(finished)
            logical_checkpoint_manifest_hash = _logical_checkpoint_manifest_hash(checkpoint)
            reproducibility_hash = canonical_sha256(
                {
                    "logicalRunManifestHash": logical_run_manifest_hash,
                    "logicalCheckpointManifestHash": logical_checkpoint_manifest_hash,
                    "materializationHash": materialized.manifest["materializationHash"],
                    "outputDigest": verification["outputDigest"],
                }
            )
            results.append(
                {
                    "fixture": fixture_name,
                    "runId": run_id,
                    "status": "succeeded",
                    "loss": float(loss.detach().cpu()),
                    "materializationHash": materialized.manifest["materializationHash"],
                    "checkpoint": checkpoint.model_dump(mode="json", by_alias=True),
                    "freshProcessVerified": True,
                    "optimizerRestored": True,
                    "outputDigest": verification["outputDigest"],
                    "runManifestHash": run_manifest_hash,
                    "checkpointManifestHash": checkpoint_manifest_hash,
                    "logicalRunManifestHash": logical_run_manifest_hash,
                    "logicalCheckpointManifestHash": logical_checkpoint_manifest_hash,
                    "reproducibilityHash": reproducibility_hash,
                }
            )
        except RunCancelled as error:
            cancelled = SmokeTrainingRunManifest(
                **{
                    **running.model_dump(mode="python", by_alias=True),
                    "status": RunStatus.CANCELLED,
                    "finishedAt": datetime.now(UTC),
                }
            )
            registry.record_run(cancelled)
            context.log("run_cancelled", code=error.code, message=str(error))
            raise
        except BaseException as error:
            failed = SmokeTrainingRunManifest(
                **{
                    **running.model_dump(mode="python", by_alias=True),
                    "status": RunStatus.FAILED,
                    "finishedAt": datetime.now(UTC),
                    "failureCode": getattr(error, "code", type(error).__name__),
                }
            )
            registry.record_run(failed)
            context.log("run_failed", code=failed.failure_code, message=str(error))
            raise
    result = {
        "schemaVersion": SMOKE_SCHEMA_VERSION,
        "ok": True,
        "runs": results,
        "elapsedSeconds": round(time.perf_counter() - started_clock, 6),
        "maxMemoryMb": round(_peak_memory_mb(torch, device), 3),
    }
    result["manifestHash"] = canonical_sha256(result)
    return result


def verify_in_subprocess(*, manifest_path: Path, fixture: str, device: str) -> dict[str, Any]:
    environment = os.environ.copy()
    source_root = str(Path(__file__).resolve().parents[1])
    environment["PYTHONPATH"] = source_root + os.pathsep + environment.get("PYTHONPATH", "")
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "socialgraph_gfm.smoke_verify",
            "--manifest",
            str(manifest_path),
            "--fixture",
            fixture,
            "--device",
            device,
        ],
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=90,
        check=False,
        env=environment,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            "fresh-process checkpoint verification failed: "
            + (completed.stderr.strip() or completed.stdout.strip())
        )
    return json.loads(completed.stdout)


def verify_checkpoint_output(manifest_path: str | Path, fixture: str, device: str) -> dict[str, Any]:
    torch, _ = require_ml_runtime(device)
    manifest = read_manifest(manifest_path)
    from .checkpoint import load_checkpoint

    payload = load_checkpoint(manifest, map_location=device)
    config = payload["config"]
    snapshot = get_fixture(fixture)
    materialized = materialize(
        snapshot,
        purpose="training_smoke",
        fit_node_ids=smoke_fit_node_ids(fixture),
        device=device,
    )
    x, edge_index = homogeneous_tensors(materialized)
    x = x.to(device)
    edge_index = edge_index.to(device)
    model = _model(torch, int(config["inputDim"]), int(config["hiddenDim"])).to(device)
    model.load_state_dict(payload["model_state"])
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    optimizer.load_state_dict(payload["optimizer_state"])
    if not optimizer.state or int(payload.get("step", -1)) != manifest.step:
        raise RuntimeError("checkpoint optimizer state or step was not restored")
    model.eval()
    with torch.no_grad():
        output = model(x, edge_index)
    return {"ok": True, "outputDigest": _output_digest(output), "optimizerRestored": True}
