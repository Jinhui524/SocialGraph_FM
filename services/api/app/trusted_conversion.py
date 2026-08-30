from __future__ import annotations

import asyncio
import ctypes
import hashlib
import ipaddress
import json
import os
import secrets
import shutil
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from fastapi import HTTPException, status

from .config import Settings
from .dataset_imports import DatasetImportService
from .dataset_schemas import (
    DatasetIssue,
    TrustedConversionJob,
    TrustedDiscoveredDataset,
    TrustedLocalInspection,
)


def is_loopback_host(host: str | None) -> bool:
    if not host:
        return False
    if host.casefold() == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def _is_reparse_point(path: Path) -> bool:
    try:
        if path.is_symlink():
            return True
        if os.name != "nt":
            return False
        attributes = getattr(path.stat(follow_symlinks=False), "st_file_attributes", 0)
        return bool(attributes & 0x400)  # FILE_ATTRIBUTE_REPARSE_POINT
    except OSError:
        return True


def _inside(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def _process_memory_bytes(pid: int) -> int | None:
    if os.name != "nt":
        try:
            status_text = Path(f"/proc/{pid}/status").read_text(encoding="utf-8")
            line = next(item for item in status_text.splitlines() if item.startswith("VmRSS:"))
            return int(line.split()[1]) * 1024
        except (OSError, StopIteration, ValueError):
            return None
    try:
        class ProcessMemoryCounters(ctypes.Structure):
            _fields_ = [
                ("cb", ctypes.c_ulong),
                ("PageFaultCount", ctypes.c_ulong),
                ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t),
            ]

        kernel32 = ctypes.windll.kernel32
        psapi = ctypes.windll.psapi
        kernel32.OpenProcess.argtypes = [ctypes.c_ulong, ctypes.c_int, ctypes.c_ulong]
        kernel32.OpenProcess.restype = ctypes.c_void_p
        kernel32.CloseHandle.argtypes = [ctypes.c_void_p]
        psapi.GetProcessMemoryInfo.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_ulong,
        ]
        handle = kernel32.OpenProcess(0x1000 | 0x0010, False, pid)
        if not handle:
            return None
        try:
            counters = ProcessMemoryCounters()
            counters.cb = ctypes.sizeof(counters)
            if not psapi.GetProcessMemoryInfo(
                handle,
                ctypes.byref(counters),
                counters.cb,
            ):
                return None
            return int(counters.WorkingSetSize)
        finally:
            kernel32.CloseHandle(handle)
    except (AttributeError, OSError):
        return None


class TrustedConversionService:
    """Orchestrates explicitly trusted local conversion in a separate process.

    Process separation limits accidental credential/file access. It is not an
    adversarial pickle sandbox and therefore remains loopback + allow-list only.
    """

    def __init__(self, settings: Settings, imports: DatasetImportService) -> None:
        self.settings = settings
        self.imports = imports
        self.store = imports.store
        self.store.mark_interrupted_jobs()
        self._tasks: dict[str, asyncio.Task[None]] = {}
        self._processes: dict[str, asyncio.subprocess.Process] = {}
        self._cancelled: set[str] = set()

    def _ensure_available(self, client_host: str | None) -> None:
        if not is_loopback_host(client_host):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail={"code": "TRUSTED_CONVERSION_LOOPBACK_ONLY"},
            )
        if not self.settings.enable_trusted_local_conversion:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail={"code": "TRUSTED_CONVERSION_DISABLED"},
            )
        if not self.settings.trusted_roots:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail={"code": "TRUSTED_ROOTS_NOT_CONFIGURED"},
            )

    def _resolve_source(self, raw_path: str) -> tuple[Path, Path]:
        try:
            requested = Path(raw_path).expanduser()
            resolved = requested.resolve(strict=True)
        except (OSError, RuntimeError) as exc:
            raise HTTPException(status_code=404, detail="本地数据目录不存在") from exc
        if not resolved.is_dir():
            raise HTTPException(status_code=400, detail="sourcePath 必须是本地目录")
        roots: list[Path] = []
        for configured in self.settings.trusted_roots:
            try:
                roots.append(configured.resolve(strict=True))
            except OSError:
                continue
        trusted_root = next((root for root in roots if _inside(resolved, root)), None)
        if trusted_root is None:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail={"code": "SOURCE_OUTSIDE_TRUSTED_ROOTS"},
            )
        if _is_reparse_point(trusted_root) or _is_reparse_point(resolved):
            raise HTTPException(
                status_code=400,
                detail={"code": "REPARSE_POINT_NOT_ALLOWED", "path": str(resolved)},
            )
        current = resolved
        while current != trusted_root:
            if _is_reparse_point(current):
                raise HTTPException(
                    status_code=400,
                    detail={"code": "REPARSE_POINT_NOT_ALLOWED", "path": str(current)},
                )
            current = current.parent
        return resolved, trusted_root

    def _scan(self, source: Path) -> tuple[int, int, list[TrustedDiscoveredDataset]]:
        file_count = 0
        total_bytes = 0
        per_dataset: dict[str, dict[str, Any]] = {}
        for root_text, directories, files in os.walk(source, followlinks=False):
            root = Path(root_text)
            for directory in list(directories):
                candidate = root / directory
                if _is_reparse_point(candidate):
                    raise HTTPException(
                        status_code=400,
                        detail={"code": "REPARSE_POINT_NOT_ALLOWED", "path": str(candidate)},
                    )
            for name in files:
                path = root / name
                if _is_reparse_point(path):
                    raise HTTPException(
                        status_code=400,
                        detail={"code": "REPARSE_POINT_NOT_ALLOWED", "path": str(path)},
                    )
                try:
                    size = path.stat().st_size
                except OSError as exc:
                    raise HTTPException(status_code=400, detail=f"无法检查文件: {path}") from exc
                file_count += 1
                total_bytes += size
                if file_count > self.settings.trusted_conversion_max_files:
                    raise HTTPException(status_code=413, detail="可信转换源文件数量超过限制")
                if total_bytes > self.settings.trusted_conversion_max_source_bytes:
                    raise HTTPException(status_code=413, detail="可信转换源目录大小超过限制")
                relative = path.relative_to(source)
                dataset = relative.parts[0] if len(relative.parts) > 1 else source.name
                record = per_dataset.setdefault(dataset, {"files": 0, "suffixes": set()})
                record["files"] += 1
                record["suffixes"].add(path.suffix.casefold())

        discovered: list[TrustedDiscoveredDataset] = []
        for name, record in sorted(per_dataset.items()):
            suffixes = record["suffixes"]
            lowered = name.casefold()
            if lowered == "fewshot_cora":
                detected = "fewshot_torch_episodes"
            elif lowered in {"ogbl-collab", "ogbl_collab"}:
                detected = "ogb_link_prediction_cache"
            elif ".txt" in suffixes:
                detected = "geom_gcn_or_text"
            elif ".pt" in suffixes or ".pth" in suffixes:
                detected = "torch_pyg_archive"
            else:
                detected = "legacy_planetoid_or_numeric"
            discovered.append(
                TrustedDiscoveredDataset(
                    name=name,
                    detectedFormat=detected,
                    fileCount=record["files"],
                )
            )
        return file_count, total_bytes, discovered

    def inspect_local(self, raw_path: str, *, client_host: str | None) -> TrustedLocalInspection:
        self._ensure_available(client_host)
        source, trusted_root = self._resolve_source(raw_path)
        file_count, total_bytes, datasets = self._scan(source)
        token = secrets.token_urlsafe(32)
        token_hash = hashlib.sha256(token.encode("utf-8")).hexdigest()
        now = datetime.now(UTC)
        converter = self.settings.trusted_converter_python or sys.executable
        job = TrustedConversionJob(
            id=str(uuid4()),
            sourcePath=str(source),
            trustedRoot=str(trusted_root),
            status="awaiting_authorization",
            progress=0,
            fileCount=file_count,
            totalBytes=total_bytes,
            datasets=datasets,
            artifactIds=[],
            issues=[
                DatasetIssue(
                    severity="warning",
                    code="TRUSTED_PICKLE_EXECUTION",
                    message=(
                        "该作业会在独立进程中读取已授权的 PT/Pickle；进程隔离不是恶意文件沙箱。"
                    ),
                )
            ],
            converterPython=converter,
            createdAt=now,
            updatedAt=now,
        )
        self.store.create_job(job, token_hash)
        return TrustedLocalInspection(
            **job.model_dump(by_alias=True),
            authorizationToken=token,
        )

    def get_job(self, job_id: str, *, client_host: str | None) -> TrustedConversionJob:
        self._ensure_available(client_host)
        stored = self.store.get_job(job_id)
        if stored is None:
            raise HTTPException(status_code=404, detail="转换作业不存在")
        return stored[0]

    async def authorize(
        self,
        job_id: str,
        authorization_token: str,
        *,
        client_host: str | None,
    ) -> TrustedConversionJob:
        self._ensure_available(client_host)
        stored = self.store.get_job(job_id)
        if stored is None:
            raise HTTPException(status_code=404, detail="转换作业不存在")
        job, expected_hash = stored
        supplied_hash = hashlib.sha256(authorization_token.encode("utf-8")).hexdigest()
        if job.status != "awaiting_authorization" or not expected_hash:
            raise HTTPException(status_code=409, detail={"code": "AUTHORIZATION_ALREADY_USED"})
        if not secrets.compare_digest(expected_hash, supplied_hash):
            raise HTTPException(status_code=403, detail={"code": "INVALID_AUTHORIZATION_TOKEN"})
        job.status = "queued"
        job.progress = 1
        job.updated_at = datetime.now(UTC)
        self.store.update_job(job, clear_authorization=True)
        task = asyncio.create_task(self._run(job.id), name=f"trusted-conversion-{job.id}")
        self._tasks[job.id] = task

        def discard_task(_task: asyncio.Task[None], key: str = job.id) -> None:
            self._tasks.pop(key, None)

        task.add_done_callback(discard_task)
        return job

    async def cancel(self, job_id: str, *, client_host: str | None) -> TrustedConversionJob:
        self._ensure_available(client_host)
        stored = self.store.get_job(job_id)
        if stored is None:
            raise HTTPException(status_code=404, detail="转换作业不存在")
        job = stored[0]
        if job.status in {"succeeded", "failed", "cancelled"}:
            return job
        self._cancelled.add(job_id)
        process = self._processes.get(job_id)
        if process is not None and process.returncode is None:
            await self._kill_process_tree(process)
        job.status = "cancelled"
        job.updated_at = datetime.now(UTC)
        job.issues.append(
            DatasetIssue(severity="warning", code="CONVERSION_CANCELLED", message="转换已取消。")
        )
        self.store.update_job(job, clear_authorization=True)
        return job

    def _minimal_environment(self, working_directory: Path) -> dict[str, str]:
        environment: dict[str, str] = {
            "PYTHONIOENCODING": "utf-8",
            "PYTHONUNBUFFERED": "1",
            "PYTHONPATH": str(Path(__file__).resolve().parent.parent),
            "SGFM_CONVERTER_MEMORY_LIMIT_MB": str(self.settings.trusted_conversion_memory_mb),
        }
        for name in ("PATH", "SYSTEMROOT", "WINDIR"):
            if value := os.environ.get(name):
                environment[name] = value
        environment["TEMP"] = str(working_directory)
        environment["TMP"] = str(working_directory)
        return environment

    async def _run(self, job_id: str) -> None:
        stored = self.store.get_job(job_id)
        if stored is None:
            return
        job = stored[0]
        workspace = self.store.jobs_root / job_id
        workspace.mkdir(parents=True, exist_ok=True)
        output = workspace / "converted.sgfm.zip"
        stdout_path = workspace / "converter.stdout.log"
        stderr_path = workspace / "converter.stderr.log"
        converter = Path(job.converter_python)
        if not converter.exists() and shutil.which(job.converter_python) is None:
            self._fail(job, "CONVERTER_PYTHON_NOT_FOUND", f"转换器 Python 不存在: {converter}")
            return
        job.status = "running"
        job.progress = 5
        job.updated_at = datetime.now(UTC)
        self.store.update_job(job)
        use_ogbl_collab = any(
            item.detected_format == "ogb_link_prediction_cache" for item in job.datasets
        ) or Path(job.source_path).name.casefold() in {"ogbl-collab", "ogbl_collab"}
        command = [
            job.converter_python,
            "-m",
            "app.dataset_tools",
            "convert-ogbl-collab" if use_ogbl_collab else "convert-pyg",
            "--input",
            job.source_path,
            "--output",
            str(output),
            "--trust-pickle",
        ]
        creationflags = 0
        start_new_session = False
        if os.name == "nt":
            creationflags = getattr(__import__("subprocess"), "CREATE_NEW_PROCESS_GROUP", 0)
        else:
            start_new_session = True
        try:
            with stdout_path.open("wb") as stdout, stderr_path.open("wb") as stderr:
                process = await asyncio.create_subprocess_exec(
                    *command,
                    cwd=workspace,
                    env=self._minimal_environment(workspace),
                    stdin=asyncio.subprocess.DEVNULL,
                    stdout=stdout,
                    stderr=stderr,
                    creationflags=creationflags,
                    start_new_session=start_new_session,
                )
                self._processes[job_id] = process
                started = time.monotonic()
                while process.returncode is None:
                    if job_id in self._cancelled:
                        await self._kill_process_tree(process)
                        return
                    if time.monotonic() - started > self.settings.trusted_conversion_timeout_seconds:
                        await self._kill_process_tree(process)
                        self._fail(job, "CONVERTER_TIMEOUT", "转换超时，进程树已终止。")
                        return
                    memory = _process_memory_bytes(process.pid)
                    if memory and memory > self.settings.trusted_conversion_memory_mb * 1024 * 1024:
                        await self._kill_process_tree(process)
                        self._fail(job, "CONVERTER_MEMORY_LIMIT", "转换器超过内存限制。")
                        return
                    if output.exists() and output.stat().st_size > self.settings.trusted_conversion_max_output_bytes:
                        await self._kill_process_tree(process)
                        self._fail(job, "CONVERTER_OUTPUT_LIMIT", "转换产物超过大小限制。")
                        return
                    try:
                        await asyncio.wait_for(process.wait(), timeout=0.2)
                    except TimeoutError:
                        pass
                return_code = await process.wait()
        except (OSError, ValueError) as exc:
            self._fail(job, "CONVERTER_START_FAILED", str(exc))
            return
        finally:
            self._processes.pop(job_id, None)
        if job_id in self._cancelled:
            return
        if return_code != 0:
            message = stderr_path.read_text(encoding="utf-8", errors="replace")[-4000:]
            hint = "请为 TRUSTED_CONVERTER_PYTHON 配置含 PyTorch/SciPy 的研究环境。"
            self._fail(job, "CONVERTER_DEPENDENCY_OR_DATA_ERROR", f"{message}\n{hint}".strip())
            return
        if not output.is_file():
            self._fail(job, "CONVERTER_OUTPUT_MISSING", "转换器没有生成 SGFM 产物。")
            return
        if output.stat().st_size > self.settings.trusted_conversion_max_output_bytes:
            self._fail(job, "CONVERTER_OUTPUT_LIMIT", "转换产物超过大小限制。")
            return
        try:
            import zipfile

            with zipfile.ZipFile(output) as archive:
                package_manifest = json.loads(archive.read("manifest.json"))
            skipped = package_manifest.get("skipped", [])
            if skipped:
                reasons = "; ".join(
                    str(item.get("reason", "unknown"))[:200]
                    for item in skipped[:5]
                    if isinstance(item, dict)
                )
                job.issues.append(
                    DatasetIssue(
                        severity="warning",
                        code="CONVERTER_PARTIAL_RESULT",
                        message=f"部分数据未转换：{reasons}",
                    )
                )
        except (OSError, KeyError, ValueError, zipfile.BadZipFile):
            # The safe importer below performs the authoritative package validation.
            pass
        job.progress = 80
        job.updated_at = datetime.now(UTC)
        self.store.update_job(job)
        try:
            artifacts = await asyncio.to_thread(
                self.imports.import_trusted_package,
                str(output),
                job_id=job.id,
                source_path=job.source_path,
            )
        except Exception as exc:  # noqa: BLE001 - convert into durable job diagnostic
            self._fail(job, "ARTIFACT_COMMIT_FAILED", str(exc))
            return
        if job_id in self._cancelled:
            return
        job.status = "succeeded"
        job.progress = 100
        job.artifact_ids = [artifact.id for artifact in artifacts]
        job.updated_at = datetime.now(UTC)
        self.store.update_job(job)

    def _fail(self, job: TrustedConversionJob, code: str, message: str) -> None:
        job.status = "failed"
        job.updated_at = datetime.now(UTC)
        job.issues.append(
            DatasetIssue(severity="error", code=code, message=message[:5000])
        )
        self.store.update_job(job, clear_authorization=True)

    async def _kill_process_tree(self, process: asyncio.subprocess.Process) -> None:
        if process.returncode is not None:
            return
        if os.name == "nt":
            try:
                killer = await asyncio.create_subprocess_exec(
                    "taskkill",
                    "/PID",
                    str(process.pid),
                    "/T",
                    "/F",
                    stdout=asyncio.subprocess.DEVNULL,
                    stderr=asyncio.subprocess.DEVNULL,
                )
                await asyncio.wait_for(killer.wait(), timeout=5)
            except (OSError, TimeoutError):
                process.kill()
        else:
            try:
                os.killpg(process.pid, 15)  # type: ignore[attr-defined]
                await asyncio.wait_for(process.wait(), timeout=3)
            except (OSError, TimeoutError):
                process.kill()
        try:
            await asyncio.wait_for(process.wait(), timeout=5)
        except TimeoutError:
            pass

    async def close(self) -> None:
        for process in list(self._processes.values()):
            await self._kill_process_tree(process)
        for task in list(self._tasks.values()):
            task.cancel()
        if self._tasks:
            await asyncio.gather(*self._tasks.values(), return_exceptions=True)
