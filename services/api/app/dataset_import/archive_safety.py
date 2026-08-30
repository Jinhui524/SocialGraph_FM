"""Bounded upload ingestion and in-memory archive safety checks."""

from __future__ import annotations

import io
import zipfile
from pathlib import PurePosixPath

from fastapi import HTTPException, UploadFile

from ..config import Settings

from .models import UploadedEntry

def _normalized_name(name: str) -> str:
    candidate = name.replace("\\", "/").strip()
    if "\x00" in candidate:
        raise HTTPException(status_code=400, detail="文件名包含非法空字符")
    path = PurePosixPath(candidate)
    if not candidate or path.is_absolute() or ".." in path.parts:
        raise HTTPException(status_code=400, detail="文件路径不安全")
    return path.as_posix()


def _is_zip(entry: UploadedEntry) -> bool:
    lowered = entry.name.casefold()
    # NPZ is itself a ZIP container and must stay intact for allow_pickle=False
    # validation. Only generic PK data without a known NPZ suffix is expanded.
    return lowered.endswith(".zip") or (
        not lowered.endswith(".npz") and entry.data.startswith(b"PK\x03\x04")
    )


def _validate_zip_member(name: str) -> str:
    normalized = _normalized_name(name)
    if normalized.endswith("/"):
        return normalized
    return normalized


def _expand_zip(entry: UploadedEntry, settings: Settings) -> list[UploadedEntry]:
    result: list[UploadedEntry] = []
    try:
        archive = zipfile.ZipFile(io.BytesIO(entry.data))
    except zipfile.BadZipFile as exc:
        raise HTTPException(status_code=400, detail="ZIP 文件损坏或格式无效") from exc

    with archive:
        members = archive.infolist()
        if len(members) > settings.dataset_archive_max_files:
            raise HTTPException(status_code=413, detail="ZIP 内文件数量超过限制")
        total_size = 0
        for member in members:
            name = _validate_zip_member(member.filename)
            if member.is_dir():
                continue
            # Unix symlinks can escape the archive root after extraction. We never
            # extract to disk, but rejecting them keeps the contract unambiguous.
            if (member.external_attr >> 16) & 0o170000 == 0o120000:
                raise HTTPException(status_code=400, detail="ZIP 不允许包含符号链接")
            total_size += member.file_size
            if member.file_size > settings.dataset_upload_max_bytes:
                raise HTTPException(status_code=413, detail=f"ZIP 成员过大: {name}")
            if total_size > settings.dataset_archive_max_bytes:
                raise HTTPException(status_code=413, detail="ZIP 解压后总大小超过限制")
            if member.compress_size and member.file_size / member.compress_size > 100:
                raise HTTPException(status_code=413, detail=f"ZIP 压缩比异常: {name}")
            data = archive.read(member)
            result.append(UploadedEntry(name=name, data=data))
    return result


async def read_uploads(files: list[UploadFile], settings: Settings) -> list[UploadedEntry]:
    if not files:
        raise HTTPException(status_code=400, detail="至少上传一个数据文件")
    if len(files) > settings.dataset_archive_max_files:
        raise HTTPException(status_code=413, detail="上传文件数量超过限制")

    raw_entries: list[UploadedEntry] = []
    total = 0
    for upload in files:
        name = _normalized_name(upload.filename or "unnamed")
        data = await upload.read(settings.dataset_upload_max_bytes + 1)
        await upload.close()
        if len(data) > settings.dataset_upload_max_bytes:
            raise HTTPException(status_code=413, detail=f"文件超过单文件限制: {name}")
        total += len(data)
        if total > settings.dataset_archive_max_bytes:
            raise HTTPException(status_code=413, detail="上传总大小超过限制")
        raw_entries.append(UploadedEntry(name=name, data=data))

    expanded: list[UploadedEntry] = []
    for entry in raw_entries:
        if _is_zip(entry):
            expanded.extend(_expand_zip(entry, settings))
        else:
            expanded.append(entry)

    names: set[str] = set()
    for entry in expanded:
        key = entry.name.casefold()
        if key in names:
            raise HTTPException(status_code=400, detail=f"存在重复文件路径: {entry.name}")
        names.add(key)
    if not expanded:
        raise HTTPException(status_code=400, detail="压缩包中没有可读取文件")
    return expanded


def _entry_map(entries: list[UploadedEntry]) -> dict[str, UploadedEntry]:
    return {entry.name.casefold(): entry for entry in entries}


def _find_suffix(entries: dict[str, UploadedEntry], suffix: str) -> UploadedEntry | None:
    lowered = suffix.casefold()
    return next((entry for key, entry in entries.items() if key.endswith(lowered)), None)
