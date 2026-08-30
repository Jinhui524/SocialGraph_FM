"""Fail-closed path confinement using held, non-following OS handles."""

from __future__ import annotations

import os
import stat
from pathlib import Path, PurePosixPath


_REPARSE_POINT = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)


def _PATH_WALK_SEAM(_path: Path) -> None:
    return


def _is_link_or_reparse(path: Path) -> bool:
    details = path.lstat()
    return stat.S_ISLNK(details.st_mode) or bool(
        getattr(details, "st_file_attributes", 0) & _REPARSE_POINT
    )


def reject_link_components(path: str | Path) -> Path:
    """Inspect every existing lexical component before any resolving operation."""

    candidate = Path(os.path.abspath(os.path.expanduser(os.fspath(path))))
    current = Path(candidate.anchor)
    for part in candidate.parts[1:]:
        current /= part
        if current.exists() or current.is_symlink():
            if _is_link_or_reparse(current):
                raise ValueError("path contains a link or reparse component")
    return candidate


def secure_existing_root(path: str | Path) -> Path:
    lexical = reject_link_components(path)
    if not lexical.is_dir():
        raise ValueError("authorized root must be an existing directory")
    return lexical.resolve(strict=True)


def _relative_parts(relative_path: str) -> tuple[str, ...]:
    parsed = PurePosixPath(relative_path.replace("\\", "/"))
    if parsed.is_absolute() or not parsed.parts or ".." in parsed.parts or ":" in relative_path:
        raise ValueError("path must be a safe relative path")
    return parsed.parts


def secure_relative_file(root: Path, relative_path: str) -> Path:
    parts = _relative_parts(relative_path)
    lexical = reject_link_components(root.joinpath(*parts))
    if not lexical.is_file() or _is_link_or_reparse(lexical):
        raise ValueError("path must identify a regular non-link file")
    resolved = lexical.resolve(strict=True)
    try:
        resolved.relative_to(root)
    except ValueError as error:
        raise ValueError("path escapes authorized root") from error
    return resolved


def _bounded_read_descriptor(descriptor: int, *, size: int, max_bytes: int) -> bytes:
    if size < 1 or size > max_bytes:
        raise ValueError("file size is outside the authorized bound")
    chunks: list[bytes] = []
    remaining = size
    while remaining:
        chunk = os.read(descriptor, min(1024 * 1024, remaining))
        if not chunk:
            raise ValueError("file changed while reading immutable snapshot")
        chunks.append(chunk)
        remaining -= len(chunk)
    if os.read(descriptor, 1):
        raise ValueError("file grew while reading immutable snapshot")
    return b"".join(chunks)


def _read_posix_snapshot(root: Path, parts: tuple[str, ...], *, max_bytes: int) -> bytes:
    paths = [root]
    for index in range(1, len(parts) + 1):
        paths.append(root.joinpath(*parts[:index]))
    before = tuple((item.lstat().st_dev, item.lstat().st_ino) for item in paths)
    if any(_is_link_or_reparse(item) for item in paths):
        raise ValueError("path contains a link component")
    _PATH_WALK_SEAM(paths[-1])
    descriptors: list[int] = []
    no_follow = getattr(os, "O_NOFOLLOW", 0)
    directory_flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | no_follow
    try:
        descriptor = os.open(root, directory_flags)
        descriptors.append(descriptor)
        opened = os.fstat(descriptor)
        if (opened.st_dev, opened.st_ino) != before[0]:
            raise ValueError("authorized path component changed during handle walk")
        for index, part in enumerate(parts[:-1], start=1):
            descriptor = os.open(part, directory_flags, dir_fd=descriptors[-1])
            descriptors.append(descriptor)
            opened = os.fstat(descriptor)
            if (opened.st_dev, opened.st_ino) != before[index]:
                raise ValueError("authorized path component changed during handle walk")
        final = os.open(
            parts[-1],
            os.O_RDONLY | getattr(os, "O_BINARY", 0) | no_follow,
            dir_fd=descriptors[-1],
        )
        descriptors.append(final)
        opened = os.fstat(final)
        if (opened.st_dev, opened.st_ino) != before[-1] or not stat.S_ISREG(opened.st_mode):
            raise ValueError("authorized final file changed during handle walk")
        return _bounded_read_descriptor(final, size=opened.st_size, max_bytes=max_bytes)
    finally:
        for descriptor in reversed(descriptors):
            os.close(descriptor)


if os.name == "nt":  # pragma: win32 cover
    import ctypes
    from ctypes import wintypes

    class _BY_HANDLE_FILE_INFORMATION(ctypes.Structure):
        _fields_ = [
            ("dwFileAttributes", wintypes.DWORD),
            ("ftCreationTime", wintypes.FILETIME),
            ("ftLastAccessTime", wintypes.FILETIME),
            ("ftLastWriteTime", wintypes.FILETIME),
            ("dwVolumeSerialNumber", wintypes.DWORD),
            ("nFileSizeHigh", wintypes.DWORD),
            ("nFileSizeLow", wintypes.DWORD),
            ("nNumberOfLinks", wintypes.DWORD),
            ("nFileIndexHigh", wintypes.DWORD),
            ("nFileIndexLow", wintypes.DWORD),
        ]

    _kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    _CreateFileW = _kernel32.CreateFileW
    _CreateFileW.argtypes = [
        wintypes.LPCWSTR,
        wintypes.DWORD,
        wintypes.DWORD,
        wintypes.LPVOID,
        wintypes.DWORD,
        wintypes.DWORD,
        wintypes.HANDLE,
    ]
    _CreateFileW.restype = wintypes.HANDLE
    _GetFileInformationByHandle = _kernel32.GetFileInformationByHandle
    _GetFileInformationByHandle.argtypes = [
        wintypes.HANDLE,
        ctypes.POINTER(_BY_HANDLE_FILE_INFORMATION),
    ]
    _GetFileInformationByHandle.restype = wintypes.BOOL
    _GetFinalPathNameByHandleW = _kernel32.GetFinalPathNameByHandleW
    _GetFinalPathNameByHandleW.argtypes = [
        wintypes.HANDLE,
        wintypes.LPWSTR,
        wintypes.DWORD,
        wintypes.DWORD,
    ]
    _GetFinalPathNameByHandleW.restype = wintypes.DWORD
    _CloseHandle = _kernel32.CloseHandle
    _CloseHandle.argtypes = [wintypes.HANDLE]
    _CloseHandle.restype = wintypes.BOOL
    _ReadFile = _kernel32.ReadFile
    _ReadFile.argtypes = [
        wintypes.HANDLE,
        wintypes.LPVOID,
        wintypes.DWORD,
        ctypes.POINTER(wintypes.DWORD),
        wintypes.LPVOID,
    ]
    _ReadFile.restype = wintypes.BOOL
    _INVALID_HANDLE_VALUE = ctypes.c_void_p(-1).value
    _GENERIC_READ = 0x80000000
    _FILE_READ_ATTRIBUTES = 0x80
    _FILE_SHARE_READ = 0x1
    _FILE_SHARE_WRITE = 0x2
    _OPEN_EXISTING = 3
    _FILE_FLAG_OPEN_REPARSE_POINT = 0x00200000
    _FILE_FLAG_BACKUP_SEMANTICS = 0x02000000
    _FILE_ATTRIBUTE_DIRECTORY = 0x10
    _FILE_ATTRIBUTE_REPARSE_POINT = 0x400

    def _win_open(path: Path, *, final: bool) -> int:
        access = _FILE_READ_ATTRIBUTES | (_GENERIC_READ if final else 0)
        handle = _CreateFileW(
            str(path),
            access,
            _FILE_SHARE_READ | (_FILE_SHARE_WRITE if not final else 0),
            None,
            _OPEN_EXISTING,
            _FILE_FLAG_OPEN_REPARSE_POINT | _FILE_FLAG_BACKUP_SEMANTICS,
            None,
        )
        if handle == _INVALID_HANDLE_VALUE:
            raise OSError(ctypes.get_last_error(), "failed to open confined path handle")
        return handle

    def _win_info(handle: int) -> _BY_HANDLE_FILE_INFORMATION:
        details = _BY_HANDLE_FILE_INFORMATION()
        if not _GetFileInformationByHandle(handle, ctypes.byref(details)):
            raise OSError(ctypes.get_last_error(), "failed to inspect confined path handle")
        return details

    def _win_identity(handle: int) -> tuple[int, int]:
        details = _win_info(handle)
        return (
            details.dwVolumeSerialNumber,
            (details.nFileIndexHigh << 32) | details.nFileIndexLow,
        )

    def _win_identity_path(path: Path) -> tuple[int, int]:
        handle = _win_open(path, final=path.is_file())
        try:
            return _win_identity(handle)
        finally:
            _CloseHandle(handle)

    def _win_final_path(handle: int) -> Path:
        required = _GetFinalPathNameByHandleW(handle, None, 0, 0)
        if not required:
            raise OSError(ctypes.get_last_error(), "failed to size final handle path")
        buffer = ctypes.create_unicode_buffer(required + 1)
        if not _GetFinalPathNameByHandleW(handle, buffer, len(buffer), 0):
            raise OSError(ctypes.get_last_error(), "failed to resolve final handle path")
        value = buffer.value
        if value.startswith("\\\\?\\UNC\\"):
            value = "\\\\" + value[8:]
        elif value.startswith("\\\\?\\"):
            value = value[4:]
        return Path(value)

    def _read_windows_snapshot(root: Path, parts: tuple[str, ...], *, max_bytes: int) -> bytes:
        paths = [root]
        for index in range(1, len(parts) + 1):
            paths.append(root.joinpath(*parts[:index]))
        if any(_is_link_or_reparse(item) for item in paths):
            raise ValueError("path contains a link or reparse component")
        before = tuple(_win_identity_path(item) for item in paths)
        _PATH_WALK_SEAM(paths[-1])
        handles: list[int] = []
        try:
            for index, path in enumerate(paths):
                handle = _win_open(path, final=index == len(paths) - 1)
                handles.append(handle)
                details = _win_info(handle)
                if details.dwFileAttributes & _FILE_ATTRIBUTE_REPARSE_POINT:
                    raise ValueError("opened path is a reparse point")
                if _win_identity(handle) != before[index]:
                    raise ValueError("authorized path component changed during handle walk")
                if index < len(paths) - 1 and not (
                    details.dwFileAttributes & _FILE_ATTRIBUTE_DIRECTORY
                ):
                    raise ValueError("authorized parent component is not a directory")
            final_handle = handles[-1]
            details = _win_info(final_handle)
            if details.dwFileAttributes & _FILE_ATTRIBUTE_DIRECTORY:
                raise ValueError("authorized final path is not a regular file")
            expected = os.path.normcase(os.path.abspath(paths[-1]))
            observed = os.path.normcase(os.path.abspath(_win_final_path(final_handle)))
            trusted = os.path.normcase(os.path.abspath(root))
            if observed != expected or os.path.commonpath((observed, trusted)) != trusted:
                raise ValueError("final handle resolved outside the trusted root")
            size = (details.nFileSizeHigh << 32) | details.nFileSizeLow
            if size < 1 or size > max_bytes:
                raise ValueError("file size is outside the authorized bound")
            chunks: list[bytes] = []
            remaining = size
            while remaining:
                length = min(1024 * 1024, remaining)
                buffer = ctypes.create_string_buffer(length)
                read = wintypes.DWORD()
                if not _ReadFile(final_handle, buffer, length, ctypes.byref(read), None):
                    raise OSError(ctypes.get_last_error(), "failed to read confined snapshot")
                if read.value == 0:
                    raise ValueError("file changed while reading immutable snapshot")
                chunks.append(buffer.raw[: read.value])
                remaining -= read.value
            extra = ctypes.create_string_buffer(1)
            read = wintypes.DWORD()
            if not _ReadFile(final_handle, extra, 1, ctypes.byref(read), None):
                raise OSError(ctypes.get_last_error(), "failed to finish confined snapshot")
            if read.value:
                raise ValueError("file grew while reading immutable snapshot")
            return b"".join(chunks)
        finally:
            for handle in reversed(handles):
                _CloseHandle(handle)


def read_confined_snapshot(root: Path, relative_path: str, *, max_bytes: int) -> bytes:
    """Read bytes through a held-root component walk and an immutable final handle."""

    parts = _relative_parts(relative_path)
    lexical_root = secure_existing_root(root)
    final = lexical_root.joinpath(*parts)
    reject_link_components(final)
    if not final.is_file():
        raise ValueError("path must identify an existing regular file")
    if os.name == "nt":  # pragma: win32 cover
        return _read_windows_snapshot(lexical_root, parts, max_bytes=max_bytes)
    return _read_posix_snapshot(lexical_root, parts, max_bytes=max_bytes)


__all__ = [
    "read_confined_snapshot",
    "reject_link_components",
    "secure_existing_root",
    "secure_relative_file",
]
