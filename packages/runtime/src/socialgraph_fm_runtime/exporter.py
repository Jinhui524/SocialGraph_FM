"""Fail-closed export of the public repository and its Download ZIP snapshot."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import stat
import subprocess
import sys
import tempfile
import unicodedata
import zipfile
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath


NEUTRAL_AUTHOR_NAME = "SocialGraph-FM Contributors"
NEUTRAL_AUTHOR_EMAIL = "socialgraph-fm@users.noreply.github.com"
DEFAULT_COMMIT_MESSAGE = "Initial public SocialGraph-FM complete runtime snapshot"
_REPOSITORY_STAGING_PREFIX = ".socialgraph-fm-export-repository-"
_TEMPLATE_STAGING_PREFIX = ".socialgraph-fm-export-template-"
_ZIP_STAGING_PREFIX = ".socialgraph-fm-export-archive-"
_WINDOWS_RESERVED_NAMES = {
    "CON",
    "PRN",
    "AUX",
    "NUL",
    *(f"COM{index}" for index in range(1, 10)),
    *(f"LPT{index}" for index in range(1, 10)),
}
_MAX_SCAN_FILE_BYTES = 50 * 1024 * 1024
_MAX_EXPORT_ZIP_BYTES = 30 * 1024 * 1024
_MAX_ARCHIVE_MEMBERS = 100
_MAX_ARCHIVE_EXPANDED_BYTES = 20 * 1024 * 1024
_RUNTIME_BINARY_EXTENSIONS = {
    ".pt",
    ".pth",
    ".ckpt",
    ".safetensors",
    ".onnx",
    ".pkl",
    ".pickle",
    ".npy",
    ".npz",
    ".joblib",
    ".h5",
    ".hdf5",
    ".pb",
    ".parquet",
    ".sqlite",
    ".sqlite3",
    ".tar",
    ".tgz",
    ".7z",
}
_ARCHIVE_EXTENSIONS = {".zip", ".npz", ".pt"}
_VISUAL_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".woff2"}
_TEXT_EXTENSIONS = {
    ".cff",
    ".css",
    ".csv",
    ".example",
    ".gexf",
    ".graphml",
    ".html",
    ".js",
    ".json",
    ".lock",
    ".md",
    ".mjs",
    ".ps1",
    ".py",
    ".toml",
    ".ts",
    ".tsv",
    ".tsx",
    ".txt",
    ".yaml",
    ".yml",
}
_SECRET_PATTERNS = tuple(
    re.compile(pattern)
    for pattern in (
        rb"(?<![A-Za-z0-9_-])sk-[A-Za-z0-9_-]{20,}",
        rb"(?<![A-Za-z0-9_])gh[oprsu]_[A-Za-z0-9]{30,}",
        rb"(?<![A-Za-z0-9_])github_pat_[A-Za-z0-9_]{40,}",
        rb"(?<![A-Z0-9])AKIA[A-Z0-9]{16}(?![A-Z0-9])",
        rb"(?<![A-Za-z0-9_-])AIza[A-Za-z0-9_-]{35}(?![A-Za-z0-9_-])",
        rb"(?<![A-Za-z0-9_-])xox[baprs]-[A-Za-z0-9-]{20,}",
        rb"(?<![A-Za-z0-9_-])eyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}",
        rb"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----",
        rb"(?i)authorization\s*[:=]\s*bearer\s+[A-Za-z0-9._~+/-]{20,}",
        rb"(?i)(?:api[_-]?key|secret|authorization|password|token)\s*[:=]\s*[\"'](?![^\"']*(?:test|fixture|example|with-at-least))[A-Za-z0-9_-]{24,}[\"']",
        rb"(?im)^\s*[A-Z0-9_]*(?:API_KEY|SECRET|PASSWORD|TOKEN)\s*=\s*(?!test|fixture|example|replace|dummy|hidden)[A-Za-z0-9_./+=-]{16,}\s*$",
    )
)
_PERSONAL_PATH_PATTERNS = tuple(
    re.compile(pattern)
    for pattern in (
        rb"(?i)[A-Z]:[\\/](?:Users|Documents and Settings)[\\/]",
        rb"(?i)[A-Z]:[\\/]project[\\/]SocialGraph_FM(?:-public)?",
        rb"(?i)/(?:Users|home)/[A-Za-z0-9._-]+/",
    )
)


class PublicExportError(RuntimeError):
    """A fail-closed public export validation or publication failure."""


@dataclass(frozen=True, slots=True)
class ExportResult:
    """Identity and location of one verified public export."""

    source_head: str
    source_tree: str
    repository_commit: str
    repository_destination: Path
    zip_destination: Path
    zip_sha256: str
    zip_bytes: int
    tracked_file_count: int

    def to_document(self) -> dict[str, object]:
        return {
            "schemaVersion": "socialgraph-fm.public-export/1.0",
            "sourceHead": self.source_head,
            "sourceTree": self.source_tree,
            "repositoryCommit": self.repository_commit,
            "repositoryDestination": str(self.repository_destination),
            "zipDestination": str(self.zip_destination),
            "zipSha256": self.zip_sha256,
            "zipBytes": self.zip_bytes,
            "trackedFileCount": self.tracked_file_count,
        }


@dataclass(frozen=True, slots=True)
class _TreeEntry:
    mode: str
    object_id: str
    size: int
    path: str


ScanRunner = Callable[[Path], None]


def _git_environment() -> dict[str, str]:
    environment = dict(os.environ)
    for name in (
        "GIT_DIR",
        "GIT_WORK_TREE",
        "GIT_INDEX_FILE",
        "GIT_OBJECT_DIRECTORY",
        "GIT_ALTERNATE_OBJECT_DIRECTORIES",
    ):
        environment.pop(name, None)
    environment.update(
        {
            "GIT_CONFIG_NOSYSTEM": "1",
            "GIT_CONFIG_GLOBAL": os.devnull,
            "GIT_TERMINAL_PROMPT": "0",
            "LC_ALL": "C",
            "LANG": "C",
        }
    )
    return environment


def _run(
    command: Sequence[str],
    *,
    cwd: Path,
    timeout: float = 900,
    environment: dict[str, str] | None = None,
    input_data: bytes | None = None,
) -> subprocess.CompletedProcess[bytes]:
    try:
        completed = subprocess.run(
            list(command),
            cwd=cwd,
            env=environment,
            stdin=subprocess.DEVNULL if input_data is None else None,
            input=input_data,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            timeout=timeout,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise PublicExportError(f"Could not run {command[0]}: {exc}") from None
    if completed.returncode != 0:
        detail = completed.stderr.decode("utf-8", "replace").strip()
        if not detail:
            detail = completed.stdout.decode("utf-8", "replace").strip()
        raise PublicExportError(
            f"Command failed ({completed.returncode}): {' '.join(command)}"
            + (f"\n{detail}" if detail else "")
        )
    return completed


def _git(repository: Path, *arguments: str, timeout: float = 900) -> bytes:
    executable = shutil.which("git")
    if executable is None:
        raise PublicExportError("Git is required to export the public repository")
    return _run(
        [executable, "-C", str(repository), *arguments],
        cwd=repository,
        timeout=timeout,
        environment=_git_environment(),
    ).stdout


def _has_link_component(path: Path) -> bool:
    selected = path
    while True:
        if os.path.lexists(selected):
            if selected.is_symlink():
                return True
            is_junction = getattr(selected, "is_junction", None)
            if callable(is_junction) and is_junction():
                return True
        if selected.parent == selected:
            return False
        selected = selected.parent


def _absolute_path(value: Path, *, label: str) -> Path:
    text = os.fspath(value)
    if not text or any(ord(character) < 32 for character in text):
        raise PublicExportError(f"{label} is empty or contains control characters")
    return Path(os.path.abspath(text))


def _validate_destination(
    value: Path,
    *,
    label: str,
    source: Path,
    require_zip: bool,
) -> Path:
    raw = _absolute_path(value, label=label)
    if raw == Path(raw.anchor) or raw.name in {"", ".", ".."}:
        raise PublicExportError(f"{label} is a dangerous filesystem target: {raw}")
    if require_zip and raw.suffix.casefold() != ".zip":
        raise PublicExportError(f"{label} must end in .zip: {raw}")
    if os.path.lexists(raw):
        raise PublicExportError(f"{label} already exists: {raw}")
    parent = raw.parent
    if not parent.is_dir():
        raise PublicExportError(f"{label} parent directory does not exist: {parent}")
    if _has_link_component(parent):
        raise PublicExportError(f"{label} parent cannot contain a link or junction: {parent}")
    resolved = raw.resolve(strict=False)
    if resolved == source or resolved.is_relative_to(source):
        raise PublicExportError(f"{label} must be outside the source repository")
    return resolved


def _validate_targets(source: Path, repository: Path, archive: Path) -> tuple[Path, Path]:
    repository = _validate_destination(
        repository,
        label="Repository destination",
        source=source,
        require_zip=False,
    )
    archive = _validate_destination(
        archive,
        label="ZIP destination",
        source=source,
        require_zip=True,
    )
    if repository == archive:
        raise PublicExportError("Repository and ZIP destinations must differ")
    if archive.is_relative_to(repository) or repository.is_relative_to(archive):
        raise PublicExportError("Repository and ZIP destinations cannot contain one another")
    return repository, archive


def _validate_source(value: Path) -> tuple[Path, str, str]:
    raw_source = _absolute_path(value, label="Source repository")
    try:
        source = raw_source.resolve(strict=True)
    except OSError as exc:
        raise PublicExportError(f"Source repository does not exist: {raw_source}: {exc}") from None
    if not source.is_dir() or _has_link_component(source):
        raise PublicExportError("Source repository must be a real directory without links")
    root = Path(
        _git(source, "rev-parse", "--show-toplevel").decode("utf-8", "strict").strip()
    ).resolve(strict=True)
    if root != source:
        raise PublicExportError(f"Source must name the Git worktree root: {root}")
    status = _git(source, "status", "--porcelain=v1", "--untracked-files=all")
    if status:
        raise PublicExportError(
            "Source repository must be clean, including non-ignored untracked files"
        )
    head = _git(source, "rev-parse", "--verify", "HEAD").decode("ascii").strip()
    tree = _git(source, "rev-parse", "--verify", "HEAD^{tree}").decode("ascii").strip()
    return source, head, tree


def _portable_path(value: str) -> str:
    if (
        not value
        or value.startswith("/")
        or "\\" in value
        or ":" in value
        or any(ord(character) < 32 or ord(character) == 127 for character in value)
    ):
        raise PublicExportError(f"HEAD contains an unsafe path: {value!r}")
    selected = PurePosixPath(value)
    if selected.as_posix() != value or any(part in {"", ".", ".."} for part in selected.parts):
        raise PublicExportError(f"HEAD contains an unsafe path: {value!r}")
    for part in selected.parts:
        if part.casefold() == ".git":
            raise PublicExportError(f"HEAD contains a forbidden .git path: {value}")
        if part.rstrip(" .") != part:
            raise PublicExportError(f"HEAD contains a non-portable path segment: {value}")
        if part.split(".", 1)[0].upper() in _WINDOWS_RESERVED_NAMES:
            raise PublicExportError(f"HEAD contains a Windows-reserved path: {value}")
    return value


def _head_tree(source: Path) -> tuple[_TreeEntry, ...]:
    raw = _git(source, "ls-tree", "-r", "-z", "--full-tree", "-l", "HEAD")
    entries: list[_TreeEntry] = []
    portable_names: dict[str, str] = {}
    for encoded in raw.split(b"\0"):
        if not encoded:
            continue
        try:
            metadata, encoded_path = encoded.split(b"\t", 1)
            mode, kind, object_id, encoded_size = metadata.split(b" ", 3)
            path = _portable_path(encoded_path.decode("utf-8", "strict"))
        except (UnicodeDecodeError, ValueError):
            raise PublicExportError("HEAD tree contains malformed or non-UTF-8 metadata") from None
        if kind != b"blob" or mode not in {b"100644", b"100755"}:
            raise PublicExportError(
                f"HEAD may contain only regular tracked files: {path} ({mode.decode()})"
            )
        normalized = unicodedata.normalize("NFC", path).casefold()
        previous = portable_names.setdefault(normalized, path)
        if previous != path:
            raise PublicExportError(
                f"HEAD contains cross-platform path aliases: {previous!r}, {path!r}"
            )
        entries.append(
            _TreeEntry(
                mode=mode.decode("ascii"),
                object_id=object_id.decode("ascii"),
                size=int(encoded_size),
                path=path,
            )
        )
    if not entries:
        raise PublicExportError("HEAD contains no tracked files")
    return tuple(entries)


def _json_object(path: Path, *, label: str) -> dict[str, object]:
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PublicExportError(f"{label} is unreadable or invalid JSON: {path}: {exc}") from None
    if not isinstance(document, dict):
        raise PublicExportError(f"{label} must be a JSON object: {path}")
    return document


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _is_runtime_binary(path: str) -> bool:
    lowered = path.casefold()
    return (
        PurePosixPath(lowered).suffix in _RUNTIME_BINARY_EXTENSIONS
        or lowered.endswith((".tar.gz", ".tar.bz2", ".tar.xz"))
    )


def _runtime_manifest_assets(
    repository: Path, tracked: set[str]
) -> dict[str, dict[str, object]]:
    manifest_path = repository / "bundles" / "runtime-manifest.json"
    if "bundles/runtime-manifest.json" not in tracked:
        raise PublicExportError("Runtime bundle manifest is not tracked")
    manifest = _json_object(manifest_path, label="Runtime bundle manifest")
    if manifest.get("schemaVersion") != "socialgraph-fm.runtime-bundle/1.0":
        raise PublicExportError("Runtime bundle manifest schema is unsupported")
    raw_assets = manifest.get("assets")
    if not isinstance(raw_assets, list):
        raise PublicExportError("Runtime bundle manifest assets must be an array")
    assets: dict[str, dict[str, object]] = {}
    for raw_asset in raw_assets:
        if not isinstance(raw_asset, dict):
            raise PublicExportError("Runtime bundle manifest contains an invalid asset")
        raw_path = raw_asset.get("path")
        raw_hash = raw_asset.get("sha256")
        raw_bytes = raw_asset.get("bytes")
        if not isinstance(raw_path, str):
            raise PublicExportError("Runtime bundle asset path must be a string")
        relative = _portable_path(raw_path)
        if (
            not isinstance(raw_hash, str)
            or re.fullmatch(r"[0-9a-f]{64}", raw_hash) is None
            or not isinstance(raw_bytes, int)
            or isinstance(raw_bytes, bool)
            or raw_bytes < 0
            or relative in assets
        ):
            raise PublicExportError(f"Runtime bundle asset identity is invalid: {relative}")
        if relative not in tracked:
            raise PublicExportError(f"Runtime bundle asset is not tracked: {relative}")
        path = repository.joinpath(*PurePosixPath(relative).parts)
        if not path.is_file() or path.is_symlink():
            raise PublicExportError(f"Runtime bundle asset is missing or linked: {relative}")
        if path.stat().st_size != raw_bytes or _sha256_file(path) != raw_hash:
            raise PublicExportError(f"Runtime bundle asset hash or size differs: {relative}")
        assets[relative] = raw_asset
    if manifest.get("fileCount") != len(assets):
        raise PublicExportError("Runtime bundle manifest file count differs from its assets")
    return assets


def _publication_allowlist(repository: Path) -> dict[str, object]:
    path = repository / "scripts" / "publication-allowlist.json"
    tracked = {
        item.decode("utf-8", "strict")
        for item in _git(repository, "ls-files", "--cached", "-z").split(b"\0")
        if item
    }
    if "scripts/publication-allowlist.json" not in tracked:
        raise PublicExportError("Publication allowlist is not tracked")
    document = _json_object(path, label="Publication allowlist")
    if document.get("schemaVersion") != "socialgraph-fm.publication-allowlist/1.0":
        raise PublicExportError("Publication allowlist schema is unsupported")
    if not isinstance(document.get("archives", {}), dict) or not isinstance(
        document.get("binaryMetadata", {}), dict
    ):
        raise PublicExportError("Publication allowlist maps are invalid")
    return document


def _scan_content(path: str, content: bytes) -> None:
    candidates = [content]
    if b"\0" in content:
        try:
            candidates.append(content.decode("utf-16-le").encode("utf-8"))
        except UnicodeError:
            pass
    for selected in candidates:
        if any(pattern.search(selected) is not None for pattern in _SECRET_PATTERNS):
            raise PublicExportError(f"Potential secret in tracked file: {path}")
        if any(pattern.search(selected) is not None for pattern in _PERSONAL_PATH_PATTERNS):
            raise PublicExportError(f"Personal absolute path in tracked file: {path}")


def _validate_visual(
    relative: str,
    content: bytes,
    binary_metadata: dict[str, object],
) -> None:
    suffix = PurePosixPath(relative).suffix.casefold()
    magic_matches = {
        ".png": content.startswith(b"\x89PNG\r\n\x1a\n"),
        ".jpg": content.startswith(b"\xff\xd8\xff"),
        ".jpeg": content.startswith(b"\xff\xd8\xff"),
        ".webp": len(content) >= 12
        and content[:4] == b"RIFF"
        and content[8:12] == b"WEBP",
        ".woff2": content.startswith(b"wOF2"),
    }.get(suffix, False)
    if not magic_matches:
        raise PublicExportError(f"Visual asset extension/magic mismatch: {relative}")
    lowered = content.lower()
    if b"c2pa" in lowered or b"cabx" in lowered:
        policy = binary_metadata.get(relative)
        if not isinstance(policy, dict) or policy.get("allowC2pa") is not True:
            raise PublicExportError(f"Unapproved C2PA metadata in visual asset: {relative}")


def _validate_archive_member_name(value: str) -> str:
    selected = value[:-1] if value.endswith("/") else value
    if not selected:
        raise PublicExportError("Archive contains an empty member path")
    return _portable_path(selected)


def _scan_archive(
    path: Path,
    *,
    relative: str,
    runtime_policy: dict[str, object] | None,
    archive_policy: object,
) -> None:
    if runtime_policy is None:
        if not isinstance(archive_policy, dict) or archive_policy.get("synthetic") is not True:
            raise PublicExportError(
                f"Archive is neither runtime-manifest-bound nor approved synthetic data: {relative}"
            )
        expected_hash = archive_policy.get("sha256")
    else:
        expected_hash = runtime_policy.get("sha256")
    if not isinstance(expected_hash, str) or _sha256_file(path) != expected_hash:
        raise PublicExportError(f"Approved archive hash differs: {relative}")
    try:
        with zipfile.ZipFile(path, "r") as archive:
            members = archive.infolist()
            if len(members) > _MAX_ARCHIVE_MEMBERS:
                raise PublicExportError(f"Archive contains too many members: {relative}")
            if runtime_policy is None:
                assert isinstance(archive_policy, dict)
                expected_members = archive_policy.get("members")
                if not isinstance(expected_members, list) or not all(
                    isinstance(member, str) for member in expected_members
                ):
                    raise PublicExportError(
                        f"Synthetic archive member policy is invalid: {relative}"
                    )
                if sorted(member.filename for member in members) != sorted(expected_members):
                    raise PublicExportError(f"Synthetic archive inventory differs: {relative}")
            expanded = 0
            normalized: set[str] = set()
            for member in members:
                name = _validate_archive_member_name(member.filename)
                portable = unicodedata.normalize("NFC", name).casefold()
                if portable in normalized:
                    raise PublicExportError(f"Archive contains duplicate paths: {relative}")
                normalized.add(portable)
                if member.flag_bits & 0x1:
                    raise PublicExportError(f"Archive contains encrypted data: {relative}")
                member_mode = (member.external_attr >> 16) & 0xFFFF
                if stat.S_ISLNK(member_mode):
                    raise PublicExportError(f"Archive contains a symbolic link: {relative}")
                expanded += member.file_size
                if expanded > _MAX_ARCHIVE_EXPANDED_BYTES:
                    raise PublicExportError(f"Archive exceeds expanded byte budget: {relative}")
                if member.is_dir():
                    continue
                _scan_content(f"{relative} -> {member.filename}", archive.read(member))
            corrupt = archive.testzip()
            if corrupt is not None:
                raise PublicExportError(f"Archive CRC verification failed: {relative} -> {corrupt}")
    except (OSError, zipfile.BadZipFile, RuntimeError) as exc:
        if isinstance(exc, PublicExportError):
            raise
        raise PublicExportError(f"Tracked archive is invalid: {relative}: {exc}") from None


def run_builtin_publication_scan(
    repository: Path, tree: tuple[_TreeEntry, ...] | None = None
) -> None:
    """Run the cross-platform, standard-library publication safety gate."""

    selected_tree = tree or _head_tree(repository)
    tracked = {
        item.decode("utf-8", "strict")
        for item in _git(repository, "ls-files", "--cached", "-z").split(b"\0")
        if item
    }
    expected = {entry.path for entry in selected_tree}
    if tracked != expected:
        raise PublicExportError("Git index inventory differs from the HEAD tree")
    candidates = _git(
        repository, "ls-files", "--cached", "--others", "--exclude-standard", "-z"
    )
    candidate_paths = {
        item.decode("utf-8", "strict") for item in candidates.split(b"\0") if item
    }
    if candidate_paths != tracked:
        raise PublicExportError("Publication candidate includes untracked, non-ignored files")

    runtime_assets = _runtime_manifest_assets(repository, tracked)
    allowlist = _publication_allowlist(repository)
    raw_archives = allowlist.get("archives", {})
    raw_binary_metadata = allowlist.get("binaryMetadata", {})
    assert isinstance(raw_archives, dict)
    assert isinstance(raw_binary_metadata, dict)

    for relative in sorted(tracked):
        parts = PurePosixPath(relative).parts
        lowered = relative.casefold()
        name = parts[-1].casefold()
        if ".superpowers" in (part.casefold() for part in parts) or lowered.startswith(
            "platform/"
        ):
            raise PublicExportError(f"Forbidden internal path is tracked: {relative}")
        if lowered.startswith("var/") and relative != "var/.gitkeep":
            raise PublicExportError(f"Runtime state is tracked: {relative}")
        if re.search(r"(^|/)\.env(?:\.|$)", relative, re.IGNORECASE) and name != ".env.example":
            raise PublicExportError(f"Private environment file is tracked: {relative}")
        if _is_runtime_binary(relative) and relative not in runtime_assets:
            raise PublicExportError(f"Unknown tracked runtime binary: {relative}")
        if (
            lowered.startswith("bundles/models/")
            or lowered.startswith("bundles/governance/")
            or lowered.startswith("examples/governance/")
        ) and relative not in runtime_assets:
            raise PublicExportError(f"Unmanifested runtime payload is tracked: {relative}")

        path = repository.joinpath(*parts)
        if not path.is_file() or path.is_symlink():
            raise PublicExportError(f"Tracked path is missing, linked, or not a file: {relative}")
        if path.stat().st_size > _MAX_SCAN_FILE_BYTES:
            raise PublicExportError(f"Tracked file exceeds publication byte budget: {relative}")
        content = path.read_bytes()
        _scan_content(relative, content)
        suffix = PurePosixPath(relative).suffix.casefold()
        visual = suffix in _VISUAL_EXTENSIONS
        approved_archive = relative in runtime_assets or relative in raw_archives
        binary_content = b"\0" in content[:8192]
        if binary_content and not visual and not approved_archive:
            raise PublicExportError(f"Unknown tracked binary content: {relative}")
        if visual:
            _validate_visual(relative, content, raw_binary_metadata)
        if suffix in _ARCHIVE_EXTENSIONS:
            runtime_policy = runtime_assets.get(relative)
            _scan_archive(
                path,
                relative=relative,
                runtime_policy=runtime_policy,
                archive_policy=raw_archives.get(relative),
            )


def run_powershell_publication_scan(repository: Path) -> None:
    """Optionally run the legacy PowerShell publication gate in addition to Python."""

    candidates = ("pwsh", "powershell") if os.name == "nt" else ("pwsh",)
    shell = next((selected for name in candidates if (selected := shutil.which(name))), None)
    if shell is None:
        raise PublicExportError("The optional PowerShell publication scan is unavailable")
    script = repository / "scripts" / "publication-check.ps1"
    if not script.is_file() or script.is_symlink():
        raise PublicExportError(f"Publication safety scan entry is missing: {script}")
    command = [shell, "-NoProfile", "-NonInteractive"]
    if os.name == "nt":
        command.extend(("-ExecutionPolicy", "Bypass"))
    command.extend(("-File", str(script), "-RepositoryRoot", str(repository)))
    _run(command, cwd=repository, timeout=1800, environment=dict(os.environ))


def run_brand_publication_scan(repository: Path) -> None:
    """Run the tracked cross-platform brand gate in an isolated Python child."""

    script = repository / "scripts" / "brand-scan.py"
    if not script.is_file() or script.is_symlink():
        raise PublicExportError(f"Brand scan entry is missing: {script}")
    environment = dict(os.environ)
    for name in tuple(environment):
        if name.upper().startswith("PYTHON"):
            environment.pop(name, None)
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    _run(
        [
            sys.executable,
            "-I",
            str(script),
            "--repository-root",
            str(repository),
        ],
        cwd=repository,
        timeout=1800,
        environment=environment,
    )


def _assert_source_unchanged(source: Path, head: str) -> None:
    current = _git(source, "rev-parse", "--verify", "HEAD").decode("ascii").strip()
    status = _git(source, "status", "--porcelain=v1", "--untracked-files=all")
    if current != head or status:
        raise PublicExportError("Source repository changed during public export")


def _tree_blob_contents(
    source: Path, tree: tuple[_TreeEntry, ...]
) -> tuple[bytes, ...]:
    executable = shutil.which("git")
    if executable is None:
        raise PublicExportError("Git is required to export the public repository")
    request = b"".join(entry.object_id.encode("ascii") + b"\n" for entry in tree)
    raw = _run(
        [executable, "-C", str(source), "cat-file", "--batch"],
        cwd=source,
        timeout=1800,
        environment=_git_environment(),
        input_data=request,
    ).stdout
    contents: list[bytes] = []
    offset = 0
    for entry in tree:
        header_end = raw.find(b"\n", offset)
        if header_end < 0:
            raise PublicExportError("Git blob batch ended before its header")
        header = raw[offset:header_end].split(b" ")
        expected = [
            entry.object_id.encode("ascii"),
            b"blob",
            str(entry.size).encode("ascii"),
        ]
        if header != expected:
            raise PublicExportError(f"Git blob metadata differs from HEAD: {entry.path}")
        content_start = header_end + 1
        content_end = content_start + entry.size
        if content_end >= len(raw) or raw[content_end : content_end + 1] != b"\n":
            raise PublicExportError(f"Git blob payload is truncated: {entry.path}")
        contents.append(raw[content_start:content_end])
        offset = content_end + 1
    if offset != len(raw):
        raise PublicExportError("Git blob batch returned unexpected trailing data")
    return tuple(contents)


def _create_archive(
    source: Path, destination: Path, tree: tuple[_TreeEntry, ...]
) -> tuple[bytes, ...]:
    """Write deterministic Git-blob bytes without platform EOL conversion."""

    contents = _tree_blob_contents(source, tree)
    try:
        with zipfile.ZipFile(
            destination, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9
        ) as selected:
            for entry, content in zip(tree, contents, strict=True):
                member = zipfile.ZipInfo(entry.path, date_time=(1980, 1, 1, 0, 0, 0))
                member.create_system = 3
                member.external_attr = (
                    (0o100755 if entry.mode == "100755" else 0o100644) << 16
                )
                member.compress_type = zipfile.ZIP_DEFLATED
                selected.writestr(
                    member,
                    content,
                    compress_type=zipfile.ZIP_DEFLATED,
                    compresslevel=9,
                )
    except OSError as exc:
        raise PublicExportError(f"Could not create exported ZIP: {exc}") from None
    return contents


def _verify_archive(
    archive: Path,
    tree: tuple[_TreeEntry, ...],
    contents: tuple[bytes, ...],
) -> None:
    expected = {entry.path: entry for entry in tree}
    actual: dict[str, zipfile.ZipInfo] = {}
    normalized_names: set[str] = set()
    try:
        with zipfile.ZipFile(archive, "r") as selected:
            for member in selected.infolist():
                name = member.filename[:-1] if member.is_dir() else member.filename
                name = _portable_path(name)
                normalized = unicodedata.normalize("NFC", name).casefold()
                if normalized in normalized_names:
                    raise PublicExportError(f"ZIP contains a duplicate portable path: {name}")
                normalized_names.add(normalized)
                if member.flag_bits & 0x1:
                    raise PublicExportError(f"ZIP contains an encrypted member: {name}")
                if not member.is_dir():
                    actual[name] = member
            if set(actual) != set(expected):
                missing = sorted(set(expected) - set(actual))[:5]
                extra = sorted(set(actual) - set(expected))[:5]
                raise PublicExportError(
                    f"ZIP/HEAD inventory mismatch; missing={missing}, extra={extra}"
                )
            expected_content = {
                entry.path: content for entry, content in zip(tree, contents, strict=True)
            }
            for name, member in actual.items():
                if member.file_size != expected[name].size:
                    raise PublicExportError(f"ZIP member size differs from HEAD: {name}")
                if selected.read(member) != expected_content[name]:
                    raise PublicExportError(f"ZIP member bytes differ from HEAD: {name}")
            corrupt = selected.testzip()
            if corrupt is not None:
                raise PublicExportError(f"ZIP CRC verification failed: {corrupt}")
    except (OSError, zipfile.BadZipFile) as exc:
        raise PublicExportError(f"Could not verify exported ZIP: {exc}") from None


def _extract_archive(
    archive: Path, destination: Path, tree: tuple[_TreeEntry, ...]
) -> None:
    with zipfile.ZipFile(archive, "r") as selected:
        for entry in tree:
            target = destination.joinpath(*PurePosixPath(entry.path).parts)
            target.parent.mkdir(parents=True, exist_ok=True)
            with selected.open(entry.path, "r") as source, target.open("xb") as output:
                shutil.copyfileobj(source, output, length=1024 * 1024)
            if os.name != "nt":
                target.chmod(0o755 if entry.mode == "100755" else 0o644)


def _initialize_repository(
    repository: Path,
    *,
    template: Path,
    source_tree: str,
    message: str,
) -> str:
    if not message.strip() or "\0" in message:
        raise PublicExportError("Initial commit message cannot be empty or contain NUL")
    executable = shutil.which("git")
    if executable is None:
        raise PublicExportError("Git is required to export the public repository")
    environment = _git_environment()
    _run(
        [
            executable,
            "-c",
            "init.defaultBranch=main",
            "init",
            "--initial-branch=main",
            f"--template={template}",
            str(repository),
        ],
        cwd=repository.parent,
        environment=environment,
    )
    _git(repository, "config", "core.autocrlf", "false")
    _git(repository, "add", "--all")
    exported_tree = _git(repository, "write-tree").decode("ascii").strip()
    if exported_tree != source_tree:
        raise PublicExportError("Exported repository tree differs from source HEAD")
    commit_environment = _git_environment()
    commit_environment.update(
        {
            "GIT_AUTHOR_NAME": NEUTRAL_AUTHOR_NAME,
            "GIT_AUTHOR_EMAIL": NEUTRAL_AUTHOR_EMAIL,
            "GIT_COMMITTER_NAME": NEUTRAL_AUTHOR_NAME,
            "GIT_COMMITTER_EMAIL": NEUTRAL_AUTHOR_EMAIL,
        }
    )
    _run(
        [
            executable,
            "-C",
            str(repository),
            "-c",
            f"core.hooksPath={template}",
            "-c",
            "commit.gpgsign=false",
            "commit",
            "--no-gpg-sign",
            "--no-verify",
            "--message",
            message,
        ],
        cwd=repository,
        environment=commit_environment,
    )
    return _git(repository, "rev-parse", "HEAD").decode("ascii").strip()


def _verify_repository(repository: Path, source_tree: str) -> str:
    if _git(repository, "status", "--porcelain=v1", "--untracked-files=all"):
        raise PublicExportError("Exported repository worktree is not clean")
    if _git(repository, "ls-files", "--others", "--ignored", "--exclude-standard", "-z"):
        raise PublicExportError("Exported repository contains ignored worktree state")
    if _git(repository, "symbolic-ref", "--short", "HEAD").decode().strip() != "main":
        raise PublicExportError("Exported repository HEAD is not main")
    refs = _git(repository, "for-each-ref", "--format=%(refname)").decode().splitlines()
    if refs != ["refs/heads/main"]:
        raise PublicExportError(f"Exported repository contains unexpected refs: {refs}")
    if _git(repository, "remote").strip():
        raise PublicExportError("Exported repository contains a remote")
    if _git(repository, "rev-list", "--all", "--count").decode().strip() != "1":
        raise PublicExportError("Exported repository must contain exactly one commit")
    identity = _git(
        repository, "show", "-s", "--format=%an%x00%ae%x00%cn%x00%ce", "HEAD"
    ).decode("utf-8").strip().split("\0")
    expected_identity = [
        NEUTRAL_AUTHOR_NAME,
        NEUTRAL_AUTHOR_EMAIL,
        NEUTRAL_AUTHOR_NAME,
        NEUTRAL_AUTHOR_EMAIL,
    ]
    if identity != expected_identity:
        raise PublicExportError("Exported repository commit identity is not neutral")
    revision = _git(repository, "rev-list", "--parents", "--max-count=1", "HEAD")
    if len(revision.decode("ascii").split()) != 1:
        raise PublicExportError("Exported repository root commit has a parent")
    tree = _git(repository, "rev-parse", "HEAD^{tree}").decode("ascii").strip()
    if tree != source_tree:
        raise PublicExportError("Exported repository does not preserve the source HEAD tree")
    return _git(repository, "rev-parse", "HEAD").decode("ascii").strip()


def _finalize_repository(repository: Path, source_tree: str) -> str:
    _git(repository, "reflog", "expire", "--expire=now", "--all")
    _git(repository, "gc", "--prune=now")
    fsck = _git(repository, "fsck", "--full", "--unreachable", "--no-reflogs")
    if b"unreachable" in fsck.lower():
        raise PublicExportError("Exported repository contains unreachable Git objects")
    if _git(repository, "reflog", "show", "--all").strip():
        raise PublicExportError("Exported repository retains reflog history")
    return _verify_repository(repository, source_tree)


def _safe_cleanup_directory(path: Path | None, *, parent: Path, prefix: str) -> None:
    if path is None or not os.path.lexists(path):
        return
    if path.parent != parent or not path.name.startswith(prefix):
        raise PublicExportError(f"Refusing to clean unsafe staging directory: {path}")
    _remove_tree(path)


def _remove_tree(path: Path) -> None:
    def make_writable(function: Callable[[str], object], selected: str, _: BaseException) -> None:
        os.chmod(selected, stat.S_IWRITE)
        function(selected)

    shutil.rmtree(path, onexc=make_writable)


def _safe_cleanup_file(path: Path | None, *, parent: Path, prefix: str) -> None:
    if path is None or not os.path.lexists(path):
        return
    if path.parent != parent or not path.name.startswith(prefix):
        raise PublicExportError(f"Refusing to clean unsafe staging file: {path}")
    path.unlink()


def export_public_snapshot(
    source_repository: Path,
    repository_destination: Path,
    zip_destination: Path,
    *,
    scan_runner: ScanRunner | None = None,
    commit_message: str = DEFAULT_COMMIT_MESSAGE,
) -> ExportResult:
    """Create a verified Download ZIP and a fresh, single-commit public repository."""

    source, source_head, source_tree = _validate_source(source_repository)
    repository_destination, zip_destination = _validate_targets(
        source, repository_destination, zip_destination
    )
    tree = _head_tree(source)
    run_builtin_publication_scan(source, tree)
    run_brand_publication_scan(source)
    if scan_runner is not None:
        scan_runner(source)
    _assert_source_unchanged(source, source_head)

    repository_staging: Path | None = None
    template_staging: Path | None = None
    zip_staging: Path | None = None
    published_zip = False
    published_repository = False
    try:
        repository_staging = Path(
            tempfile.mkdtemp(
                prefix=_REPOSITORY_STAGING_PREFIX,
                dir=repository_destination.parent,
            )
        ).resolve()
        template_staging = Path(
            tempfile.mkdtemp(
                prefix=_TEMPLATE_STAGING_PREFIX,
                dir=repository_destination.parent,
            )
        ).resolve()
        descriptor, raw_zip_staging = tempfile.mkstemp(
            prefix=_ZIP_STAGING_PREFIX,
            suffix=".zip",
            dir=zip_destination.parent,
        )
        os.close(descriptor)
        zip_staging = Path(raw_zip_staging).resolve()

        blob_contents = _create_archive(source, zip_staging, tree)
        if zip_staging.stat().st_size > _MAX_EXPORT_ZIP_BYTES:
            raise PublicExportError(
                "Public Download ZIP exceeds the 30 MiB release budget"
            )
        _verify_archive(zip_staging, tree, blob_contents)
        _extract_archive(zip_staging, repository_staging, tree)
        repository_commit = _initialize_repository(
            repository_staging,
            template=template_staging,
            source_tree=source_tree,
            message=commit_message,
        )
        run_builtin_publication_scan(repository_staging, tree)
        run_brand_publication_scan(repository_staging)
        if scan_runner is not None:
            scan_runner(repository_staging)
        repository_commit = _finalize_repository(repository_staging, source_tree)
        _assert_source_unchanged(source, source_head)
        _verify_archive(zip_staging, tree, blob_contents)

        if os.path.lexists(zip_destination) or os.path.lexists(repository_destination):
            raise PublicExportError("An export destination appeared while staging")
        os.rename(zip_staging, zip_destination)
        published_zip = True
        zip_staging = None
        try:
            os.rename(repository_staging, repository_destination)
            published_repository = True
            repository_staging = None
        except BaseException:
            zip_destination.unlink(missing_ok=True)
            published_zip = False
            raise

        digest = _sha256_file(zip_destination)
        return ExportResult(
            source_head=source_head,
            source_tree=source_tree,
            repository_commit=repository_commit,
            repository_destination=repository_destination,
            zip_destination=zip_destination,
            zip_sha256=digest,
            zip_bytes=zip_destination.stat().st_size,
            tracked_file_count=len(tree),
        )
    except PublicExportError:
        raise
    except Exception as exc:
        if published_repository and os.path.lexists(repository_destination):
            _remove_tree(repository_destination)
            published_repository = False
        if published_zip and os.path.lexists(zip_destination):
            zip_destination.unlink()
            published_zip = False
        raise PublicExportError(f"Public export failed: {exc}") from None
    finally:
        _safe_cleanup_directory(
            template_staging,
            parent=repository_destination.parent,
            prefix=_TEMPLATE_STAGING_PREFIX,
        )
        _safe_cleanup_directory(
            repository_staging,
            parent=repository_destination.parent,
            prefix=_REPOSITORY_STAGING_PREFIX,
        )
        _safe_cleanup_file(
            zip_staging,
            parent=zip_destination.parent,
            prefix=_ZIP_STAGING_PREFIX,
        )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Export a safe Download ZIP and a fresh public Git repository."
    )
    parser.add_argument("--source", type=Path, default=Path.cwd())
    parser.add_argument("--repository", type=Path, required=True)
    parser.add_argument("--zip", dest="zip_destination", type=Path, required=True)
    parser.add_argument("--message", default=DEFAULT_COMMIT_MESSAGE)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    try:
        result = export_public_snapshot(
            arguments.source,
            arguments.repository,
            arguments.zip_destination,
            commit_message=arguments.message,
        )
    except PublicExportError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    print(json.dumps(result.to_document(), ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
