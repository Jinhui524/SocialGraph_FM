from __future__ import annotations

import argparse
import json
import re
import subprocess
import zipfile
from pathlib import Path, PurePosixPath


LONG_TOKENS = tuple(
    value.casefold()
    for value in (
        "io" + "hunter",
        "info" + "opsgfm",
        "py" + "gfm",
        "generic" + "agent",
        "md" + "gfm",
        "static" + "-v2",
        "static" + "_v2",
        "research" + "-v1",
        "research" + "_v1",
        "ioh" + "2",
        "bridge" + "top2router",
        "bridge" + "-inspired",
    )
)
SHORT_PATTERN = re.compile(r"(?<![a-z0-9])r" + r"q[123](?![a-z0-9])", re.IGNORECASE)
IDENTIFIER_PATTERN = re.compile(r"(?<![A-Za-z0-9])(?:rq|RQ)[123](?=[A-Z_]|$)")
TEXT_SUFFIXES = {
    ".css",
    ".csv",
    ".html",
    ".js",
    ".json",
    ".md",
    ".mjs",
    ".ps1",
    ".py",
    ".toml",
    ".ts",
    ".tsx",
    ".txt",
    ".xml",
    ".yaml",
    ".yml",
}
ARCHIVE_SUFFIXES = {".zip", ".npz"}
RAW_BINARY_SUFFIXES = {".pt", ".sqlite", ".sqlite3"}
CONTENT_EXCEPTIONS = {
    "THIRD_PARTY_NOTICES.md",
}
MODEL_CARD = "bundles/models/socialgraph-global/exports/socialgraph-global/model-card.json"
MAX_ARCHIVE_DEPTH = 3
MAX_MEMBER_BYTES = 64 * 1024 * 1024


def _matches(value: str) -> list[str]:
    folded = value.casefold()
    found = [token for token in LONG_TOKENS if token in folded]
    if SHORT_PATTERN.search(value):
        found.append("research-question protocol label")
    if IDENTIFIER_PATTERN.search(value):
        found.append("research-question identifier")
    return found


def _portable_member(value: str) -> str:
    if not value or "\\" in value or value.startswith("/") or ":" in value:
        raise ValueError(f"unsafe archive member path: {value!r}")
    path = PurePosixPath(value)
    if any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError(f"unsafe archive member path: {value!r}")
    return value


def _scan_bytes(label: str, value: bytes, *, short_tokens: bool) -> list[str]:
    folded = value.lower()
    findings = [token for token in LONG_TOKENS if token.encode("ascii") in folded]
    if short_tokens:
        text = value.decode("utf-8", "ignore")
        if SHORT_PATTERN.search(text):
            findings.append("research-question protocol label")
        if IDENTIFIER_PATTERN.search(text):
            findings.append("research-question identifier")
    return [f"{label}: {item}" for item in findings]


def _scan_archive(label: str, value: bytes, *, depth: int) -> list[str]:
    if depth > MAX_ARCHIVE_DEPTH:
        raise ValueError(f"archive nesting exceeds {MAX_ARCHIVE_DEPTH}: {label}")
    findings: list[str] = []
    from io import BytesIO

    with zipfile.ZipFile(BytesIO(value), "r") as archive:
        for member in archive.infolist():
            name = _portable_member(member.filename)
            findings.extend(f"{label} -> {name}: {item}" for item in _matches(name))
            if member.is_dir():
                continue
            if member.file_size > MAX_MEMBER_BYTES:
                continue
            content = archive.read(member)
            suffix = PurePosixPath(name).suffix.casefold()
            if suffix in ARCHIVE_SUFFIXES and zipfile.is_zipfile(BytesIO(content)):
                findings.extend(_scan_archive(f"{label} -> {name}", content, depth=depth + 1))
            elif suffix in TEXT_SUFFIXES:
                findings.extend(_scan_bytes(f"{label} -> {name}", content, short_tokens=True))
            elif suffix in RAW_BINARY_SUFFIXES:
                findings.extend(_scan_bytes(f"{label} -> {name}", content, short_tokens=False))
    return findings


def _tracked(root: Path) -> list[str]:
    completed = subprocess.run(
        ["git", "-C", str(root), "ls-files", "-z"],
        check=False,
        capture_output=True,
    )
    if completed.returncode == 0:
        return sorted(
            value.decode("utf-8")
            for value in completed.stdout.split(b"\0")
            if value
        )
    return sorted(
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() and ".git" not in path.relative_to(root).parts
    )


def _scan_model_card(label: str, value: bytes) -> list[str]:
    try:
        candidate = json.loads(value)
    except (UnicodeDecodeError, json.JSONDecodeError):
        return _scan_bytes(label, value, short_tokens=True)
    if not isinstance(candidate, dict):
        return _scan_bytes(label, value, short_tokens=True)
    public_content = dict(candidate)
    public_content.pop("licenses", None)
    public_content.pop("sourceAttribution", None)
    return _scan_bytes(
        label,
        json.dumps(public_content, ensure_ascii=False, sort_keys=True).encode("utf-8"),
        short_tokens=True,
    )


def scan(root: Path) -> list[str]:
    findings: list[str] = []
    for relative in _tracked(root):
        findings.extend(f"path {relative}: {item}" for item in _matches(relative))
        if relative in CONTENT_EXCEPTIONS:
            continue
        path = root.joinpath(*PurePosixPath(relative).parts)
        if not path.is_file():
            findings.append(f"missing tracked file: {relative}")
            continue
        suffix = path.suffix.casefold()
        if relative == MODEL_CARD:
            findings.extend(_scan_model_card(relative, path.read_bytes()))
        elif suffix in ARCHIVE_SUFFIXES:
            findings.extend(_scan_archive(relative, path.read_bytes(), depth=0))
        elif suffix in TEXT_SUFFIXES:
            findings.extend(_scan_bytes(relative, path.read_bytes(), short_tokens=True))
        elif suffix in RAW_BINARY_SUFFIXES:
            findings.extend(_scan_bytes(relative, path.read_bytes(), short_tokens=False))
    return findings


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository-root", type=Path, required=True)
    arguments = parser.parse_args()
    root = arguments.repository_root.resolve(strict=True)
    findings = scan(root)
    if findings:
        print("Brand scan rejected the publication candidate:")
        for finding in findings[:100]:
            print(f"- {finding}")
        if len(findings) > 100:
            print(f"- ... {len(findings) - 100} additional findings")
        return 1
    print(f"Brand scan passed for {len(_tracked(root))} tracked files.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
