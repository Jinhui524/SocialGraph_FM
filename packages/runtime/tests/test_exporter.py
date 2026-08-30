from __future__ import annotations

import hashlib
import json
import os
import subprocess
import zipfile
from pathlib import Path

import pytest

from socialgraph_fm_runtime import exporter
from socialgraph_fm_runtime.exporter import PublicExportError, export_public_snapshot


def _run(repository: Path, *arguments: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(repository), *arguments],
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _write(path: Path, content: str | bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(content, str):
        path.write_text(content, encoding="utf-8", newline="\n")
    else:
        path.write_bytes(content)


def _runtime_checkpoint(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("checkpoint/data.pkl", b"safe model fixture")
        archive.writestr("checkpoint/tensor.bin", b"\x00\x01\x02\x03")


def _source_repository(tmp_path: Path) -> Path:
    source = tmp_path / "source repository"
    source.mkdir()
    empty_template = tmp_path / "empty-template"
    empty_template.mkdir()
    subprocess.run(
        [
            "git",
            "init",
            "--initial-branch=main",
            f"--template={empty_template}",
            str(source),
        ],
        check=True,
        capture_output=True,
    )
    _write(source / ".gitignore", "ignored/\n")
    _write(source / ".gitattributes", "* text=auto\n*.ps1 text eol=crlf\n")
    _write(source / ".editorconfig", "root = true\n[*]\nend_of_line = lf\n")
    _write(source / "README.md", "# Safe public fixture\n")
    _write(source / "scripts/publication-check.ps1", "Write-Output 'fixture scan'\n")
    _write(
        source / "scripts/brand-scan.py",
        "from __future__ import annotations\nraise SystemExit(0)\n",
    )
    _write(source / "var/.gitkeep", "\n")
    _write(source / "apps/web/public/assets/brand-mark.png", b"\x89PNG\r\n\x1a\nfixture")
    checkpoint = source / "bundles/models/socialgraph-global/checkpoints/global.pt"
    _runtime_checkpoint(checkpoint)
    relative_checkpoint = checkpoint.relative_to(source).as_posix()
    checkpoint_bytes = checkpoint.read_bytes()
    manifest = {
        "schemaVersion": "socialgraph-fm.runtime-bundle/1.0",
        "fileCount": 1,
        "assets": [
            {
                "path": relative_checkpoint,
                "bytes": len(checkpoint_bytes),
                "sha256": hashlib.sha256(checkpoint_bytes).hexdigest(),
            }
        ],
    }
    _write(
        source / "bundles/runtime-manifest.json",
        json.dumps(manifest, sort_keys=True) + "\n",
    )
    allowlist = {
        "schemaVersion": "socialgraph-fm.publication-allowlist/1.0",
        "archives": {},
        "binaryMetadata": {},
    }
    _write(
        source / "scripts/publication-allowlist.json",
        json.dumps(allowlist, sort_keys=True) + "\n",
    )
    _write(source / "ignored/local-state.bin", b"must not be exported")
    _run(source, "add", "--all")
    _run(
        source,
        "-c",
        "user.name=Private Developer",
        "-c",
        "user.email=private@example.invalid",
        "commit",
        "--message=private history",
    )
    return source


def _assert_no_staging(parent: Path) -> None:
    assert not [
        path
        for path in parent.iterdir()
        if path.name.startswith(
            (
                exporter._REPOSITORY_STAGING_PREFIX,
                exporter._TEMPLATE_STAGING_PREFIX,
                exporter._ZIP_STAGING_PREFIX,
            )
        )
    ]


def test_export_preserves_head_assets_and_builds_two_clean_atomic_outputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = _source_repository(tmp_path)
    repository = tmp_path / "public repository"
    archive = tmp_path / "public download.zip"
    scans: list[Path] = []
    original_which = exporter.shutil.which

    def without_powershell(name: str) -> str | None:
        if name in {"pwsh", "powershell"}:
            return None
        return original_which(name)

    monkeypatch.setattr(exporter.shutil, "which", without_powershell)
    result = export_public_snapshot(
        source,
        repository,
        archive,
        scan_runner=lambda selected: scans.append(selected),
    )

    assert scans[0] == source.resolve()
    assert scans[1].parent == repository.parent.resolve()
    assert scans[1].name.startswith(exporter._REPOSITORY_STAGING_PREFIX)
    assert result.repository_destination == repository.resolve()
    assert result.zip_destination == archive.resolve()
    assert result.tracked_file_count == len(_run(source, "ls-files").splitlines())
    assert result.zip_sha256 == hashlib.sha256(archive.read_bytes()).hexdigest()
    assert _run(repository, "symbolic-ref", "--short", "HEAD") == "main"
    assert _run(repository, "rev-list", "--all", "--count") == "1"
    assert len(_run(repository, "rev-list", "--parents", "-n", "1", "HEAD").split()) == 1
    assert _run(repository, "rev-parse", "HEAD^{tree}") == _run(
        source, "rev-parse", "HEAD^{tree}"
    )
    assert _run(repository, "remote") == ""
    assert _run(repository, "tag") == ""
    assert _run(repository, "status", "--porcelain=v1", "--untracked-files=all") == ""
    identity = _run(repository, "show", "-s", "--format=%an|%ae|%cn|%ce", "HEAD")
    assert identity == (
        "SocialGraph-FM Contributors|socialgraph-fm@users.noreply.github.com|"
        "SocialGraph-FM Contributors|socialgraph-fm@users.noreply.github.com"
    )
    assert _run(repository, "fsck", "--full", "--unreachable", "--no-reflogs") == ""
    git_config = (repository / ".git/config").read_text(encoding="utf-8")
    assert "hooksPath" not in git_config
    assert str(source.resolve()) not in git_config
    assert exporter._TEMPLATE_STAGING_PREFIX not in git_config
    assert not (repository / "ignored").exists()
    with zipfile.ZipFile(archive, "r") as selected:
        members = {member.filename for member in selected.infolist() if not member.is_dir()}
        assert members == set(_run(source, "ls-files").splitlines())
        assert not any(".git" in Path(member).parts for member in members)
        expected_editorconfig = subprocess.run(
            ["git", "-C", str(source), "cat-file", "blob", "HEAD:.editorconfig"],
            check=True,
            capture_output=True,
        ).stdout
        assert selected.read(".editorconfig") == expected_editorconfig
        assert selected.read("bundles/models/socialgraph-global/checkpoints/global.pt")
    _assert_no_staging(tmp_path)


@pytest.mark.parametrize("dirty_kind", ["modified", "untracked"])
def test_export_rejects_a_dirty_source_before_scanning(
    tmp_path: Path, dirty_kind: str
) -> None:
    source = _source_repository(tmp_path)
    if dirty_kind == "modified":
        _write(source / "README.md", "changed\n")
    else:
        _write(source / "untracked.txt", "new\n")
    calls: list[Path] = []

    with pytest.raises(PublicExportError, match="must be clean"):
        export_public_snapshot(
            source,
            tmp_path / "public",
            tmp_path / "public.zip",
            scan_runner=lambda selected: calls.append(selected),
        )

    assert calls == []
    assert not (tmp_path / "public").exists()
    assert not (tmp_path / "public.zip").exists()


def test_extra_scan_failure_leaves_no_output_or_staging(tmp_path: Path) -> None:
    source = _source_repository(tmp_path)

    def fail_scan(_: Path) -> None:
        raise PublicExportError("external scan rejected fixture")

    with pytest.raises(PublicExportError, match="external scan rejected"):
        export_public_snapshot(
            source,
            tmp_path / "public",
            tmp_path / "public.zip",
            scan_runner=fail_scan,
        )

    assert not (tmp_path / "public").exists()
    assert not (tmp_path / "public.zip").exists()
    _assert_no_staging(tmp_path)


@pytest.mark.parametrize(
    ("repository_name", "zip_name", "message"),
    [
        ("existing", "download.zip", "already exists"),
        ("public", "not-an-archive.bin", "must end in .zip"),
        ("same.zip", "same.zip", "must differ"),
    ],
)
def test_export_rejects_existing_or_unsafe_destinations(
    tmp_path: Path, repository_name: str, zip_name: str, message: str
) -> None:
    source = _source_repository(tmp_path)
    repository = tmp_path / repository_name
    archive = tmp_path / zip_name
    if repository_name == "existing":
        repository.mkdir()

    with pytest.raises(PublicExportError, match=message):
        export_public_snapshot(source, repository, archive)


def test_export_rejects_destinations_inside_the_source(tmp_path: Path) -> None:
    source = _source_repository(tmp_path)
    with pytest.raises(PublicExportError, match="outside the source"):
        export_public_snapshot(
            source,
            source / "public-copy",
            tmp_path / "public.zip",
        )


def test_publish_failure_rolls_back_the_first_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = _source_repository(tmp_path)
    repository = (tmp_path / "public").resolve()
    archive = (tmp_path / "public.zip").resolve()
    original_rename = exporter.os.rename

    def fail_repository_publish(source_path: os.PathLike[str], target_path: os.PathLike[str]) -> None:
        if Path(target_path) == repository:
            raise OSError("simulated repository publish failure")
        original_rename(source_path, target_path)

    monkeypatch.setattr(exporter.os, "rename", fail_repository_publish)
    with pytest.raises(PublicExportError, match="simulated repository publish failure"):
        export_public_snapshot(source, repository, archive)

    assert not repository.exists()
    assert not archive.exists()
    _assert_no_staging(tmp_path)


@pytest.mark.parametrize(
    ("relative", "content", "message"),
    [
        (
            "credentials.txt",
            b"OPENAI_API_KEY=" + b"sk-" + b"abcdefghijklmnopqrstuvwxyz1234\n",
            "secret",
        ),
        (".env", b"SAFE=value\n", "environment"),
        (
            "notes.txt",
            b"C:" + b"\\Users\\private-person\\secret.txt\n",
            "absolute path",
        ),
        ("unknown.bin", b"\x00\x01\x02", "Unknown tracked binary"),
    ],
)
def test_builtin_scan_rejects_secrets_env_paths_and_unknown_binaries(
    tmp_path: Path, relative: str, content: bytes, message: str
) -> None:
    source = _source_repository(tmp_path)
    _write(source / relative, content)
    _run(source, "add", relative)
    _run(
        source,
        "-c",
        "user.name=Private Developer",
        "-c",
        "user.email=private@example.invalid",
        "commit",
        "--message=unsafe fixture",
    )

    with pytest.raises(PublicExportError, match=message):
        export_public_snapshot(source, tmp_path / "public", tmp_path / "public.zip")


def test_existing_zip_is_never_overwritten(tmp_path: Path) -> None:
    source = _source_repository(tmp_path)
    archive = tmp_path / "public.zip"
    archive.write_bytes(b"keep me")

    with pytest.raises(PublicExportError, match="already exists"):
        export_public_snapshot(source, tmp_path / "public", archive)

    assert archive.read_bytes() == b"keep me"


def test_export_enforces_the_download_zip_release_budget(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = _source_repository(tmp_path)
    repository = tmp_path / "public"
    archive = tmp_path / "public.zip"
    monkeypatch.setattr(exporter, "_MAX_EXPORT_ZIP_BYTES", 1)

    with pytest.raises(PublicExportError, match="30 MiB release budget"):
        export_public_snapshot(source, repository, archive)

    assert not repository.exists()
    assert not archive.exists()
    _assert_no_staging(tmp_path)
