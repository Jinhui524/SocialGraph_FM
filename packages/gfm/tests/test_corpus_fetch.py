from __future__ import annotations

import zipfile
from pathlib import Path

import pytest

import socialgraph_gfm.corpus_fetch as corpus_fetch
from socialgraph_gfm.corpus_fetch import _safe_extract_archive
from socialgraph_gfm.runtime import RuntimeLayout


def _release_archive(path: Path, *, extra_name: str | None = None) -> None:
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("collab/raw/edge.csv.gz", b"edge")
        archive.writestr("collab/raw/node-feat.csv.gz", b"feature")
        archive.writestr("collab/split/time/train.pt", b"trusted")
        archive.writestr("collab/RELEASE_v1.txt", b"release 1")
        if extra_name is not None:
            archive.writestr(extra_name, b"escape")


def test_safe_extract_builds_expected_ogb_cache(tmp_path: Path) -> None:
    archive = tmp_path / "collab.zip"
    _release_archive(archive)

    target = _safe_extract_archive(archive, tmp_path)

    assert target == tmp_path / "ogbl_collab"
    assert (target / "raw" / "edge.csv.gz").read_bytes() == b"edge"
    with pytest.raises(ValueError, match="refusing to reuse"):
        _safe_extract_archive(archive, tmp_path)


def test_safe_extract_rejects_prepopulated_pickle_cache(tmp_path: Path) -> None:
    archive = tmp_path / "collab.zip"
    _release_archive(archive)
    malicious = tmp_path / "ogbl_collab" / "processed" / "geometric_data_processed.pt"
    malicious.parent.mkdir(parents=True)
    malicious.write_bytes(b"untrusted pickle")

    with pytest.raises(ValueError, match="refusing to reuse"):
        _safe_extract_archive(archive, tmp_path)

    assert malicious.read_bytes() == b"untrusted pickle"


def test_safe_extract_rejects_path_traversal(tmp_path: Path) -> None:
    archive = tmp_path / "collab.zip"
    _release_archive(archive, extra_name="../outside.txt")

    with pytest.raises(ValueError, match="path traversal"):
        _safe_extract_archive(archive, tmp_path)

    assert not (tmp_path.parent / "outside.txt").exists()


def test_fetch_converts_only_inside_an_ephemeral_cache(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: dict[str, Path] = {}

    def fake_download(destination: Path) -> None:
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(b"pinned archive test marker")

    def fake_extract(archive: Path, cache_root: Path) -> Path:
        assert archive == tmp_path / "datasets" / "raw" / "ogb" / "collab.zip"
        observed["cache_root"] = cache_root
        target = cache_root / "ogbl_collab"
        target.mkdir(parents=True)
        return target

    def fake_convert(cache_root: Path, package: Path) -> dict[str, object]:
        assert cache_root == observed["cache_root"]
        package.parent.mkdir(parents=True, exist_ok=True)
        package.write_bytes(b"safe json npz package")
        return {"converted": True}

    monkeypatch.setattr(corpus_fetch, "_download_pinned_archive", fake_download)
    monkeypatch.setattr(corpus_fetch, "_safe_extract_archive", fake_extract)
    monkeypatch.setattr(corpus_fetch, "_convert_with_api", fake_convert)
    layout = RuntimeLayout.from_root(tmp_path)
    layout.prepare()
    monkeypatch.setattr(corpus_fetch, "prepare_runtime_layout", lambda *_args, **_kwargs: layout)
    monkeypatch.setattr(
        corpus_fetch,
        "require_storage_reserve",
        lambda *_args, **_kwargs: {"ready": True},
    )

    result = corpus_fetch.fetch_ogbl_collab(
        tmp_path,
        accept_license=corpus_fetch.OGBL_COLLAB_LICENSE,
    )

    isolated = observed["cache_root"]
    assert result["conversionCache"] == {
        "isolated": True,
        "retained": False,
        "provenance": "pinned-archive-sha256",
    }
    assert not isolated.exists()
    assert not (tmp_path / "datasets" / "raw" / "ogb" / "ogbl_collab").exists()
