from __future__ import annotations

import gzip
import io
import stat
import zipfile
from pathlib import Path

import numpy as np
import pytest
from scipy.io import savemat

from socialgraph_gfm.core.datasets.acquire import (
    download_source,
    extract_source_atomic,
    safe_load_mat_arrays,
    safe_load_numpy_arrays,
)
from socialgraph_gfm.core.datasets.recipes import SourceRecipe


def _source(**overrides: object) -> SourceRecipe:
    payload = {
        "sourceId": "fixture",
        "url": "https://downloads.example.test/data.zip",
        "expectedSha256": None,
        "archiveType": "zip",
        "maxBytes": 1024,
        "inventory": ["graph/edges.csv"],
    }
    payload.update(overrides)
    return SourceRecipe.model_validate(payload)


class _Response(io.BytesIO):
    def __init__(self, payload: bytes, *, url: str, content_length: int | None = None) -> None:
        super().__init__(payload)
        self._url = url
        self.headers = {} if content_length is None else {"Content-Length": str(content_length)}

    def geturl(self) -> str:
        return self._url

    def __enter__(self):
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()


def test_downloader_streams_to_runtime_and_records_observed_hash(tmp_path) -> None:
    payload = b"bounded source bytes"
    source_url = "https://snap.stanford.edu/data/email-Eu-core.txt.gz"

    result = download_source(
        recipe_id="email-eu-core",
        source_id="edges",
        runtime_root=tmp_path,
        open_url=lambda _request, _timeout: _Response(payload, url=source_url),
    )

    assert result.path.read_bytes() == payload
    assert result.path.is_relative_to(tmp_path.resolve())
    assert result.observed_sha256 == "20386285d2d5a2c88ec67a96bcee18f854882bbcb258c74f7c0cecf35e912ef5"
    assert not list(tmp_path.rglob("*.part"))


@pytest.mark.parametrize(
    ("response_url", "content_length", "payload", "message"),
    [
        ("http://snap.stanford.edu/data/email-Eu-core.txt.gz", None, b"ok", "HTTPS"),
        ("https://evil.example/data.zip", None, b"ok", "allowlist"),
        ("https://snap.stanford.edu/data/email-Eu-core.txt.gz", 200001, b"ok", "maximum"),
        (
            "https://snap.stanford.edu/data/email-Eu-core.txt.gz",
            None,
            b"x" * 200001,
            "maximum",
        ),
    ],
    ids=["http-redirect", "host-redirect", "content-length", "stream-size"],
)
def test_downloader_rejects_bad_redirects_and_oversized_responses(
    tmp_path, response_url: str, content_length: int | None, payload: bytes, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        download_source(
            recipe_id="email-eu-core",
            source_id="edges",
            runtime_root=tmp_path,
            open_url=lambda _request, _timeout: _Response(
                payload, url=response_url, content_length=content_length
            ),
        )


def test_downloader_resolves_only_catalog_ids_and_contains_targets(tmp_path) -> None:
    opened = False

    def forbidden_open(_request, _timeout):
        nonlocal opened
        opened = True
        raise AssertionError("unsafe identifier reached the network")

    for recipe_id, source_id in (
        ("../escape", "edges"),
        ("email-eu-core", "../escape"),
        ("not-in-catalog", "edges"),
    ):
        with pytest.raises(ValueError, match="catalog|identifier"):
            download_source(
                recipe_id=recipe_id,
                source_id=source_id,
                runtime_root=tmp_path,
                open_url=forbidden_open,
            )
    assert opened is False
    assert not (tmp_path.parent / "escape").exists()


def test_downloader_refuses_conflicting_existing_raw_file(tmp_path) -> None:
    target = tmp_path / "raw" / "email-eu-core" / "1.0.0" / "email-Eu-core.txt.gz"
    target.parent.mkdir(parents=True)
    target.write_bytes(b"existing different bytes")
    source_url = "https://snap.stanford.edu/data/email-Eu-core.txt.gz"

    with pytest.raises(FileExistsError, match="conflicting"):
        download_source(
            recipe_id="email-eu-core",
            source_id="edges",
            runtime_root=tmp_path,
            open_url=lambda _request, _timeout: _Response(b"new bytes", url=source_url),
        )
    assert target.read_bytes() == b"existing different bytes"


def _write_zip(path: Path, members: list[tuple[zipfile.ZipInfo | str, bytes]]) -> None:
    with zipfile.ZipFile(path, "w") as archive:
        for name, payload in members:
            archive.writestr(name, payload)


@pytest.mark.parametrize("bad_name", ["../escape.txt", "/absolute.txt", "graph\\edge.csv"])
def test_zip_extraction_rejects_path_traversal(tmp_path, bad_name: str) -> None:
    source_path = tmp_path / "bad.zip"
    _write_zip(source_path, [(bad_name, b"bad")])

    with pytest.raises(ValueError, match="path"):
        extract_source_atomic(
            source_path=source_path,
            source=_source(inventory=[bad_name]),
            target_directory=tmp_path / "published",
            max_expanded_bytes=1024,
        )


def test_zip_extraction_rejects_symlinks_duplicates_and_unexpected_inventory(tmp_path) -> None:
    symlink = zipfile.ZipInfo("graph/edges.csv")
    symlink.create_system = 3
    symlink.external_attr = (stat.S_IFLNK | 0o777) << 16
    cases = [
        ([(symlink, b"target")], "symlink"),
        (
            [("graph/edges.csv", b"a"), ("graph/edges.csv", b"b")],
            "duplicate",
        ),
        ([("graph/extra.csv", b"a")], "inventory"),
    ]
    for index, (members, message) in enumerate(cases):
        source_path = tmp_path / f"bad-{index}.zip"
        if message == "duplicate":
            with pytest.warns(UserWarning, match="Duplicate name"):
                _write_zip(source_path, members)
        else:
            _write_zip(source_path, members)
        with pytest.raises(ValueError, match=message):
            extract_source_atomic(
                source_path=source_path,
                source=_source(),
                target_directory=tmp_path / f"published-{index}",
                max_expanded_bytes=1024,
            )


def test_gzip_extraction_is_bounded_and_atomically_published(tmp_path) -> None:
    source_path = tmp_path / "data.gz"
    with gzip.open(source_path, "wb") as stream:
        stream.write(b"edge\n" * 20)
    source = _source(
        archiveType="gzip", inventory=["graph.txt"], expectedSha256=None, maxBytes=1024
    )

    with pytest.raises(ValueError, match="expanded"):
        extract_source_atomic(
            source_path=source_path,
            source=source,
            target_directory=tmp_path / "too-large",
            max_expanded_bytes=25,
        )
    target = extract_source_atomic(
        source_path=source_path,
        source=source,
        target_directory=tmp_path / "published",
        max_expanded_bytes=1024,
    )
    assert (target / "graph.txt").read_bytes() == b"edge\n" * 20


def test_numpy_loader_rejects_pickle_objects_and_unexpected_keys(tmp_path) -> None:
    object_path = tmp_path / "object.npy"
    np.save(object_path, np.array([{"unsafe": True}], dtype=object))
    with pytest.raises(ValueError, match="pickle|object"):
        safe_load_numpy_arrays(
            object_path,
            expected_keys=None,
            max_array_elements=10,
            max_total_array_bytes=1024,
        )

    archive_path = tmp_path / "arrays.npz"
    np.savez(archive_path, edges=np.array([[0, 1]]), surprise=np.array([1]))
    with pytest.raises(ValueError, match="inventory"):
        safe_load_numpy_arrays(
            archive_path,
            expected_keys={"edges"},
            max_array_elements=10,
            max_total_array_bytes=1024,
        )


def test_numpy_preflight_rejects_compressed_huge_shape_before_loading(tmp_path) -> None:
    header = io.BytesIO()
    np.lib.format.write_array_header_1_0(
        header,
        {"descr": np.dtype("<i8").str, "fortran_order": False, "shape": (1_000_000_000,)},
    )
    archive_path = tmp_path / "huge.npz"
    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("edges.npy", header.getvalue())

    with pytest.raises(ValueError, match="element|uncompressed|allocation"):
        safe_load_numpy_arrays(
            archive_path,
            expected_keys={"edges"},
            max_array_elements=100,
            max_total_array_bytes=1024,
        )


def test_compressed_mat_is_parsed_in_bounded_worker(tmp_path) -> None:
    mat_path = tmp_path / "compressed.mat"
    savemat(
        mat_path,
        {"A": np.zeros((1000, 1000), dtype=np.uint8), "local_info": np.zeros((1000, 7))},
        do_compression=True,
    )

    with pytest.raises(ValueError, match="element limit"):
        safe_load_mat_arrays(
            mat_path,
            expected_keys={"A", "local_info"},
            max_array_elements=100,
            max_worker_memory_bytes=2 * 1024 * 1024 * 1024,
            timeout_seconds=30,
        )
