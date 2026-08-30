from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Callable

import numpy as np
import pytest

from socialgraph_gfm.canonical import canonical_sha256
from socialgraph_gfm.errors import ContractViolation
from socialgraph_gfm.gfm.corpus import wikimedia
from socialgraph_gfm.gfm.corpus.common import atomic_write_json
from socialgraph_gfm.gfm.corpus.domains import load_domain


def _downloader(payload: bytes) -> Callable[[str, Path], None]:
    def download(_url: str, path: Path) -> None:
        path.write_bytes(payload)

    return download


def _metadata_for_years(
    payload: bytes, years: tuple[int, ...]
) -> dict[str, object]:
    return {
        "id": wikimedia.ARTICLE_ID,
        "license": {"name": "CC0"},
        "files": [
            {
                "id": wikimedia.EXPECTED_FILES[year]["id"],
                "name": f"comments_article_{year}.tar.gz",
                "size": len(payload),
                "computed_md5": hashlib.md5(payload, usedforsecurity=False).hexdigest(),  # noqa: S324
                "download_url": (
                    "https://ndownloader.figshare.com/files/"
                    f"{wikimedia.EXPECTED_FILES[year]['id']}"
                ),
            }
            for year in years
        ],
    }


def _metadata(payload: bytes) -> dict[str, object]:
    return _metadata_for_years(payload, (2011,))


def test_wikimedia_fetch_canonicalizes_cc0_and_hashes_file(tmp_path: Path) -> None:
    payload = b"small mocked official archive"

    def downloader(url: str, path: Path) -> None:
        path.write_bytes(payload)

    receipt = wikimedia.fetch_wikimedia(
        tmp_path,
        accept_license="CC0",
        years=[2011],
        metadata_client=lambda: _metadata(payload),
        downloader=downloader,
        enforce_fixed_metadata=False,
    )
    assert receipt["licenseId"] == "CC0-1.0"
    assert receipt["files"][0]["sha256"] == hashlib.sha256(payload).hexdigest()
    assert receipt["formalEligible"] is False


def test_wikimedia_invalid_download_never_publishes_file_or_receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = b"expected-complete-payload"
    monkeypatch.setitem(
        wikimedia.EXPECTED_FILES,
        2011,
        {
            **wikimedia.EXPECTED_FILES[2011],
            "size": len(payload),
            "md5": hashlib.md5(payload, usedforsecurity=False).hexdigest(),  # noqa: S324
        },
    )
    with pytest.raises(ContractViolation, match="checksum/size mismatch"):
        wikimedia.fetch_wikimedia(
            tmp_path,
            accept_license="CC0",
            years=[2011],
            metadata_client=lambda: _metadata(payload),
            downloader=lambda _url, path: path.write_bytes(b"truncated"),
            enforce_fixed_metadata=True,
        )
    raw = tmp_path / "datasets/raw/gfm/wikimedia-talk"
    assert not (raw / "comments_article_2011.tar.gz").exists()
    assert not (raw / "fetch-receipt.json").exists()
    assert not list(raw.glob(".comments_article_2011.tar.gz.*.tmp"))


def test_wikimedia_existing_file_is_revalidated_and_not_overwritten(
    tmp_path: Path,
) -> None:
    payload = b"stable-fixture"
    calls = 0

    def downloader(_url: str, path: Path) -> None:
        nonlocal calls
        calls += 1
        path.write_bytes(payload)

    first = wikimedia.fetch_wikimedia(
        tmp_path,
        accept_license="CC0",
        years=[2011],
        metadata_client=lambda: _metadata(payload),
        downloader=downloader,
        enforce_fixed_metadata=False,
    )
    source = tmp_path / "datasets/raw/gfm/wikimedia-talk/comments_article_2011.tar.gz"
    before = source.read_bytes()
    second = wikimedia.fetch_wikimedia(
        tmp_path,
        accept_license="CC0",
        years=[2011],
        metadata_client=lambda: _metadata(payload),
        downloader=downloader,
        enforce_fixed_metadata=False,
    )
    assert calls == 1
    assert source.read_bytes() == before
    assert second["files"][0]["sha256"] == first["files"][0]["sha256"]


def test_two_pass_sampler_excludes_anonymous_bot_and_preserves_whole_pages(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("SOCIALGRAPH_GFM_PSEUDONYM_SALT", "fixture-secret-not-persisted")
    payload = b"fixture"

    def downloader(url: str, path: Path) -> None:
        path.write_bytes(payload)

    receipt = wikimedia.fetch_wikimedia(
        tmp_path,
        accept_license="CC0-1.0",
        years=[2011],
        metadata_client=lambda: _metadata(payload),
        downloader=downloader,
        enforce_fixed_metadata=False,
    )
    raw = tmp_path / "datasets/raw/gfm/wikimedia-talk"
    source = raw / receipt["files"][0]["name"]
    rows = [
        {
            "rev_id": "1",
            "comment": "visible text one",
            "raw_comment": "secret raw",
            "timestamp": "2011-01-01T00:00:00Z",
            "page_id": "10",
            "page_title": "Sensitive title",
            # The official nullable user column renders registered integral
            # identifiers with a .0 suffix.
            "user_id": "7.0",
            "user_text": "RealUsername",
            "bot": "0",
            "admin": "0",
        },
        {
            "rev_id": "2",
            "comment": "visible text two",
            "raw_comment": "",
            "timestamp": "2011-01-02T00:00:00Z",
            "page_id": "10",
            "page_title": "Sensitive title",
            "user_id": "8.0",
            "user_text": "OtherUsername",
            "bot": "0",
            "admin": "0",
        },
        {
            "rev_id": "3",
            "comment": "IP contribution",
            "raw_comment": "",
            "timestamp": "2011-01-03T00:00:00Z",
            "page_id": "11",
            "page_title": "Anonymous",
            # Empty means an anonymous/IP edit in the official article dump.
            "user_id": "",
            "user_text": "192.0.2.1",
            "bot": "0",
            "admin": "0",
        },
        {
            "rev_id": "4",
            "comment": "Bot contribution",
            "raw_comment": "",
            "timestamp": "2011-01-04T00:00:00Z",
            "page_id": "12",
            "page_title": "Bot",
            "user_id": "9",
            "user_text": "HelperBot",
            "bot": "1",
            "admin": "0",
        },
    ]
    manifest = wikimedia.prepare_wikimedia(
        [source], tmp_path, max_comments=2, row_source=lambda _: list(rows)
    )
    assert manifest["eventCount"] == 2
    assert manifest["pageCount"] == 1
    assert manifest["nodeOffsets"] == {"user": 0, "page": 2}
    assert manifest["fullPageHistories"] is True
    assert manifest["privacy"]["anonymousExcluded"] == 1
    assert manifest["privacy"]["botsExcluded"] == 1
    combined = "".join(path.read_text(errors="ignore") for path in (tmp_path / "datasets/processed/gfm/wikimedia-talk-article-2011-2015").glob("*.json*"))
    assert "RealUsername" not in combined
    assert "192.0.2.1" not in combined
    assert "Sensitive title" not in combined
    loaded = load_domain(tmp_path, wikimedia.DOMAIN_ID)["arrays"]
    assert loaded["dst"].min() >= manifest["nodeOffsets"]["page"]
    assert loaded["node_type"].tolist() == [0, 0, 1]
    assert set(loaded["src"].tolist()) == {0, 1}
    assert set(loaded["dst"].tolist()) == {2}
    assert "revision_id" not in loaded
    pseudonyms = loaded["revision_pseudonym"].tolist()
    assert len(set(pseudonyms)) == 2
    assert set(pseudonyms).isdisjoint({1, 2})
    text_rows = [
        json.loads(line)
        for line in (
            tmp_path
            / "datasets/processed/gfm/wikimedia-talk-article-2011-2015/text.jsonl"
        ).read_text(encoding="utf-8").splitlines()
    ]
    assert [int(row["id"]) for row in text_rows] == pseudonyms
    assert manifest["privacy"]["sourceRevisionIdsPersisted"] is False
    assert manifest["privacy"]["sourceUserIdsPersisted"] is False
    assert manifest["privacy"]["sourcePageIdsPersisted"] is False
    assert manifest["privacy"]["ipAddressesPersisted"] is False
    assert manifest["privacy"]["usernamesPersisted"] is False
    assert manifest["privacy"]["publicCheckpointEligible"] is False


def test_wikimedia_user_id_parser_rejects_nonintegral_or_ambiguous_values() -> None:
    assert wikimedia._parse_user_id("") == 0
    assert wikimedia._parse_user_id("42") == 42
    assert wikimedia._parse_user_id("42.0") == 42
    for value in ("42.5", "1e3", "NaN", "-1.0", "42.00"):
        with pytest.raises(ContractViolation, match="invalid user ID"):
            wikimedia._parse_user_id(value)


def test_wikimedia_fixed_metadata_drift_fails_closed(tmp_path: Path) -> None:
    metadata = _metadata(b"fixture")
    metadata["files"][0]["computed_md5"] = "0" * 32  # type: ignore[index]
    with pytest.raises(ContractViolation, match="metadata drifted"):
        wikimedia.fetch_wikimedia(
            tmp_path,
            accept_license="CC0",
            years=[2011],
            metadata_client=lambda: metadata,
            downloader=_downloader(b"fixture"),
        )


def test_wikimedia_streams_multiple_bounded_shards_in_global_order(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(
        "SOCIALGRAPH_GFM_PSEUDONYM_SALT", "multi-shard-secret-not-persisted"
    )
    payload = b"multi-shard-fixture"

    receipt = wikimedia.fetch_wikimedia(
        tmp_path,
        accept_license="CC0",
        years=[2011],
        metadata_client=lambda: _metadata(payload),
        downloader=_downloader(payload),
        enforce_fixed_metadata=False,
    )
    source = (
        tmp_path
        / "datasets/raw/gfm/wikimedia-talk"
        / receipt["files"][0]["name"]
    )

    def row(
        revision: int, page: int, user: int, timestamp: str
    ) -> dict[str, str]:
        return {
            "rev_id": str(revision),
            "comment": f"cleaned text {revision}",
            "raw_comment": "must never be copied",
            "timestamp": timestamp,
            "page_id": str(page),
            "page_title": f"private title {page}",
            "user_id": str(user),
            "user_text": f"private user {user}",
            "bot": "0",
            "admin": "0",
        }

    # Deliberately unsorted, including a timestamp tie.  Both complete page
    # histories fit the cap and therefore must be preserved in external order.
    rows = [
        row(9, 20, 100, "2011-01-04T00:00:00Z"),
        row(2, 10, 101, "2011-01-01T00:00:00Z"),
        row(4, 20, 102, "2011-01-02T00:00:00Z"),
        row(3, 10, 103, "2011-01-02T00:00:00Z"),
        row(5, 10, 104, "2011-01-03T00:00:00Z"),
    ]
    manifest = wikimedia.prepare_wikimedia(
        [source],
        tmp_path,
        max_comments=5,
        event_rows_per_shard=2,
        row_source=lambda _path: iter(rows),
    )
    output = (
        tmp_path
        / "datasets/processed/gfm/wikimedia-talk-article-2011-2015"
    )
    event_records = [
        item for item in manifest["shards"] if item["path"].startswith("events-")
    ]
    node_records = [
        item for item in manifest["shards"] if item["path"].startswith("nodes-")
    ]
    assert [item["path"] for item in event_records] == [
        "events-00000.npz",
        "events-00001.npz",
        "events-00002.npz",
    ]
    assert [item["rows"] for item in event_records] == [2, 2, 1]
    assert len(node_records) > 1
    assert max(item["rows"] for item in manifest["shards"] if item["arrays"]) <= 2

    timestamps: list[int] = []
    pseudonyms: list[int] = []
    for item in event_records:
        with np.load(output / item["path"], allow_pickle=False) as arrays:
            timestamps.extend(int(value) for value in arrays["timestamp"])
            pseudonyms.extend(int(value) for value in arrays["revision_pseudonym"])
    assert timestamps == sorted(timestamps)
    order_keys = [
        (timestamp, pseudonym.to_bytes(8, "little"))
        for timestamp, pseudonym in zip(timestamps, pseudonyms, strict=True)
    ]
    assert order_keys == sorted(order_keys)
    text_rows = [
        json.loads(line)
        for line in (output / "text.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert [row["timestamp"] for row in text_rows] == timestamps
    assert [int(row["id"]) for row in text_rows] == pseudonyms
    loaded = load_domain(tmp_path, wikimedia.DOMAIN_ID)["arrays"]
    assert loaded["src"].shape == (5,)
    assert loaded["timestamp"].tolist() == timestamps
    assert loaded["revision_pseudonym"].tolist() == pseudonyms
    assert loaded["node_type"].shape == (manifest["nodeCount"],)
    assert not (output / "spool.sqlite3").exists()
    assert not list(output.parent.glob(f".{wikimedia.CORPUS_ID}.*.staging"))
    assert wikimedia.check_wikimedia(tmp_path)["logicalHash"] == manifest["logicalHash"]


def test_cross_year_page_has_one_split_from_its_last_event(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(
        "SOCIALGRAPH_GFM_PSEUDONYM_SALT", "page-split-secret-not-persisted"
    )
    payload = b"cross-year-page-fixture"
    years = (2011, 2012, 2014, 2015)
    receipt = wikimedia.fetch_wikimedia(
        tmp_path,
        accept_license="CC0",
        years=years,
        metadata_client=lambda: _metadata_for_years(payload, years),
        downloader=_downloader(payload),
        enforce_fixed_metadata=False,
    )
    raw = tmp_path / "datasets/raw/gfm/wikimedia-talk"
    sources = [raw / str(item["name"]) for item in receipt["files"]]

    def row(revision: int, page: int, user: int, timestamp: str) -> dict[str, str]:
        return {
            "rev_id": str(revision),
            "comment": f"cleaned text {revision}",
            "raw_comment": "must never be copied",
            "timestamp": timestamp,
            "page_id": str(page),
            "page_title": f"private title {page}",
            "user_id": str(user),
            "user_text": f"private user {user}",
            "bot": "0",
            "admin": "0",
        }

    rows_by_year = {
        2011: [row(1, 20, 101, "2011-01-01T00:00:00Z")],
        2012: [row(2, 10, 102, "2012-01-01T00:00:00Z")],
        2014: [
            row(3, 10, 103, "2014-06-01T00:00:00Z"),
            row(4, 30, 104, "2014-07-01T00:00:00Z"),
        ],
        2015: [row(5, 40, 105, "2015-01-01T00:00:00Z")],
    }

    def source_rows(path: Path) -> list[dict[str, str]]:
        year = int(path.name.removeprefix("comments_article_").removesuffix(".tar.gz"))
        return rows_by_year[year]

    manifest = wikimedia.prepare_wikimedia(
        sources,
        tmp_path,
        max_comments=5,
        event_rows_per_shard=2,
        row_source=source_rows,
    )
    loaded = load_domain(tmp_path, wikimedia.DOMAIN_ID)["arrays"]
    destinations = loaded["dst"].tolist()
    splits = loaded["split"].tolist()
    timestamps = loaded["timestamp"].tolist()
    page_splits: dict[int, set[int]] = {}
    for page, split in zip(destinations, splits, strict=True):
        page_splits.setdefault(int(page), set()).add(int(split))
    assert all(len(values) == 1 for values in page_splits.values())

    cross_year_page = destinations[timestamps.index(1325376000)]
    cross_year_indices = [
        index for index, page in enumerate(destinations) if page == cross_year_page
    ]
    assert [splits[index] for index in cross_year_indices] == [1, 1]
    assert manifest["splits"]["strategy"] == "page-last-event-time"
    assert manifest["splits"]["pageDisjoint"] is True
    assert manifest["splits"]["counts"] == {
        "events": {"train": 1, "validation": 3, "test": 1},
        "pages": {"train": 1, "validation": 2, "test": 1},
    }
    assert wikimedia.check_wikimedia(tmp_path)["logicalHash"] == manifest["logicalHash"]


def test_load_domain_rejects_resigned_page_split_semantic_forgery(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(
        "SOCIALGRAPH_GFM_PSEUDONYM_SALT", "resigned-split-forgery-secret"
    )
    payload = b"resigned-split-forgery"
    years = (2011, 2014, 2015)
    receipt = wikimedia.fetch_wikimedia(
        tmp_path,
        accept_license="CC0",
        years=years,
        metadata_client=lambda: _metadata_for_years(payload, years),
        downloader=_downloader(payload),
        enforce_fixed_metadata=False,
    )
    raw = tmp_path / "datasets/raw/gfm/wikimedia-talk"
    sources = [raw / str(item["name"]) for item in receipt["files"]]

    def source_rows(path: Path) -> list[dict[str, str]]:
        year = int(path.name.removeprefix("comments_article_").removesuffix(".tar.gz"))
        return [
            {
                "rev_id": str(year),
                "comment": f"cleaned {year}",
                "raw_comment": "forbidden",
                "timestamp": f"{year}-06-01T00:00:00Z",
                "page_id": str(year),
                "page_title": "private",
                "user_id": str(year + 10),
                "user_text": "private",
                "bot": "0",
                "admin": "0",
            }
        ]

    wikimedia.prepare_wikimedia(
        sources,
        tmp_path,
        max_comments=3,
        row_source=source_rows,
    )
    path = (
        tmp_path
        / "datasets/processed/gfm"
        / wikimedia.CORPUS_ID
        / "manifest.json"
    )
    forged = json.loads(path.read_text(encoding="utf-8"))
    forged["splits"]["pageDisjoint"] = False
    logical_payload = {
        key: value
        for key, value in forged.items()
        if key not in {"logicalHash", "createdAt"}
    }
    forged["logicalHash"] = canonical_sha256(logical_payload)
    atomic_write_json(path, forged)
    with pytest.raises(ContractViolation, match="page-disjoint split contract"):
        load_domain(tmp_path, wikimedia.DOMAIN_ID)


def test_wikimedia_failed_second_pass_never_publishes_partial_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(
        "SOCIALGRAPH_GFM_PSEUDONYM_SALT", "atomic-failure-secret-not-persisted"
    )
    payload = b"atomic-fixture"
    receipt = wikimedia.fetch_wikimedia(
        tmp_path,
        accept_license="CC0",
        years=[2011],
        metadata_client=lambda: _metadata(payload),
        downloader=_downloader(payload),
        enforce_fixed_metadata=False,
    )
    source = (
        tmp_path
        / "datasets/raw/gfm/wikimedia-talk"
        / receipt["files"][0]["name"]
    )
    rows = [
        {
            "rev_id": "1",
            "comment": "cleaned text",
            "raw_comment": "raw text",
            "timestamp": "2011-01-01T00:00:00Z",
            "page_id": "10",
            "page_title": "private title",
            "user_id": "7",
            "user_text": "private user",
            "bot": "0",
            "admin": "0",
        }
    ]
    calls = 0

    def source_rows(_path: Path) -> object:
        nonlocal calls
        calls += 1
        if calls == 1:
            return iter(rows)

        def broken_second_pass() -> object:
            yield rows[0]
            raise RuntimeError("simulated source read failure")

        return broken_second_pass()

    with pytest.raises(RuntimeError, match="simulated source read failure"):
        wikimedia.prepare_wikimedia(
            [source], tmp_path, max_comments=1, row_source=source_rows  # type: ignore[arg-type]
        )
    output = (
        tmp_path
        / "datasets/processed/gfm/wikimedia-talk-article-2011-2015"
    )
    assert not output.exists()
    assert not list(output.parent.glob(f".{wikimedia.CORPUS_ID}.*.staging"))
