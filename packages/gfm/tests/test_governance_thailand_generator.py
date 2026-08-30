from __future__ import annotations

import csv
import hashlib
import inspect
import io
import json
import re
import shutil
import subprocess
import zipfile
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from socialgraph_gfm.governance import thailand as thailand_module
from socialgraph_gfm.governance.cli import main
from socialgraph_gfm.governance.thailand import (
    PINNED_ENCODER_MODEL_ID,
    PINNED_ENCODER_REVISION,
    SourceValidationError,
    aggregate_account_content,
    anonymized_node_id,
    generate_thailand_package,
)


class DeterministicEncoder:
    model_id = "fixture/deterministic-encoder"
    revision = "fixture-v1"
    cache_sha256 = "1" * 64

    def encode(self, texts: list[str]) -> np.ndarray:
        result = np.zeros((len(texts), 768), dtype=np.float32)
        for row, text in enumerate(texts):
            match = re.search(r"cluster-(\d+)", text)
            bucket = int(match.group(1)) if match else 767
            result[row, bucket % 767] = 1.0
            result[row, 767] = 0.01
            result[row] /= np.linalg.norm(result[row])
        return result


def _post(
    *,
    postid: str,
    accountid: str,
    text: str,
    at: datetime,
    is_control: bool,
    reposted_postid: str | None = None,
    urls: list[str] | None = None,
    hashtags: list[str] | None = None,
) -> dict[str, Any]:
    return {
        "postid": postid,
        "post_text": text,
        "post_time": at.isoformat().replace("+00:00", "Z"),
        "accountid": accountid,
        "is_repost": reposted_postid is not None,
        "reposted_accountid": "origin" if reposted_postid is not None else None,
        "reposted_postid": reposted_postid,
        "hashtags": hashtags or [],
        "urls": urls or [],
        "account_mentions": [],
        "in_reply_to_accountid": None,
        "is_control": is_control,
    }


def _write_authorized_source(
    runtime: Path,
    posts: list[dict[str, Any]],
    *,
    source_schema: str = "socialgraph-fm.anonymized-posts/1.0",
) -> Path:
    runtime.mkdir(parents=True, exist_ok=True)
    (runtime / "runtime-authorization.json").write_text(
        json.dumps(
            {
                "schemaVersion": "socialgraph-fm.defense-runtime-authorization/1.0",
                "purpose": "authorized-anonymized-defense-data",
            }
        ),
        encoding="utf-8",
    )
    source = runtime / "sources" / "thailand-authorized"
    source.mkdir(parents=True)
    payload = b"".join(
        json.dumps(row, sort_keys=True, separators=(",", ":")).encode("utf-8") + b"\n"
        for row in posts
    )
    (source / "posts.jsonl").write_bytes(payload)
    (source / "authorization.json").write_text(
        json.dumps(
            {
                "schemaVersion": "socialgraph-fm.authorized-source/1.0",
                "datasetId": "authorized-thailand-fixture",
                "country": "TH",
                "sourceSchemaVersion": source_schema,
                "sourceFile": "posts.jsonl",
                "sourceSha256": hashlib.sha256(payload).hexdigest(),
                "authorizationReference": "fixture-approval-2026-08-20",
                "license": "Fixture-only synthetic data",
                "approvedAt": "2026-08-20T00:00:00Z",
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return source


def _coverage_posts(count: int = 128) -> list[dict[str, Any]]:
    base = datetime(2026, 8, 20, tzinfo=UTC)
    posts: list[dict[str, Any]] = []
    for index in range(count):
        account = f"account-{index:03d}"
        control = index >= 32
        cluster = index // 2
        for content_index in range(6):
            posts.append(
                _post(
                    postid=f"original-{index:03d}-{content_index}",
                    accountid=account,
                    text=f"cluster-{cluster} original-{content_index} account-{index:03d}",
                    at=base + timedelta(minutes=index, seconds=content_index),
                    is_control=control,
                )
            )
        posts.append(
            _post(
                postid=f"repost-{index:03d}",
                accountid=account,
                text=f"cluster-{cluster} shared repost",
                at=base + timedelta(hours=4, seconds=cluster * 20 + index % 2 * 5),
                is_control=control,
                reposted_postid=f"shared-{cluster}",
                urls=[
                    "https://example.invalid/authorized-fixture",
                    f"https://example.invalid/pair/{cluster}",
                ],
                hashtags=["Defense", f"Pair{cluster}"],
            )
        )
    return posts


def _read_bundle(path: Path) -> tuple[dict[str, Any], list[dict[str, str]], np.ndarray]:
    with zipfile.ZipFile(path) as archive:
        assert archive.namelist() == [
            "manifest.json",
            "nodes.csv",
            "relations.csv",
            "features.npz",
        ]
        manifest = json.loads(archive.read("manifest.json"))
        relations = list(csv.DictReader(io.StringIO(archive.read("relations.csv").decode())))
        with np.load(io.BytesIO(archive.read("features.npz")), allow_pickle=False) as values:
            features = np.asarray(values["text_features"])
    return manifest, relations, features


def test_account_content_uses_top_five_observed_repost_counts_with_hash_ties() -> None:
    base = datetime(2026, 8, 20, tzinfo=UTC)
    posts = [
        _post(
            postid=f"owned-{index}",
            accountid="owner",
            text=f"content-{index}",
            at=base + timedelta(seconds=index),
            is_control=False,
        )
        for index in range(6)
    ]
    posts.extend(
        _post(
            postid=f"tie-{index}",
            accountid="tie-owner",
            text=f"tie-content-{index}",
            at=base + timedelta(seconds=30 + index),
            is_control=False,
        )
        for index in range(6)
    )
    for original_index in range(6):
        for repeat in range(original_index):
            posts.append(
                _post(
                    postid=f"external-{original_index}-{repeat}",
                    accountid=f"reposter-{original_index}-{repeat}",
                    text="repost",
                    at=base + timedelta(minutes=original_index, seconds=repeat),
                    is_control=True,
                    reposted_postid=f"owned-{original_index}",
                )
            )

    content = aggregate_account_content(posts)

    assert content["owner"].split("\n") == [
        "content-5",
        "content-4",
        "content-3",
        "content-2",
        "content-1",
    ]
    assert content["tie-owner"].split("\n") == [
        "tie-content-3",
        "tie-content-0",
        "tie-content-5",
        "tie-content-2",
        "tie-content-1",
    ]


def test_generator_builds_reproducible_contract_complete_128_node_package(tmp_path: Path) -> None:
    runtime = tmp_path / "var" / "defense"
    source = _write_authorized_source(runtime, _coverage_posts())
    first = thailand_module._generate_thailand_package_with_encoder(
        source,
        runtime,
        runtime / "derived" / "thailand-first.zip",
        encoder=DeterministicEncoder(),
    )
    second = thailand_module._generate_thailand_package_with_encoder(
        source,
        runtime,
        runtime / "derived" / "thailand-second.zip",
        encoder=DeterministicEncoder(),
    )

    assert first.bundle_path.read_bytes() == second.bundle_path.read_bytes()
    manifest, relations, features = _read_bundle(first.bundle_path)
    assert manifest["nodeCount"] == 128
    assert set(manifest["modalities"]) == {"coRT", "coURL", "hashSeq", "fastRT", "tweetSim"}
    assert features.shape == (128, 768)
    assert features.dtype == np.float32
    assert bool(np.isfinite(features).all())
    assert {row["modality"] for row in relations} == {
        "coRT",
        "coURL",
        "hashSeq",
        "fastRT",
        "tweetSim",
    }
    account_zero = anonymized_node_id("account-000")
    account_one = anonymized_node_id("account-001")
    account_two = anonymized_node_id("account-002")
    pair_zero_one = {
        row["modality"]
        for row in relations
        if {row["source"], row["target"]} == {account_zero, account_one}
    }
    pair_zero_two = {
        row["modality"]
        for row in relations
        if {row["source"], row["target"]} == {account_zero, account_two}
    }
    assert pair_zero_one == {"coRT", "coURL", "hashSeq", "fastRT", "tweetSim"}
    assert pair_zero_two == {"coURL"}

    label_document = json.loads(first.labels_path.read_text(encoding="utf-8"))
    labels = label_document["labels"]
    assert len(labels) == 16
    assert sum(item["label"] == "io" for item in labels) == 8
    assert sum(item["label"] == "control" for item in labels) == 8
    assert "Global" not in json.dumps(label_document)
    assert label_document["selectionRecipe"]["stratification"] == "graph-fused-degree-rank-quartile"

    receipt = json.loads(first.receipt_path.read_text(encoding="utf-8"))
    assert receipt["sourceSha256"] == hashlib.sha256(
        (source / "posts.jsonl").read_bytes()
    ).hexdigest()
    assert receipt["encoder"] == {
        "modelId": "fixture/deterministic-encoder",
        "revision": "fixture-v1",
        "cacheSha256": "1" * 64,
        "compatibility": "dimension-only-unverified",
        "dimension": 768,
    }
    assert receipt["selectionRecipe"]["requiredIo"] == 16
    assert receipt["selectionRecipe"]["requiredControls"] == 64
    assert receipt["selectionRecipe"]["minimumNonemptyModalities"] == 4
    assert receipt["selectionRecipe"]["groupRelations"] == {
        "maxGroupAccounts": 256,
        "totalPotentialPairBudget": 50_000,
    }
    assert receipt["selectionRecipe"]["fastRT"] == {
        "windowSeconds": 10,
        "pairBudget": 50_000,
        "algorithm": "sorted-sliding-window-v1",
    }
    assert receipt["coverage"]["ioCount"] >= 16
    assert receipt["coverage"]["controlCount"] >= 64
    assert receipt["coverage"]["connected"] is True

    node_ids = {item["nodeId"] for item in labels}
    related = {row["source"] for row in relations} | {row["target"] for row in relations}
    assert node_ids <= related
    assert receipt["schemaVersion"] == "socialgraph-fm.governance-target-package-receipt/1.1"
    assert label_document["schemaVersion"] == "socialgraph-fm.governance-target-label-recipe/1.1"
    assert receipt["labelsSha256"] == hashlib.sha256(first.labels_path.read_bytes()).hexdigest()
    logical_receipt = {key: value for key, value in receipt.items() if key != "receiptHash"}
    assert receipt["receiptHash"] == thailand_module.canonical_sha256(logical_receipt)
    for label in ("io", "control"):
        assert {
            stratum: sum(
                item["label"] == label and item["structuralStratum"] == stratum
                for item in labels
            )
            for stratum in range(4)
        } == {0: 2, 1: 2, 2: 2, 3: 2}


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("missing-authorization", "authorization metadata"),
        ("unsupported-schema", "source schema"),
        ("hash-mismatch", "source hash"),
        ("insufficient-coverage", "128 nodes"),
    ],
)
def test_generator_fails_closed_for_untrusted_or_insufficient_sources(
    tmp_path: Path, mutation: str, message: str
) -> None:
    runtime = tmp_path / mutation / "var" / "defense"
    posts = _coverage_posts(127 if mutation == "insufficient-coverage" else 128)
    source = _write_authorized_source(
        runtime,
        posts,
        source_schema=(
            "socialgraph-fm.unsupported/9.9"
            if mutation == "unsupported-schema"
            else "socialgraph-fm.anonymized-posts/1.0"
        ),
    )
    if mutation == "missing-authorization":
        (source / "authorization.json").unlink()
    elif mutation == "hash-mismatch":
        authorization = json.loads((source / "authorization.json").read_text(encoding="utf-8"))
        authorization["sourceSha256"] = "0" * 64
        (source / "authorization.json").write_text(json.dumps(authorization), encoding="utf-8")

    with pytest.raises(SourceValidationError, match=message):
        thailand_module._generate_thailand_package_with_encoder(
            source,
            runtime,
            runtime / "derived" / "target.zip",
            encoder=DeterministicEncoder(),
        )


def test_generator_rejects_output_or_source_outside_authorized_runtime(tmp_path: Path) -> None:
    runtime = tmp_path / "var" / "defense"
    source = _write_authorized_source(runtime, _coverage_posts())
    with pytest.raises(SourceValidationError, match="output.*runtime root"):
        thailand_module._generate_thailand_package_with_encoder(
            source,
            runtime,
            tmp_path / "outside.zip",
            encoder=DeterministicEncoder(),
        )

    outside_source = tmp_path / "outside-source"
    outside_source.mkdir()
    for name in ("authorization.json", "posts.jsonl"):
        (outside_source / name).write_bytes((source / name).read_bytes())
    with pytest.raises(SourceValidationError, match="source.*runtime root"):
        thailand_module._generate_thailand_package_with_encoder(
            outside_source,
            runtime,
            runtime / "derived" / "target.zip",
            encoder=DeterministicEncoder(),
        )


def test_generator_refuses_an_ordinary_runtime_even_with_a_copied_marker(tmp_path: Path) -> None:
    ordinary_runtime = tmp_path / "var" / "gfm"
    source = _write_authorized_source(ordinary_runtime, _coverage_posts())
    with pytest.raises(SourceValidationError, match="var/defense"):
        thailand_module._generate_thailand_package_with_encoder(
            source,
            ordinary_runtime,
            ordinary_runtime / "derived" / "target.zip",
            encoder=DeterministicEncoder(),
        )


@pytest.mark.parametrize(
    "mutate",
    [
        lambda row: row.update(accountid=7),
        lambda row: row.update(post_text="x" * 20_001),
        lambda row: row.update(urls="https://example.invalid"),
        lambda row: row.update(Global=0.99),
    ],
)
def test_source_rows_validate_types_limits_and_forbid_model_scores(
    tmp_path: Path, mutate: Any
) -> None:
    runtime = tmp_path / "row-validation" / "var" / "defense"
    posts = _coverage_posts()
    mutate(posts[0])
    source = _write_authorized_source(runtime, posts)

    with pytest.raises(SourceValidationError, match="source row"):
        thailand_module._generate_thailand_package_with_encoder(
            source,
            runtime,
            runtime / "derived" / "target.zip",
            encoder=DeterministicEncoder(),
        )


def test_production_encoder_is_pinned_offline_and_fails_before_download(tmp_path: Path) -> None:
    runtime = tmp_path / "var" / "defense"
    source = _write_authorized_source(runtime, _coverage_posts())
    with pytest.raises(SourceValidationError, match="offline encoder cache"):
        generate_thailand_package(
            source,
            runtime,
            runtime / "derived" / "target.zip",
            encoder_cache=runtime / "missing-model-cache",
        )
    assert PINNED_ENCODER_MODEL_ID == (
        "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
    )
    assert PINNED_ENCODER_REVISION == "4328cf26390c98c5e3c738b4460a05b95f4911f5"


def test_anonymized_ids_are_stable_and_do_not_expose_source_ids() -> None:
    first = anonymized_node_id("sensitive-account-id")
    assert first == anonymized_node_id("sensitive-account-id")
    assert first.startswith("th:")
    assert "sensitive" not in first


def test_cli_requires_the_explicit_pinned_offline_cache(tmp_path: Path) -> None:
    runtime = tmp_path / "cli" / "var" / "defense"
    source = _write_authorized_source(runtime, _coverage_posts())
    with pytest.raises(SourceValidationError, match="offline encoder cache"):
        main(
            [
                "thailand-package",
                "--source-directory",
                str(source),
                "--runtime-root",
                str(runtime),
                "--output",
                str(runtime / "derived" / "target.zip"),
                "--encoder-cache",
                str(runtime / "missing-model-cache"),
            ]
        )


def test_public_generator_does_not_expose_encoder_or_provenance_injection(tmp_path: Path) -> None:
    parameters = inspect.signature(generate_thailand_package).parameters
    assert {"encoder", "model_id", "revision", "cache_sha256"}.isdisjoint(parameters)
    assert parameters["encoder_cache"].default is inspect.Parameter.empty
    assert "_generate_thailand_package_with_encoder" not in thailand_module.__all__
    runtime = tmp_path / "public-boundary" / "var" / "defense"
    source = _write_authorized_source(runtime, _coverage_posts())
    with pytest.raises(TypeError):
        generate_thailand_package(  # type: ignore[call-arg]
            source,
            runtime,
            runtime / "derived" / "target.zip",
            encoder=DeterministicEncoder(),
        )


def _symlink_or_skip(link: Path, target: Path, *, directory: bool = False) -> None:
    try:
        link.symlink_to(target, target_is_directory=directory)
    except OSError as error:
        if not directory:
            pytest.skip(f"symlink creation is unavailable: {error}")
        command_shell = shutil.which("cmd")
        if command_shell is None:
            pytest.skip(f"directory reparse creation is unavailable: {error}")
        completed = subprocess.run(
            [
                command_shell,
                "/d",
                "/c",
                "mklink",
                "/J",
                str(link),
                str(target),
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        if completed.returncode != 0 or not thailand_module._is_reparse(link):
            pytest.skip(
                f"directory reparse creation is unavailable: "
                f"{completed.stderr or completed.stdout or error}"
            )


@pytest.mark.parametrize("metadata_name", ["runtime-authorization.json", "authorization.json", "posts.jsonl"])
def test_generator_rejects_symlinked_authorization_and_source_files(
    tmp_path: Path, metadata_name: str
) -> None:
    runtime = tmp_path / metadata_name / "var" / "defense"
    source = _write_authorized_source(runtime, _coverage_posts())
    path = runtime / metadata_name if metadata_name.startswith("runtime-") else source / metadata_name
    target = path.with_name(f"real-{metadata_name}")
    path.replace(target)
    _symlink_or_skip(path, target)

    with pytest.raises(SourceValidationError, match="reparse|symlink|unsafe"):
        thailand_module._generate_thailand_package_with_encoder(
            source,
            runtime,
            runtime / "derived" / "target.zip",
            encoder=DeterministicEncoder(),
        )


@pytest.mark.parametrize("kind", ["source-parent", "output-parent"])
def test_generator_rejects_reparse_source_and_output_parents(tmp_path: Path, kind: str) -> None:
    runtime = tmp_path / kind / "var" / "defense"
    source = _write_authorized_source(runtime, _coverage_posts())
    if kind == "source-parent":
        linked = runtime / "linked-source"
        _symlink_or_skip(linked, source, directory=True)
        requested_source = linked
        output = runtime / "derived" / "target.zip"
    else:
        real_output = runtime / "real-output"
        real_output.mkdir()
        linked_output = runtime / "linked-output"
        _symlink_or_skip(linked_output, real_output, directory=True)
        assert thailand_module._is_reparse(linked_output)
        requested_source = source
        output = linked_output / "target.zip"

    with pytest.raises(SourceValidationError, match="reparse|symlink|unsafe"):
        thailand_module._generate_thailand_package_with_encoder(
            requested_source,
            runtime,
            output,
            encoder=DeterministicEncoder(),
        )


def test_generator_parses_the_exact_authorized_source_snapshot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runtime = tmp_path / "snapshot" / "var" / "defense"
    source = _write_authorized_source(runtime, _coverage_posts())
    source_path = source / "posts.jsonl"
    original_hash = hashlib.sha256(source_path.read_bytes()).hexdigest()
    original_read_bytes = Path.read_bytes
    mutated = False

    def read_then_mutate(path: Path) -> bytes:
        nonlocal mutated
        snapshot = original_read_bytes(path)
        if path == source_path and not mutated:
            mutated = True
            path.write_bytes(b'{"tampered":true}\n')
        return snapshot

    monkeypatch.setattr(Path, "read_bytes", read_then_mutate)
    package = thailand_module._generate_thailand_package_with_encoder(
        source,
        runtime,
        runtime / "derived" / "target.zip",
        encoder=DeterministicEncoder(),
    )

    assert mutated is True
    assert source_path.read_bytes() == b'{"tampered":true}\n'
    receipt = json.loads(package.receipt_path.read_text(encoding="utf-8"))
    assert receipt["sourceSha256"] == original_hash
    manifest, _, _ = _read_bundle(package.bundle_path)
    assert manifest["nodeCount"] == 128
