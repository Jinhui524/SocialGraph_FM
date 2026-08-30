"""Deterministic SocialGraph-FM Governance demo bundle generation from the Russia corpus."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import tempfile
import zipfile
from pathlib import Path
from typing import Any

import numpy as np

from socialgraph_gfm.global_model.corpus import load_corpus_index

from .contracts import INPUT_SCHEMA_VERSION, MODALITIES, GovernanceInputManifest


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_zip_member(archive: zipfile.ZipFile, source: Path, name: str) -> None:
    info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
    info.compress_type = zipfile.ZIP_DEFLATED
    info.external_attr = 0o100644 << 16
    archive.writestr(info, source.read_bytes(), compresslevel=6)


def create_russia_demo_bundle(global_model_root: str | Path, output: str | Path) -> Path:
    """Create a contract-complete Russia replay archive without labels or split masks."""

    root = Path(global_model_root).expanduser().resolve(strict=True)
    destination = Path(output).expanduser().resolve()
    if destination.exists():
        raise FileExistsError(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    index = load_corpus_index(root / "corpus", verify_manifests=True)
    corpus = index.load_country(
        "russia", verify_hashes=True, verify_values=True, mmap_mode="r"
    )
    node_count = corpus.manifest.node_count
    node_ids = tuple(f"russia:{index}" for index in range(node_count))
    with tempfile.TemporaryDirectory(prefix="governance-demo-") as raw_temporary:
        temporary = Path(raw_temporary)
        nodes_path = temporary / "nodes.csv"
        with nodes_path.open("x", encoding="utf-8", newline="") as stream:
            writer = csv.writer(stream, lineterminator="\n")
            writer.writerow(("node_id", "display_name"))
            for index_value, node_id in enumerate(node_ids):
                writer.writerow((node_id, f"Anonymous account {index_value}"))

        relations_path = temporary / "relations.csv"
        relation_rows = 0
        observed: list[str] = []
        with relations_path.open("x", encoding="utf-8", newline="") as stream:
            writer = csv.writer(stream, lineterminator="\n")
            writer.writerow(("source", "target", "modality", "weight"))
            for modality in MODALITIES:
                relation = corpus.relation(modality)
                before = relation_rows
                for source in range(node_count):
                    start, stop = int(relation.indptr[source]), int(relation.indptr[source + 1])
                    for position in range(start, stop):
                        target = int(relation.indices[position])
                        if source >= target:
                            continue
                        writer.writerow(
                            (
                                node_ids[source],
                                node_ids[target],
                                modality,
                                format(float(relation.weights[position]), ".17g"),
                            )
                        )
                        relation_rows += 1
                if relation_rows > before:
                    observed.append(modality)

        features_path = temporary / "features.npz"
        feature_ids = np.asarray(node_ids)
        with features_path.open("xb") as stream:
            np.savez_compressed(
                stream,
                node_ids=feature_ids,
                text_features=np.asarray(corpus.text_features, dtype=np.float32),
            )
            stream.flush()
            os.fsync(stream.fileno())
        files = {
            name: {"sha256": _sha256(path), "bytes": path.stat().st_size}
            for name, path in (
                ("nodes.csv", nodes_path),
                ("relations.csv", relations_path),
                ("features.npz", features_path),
            )
        }
        manifest = GovernanceInputManifest.model_validate(
            {
                "schemaVersion": INPUT_SCHEMA_VERSION,
                "datasetId": "socialgraph-fm:russia:dynamic-replay",
                "displayName": "Governance Russia dynamic replay",
                "nodeCount": node_count,
                "relationRowCount": relation_rows,
                "featureDimension": 768,
                "modalities": observed,
                "files": files,
                "license": "CC-BY-4.0",
                "sourceUri": "https://zenodo.org/records/13357621",
            }
        )
        manifest_path = temporary / "manifest.json"
        manifest_payload: dict[str, Any] = manifest.model_dump(mode="json")
        manifest_path.write_text(
            json.dumps(
                manifest_payload,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ),
            encoding="utf-8",
            newline="\n",
        )
        staging = destination.parent / f".{destination.name}.{os.getpid()}.tmp"
        try:
            with zipfile.ZipFile(staging, mode="x", allowZip64=False) as archive:
                for name in ("manifest.json", "nodes.csv", "relations.csv", "features.npz"):
                    _write_zip_member(archive, temporary / name, name)
            os.replace(staging, destination)
        finally:
            staging.unlink(missing_ok=True)
    return destination


def create_tiny_contract_bundle(output: str | Path) -> Path:
    """Create a deterministic synthetic bundle for contract and UI smoke tests only."""

    destination = Path(output).expanduser().resolve()
    if destination.exists():
        raise FileExistsError(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    node_ids = tuple(f"synthetic:{index}" for index in range(6))
    relation_values = (
        (0, 1, "coRT", 0.8),
        (1, 2, "coURL", 0.7),
        (2, 3, "hashSeq", 0.6),
        (3, 4, "fastRT", 0.5),
        (4, 5, "tweetSim", 0.4),
        (0, 5, "coRT", 0.3),
    )
    with tempfile.TemporaryDirectory(prefix="governance-tiny-") as raw_temporary:
        temporary = Path(raw_temporary)
        nodes_path = temporary / "nodes.csv"
        with nodes_path.open("x", encoding="utf-8", newline="") as stream:
            writer = csv.writer(stream, lineterminator="\n")
            writer.writerow(("node_id", "display_name"))
            for index_value, node_id in enumerate(node_ids):
                writer.writerow((node_id, f"Synthetic account {index_value}"))
        relations_path = temporary / "relations.csv"
        with relations_path.open("x", encoding="utf-8", newline="") as stream:
            writer = csv.writer(stream, lineterminator="\n")
            writer.writerow(("source", "target", "modality", "weight"))
            for source, target, modality, weight in relation_values:
                writer.writerow((node_ids[source], node_ids[target], modality, format(weight, ".17g")))
        features_path = temporary / "features.npz"
        features = np.linspace(-1.0, 1.0, num=len(node_ids) * 768, dtype=np.float32).reshape(
            len(node_ids), 768
        )
        with features_path.open("xb") as stream:
            np.savez_compressed(stream, node_ids=np.asarray(node_ids), text_features=features)
            stream.flush()
            os.fsync(stream.fileno())
        files = {
            name: {"sha256": _sha256(path), "bytes": path.stat().st_size}
            for name, path in (
                ("nodes.csv", nodes_path),
                ("relations.csv", relations_path),
                ("features.npz", features_path),
            )
        }
        manifest = GovernanceInputManifest.model_validate(
            {
                "schemaVersion": INPUT_SCHEMA_VERSION,
                "datasetId": "governance:synthetic:contract-only",
                "displayName": "Synthetic Governance contract sample",
                "nodeCount": len(node_ids),
                "relationRowCount": len(relation_values),
                "featureDimension": 768,
                "modalities": list(MODALITIES),
                "files": files,
                "license": "Synthetic contract fixture; not governance evidence",
            }
        )
        manifest_path = temporary / "manifest.json"
        manifest_path.write_text(
            json.dumps(
                manifest.model_dump(mode="json"),
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ),
            encoding="utf-8",
            newline="\n",
        )
        staging = destination.parent / f".{destination.name}.{os.getpid()}.tmp"
        try:
            with zipfile.ZipFile(staging, mode="x", allowZip64=False) as archive:
                for name in ("manifest.json", "nodes.csv", "relations.csv", "features.npz"):
                    _write_zip_member(archive, temporary / name, name)
            os.replace(staging, destination)
        finally:
            staging.unlink(missing_ok=True)
    return destination


__all__ = ["create_russia_demo_bundle", "create_tiny_contract_bundle"]
