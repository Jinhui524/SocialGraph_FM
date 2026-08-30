"""Registered raw-to-experiment conversions for formal core datasets.

Formal conversion is deliberately narrower than the smoke/dev publisher.  Every
registered converter consumes the already hash-checked raw byte snapshots, calls
the production parser and split primitive, and returns the only bundle/label
semantics that the formal preflight may accept.
"""

from __future__ import annotations

import gzip
import hashlib
import io
import tempfile
from dataclasses import dataclass
from pathlib import Path

from socialgraph_gfm.canonical import canonical_sha256

from .bundle import SourceProvenance, SplitManifest, CoreGraphBundle
from .datasets.parsers import parse_email_files, parse_wiki_rfa
from .datasets.recipes import DatasetRecipe
from .experiment_data import bundle_from_parsed_graph
from .splits import spanning_forest_link_split, stratified_signed_edge_split


FORMAL_SPLIT_SEED = 1729
_MATERIALIZER_VERSION = "socialgraph-fm.core-formal-materializer/1.0"
_STREAM_CHUNK_BYTES = 1024 * 1024


ScalarLabel = int | float | str


@dataclass(frozen=True)
class FormalMaterializerBinding:
    materializer_id: str
    materializer_version: str
    materializer_code_sha256: str
    materialization_protocol_hash: str


@dataclass(frozen=True)
class DerivedFormalDataset:
    bundle: CoreGraphBundle
    labels: dict[str, dict[str, ScalarLabel]]
    split_manifests: tuple[SplitManifest, ...]


_REGISTERED: dict[str, str] = {
    "email-eu-core": "email-eu-core",
    "wiki-rfa": "wiki-rfa",
}


def _implementation_hash() -> str:
    directory = Path(__file__).resolve().parent
    files = (
        directory / "formal_materialization.py",
        directory / "experiment_data.py",
        directory / "splits.py",
        directory / "datasets" / "parsers.py",
    )
    return canonical_sha256(
        {
            path.name if path.parent == directory else f"datasets/{path.name}": hashlib.sha256(
                path.read_bytes()
            ).hexdigest()
            for path in files
        }
    )


def formal_materializer_binding(
    *,
    requirement_id: str,
    recipe: DatasetRecipe,
    graph_id: str,
    bundle_split_policy: str,
    experiment_split_policy: str,
) -> FormalMaterializerBinding | None:
    converter = _REGISTERED.get(requirement_id)
    if converter is None:
        return None
    materializer_id = f"socialgraph-fm.{converter}-formal"
    code_hash = _implementation_hash()
    protocol_hash = canonical_sha256(
        {
            "materializerId": materializer_id,
            "materializerVersion": _MATERIALIZER_VERSION,
            "materializerCodeSha256": code_hash,
            "requirementId": requirement_id,
            "recipeId": recipe.recipe_id,
            "recipeVersion": recipe.recipe_version,
            "recipeSha256": recipe.recipe_sha256,
            "graphId": graph_id,
            "bundleSplitPolicy": bundle_split_policy,
            "experimentSplitPolicy": experiment_split_policy,
            "splitSeed": FORMAL_SPLIT_SEED,
        }
    )
    return FormalMaterializerBinding(
        materializer_id=materializer_id,
        materializer_version=_MATERIALIZER_VERSION,
        materializer_code_sha256=code_hash,
        materialization_protocol_hash=protocol_hash,
    )


def _write_gzip_member(source: bytes, target: Path, *, maximum: int) -> None:
    total = 0
    with gzip.GzipFile(fileobj=io.BytesIO(source)) as compressed:
        with target.open("xb") as output:
            while chunk := compressed.read(_STREAM_CHUNK_BYTES):
                total += len(chunk)
                if total > maximum:
                    raise ValueError("formal gzip expansion exceeds the dataset limit")
                output.write(chunk)
    if total == 0:
        raise ValueError("formal gzip source expands to an empty file")


def _source(
    *, recipe: DatasetRecipe, graph_id: str, source_sha256: str
) -> SourceProvenance:
    return SourceProvenance(
        sourceName=f"{recipe.recipe_id}:{graph_id}",
        sourceUri=recipe.sources[0].url,
        citation=recipe.citation,
        sourceSha256=source_sha256,
    )


def _edge_labels(bundle: CoreGraphBundle) -> dict[str, ScalarLabel]:
    return {
        f"edge:{edge.source_id}:{edge.target_id}": 1 for edge in bundle.edges
    }


def _derive_email(
    *,
    recipe: DatasetRecipe,
    raw_sources: dict[str, bytes],
    combined_source_sha256: str,
) -> DerivedFormalDataset:
    if set(raw_sources) != {"edges", "departments"}:
        raise ValueError("Email formal raw inventory is incomplete")
    with tempfile.TemporaryDirectory(prefix="socialgraph-formal-email-") as temporary:
        directory = Path(temporary)
        edges = directory / "email-Eu-core.txt"
        departments = directory / "email-Eu-core-department-labels.txt"
        _write_gzip_member(raw_sources["edges"], edges, maximum=64 * 1024 * 1024)
        _write_gzip_member(
            raw_sources["departments"], departments, maximum=16 * 1024 * 1024
        )
        parsed = parse_email_files(edges, departments)
    split = spanning_forest_link_split(
        num_nodes=len(parsed.node_ids), edges=parsed.edges, seed=FORMAL_SPLIT_SEED
    )
    bundle = bundle_from_parsed_graph(
        parsed,
        source=_source(
            recipe=recipe,
            graph_id=parsed.graph_id,
            source_sha256=combined_source_sha256,
        ),
        split=split,
        excluded_feature_names=recipe.excluded_model_fields,
    )
    departments_by_node: dict[str, ScalarLabel] = {
        node_id: value
        for node_id, value in zip(
            parsed.node_ids, parsed.offline_labels["department"], strict=True
        )
    }
    return DerivedFormalDataset(
        bundle=bundle,
        labels={
            "departmentValidation": departments_by_node,
            "relationCompletion": _edge_labels(bundle),
        },
        split_manifests=(bundle.split_manifest,),
    )


def _derive_wiki(
    *,
    recipe: DatasetRecipe,
    raw_sources: dict[str, bytes],
    combined_source_sha256: str,
) -> DerivedFormalDataset:
    if set(raw_sources) != {"wiki-rfa"}:
        raise ValueError("Wiki-RfA formal raw inventory is incomplete")
    with tempfile.TemporaryDirectory(prefix="socialgraph-formal-wiki-") as temporary:
        path = Path(temporary) / "wiki-RfA.txt"
        _write_gzip_member(raw_sources["wiki-rfa"], path, maximum=512 * 1024 * 1024)
        parsed = parse_wiki_rfa(path)
    split = stratified_signed_edge_split(
        edges=parsed.signed_edges, seed=FORMAL_SPLIT_SEED
    )
    bundle = bundle_from_parsed_graph(
        parsed,
        source=_source(
            recipe=recipe,
            graph_id=parsed.graph_id,
            source_sha256=combined_source_sha256,
        ),
        split=split,
        excluded_feature_names=recipe.excluded_model_fields,
    )
    sign_by_id: dict[str, ScalarLabel] = {
        f"edge:{parsed.node_ids[left]}:{parsed.node_ids[right]}": sign
        for left, right, sign in parsed.signed_edges
    }
    return DerivedFormalDataset(
        bundle=bundle,
        labels={"voteSign": sign_by_id},
        split_manifests=(bundle.split_manifest,),
    )


def derive_registered_formal_dataset(
    *,
    requirement_id: str,
    recipe: DatasetRecipe,
    graph_id: str,
    raw_sources: dict[str, bytes],
    combined_source_sha256: str,
    bundle_split_policy: str,
) -> DerivedFormalDataset:
    """Re-materialize one registered formal graph from its exact raw snapshots."""

    if requirement_id == "email-eu-core":
        derived = _derive_email(
            recipe=recipe,
            raw_sources=raw_sources,
            combined_source_sha256=combined_source_sha256,
        )
    elif requirement_id == "wiki-rfa":
        derived = _derive_wiki(
            recipe=recipe,
            raw_sources=raw_sources,
            combined_source_sha256=combined_source_sha256,
        )
    else:
        raise ValueError("no dataset-specific formal materializer is registered")
    if derived.bundle.split_manifest.strategy != bundle_split_policy:
        raise ValueError("formal materializer emitted the wrong bundle split policy")
    if derived.bundle.source.source_sha256 != combined_source_sha256:
        raise ValueError("formal materializer did not bind the raw snapshot hash")
    return derived


__all__ = [
    "DerivedFormalDataset",
    "FORMAL_SPLIT_SEED",
    "FormalMaterializerBinding",
    "derive_registered_formal_dataset",
    "formal_materializer_binding",
]
