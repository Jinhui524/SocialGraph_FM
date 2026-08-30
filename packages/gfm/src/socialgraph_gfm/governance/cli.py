"""CLI utilities for governed SocialGraph-FM Governance online input artifacts."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from socialgraph_gfm.canonical import canonical_sha256, file_sha256

from .bundle import create_russia_demo_bundle, create_tiny_contract_bundle
from .knowledge import (
    KnowledgeIndex,
    KnowledgeSource,
    build_knowledge_index,
    default_source_uri,
)
from .materialize import materialize_bundle
from .russia_answer_packs import (
    generate_russia_answer_packs,
    verify_russia_answer_pack_catalog,
)
from .russia_protocols import (
    generate_russia_protocol_focus,
    verify_russia_protocol_focus,
)
from .russia_shards import generate_russia_shards, verify_russia_shard_catalog
from .target_tasks import (
    generate_governance_target_tasks,
    reset_governance_target_tasks,
    verify_governance_target_tasks,
)
from .thailand import generate_thailand_package


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="socialgraph-gfm-governance")
    subparsers = parser.add_subparsers(dest="command", required=True)
    demo = subparsers.add_parser("demo-bundle")
    demo.add_argument("--global-model-root", type=Path, required=True)
    demo.add_argument("--output", type=Path, required=True)
    tiny = subparsers.add_parser("tiny-bundle")
    tiny.add_argument("--output", type=Path, required=True)
    shards = subparsers.add_parser("russia-shards")
    shards.add_argument("--source", type=Path, required=True)
    shards.add_argument("--output-dir", type=Path, required=True)
    verify_shards = subparsers.add_parser("verify-russia-shards")
    verify_shards.add_argument("--source", type=Path, required=True)
    verify_shards.add_argument("--catalog", type=Path, required=True)
    answer_packs = subparsers.add_parser("russia-answer-packs")
    answer_packs.add_argument("--source", type=Path, required=True)
    answer_packs.add_argument("--output-dir", type=Path, required=True)
    answer_packs.add_argument("--frozen-scores", type=Path)
    verify_answer_packs = subparsers.add_parser("verify-russia-answer-packs")
    verify_answer_packs.add_argument("--source", type=Path, required=True)
    verify_answer_packs.add_argument("--catalog", type=Path, required=True)
    protocols = subparsers.add_parser("russia-protocol-focus")
    protocols.add_argument("--global-model-root", type=Path, required=True)
    protocols.add_argument("--output-dir", type=Path, required=True)
    verify_protocols = subparsers.add_parser("verify-russia-protocol-focus")
    verify_protocols.add_argument("--global-model-root", type=Path, required=True)
    verify_protocols.add_argument("--output-dir", type=Path, required=True)
    knowledge = subparsers.add_parser("knowledge-import")
    knowledge.add_argument("--root", type=Path, required=True)
    knowledge.add_argument("--source", action="append", required=True)
    knowledge.add_argument("--source-uri", action="append", default=[])
    materialize = subparsers.add_parser("materialize")
    materialize.add_argument("--root", type=Path, required=True)
    materialize.add_argument("--artifact-id", required=True)
    materialize.add_argument("--dataset-content-hash", required=True)
    materialize.add_argument("--graph-version-hash", required=True)
    materialize.add_argument("--clean-self-loops", action="store_true")
    thailand = subparsers.add_parser("thailand-package")
    thailand.add_argument("--source-directory", type=Path, required=True)
    thailand.add_argument("--runtime-root", type=Path, required=True)
    thailand.add_argument("--output", type=Path, required=True)
    thailand.add_argument("--encoder-cache", type=Path, required=True)
    governance_tasks = subparsers.add_parser("governance-target-tasks")
    governance_tasks.add_argument("--corpus-root", type=Path, required=True)
    governance_tasks.add_argument("--output-dir", type=Path, required=True)
    verify_governance = subparsers.add_parser("verify-governance-target-tasks")
    verify_governance.add_argument("--output-dir", type=Path, required=True)
    verify_governance.add_argument("--corpus-root", type=Path, required=True)
    reset_governance = subparsers.add_parser("reset-governance-target-tasks")
    reset_governance.add_argument("--output-dir", type=Path, required=True)
    return parser


def _label_values(values: Sequence[str], *, option: str) -> dict[str, str]:
    parsed: dict[str, str] = {}
    for value in values:
        label, separator, payload = value.partition("=")
        if not separator or not label or not payload or label in parsed:
            raise ValueError(f"{option} must use unique label=value entries")
        parsed[label] = payload
    return parsed


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    if arguments.command == "demo-bundle":
        path = create_russia_demo_bundle(arguments.global_model_root, arguments.output)
        print(path)
        return 0
    if arguments.command == "tiny-bundle":
        path = create_tiny_contract_bundle(arguments.output)
        print(path)
        return 0
    if arguments.command == "russia-shards":
        path = generate_russia_shards(arguments.source, arguments.output_dir)
        print(path)
        return 0
    if arguments.command == "verify-russia-shards":
        catalog = verify_russia_shard_catalog(arguments.source, arguments.catalog)
        print(catalog.catalog_hash)
        return 0
    if arguments.command == "russia-answer-packs":
        scores = None
        if arguments.frozen_scores is not None:
            with np.load(arguments.frozen_scores, allow_pickle=False) as archive:
                if "scores" not in archive.files:
                    raise ValueError("--frozen-scores NPZ must contain a scores array")
                scores = np.asarray(archive["scores"], dtype=np.float32)
        path = generate_russia_answer_packs(
            arguments.source,
            arguments.output_dir,
            frozen_scores=scores,
        )
        print(path)
        return 0
    if arguments.command == "verify-russia-answer-packs":
        answer_catalog = verify_russia_answer_pack_catalog(arguments.source, arguments.catalog)
        print(answer_catalog.catalog_hash)
        return 0
    if arguments.command == "russia-protocol-focus":
        path = generate_russia_protocol_focus(arguments.global_model_root, arguments.output_dir)
        print(path)
        return 0
    if arguments.command == "verify-russia-protocol-focus":
        manifests = verify_russia_protocol_focus(
            arguments.global_model_root, arguments.output_dir
        )
        print(
            canonical_sha256(
                {item.directory_protocol: item.manifest_hash for item in manifests}
            )
        )
        return 0
    if arguments.command == "knowledge-import":
        source_values = _label_values(arguments.source, option="--source")
        uri_values = _label_values(arguments.source_uri, option="--source-uri")
        if set(uri_values) - set(source_values):
            raise ValueError("--source-uri labels must also appear in --source")
        sources = tuple(
            KnowledgeSource(
                label=label,
                path=Path(path),
                uri=uri_values.get(label, default_source_uri(label)),
            )
            for label, path in sorted(source_values.items())
        )
        destination = arguments.root.expanduser().resolve() / "knowledge"
        build_knowledge_index(destination, sources)
        print(KnowledgeIndex(destination).verify())
        return 0
    if arguments.command == "thailand-package":
        package = generate_thailand_package(
            arguments.source_directory,
            arguments.runtime_root,
            arguments.output,
            encoder_cache=arguments.encoder_cache,
        )
        print(package.bundle_path)
        print(package.labels_path)
        print(package.receipt_path)
        return 0
    if arguments.command == "governance-target-tasks":
        generated = generate_governance_target_tasks(
            arguments.corpus_root, arguments.output_dir
        )
        print(generated.zero_shot)
        print(generated.few_shot)
        return 0
    if arguments.command == "verify-governance-target-tasks":
        for verified in verify_governance_target_tasks(
            arguments.output_dir, corpus_root=arguments.corpus_root
        ):
            print(f"{verified.path.name} {file_sha256(verified.path)}")
        return 0
    if arguments.command == "reset-governance-target-tasks":
        reset_governance_target_tasks(arguments.output_dir)
        print(Path(arguments.output_dir).expanduser().resolve())
        return 0
    artifact = materialize_bundle(
        arguments.root,
        arguments.artifact_id,
        expected_dataset_content_hash=arguments.dataset_content_hash,
        expected_graph_version_hash=arguments.graph_version_hash,
        clean_self_loops=arguments.clean_self_loops,
    )
    print(artifact.root / "artifact.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
