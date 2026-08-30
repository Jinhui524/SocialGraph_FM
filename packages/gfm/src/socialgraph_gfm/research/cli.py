"""Six-stage command line workflow for the isolated SocialGraph-FM Research release."""

from __future__ import annotations

import argparse
import json
import os
from collections.abc import Sequence
from pathlib import Path

from socialgraph_gfm.runtime import ARTIFACT_ROOT_ENV

from .workflow import (
    evaluate_research_model,
    export_research_model,
    load_corpus_manifest,
    load_export_manifest,
    load_registry,
    materialize_fixture_corpus,
    materialize_research_corpus,
    publish_research_model,
    research_root_from_home,
    smoke_research_export,
    train_research_comparison_matrix,
    train_research_model,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="socialgraph-gfm-research")
    subparsers = parser.add_subparsers(dest="command", required=True)

    def command(name: str) -> argparse.ArgumentParser:
        selected = subparsers.add_parser(name)
        selected.add_argument("--research-root", type=Path)
        return selected

    materialize = command("materialize")
    materialize.add_argument("--fixture-root", type=Path)

    train = command("train")
    train.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    train.add_argument("--pretrain-epochs", type=int, default=60)
    train.add_argument("--head-epochs", type=int, default=100)
    train.add_argument(
        "--skip-comparison",
        action="store_true",
        help="test-only shortcut; resulting evaluation cannot be exported or published",
    )

    evaluate = command("evaluate")
    evaluate.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    command("export")
    command("smoke")
    command("publish")
    return parser


def _root(argument: Path | None, parser: argparse.ArgumentParser) -> Path:
    if argument is not None:
        return argument.expanduser().resolve()
    home = os.environ.get(ARTIFACT_ROOT_ENV, "").strip()
    if not home:
        parser.error(f"--research-root or {ARTIFACT_ROOT_ENV} is required")
    return research_root_from_home(home)


def _summary(command: str, root: Path, artifact: Path, identity: dict[str, object]) -> None:
    print(
        json.dumps(
            {
                "schemaVersion": "socialgraph-fm.research-cli-result/1.0",
                "command": command,
                "researchRoot": str(root),
                "artifact": str(artifact.resolve()),
                **identity,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    arguments = parser.parse_args(argv)
    root = _root(arguments.research_root, parser)
    try:
        if arguments.command == "materialize":
            artifact = (
                materialize_fixture_corpus(root, arguments.fixture_root)
                if arguments.fixture_root is not None
                else materialize_research_corpus(root)
            )
            payload = load_corpus_manifest(root)
            identity = {
                "corpusHash": payload["corpusHash"],
                "graphCount": payload["graphCount"],
                "nodeCount": payload["nodeCount"],
                "edgeCount": payload["edgeCount"],
            }
        elif arguments.command == "train":
            artifact = train_research_model(
                root,
                device=arguments.device,
                pretrain_epochs=arguments.pretrain_epochs,
                head_epochs=arguments.head_epochs,
            )
            payload = json.loads(artifact.read_text(encoding="utf-8"))
            identity = {
                "trainingHash": payload["trainingHash"],
                "checkpointSha256": payload["checkpointSha256"],
            }
            if not arguments.skip_comparison:
                comparison_path = train_research_comparison_matrix(
                    root,
                    device=arguments.device,
                    pretrain_epochs=arguments.pretrain_epochs,
                    downstream_epochs=arguments.head_epochs,
                )
                comparison = json.loads(comparison_path.read_text(encoding="utf-8"))
                identity["comparisonMatrixHash"] = comparison["matrixHash"]
                identity["comparisonRunCount"] = comparison["runCount"]
        elif arguments.command == "evaluate":
            artifact = evaluate_research_model(root, device=arguments.device)
            payload = json.loads(artifact.read_text(encoding="utf-8"))
            identity = {"evaluationHash": payload["evaluationHash"]}
        elif arguments.command == "export":
            artifact = export_research_model(root)
            payload = load_export_manifest(root)
            identity = {
                "artifactHash": payload["artifactHash"],
                "modelVersionId": payload["modelVersionId"],
                "modelVersionHash": payload["modelVersionHash"],
            }
        elif arguments.command == "smoke":
            artifact = smoke_research_export(root)
            payload = json.loads(artifact.read_text(encoding="utf-8"))
            identity = {"passed": payload["passed"], "smokeHash": payload["smokeHash"]}
        else:
            artifact = publish_research_model(root)
            payload = load_registry(root)
            identity = {
                "registryHash": payload["registryHash"],
                "modelVersionId": payload["modelVersionId"],
                "modelVersionHash": payload["modelVersionHash"],
            }
    except (FileNotFoundError, FileExistsError, OSError, RuntimeError, ValueError) as error:
        parser.error(str(error))
    _summary(arguments.command, root, artifact, identity)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
