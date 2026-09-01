"""Command-line entry point for the isolated core loopback service."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from socialgraph_gfm.global_model.service import GlobalServingRuntime
from socialgraph_gfm.research.service import ResearchServingRuntime

from .artifact_catalog import ArtifactCatalog
from .inference_service import (
    InferenceRuntime,
    RunStore,
    atomic_publish_session_token,
    create_server,
)
from .serving_control import ServingControlStore
from .serving_registry import ServingRegistry


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Serve accepted core GFM checkpoints")
    parser.add_argument("--runtime-root", type=Path, required=True)
    parser.add_argument("--serving-control", type=Path, required=True)
    parser.add_argument("--published-serving-root", type=Path)
    parser.add_argument("--published-artifact-root", type=Path)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--token-file", type=Path, required=True)
    parser.add_argument("--research-root", type=Path)
    parser.add_argument("--global-model-root", type=Path)
    parser.add_argument("--governance-root", type=Path)
    parser.add_argument("--dataset-store-root", type=Path)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8766)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    runtime_root = arguments.runtime_root.expanduser().resolve(strict=True)
    token_file = arguments.token_file.expanduser().resolve()
    try:
        token_file.relative_to(runtime_root)
    except ValueError as error:
        raise SystemExit("token file must be inside the authorized runtime root") from error
    published_serving_root = (
        arguments.published_serving_root.expanduser().resolve(strict=True)
        if arguments.published_serving_root is not None
        else arguments.serving_control.expanduser().resolve(strict=True).parent
    )
    published_artifact_root = (
        arguments.published_artifact_root.expanduser().resolve(strict=True)
        if arguments.published_artifact_root is not None
        else arguments.artifact_root.expanduser().resolve(strict=True)
    )
    serving_control = ServingControlStore.load(
        arguments.serving_control,
        high_water_root=runtime_root / "serving-control",
    )
    control = serving_control.capture()
    registry_path = published_serving_root / control.document.registry.relative_path
    catalog_path = published_serving_root / control.document.catalog.relative_path
    registry = ServingRegistry.load(registry_path, runtime_root=published_serving_root)
    artifact_catalog = ArtifactCatalog.load(
        catalog_path,
        artifact_root=published_artifact_root,
    )
    registry.capabilities(registry_snapshot=control.registry_snapshot)
    serving_control.accept(control)
    store = RunStore(
        runtime_root / "inference",
        registry=registry,
        artifact_catalog=artifact_catalog,
        serving_control=serving_control,
    )
    research_runtime = None
    global_model_runtime = None
    governance_runtime = None
    server = None
    token = atomic_publish_session_token(token_file)
    try:
        if arguments.research_root is not None:
            research_runtime = ResearchServingRuntime(
                arguments.research_root,
                arguments.dataset_store_root,
            )
        if arguments.global_model_root is not None:
            global_model_runtime = GlobalServingRuntime(arguments.global_model_root)
        if arguments.governance_root is not None:
            model_root = arguments.global_model_root
            if model_root is None:
                raise SystemExit(
                    "--global-model-root is required with --governance-root"
                )
            from socialgraph_gfm.governance.service import GovernanceServingRuntime

            governance_runtime = GovernanceServingRuntime(
                arguments.governance_root,
                global_model_root=model_root,
                device="cpu",
            )
            if governance_runtime.health().get("servingReady") is not True:
                raise RuntimeError(
                    "SocialGraph-FM Governance Global model failed fail-closed startup validation"
                )
        server = create_server(
            arguments.host,
            arguments.port,
            token=token,
            runtime=InferenceRuntime(store, registry, serving_control),
            research_runtime=research_runtime,
            global_model_runtime=global_model_runtime,
            governance_runtime=governance_runtime,
        )
        server.serve_forever(poll_interval=0.25)
    except KeyboardInterrupt:
        return 0
    finally:
        if server is not None:
            server.server_close()
        else:
            if research_runtime is not None:
                research_runtime.close()
            if global_model_runtime is not None:
                global_model_runtime.close()
            if governance_runtime is not None:
                governance_runtime.close()
        token_file.unlink(missing_ok=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
