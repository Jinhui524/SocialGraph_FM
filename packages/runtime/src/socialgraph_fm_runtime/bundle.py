"""Cross-platform verification and installation of the tracked runtime bundle."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Iterable

from .environment import run_clean_python
from .layout import RuntimeLayout
from .profile import atomic_write_json


BUNDLE_SCHEMA_VERSION = "socialgraph-fm.runtime-bundle/1.0"
MODEL_INSTALL_SCHEMA = "socialgraph-fm.runtime-install/1.0"
SEED_INSTALL_SCHEMA = "socialgraph-fm.runtime-seed-install/1.0"
TARGET_CATALOG_SCHEMA = "socialgraph-fm.governance-target-catalog/1.0"
CHECKPOINT_FORWARD_SCHEMA = "socialgraph-fm.global-model-forward-smoke/1.0"
TARGET_EXAMPLE_NAMES = {
    "zero_shot": ("zeroShot", "target-domain-a-zero.sgtask.zip"),
    "few_shot": ("fewShot", "target-domain-b-few.sgtask.zip"),
}
KNOWLEDGE_SEED_FILES = frozenset({"knowledge.sqlite3", "manifest.json"})
ALLOWED_ROOTS = (
    "bundles/models/socialgraph-global",
    "bundles/governance/knowledge",
    "bundles/governance/reviewed-cases",
    "bundles/web",
    "examples/governance/russia",
    "examples/governance/target-domain",
)


_FORWARD_PROBE_SOURCE = r"""
import json
import shutil
import sys
import tempfile
import zipfile
from pathlib import Path

import numpy as np

from socialgraph_gfm.governance.contracts import MODALITIES
from socialgraph_gfm.governance.inference import load_global_model, run_online_inference
from socialgraph_gfm.governance.materialize import (
    _dataset_content_hash,
    _graph_arrays,
    _graph_version_hash,
    _hash_bytes,
    _load_features,
    _parse_manifest,
    _parse_nodes,
    _parse_relations,
    _read_member,
    _validate_zip,
    load_materialized_artifact,
    materialize_bundle,
)

model_root = Path(sys.argv[1]).resolve(strict=True)
bundle_path = Path(sys.argv[2]).resolve(strict=True)
device = "cpu"
runtime_root = Path(tempfile.mkdtemp(prefix="sgfm-forward-probe-"))
try:
    _validate_zip(bundle_path)
    with zipfile.ZipFile(bundle_path) as archive:
        raw_manifest = _read_member(archive, "manifest.json", 2 * 1024 * 1024)
        raw_nodes = _read_member(archive, "nodes.csv", 16 * 1024 * 1024)
        raw_relations = _read_member(archive, "relations.csv", 256 * 1024 * 1024)
        raw_features = _read_member(archive, "features.npz", 128 * 1024 * 1024)
    manifest = _parse_manifest(raw_manifest)
    raw_files = {"nodes.csv": raw_nodes, "relations.csv": raw_relations,
                 "features.npz": raw_features}
    digests = {name: _hash_bytes(value) for name, value in raw_files.items()}
    node_ids, _labels = _parse_nodes(raw_nodes, manifest.nodeCount)
    _load_features(raw_features, node_ids)
    relations, removed, observed = _parse_relations(
        raw_relations, node_ids=node_ids,
        expected_rows=manifest.relationRowCount, clean_self_loops=True,
    )
    if observed != manifest.modalities:
        raise RuntimeError("Russia probe modalities differ from its manifest")
    dataset_hash = _dataset_content_hash(
        manifest_hash=_hash_bytes(raw_manifest), file_digests=digests,
        clean=True, removed=removed,
    )
    undirected, *_rest = _graph_arrays(relations, len(node_ids))
    graph_hash = _graph_version_hash(node_ids, undirected)
    artifact_id = f"governance-artifact-{dataset_hash[:32]}"
    incoming = runtime_root / "incoming" / artifact_id
    incoming.mkdir(parents=True)
    shutil.copyfile(bundle_path, incoming / "bundle.zip")
    materialized = materialize_bundle(
        runtime_root, artifact_id,
        expected_dataset_content_hash=dataset_hash,
        expected_graph_version_hash=graph_hash,
        clean_self_loops=True,
    )
    data = load_materialized_artifact(materialized.root)
    loaded = load_global_model(model_root, device=device)
    outputs = run_online_inference(data, loaded)
    if (
        outputs.scores.shape != (len(node_ids),)
        or not np.isfinite(outputs.scores).all()
        or not np.isfinite(outputs.embeddings).all()
    ):
        raise RuntimeError("Global forward did not return finite all-node outputs")
    print(json.dumps({
        "passed": True,
        "device": loaded.device_name,
        "dtype": loaded.dtype_name,
        "nodeCount": len(node_ids),
        "batchSize": outputs.batch_size,
        "modelVersionHash": loaded.model_version_hash,
    }, sort_keys=True))
finally:
    shutil.rmtree(runtime_root, ignore_errors=True)
"""


_RUNTIME_STATE_PROBE_SOURCE = r"""
import json
import sys
from pathlib import Path

from socialgraph_gfm.global_model.service import GlobalServingRuntime
from socialgraph_gfm.governance.reviewed_cases import ReviewedCaseIndex

model = GlobalServingRuntime(Path(sys.argv[1]))
try:
    health = model.health()
    if health.get("servingReady") is not True:
        raise RuntimeError("SocialGraph-FM Global serving runtime is not ready")
finally:
    model.close()
reviewed = ReviewedCaseIndex(Path(sys.argv[2]))
print(json.dumps({
    "passed": True,
    "modelVersionHash": health.get("modelVersionHash"),
    "reviewedCaseIndexHash": reviewed.index_hash,
}, sort_keys=True))
"""


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(document: dict[str, Any]) -> str:
    encoded = json.dumps(
        document,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _target_example_declarations(
    layout: RuntimeLayout, bundle: "RuntimeBundle"
) -> tuple[dict[str, Any], tuple[tuple[str, Path, dict[str, Any]], ...]]:
    catalog_relative = "examples/governance/target-domain/governance-target-tasks.catalog.json"
    catalog_path = _resolve_source(layout.project_root, catalog_relative)
    try:
        catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise RuntimeError("The published target-domain catalog is invalid") from error
    if not isinstance(catalog, dict) or set(catalog) != {
        "schemaVersion",
        "generationId",
        "targets",
        "catalogHash",
    }:
        raise RuntimeError("The published target-domain catalog inventory is invalid")
    logical = {key: value for key, value in catalog.items() if key != "catalogHash"}
    generation = catalog.get("generationId")
    if (
        catalog.get("schemaVersion") != TARGET_CATALOG_SCHEMA
        or not isinstance(generation, str)
        or len(generation) != 64
        or any(character not in "0123456789abcdef" for character in generation)
        or catalog.get("catalogHash") != _canonical_sha256(logical)
    ):
        raise RuntimeError("The published target-domain catalog identity is invalid")
    catalog_asset = next(
        (asset for asset in bundle.assets if asset.get("path") == catalog_relative),
        None,
    )
    if catalog_asset is None or catalog_asset.get("role") != "target_domain_example":
        raise RuntimeError("The target-domain catalog is not bound to the runtime manifest")
    targets = catalog.get("targets")
    if not isinstance(targets, list) or len(targets) != len(TARGET_EXAMPLE_NAMES):
        raise RuntimeError("The published target-domain catalog targets are invalid")
    declared: list[tuple[str, Path, dict[str, Any]]] = []
    assets_by_path = {str(asset.get("path")): asset for asset in bundle.assets}
    for entry, (role, (label, name)) in zip(
        targets, TARGET_EXAMPLE_NAMES.items(), strict=True
    ):
        expected_relative = PurePosixPath(".governance-target-catalog", generation, name)
        if (
            not isinstance(entry, dict)
            or set(entry) != {"role", "path", "sha256", "bytes"}
            or entry.get("role") != role
            or entry.get("path") != expected_relative.as_posix()
            or not isinstance(entry.get("bytes"), int)
            or not isinstance(entry.get("sha256"), str)
        ):
            raise RuntimeError("The published target-domain catalog role binding is invalid")
        source_relative = f"examples/governance/target-domain/{expected_relative.as_posix()}"
        asset = assets_by_path.get(source_relative)
        if (
            asset is None
            or asset.get("role") != "target_domain_example"
            or asset.get("bytes") != entry["bytes"]
            or asset.get("sha256") != entry["sha256"]
        ):
            raise RuntimeError("A target-domain example is not bound to the runtime manifest")
        source = _resolve_source(layout.project_root, source_relative)
        if source.is_symlink() or source.stat().st_size != entry["bytes"]:
            raise RuntimeError(f"Published target-domain example is invalid: {source}")
        if file_sha256(source) != entry["sha256"]:
            raise RuntimeError(f"Published target-domain example hash differs: {source}")
        declared.append((label, source, entry))
    return catalog, tuple(declared)


def verify_target_examples(
    layout: RuntimeLayout, bundle: "RuntimeBundle"
) -> dict[str, Any]:
    """Verify the two visible, upload-ready target-domain copies under ``var``."""

    catalog, declarations = _target_example_declarations(layout, bundle)
    report: dict[str, Any] = {
        "catalogHash": catalog["catalogHash"],
        "generationId": catalog["generationId"],
    }
    for label, _source, entry in declarations:
        _role, name = next(
            (role, value[1])
            for role, value in TARGET_EXAMPLE_NAMES.items()
            if value[0] == label
        )
        destination = layout.target_examples_root / name
        layout.assert_safe_var_path(destination)
        if not destination.is_file() or destination.is_symlink():
            raise RuntimeError(f"Visible target-domain example is missing: {destination}")
        if destination.stat().st_size != entry["bytes"] or file_sha256(destination) != entry["sha256"]:
            raise RuntimeError(f"Visible target-domain example hash differs: {destination}")
        report[label] = {
            "path": str(destination),
            "bytes": entry["bytes"],
            "sha256": entry["sha256"],
        }
    return report


def materialize_target_examples(
    layout: RuntimeLayout, bundle: "RuntimeBundle"
) -> dict[str, Any]:
    """Create upload-friendly copies without changing the frozen source catalog."""

    _catalog, declarations = _target_example_declarations(layout, bundle)
    root = layout.target_examples_root
    layout.assert_safe_var_path(root)
    root.mkdir(parents=True, exist_ok=True)
    layout.assert_safe_var_path(root)
    for label, source, _entry in declarations:
        _role, name = next(
            (role, value[1])
            for role, value in TARGET_EXAMPLE_NAMES.items()
            if value[0] == label
        )
        destination = root / name
        layout.assert_safe_var_path(destination)
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{name}.", suffix=".tmp", dir=root
        )
        temporary = Path(temporary_name)
        try:
            with os.fdopen(descriptor, "wb") as stream:
                with source.open("rb") as source_stream:
                    shutil.copyfileobj(source_stream, stream, length=1024 * 1024)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, destination)
        finally:
            temporary.unlink(missing_ok=True)
    return verify_target_examples(layout, bundle)


def _safe_relative(value: str) -> PurePosixPath:
    if not value or "\\" in value or "\x00" in value:
        raise RuntimeError(f"Unsafe runtime bundle path: {value!r}")
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise RuntimeError(f"Unsafe runtime bundle path: {value!r}")
    if ":" in path.parts[0]:
        raise RuntimeError(f"Unsafe runtime bundle path: {value!r}")
    return path


def _resolve_source(root: Path, relative: str) -> Path:
    parts = _safe_relative(relative).parts
    selected = root.joinpath(*parts).resolve()
    try:
        selected.relative_to(root.resolve())
    except ValueError as error:
        raise RuntimeError(f"Runtime bundle path escapes the repository: {relative}") from error
    return selected


@dataclass(frozen=True)
class RuntimeBundle:
    document: dict[str, Any]
    assets: tuple[dict[str, Any], ...]
    manifest_sha256: str


def load_and_verify_bundle(layout: RuntimeLayout, *, exact_inventory: bool = True) -> RuntimeBundle:
    manifest = layout.bundle_manifest
    try:
        document = json.loads(manifest.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"Runtime bundle manifest is invalid: {manifest}") from error
    if document.get("schemaVersion") != BUNDLE_SCHEMA_VERSION:
        raise RuntimeError(f"Unsupported runtime bundle schema: {manifest}")
    raw_assets = document.get("assets")
    if not isinstance(raw_assets, list) or not raw_assets:
        raise RuntimeError("Runtime bundle manifest has no assets")
    assets: list[dict[str, Any]] = []
    seen: set[str] = set()
    for raw in raw_assets:
        if not isinstance(raw, dict):
            raise RuntimeError("Runtime bundle asset entry is invalid")
        relative = str(raw.get("path", ""))
        _safe_relative(relative)
        if relative in seen:
            raise RuntimeError(f"Duplicate runtime bundle asset: {relative}")
        seen.add(relative)
        if not any(relative == prefix or relative.startswith(prefix + "/") for prefix in ALLOWED_ROOTS):
            raise RuntimeError(f"Runtime bundle asset uses an unapproved root: {relative}")
        source = _resolve_source(layout.project_root, relative)
        if not source.is_file() or source.is_symlink():
            raise RuntimeError(f"Runtime bundle asset is missing or linked: {relative}")
        expected_bytes = int(raw.get("bytes", -1))
        expected_hash = str(raw.get("sha256", ""))
        if source.stat().st_size != expected_bytes:
            raise RuntimeError(f"Runtime bundle asset size mismatch: {relative}")
        if file_sha256(source) != expected_hash:
            raise RuntimeError(f"Runtime bundle asset SHA-256 mismatch: {relative}")
        assets.append(dict(raw))
    if exact_inventory:
        actual: set[str] = set()
        for prefix in ALLOWED_ROOTS:
            directory = _resolve_source(layout.project_root, prefix)
            if not directory.exists():
                continue
            for file in directory.rglob("*"):
                if file.is_file() or file.is_symlink():
                    actual.add(file.relative_to(layout.project_root).as_posix())
        if actual != seen:
            missing = sorted(seen - actual)
            extra = sorted(actual - seen)
            raise RuntimeError(
                "Runtime bundle inventory differs from the manifest; "
                f"missing={missing[:3]}, extra={extra[:3]}"
            )
    return RuntimeBundle(document, tuple(assets), file_sha256(manifest))


def _asset_relative(asset: dict[str, Any], prefix: str) -> Path:
    relative = str(asset["path"])
    marker = prefix.rstrip("/") + "/"
    if not relative.startswith(marker):
        raise RuntimeError(f"Runtime bundle asset does not use {prefix}: {relative}")
    remainder = relative[len(marker) :]
    return Path(*_safe_relative(remainder).parts)


def _copy_assets(
    layout: RuntimeLayout,
    assets: Iterable[dict[str, Any]],
    prefix: str,
    destination: Path,
) -> None:
    for asset in assets:
        source = _resolve_source(layout.project_root, str(asset["path"]))
        target = destination / _asset_relative(asset, prefix)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        observed_bytes = target.stat().st_size
        observed_hash = file_sha256(target)
        if observed_bytes != int(asset["bytes"]) or observed_hash != asset["sha256"]:
            raise RuntimeError(
                f"Runtime bundle copy failed verification: {asset['path']} "
                f"(bytes={observed_bytes}, sha256={observed_hash})"
            )


def _verify_installed(
    assets: Iterable[dict[str, Any]], prefix: str, destination: Path
) -> None:
    for asset in assets:
        target = destination / _asset_relative(asset, prefix)
        if (
            not target.is_file()
            or target.is_symlink()
            or target.stat().st_size != int(asset["bytes"])
            or file_sha256(target) != asset["sha256"]
        ):
            raise RuntimeError(f"Installed runtime bundle asset differs: {target}")


def _staging_directory(parent: Path, _name: str) -> Path:
    parent.mkdir(parents=True, exist_ok=True)
    # Keep the stage basename short: deeply nested catalog assets otherwise cross
    # the legacy Windows path boundary even when the final destination is valid.
    return Path(tempfile.mkdtemp(prefix=".sg-", suffix=".stage", dir=parent))


def _is_link_or_reparse_point(path: Path) -> bool:
    if path.is_symlink():
        return True
    try:
        attributes = getattr(path.lstat(), "st_file_attributes", 0)
    except OSError as error:
        raise RuntimeError(f"Could not inspect immutable seed path: {path}") from error
    return bool(attributes & 0x400)  # FILE_ATTRIBUTE_REPARSE_POINT


def _assert_exact_knowledge_seed(directory: Path) -> None:
    """Require the two-file managed knowledge inventory before any replacement."""

    if not directory.is_dir() or _is_link_or_reparse_point(directory):
        raise RuntimeError(f"Knowledge seed is not a plain managed directory: {directory}")
    try:
        entries = tuple(directory.iterdir())
    except OSError as error:
        raise RuntimeError(f"Could not inspect knowledge seed: {directory}") from error
    if {entry.name for entry in entries} != KNOWLEDGE_SEED_FILES:
        raise RuntimeError(
            "Knowledge seed refresh requires exactly knowledge.sqlite3 and manifest.json"
        )
    for entry in entries:
        if _is_link_or_reparse_point(entry) or not entry.is_file():
            raise RuntimeError(f"Knowledge seed contains a non-regular file: {entry}")


def _refresh_knowledge_seed(
    layout: RuntimeLayout,
    assets: tuple[dict[str, Any], ...],
    prefix: str,
    destination: Path,
) -> None:
    """Atomically replace the exact immutable knowledge seed, with rollback."""

    relative_inventory = {
        _asset_relative(asset, prefix).as_posix() for asset in assets
    }
    if relative_inventory != KNOWLEDGE_SEED_FILES or len(assets) != len(
        KNOWLEDGE_SEED_FILES
    ):
        raise RuntimeError(
            "Knowledge seed refresh requires exactly knowledge.sqlite3 and manifest.json assets"
        )
    _assert_exact_knowledge_seed(destination)
    try:
        _verify_installed(assets, prefix, destination)
    except RuntimeError:
        pass
    else:
        return
    staging = _staging_directory(destination.parent, destination.name)
    layout.assert_safe_var_path(staging)
    backup: Path | None = None
    previous_moved = False
    try:
        _copy_assets(layout, assets, prefix, staging)
        _verify_installed(assets, prefix, staging)
        _assert_exact_knowledge_seed(staging)
        backup = _staging_directory(destination.parent, destination.name)
        backup.rmdir()
        layout.assert_safe_var_path(backup)
        try:
            os.replace(destination, backup)
            previous_moved = True
            os.replace(staging, destination)
            _verify_installed(assets, prefix, destination)
            _assert_exact_knowledge_seed(destination)
        except Exception as error:
            if not previous_moved:
                raise RuntimeError(
                    "Knowledge seed refresh failed without changing the active seed"
                ) from error
            try:
                if destination.exists():
                    _assert_exact_knowledge_seed(destination)
                    shutil.rmtree(destination)
                if backup is None or not backup.exists():
                    raise RuntimeError("Knowledge seed backup is missing")
                os.replace(backup, destination)
                previous_moved = False
            except Exception as rollback_error:
                raise RuntimeError(
                    "Knowledge seed refresh failed and rollback could not be completed"
                ) from rollback_error
            raise RuntimeError(
                "Knowledge seed refresh failed; the previous seed was restored"
            ) from error
        if backup is not None and backup.exists():
            # The active replacement is verified. Failure to remove an inert hidden
            # backup is cleanup debt and must not invalidate the active seed.
            shutil.rmtree(backup, ignore_errors=True)
            backup = None
    finally:
        if staging.exists():
            shutil.rmtree(staging, ignore_errors=True)


def _invoke_model_cli(layout: RuntimeLayout, python: Path, root: Path, operation: str) -> None:
    arguments = {
        "verify": ("_verify-export", "--root", str(root)),
        "smoke": ("smoke", "--root", str(root)),
        "publish": ("publish", "--root", str(root)),
    }[operation]
    completed = run_clean_python(
        python,
        ("-m", "socialgraph_gfm.global_model.cli", *arguments),
        python_path=str(layout.gfm_package / "src"),
        cwd=layout.gfm_package,
        timeout=300,
    )
    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip()
        raise RuntimeError(f"Bundled SocialGraph-FM Global model {operation} failed: {detail}")


def _model_marker(bundle: RuntimeBundle) -> dict[str, Any]:
    model = bundle.document["model"]
    assets = [asset for asset in bundle.assets if asset.get("role") == "model"]
    return {
        "schemaVersion": MODEL_INSTALL_SCHEMA,
        "bundleVersion": bundle.document["bundleVersion"],
        "bundleManifestSha256": bundle.manifest_sha256,
        "modelVersionId": model["modelVersionId"],
        "modelVersionHash": model["modelVersionHash"],
        "artifactHash": model["artifactHash"],
        "corpusHash": model["corpusHash"],
        "sourceFileCount": len(assets),
    }


def _verify_registry_identity(
    registry: dict[str, Any], bundle: RuntimeBundle
) -> None:
    model = bundle.document["model"]
    if (
        registry.get("state") != "servingReady"
        or registry.get("modelVersionHash") != model["modelVersionHash"]
        or registry.get("artifactHash") != model["artifactHash"]
        or registry.get("corpusHash") != model["corpusHash"]
    ):
        raise RuntimeError("Published bundled SocialGraph-FM Global registry identity is invalid")


def install_model(layout: RuntimeLayout, python: Path, bundle: RuntimeBundle) -> None:
    assets = tuple(asset for asset in bundle.assets if asset.get("role") == "model")
    prefix = "bundles/models/socialgraph-global"
    destination = layout.model_root
    layout.assert_safe_var_path(destination)
    marker = _model_marker(bundle)
    marker_path = destination / "bundle-install.json"
    if destination.exists():
        try:
            installed_marker = json.loads(marker_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise RuntimeError(f"Existing SocialGraph-FM Global model is unmanaged: {destination}") from error
        identity_fields = (
            "schemaVersion",
            "bundleVersion",
            "modelVersionHash",
            "artifactHash",
            "corpusHash",
            "sourceFileCount",
        )
        if any(installed_marker.get(name) != marker.get(name) for name in identity_fields):
            raise RuntimeError("A different SocialGraph-FM Global model bundle is already installed")
        _verify_installed(assets, prefix, destination)
        _invoke_model_cli(layout, python, destination, "verify")
        registry_path = destination / "registry" / "socialgraph-global.json"
        try:
            registry = json.loads(registry_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise RuntimeError("Installed SocialGraph-FM Global serving registry is invalid") from error
        _verify_registry_identity(registry, bundle)
        registry_sha256 = file_sha256(registry_path)
        if installed_marker.get("registrySha256") not in {None, registry_sha256}:
            raise RuntimeError("Installed SocialGraph-FM Global serving registry bytes differ")
        if installed_marker.get("registrySha256") is None:
            atomic_write_json(marker_path, {**installed_marker, "registrySha256": registry_sha256})
        return
    staging = _staging_directory(destination.parent, destination.name)
    try:
        _copy_assets(layout, assets, prefix, staging)
        _verify_installed(assets, prefix, staging)
        _invoke_model_cli(layout, python, staging, "verify")
        _invoke_model_cli(layout, python, staging, "smoke")
        _invoke_model_cli(layout, python, staging, "publish")
        registry_path = staging / "registry" / "socialgraph-global.json"
        registry = json.loads(registry_path.read_text(encoding="utf-8"))
        _verify_registry_identity(registry, bundle)
        atomic_write_json(
            staging / "bundle-install.json",
            {**marker, "registrySha256": file_sha256(registry_path)},
        )
        os.replace(staging, destination)
    finally:
        if staging.exists():
            shutil.rmtree(staging)


def verify_installed_runtime_bundle(
    layout: RuntimeLayout, bundle: RuntimeBundle
) -> dict[str, Any]:
    """Verify every installed model byte plus the generated serving identity."""

    layout.assert_safe_var_path(layout.model_root)
    assets = tuple(asset for asset in bundle.assets if asset.get("role") == "model")
    marker = _model_marker(bundle)
    marker_path = layout.model_root / "bundle-install.json"
    registry_path = layout.model_root / "registry" / "socialgraph-global.json"
    try:
        installed_marker = json.loads(marker_path.read_text(encoding="utf-8"))
        registry = json.loads(registry_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError("Installed SocialGraph-FM Global model metadata is invalid") from error
    identity_fields = (
        "schemaVersion",
        "bundleVersion",
        "modelVersionHash",
        "artifactHash",
        "corpusHash",
        "sourceFileCount",
    )
    if any(installed_marker.get(name) != marker.get(name) for name in identity_fields):
        raise RuntimeError("Installed SocialGraph-FM Global bundle marker identity differs")
    _verify_installed(
        assets, "bundles/models/socialgraph-global", layout.model_root
    )
    _verify_registry_identity(registry, bundle)
    if installed_marker.get("registrySha256") != file_sha256(registry_path):
        raise RuntimeError("Installed SocialGraph-FM Global serving registry bytes differ")
    model = bundle.document["model"]
    return {
        "modelVersionHash": model["modelVersionHash"],
        "artifactHash": model["artifactHash"],
        "corpusHash": model["corpusHash"],
        "assetCount": len(assets),
    }


def verify_installed_runtime_seeds(
    layout: RuntimeLayout, bundle: RuntimeBundle
) -> dict[str, int]:
    """Verify immutable seeds and the identity marker for mutable reviewed cases."""

    checks = (
        (
            "knowledge",
            "bundles/governance/knowledge",
            layout.governance_root / "knowledge",
            lambda _asset: True,
        ),
        (
            "russia_example",
            "examples/governance/russia",
            layout.governance_root / "answer-packs" / "russia",
            lambda asset: not str(asset["path"]).endswith("/russia-full.zip"),
        ),
        (
            "russia_example",
            "examples/governance/russia",
            layout.governance_root / "samples" / "russia",
            lambda asset: str(asset["path"]).endswith("/russia-full.zip"),
        ),
        (
            "target_domain_example",
            "examples/governance/target-domain",
            layout.target_input_root,
            lambda _asset: True,
        ),
    )
    counts: dict[str, int] = {}
    for role, prefix, destination, predicate in checks:
        layout.assert_safe_var_path(destination)
        assets = tuple(
            asset
            for asset in bundle.assets
            if asset.get("role") == role and predicate(asset)
        )
        _verify_installed(assets, prefix, destination)
        counts[str(destination.relative_to(layout.var_root))] = len(assets)

    reviewed = tuple(
        asset for asset in bundle.assets if asset.get("role") == "reviewed_cases"
    )
    reviewed_root = layout.governance_root / "reviewed-cases"
    layout.assert_safe_var_path(reviewed_root)
    marker_path = reviewed_root / ".runtime-bundle-seed.json"
    try:
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError("Reviewed-case seed marker is invalid") from error
    expected = {
        "schemaVersion": SEED_INSTALL_SCHEMA,
        "bundleVersion": bundle.document["bundleVersion"],
        "sourcePrefix": "bundles/governance/reviewed-cases",
        "seedIdentity": _seed_identity(reviewed),
        "sourceFileCount": len(reviewed),
    }
    if marker != expected:
        raise RuntimeError("Reviewed-case seed identity differs from the runtime bundle")
    counts[str(reviewed_root.relative_to(layout.var_root))] = len(reviewed)
    return counts


def verify_gfm_runtime_state(
    layout: RuntimeLayout, python: Path
) -> dict[str, Any]:
    """Use the production GFM validators for registry and mutable case state."""

    layout.assert_safe_var_path(layout.model_root)
    reviewed_root = layout.governance_root / "reviewed-cases"
    layout.assert_safe_var_path(reviewed_root)
    completed = run_clean_python(
        python,
        (
            "-c",
            _RUNTIME_STATE_PROBE_SOURCE,
            str(layout.model_root),
            str(reviewed_root),
        ),
        python_path=str(layout.gfm_package / "src"),
        cwd=layout.gfm_package,
        timeout=180,
    )
    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip()
        raise RuntimeError(f"Installed GFM runtime state validation failed: {detail}")
    try:
        report = json.loads(completed.stdout)
    except json.JSONDecodeError as error:
        raise RuntimeError("Installed GFM runtime state returned invalid JSON") from error
    if (
        report.get("passed") is not True
        or not isinstance(report.get("modelVersionHash"), str)
        or not isinstance(report.get("reviewedCaseIndexHash"), str)
    ):
        raise RuntimeError("Installed GFM runtime state did not pass validation")
    return report


def run_checkpoint_forward_probe(
    layout: RuntimeLayout, python: Path
) -> dict[str, Any]:
    """Run all four frozen checkpoints only after installed assets verify exactly."""

    bundle = load_and_verify_bundle(layout)
    verify_installed_runtime_bundle(layout, bundle)
    completed = run_clean_python(
        python,
        (
            "-m",
            "socialgraph_gfm.global_model.cli",
            "forward-smoke",
            "--root",
            str(layout.model_root),
            "--device",
            "cpu",
        ),
        python_path=str(layout.gfm_package / "src"),
        cwd=layout.gfm_package,
        timeout=1800,
    )
    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip()
        raise RuntimeError(f"Four-checkpoint forward smoke failed: {detail}")
    try:
        report = json.loads(completed.stdout)
    except json.JSONDecodeError as error:
        raise RuntimeError("Four-checkpoint forward smoke returned invalid JSON") from error
    protocols = report.get("protocols")
    if (
        report.get("ok") is not True
        or report.get("command") != "forward-smoke"
        or report.get("schemaVersion") != CHECKPOINT_FORWARD_SCHEMA
        or report.get("passed") is not True
        or report.get("readOnly") is not True
        or report.get("device") != "cpu"
        or report.get("protocolCount") != 4
        or not isinstance(protocols, list)
        or [item.get("protocol") for item in protocols if isinstance(item, dict)]
        != ["global", "in_domain", "low_label", "cross_domain"]
    ):
        raise RuntimeError("Four-checkpoint forward smoke did not pass its release contract")

    summaries: list[dict[str, Any]] = []
    for protocol in protocols:
        if not isinstance(protocol, dict):
            raise RuntimeError("Four-checkpoint forward smoke protocol report is invalid")
        checkpoint = protocol.get("checkpoint")
        model = protocol.get("model")
        router = protocol.get("router")
        shape = protocol.get("shape")
        if (
            not isinstance(checkpoint, dict)
            or not isinstance(model, dict)
            or not isinstance(router, dict)
            or not isinstance(shape, dict)
            or protocol.get("finite") is not True
            or protocol.get("modelStateUnchanged") is not True
            or protocol.get("modalityContributionsValid") is not True
            or router.get("routesAllowed") is not True
            or router.get("weightsValid") is not True
            or not isinstance(protocol.get("allowedExpertMask"), list)
        ):
            raise RuntimeError(
                f"Checkpoint forward validation failed for {protocol.get('protocol')}"
            )
        hashes = (
            checkpoint.get("sha256"),
            model.get("modelVersionHash"),
            model.get("modelStateHash"),
            protocol.get("outputHash"),
        )
        if any(
            not isinstance(value, str)
            or len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)
            for value in hashes
        ):
            raise RuntimeError(
                f"Checkpoint forward identity is invalid for {protocol.get('protocol')}"
            )
        summaries.append(
            {
                "protocol": protocol["protocol"],
                "checkpointSha256": checkpoint["sha256"],
                "modelVersionHash": model["modelVersionHash"],
                "modelStateHash": model["modelStateHash"],
                "allowedExpertMask": protocol["allowedExpertMask"],
                "shape": shape,
                "outputHash": protocol["outputHash"],
            }
        )
    report_hash = report.get("reportHash")
    batch = report.get("batch")
    corpus = report.get("corpus")
    if (
        not isinstance(report_hash, str)
        or len(report_hash) != 64
        or not isinstance(batch, dict)
        or not isinstance(batch.get("batchHash"), str)
        or not isinstance(corpus, dict)
    ):
        raise RuntimeError("Four-checkpoint forward summary identity is invalid")
    return {
        "schemaVersion": report["schemaVersion"],
        "passed": True,
        "readOnly": True,
        "device": report["device"],
        "deviceName": report.get("deviceName"),
        "torchVersion": report.get("torchVersion"),
        "torchGeometricVersion": report.get("torchGeometricVersion"),
        "exportHash": report.get("exportHash"),
        "corpus": corpus,
        "batchHash": batch["batchHash"],
        "protocolCount": 4,
        "protocols": summaries,
        "reportHash": report_hash,
    }


def _seed_identity(assets: Iterable[dict[str, Any]]) -> str:
    text = "".join(
        f"{asset['path']}\t{asset['bytes']}\t{asset['sha256']}\n" for asset in assets
    )
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def install_seed(
    layout: RuntimeLayout,
    bundle: RuntimeBundle,
    *,
    role: str,
    prefix: str,
    destination: Path,
    mutable: bool = False,
    refresh_immutable: bool = False,
    predicate: Any | None = None,
) -> None:
    layout.assert_safe_var_path(destination)
    assets = tuple(
        asset
        for asset in bundle.assets
        if asset.get("role") == role and (predicate is None or predicate(asset))
    )
    if not assets:
        raise RuntimeError(f"Runtime bundle has no {role} seed assets")
    if refresh_immutable:
        expected_destination = layout.governance_root / "knowledge"
        if (
            mutable
            or role != "knowledge"
            or prefix != "bundles/governance/knowledge"
            or predicate is not None
            or os.path.normcase(os.path.abspath(destination))
            != os.path.normcase(os.path.abspath(expected_destination))
        ):
            raise RuntimeError("Immutable seed refresh is allowed only for Governance knowledge")
        relative_inventory = {
            _asset_relative(asset, prefix).as_posix() for asset in assets
        }
        if relative_inventory != KNOWLEDGE_SEED_FILES or len(assets) != len(
            KNOWLEDGE_SEED_FILES
        ):
            raise RuntimeError(
                "Knowledge seed refresh requires exactly knowledge.sqlite3 and manifest.json assets"
            )
    marker = {
        "schemaVersion": SEED_INSTALL_SCHEMA,
        "bundleVersion": bundle.document["bundleVersion"],
        "sourcePrefix": prefix,
        "seedIdentity": _seed_identity(assets),
        "sourceFileCount": len(assets),
    }
    marker_path = destination / ".runtime-bundle-seed.json"
    if destination.is_dir() and not any(destination.iterdir()):
        # Older layout initialization created this seed destination eagerly.
        # An empty directory has no user state and can be replaced atomically.
        destination.rmdir()
    if destination.exists():
        if mutable:
            try:
                existing = json.loads(marker_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as error:
                raise RuntimeError(f"Existing mutable seed is unmanaged: {destination}") from error
            if existing != marker:
                raise RuntimeError(f"A different mutable seed is installed: {destination}")
        elif refresh_immutable:
            _refresh_knowledge_seed(layout, assets, prefix, destination)
        else:
            _verify_installed(assets, prefix, destination)
        return
    staging = _staging_directory(destination.parent, destination.name)
    try:
        _copy_assets(layout, assets, prefix, staging)
        _verify_installed(assets, prefix, staging)
        if mutable:
            atomic_write_json(staging / marker_path.name, marker)
        os.replace(staging, destination)
    finally:
        if staging.exists():
            shutil.rmtree(staging)


def install_public_runtime_bundle(layout: RuntimeLayout, python: Path) -> RuntimeBundle:
    bundle = load_and_verify_bundle(layout)
    install_model(layout, python, bundle)
    install_seed(
        layout,
        bundle,
        role="knowledge",
        prefix="bundles/governance/knowledge",
        destination=layout.governance_root / "knowledge",
        refresh_immutable=True,
    )
    install_seed(
        layout,
        bundle,
        role="reviewed_cases",
        prefix="bundles/governance/reviewed-cases",
        destination=layout.governance_root / "reviewed-cases",
        mutable=True,
    )
    install_seed(
        layout,
        bundle,
        role="russia_example",
        prefix="examples/governance/russia",
        destination=layout.governance_root / "answer-packs" / "russia",
        predicate=lambda asset: not str(asset["path"]).endswith("/russia-full.zip"),
    )
    install_seed(
        layout,
        bundle,
        role="russia_example",
        prefix="examples/governance/russia",
        destination=layout.governance_root / "samples" / "russia",
        predicate=lambda asset: str(asset["path"]).endswith("/russia-full.zip"),
    )
    install_seed(
        layout,
        bundle,
        role="target_domain_example",
        prefix="examples/governance/target-domain",
        destination=layout.target_input_root,
    )
    return bundle


def run_full_gfm_probe(
    layout: RuntimeLayout,
    python: Path,
    *,
    use_installed_model: bool = False,
) -> dict[str, Any]:
    """Run a real Russia graph forward through the bundled Global checkpoint."""

    bundle = load_and_verify_bundle(layout)
    temporary: Path | None = None
    try:
        if use_installed_model:
            model_root = layout.model_root
            layout.assert_safe_var_path(model_root)
            if not model_root.is_dir():
                raise RuntimeError("The installed Global model is missing")
            verify_installed_runtime_bundle(layout, bundle)
        else:
            assets = tuple(asset for asset in bundle.assets if asset.get("role") == "model")
            temporary = _staging_directory(layout.temp_root, "forward-model")
            model_root = temporary
            _copy_assets(layout, assets, "bundles/models/socialgraph-global", model_root)
            _verify_installed(assets, "bundles/models/socialgraph-global", model_root)
            _invoke_model_cli(layout, python, model_root, "verify")
            _invoke_model_cli(layout, python, model_root, "smoke")
            _invoke_model_cli(layout, python, model_root, "publish")
        reports: list[dict[str, Any]] = []
        for index in range(1, 5):
            russia = (
                layout.project_root
                / "examples"
                / "governance"
                / "russia"
                / f"russia-{index:02d}.zip"
            )
            completed = run_clean_python(
                python,
                ("-c", _FORWARD_PROBE_SOURCE, str(model_root), str(russia)),
                python_path=str(layout.gfm_package / "src"),
                cwd=layout.gfm_package,
                timeout=600,
            )
            if completed.returncode != 0:
                detail = completed.stderr.strip() or completed.stdout.strip()
                raise RuntimeError(
                    f"Global {russia.name} forward probe failed: {detail}"
                )
            try:
                report = json.loads(completed.stdout)
            except json.JSONDecodeError as error:
                raise RuntimeError(
                    f"Global {russia.name} forward probe returned invalid JSON"
                ) from error
            if report.get("passed") is not True:
                raise RuntimeError(f"Global {russia.name} forward probe did not pass")
            reports.append({"input": russia.name, **report})
        model_hashes = {str(report.get("modelVersionHash")) for report in reports}
        devices = {str(report.get("device")) for report in reports}
        if len(model_hashes) != 1 or len(devices) != 1:
            raise RuntimeError("Russia forward probes disagreed on model or device identity")
        return {
            "passed": True,
            "device": next(iter(devices)),
            "modelVersionHash": next(iter(model_hashes)),
            "bundleCount": len(reports),
            "nodeCount": sum(int(report["nodeCount"]) for report in reports),
            "bundles": reports,
        }
    finally:
        if temporary is not None and temporary.exists():
            shutil.rmtree(temporary)
