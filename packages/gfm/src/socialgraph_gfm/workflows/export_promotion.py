"""Accepted model export and promotion.

The implementation is installed into the shared compatibility namespace by
:mod:`socialgraph_gfm.workflows` after all workflow stages are imported.
"""

# ruff: noqa: F403, F405
# mypy: disable-error-code=name-defined
from __future__ import annotations

from ._shared import *


def export_gfm(*, root: str | Path | None, experiment_id: str) -> dict[str, Any]:
    """Create an offline export reference only after formal acceptance."""

    layout = prepare_runtime_layout(root, operation="run")
    registry = _registry(layout)
    acceptance = registry.latest_acceptance(experiment_id=experiment_id)
    if acceptance is None or not acceptance.accepted:
        raise GfmAcceptanceRejected(
            "GFM export is blocked until the latest formal acceptance is true"
        )
    model_id = f"socialgraph-core-{acceptance.report_hash[:12]}"
    promotion = registry.promote_model(model_id=model_id, experiment_id=experiment_id)
    checkpoint = registry.get_checkpoint(acceptance.checkpoint_id)
    if checkpoint is None:
        raise RegistrationRejected("Accepted export checkpoint is absent")
    load_gfm_checkpoint(checkpoint, map_location="cpu")
    export_dir = layout.exports / model_id
    if export_dir.exists():
        manifest = read_json_object(export_dir / "export-manifest.json")
        if (
            manifest.get("acceptanceReportHash") != acceptance.report_hash
            or manifest.get("checkpointId") != checkpoint.checkpoint_id
            or manifest.get("checkpointSha256") != checkpoint.artifact_sha256
            or manifest.get("checkpointStateHash") != checkpoint.state_hash
            or manifest.get("deliveryEvidenceReportHashes")
            != list(acceptance.delivery_evidence_report_hashes)
        ):
            raise RegistrationRejected("Existing export directory has different evidence")
        exported = export_dir / str(manifest.get("checkpointFile", ""))
        if not exported.is_file() or file_sha256(exported) != manifest.get("checkpointSha256"):
            raise RegistrationRejected("Existing export checkpoint failed rehash")
        checked_manifest = dict(manifest)
        export_hash = checked_manifest.pop("exportHash", None)
        if export_hash != canonical_sha256(checked_manifest):
            raise RegistrationRejected("Existing export manifest hash is invalid")
    else:
        staging = layout.exports / f".{model_id}.{uuid.uuid4().hex}.tmp"
        if staging.exists():
            raise RegistrationRejected("Owned export staging identity already exists")
        staging.mkdir(parents=False)
        try:
            artifact = Path(checkpoint.artifact_path)
            copied_checkpoint = staging / artifact.name
            shutil.copy2(artifact, copied_checkpoint)
            payload = load_gfm_checkpoint(checkpoint, map_location="cpu")
            components = payload.get("components")
            required = {
                "collaboration",
                "collaboration_config",
                "newcomer",
                "newcomer_config",
                "suite_config",
            }
            if not isinstance(components, dict) or set(components) != required:
                raise RegistrationRejected(
                    "Accepted export is not the complete core plus two-head product suite"
                )
            product_configs = {
                task: components[f"{task}_config"] for task in ("collaboration", "newcomer")
            }
            for task, product_config in product_configs.items():
                if (
                    not isinstance(product_config, dict)
                    or "featureTransform" not in product_config
                    or "temperature" not in product_config
                    or "protocolHash" not in product_config
                ):
                    raise RegistrationRejected(
                        f"Exported {task} head lacks transform/calibration/protocol evidence"
                    )
            suite_config = components["suite_config"]
            if not isinstance(suite_config, dict):
                raise RegistrationRejected("Exported suite configuration is invalid")
            checked_suite_config = dict(suite_config)
            suite_config_hash = checked_suite_config.pop("taskConfigHash", None)
            if suite_config_hash != canonical_sha256(checked_suite_config):
                raise RegistrationRejected("Exported suite configuration hash is invalid")
            source_bindings = suite_config.get("taskCheckpoints")
            if not isinstance(source_bindings, dict) or set(source_bindings) != {
                "collaboration",
                "newcomer",
            }:
                raise RegistrationRejected("Exported suite lacks its two source bindings")
            manifest = {
                "schemaVersion": "gfm.offline-export/1.0",
                "modelId": model_id,
                "experimentId": experiment_id,
                "checkpointId": checkpoint.checkpoint_id,
                "checkpointFile": copied_checkpoint.name,
                "checkpointSha256": file_sha256(copied_checkpoint),
                "checkpointStateHash": checkpoint.state_hash,
                "componentStateHashes": {
                    task: _state_digest(components[task]) for task in ("collaboration", "newcomer")
                },
                "productConfigHashes": {
                    task: canonical_sha256(product_configs[task])
                    for task in ("collaboration", "newcomer")
                },
                "suiteConfigHash": suite_config_hash,
                "sourceBindings": source_bindings,
                "featureTransformHashes": {
                    task: canonical_sha256(product_configs[task]["featureTransform"])
                    for task in ("collaboration", "newcomer")
                },
                "temperatures": {
                    task: float(product_configs[task]["temperature"])
                    for task in ("collaboration", "newcomer")
                },
                "protocolHashes": {
                    task: product_configs[task]["protocolHash"]
                    for task in ("collaboration", "newcomer")
                },
                "pretrainConfigHash": acceptance.config_hash,
                "acceptanceReportHash": acceptance.report_hash,
                "deliveryEvidenceReportHashes": list(acceptance.delivery_evidence_report_hashes),
                "servingReady": False,
                "promotion": promotion,
            }
            manifest["exportHash"] = canonical_sha256(manifest)
            atomic_write_json(staging / "export-manifest.json", manifest)
            # Rehash every staged object before the single same-volume publish.
            if (
                file_sha256(copied_checkpoint) != checkpoint.artifact_sha256
                or read_json_object(staging / "export-manifest.json") != manifest
            ):
                raise RegistrationRejected("Staged export failed its final integrity check")
            os.replace(staging, export_dir)
        finally:
            # ``staging`` is a UUID-owned child of the fixed exports root.  It
            # remains present only when publication failed before os.replace.
            if staging.exists():
                try:
                    staging.resolve().relative_to(layout.exports.resolve())
                except ValueError as error:
                    raise RegistrationRejected(
                        "Export staging escaped the runtime exports root"
                    ) from error
                shutil.rmtree(staging)
    return {
        "schemaVersion": "gfm.workflow-export/1.0",
        "ok": True,
        "modelId": model_id,
        "experimentId": experiment_id,
        "export": str(export_dir / "export-manifest.json"),
        "servingReady": False,
    }


__all__ = [
    "export_gfm",
]
