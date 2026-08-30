"""Dataset import cache, persistence, handoff, and orchestration service."""

from __future__ import annotations

import hashlib
import json
import re
import secrets
import threading
import zipfile
from collections import OrderedDict
from datetime import UTC, datetime, timedelta
from pathlib import Path, PurePosixPath
from uuid import uuid4

import numpy as np
from fastapi import HTTPException, UploadFile, status

from ..config import Settings
from ..dataset_schemas import (
    ArrayDescriptor,
    DataGovernancePolicy,
    DatasetArtifact,
    DatasetArtifactDeletionImpact,
    DatasetArtifactLifecycleResponse,
    DatasetArtifactPurgeResponse,
    DatasetArtifactRef,
    DatasetFileProfile,
    DatasetInspection,
    DatasetInspectionCancellation,
    DatasetPreparationSpec,
    DatasetReadiness,
    DatasetReadinessIssue,
    FeatureSchema,
    GraphDatasetBinding,
    GraphDatasetHandoffRequest,
    GraphDatasetHandoffResponse,
    GraphHandoffReservation,
    GraphHandoffReserveRequest,
    GraphSemantics,
    LabelSchema,
    LinkPredictionProtocol,
    MaterializedDatasetBundle,
    NodeIdentitySchema,
    OrphanArtifactDirectory,
    OrphanArtifactRecoveryResponse,
    SourceFileDigest,
    SplitFoldCounts,
    SplitSet,
    TaskSpec,
    TrainingDatasetRef,
    TrainingRefResolveRequest,
    TrainingRefResolveResponse,
)
from ..dataset_storage import DatasetArtifactStore
from ..gfm_research import graph_envelope_research_compatibility

from .adapters import GraphVersionTargetDomainAdapter, _adapter_registry
from .archive_safety import _entry_map, _normalized_name, read_uploads
from .array_validation import (
    _edge_array_count,
    _graph_from_arrays,
    _profile,
    _read_npz,
    _validate_artifact_arrays,
)
from .contracts import (
    _array_descriptors,
    _array_role,
    _array_sha256,
    _attachment_arrays,
    _build_view,
    _canonical_graph_hash,
    _checksum,
    _contract_content_hash,
    _contract_manifest_hash,
    _dataset_role,
    _default_transform_recipe,
    _file_role,
    _graph_variants,
    _license_evidence,
    _license_policy,
    _normalise_feature_recipes,
    _payload_arrays,
    _source_file_digests,
    _split_sets,
    _training_ref_hash,
)
from .models import (
    MAX_STORED_INSPECTIONS,
    GraphPayload,
    InspectionRecord,
    UploadedEntry,
    _inspection_record_retained_bytes,
    dataset_preparation_hash_v1,
    graph_fact_hash_v1,
)

class DatasetImportService:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._inspections: OrderedDict[str, InspectionRecord] = OrderedDict()
        self._inspection_cache_bytes = 0
        self._inspection_cache_project_bytes: dict[str, int] = {}
        self._inspection_lock = threading.RLock()
        self.store = DatasetArtifactStore(settings.dataset_storage_root)
        self._handoff_lock = threading.RLock()

    @property
    def inspection_cache_bytes(self) -> int:
        with self._inspection_lock:
            return self._inspection_cache_bytes

    @property
    def inspection_cache_count(self) -> int:
        with self._inspection_lock:
            return len(self._inspections)

    def inspection_cache_project_bytes(self, project_id: str) -> int:
        with self._inspection_lock:
            return self._inspection_cache_project_bytes.get(project_id, 0)

    def _release_inspection_locked(self, inspection_id: str) -> bool:
        record = self._inspections.pop(inspection_id, None)
        if record is None:
            return False
        self._inspection_cache_bytes = max(
            0,
            self._inspection_cache_bytes - record.retained_bytes,
        )
        project_bytes = (
            self._inspection_cache_project_bytes.get(record.project_id, 0)
            - record.retained_bytes
        )
        if project_bytes > 0:
            self._inspection_cache_project_bytes[record.project_id] = project_bytes
        else:
            self._inspection_cache_project_bytes.pop(record.project_id, None)
        return True

    def _prune_inspections_locked(
        self,
        *,
        now: datetime,
        incoming_bytes: int = 0,
        incoming_project_id: str | None = None,
    ) -> None:
        expires_before = now - timedelta(seconds=self.settings.inspection_cache_ttl_seconds)
        expired = [
            inspection_id
            for inspection_id, record in self._inspections.items()
            if record.last_accessed_at <= expires_before
        ]
        for inspection_id in expired:
            self._release_inspection_locked(inspection_id)

        if incoming_project_id is not None:
            while (
                self._inspection_cache_project_bytes.get(incoming_project_id, 0)
                + incoming_bytes
                > self.settings.inspection_cache_max_project_bytes
            ):
                oldest_project_id = next(
                    (
                        inspection_id
                        for inspection_id, record in self._inspections.items()
                        if record.project_id == incoming_project_id
                    ),
                    None,
                )
                if oldest_project_id is None:
                    break
                self._release_inspection_locked(oldest_project_id)

        while self._inspections and (
            len(self._inspections) >= MAX_STORED_INSPECTIONS
            or self._inspection_cache_bytes + incoming_bytes
            > self.settings.inspection_cache_max_bytes
        ):
            oldest_id = next(iter(self._inspections))
            self._release_inspection_locked(oldest_id)

    async def inspect(
        self,
        files: list[UploadFile],
        selected_dataset: str | None = None,
        project_id: str = "local-default",
    ) -> DatasetInspection:
        entries = await read_uploads(files, self.settings)
        mapped = _entry_map(entries)
        adapter = next(
            adapter
            for adapter in _adapter_registry(mapped, selected_dataset)
            if adapter.matches(mapped)
        )
        result = adapter.inspect(mapped)
        inspection_id = str(uuid4())
        created_at = datetime.now(UTC)
        graph_handoff = (
            result.raw_manifest.get("graphVersionHandoff")
            if isinstance(result.raw_manifest, dict)
            else None
        )
        graph_fact_hash = (
            graph_handoff.get("graphFactHash")
            if isinstance(graph_handoff, dict)
            else None
        )
        response = DatasetInspection(
            id=inspection_id,
            detectedFormat=result.detected_format,
            status=result.status,
            profile=result.profile,
            files=[
                DatasetFileProfile(name=entry.name, size=len(entry.data), role=_file_role(entry.name))
                for entry in entries
            ],
            issues=result.issues,
            datasetCandidates=result.dataset_candidates,
            serverGraphFactHash=str(graph_fact_hash) if graph_fact_hash else None,
            createdAt=created_at,
        )
        record = InspectionRecord(
            project_id=project_id,
            response=response,
            payload=result.payload,
            checksum=_checksum(entries),
            source_files=[entry.name for entry in entries],
            source_file_digests=_source_file_digests(entries),
            dataset_name=result.dataset_name,
            raw_manifest=result.raw_manifest,
            derived_manifest=result.derived_manifest,
            attachments=result.attachments,
        )
        record.retained_bytes = _inspection_record_retained_bytes(record)
        if record.retained_bytes > self.settings.inspection_cache_max_entry_bytes:
            raise HTTPException(
                status_code=status.HTTP_413_CONTENT_TOO_LARGE,
                detail={
                    "code": "INSPECTION_CACHE_ENTRY_TOO_LARGE",
                    "retainedBytes": record.retained_bytes,
                    "maxBytes": self.settings.inspection_cache_max_entry_bytes,
                },
            )
        if record.retained_bytes > self.settings.inspection_cache_max_project_bytes:
            raise HTTPException(
                status_code=status.HTTP_413_CONTENT_TOO_LARGE,
                detail={
                    "code": "INSPECTION_CACHE_PROJECT_ENTRY_TOO_LARGE",
                    "projectId": project_id,
                    "retainedBytes": record.retained_bytes,
                    "maxProjectBytes": self.settings.inspection_cache_max_project_bytes,
                },
            )
        with self._inspection_lock:
            self._prune_inspections_locked(
                now=created_at,
                incoming_bytes=record.retained_bytes,
                incoming_project_id=project_id,
            )
            if (
                self._inspection_cache_project_bytes.get(project_id, 0)
                + record.retained_bytes
                > self.settings.inspection_cache_max_project_bytes
            ):
                raise HTTPException(
                    status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                    detail={
                        "code": "INSPECTION_CACHE_PROJECT_CAPACITY_UNAVAILABLE",
                        "projectId": project_id,
                    },
                )
            if (
                self._inspection_cache_bytes + record.retained_bytes
                > self.settings.inspection_cache_max_bytes
            ):
                raise HTTPException(
                    status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                    detail={"code": "INSPECTION_CACHE_CAPACITY_UNAVAILABLE"},
                )
            self._inspections[inspection_id] = record
            self._inspection_cache_bytes += record.retained_bytes
            self._inspection_cache_project_bytes[project_id] = (
                self._inspection_cache_project_bytes.get(project_id, 0)
                + record.retained_bytes
            )
        return response

    def commit(self, inspection_id: str) -> DatasetArtifact:
        with self._inspection_lock:
            now = datetime.now(UTC)
            self._prune_inspections_locked(now=now)
            record = self._inspections.get(inspection_id)
            if record is None:
                raise HTTPException(status_code=404, detail="数据检查记录不存在、已过期或已释放")
            record.last_accessed_at = now
            self._inspections.move_to_end(inspection_id)
            try:
                if record.response.status != "accepted" or record.payload is None:
                    raise HTTPException(
                        status_code=status.HTTP_409_CONFLICT,
                        detail={
                            "code": "DATASET_NOT_COMMITTABLE",
                            "status": record.response.status,
                            "issues": [
                                issue.model_dump(by_alias=True)
                                for issue in record.response.issues
                            ],
                        },
                    )
                return self._commit_payload(
                    record.payload,
                    inspection_id=inspection_id,
                    source_format=record.response.detected_format,
                    source_files=record.source_files,
                    source_file_digests=record.source_file_digests,
                    checksum=record.checksum,
                    dataset_name=record.dataset_name,
                    raw_manifest=record.raw_manifest,
                    derived_manifest=record.derived_manifest,
                    attachments=record.attachments,
                )
            finally:
                self._release_inspection_locked(inspection_id)

    def cancel_inspection(self, inspection_id: str) -> DatasetInspectionCancellation:
        with self._inspection_lock:
            self._prune_inspections_locked(now=datetime.now(UTC))
            if not self._release_inspection_locked(inspection_id):
                raise HTTPException(status_code=404, detail="数据检查记录不存在、已过期或已释放")
        return DatasetInspectionCancellation()

    def _commit_payload(
        self,
        payload: GraphPayload,
        *,
        inspection_id: str,
        source_format: str,
        source_files: list[str],
        source_file_digests: list[SourceFileDigest] | None = None,
        checksum: str,
        dataset_name: str | None = None,
        raw_manifest: dict[str, object] | None = None,
        derived_manifest: dict[str, object] | None = None,
        attachments: dict[str, bytes] | None = None,
        preparation_spec: DatasetPreparationSpec | None = None,
        stage: bool = False,
    ) -> DatasetArtifact:
        artifact_id = str(uuid4())
        canonical_hash = _canonical_graph_hash(payload)
        resolved_raw = dict(
            raw_manifest
            or {
                "schemaVersion": "2.2",
                "sourceFormat": source_format,
                "sourceFiles": source_files,
                "sourceChecksum": checksum,
                "license": "unknown",
            }
        )
        incoming_derived = dict(derived_manifest or {})
        raw_recipes = incoming_derived.get("transformRecipes") or [
            _default_transform_recipe(payload)
        ]
        stored_arrays = _payload_arrays(payload)
        stored_attachments = dict(attachments or {})
        descriptors, attachment_values = _array_descriptors(stored_arrays, stored_attachments)
        node_identity = NodeIdentitySchema(
            id="node-identity-v1",
            arrayName="node_id_map",
            kind=payload.node_identity_kind,
            count=payload.node_count,
            unique=True,
        )
        graph_semantics = GraphSemantics(
            directed=payload.directed,
            directedness=payload.directedness
            or ("directed" if payload.directed else "undirected"),
            edgeDirectedArray="edge_directed" if payload.edge_directed is not None else None,
            selfLoopPolicy="preserve",
            duplicateEdgePolicy="preserve",
            weighted=payload.edge_weights is not None
            and bool(np.any(~np.isnan(payload.edge_weights.astype(np.float64)))),
            temporal=payload.edge_timestamps is not None
            and bool(np.any(payload.edge_timestamps.astype(np.str_) != "")),
            heterogeneous=(
                payload.node_types is not None
                and bool(np.any(payload.node_types.astype(np.str_) != ""))
            )
            or (
                payload.edge_types is not None
                and bool(np.any(payload.edge_types.astype(np.str_) != ""))
            ),
        )
        feature_recipes = _normalise_feature_recipes(raw_recipes, stored_arrays)
        graph_variants = _graph_variants(payload, feature_recipes, raw_recipes)
        feature_arrays = sorted(
            {
                recipe.output_array
                for recipe in feature_recipes
                if recipe.output_array is not None and recipe.output_array in stored_arrays
            }
        )
        feature_schemas = [
            FeatureSchema(
                id=f"feature-{name}",
                arrayName=name,
                dtype=stored_arrays[name].dtype.str,
                shape=list(stored_arrays[name].shape),
            )
            for name in feature_arrays
        ]
        label_schemas: list[LabelSchema] = []
        if payload.labels is not None:
            valid_labels = payload.labels[payload.labels >= 0]
            values = sorted(np.unique(valid_labels).tolist()) if valid_labels.size else []
            label_schemas.append(
                LabelSchema(
                    id="node-labels-v1",
                    arrayName="y",
                    dtype=stored_arrays["y"].dtype.str,
                    shape=list(stored_arrays["y"].shape),
                    classCount=len(values),
                    classValues=values,
                )
            )
        split_sets = _split_sets(payload, resolved_raw, attachment_values)
        task_specs = (
            [
                TaskSpec(
                    id="node-classification-v1",
                    kind="node_classification",
                    target="node",
                    labelSchemaId="node-labels-v1",
                    splitSetIds=[item.id for item in split_sets],
                    evaluationProtocol="transductive",
                    metrics=["accuracy", "macro_f1"],
                )
            ]
            if label_schemas and split_sets
            else []
        )
        raw_link_protocol = resolved_raw.get("linkPredictionProtocol")
        if not isinstance(raw_link_protocol, dict):
            selected_manifest = resolved_raw.get("selectedDatasetManifest")
            if isinstance(selected_manifest, dict):
                raw_link_protocol = selected_manifest.get("linkPredictionProtocol")
        if isinstance(raw_link_protocol, dict):
            protocol = LinkPredictionProtocol.model_validate(raw_link_protocol)
            required_arrays = {
                protocol.message_passing_edge_array,
                protocol.train_positive_array,
                protocol.validation_positive_array,
                protocol.test_positive_array,
                *(
                    [protocol.validation_negative_array]
                    if protocol.validation_negative_array
                    else []
                ),
                *([protocol.test_negative_array] if protocol.test_negative_array else []),
                *([protocol.edge_year_array] if protocol.edge_year_array else []),
                *([protocol.edge_weight_array] if protocol.edge_weight_array else []),
            }
            missing = sorted(required_arrays.difference(stored_arrays))
            if missing:
                raise ValueError(f"linkPredictionProtocol 缺少数组: {', '.join(missing)}")
            link_split = SplitSet(
                id="ogb-time-split" if dataset_name == "ogbl-collab" else "temporal-link-split",
                kind="official",
                target="edge",
                representation="index",
                arrays={
                    "train": protocol.train_positive_array,
                    "validation": protocol.validation_positive_array,
                    "test": protocol.test_positive_array,
                },
                foldCount=1,
                foldCounts=[
                    SplitFoldCounts(
                        train=_edge_array_count(stored_arrays[protocol.train_positive_array]),
                        validation=_edge_array_count(stored_arrays[protocol.validation_positive_array]),
                        test=_edge_array_count(stored_arrays[protocol.test_positive_array]),
                    )
                ],
                source="OGB official temporal split" if dataset_name == "ogbl-collab" else None,
            )
            split_sets = [link_split]
            task_specs = [
                TaskSpec(
                    id="ogbl-collab-link-prediction-v1"
                    if dataset_name == "ogbl-collab"
                    else "temporal-link-prediction-v1",
                    kind="link_prediction",
                    target="edge",
                    splitSetIds=[link_split.id],
                    evaluationProtocol="temporal",
                    metrics=["Hits@50"] if dataset_name == "ogbl-collab" else ["hits_at_k"],
                    linkPredictionProtocol=protocol,
                )
            ]
        content_hash = _contract_content_hash(
            descriptors=descriptors,
            node_identity=node_identity,
            graph_semantics=graph_semantics,
            graph_variants=graph_variants,
            feature_schemas=feature_schemas,
            label_schemas=label_schemas,
            feature_recipes=feature_recipes,
            split_sets=split_sets,
            task_specs=task_specs,
            schema_version="2.2",
        )
        role = _dataset_role(resolved_raw)
        license_evidence = _license_evidence(resolved_raw)
        license_policy = _license_policy(
            resolved_raw,
            trusted_generated=source_format.startswith("trusted_local"),
        )
        if role == "pretraining_candidate" and (
            not source_format.startswith("trusted_local")
            or license_policy.status != "verified"
            or "pretraining" not in license_policy.allowed_uses
        ):
            role = "target_domain"
        if preparation_spec is not None:
            governance = preparation_spec.governance
        else:
            raw_governance = resolved_raw.get("dataGovernance")
            governance = (
                DataGovernancePolicy.model_validate(raw_governance)
                if isinstance(raw_governance, dict)
                else DataGovernancePolicy(
                    excludedAttributes=["*"]
                    if source_format == "graph_version_target_domain"
                    else []
                )
            )
        training_refs: list[TrainingDatasetRef] = []
        for recipe in feature_recipes:
            for split_set in split_sets:
                for task_spec in task_specs:
                    if split_set.id not in task_spec.split_set_ids:
                        continue
                    for split_fold in range(split_set.fold_count):
                        reference = TrainingDatasetRef(
                            schemaVersion="1.1",
                            artifactId=artifact_id,
                            contentHash=content_hash,
                            graphVariant=recipe.graph_variant,
                            splitSetId=split_set.id,
                            splitFold=split_fold,
                            featureRecipeId=recipe.id,
                            taskSpecId=task_spec.id,
                            datasetRole=role,
                            intendedUse="evaluation",
                        )
                        reference.ref_hash = _training_ref_hash(reference)
                        training_refs.append(reference)
        training_ref = training_refs[0] if training_refs else None
        resolved_derived: dict[str, object] = {
            **incoming_derived,
            "schemaVersion": "2.2",
            "canonicalGraphHash": canonical_hash,
            "contentHash": content_hash,
            "transformRecipes": raw_recipes,
            "splitNames": payload.split_names,
            "trainingReference": training_ref.model_dump(by_alias=True) if training_ref else None,
            "trainingReferences": [
                reference.model_dump(by_alias=True) for reference in training_refs
            ],
        }
        digests = list(source_file_digests or [])
        original_digests = resolved_raw.get("sourceFileDigests")
        selected_manifest = resolved_raw.get("selectedDatasetManifest")
        if not isinstance(original_digests, list) and isinstance(selected_manifest, dict):
            original_digests = selected_manifest.get("sourceFileDigests")
        if isinstance(original_digests, list):
            for raw_digest in original_digests:
                if not isinstance(raw_digest, dict):
                    continue
                raw_path = raw_digest.get("path")
                raw_sha = raw_digest.get("sha256")
                raw_size = raw_digest.get("size")
                if (
                    not isinstance(raw_path, str)
                    or not isinstance(raw_sha, str)
                    or re.fullmatch(r"[0-9a-f]{64}", raw_sha) is None
                    or not isinstance(raw_size, int)
                    or raw_size < 0
                ):
                    continue
                normalized = PurePosixPath(raw_path.replace("\\", "/"))
                if normalized.is_absolute() or ".." in normalized.parts:
                    continue
                digests.append(
                    SourceFileDigest(
                        path=f"original/{normalized.as_posix()}",
                        role="original_source",
                        size=raw_size,
                        sha256=raw_sha,
                    )
                )
        if not digests:
            digests = [
                SourceFileDigest(
                    path=name,
                    role=_file_role(name),
                    size=0,
                    sha256=checksum if re.fullmatch(r"[0-9a-f]{64}", checksum) else hashlib.sha256(name.encode()).hexdigest(),
                )
                for name in source_files
            ]
        manifest_hash = _contract_manifest_hash(
            content_hash=content_hash,
            dataset_role=role,
            source_file_digests=digests,
            license_policy=license_policy,
            raw_manifest=resolved_raw,
            derived_manifest=resolved_derived,
            schema_version="2.2",
            license_evidence=license_evidence,
            data_governance=governance,
        )
        resolved_derived["manifestHash"] = manifest_hash
        artifact = DatasetArtifact(
            schemaVersion="2.2",
            id=artifact_id,
            inspectionId=inspection_id,
            sourceFormat=source_format,
            sourceFiles=source_files,
            checksum=checksum,
            profile=_profile(payload),
            graphView=_build_view(payload, artifact_id),
            datasetName=dataset_name,
            canonicalGraphHash=canonical_hash,
            contentHash=content_hash,
            manifestHash=manifest_hash,
            datasetRole=role,
            sourceFileDigests=digests,
            arrays=descriptors,
            nodeIdentity=node_identity,
            graphSemantics=graph_semantics,
            graphVariants=graph_variants,
            featureSchemas=feature_schemas,
            labelSchemas=label_schemas,
            featureRecipes=feature_recipes,
            splitSets=split_sets,
            taskSpecs=task_specs,
            licensePolicy=license_policy,
            licenseEvidence=license_evidence,
            dataGovernance=governance,
            preparationSpec=preparation_spec,
            trainingRef=training_ref,
            trainingRefs=training_refs,
            scope="complete",
            rawManifest=resolved_raw,
            derivedManifest=resolved_derived,
            createdAt=datetime.now(UTC),
        )
        if stage:
            self.store.stage_artifact(
                artifact,
                stored_arrays,
                attachments=stored_attachments,
            )
        else:
            self.store.save_artifact(
                artifact,
                stored_arrays,
                attachments=stored_attachments,
            )
        return artifact

    def get_artifact(self, artifact_id: str) -> DatasetArtifact:
        artifact = self.store.get_artifact(artifact_id)
        if artifact is None:
            raise HTTPException(status_code=404, detail="数据产物不存在")
        return artifact

    def list_artifacts(self, *, include_trashed: bool = False) -> list[DatasetArtifactRef]:
        return self.store.list_artifacts(include_trashed=include_trashed)

    @staticmethod
    def _lifecycle_error(exc: ValueError) -> HTTPException:
        code = str(exc)
        if code in {"ARTIFACT_NOT_FOUND", "ORPHAN_NOT_FOUND"}:
            return HTTPException(status_code=404, detail={"code": code})
        if code in {"CONFIRMATION_MISMATCH", "ARTIFACT_ID_UNSAFE"}:
            return HTTPException(status_code=422, detail={"code": code})
        return HTTPException(status_code=409, detail={"code": code})

    def artifact_deletion_impact(self, artifact_id: str) -> DatasetArtifactDeletionImpact:
        try:
            return self.store.deletion_impact(artifact_id)
        except ValueError as exc:
            raise self._lifecycle_error(exc) from exc

    def trash_artifact(self, artifact_id: str) -> DatasetArtifactLifecycleResponse:
        try:
            lifecycle = self.store.set_lifecycle(artifact_id, "trashed")
            impact = self.store.deletion_impact(artifact_id)
        except ValueError as exc:
            raise self._lifecycle_error(exc) from exc
        return DatasetArtifactLifecycleResponse(lifecycle=lifecycle, impact=impact)

    def restore_artifact(self, artifact_id: str) -> DatasetArtifactLifecycleResponse:
        try:
            lifecycle = self.store.set_lifecycle(artifact_id, "active")
            impact = self.store.deletion_impact(artifact_id)
        except ValueError as exc:
            raise self._lifecycle_error(exc) from exc
        return DatasetArtifactLifecycleResponse(lifecycle=lifecycle, impact=impact)

    def purge_artifact(
        self,
        artifact_id: str,
        *,
        impact_hash: str,
        confirmation: str,
    ) -> DatasetArtifactPurgeResponse:
        try:
            cleanup_pending = self.store.purge_artifact(
                artifact_id,
                expected_impact_hash=impact_hash,
                confirmation=confirmation,
            )
        except ValueError as exc:
            raise self._lifecycle_error(exc) from exc
        return DatasetArtifactPurgeResponse(
            artifactId=artifact_id,
            cleanupPending=cleanup_pending,
        )

    def list_orphan_artifacts(self) -> list[OrphanArtifactDirectory]:
        return self.store.list_orphan_artifacts()

    def recover_orphan_artifact(self, artifact_id: str) -> OrphanArtifactRecoveryResponse:
        try:
            artifact, lifecycle = self.store.recover_orphan_artifact(artifact_id)
        except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
            if isinstance(exc, ValueError):
                raise self._lifecycle_error(exc) from exc
            raise HTTPException(
                status_code=409,
                detail={"code": "ORPHAN_NOT_RECOVERABLE"},
            ) from exc
        return OrphanArtifactRecoveryResponse(artifact=artifact, lifecycle=lifecycle)

    def reserve_graph_handoff(
        self,
        body: GraphHandoffReserveRequest,
    ) -> GraphHandoffReservation:
        token = secrets.token_urlsafe(32)
        expires = datetime.now(UTC) + timedelta(
            seconds=self.settings.graph_handoff_token_ttl_seconds
        )
        self.store.create_handoff_reservation(
            token_hash=hashlib.sha256(token.encode("utf-8")).hexdigest(),
            graph_version_id=body.graph_version_id,
            graph_fact_hash=body.graph_fact_hash,
            expires_at=expires,
        )
        return GraphHandoffReservation(
            token=token,
            graphVersionId=body.graph_version_id,
            graphFactHash=body.graph_fact_hash,
            expiresAt=expires,
        )

    def cancel_graph_handoff(self, token: str) -> None:
        try:
            self.store.cancel_handoff_token(
                token_hash=hashlib.sha256(token.encode("utf-8")).hexdigest(),
                now=datetime.now(UTC),
            )
        except ValueError as exc:
            raise HTTPException(status_code=409, detail={"code": str(exc)}) from exc

    def commit_graph_handoff(
        self,
        body: GraphDatasetHandoffRequest,
    ) -> GraphDatasetHandoffResponse:
        with self._handoff_lock:
            return self._commit_graph_handoff_locked(body)

    def _commit_graph_handoff_locked(
        self,
        body: GraphDatasetHandoffRequest,
    ) -> GraphDatasetHandoffResponse:
        envelope = body.envelope
        if envelope.graph_version_id != body.preparation.graph_version_id:
            raise HTTPException(
                status_code=409,
                detail={"code": "PREPARATION_GRAPH_VERSION_MISMATCH"},
            )
        available_attributes = {
            key for node in envelope.nodes for key in node.attributes
        }
        requested_attributes = set(body.preparation.feature_attributes)
        if body.preparation.label_attribute:
            requested_attributes.add(body.preparation.label_attribute)
        missing_attributes = sorted(requested_attributes.difference(available_attributes))
        if missing_attributes:
            raise HTTPException(
                status_code=422,
                detail={
                    "code": "PREPARATION_ATTRIBUTE_MISSING",
                    "attributes": missing_attributes,
                },
            )
        actual_hash = graph_fact_hash_v1(envelope)
        if envelope.graph_fact_hash is not None and envelope.graph_fact_hash != actual_hash:
            raise HTTPException(status_code=409, detail={"code": "GRAPH_FACT_HASH_MISMATCH"})
        preparation_hash = dataset_preparation_hash_v1(body.preparation)
        token_hash = hashlib.sha256(body.token.encode("utf-8")).hexdigest()
        existing = self.store.find_binding(
            graph_version_id=envelope.graph_version_id,
            graph_fact_hash=actual_hash,
            preparation_hash=preparation_hash,
        )
        if existing is not None:
            try:
                existing = self.store.commit_binding(
                    binding=existing,
                    token_hash=token_hash,
                    now=datetime.now(UTC),
                    allow_consumed=True,
                )
            except ValueError as exc:
                raise HTTPException(status_code=409, detail={"code": str(exc)}) from exc
            return GraphDatasetHandoffResponse(
                binding=existing,
                artifact=self.get_artifact(existing.artifact_id),
                reused=True,
                researchCompatibility=(
                    graph_envelope_research_compatibility(envelope)
                    if body.intended_use == "gfm_research"
                    else None
                ),
            )
        try:
            self.store.validate_handoff_token(
                token_hash=token_hash,
                graph_version_id=envelope.graph_version_id,
                graph_fact_hash=actual_hash,
                now=datetime.now(UTC),
            )
        except ValueError as exc:
            raise HTTPException(status_code=409, detail={"code": str(exc)}) from exc

        serialized = envelope.model_dump_json(by_alias=True).encode("utf-8")
        adapter = GraphVersionTargetDomainAdapter()
        result = adapter.inspect(
            {f"{envelope.graph_version_id}.sgfm-graph.json": UploadedEntry(
                name=f"{envelope.graph_version_id}.sgfm-graph.json", data=serialized
            )}
        )
        if result.status != "accepted" or result.payload is None:
            raise HTTPException(
                status_code=409,
                detail={
                    "code": "GRAPH_HANDOFF_REJECTED",
                    "issues": [item.model_dump(by_alias=True) for item in result.issues],
                },
            )
        artifact = self._commit_payload(
            result.payload,
            inspection_id=f"handoff:{envelope.graph_version_id}",
            source_format="graph_version_target_domain",
            source_files=[f"{envelope.graph_version_id}.sgfm-graph.json"],
            source_file_digests=[
                SourceFileDigest(
                    path=f"{envelope.graph_version_id}.sgfm-graph.json",
                    role="graph_version_handoff",
                    size=len(serialized),
                    sha256=hashlib.sha256(serialized).hexdigest(),
                )
            ],
            checksum=hashlib.sha256(serialized).hexdigest(),
            dataset_name=envelope.source_file,
            raw_manifest=result.raw_manifest,
            derived_manifest=result.derived_manifest,
            preparation_spec=body.preparation,
            stage=True,
        )
        now = datetime.now(UTC)
        binding = GraphDatasetBinding(
            id=str(uuid4()),
            graphVersionId=envelope.graph_version_id,
            graphFactHash=actual_hash,
            artifactId=artifact.id,
            preparationHash=preparation_hash,
            createdAt=now,
        )
        try:
            binding = self.store.activate_staged_handoff(
                artifact=artifact,
                binding=binding,
                token_hash=token_hash,
                now=now,
            )
        except ValueError as exc:
            raise HTTPException(status_code=409, detail={"code": str(exc)}) from exc
        reused = binding.artifact_id != artifact.id
        return GraphDatasetHandoffResponse(
            binding=binding,
            artifact=self.get_artifact(binding.artifact_id) if reused else artifact,
            reused=reused,
            researchCompatibility=(
                graph_envelope_research_compatibility(envelope)
                if body.intended_use == "gfm_research"
                else None
            ),
        )

    def _reference_issues(
        self,
        artifact: DatasetArtifact,
        reference: TrainingDatasetRef,
    ) -> tuple[list[DatasetReadinessIssue], list[DatasetReadinessIssue]]:
        blockers: list[DatasetReadinessIssue] = []
        warnings: list[DatasetReadinessIssue] = []

        def block(code: str, message: str) -> None:
            blockers.append(DatasetReadinessIssue(code=code, message=message))

        if reference.artifact_id != artifact.id or reference.content_hash != artifact.content_hash:
            block("REFERENCE_IDENTITY_MISMATCH", "训练引用与 Artifact 身份不一致。")
        variants = {item.id: item for item in artifact.graph_variants}
        recipes = {item.id: item for item in artifact.feature_recipes}
        split_sets = {item.id: item for item in artifact.split_sets}
        tasks = {item.id: item for item in artifact.task_specs}
        labels = {item.id: item for item in artifact.label_schemas}
        if reference.graph_variant not in variants:
            block("REFERENCE_TARGET_MISSING", "训练引用的 graphVariant 不存在。")
        recipe = recipes.get(reference.feature_recipe_id)
        if recipe is None:
            block("REFERENCE_TARGET_MISSING", "训练引用的 FeatureRecipe 不存在。")
        elif recipe.graph_variant != reference.graph_variant:
            block("REFERENCE_INCOMPATIBLE", "FeatureRecipe 与 graphVariant 不匹配。")
        split_set = split_sets.get(reference.split_set_id or "")
        if split_set is None:
            block("SPLIT_INCOMPATIBLE", "训练引用缺少可解析的 SplitSet。")
        else:
            if "train" not in split_set.arrays or "test" not in split_set.arrays:
                block("SPLIT_INCOMPATIBLE", "SplitSet 至少需要 train 与 test。")
            if split_set.kind != "few_shot" and "validation" not in split_set.arrays:
                block("SPLIT_INCOMPATIBLE", "正式评测 SplitSet 缺少 validation。")
            if reference.schema_version == "1.1" and reference.split_fold is None:
                block("SPLIT_FOLD_MISSING", "TrainingDatasetRef 1.1 必须固定 splitFold。")
            elif reference.split_fold is not None and reference.split_fold >= split_set.fold_count:
                block("SPLIT_FOLD_OUT_OF_RANGE", "TrainingDatasetRef 的 splitFold 越界。")
        task = tasks.get(reference.task_spec_id or "")
        if task is None:
            block("TASK_SPEC_MISSING", "训练引用缺少 TaskSpec。")
        else:
            if split_set is not None and split_set.id not in task.split_set_ids:
                block("SPLIT_INCOMPATIBLE", "TaskSpec 未声明该 SplitSet。")
            if task.label_schema_id and task.label_schema_id not in labels:
                block("LABEL_SCHEMA_MISSING", "TaskSpec 引用的 LabelSchema 不存在。")
            if recipe is not None and recipe.fit_scope == "all_nodes_transductive" and task.evaluation_protocol != "transductive":
                block("RECIPE_LEAKAGE_RISK", "全节点拟合 recipe 不能用于 inductive/temporal 任务。")
        if not artifact.feature_schemas:
            block("FEATURE_SCHEMA_MISSING", "Artifact 没有可训练的 FeatureSchema。")
        if artifact.node_identity is None:
            block("NODE_ID_MAP_MISSING", "Artifact 缺少可回映实体的节点身份数组。")
        license_policy = artifact.license_policy
        if license_policy is None or license_policy.status == "unknown":
            block("LICENSE_UNRESOLVED", "许可证用途尚未确认。")
        elif reference.intended_use not in license_policy.allowed_uses:
            block("LICENSE_USE_NOT_ALLOWED", "许可证不允许该训练用途。")
        if artifact.schema_version == "2.2" and license_policy is not None:
            evidence = {item.id: item for item in artifact.license_evidence}
            if any(item_id not in evidence for item_id in license_policy.evidence_ids):
                block("LICENSE_EVIDENCE_MISSING", "许可证策略引用了不存在的证据。")
            if license_policy.status == "verified" and not any(
                item.kind in {"official_metadata", "official_license"}
                for item in evidence.values()
            ):
                block("LICENSE_EVIDENCE_MISSING", "verified 许可证缺少官方证据。")
            if license_policy.status == "user_attested" and not any(
                item.kind == "user_attestation" for item in evidence.values()
            ):
                block("LICENSE_EVIDENCE_MISSING", "用户声明许可证缺少显式证明记录。")
        if reference.intended_use == "pretraining" and artifact.dataset_role != "pretraining_candidate":
            block("DATASET_ROLE_NOT_PRETRAINING", "目标域或 benchmark 数据不能自动作为预训练语料。")
        if artifact.dataset_role == "target_domain":
            warnings.append(
                DatasetReadinessIssue(
                    code="TARGET_DOMAIN_ONLY",
                    message="该数据仅作为用户目标域事实、适配或评测数据，不是基础模型预训练语料。",
                    severity="warning",
                )
            )
        return blockers, warnings

    def readiness(
        self,
        artifact_id: str,
        *,
        training_ref_hash: str | None = None,
        reference: TrainingDatasetRef | None = None,
    ) -> DatasetReadiness:
        artifact = self.get_artifact(artifact_id)
        now = datetime.now(UTC)
        if artifact.schema_version not in {"2.1", "2.2"}:
            return DatasetReadiness(
                artifactId=artifact.id,
                status="legacy",
                contentHash=artifact.content_hash,
                manifestHash=artifact.manifest_hash,
                blockers=[
                    DatasetReadinessIssue(
                        code="LEGACY_ARTIFACT_REIMPORT_REQUIRED",
                        message="v1/v2.0 Artifact 只读兼容；请重新导入生成 2.2。",
                    )
                ],
                checkedAt=now,
            )
        selected = reference
        if selected is None and training_ref_hash:
            selected = next(
                (item for item in artifact.training_refs if item.ref_hash == training_ref_hash),
                None,
            )
            if selected is None:
                return DatasetReadiness(
                    artifactId=artifact.id,
                    status="blocked",
                    contentHash=artifact.content_hash,
                    manifestHash=artifact.manifest_hash,
                    blockers=[
                        DatasetReadinessIssue(
                            code="REFERENCE_TARGET_MISSING",
                            message="指定 trainingRefHash 不属于该 Artifact。",
                        )
                    ],
                    checkedAt=now,
                )
        selected = selected or artifact.training_ref or next(iter(artifact.training_refs), None)
        try:
            stored_arrays = self.store.load_arrays(artifact.id)
            actual_descriptors = [
                ArrayDescriptor(
                    name=name,
                    role=_array_role(name),
                    dtype=value.dtype.str,
                    shape=list(value.shape),
                    sha256=_array_sha256(value),
                )
                for name, value in sorted(stored_arrays.items())
            ]
            attachment_paths = sorted(
                {
                    descriptor.name.split("#", 1)[0]
                    for descriptor in artifact.arrays
                    if "#" in descriptor.name
                }
            )
            for path in attachment_paths:
                data = self.store.load_attachment(artifact.id, path)
                values, descriptors = _attachment_arrays({path: data})
                actual_descriptors.extend(descriptors)
                stored_arrays.update(values)
            expected = {
                item.name: item.model_dump(mode="json", by_alias=True) for item in artifact.arrays
            }
            actual = {
                item.name: item.model_dump(mode="json", by_alias=True) for item in actual_descriptors
            }
            if expected != actual:
                raise ValueError("ARRAY_HASH_MISMATCH")
            if artifact.node_identity is None or artifact.graph_semantics is None or artifact.license_policy is None:
                raise ValueError("CONTRACT_MISSING")
            node_ids = stored_arrays.get(artifact.node_identity.array_name)
            if node_ids is None or node_ids.reshape(-1).size != artifact.node_identity.count:
                raise ValueError("NODE_ID_MAP_MISMATCH")
            if len({str(value) for value in node_ids.reshape(-1).tolist()}) != int(node_ids.size):
                raise ValueError("NODE_ID_MAP_NOT_UNIQUE")
            _validate_artifact_arrays(artifact, stored_arrays)
            recomputed_content = _contract_content_hash(
                descriptors=actual_descriptors,
                node_identity=artifact.node_identity,
                graph_semantics=artifact.graph_semantics,
                graph_variants=artifact.graph_variants,
                feature_schemas=artifact.feature_schemas,
                label_schemas=artifact.label_schemas,
                feature_recipes=artifact.feature_recipes,
                split_sets=artifact.split_sets,
                task_specs=artifact.task_specs,
                schema_version=artifact.schema_version,
            )
            recomputed_manifest = _contract_manifest_hash(
                content_hash=recomputed_content,
                dataset_role=artifact.dataset_role,
                source_file_digests=artifact.source_file_digests,
                license_policy=artifact.license_policy,
                raw_manifest=artifact.raw_manifest,
                derived_manifest=artifact.derived_manifest,
                schema_version=artifact.schema_version,
                license_evidence=artifact.license_evidence,
                data_governance=artifact.data_governance,
            )
            if recomputed_content != artifact.content_hash or recomputed_manifest != artifact.manifest_hash:
                raise ValueError("ARTIFACT_HASH_MISMATCH")
        except (FileNotFoundError, OSError, ValueError, zipfile.BadZipFile) as exc:
            return DatasetReadiness(
                artifactId=artifact.id,
                status="corrupt",
                contentHash=artifact.content_hash,
                manifestHash=artifact.manifest_hash,
                trainingRef=selected,
                blockers=[
                    DatasetReadinessIssue(
                        code="ARTIFACT_INTEGRITY_FAILURE",
                        message=f"Artifact 安全重读或哈希校验失败：{exc}",
                    )
                ],
                checkedAt=now,
            )
        if selected is None:
            return DatasetReadiness(
                artifactId=artifact.id,
                status="blocked",
                contentHash=artifact.content_hash,
                manifestHash=artifact.manifest_hash,
                blockers=[
                    DatasetReadinessIssue(
                        code="TRAINING_REFERENCE_MISSING",
                        message="尚无完整的 variant/recipe/split/task 组合。",
                    )
                ],
                checkedAt=now,
            )
        blockers, warnings = self._reference_issues(artifact, selected)
        expected_ref_hash = _training_ref_hash(selected)
        if selected.ref_hash != expected_ref_hash:
            blockers.append(
                DatasetReadinessIssue(
                    code="REFERENCE_HASH_MISMATCH",
                    message="TrainingDatasetRef refHash 不一致。",
                )
            )
        return DatasetReadiness(
            artifactId=artifact.id,
            status="blocked" if blockers else "ready",
            contentHash=artifact.content_hash,
            manifestHash=artifact.manifest_hash,
            trainingRef=selected,
            blockers=blockers,
            warnings=warnings,
            checkedAt=now,
        )

    def resolve_training_ref(
        self,
        body: TrainingRefResolveRequest,
    ) -> TrainingRefResolveResponse:
        artifact = self.get_artifact(body.artifact_id)
        reference = TrainingDatasetRef(
            schemaVersion="1.1" if artifact.schema_version == "2.2" else "1.0",
            artifactId=body.artifact_id,
            contentHash=body.content_hash,
            graphVariant=body.graph_variant,
            splitSetId=body.split_set_id,
            splitFold=body.split_fold if artifact.schema_version == "2.2" else None,
            featureRecipeId=body.feature_recipe_id,
            taskSpecId=body.task_spec_id,
            datasetRole=artifact.dataset_role,
            intendedUse=body.intended_use,
        )
        reference.ref_hash = _training_ref_hash(reference)
        return TrainingRefResolveResponse(
            reference=reference,
            readiness=self.readiness(artifact.id, reference=reference),
        )

    def materialize_contract(
        self,
        artifact_id: str,
        *,
        training_ref_hash: str,
    ) -> MaterializedDatasetBundle:
        """Read-only trainer boundary: validate and expose shapes, never raw paths."""

        artifact = self.get_artifact(artifact_id)
        reference = next(
            (item for item in artifact.training_refs if item.ref_hash == training_ref_hash),
            None,
        )
        if reference is None:
            raise HTTPException(status_code=404, detail={"code": "TRAINING_REFERENCE_NOT_FOUND"})
        readiness = self.readiness(artifact_id, reference=reference)
        if readiness.status != "ready":
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail={
                    "code": "DATASET_NOT_READY",
                    "readiness": readiness.model_dump(mode="json", by_alias=True),
                },
            )
        arrays = self.store.load_arrays(artifact_id)
        attachment_paths = sorted(
            {
                descriptor.name.split("#", 1)[0]
                for descriptor in artifact.arrays
                if "#" in descriptor.name
            }
        )
        for path in attachment_paths:
            values, _descriptors = _attachment_arrays(
                {path: self.store.load_attachment(artifact_id, path)}
            )
            arrays.update(values)
        recipe = next(item for item in artifact.feature_recipes if item.id == reference.feature_recipe_id)
        feature = arrays.get(recipe.output_array or "")
        if feature is None:
            raise HTTPException(status_code=409, detail={"code": "FEATURE_ARRAY_MISSING"})
        task = next(item for item in artifact.task_specs if item.id == reference.task_spec_id)
        split_set = next(item for item in artifact.split_sets if item.id == reference.split_set_id)
        split_sizes: dict[str, int] = {}
        for part, name in split_set.arrays.items():
            value = arrays.get(name)
            if value is None:
                raise HTTPException(status_code=409, detail={"code": "SPLIT_ARRAY_MISSING"})
            if split_set.target == "edge":
                split_sizes[part] = _edge_array_count(value)
                continue
            selected_values = (
                value[:, reference.split_fold or 0]
                if value.ndim == 2
                else value
            )
            split_sizes[part] = (
                int(np.count_nonzero(selected_values))
                if split_set.representation == "mask"
                else int(selected_values.size)
            )
        label_shape: list[int] | None = None
        if task.label_schema_id:
            label_schema = next(
                item for item in artifact.label_schemas if item.id == task.label_schema_id
            )
            label_shape = list(arrays[label_schema.array_name].shape)
        variant = next(item for item in artifact.graph_variants if item.id == reference.graph_variant)
        return MaterializedDatasetBundle(
            artifactId=artifact.id,
            trainingRefHash=reference.ref_hash,
            nodeCount=artifact.node_identity.count if artifact.node_identity else 0,
            edgeCount=_edge_array_count(arrays[variant.edge_index_array]),
            featureShape=list(feature.shape),
            labelShape=label_shape,
            splitSizes=split_sizes,
            taskKind=task.kind,
        )

    def import_trusted_package(
        self,
        package_path: str,
        *,
        job_id: str,
        source_path: str,
    ) -> list[DatasetArtifact]:
        """Import safe outputs from the authorized subprocess; never loads pickle/PT."""

        package_file = Path(package_path)
        digest = hashlib.sha256()
        with package_file.open("rb") as handle:
            while chunk := handle.read(1024 * 1024):
                digest.update(chunk)
        checksum = digest.hexdigest()
        artifacts: list[DatasetArtifact] = []
        try:
            archive = zipfile.ZipFile(package_file)
        except zipfile.BadZipFile as exc:
            raise ValueError("转换器输出的 SGFM 包损坏") from exc
        with archive:
            try:
                manifest = json.loads(archive.read("manifest.json"))
            except (KeyError, json.JSONDecodeError) as exc:
                raise ValueError("转换器输出缺少有效 manifest.json") from exc
            if manifest.get("schemaVersion") != "socialgraph-fm-dataset-package/1.0":
                raise ValueError("转换器输出的 schemaVersion 不受支持")
            datasets = manifest.get("datasets")
            if not isinstance(datasets, list):
                raise TypeError("转换器输出的数据集清单无效")
            names = set(archive.namelist())
            for item in datasets:
                if not isinstance(item, dict):
                    raise TypeError("转换器输出包含无效数据集条目")
                dataset_name = str(item.get("name", "unnamed"))[:200]
                graph_path = _normalized_name(str(item.get("path", "")))
                if graph_path not in names:
                    raise ValueError(f"转换器输出缺少图数组: {graph_path}")
                graph_entry = UploadedEntry(graph_path, archive.read(graph_path))
                payload = _graph_from_arrays(
                    _read_npz(
                        graph_entry,
                        trusted_generated=True,
                        trusted_max_bytes=self.settings.trusted_array_max_bytes,
                    )
                )
                if "directed" in item:
                    payload.directed = bool(item.get("directed"))
                episode_entries: list[dict[str, object]] = []
                attachments: dict[str, bytes] = {}
                digest_entries = [graph_entry]
                for episode in item.get("fewShotEpisodes", []):
                    if not isinstance(episode, dict) or not isinstance(episode.get("path"), str):
                        continue
                    episode_path = _normalized_name(str(episode["path"]))
                    if episode_path not in names:
                        raise ValueError(f"转换器输出缺少 few-shot episode: {episode_path}")
                    relative = f"episodes/{PurePosixPath(episode_path).name}"
                    episode_payload = archive.read(episode_path)
                    attachments[relative] = episode_payload
                    digest_entries.append(UploadedEntry(episode_path, episode_payload))
                    episode_entries.append({**episode, "artifactPath": relative})
                raw_manifest: dict[str, object] = {
                    "schemaVersion": "2.2",
                    "datasetName": dataset_name,
                    "sourcePath": source_path,
                    "sourceFormat": item.get("sourceFormat", "trusted_local_conversion"),
                    "sourceChecksum": checksum,
                    "sourceFingerprint": manifest.get("sourceFingerprint"),
                    "license": item.get("license", "unknown"),
                    "licensePolicy": item.get("licensePolicy"),
                    "licenseEvidence": item.get("licenseEvidence", []),
                    "dataGovernance": item.get("dataGovernance"),
                    "linkPredictionProtocol": item.get("linkPredictionProtocol"),
                    "datasetRole": item.get("datasetRole", "benchmark"),
                    "splitKind": item.get("splitKind", "source"),
                    "sourceFiles": item.get("sourceFiles", []),
                    "sourceFileDigests": item.get("sourceFileDigests", []),
                    "splitFiles": item.get("splitFiles", []),
                    "fewShotEpisodes": episode_entries,
                    "conversionSkipped": manifest.get("skipped", []),
                }
                canonical_hash = _canonical_graph_hash(payload)
                derived_manifest: dict[str, object] = {
                    "schemaVersion": "2.2",
                    "canonicalGraphHash": canonical_hash,
                    "transforms": item.get(
                        "transforms",
                        ["trusted_pickle_to_safe_npz", "preserve_source_topology"],
                    ),
                    "splitNames": payload.split_names,
                    "transformRecipes": item.get("transformRecipes", []),
                    "conversionJobId": job_id,
                }
                artifacts.append(
                    self._commit_payload(
                        payload,
                        inspection_id=job_id,
                        source_format=(
                            "trusted_local_ogb"
                            if dataset_name == "ogbl-collab"
                            else "trusted_local_pyg"
                        ),
                        source_files=[entry.name for entry in digest_entries],
                        source_file_digests=_source_file_digests(digest_entries),
                        checksum=checksum,
                        dataset_name=dataset_name,
                        raw_manifest=raw_manifest,
                        derived_manifest=derived_manifest,
                        attachments=attachments,
                    )
                )
        return artifacts
