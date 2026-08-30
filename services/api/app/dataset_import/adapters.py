"""Format detectors and safe, data-only dataset adapters."""

from __future__ import annotations

import json
import re
from pathlib import PurePosixPath

import numpy as np
from fastapi import HTTPException
from pydantic import ValidationError

from ..dataset_schemas import (
    DatasetProfile,
    GraphVersionTargetDomainEnvelope,
)

from .archive_safety import _find_suffix, _normalized_name
from .array_validation import (
    _combine_split_sources,
    _graph_from_arrays,
    _merge_payload_splits,
    _profile,
    _read_npz,
    _validated_split_arrays,
)
from .contracts import _default_transform_recipe
from .models import (
    MAX_EDGES,
    MAX_NODES,
    AdapterResult,
    DatasetAdapter,
    GraphPayload,
    UploadedEntry,
    _issue,
    graph_fact_hash_v1,
)

class TorchPygArchiveDetector:
    def matches(self, entries: dict[str, UploadedEntry]) -> bool:
        return any(key.endswith((".pt", ".pth")) for key in entries)

    def inspect(self, entries: dict[str, UploadedEntry]) -> AdapterResult:
        files = [entry.name for key, entry in entries.items() if key.endswith((".pt", ".pth"))]
        return AdapterResult(
            detected_format="torch_pyg_archive",
            status="conversion_required",
            profile=None,
            issues=[
                _issue(
                    "TRUSTED_LOCAL_CONVERSION_REQUIRED",
                    "PT/PTH 可能执行反序列化代码；Web 请求只检测，不加载。请使用受信本地转换命令。",
                    file=files[0],
                )
            ],
        )


class LegacyPlanetoidPickleDetector:
    _pattern = re.compile(
        r"(?:^|/)ind\.[^/]+\.(?:x|y|tx|ty|allx|ally|graph)$",
        re.IGNORECASE,
    )

    def matches(self, entries: dict[str, UploadedEntry]) -> bool:
        return any(self._pattern.search(key) for key in entries)

    def inspect(self, entries: dict[str, UploadedEntry]) -> AdapterResult:
        file = next(entry.name for key, entry in entries.items() if self._pattern.search(key))
        return AdapterResult(
            detected_format="legacy_planetoid_pickle",
            status="conversion_required",
            profile=None,
            issues=[
                _issue(
                    "TRUSTED_LOCAL_CONVERSION_REQUIRED",
                    "Planetoid pickle 只在显式信任的离线转换中读取；Web 请求不会调用 pickle.load。",
                    file=file,
                )
            ],
        )


class GraphVersionTargetDomainAdapter:
    """Strict, data-only GraphVersion handoff; never grants training permission."""

    _suffix = ".sgfm-graph.json"

    def matches(self, entries: dict[str, UploadedEntry]) -> bool:
        return any(key.endswith(self._suffix) for key in entries)

    def inspect(self, entries: dict[str, UploadedEntry]) -> AdapterResult:
        candidates = [entry for key, entry in entries.items() if key.endswith(self._suffix)]
        if len(entries) != 1 or len(candidates) != 1:
            return AdapterResult(
                detected_format="graph_version_target_domain",
                status="rejected",
                profile=None,
                issues=[
                    _issue(
                        "GRAPH_VERSION_SINGLE_FILE_REQUIRED",
                        ".sgfm-graph.json 必须作为单一文件提交，不能与其他附件混合。",
                    )
                ],
            )
        entry = candidates[0]
        try:
            decoded = entry.data.decode("utf-8")
            raw = json.loads(decoded)
            envelope = GraphVersionTargetDomainEnvelope.model_validate(raw)
        except UnicodeDecodeError:
            return AdapterResult(
                detected_format="graph_version_target_domain",
                status="rejected",
                profile=None,
                issues=[_issue("GRAPH_VERSION_UTF8_REQUIRED", "GraphVersion 交接文件必须是 UTF-8。", file=entry.name)],
            )
        except (json.JSONDecodeError, ValidationError) as exc:
            return AdapterResult(
                detected_format="graph_version_target_domain",
                status="rejected",
                profile=None,
                issues=[
                    _issue(
                        "GRAPH_VERSION_SCHEMA_INVALID",
                        f"GraphVersion 交接合同无效：{exc}",
                        file=entry.name,
                    )
                ],
            )

        actual_fact_hash = graph_fact_hash_v1(envelope)
        if envelope.graph_fact_hash is not None and envelope.graph_fact_hash != actual_fact_hash:
            return AdapterResult(
                detected_format="graph_version_target_domain",
                status="rejected",
                profile=None,
                issues=[
                    _issue(
                        "GRAPH_FACT_HASH_MISMATCH",
                        "GraphFactHash v1 与后端重算结果不一致。",
                        file=entry.name,
                    )
                ],
            )

        if len(envelope.nodes) > MAX_NODES or len(envelope.edges) > MAX_EDGES:
            return AdapterResult(
                detected_format="graph_version_target_domain",
                status="rejected",
                profile=None,
                issues=[_issue("GRAPH_VERSION_LIMIT_EXCEEDED", "GraphVersion 节点或边数量超过安全上限。", file=entry.name)],
            )
        node_ids = [node.id for node in envelope.nodes]
        if len(set(node_ids)) != len(node_ids):
            return AdapterResult(
                detected_format="graph_version_target_domain",
                status="rejected",
                profile=None,
                issues=[_issue("GRAPH_VERSION_NODE_ID_DUPLICATE", "GraphVersion 节点 ID 必须唯一。", file=entry.name)],
            )
        edge_ids = [edge.id for edge in envelope.edges]
        if len(set(edge_ids)) != len(edge_ids):
            return AdapterResult(
                detected_format="graph_version_target_domain",
                status="rejected",
                profile=None,
                issues=[_issue("GRAPH_VERSION_EDGE_ID_DUPLICATE", "GraphVersion 边 ID 必须唯一。", file=entry.name)],
            )
        node_index = {node_id: index for index, node_id in enumerate(node_ids)}
        dangling = next(
            (
                edge
                for edge in envelope.edges
                if edge.source not in node_index or edge.target not in node_index
            ),
            None,
        )
        if dangling is not None:
            return AdapterResult(
                detected_format="graph_version_target_domain",
                status="rejected",
                profile=None,
                issues=[
                    _issue(
                        "GRAPH_VERSION_DANGLING_ENDPOINT",
                        f"边 {dangling.id} 引用了不存在的节点；禁止静默补点。",
                        file=entry.name,
                    )
                ],
            )

        explicit = {edge.directed for edge in envelope.edges if edge.directed is not None}
        inconsistent = (
            (envelope.directedness == "directed" and explicit.difference({True}))
            or (envelope.directedness == "undirected" and explicit.difference({False}))
            or (envelope.directedness == "unspecified" and explicit)
            or (envelope.directedness == "mixed" and len(explicit) < 2 and all(edge.directed is not None for edge in envelope.edges))
        )
        if inconsistent:
            return AdapterResult(
                detected_format="graph_version_target_domain",
                status="rejected",
                profile=None,
                issues=[
                    _issue(
                        "GRAPH_VERSION_DIRECTEDNESS_MISMATCH",
                        "图级 directedness 与逐边 directed 事实不一致。",
                        file=entry.name,
                    )
                ],
            )

        edge_index = np.asarray(
            [
                [node_index[edge.source] for edge in envelope.edges],
                [node_index[edge.target] for edge in envelope.edges],
            ],
            dtype=np.int64,
        ).reshape(2, len(envelope.edges))
        default_directed = {
            "directed": 1,
            "undirected": 0,
            "mixed": -1,
            "unspecified": -1,
        }[envelope.directedness]
        edge_directed = np.asarray(
            [
                default_directed if edge.directed is None else int(edge.directed)
                for edge in envelope.edges
            ],
            dtype=np.int8,
        )
        def canonical_attributes(value: object) -> str:
            return json.dumps(
                value,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
        payload = GraphPayload(
            node_count=len(envelope.nodes),
            edge_index=edge_index,
            node_ids=np.asarray(node_ids, dtype=np.str_),
            node_labels=np.asarray([node.label for node in envelope.nodes], dtype=np.str_),
            node_types=np.asarray([node.node_type or "" for node in envelope.nodes], dtype=np.str_),
            node_attributes=np.asarray(
                [canonical_attributes(node.attributes) for node in envelope.nodes],
                dtype=np.str_,
            ),
            edge_ids=np.asarray(edge_ids, dtype=np.str_),
            edge_types=np.asarray([edge.edge_type or "" for edge in envelope.edges], dtype=np.str_),
            edge_weights=np.asarray(
                [np.nan if edge.weight is None else edge.weight for edge in envelope.edges],
                dtype=np.float64,
            ),
            edge_timestamps=np.asarray([edge.timestamp or "" for edge in envelope.edges], dtype=np.str_),
            edge_directed=edge_directed,
            edge_attributes=np.asarray(
                [canonical_attributes(edge.attributes) for edge in envelope.edges],
                dtype=np.str_,
            ),
            node_identity_kind="source",
            directed=envelope.directedness in {"directed", "mixed"},
            directedness=envelope.directedness,
        )
        raw_manifest: dict[str, object] = {
            "schemaVersion": "2.2",
            "sourceFormat": "graph_version_target_domain",
            "datasetRole": "target_domain",
            "license": "unknown",
            "graphVersionHandoff": {
                "schemaVersion": envelope.schema_version,
                "graphVersionId": envelope.graph_version_id,
                "contentHash": envelope.content_hash,
                "buildSpecHash": envelope.build_spec_hash,
                "sourceFile": envelope.source_file,
                "directedness": envelope.directedness,
                "graphFactHash": actual_fact_hash,
            },
        }
        return AdapterResult(
            detected_format="graph_version_target_domain",
            status="accepted",
            profile=_profile(payload),
            issues=[],
            payload=payload,
            dataset_name=envelope.source_file,
            raw_manifest=raw_manifest,
            derived_manifest={
                "schemaVersion": "2.2",
                "transforms": ["browser_graph_version_handoff"],
                "transformRecipes": [_default_transform_recipe(payload)],
            },
        )


class SafeGraphNpzAdapter:
    def matches(self, entries: dict[str, UploadedEntry]) -> bool:
        return any(key.endswith(".npz") for key in entries)

    def inspect(self, entries: dict[str, UploadedEntry]) -> AdapterResult:
        graph_entries: list[tuple[UploadedEntry, dict[str, np.ndarray]]] = []
        split_sources: list[tuple[str, dict[str, np.ndarray]]] = []
        try:
            for key, entry in sorted(entries.items()):
                if not key.endswith(".npz"):
                    continue
                arrays = _read_npz(entry)
                if "edge_index" in arrays:
                    graph_entries.append((entry, arrays))
                else:
                    split_sources.append((entry.name, arrays))
            if not graph_entries:
                return StrictSplitNpzAdapter().inspect(entries)
            if len(graph_entries) != 1:
                raise ValueError("一次导入只能包含一个 GraphNPZ 主图")
            _graph_entry, arrays = graph_entries[0]
            payload = _graph_from_arrays(arrays)
            external_splits = _combine_split_sources(split_sources, payload.node_count)
            _merge_payload_splits(payload, external_splits)
        except ValueError as exc:
            return AdapterResult(
                detected_format="graph_npz",
                status="rejected",
                profile=None,
                issues=[_issue("INVALID_SAFE_NPZ", str(exc))],
            )
        return AdapterResult(
            detected_format="graph_npz",
            status="accepted",
            profile=_profile(payload),
            issues=[
                _issue(
                    "ADDITIONAL_FILES_IGNORED",
                    "除 GraphNPZ 与严格 split NPZ 外的文件未参与构图。",
                    severity="warning",
                )
            ]
            if len(entries) > 1 + len(split_sources)
            else [],
            payload=payload,
        )


class StrictSplitNpzAdapter:
    def matches(self, entries: dict[str, UploadedEntry]) -> bool:
        return any(key.endswith(".npz") for key in entries)

    def inspect(self, entries: dict[str, UploadedEntry]) -> AdapterResult:
        try:
            split_sources: list[tuple[str, dict[str, np.ndarray]]] = []
            for key, entry in sorted(entries.items()):
                if key.endswith(".npz"):
                    arrays = _read_npz(entry)
                    if not arrays:
                        raise ValueError("split NPZ 不能为空")
                    splits = _validated_split_arrays(arrays)
                    if set(splits) != set(arrays):
                        raise ValueError("split NPZ 只能包含 mask 或 idx 数组")
                    split_sources.append((entry.name, arrays))
            combined = _combine_split_sources(split_sources, None)
        except ValueError as exc:
            return AdapterResult(
                detected_format="strict_split_npz",
                status="rejected",
                profile=None,
                issues=[_issue("INVALID_SPLIT_NPZ", str(exc))],
            )
        return AdapterResult(
            detected_format="strict_split_npz",
            status="mapping_required",
            profile=DatasetProfile(splitNames=list(combined)),
            issues=[
                _issue(
                    "GRAPH_FILE_REQUIRED",
                    "该文件只包含训练划分；请同时提供 GraphNPZ 或 Geom-GCN 主图。",
                )
            ],
        )


def _decode_text(entry: UploadedEntry) -> str:
    try:
        return entry.data.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise ValueError(f"文本必须使用 UTF-8: {entry.name}") from exc


class GeomGcnTextDirectoryAdapter:
    def matches(self, entries: dict[str, UploadedEntry]) -> bool:
        return bool(
            _find_suffix(entries, "out1_graph_edges.txt")
            or _find_suffix(entries, "out1_node_feature_label.txt")
        )

    def inspect(self, entries: dict[str, UploadedEntry]) -> AdapterResult:
        edge_file = _find_suffix(entries, "out1_graph_edges.txt")
        node_file = _find_suffix(entries, "out1_node_feature_label.txt")
        if edge_file is None or node_file is None:
            return AdapterResult(
                detected_format="geom_gcn_text",
                status="mapping_required",
                profile=None,
                issues=[
                    _issue(
                        "GEOM_GCN_PAIR_REQUIRED",
                        "Geom-GCN 必须同时包含节点特征标签文件和边文件。",
                    )
                ],
            )
        try:
            node_ids: list[str] = []
            seen_node_ids: set[str] = set()
            labels: list[int] = []
            feature_rows: list[np.ndarray] = []
            feature_dimension: int | None = None
            for line_no, line in enumerate(_decode_text(node_file).splitlines(), start=1):
                if not line.strip() or line_no == 1 and line.casefold().startswith("node_id"):
                    continue
                parts = line.rstrip("\r\n").split("\t")
                if len(parts) != 3:
                    raise ValueError(f"节点文件第 {line_no} 行必须为三列")
                node_id, feature_text, label_text = parts
                if not node_id or node_id in seen_node_ids:
                    raise ValueError(f"节点文件第 {line_no} 行 ID 为空或重复")
                dimension = feature_text.count(",") + 1 if feature_text else 0
                if feature_dimension is None:
                    feature_dimension = dimension
                elif dimension != feature_dimension:
                    raise ValueError(f"节点文件第 {line_no} 行特征维度不一致")
                try:
                    label = int(label_text)
                except ValueError as exc:
                    raise ValueError(f"节点文件第 {line_no} 行标签不是整数") from exc
                node_ids.append(node_id)
                seen_node_ids.add(node_id)
                labels.append(label)
                feature_rows.append(np.fromstring(feature_text, sep=",", dtype=np.float32))
                if len(node_ids) > MAX_NODES:
                    raise ValueError("节点数量超过限制")
            node_to_index = {node_id: index for index, node_id in enumerate(node_ids)}
            edge_pairs: list[tuple[int, int]] = []
            for line_no, line in enumerate(_decode_text(edge_file).splitlines(), start=1):
                if not line.strip() or line_no == 1 and line.casefold().startswith("node_id"):
                    continue
                parts = line.split()
                if len(parts) != 2:
                    raise ValueError(f"边文件第 {line_no} 行必须为两列")
                if parts[0] not in node_to_index or parts[1] not in node_to_index:
                    raise ValueError(f"边文件第 {line_no} 行引用了未知节点")
                edge_pairs.append((node_to_index[parts[0]], node_to_index[parts[1]]))
                if len(edge_pairs) > MAX_EDGES:
                    raise ValueError("边数量超过限制")
            edge_index = (
                np.asarray(edge_pairs, dtype=np.int64).T
                if edge_pairs
                else np.empty((2, 0), dtype=np.int64)
            )
            payload = GraphPayload(
                node_count=len(node_ids),
                edge_index=edge_index,
                features=np.stack(feature_rows) if feature_rows else None,
                feature_dimension=feature_dimension,
                labels=np.asarray(labels, dtype=np.int64),
                node_ids=np.asarray(node_ids, dtype=np.str_),
                node_identity_kind="source",
            )
            split_sources = [
                (entry.name, _read_npz(entry))
                for key, entry in sorted(entries.items())
                if key.endswith(".npz")
            ]
            _merge_payload_splits(
                payload,
                _combine_split_sources(split_sources, payload.node_count),
            )
        except ValueError as exc:
            return AdapterResult(
                detected_format="geom_gcn_text",
                status="rejected",
                profile=None,
                issues=[_issue("INVALID_GEOM_GCN", str(exc))],
            )
        return AdapterResult(
            detected_format="geom_gcn_text",
            status="accepted",
            profile=_profile(payload),
            issues=[],
            payload=payload,
        )


class FewShotJsonNpzAdapter:
    def matches(self, entries: dict[str, UploadedEntry]) -> bool:
        return any(key.endswith(".json") for key in entries) and any(
            key.endswith(".npz") for key in entries
        )

    def inspect(self, entries: dict[str, UploadedEntry]) -> AdapterResult:
        manifest_entry = next(entry for key, entry in entries.items() if key.endswith(".json"))
        try:
            manifest = json.loads(_decode_text(manifest_entry))
            if not isinstance(manifest, dict):
                raise TypeError("few-shot manifest 必须是 JSON 对象")
            graph_name = manifest.get("graph")
            split_names = manifest.get("splits", [])
            if not isinstance(graph_name, str) or not isinstance(split_names, list):
                raise TypeError("manifest 必须包含 graph 字符串与 splits 数组")
            graph_entry = entries.get(graph_name.replace("\\", "/").casefold())
            if graph_entry is None:
                raise ValueError("manifest 指定的 GraphNPZ 不存在")
            payload = _graph_from_arrays(_read_npz(graph_entry))
            split_sources: list[tuple[str, dict[str, np.ndarray]]] = []
            for split_name in split_names:
                if not isinstance(split_name, str):
                    raise TypeError("splits 只能包含文件名")
                split = entries.get(split_name.replace("\\", "/").casefold())
                if split is None:
                    raise ValueError(f"split 文件不存在: {split_name}")
                arrays = _read_npz(split)
                split_sources.append((split_name, arrays))
            _merge_payload_splits(
                payload,
                _combine_split_sources(split_sources, payload.node_count),
            )
        except (json.JSONDecodeError, TypeError, ValueError) as exc:
            return AdapterResult(
                detected_format="fewshot_json_npz",
                status="rejected",
                profile=None,
                issues=[_issue("INVALID_FEWSHOT_PACKAGE", str(exc))],
            )
        return AdapterResult(
            detected_format="fewshot_json_npz",
            status="accepted",
            profile=_profile(payload),
            issues=[],
            payload=payload,
        )


class SocialGraphDatasetPackageAdapter:
    def __init__(self, selected_dataset: str | None = None) -> None:
        self.selected_dataset = selected_dataset.strip() if selected_dataset else None

    @staticmethod
    def _manifest(entries: dict[str, UploadedEntry]) -> dict[str, object] | None:
        manifest_entry = _find_suffix(entries, "manifest.json")
        if manifest_entry is None:
            return None
        try:
            value = json.loads(_decode_text(manifest_entry))
        except (json.JSONDecodeError, ValueError):
            return None
        if not isinstance(value, dict):
            return None
        if value.get("schemaVersion") != "socialgraph-fm-dataset-package/1.0":
            return None
        return value

    def matches(self, entries: dict[str, UploadedEntry]) -> bool:
        return self._manifest(entries) is not None

    def inspect(self, entries: dict[str, UploadedEntry]) -> AdapterResult:
        manifest = self._manifest(entries)
        assert manifest is not None
        datasets = manifest.get("datasets")
        if not isinstance(datasets, list) or not datasets:
            return AdapterResult(
                detected_format="socialgraph_dataset_package",
                status="rejected",
                profile=None,
                issues=[_issue("INVALID_PACKAGE_MANIFEST", "manifest 的 datasets 必须是非空数组")],
            )
        candidates = [item for item in datasets if isinstance(item, dict)]
        if len(candidates) != len(datasets):
            return AdapterResult(
                detected_format="socialgraph_dataset_package",
                status="rejected",
                profile=None,
                issues=[_issue("INVALID_PACKAGE_MANIFEST", "datasets 中存在非对象条目")],
            )
        names = [str(item.get("name", "unnamed"))[:200] for item in candidates]
        if len({name.casefold() for name in names}) != len(names):
            return AdapterResult(
                detected_format="socialgraph_dataset_package",
                status="rejected",
                profile=None,
                issues=[_issue("INVALID_PACKAGE_MANIFEST", "datasets 中存在重复名称")],
            )
        selected: dict[str, object] | None = None
        if self.selected_dataset:
            selected = next(
                (
                    item
                    for item in candidates
                    if str(item.get("name", "")).casefold() == self.selected_dataset.casefold()
                ),
                None,
            )
            if selected is None:
                return AdapterResult(
                    detected_format="socialgraph_dataset_package",
                    status="mapping_required",
                    profile=None,
                    issues=[
                        _issue(
                            "DATASET_SELECTION_NOT_FOUND",
                            f"包中不存在数据集: {self.selected_dataset}",
                        )
                    ],
                    dataset_candidates=names,
                )
        elif len(candidates) == 1:
            selected = candidates[0]
        else:
            return AdapterResult(
                detected_format="socialgraph_dataset_package",
                status="mapping_required",
                profile=None,
                issues=[
                    _issue(
                        "DATASET_SELECTION_REQUIRED",
                        "该包包含多个数据集，请用 multipart dataset 字段选择: "
                        + ", ".join(names[:20]),
                    )
                ],
                dataset_candidates=names,
            )
        graph_path = selected.get("path") if selected else None
        if not isinstance(graph_path, str):
            return AdapterResult(
                detected_format="socialgraph_dataset_package",
                status="rejected",
                profile=None,
                issues=[_issue("INVALID_PACKAGE_MANIFEST", "选中数据集缺少 path")],
            )
        attachments: dict[str, bytes] = {}
        episode_entries: list[dict[str, object]] = []
        try:
            normalized_path = _normalized_name(graph_path).casefold()
            graph_entry = entries.get(normalized_path)
            if graph_entry is None:
                raise ValueError("manifest 指定的 graph.npz 不存在")
            payload = _graph_from_arrays(_read_npz(graph_entry))
            if selected is not None and "directed" in selected:
                payload.directed = bool(selected.get("directed"))
            raw_episodes = selected.get("fewShotEpisodes", []) if selected else []
            if not isinstance(raw_episodes, list):
                raise TypeError("fewShotEpisodes 必须是数组")
            for episode in raw_episodes:
                if not isinstance(episode, dict) or not isinstance(episode.get("path"), str):
                    raise TypeError("fewShotEpisodes 包含无效条目")
                episode_path = _normalized_name(str(episode["path"])).casefold()
                episode_entry = entries.get(episode_path)
                if episode_entry is None:
                    raise ValueError(f"包中缺少 few-shot episode: {episode['path']}")
                _read_npz(episode_entry)
                relative = f"episodes/{PurePosixPath(episode_path).name}"
                if relative in attachments:
                    raise ValueError(f"few-shot episode 文件名冲突: {relative}")
                attachments[relative] = episode_entry.data
                episode_entries.append({**episode, "artifactPath": relative})
        except (HTTPException, TypeError, ValueError) as exc:
            detail = exc.detail if isinstance(exc, HTTPException) else str(exc)
            return AdapterResult(
                detected_format="socialgraph_dataset_package",
                status="rejected",
                profile=None,
                issues=[_issue("INVALID_PACKAGE_GRAPH", str(detail))],
            )
        dataset_name = str(selected.get("name", "unnamed"))[:200] if selected else None
        raw_manifest: dict[str, object] = {
            "schemaVersion": "2.1",
            "packageSchemaVersion": manifest.get("schemaVersion"),
            "datasetName": dataset_name,
            "sourceFormat": selected.get("sourceFormat", "socialgraph_dataset_package")
            if selected
            else "socialgraph_dataset_package",
            "license": selected.get("license", "unknown") if selected else "unknown",
            "licensePolicy": selected.get("licensePolicy") if selected else None,
            "licenseEvidence": selected.get("licenseEvidence", []) if selected else [],
            "dataGovernance": selected.get("dataGovernance") if selected else None,
            "linkPredictionProtocol": selected.get("linkPredictionProtocol") if selected else None,
            "datasetRole": selected.get("datasetRole", "target_domain") if selected else "target_domain",
            "splitKind": selected.get("splitKind", "source") if selected else "source",
            "sourceFiles": selected.get("sourceFiles", []) if selected else [],
            "splitFiles": selected.get("splitFiles", []) if selected else [],
            "fewShotEpisodes": episode_entries,
            "sourceFingerprint": manifest.get("sourceFingerprint"),
            "conversionSkipped": manifest.get("skipped", []),
            "packageManifest": manifest,
            "selectedDatasetManifest": selected or {},
        }
        derived_manifest: dict[str, object] = {
            "schemaVersion": "2.1",
            "transforms": selected.get("transforms", []) if selected else [],
            "transformRecipes": selected.get("transformRecipes", []) if selected else [],
            "splitNames": payload.split_names,
        }
        return AdapterResult(
            detected_format="socialgraph_dataset_package",
            status="accepted",
            profile=_profile(payload),
            issues=[],
            payload=payload,
            dataset_candidates=names,
            dataset_name=dataset_name,
            raw_manifest=raw_manifest,
            derived_manifest=derived_manifest,
            attachments=attachments,
        )


class UnsupportedAdapter:
    def matches(self, entries: dict[str, UploadedEntry]) -> bool:
        return True

    def inspect(self, entries: dict[str, UploadedEntry]) -> AdapterResult:
        return AdapterResult(
            detected_format="unsupported",
            status="rejected",
            profile=None,
            issues=[
                _issue(
                    "UNSUPPORTED_DATASET_FORMAT",
                    "支持 Geom-GCN 文本、安全 GraphNPZ、严格 split NPZ 与 few-shot JSON+NPZ。",
                )
            ],
        )


def _adapter_registry(
    entries: dict[str, UploadedEntry],
    selected_dataset: str | None = None,
) -> list[DatasetAdapter]:
    # Unsafe formats always win, even if a ZIP also contains a safe derivative.
    unsafe: list[DatasetAdapter] = [TorchPygArchiveDetector(), LegacyPlanetoidPickleDetector()]
    if any(adapter.matches(entries) for adapter in unsafe):
        return unsafe
    return [
        GraphVersionTargetDomainAdapter(),
        SocialGraphDatasetPackageAdapter(selected_dataset),
        GeomGcnTextDirectoryAdapter(),
        FewShotJsonNpzAdapter(),
        SafeGraphNpzAdapter(),
        StrictSplitNpzAdapter(),
        UnsupportedAdapter(),
    ]
