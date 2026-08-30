"""Data-only NumPy parsing, validation, and graph payload construction."""

from __future__ import annotations

import io
import zipfile
from typing import Literal

import numpy as np

from ..dataset_schemas import (
    DatasetArtifact,
    DatasetProfile,
)

from .models import (
    MAX_ARRAYS,
    MAX_ARRAY_ELEMENTS,
    MAX_EDGES,
    MAX_NODES,
    MAX_TRUSTED_ARRAY_BYTES,
    _SPLIT_KEYS,
    GraphPayload,
    UploadedEntry,
)

def _validate_npz_container(
    entry: UploadedEntry,
    *,
    trusted_generated: bool = False,
    trusted_max_bytes: int = MAX_TRUSTED_ARRAY_BYTES,
) -> None:
    try:
        archive = zipfile.ZipFile(io.BytesIO(entry.data))
    except zipfile.BadZipFile as exc:
        raise ValueError("NPZ 容器损坏") from exc
    with archive:
        members = archive.infolist()
        array_limit = 256 if trusted_generated else MAX_ARRAYS
        if len(members) > array_limit:
            raise ValueError("NPZ 数组数量超过限制")
        expanded = sum(member.file_size for member in members)
        expanded_limit = trusted_max_bytes if trusted_generated else 128 * 1024 * 1024
        if expanded > expanded_limit:
            raise ValueError("NPZ 解压后超过安全限制")
        for member in members:
            member_limit = trusted_max_bytes if trusted_generated else 64 * 1024 * 1024
            if member.file_size > member_limit:
                raise ValueError(f"NPZ 数组过大: {member.filename}")
            if (
                not trusted_generated
                and member.compress_size
                and member.file_size / member.compress_size > 1_000
            ):
                raise ValueError(f"NPZ 数组压缩比异常: {member.filename}")


def _read_npz(
    entry: UploadedEntry,
    *,
    trusted_generated: bool = False,
    trusted_max_bytes: int = MAX_TRUSTED_ARRAY_BYTES,
) -> dict[str, np.ndarray]:
    _validate_npz_container(
        entry,
        trusted_generated=trusted_generated,
        trusted_max_bytes=trusted_max_bytes,
    )
    arrays: dict[str, np.ndarray] = {}
    try:
        with np.load(io.BytesIO(entry.data), allow_pickle=False) as payload:
            for key in payload.files:
                array = np.asarray(payload[key])
                if array.dtype.hasobject:
                    raise ValueError(f"数组 {key} 使用 object dtype")
                exceeds_limit = (
                    array.nbytes > trusted_max_bytes
                    if trusted_generated
                    else array.size > MAX_ARRAY_ELEMENTS
                )
                if array.ndim > 3 or exceeds_limit:
                    raise ValueError(f"数组 {key} 的形状超过限制")
                arrays[key] = array
    except (OSError, ValueError, zipfile.BadZipFile) as exc:
        if isinstance(exc, ValueError) and str(exc).startswith("数组"):
            raise
        raise ValueError(f"无法安全读取 NPZ: {exc}") from exc
    return arrays


def _normalize_edge_index(array: np.ndarray) -> np.ndarray:
    if array.ndim != 2:
        raise ValueError("edge_index 必须是二维数组")
    if array.shape[0] == 2:
        result = array
    elif array.shape[1] == 2:
        result = array.T
    else:
        raise ValueError("edge_index 必须为 [2,E] 或 [E,2]")
    if not np.issubdtype(result.dtype, np.integer):
        raise ValueError("edge_index 必须使用整数类型")
    result = result.astype(np.int64, copy=False)
    if result.shape[1] > MAX_EDGES:
        raise ValueError("边数量超过限制")
    if result.size and int(result.min()) < 0:
        raise ValueError("edge_index 包含负节点索引")
    return result


def _edge_array_count(array: np.ndarray) -> int:
    value = np.asarray(array)
    if value.ndim != 2 or 2 not in value.shape:
        raise ValueError("链路划分数组必须是 [2,E] 或 [E,2]")
    return int(value.shape[1] if value.shape[0] == 2 else value.shape[0])


def _edge_keys(
    array: np.ndarray,
    *,
    node_count: int,
    undirected: bool,
) -> np.ndarray:
    value = np.asarray(array)
    if value.ndim != 2:
        raise ValueError("LINK_EDGE_SHAPE_INVALID")
    if value.shape[0] == 2:
        edges = value
    elif value.shape[1] == 2:
        edges = value.T
    else:
        raise ValueError("LINK_EDGE_SHAPE_INVALID")
    if not np.issubdtype(edges.dtype, np.integer):
        raise ValueError("LINK_EDGE_DTYPE_INVALID")
    edges = edges.astype(np.int64, copy=False)
    if edges.size and (int(edges.min()) < 0 or int(edges.max()) >= node_count):
        raise ValueError("LINK_ENDPOINT_OUT_OF_RANGE")
    source, target = edges
    if undirected:
        source, target = np.minimum(source, target), np.maximum(source, target)
    return np.unique(source * np.int64(node_count) + target)


def _split_memberships(
    value: np.ndarray,
    *,
    node_count: int,
    representation: str,
) -> list[set[int]]:
    array = np.asarray(value)
    if representation == "mask":
        if not np.issubdtype(array.dtype, np.bool_) and not (
            np.issubdtype(array.dtype, np.integer)
            and bool(np.all(np.isin(array, np.asarray([0, 1], dtype=array.dtype))))
        ):
            raise ValueError("SPLIT_MASK_VALUE_INVALID")
        array = array.astype(np.bool_, copy=False)
        if array.ndim == 1:
            if array.size != node_count:
                raise ValueError("SPLIT_SHAPE_MISMATCH")
            return [set(np.flatnonzero(array).tolist())]
        if array.ndim == 2 and array.shape[0] == node_count:
            return [set(np.flatnonzero(array[:, fold]).tolist()) for fold in range(array.shape[1])]
        raise ValueError("SPLIT_SHAPE_MISMATCH")
    if not np.issubdtype(array.dtype, np.integer) or array.ndim not in {1, 2}:
        raise ValueError("SPLIT_DTYPE_INVALID")
    folds = [array.reshape(-1)] if array.ndim == 1 else [array[:, fold] for fold in range(array.shape[1])]
    result: list[set[int]] = []
    for fold in folds:
        indices = [int(value) for value in fold.tolist()]
        if any(index < 0 or index >= node_count for index in indices):
            raise ValueError("SPLIT_INDEX_OUT_OF_RANGE")
        if len(indices) != len(set(indices)):
            raise ValueError("SPLIT_INDEX_DUPLICATE")
        result.append(set(indices))
    return result


def _validate_artifact_arrays(
    artifact: DatasetArtifact,
    arrays: dict[str, np.ndarray],
) -> None:
    """Defense-in-depth checks performed from disk on every readiness call."""

    if artifact.node_identity is None:
        raise ValueError("NODE_ID_MAP_MISSING")
    node_count = artifact.node_identity.count
    for feature_schema in artifact.feature_schemas:
        value = arrays.get(feature_schema.array_name)
        if value is None:
            raise ValueError("FEATURE_ARRAY_MISSING")
        if value.dtype.str != feature_schema.dtype or list(value.shape) != feature_schema.shape:
            raise ValueError("FEATURE_SCHEMA_MISMATCH")
        if feature_schema.target == "node" and (
            value.ndim == 0 or value.shape[0] != node_count
        ):
            raise ValueError("FEATURE_NODE_COUNT_MISMATCH")
        if (
            feature_schema.missing_value_policy == "reject"
            and np.issubdtype(value.dtype, np.number)
            and not bool(np.all(np.isfinite(value)))
        ):
            raise ValueError("FEATURE_NON_FINITE")

    for label_schema in artifact.label_schemas:
        value = arrays.get(label_schema.array_name)
        if value is None:
            raise ValueError("LABEL_ARRAY_MISSING")
        if value.dtype.str != label_schema.dtype or list(value.shape) != label_schema.shape:
            raise ValueError("LABEL_SCHEMA_MISMATCH")
        if label_schema.target == "node" and (
            value.ndim == 0 or value.shape[0] != node_count
        ):
            raise ValueError("LABEL_NODE_COUNT_MISMATCH")
        if np.issubdtype(value.dtype, np.number) and not bool(np.all(np.isfinite(value))):
            raise ValueError("LABEL_NON_FINITE")
        flattened = value.reshape(-1)
        if label_schema.ignore_value is not None:
            flattened = flattened[flattened != label_schema.ignore_value]
        if label_schema.mode == "single_label":
            observed = set(flattened.tolist())
            declared = set(label_schema.class_values)
            if declared and not observed.issubset(declared):
                raise ValueError("LABEL_CLASS_VALUE_MISMATCH")
            if (
                label_schema.class_count is not None
                and declared
                and label_schema.class_count != len(declared)
            ):
                raise ValueError("LABEL_CLASS_COUNT_MISMATCH")

    feature_array_names = {schema.array_name for schema in artifact.feature_schemas}
    for recipe in artifact.feature_recipes:
        if recipe.input_array is not None and recipe.input_array not in arrays:
            raise ValueError("RECIPE_INPUT_MISSING")
        if recipe.output_array is not None and recipe.output_array not in arrays:
            raise ValueError("RECIPE_OUTPUT_MISSING")
        if recipe.output_array is not None and recipe.output_array not in feature_array_names:
            raise ValueError("RECIPE_FEATURE_SCHEMA_MISSING")

    variants = {variant.id: variant for variant in artifact.graph_variants}
    raw_edges = arrays.get("edge_index")
    if raw_edges is None:
        raise ValueError("EDGE_INDEX_MISSING")
    raw_edge_count = _edge_array_count(raw_edges)
    for name in (
        "edge_id_map",
        "edge_type",
        "edge_weight",
        "edge_timestamp",
        "edge_directed",
        "edge_attributes_json",
    ):
        value = arrays.get(name)
        if value is not None and value.reshape(-1).size != raw_edge_count:
            raise ValueError("EDGE_ATTRIBUTE_LENGTH_MISMATCH")
    weights = arrays.get("edge_weight")
    if weights is not None and (
        not np.issubdtype(weights.dtype, np.number)
        or not bool(np.all(np.isfinite(weights) | np.isnan(weights)))
    ):
        raise ValueError("EDGE_WEIGHT_INVALID")
    if artifact.graph_semantics is not None:
        if artifact.graph_semantics.weighted and weights is None:
            raise ValueError("EDGE_WEIGHT_ARRAY_MISSING")
        if artifact.graph_semantics.temporal and arrays.get("edge_timestamp") is None:
            raise ValueError("EDGE_TIMESTAMP_ARRAY_MISSING")
        if (
            artifact.graph_semantics.edge_directed_array is not None
            and artifact.graph_semantics.edge_directed_array not in arrays
        ):
            raise ValueError("EDGE_DIRECTED_ARRAY_MISSING")
    directed_values = arrays.get("edge_directed")
    if directed_values is not None and not {
        int(value) for value in directed_values.reshape(-1).tolist()
    }.issubset({-1, 0, 1}):
        raise ValueError("EDGE_DIRECTED_VALUE_INVALID")
    for variant in variants.values():
        edge_index = arrays.get(variant.edge_index_array)
        if edge_index is None:
            raise ValueError("GRAPH_VARIANT_ARRAY_MISSING")
        normalized = _normalize_edge_index(edge_index)
        if normalized.size and int(normalized.max()) >= node_count:
            raise ValueError("EDGE_ENDPOINT_OUT_OF_RANGE")

    split_members: dict[str, list[set[int]]] = {}
    for split_set in artifact.split_sets:
        if split_set.target != "node":
            continue
        for part, array_name in split_set.arrays.items():
            value = arrays.get(array_name)
            if value is None:
                raise ValueError("SPLIT_ARRAY_MISSING")
            split_members[f"{split_set.id}:{part}"] = _split_memberships(
                value,
                node_count=node_count,
                representation=split_set.representation,
            )
        parts = [part for part in ("train", "validation", "test") if part in split_set.arrays]
        fold_count = max(
            len(split_members[f"{split_set.id}:{part}"]) for part in parts
        )
        if fold_count != split_set.fold_count:
            raise ValueError("SPLIT_FOLD_COUNT_MISMATCH")
        for fold in range(fold_count):
            memberships = [
                split_members[f"{split_set.id}:{part}"][
                    min(fold, len(split_members[f"{split_set.id}:{part}"]) - 1)
                ]
                for part in parts
            ]
            for index, left in enumerate(memberships):
                if any(left.intersection(right) for right in memberships[index + 1 :]):
                    raise ValueError("SPLIT_OVERLAP")
            if split_set.fold_counts:
                if len(split_set.fold_counts) != fold_count:
                    raise ValueError("SPLIT_FOLD_COUNTS_MISMATCH")
                expected_counts = split_set.fold_counts[fold]
                actual_counts = {
                    part: len(
                        split_members[f"{split_set.id}:{part}"][
                            min(fold, len(split_members[f"{split_set.id}:{part}"]) - 1)
                        ]
                    )
                    for part in parts
                }
                if (
                    actual_counts.get("train", 0) != expected_counts.train
                    or actual_counts.get("validation", 0) != expected_counts.validation
                    or actual_counts.get("test", 0) != expected_counts.test
                ):
                    raise ValueError("SPLIT_FOLD_COUNTS_MISMATCH")

    for task in artifact.task_specs:
        protocol = task.link_prediction_protocol
        if task.kind != "link_prediction":
            continue
        if protocol is None:
            raise ValueError("LINK_PROTOCOL_MISSING")
        required = {
            protocol.message_passing_edge_array,
            protocol.train_positive_array,
            protocol.validation_positive_array,
            protocol.test_positive_array,
        }
        if protocol.negative_sampler == "stored":
            if not protocol.validation_negative_array or not protocol.test_negative_array:
                raise ValueError("LINK_NEGATIVE_ARRAY_MISSING")
            required.update({protocol.validation_negative_array, protocol.test_negative_array})
        if not required.issubset(arrays):
            raise ValueError("LINK_ARRAY_MISSING")
        undirected = not artifact.graph_semantics.directed if artifact.graph_semantics else True
        _edge_keys(
            arrays[protocol.message_passing_edge_array],
            node_count=node_count,
            undirected=undirected,
        )
        train = _edge_keys(
            arrays[protocol.train_positive_array],
            node_count=node_count,
            undirected=undirected,
        )
        validation = _edge_keys(
            arrays[protocol.validation_positive_array],
            node_count=node_count,
            undirected=undirected,
        )
        test = _edge_keys(
            arrays[protocol.test_positive_array],
            node_count=node_count,
            undirected=undirected,
        )
        if protocol.positive_overlap_policy == "reject" and (
            np.intersect1d(train, validation, assume_unique=True).size
            or np.intersect1d(train, test, assume_unique=True).size
            or np.intersect1d(validation, test, assume_unique=True).size
        ):
            raise ValueError("LINK_POSITIVE_LEAKAGE")
        positives = np.unique(np.concatenate((train, validation, test)))
        for negative_array_name in (
            protocol.validation_negative_array,
            protocol.test_negative_array,
        ):
            if negative_array_name and np.intersect1d(
                positives,
                _edge_keys(
                    arrays[negative_array_name],
                    node_count=node_count,
                    undirected=undirected,
                ),
                assume_unique=True,
            ).size:
                raise ValueError("LINK_POSITIVE_NEGATIVE_CONFLICT")
        if protocol.edge_year_array:
            years = arrays.get(protocol.edge_year_array)
            if years is None:
                raise ValueError("LINK_YEAR_ARRAY_MISSING")
            message_count = _edge_array_count(arrays[protocol.message_passing_edge_array])
            if years.reshape(-1).size != message_count:
                raise ValueError("LINK_YEAR_LENGTH_MISMATCH")
            if not np.issubdtype(years.dtype, np.integer):
                raise ValueError("LINK_YEAR_DTYPE_INVALID")
            if protocol.train_year_max is not None and years.size and int(years.max()) > protocol.train_year_max:
                raise ValueError("TEMPORAL_TRAINING_LEAKAGE")
        if (
            protocol.train_year_max is not None
            and protocol.validation_year is not None
            and protocol.test_year is not None
            and not protocol.train_year_max < protocol.validation_year < protocol.test_year
        ):
            raise ValueError("TEMPORAL_CUTOFF_INVALID")
        if protocol.edge_weight_array:
            weights = arrays.get(protocol.edge_weight_array)
            message_count = _edge_array_count(arrays[protocol.message_passing_edge_array])
            if weights is None or weights.reshape(-1).size != message_count:
                raise ValueError("LINK_WEIGHT_LENGTH_MISMATCH")
            if not np.issubdtype(weights.dtype, np.number) or not bool(np.all(np.isfinite(weights))):
                raise ValueError("LINK_WEIGHT_INVALID")


def _semantic_edge_index(
    edge_index: np.ndarray,
    node_count: int,
    *,
    directed: bool,
) -> np.ndarray:
    """Return relation-level COO while leaving the stored raw COO untouched."""

    edges = np.asarray(edge_index, dtype=np.int64)
    if directed or edges.size == 0:
        return edges
    source = np.minimum(edges[0], edges[1])
    target = np.maximum(edges[0], edges[1])
    keys = np.unique(source * np.int64(node_count) + target)
    return np.stack((keys // node_count, keys % node_count)).astype(np.int64, copy=False)


def _validated_split_arrays(
    arrays: dict[str, np.ndarray],
    node_count: int | None = None,
) -> dict[str, np.ndarray]:
    normalized: dict[str, np.ndarray] = {}
    for key in _SPLIT_KEYS:
        if key not in arrays:
            continue
        value = np.asarray(arrays[key])
        if value.ndim not in {1, 2}:
            raise ValueError(f"{key} 必须是一维或多折二维数组")
        if value.ndim == 2 and value.shape[1] == 0:
            raise ValueError(f"{key} 至少必须包含一折")
        if key.endswith("_mask"):
            if value.dtype.kind not in {"b", "u", "i"}:
                raise ValueError(f"{key} 必须是 bool/uint/int 掩码")
            if node_count is not None and value.shape[0] != node_count:
                raise ValueError(f"{key} 第一维必须等于节点数")
            if value.size and not bool(np.all((value == 0) | (value == 1))):
                raise ValueError(f"{key} 只能包含 0/1")
            normalized[key] = value.astype(np.bool_, copy=False)
        else:
            if value.ndim != 1:
                raise ValueError(f"{key} 目前只支持单折一维索引")
            if not np.issubdtype(value.dtype, np.integer):
                raise ValueError(f"{key} 必须是非负整数索引")
            if value.size and int(value.min()) < 0:
                raise ValueError(f"{key} 必须是非负整数索引")
            if value.size and node_count is not None and int(value.max()) >= node_count:
                raise ValueError(f"{key} 包含越界节点索引")
            normalized[key] = value.astype(np.int64, copy=False)

    mask_names = [name for name in _SPLIT_KEYS if name.endswith("_mask") and name in normalized]
    index_names = [name for name in _SPLIT_KEYS if name.endswith("_idx") and name in normalized]
    if mask_names and index_names:
        raise ValueError("mask 与 idx 划分不能在同一 split set 中混用")
    names = mask_names or index_names
    if not names:
        return normalized

    fold_counts = {
        1 if normalized[name].ndim == 1 else int(normalized[name].shape[1])
        for name in names
    }
    if len(fold_counts) != 1:
        raise ValueError("train/val/test 的折数必须一致")
    if mask_names and len({int(normalized[name].shape[0]) for name in names}) != 1:
        raise ValueError("train/val/test mask 的节点维必须一致")

    fold_count = next(iter(fold_counts))
    for fold in range(fold_count):
        memberships: dict[str, np.ndarray] = {}
        for name in names:
            value = normalized[name]
            column = value if value.ndim == 1 else value[:, fold]
            if name.endswith("_mask"):
                memberships[name] = np.flatnonzero(column)
            else:
                indices = np.asarray(column, dtype=np.int64).reshape(-1)
                if np.unique(indices).size != indices.size:
                    raise ValueError(f"{name} 第 {fold} 折包含重复索引")
                memberships[name] = indices
        for index, left_name in enumerate(names):
            for right_name in names[index + 1 :]:
                if np.intersect1d(
                    memberships[left_name],
                    memberships[right_name],
                    assume_unique=True,
                ).size:
                    raise ValueError(
                        f"{left_name} 与 {right_name} 在第 {fold} 折存在交叉"
                    )
    return normalized


def _split_names(arrays: dict[str, np.ndarray], node_count: int | None = None) -> list[str]:
    return list(_validated_split_arrays(arrays, node_count))


def _combine_split_sources(
    sources: list[tuple[str, dict[str, np.ndarray]]],
    node_count: int | None,
) -> dict[str, np.ndarray]:
    validated = [
        (source, splits)
        for source, arrays in sources
        if (splits := _validated_split_arrays(arrays, node_count))
    ]
    if not validated:
        return {}
    expected_names = set(validated[0][1])
    for source, splits in validated[1:]:
        if set(splits) != expected_names:
            raise ValueError(f"split 文件 {source} 的 train/val/test 字段与其他折不对齐")
    if len(validated) > 1 and any(name.endswith("_idx") for name in expected_names):
        raise ValueError(
            "多文件变长 idx split 无法无损合并；请改用单一 split NPZ 或 fewShotEpisodes"
        )

    combined: dict[str, np.ndarray] = {}
    for name in _SPLIT_KEYS:
        if name not in expected_names:
            continue
        values = [splits[name] for _source, splits in validated]
        if len(values) == 1:
            combined[name] = values[0]
            continue
        matrices = [value[:, None] if value.ndim == 1 else value for value in values]
        combined[name] = np.concatenate(matrices, axis=1)
    return _validated_split_arrays(combined, node_count)


def _merge_payload_splits(
    payload: GraphPayload,
    external: dict[str, np.ndarray],
) -> None:
    merged = _validated_split_arrays(payload.splits, payload.node_count)
    for name, value in external.items():
        if name in merged:
            if not np.array_equal(merged[name], value):
                raise ValueError(f"外部 {name} 与主图内嵌官方划分冲突")
            continue
        merged[name] = value
    payload.splits = _validated_split_arrays(merged, payload.node_count)
    payload.split_names = [name for name in _SPLIT_KEYS if name in payload.splits]


def _graph_from_arrays(arrays: dict[str, np.ndarray]) -> GraphPayload:
    if "edge_index" not in arrays:
        raise ValueError("GraphNPZ 缺少 edge_index")
    edge_index = _normalize_edge_index(arrays["edge_index"])
    candidates: list[int] = []
    feature_dimension: int | None = None
    features: np.ndarray | None = None
    labels: np.ndarray | None = None
    node_ids: np.ndarray | None = None
    node_labels: np.ndarray | None = None
    node_types: np.ndarray | None = None
    node_attributes: np.ndarray | None = None
    node_identity_kind: Literal["source", "row_index"] = "row_index"
    if "node_id_map" in arrays:
        candidate_ids = np.asarray(arrays["node_id_map"]).reshape(-1)
        if candidate_ids.dtype.hasobject:
            raise ValueError("node_id_map 不能使用 object dtype")
        text_ids = np.asarray([str(value) for value in candidate_ids], dtype=np.str_)
        if len(set(text_ids.tolist())) != int(text_ids.size):
            raise ValueError("node_id_map 必须唯一")
        node_ids = text_ids
        node_identity_kind = "source"
    if "x" in arrays:
        x = arrays["x"]
        if x.ndim != 2 or not np.issubdtype(x.dtype, np.number):
            raise ValueError("x 必须是二维数值特征矩阵")
        candidates.append(int(x.shape[0]))
        feature_dimension = int(x.shape[1])
        features = np.asarray(x)
    if "y" in arrays:
        labels = arrays["y"].reshape(-1)
        if not np.issubdtype(labels.dtype, np.number):
            raise ValueError("y 必须是数值标签数组")
        candidates.append(int(labels.shape[0]))
    if "num_nodes" in arrays:
        num_nodes_value = arrays["num_nodes"]
        if num_nodes_value.size != 1:
            raise ValueError("num_nodes 必须是标量")
        candidates.append(int(num_nodes_value.reshape(-1)[0]))
    if node_ids is not None:
        candidates.append(int(node_ids.shape[0]))
    inferred = int(edge_index.max()) + 1 if edge_index.size else 0
    node_count = candidates[0] if candidates else inferred
    if any(candidate != node_count for candidate in candidates):
        raise ValueError("x、y 与 num_nodes 的节点数量不一致")
    if node_count < inferred:
        raise ValueError("edge_index 包含越界节点索引")
    if node_count < 0 or node_count > MAX_NODES:
        raise ValueError("节点数量超过限制")
    for name in ("node_label", "node_type", "node_attributes_json"):
        if name not in arrays:
            continue
        values = np.asarray(arrays[name]).reshape(-1).astype(np.str_, copy=False)
        if int(values.size) != node_count:
            raise ValueError(f"{name} 数量必须等于节点数")
        if name == "node_label":
            node_labels = values
        elif name == "node_type":
            node_types = values
        else:
            node_attributes = values
    edge_count = int(edge_index.shape[1])
    edge_values: dict[str, np.ndarray] = {}
    for name in (
        "edge_id_map",
        "edge_type",
        "edge_weight",
        "edge_timestamp",
        "edge_directed",
        "edge_attributes_json",
    ):
        if name not in arrays:
            continue
        values = np.asarray(arrays[name]).reshape(-1)
        if int(values.size) != edge_count:
            raise ValueError(f"{name} 数量必须等于边数")
        edge_values[name] = values
    split_arrays = _validated_split_arrays(arrays, node_count)
    directed = False
    if "directed" in arrays:
        directed_value = np.asarray(arrays["directed"])
        if directed_value.size != 1:
            raise ValueError("directed 必须是标量")
        directed = bool(directed_value.reshape(-1)[0])
    return GraphPayload(
        node_count=node_count,
        edge_index=edge_index,
        features=features,
        feature_dimension=feature_dimension,
        labels=labels,
        split_names=list(split_arrays),
        splits=split_arrays,
        variant_arrays={
            name: np.asarray(value)
            for name, value in arrays.items()
            if name.startswith("variant_")
        },
        node_ids=node_ids,
        node_labels=node_labels,
        node_types=node_types,
        node_attributes=node_attributes,
        edge_ids=edge_values.get("edge_id_map"),
        edge_types=edge_values.get("edge_type"),
        edge_weights=edge_values.get("edge_weight"),
        edge_timestamps=edge_values.get("edge_timestamp"),
        edge_directed=edge_values.get("edge_directed"),
        edge_attributes=edge_values.get("edge_attributes_json"),
        node_identity_kind=node_identity_kind,
        directed=directed,
        directedness=(
            "mixed"
            if "edge_directed" in edge_values
            and {-1, 0, 1}.intersection(
                {int(value) for value in edge_values["edge_directed"].tolist()}
            ) not in ({0}, {1})
            else ("directed" if directed else "undirected")
        ),
    )


def _profile(payload: GraphPayload) -> DatasetProfile:
    label_count = None
    if payload.labels is not None and payload.labels.size:
        valid = payload.labels[payload.labels >= 0]
        label_count = int(np.unique(valid).size) if valid.size else 0
    semantic_edges = (
        np.asarray(payload.edge_index, dtype=np.int64)
        if payload.edge_ids is not None
        else _semantic_edge_index(
            payload.edge_index,
            payload.node_count,
            directed=payload.directed,
        )
    )
    return DatasetProfile(
        nodeCount=payload.node_count,
        edgeCount=int(semantic_edges.shape[1]),
        featureDimension=payload.feature_dimension,
        labelCount=label_count,
        splitNames=payload.split_names,
        directed=payload.directed,
    )
