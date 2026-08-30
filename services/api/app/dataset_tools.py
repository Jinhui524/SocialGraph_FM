from __future__ import annotations

import argparse
import contextlib
import hashlib
import io
import json
import os
import pickle
import sys
import zipfile
from pathlib import Path
from typing import Any

import numpy as np


@contextlib.contextmanager
def _trusted_legacy_torch_load():
    """Confine the OGB 1.3.6/PyG legacy pickle compatibility exception.

    Torch 2.6 and newer default ``torch.load`` to ``weights_only=True`` while
    OGB 1.3.6 still loads its processed PyG ``Data`` cache without an explicit
    argument. This converter already requires ``--trust-pickle`` and runs in an
    isolated loopback subprocess. The temporary override must never be reused by
    the public upload path and is restored even when conversion fails.
    """

    try:
        import torch  # type: ignore[import-not-found]
    except ImportError:
        yield
        return
    original = torch.load

    def trusted_load(*args: Any, **kwargs: Any) -> Any:
        kwargs.setdefault("weights_only", False)
        return original(*args, **kwargs)

    torch.load = trusted_load
    try:
        yield
    finally:
        torch.load = original


def _apply_process_memory_limit() -> None:
    raw_limit = os.environ.get("SGFM_CONVERTER_MEMORY_LIMIT_MB", "").strip()
    if not raw_limit or os.name == "nt":
        # The parent process monitors Windows working set and terminates the tree.
        return
    try:
        import resource

        limit = int(raw_limit) * 1024 * 1024
        resource.setrlimit(resource.RLIMIT_AS, (limit, limit))  # type: ignore[attr-defined]
    except (ImportError, OSError, ValueError):
        return


def _as_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def _extract_pyg_fields(
    value: Any,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, dict[str, np.ndarray]]:
    data = value[0] if isinstance(value, tuple) and value else value
    if isinstance(data, dict):
        x = data.get("x")
        edge_index = data.get("edge_index")
        y = data.get("y")
    else:
        x = getattr(data, "x", None)
        edge_index = getattr(data, "edge_index", None)
        y = getattr(data, "y", None)
    if x is None or edge_index is None:
        raise ValueError("PyG data.pt 缺少 x 或 edge_index")
    features = _as_numpy(x)
    edges = _as_numpy(edge_index)
    labels = _as_numpy(y).reshape(-1) if y is not None else None
    if features.ndim != 2 or edges.ndim != 2 or 2 not in edges.shape:
        raise ValueError("PyG 张量形状无效")
    if edges.shape[0] != 2:
        edges = edges.T
    split_arrays: dict[str, np.ndarray] = {}
    for name in ("train_mask", "val_mask", "test_mask", "train_idx", "val_idx", "test_idx"):
        split = data.get(name) if isinstance(data, dict) else getattr(data, name, None)
        if split is not None:
            split_arrays[name] = _as_numpy(split)
    return features, edges.astype(np.int64, copy=False), labels, split_arrays


def _load_trusted_torch(
    path: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, dict[str, np.ndarray]]:
    try:
        import torch  # type: ignore[import-not-found]
    except ImportError as exc:  # pragma: no cover - depends on research environment
        raise RuntimeError("转换 PT 需要在研究环境安装 PyTorch") from exc
    # Unsafe loads are intentionally confined to this CLI module. The CLI
    # refuses to reach them unless the operator passes --trust-pickle.
    value = torch.load(str(path), map_location="cpu", weights_only=False)
    return _extract_pyg_fields(value)


def _canonicalize_weighted_structure_edges(
    edges: np.ndarray, node_count: int
) -> np.ndarray:
    """Build the undirected, loop-free, sorted COO topology variant."""

    normalized = np.asarray(edges, dtype=np.int64)
    if normalized.ndim != 2 or normalized.shape[0] != 2:
        raise ValueError("edge_index must have shape [2,E]")
    if normalized.size and (int(normalized.min()) < 0 or int(normalized.max()) >= node_count):
        raise ValueError("edge_index contains an out-of-range node id")
    source, target = normalized
    keep = source != target
    forward_source, forward_target = source[keep], target[keep]
    source = np.concatenate((forward_source, forward_target))
    target = np.concatenate((forward_target, forward_source))
    if source.size == 0:
        return np.empty((2, 0), dtype=np.int64)
    keys = np.unique(source * np.int64(node_count) + target)
    return np.stack((keys // node_count, keys % node_count)).astype(np.int64, copy=False)


def _weighted_structure_variant_arrays(
    features: np.ndarray,
    edges: np.ndarray,
    *,
    pca_seed: int = 0,
) -> tuple[dict[str, np.ndarray], list[dict[str, object]]]:
    """Build the reusable weighted-structure/PCA variant when dimensions allow it."""

    node_count = int(features.shape[0])
    if min(features.shape) < 50:
        return {}, []
    values = np.asarray(features, dtype=np.float32)
    if values.shape[1] == 50:
        reduced = np.ascontiguousarray(values).copy()
        feature_transform = "identity_dimension_match"
    else:
        try:
            from sklearn.decomposition import PCA  # type: ignore[import-not-found,import-untyped]
        except ImportError:
            return {}, []
        reduced = PCA(n_components=50, random_state=pca_seed).fit_transform(values)
        feature_transform = "sklearn_pca"
    arrays = {
        "variant_weighted_structure_edge_index": _canonicalize_weighted_structure_edges(
            edges, node_count
        ),
        "variant_weighted_structure_x": np.asarray(reduced, dtype=np.float32),
    }
    recipes: list[dict[str, object]] = [
        {
            "id": "weighted-structure",
            "graphVariant": "weighted-structure",
            "edgeIndexArray": "variant_weighted_structure_edge_index",
            "featureArray": "variant_weighted_structure_x",
            "directed": False,
            "selfLoopPolicy": "remove",
            "duplicatePolicy": "deduplicate_sorted",
            "featureTransform": feature_transform,
            "fitScope": "all_nodes_transductive",
            "parameters": {"nComponents": 50, "randomState": pca_seed},
        }
    ]
    return arrays, recipes


def _load_geom(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    node_file = path / "out1_node_feature_label.txt"
    edge_file = path / "out1_graph_edges.txt"
    node_ids: list[str] = []
    feature_rows: list[np.ndarray] = []
    labels: list[int] = []
    with node_file.open("r", encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, start=1):
            if line_number == 1 and line.casefold().startswith("node_id"):
                continue
            if not line.strip():
                continue
            node_id, raw_features, raw_label = line.rstrip("\r\n").split("\t")
            node_ids.append(node_id)
            feature_rows.append(np.fromstring(raw_features, sep=",", dtype=np.float32))
            labels.append(int(raw_label))
    if not feature_rows or any(row.shape != feature_rows[0].shape for row in feature_rows):
        raise ValueError("Geom-GCN 特征为空或维度不一致")
    lookup = {node_id: index for index, node_id in enumerate(node_ids)}
    pairs: list[tuple[int, int]] = []
    with edge_file.open("r", encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, start=1):
            if line_number == 1 and line.casefold().startswith("node_id"):
                continue
            if not line.strip():
                continue
            source, target = line.split()
            pairs.append((lookup[source], lookup[target]))
    edges = (
        np.asarray(pairs, dtype=np.int64).T
        if pairs
        else np.empty((2, 0), dtype=np.int64)
    )
    return (
        np.stack(feature_rows),
        edges,
        np.asarray(labels, dtype=np.int64),
        np.asarray(node_ids, dtype=np.str_),
    )


def _load_trusted_planetoid(
    raw: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    try:
        from scipy import sparse  # type: ignore[import-untyped]
    except ImportError as exc:  # pragma: no cover - depends on research environment
        raise RuntimeError("转换 legacy Planetoid 需要安装 SciPy") from exc
    graph_file = next(raw.glob("ind.*.graph"))
    prefix = graph_file.name.removesuffix(".graph")

    def load_part(suffix: str) -> Any:
        with (raw / f"{prefix}.{suffix}").open("rb") as handle:
            return pickle.load(handle, encoding="latin1")

    x, y, tx, ty, allx, ally, graph = [
        load_part(name) for name in ("x", "y", "tx", "ty", "allx", "ally", "graph")
    ]
    train_count = int(y.shape[0])
    del x
    test_index_path = raw / f"{prefix}.test.index"
    test_reorder = np.asarray(
        [int(line) for line in test_index_path.read_text(encoding="utf-8").splitlines()],
        dtype=np.int64,
    )
    test_range = np.sort(test_reorder)
    if prefix.casefold().startswith("ind.citeseer"):
        full_range = np.arange(test_range.min(), test_range.max() + 1)
        extended_x = sparse.lil_matrix((len(full_range), allx.shape[1]))
        extended_x[test_range - full_range[0], :] = tx
        tx = extended_x
        extended_y = np.zeros((len(full_range), ty.shape[1]))
        extended_y[test_range - full_range[0], :] = ty
        ty = extended_y
    features = sparse.vstack((allx, tx)).tolil()
    features[test_reorder, :] = features[test_range, :]
    label_matrix = np.vstack((ally, ty))
    label_matrix[test_reorder, :] = label_matrix[test_range, :]
    labels = np.asarray(label_matrix.argmax(axis=1), dtype=np.int64)
    pairs = [(int(source), int(target)) for source, targets in graph.items() for target in targets]
    edges = (
        np.asarray(pairs, dtype=np.int64).T
        if pairs
        else np.empty((2, 0), dtype=np.int64)
    )
    train_mask = np.zeros(labels.shape[0], dtype=np.bool_)
    val_mask = np.zeros(labels.shape[0], dtype=np.bool_)
    test_mask = np.zeros(labels.shape[0], dtype=np.bool_)
    train_mask[:train_count] = True
    val_mask[train_count : min(labels.shape[0], train_count + 500)] = True
    test_mask[test_reorder] = True
    if np.any(train_mask & test_mask) or np.any(val_mask & test_mask):
        raise ValueError("Planetoid 官方 train/val/test 划分发生交叉")
    return (
        np.asarray(features.toarray(), dtype=np.float32),
        edges,
        labels,
        {
            "train_mask": train_mask,
            "val_mask": val_mask,
            "test_mask": test_mask,
        },
    )


def _npz_bytes(
    features: np.ndarray,
    edges: np.ndarray,
    labels: np.ndarray | None,
    extra_arrays: dict[str, np.ndarray] | None = None,
    *,
    node_ids: np.ndarray | None = None,
    directed: bool = False,
) -> bytes:
    output = io.BytesIO()
    values: dict[str, np.ndarray] = {
        "x": np.asarray(features, dtype=np.float32),
        "edge_index": np.asarray(edges, dtype=np.int64),
        "num_nodes": np.asarray(features.shape[0], dtype=np.int64),
        "directed": np.asarray(directed, dtype=np.bool_),
        "node_id_map": np.asarray(
            node_ids
            if node_ids is not None
            else [str(index) for index in range(features.shape[0])],
            dtype=np.str_,
        ),
    }
    if labels is not None:
        values["y"] = np.asarray(labels, dtype=np.int64)
    for name, array in (extra_arrays or {}).items():
        values[name] = np.asarray(array)
    np.savez_compressed(output, **values)  # type: ignore[arg-type]
    return output.getvalue()


_SPLIT_KEYS = ("train_mask", "val_mask", "test_mask", "train_idx", "val_idx", "test_idx")


def _validated_converter_splits(
    arrays: dict[str, np.ndarray],
    node_count: int,
) -> dict[str, np.ndarray]:
    normalized: dict[str, np.ndarray] = {}
    for name in _SPLIT_KEYS:
        if name not in arrays:
            continue
        value = np.asarray(arrays[name])
        if name.endswith("_mask"):
            if value.ndim not in {1, 2} or value.shape[0] != node_count:
                raise ValueError(f"{name} 必须是 [N] 或 [N,K] 掩码")
            if value.ndim == 2 and value.shape[1] == 0:
                raise ValueError(f"{name} 至少必须包含一折")
            if value.dtype.kind not in {"b", "u", "i"}:
                raise ValueError(f"{name} 必须是 bool/uint/int 掩码")
            if value.size and not bool(np.all((value == 0) | (value == 1))):
                raise ValueError(f"{name} 只能包含 0/1")
            normalized[name] = value.astype(np.bool_, copy=False)
        else:
            if value.ndim != 1 or not np.issubdtype(value.dtype, np.integer):
                raise ValueError(f"{name} 目前只支持单折一维整数索引")
            if value.size and (int(value.min()) < 0 or int(value.max()) >= node_count):
                raise ValueError(f"{name} 包含越界节点索引")
            if np.unique(value).size != value.size:
                raise ValueError(f"{name} 包含重复节点索引")
            normalized[name] = value.astype(np.int64, copy=False)

    mask_names = [name for name in _SPLIT_KEYS if name.endswith("_mask") and name in normalized]
    index_names = [name for name in _SPLIT_KEYS if name.endswith("_idx") and name in normalized]
    if mask_names and index_names:
        raise ValueError("mask 与 idx 划分不能混用")
    names = mask_names or index_names
    fold_count = 1
    if mask_names:
        fold_counts = {
            1 if normalized[name].ndim == 1 else int(normalized[name].shape[1])
            for name in names
        }
        if len(fold_counts) != 1:
            raise ValueError("train/val/test mask 折数不一致")
        fold_count = next(iter(fold_counts))
    for fold in range(fold_count):
        memberships: dict[str, np.ndarray] = {}
        for name in names:
            value = normalized[name]
            column = value if value.ndim == 1 else value[:, fold]
            memberships[name] = (
                np.flatnonzero(column) if name.endswith("_mask") else np.asarray(column)
            )
        for index, left_name in enumerate(names):
            for right_name in names[index + 1 :]:
                if np.intersect1d(
                    memberships[left_name],
                    memberships[right_name],
                    assume_unique=True,
                ).size:
                    raise ValueError(f"{left_name} 与 {right_name} 在第 {fold} 折存在交叉")
    return normalized


def _safe_split_arrays(dataset_dir: Path, node_count: int) -> tuple[dict[str, np.ndarray], list[str]]:
    sources: list[tuple[str, dict[str, np.ndarray]]] = []
    for split_path in sorted(dataset_dir.glob("**/*.npz")):
        try:
            with np.load(split_path, allow_pickle=False) as archive:
                arrays = {
                    key: np.asarray(archive[key])
                    for key in archive.files
                    if key in _SPLIT_KEYS
                }
        except (OSError, ValueError) as exc:
            if "split" in split_path.name.casefold():
                raise ValueError(f"无法安全读取 split 文件 {split_path.name}: {exc}") from exc
            continue
        if not arrays:
            continue
        relative = str(split_path.relative_to(dataset_dir)).replace("\\", "/")
        sources.append((relative, _validated_converter_splits(arrays, node_count)))
    if not sources:
        return {}, []
    expected_names = set(sources[0][1])
    for source, splits in sources[1:]:
        if set(splits) != expected_names:
            raise ValueError(f"split 文件 {source} 的 train/val/test 字段与其他折不对齐")
    if len(sources) > 1 and any(name.endswith("_idx") for name in expected_names):
        raise ValueError(
            "多文件变长 idx split 无法无损合并；请转换为 fewShotEpisodes"
        )
    result: dict[str, np.ndarray] = {}
    for name in _SPLIT_KEYS:
        if name not in expected_names:
            continue
        values = [splits[name] for _source, splits in sources]
        if len(values) == 1:
            result[name] = values[0]
        else:
            matrices = [value[:, None] if value.ndim == 1 else value for value in values]
            result[name] = np.concatenate(matrices, axis=1)
    return _validated_converter_splits(result, node_count), [source for source, _ in sources]


def _trusted_torch_arrays(path: Path) -> dict[str, np.ndarray]:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - research environment only
        raise RuntimeError("转换 few-shot PT 需要在转换器环境安装 PyTorch") from exc
    value = torch.load(str(path), map_location="cpu", weights_only=False)
    if isinstance(value, dict):
        arrays = {str(key): _as_numpy(item) for key, item in value.items()}
    elif isinstance(value, (list, tuple)):
        arrays = {f"value_{index}": _as_numpy(item) for index, item in enumerate(value)}
    else:
        arrays = {path.stem: _as_numpy(value)}
    for name, array in arrays.items():
        if array.dtype.hasobject:
            raise ValueError(f"few-shot {path.name}:{name} 不能转换 object dtype")
    return arrays


def _fewshot_payloads(
    source: Path,
) -> tuple[list[dict[str, Any]], list[tuple[str, bytes]], list[dict[str, str]]]:
    root = source / "fewshot_cora"
    if not root.is_dir():
        return [], [], []
    episodes: list[dict[str, Any]] = []
    payloads: list[tuple[str, bytes]] = []
    skipped: list[dict[str, str]] = []
    for episode_dir in sorted(
        (path for path in root.glob("**/*") if path.is_dir() and any(path.glob("*.pt"))),
        key=lambda path: str(path),
    ):
        try:
            arrays: dict[str, np.ndarray] = {}
            for tensor_path in sorted(episode_dir.glob("*.pt")):
                for name, value in _trusted_torch_arrays(tensor_path).items():
                    arrays[f"{tensor_path.stem}_{name}"] = value
            if not arrays:
                continue
            relative = episode_dir.relative_to(root)
            shot = relative.parts[0] if relative.parts else "few-shot"
            episode_id = relative.parts[-1]
            target = f"datasets/cora/episodes/{re_safe_slug(shot)}-{re_safe_slug(episode_id)}.npz"
            output = io.BytesIO()
            np.savez_compressed(output, **arrays)  # type: ignore[arg-type]
            episode_payload = output.getvalue()
            payloads.append((target, episode_payload))
            episodes.append(
                {
                    "shot": shot,
                    "episode": episode_id,
                    "path": target,
                    "splitSetId": f"fewshot-{re_safe_slug(shot)}-{re_safe_slug(episode_id)}",
                    "sha256": hashlib.sha256(episode_payload).hexdigest(),
                }
            )
        except Exception as exc:  # noqa: BLE001 - per-episode diagnostics
            skipped.append({"dataset": str(episode_dir), "reason": str(exc)[:500]})
    return episodes, payloads, skipped


def convert_pyg_dataset(
    input_path: Path,
    output_path: Path,
    *,
    trust_pickle: bool,
    pca_seed: int = 0,
) -> dict[str, Any]:
    if not trust_pickle:
        raise PermissionError(
            "拒绝读取 PT/pickle。只有确认输入目录完全可信时才可显式传入 --trust-pickle。"
        )
    source = input_path.resolve(strict=True)
    if not source.is_dir():
        raise ValueError("--input 必须指向数据目录")
    destination = output_path.resolve()
    if destination.exists():
        raise FileExistsError(f"输出文件已存在: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)

    converted: list[dict[str, Any]] = []
    skipped: list[dict[str, str]] = []
    payloads: list[tuple[str, bytes]] = []
    for dataset_dir in sorted((path for path in source.iterdir() if path.is_dir()), key=lambda p: p.name):
        if dataset_dir.name.casefold() == "fewshot_cora":
            continue
        try:
            features: np.ndarray
            edges: np.ndarray
            labels: np.ndarray | None
            embedded_splits: dict[str, np.ndarray] = {}
            node_ids: np.ndarray | None = None
            geom_raw = next(dataset_dir.glob("**/out1_node_feature_label.txt"), None)
            if geom_raw is not None:
                features, edges, labels, node_ids = _load_geom(geom_raw.parent)
                source_kind = "geom_gcn_text"
            else:
                data_pt = next(dataset_dir.glob("**/processed/data.pt"), None)
                if data_pt is not None:
                    try:
                        features, edges, labels, embedded_splits = _load_trusted_torch(data_pt)
                        source_kind = "trusted_torch_pyg"
                    except RuntimeError:
                        raw_graph = next(dataset_dir.glob("**/ind.*.graph"), None)
                        if raw_graph is None:
                            raise
                        features, edges, labels, embedded_splits = _load_trusted_planetoid(
                            raw_graph.parent
                        )
                        source_kind = "trusted_planetoid_pickle"
                else:
                    raw_graph = next(dataset_dir.glob("**/ind.*.graph"), None)
                    if raw_graph is None:
                        skipped.append({"dataset": dataset_dir.name, "reason": "未发现主图"})
                        continue
                    features, edges, labels, embedded_splits = _load_trusted_planetoid(
                        raw_graph.parent
                    )
                    source_kind = "trusted_planetoid_pickle"
            file_splits, split_files = _safe_split_arrays(dataset_dir, int(features.shape[0]))
            embedded_splits = _validated_converter_splits(
                embedded_splits,
                int(features.shape[0]),
            )
            split_arrays = dict(embedded_splits)
            for name, value in file_splits.items():
                if name in split_arrays:
                    if not np.array_equal(split_arrays[name], value):
                        raise ValueError(f"外部 {name} 与 PyG 内嵌官方划分冲突")
                    continue
                split_arrays[name] = value
            split_arrays = _validated_converter_splits(
                split_arrays,
                int(features.shape[0]),
            )
            variant_arrays, variant_recipes = _weighted_structure_variant_arrays(
                features,
                edges,
                pca_seed=pca_seed,
            )
            slug = re_safe_slug(dataset_dir.name)
            member = f"datasets/{slug}/graph.npz"
            payloads.append(
                (
                    member,
                    _npz_bytes(
                        features,
                        edges,
                        labels,
                        {**split_arrays, **variant_arrays},
                        node_ids=node_ids,
                        directed=False,
                    ),
                )
            )
            source_files = [
                str(path.relative_to(dataset_dir)).replace("\\", "/")
                for path in sorted(dataset_dir.glob("**/*"))
                if path.is_file()
            ]
            source_file_digests = [
                {
                    "path": str(path.relative_to(dataset_dir)).replace("\\", "/"),
                    "size": path.stat().st_size,
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                }
                for path in sorted(dataset_dir.glob("**/*"))
                if path.is_file()
            ]
            converted.append(
                {
                    "name": dataset_dir.name,
                    "path": member,
                    "sourceFormat": source_kind,
                    "nodeCount": int(features.shape[0]),
                    "edgeCount": int(edges.shape[1]),
                    "featureDimension": int(features.shape[1]),
                    "splitFiles": split_files,
                    "sourceFiles": source_files,
                    "sourceFileDigests": source_file_digests,
                    "license": "unknown",
                    "datasetRole": "benchmark",
                    "splitKind": (
                        "official"
                        if dataset_dir.name.casefold() in {"cora", "citeseer", "pubmed"}
                        and split_arrays
                        else "published"
                        if split_files
                        else "source"
                    ),
                    "directed": False,
                    "transformRecipes": [
                        {
                            "id": "identity-v1",
                            "graphVariant": "raw",
                            "directed": False,
                            "selfLoopPolicy": "preserve",
                            "duplicatePolicy": "preserve",
                            "featureTransform": "identity",
                        },
                        *variant_recipes,
                    ],
                    "transforms": [
                        "trusted_pickle_to_safe_npz"
                        if source_kind.startswith("trusted_")
                        else "geom_text_to_safe_npz",
                        "preserve_source_topology",
                        "preserve_features_labels_and_splits",
                    ],
                }
            )
        except Exception as exc:  # noqa: BLE001 - report per-dataset converter failures
            skipped.append({"dataset": dataset_dir.name, "reason": str(exc)[:500]})

    episodes, episode_payloads, episode_skipped = _fewshot_payloads(source)
    skipped.extend(episode_skipped)
    if episodes:
        cora = next((item for item in converted if item["name"].casefold() == "cora"), None)
        if cora is not None:
            cora["fewShotEpisodes"] = episodes
            payloads.extend(episode_payloads)
        else:
            skipped.append(
                {"dataset": "fewshot_cora", "reason": "未找到 Cora 主图，episode 未写入产物"}
            )
    if not converted:
        raise RuntimeError("没有成功转换任何主图；请检查输入结构和研究环境依赖")
    manifest = {
        "schemaVersion": "socialgraph-fm-dataset-package/1.0",
        "trustedSource": str(source),
        "sourceFingerprint": _directory_fingerprint(source),
        "datasets": converted,
        "skipped": skipped,
    }
    with zipfile.ZipFile(destination, "x", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("manifest.json", json.dumps(manifest, ensure_ascii=False, indent=2))
        for name, payload in payloads:
            archive.writestr(name, payload)
    return manifest


def convert_ogbl_collab(
    source: Path,
    destination: Path,
    *,
    trust_pickle: bool,
) -> dict[str, Any]:
    """Convert an existing local OGB cache into the safe SGFM package.

    OGB/PyG caches contain PT files, so this command is intentionally reachable
    only through the loopback trusted-conversion process.
    """

    if not trust_pickle:
        raise ValueError("ogbl-collab 本地缓存包含 PT；必须显式 --trust-pickle")
    source = source.expanduser().resolve(strict=True)
    if destination.exists():
        raise FileExistsError(f"输出已存在: {destination}")
    marker_names = {"ogbl_collab", "ogbl-collab"}
    cache_roots = [
        path
        for path in (source, *source.iterdir())
        if path.is_dir() and path.name.casefold() in marker_names
    ]
    if not cache_roots:
        raise ValueError("未在可信目录中发现 ogbl-collab 本地缓存")

    def has_processed(root: Path) -> bool:
        processed = root / "processed"
        return processed.is_dir() and any(path.is_file() for path in processed.iterdir())

    def has_complete_release_v1(root: Path) -> bool:
        required = (
            root / "RELEASE_v1.txt",
            root / "raw" / "edge.csv.gz",
            root / "raw" / "edge_weight.csv.gz",
            root / "raw" / "edge_year.csv.gz",
            root / "raw" / "node-feat.csv.gz",
            root / "raw" / "num-edge-list.csv.gz",
            root / "raw" / "num-node-list.csv.gz",
            root / "split" / "time" / "train.pt",
            root / "split" / "time" / "valid.pt",
            root / "split" / "time" / "test.pt",
        )
        return all(path.is_file() for path in required)

    # A fresh cache extracted from the pinned official archive legitimately has
    # no processed PT yet.  PyG may build it locally only when the entire
    # release-v1 raw/split set is present; an incomplete directory is rejected
    # before the dataset loader can attempt a network download.  Existing
    # processed caches remain an explicit --trust-pickle caller responsibility.
    if not any(has_processed(root) or has_complete_release_v1(root) for root in cache_roots):
        raise ValueError(
            "ogbl-collab 缓存既无 processed 产物，也无完整 release-v1 raw/split；"
            "适配器禁止联网下载"
        )
    try:
        from ogb.linkproppred import PygLinkPropPredDataset  # type: ignore[import-not-found]
    except ImportError as exc:
        raise RuntimeError("转换 ogbl-collab 需要安装 ogb 与 PyG") from exc

    ogb_root = source.parent if source.name.casefold() in marker_names else source
    with _trusted_legacy_torch_load():
        dataset = PygLinkPropPredDataset(name="ogbl-collab", root=str(ogb_root))
        data = dataset[0]
        split = dataset.get_edge_split()
    edge_index = _as_numpy(data.edge_index).astype(np.int64, copy=False)
    features = _as_numpy(data.x).astype(np.float32, copy=False)
    edge_weight = _as_numpy(data.edge_weight).reshape(-1).astype(np.float32, copy=False)
    edge_year = _as_numpy(data.edge_year).reshape(-1).astype(np.int16, copy=False)
    if edge_index.ndim != 2 or edge_index.shape[0] != 2:
        raise ValueError("OGB edge_index 不是 [2,E]")
    if edge_weight.size != edge_index.shape[1] or edge_year.size != edge_index.shape[1]:
        raise ValueError("OGB 边权重/年份与训练消息图不对齐")
    if edge_year.size and int(edge_year.max()) > 2017:
        raise ValueError("OGB 消息传递图包含 2018 年及之后边，存在时间泄漏")

    def positive(part: str) -> np.ndarray:
        value = _as_numpy(split[part]["edge"])
        if value.ndim != 2 or value.shape[1] != 2:
            raise ValueError(f"OGB {part} edge 不是 [E,2]")
        return value.T.astype(np.int64, copy=False)

    def negative(part: str) -> np.ndarray:
        value = _as_numpy(split[part]["edge_neg"])
        if value.ndim < 2 or value.shape[-1] != 2:
            raise ValueError(f"OGB {part} edge_neg 末维不是 2")
        return value.reshape(-1, 2).T.astype(np.int64, copy=False)

    train_positive = positive("train")
    validation_positive = positive("valid")
    test_positive = positive("test")
    validation_negative = negative("valid")
    test_negative = negative("test")
    arrays: dict[str, np.ndarray] = {
        "edge_index": edge_index,
        "x": features,
        "num_nodes": np.asarray(features.shape[0], dtype=np.int64),
        "node_id_map": np.asarray([str(index) for index in range(features.shape[0])]),
        "directed": np.asarray(False, dtype=np.bool_),
        "edge_weight": edge_weight,
        "edge_timestamp": edge_year,
        "variant_train_positive": train_positive,
        "variant_validation_positive": validation_positive,
        "variant_test_positive": test_positive,
        "variant_validation_negative": validation_negative,
        "variant_test_negative": test_negative,
    }
    graph_buffer = io.BytesIO()
    np.savez_compressed(graph_buffer, **arrays)  # type: ignore[arg-type]
    graph_path = "datasets/ogbl-collab/graph.npz"
    recorded_at = "2026-08-11T00:00:00Z"
    item = {
        "name": "ogbl-collab",
        "path": graph_path,
        "sourceFormat": "trusted_local_ogb",
        "sourceFiles": ["OGB local cache"],
        "datasetRole": "benchmark",
        "splitKind": "official",
        "directed": False,
        "licensePolicy": {
            "status": "verified",
            "identifier": "ODC-BY-1.0",
            "sourceUrl": "https://ogb.stanford.edu/docs/linkprop/#ogbl-collab",
            "allowedUses": ["evaluation", "adaptation", "inference", "pretraining"],
            "attribution": "Open Graph Benchmark: ogbl-collab",
            "evidenceIds": ["ogbl-collab-official-metadata"],
        },
        "licenseEvidence": [
            {
                "id": "ogbl-collab-official-metadata",
                "kind": "official_metadata",
                "sourceUrl": "https://ogb.stanford.edu/docs/linkprop/#ogbl-collab",
                "recordedAt": recorded_at,
                "recordedBy": "socialgraph-fm-ogb-adapter",
            }
        ],
        "dataGovernance": {
            "containsPersonalData": False,
            "deidentified": True,
            "attributeAllowlist": [],
            "excludedAttributes": [],
            "retention": "research_archive",
            "userDataTrainingOptIn": False,
        },
        "transformRecipes": [
            {
                "id": "ogb-identity-v1",
                "graphVariant": "raw",
                "inputArray": "x",
                "outputArray": "x",
                "featureTransform": "identity",
                "fitScope": "none",
                "parameters": {"ogbDataset": "ogbl-collab"},
            }
        ],
        "linkPredictionProtocol": {
            "messagePassingEdgeArray": "edge_index",
            "trainPositiveArray": "variant_train_positive",
            "validationPositiveArray": "variant_validation_positive",
            "testPositiveArray": "variant_test_positive",
            "validationNegativeArray": "variant_validation_negative",
            "testNegativeArray": "variant_test_negative",
            "edgeYearArray": "edge_timestamp",
            "edgeWeightArray": "edge_weight",
            "trainYearMax": 2017,
            "validationYear": 2018,
            "testYear": 2019,
            "negativeSampler": "stored",
            "undirectedCanonicalization": "min_max",
            "reverseEdgeLeakagePolicy": "reject",
            "positiveOverlapPolicy": "allow_temporal_recurrence",
            "evaluator": "ogb.linkproppred.Evaluator(ogbl-collab)",
            "evaluatorVersion": getattr(__import__("ogb"), "__version__", "unknown"),
        },
        "transforms": ["ogb_local_cache_to_safe_npz", "preserve_official_temporal_split"],
    }
    manifest = {
        "schemaVersion": "socialgraph-fm-dataset-package/1.0",
        # Do not leak or hash a machine-specific cache path into a portable
        # corpus package.  Integrity is carried by sourceFingerprint and the
        # GFM bootstrap's pinned official archive receipt.
        "trustedSource": "local-official-ogbl-collab-cache",
        "sourceFingerprint": _directory_fingerprint(source),
        "datasets": [item],
        "skipped": [],
    }

    def deterministic_entry(name: str) -> zipfile.ZipInfo:
        entry = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
        entry.compress_type = zipfile.ZIP_DEFLATED
        entry.create_system = 3
        entry.external_attr = 0o100644 << 16
        return entry

    destination.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(destination, "x", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(
            deterministic_entry("manifest.json"),
            json.dumps(
                manifest,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8"),
        )
        archive.writestr(deterministic_entry(graph_path), graph_buffer.getvalue())
    return manifest


def _directory_fingerprint(source: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted((item for item in source.glob("**/*") if item.is_file()), key=str):
        digest.update(str(path.relative_to(source)).replace("\\", "/").encode("utf-8"))
        digest.update(b"\x00")
        with path.open("rb") as handle:
            while chunk := handle.read(1024 * 1024):
                digest.update(chunk)
        digest.update(b"\x00")
    return digest.hexdigest()


def re_safe_slug(value: str) -> str:
    safe = "".join(character if character.isalnum() or character in "-_" else "-" for character in value)
    return safe.strip("-") or "dataset"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="SocialGraph-FM 可信研究数据转换工具")
    subcommands = parser.add_subparsers(dest="command", required=True)
    command = subcommands.add_parser("convert-pyg", help="转换可信的 PyG 数据目录")
    command.add_argument("--input", type=Path, required=True)
    command.add_argument("--output", type=Path, required=True)
    command.add_argument(
        "--trust-pickle",
        action="store_true",
        help="确认输入目录完全可信，允许加载 PT/Pickle（具有代码执行风险）",
    )
    ogb_command = subcommands.add_parser(
        "convert-ogbl-collab", help="转换本地 OGB ogbl-collab 缓存"
    )
    ogb_command.add_argument("--input", type=Path, required=True)
    ogb_command.add_argument("--output", type=Path, required=True)
    ogb_command.add_argument("--trust-pickle", action="store_true")
    command.add_argument(
        "--pca-seed",
        type=int,
        default=0,
        help="加权结构变体的 PCA 随机种子（会写入 transform recipe 与 contentHash）",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    _apply_process_memory_limit()
    args = _build_parser().parse_args(argv)
    try:
        if args.command == "convert-pyg":
            manifest = convert_pyg_dataset(
                args.input,
                args.output,
                trust_pickle=args.trust_pickle,
                pca_seed=args.pca_seed,
            )
            print(
                json.dumps(
                    {
                        "output": str(args.output.resolve()),
                        "converted": len(manifest["datasets"]),
                        "skipped": len(manifest["skipped"]),
                    },
                    ensure_ascii=False,
                )
            )
            return 0
        if args.command == "convert-ogbl-collab":
            manifest = convert_ogbl_collab(
                args.input,
                args.output,
                trust_pickle=args.trust_pickle,
            )
            print(
                json.dumps(
                    {"output": str(args.output.resolve()), "converted": 1, "skipped": 0},
                    ensure_ascii=False,
                )
            )
            return 0
    except (FileNotFoundError, FileExistsError, PermissionError, RuntimeError, ValueError) as exc:
        print(f"转换失败: {exc}", file=sys.stderr)
        return 2
    return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
