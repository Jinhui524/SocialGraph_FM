"""Torch-free inspection of generic Governance target-task bundles."""

from __future__ import annotations

import csv
import hashlib
import io
import stat
import zipfile
from dataclasses import dataclass

from pydantic import ValidationError

from .gfm_client import GfmProxyError
from .gfm_hashing import canonical_sha256
from .gfm_governance_artifacts import inspect_governance_bundle
from .gfm_governance_schemas import (
    TargetDomainReceipt,
    TargetLabelReceiptV2,
    TargetLabelSetV2,
    TargetTaskDocument,
)


@dataclass(frozen=True)
class InspectedTargetTask:
    task: TargetTaskDocument
    receipt: TargetDomainReceipt
    labels: TargetLabelSetV2 | None
    label_receipt: TargetLabelReceiptV2 | None
    inference: bytes
    node_degrees: dict[str, int]
    node_strata: dict[str, int]


def _sha(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def inspect_target_task_bundle(
    payload: bytes, *, max_expanded_bytes: int
) -> InspectedTargetTask:
    try:
        with zipfile.ZipFile(io.BytesIO(payload)) as archive:
            infos = archive.infolist()
            names = [info.filename for info in infos]
            if len(names) != len(set(names)) or len(names) not in {3, 5}:
                raise ValueError("outer inventory is invalid")
            expected = {"task.json", "inference.zip", "target-receipt.json"}
            if len(names) == 5:
                expected |= {"labels.json", "label-receipt.json"}
            if set(names) != expected:
                raise ValueError("outer inventory is invalid")
            expanded = 0
            for info in infos:
                mode = info.external_attr >> 16
                if (
                    info.filename.startswith(("/", "\\"))
                    or "/" in info.filename
                    or "\\" in info.filename
                    or info.filename in {".", ".."}
                    or stat.S_ISLNK(mode)
                    or info.file_size < 1
                ):
                    raise ValueError("unsafe target-task archive member")
                expanded += info.file_size
                if expanded > max_expanded_bytes:
                    raise ValueError("target-task archive is too large")
                if info.compress_size == 0 or info.file_size > info.compress_size * 200:
                    raise ValueError("target-task compression ratio is unsafe")
            entries = {name: archive.read(name) for name in names}
        task = TargetTaskDocument.model_validate_json(entries["task.json"])
        descriptors = {
            task.inference.name: task.inference,
            task.target_receipt.name: task.target_receipt,
        }
        if task.labels is not None:
            descriptors[task.labels.name] = task.labels
        if task.label_receipt is not None:
            descriptors[task.label_receipt.name] = task.label_receipt
        if set(descriptors) != set(entries) - {"task.json"}:
            raise ValueError("task descriptor inventory mismatch")
        for name, descriptor in descriptors.items():
            if descriptor.bytes != len(entries[name]) or descriptor.sha256 != _sha(entries[name]):
                raise ValueError("task descriptor digest mismatch")
        receipt = TargetDomainReceipt.model_validate_json(entries["target-receipt.json"])
        labels = (
            TargetLabelSetV2.model_validate_json(entries["labels.json"])
            if "labels.json" in entries
            else None
        )
        label_receipt = (
            TargetLabelReceiptV2.model_validate_json(entries["label-receipt.json"])
            if "label-receipt.json" in entries
            else None
        )
    except (
        KeyError,
        OSError,
        RuntimeError,
        ValueError,
        zipfile.BadZipFile,
        ValidationError,
    ) as error:
        raise GfmProxyError(400, "GOVERNANCE_TARGET_TASK_INVALID") from error

    inference = entries["inference.zip"]
    try:
        _, inspected = inspect_governance_bundle(
            inference,
            clean_self_loops=False,
            max_expanded_bytes=max_expanded_bytes,
        )
        with zipfile.ZipFile(io.BytesIO(inference)) as archive:
            node_rows = list(
                csv.DictReader(io.StringIO(archive.read("nodes.csv").decode("utf-8")))
            )
            relation_rows = list(
                csv.DictReader(
                    io.StringIO(archive.read("relations.csv").decode("utf-8"))
                )
            )
        node_ids = tuple(str(row["node_id"]) for row in node_rows)
        pairs = {
            tuple(sorted((str(row["source"]), str(row["target"]))))
            for row in relation_rows
        }
        degrees = {node_id: 0 for node_id in node_ids}
        adjacency: dict[str, set[str]] = {node_id: set() for node_id in node_ids}
        for source, target in pairs:
            degrees[source] += 1
            degrees[target] += 1
            adjacency[source].add(target)
            adjacency[target].add(source)
        reached = set()
        frontier = [node_ids[0]]
        while frontier:
            current = frontier.pop()
            if current in reached:
                continue
            reached.add(current)
            frontier.extend(adjacency[current] - reached)
        identity = (
            task.task_id == receipt.task_id
            and task.node_count == receipt.node_count == inspected["nodeCount"]
            and task.fused_edge_count == receipt.fused_edge_count == len(pairs)
            and task.modalities == receipt.modalities == tuple(inspected["modalities"])
            and task.inference.sha256 == receipt.inference_sha256 == _sha(inference)
            and receipt.node_set_sha256 == canonical_sha256(sorted(node_ids))
            and receipt.connected == (len(reached) == len(node_ids))
        )
        if not identity:
            raise ValueError("target-task reconstructed graph identity mismatch")
        if labels is not None:
            if label_receipt is None or labels.task_id != task.task_id or labels.inference_sha256 != task.inference.sha256:
                raise ValueError("detached label binding mismatch")
            ordered_label_nodes = sorted(
                node_ids, key=lambda node_id: (degrees[node_id], node_id)
            )
            authoritative_strata = {
                node_id: min(3, position * 4 // len(ordered_label_nodes))
                for position, node_id in enumerate(ordered_label_nodes)
            }
            if (
                label_receipt.task_id != task.task_id
                or label_receipt.target_receipt_hash != receipt.receipt_hash
                or label_receipt.labels_sha256 != task.labels.sha256  # type: ignore[union-attr]
                or any(
                    row.node_id not in degrees
                    or row.node_id not in label_receipt.eligible_node_ids
                    or row.fused_degree != degrees[row.node_id]
                    or row.structural_stratum != authoritative_strata[row.node_id]
                    for row in labels.labels
                )
            ):
                raise ValueError("detached label receipt mismatch")
    except (KeyError, OSError, ValueError, zipfile.BadZipFile) as error:
        raise GfmProxyError(400, "GOVERNANCE_TARGET_TASK_INVALID") from error
    ordered = sorted(node_ids, key=lambda node_id: (degrees[node_id], node_id))
    strata = {
        node_id: min(3, position * 4 // len(ordered))
        for position, node_id in enumerate(ordered)
    }
    return InspectedTargetTask(
        task, receipt, labels, label_receipt, inference, degrees, strata
    )


__all__ = ["InspectedTargetTask", "inspect_target_task_bundle"]
