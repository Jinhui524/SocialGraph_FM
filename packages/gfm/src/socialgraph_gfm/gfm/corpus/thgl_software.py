"""TGB 2.0 ``thgl-software`` acquisition and pickle-free materialization."""

from __future__ import annotations

import csv
import os
import shutil
import stat
import tempfile
import urllib.request
import uuid
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Mapping

import numpy as np

from ...canonical import file_sha256
from ...errors import ContractViolation
from ...runtime import RuntimeLayout
from .common import (
    NumericShardWriter,
    atomic_write_json,
    build_manifest,
    load_npz_safe,
    read_json_object,
    resolve_within,
    verify_manifest,
)

CORPUS_ID = "thgl-software-2.0.0"
DOMAIN_ID = "thgl-software-2.0.0"
DATASET_NAME = "thgl-software"
TGB_VERSION = "2.3.0"
TGB_REFERENCE_COMMIT = "740ff5ada7c52e38854ad13a7ac37245b162fa3d"
TGB_PREPROCESS_URL = (
    "https://github.com/shenyangHuang/TGB/blob/"
    f"{TGB_REFERENCE_COMMIT}/tgb/utils/pre_process.py"
)
DATASET_RELEASE = "2.0.0"
LICENSE_ID = "CC-BY-4.0"
LICENSE_URL = "https://tgb.complexdatalab.com/docs/thg/"
SOURCE_URL = (
    "https://object-arbutus.alliancecan.ca/swift/v1/"
    "14c95234f6cd4a21a47deafe20cce2a7/tgb/thgl-software.zip"
)
EXPECTED_NODE_TYPES = 4
EXPECTED_RELATIONS = 14
PHYSICAL_ACCESS_SCHEMA = "gfm.physical-role-views/1.0"
ACCESS_ROLES = ("train", "validation", "test", "shadow")
EXPECTED_NODE_COUNT = 681_927
EXPECTED_EDGE_COUNT = 1_489_806
EXPECTED_ARCHIVE_SIZE = 1_492_169_637
EXPECTED_ARCHIVE_ETAG = "acda1521b6c43871cb1b40ff1725f4ed"
Download = Callable[[str, Path], None]
MetadataClient = Callable[[str], Mapping[str, str]]


def _fail(message: str) -> ContractViolation:
    return ContractViolation(f"thgl-software: {message}")


def _download(url: str, destination: Path) -> None:
    # Some proxies close this 1.49 GB response cleanly before Content-Length is
    # reached.  Treat EOF as progress, never as completion, and continue only
    # when the official server acknowledges the exact byte range.  The caller
    # still validates size, ZIP structure and every member CRC before publish.
    current = destination.stat().st_size if destination.exists() else 0
    if current > EXPECTED_ARCHIVE_SIZE:
        raise _fail("partial official download exceeds the fixed archive size")
    attempts_without_completion = 0
    try:
        while current < EXPECTED_ARCHIVE_SIZE:
            headers = {"User-Agent": "SocialGraph-FM/1.0"}
            if current:
                headers["Range"] = f"bytes={current}-"
            request = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(request, timeout=120) as source:  # noqa: S310
                status = int(getattr(source, "status", source.getcode()))
                content_range = str(source.headers.get("Content-Range", ""))
                if current and (
                    status != 206
                    or not content_range.startswith(f"bytes {current}-")
                ):
                    raise _fail("official server did not honor the exact resume range")
                mode = "ab" if current else "xb"
                with destination.open(mode) as target:
                    shutil.copyfileobj(source, target, 4 * 1024 * 1024)
                    target.flush()
                    os.fsync(target.fileno())
            updated = destination.stat().st_size
            if updated <= current or updated > EXPECTED_ARCHIVE_SIZE:
                raise _fail("official ranged download made invalid progress")
            current = updated
            attempts_without_completion += 1
            if attempts_without_completion > 16:
                raise _fail("official ranged download exceeded its retry bound")
    except OSError as exc:
        # The fetch workflow owns and removes its UUID-scoped temporary on
        # failure.  A maintenance caller may retain an explicit partial in the
        # runtime tmp directory and safely resume it later.
        raise _fail("official download failed") from exc


def _official_metadata(url: str) -> Mapping[str, str]:
    request = urllib.request.Request(
        url,
        method="HEAD",
        headers={"User-Agent": "SocialGraph-FM/1.0"},
    )
    try:
        with urllib.request.urlopen(request, timeout=60) as response:  # noqa: S310
            return {
                "content-length": str(response.headers.get("Content-Length", "")),
                "etag": str(response.headers.get("ETag", "")).strip('"'),
            }
    except OSError as exc:
        raise _fail("official archive metadata request failed") from exc


def _validate_archive(path: Path, *, formal_source: bool) -> None:
    """Validate a complete archive before it can receive the stable raw name."""

    if not path.is_file() or path.is_symlink():
        raise _fail("downloaded source is not a regular archive")
    if formal_source and path.stat().st_size != EXPECTED_ARCHIVE_SIZE:
        raise _fail("downloaded official archive size differs from fixed metadata")
    try:
        with zipfile.ZipFile(path) as package:
            infos = package.infolist()
            if not infos or len(infos) > 1_000:
                raise _fail("official archive is empty")
            total = 0
            for info in infos:
                name = info.filename
                if (
                    "\\" in name
                    or name.startswith("/")
                    or ":" in name
                    or ".." in Path(name).parts
                ):
                    raise _fail("official archive contains an unsafe path")
                if info.flag_bits & 0x1:
                    raise _fail("official archive contains an encrypted member")
                if ((info.external_attr >> 16) & 0o170000) == stat.S_IFLNK:
                    raise _fail("official archive contains a symbolic link")
                total += info.file_size
                if total > 20 * 1024**3:
                    raise _fail("official archive exceeds the uncompressed safety limit")
                if (
                    info.file_size
                    and info.file_size / max(info.compress_size, 1) > 10_000
                ):
                    raise _fail("official archive has a suspicious compression ratio")
            # Reading the central directory alone is insufficient: a truncated
            # member can retain a syntactically valid directory.  testzip()
            # streams and CRC-checks every member before publication.
            if package.testzip() is not None:
                raise _fail("official archive member failed its CRC check")
    except zipfile.BadZipFile as exc:
        raise _fail("official archive is not a valid ZIP") from exc


def fetch_thgl_software(
    root: str | Path,
    *,
    accept_license: str,
    downloader: Download | None = None,
    metadata_client: MetadataClient | None = None,
) -> dict[str, Any]:
    """Download the official TGB archive only after explicit attribution consent."""

    if accept_license != LICENSE_ID:
        raise _fail(f"accept_license must be exactly {LICENSE_ID}")
    layout = RuntimeLayout.from_root(root)
    raw = layout.raw_thgl_software
    raw.mkdir(parents=True, exist_ok=True)
    archive = raw / "thgl-software-2.0.0.zip"
    formal_source = downloader is None
    source_metadata: Mapping[str, str] = {}
    if formal_source:
        source_metadata = (metadata_client or _official_metadata)(SOURCE_URL)
        try:
            source_size = int(source_metadata.get("content-length", ""))
        except ValueError as exc:
            raise _fail("official archive Content-Length is invalid") from exc
        source_etag = source_metadata.get("etag", "").strip('"').casefold()
        if (
            source_size != EXPECTED_ARCHIVE_SIZE
            or source_etag != EXPECTED_ARCHIVE_ETAG
        ):
            raise _fail("official archive size/ETag drifted from release 2.0.0")
    if not archive.exists():
        temporary = archive.with_name(f".{archive.name}.{uuid.uuid4().hex}.tmp")
        try:
            (downloader or _download)(SOURCE_URL, temporary)
            _validate_archive(temporary, formal_source=formal_source)
            os.replace(temporary, archive)
        finally:
            temporary.unlink(missing_ok=True)
    _validate_archive(archive, formal_source=formal_source)
    receipt = {
        "schemaVersion": "gfm.thgl-software-fetch/1.0",
        "dataset": DATASET_NAME,
        "tgbReferenceVersion": TGB_VERSION,
        "tgbReferenceCommit": TGB_REFERENCE_COMMIT,
        "datasetRelease": DATASET_RELEASE,
        "sourceUrl": SOURCE_URL,
        "archiveSha256": file_sha256(archive),
        "archiveSize": archive.stat().st_size,
        "sourceEtag": (
            source_metadata.get("etag", "").strip('"').casefold()
            if formal_source
            else None
        ),
        "formalEligible": formal_source,
        "licenseId": LICENSE_ID,
        "licenseEvidence": LICENSE_URL,
        "attribution": "Temporal Graph Benchmark 2.0, thgl-software",
    }
    atomic_write_json(raw / "fetch-receipt.json", receipt)
    return receipt


def _as_numpy(value: Any, *, name: str, dtype: np.dtype[Any]) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    array = np.asarray(value)
    if array.dtype.hasobject:
        raise _fail(f"runtime array {name} uses object dtype")
    return np.ascontiguousarray(array, dtype=dtype)


def _extract_official_archive(archive: Path, isolated_root: Path) -> None:
    """Safely extract only into ``isolated_root/thgl_software``."""

    dataset_root = isolated_root / "thgl_software"
    dataset_root.mkdir(parents=True, exist_ok=False)
    targets: set[Path] = set()
    try:
        package = zipfile.ZipFile(archive)
    except zipfile.BadZipFile as exc:
        raise _fail("accepted official archive is not a valid ZIP") from exc
    with package:
        infos = package.infolist()
        if not infos or len(infos) > 1_000:
            raise _fail("accepted official archive entry inventory is invalid")
        total = 0
        for info in infos:
            name = info.filename
            if not name or "\\" in name or "\x00" in name or ":" in name:
                raise _fail("accepted official archive contains an unsafe member")
            relative = PurePosixPath(name)
            if relative.is_absolute() or any(part in {"", ".", ".."} for part in relative.parts):
                raise _fail("accepted official archive contains path traversal")
            if info.flag_bits & 0x1 or ((info.external_attr >> 16) & 0o170000) == stat.S_IFLNK:
                raise _fail("accepted official archive contains an encrypted link/member")
            total += info.file_size
            if total > 20 * 1024**3:
                raise _fail("accepted official archive exceeds its extraction limit")
            parts = relative.parts
            if parts[0] in {"thgl_software", "thgl-software"}:
                parts = parts[1:]
            if not parts:
                if info.is_dir():
                    continue
                raise _fail("accepted official archive contains an empty file path")
            target = dataset_root.joinpath(*parts)
            resolved = target.resolve(strict=False)
            try:
                resolved.relative_to(dataset_root.resolve())
            except ValueError as exc:
                raise _fail("accepted official archive escapes the isolated dataset root") from exc
            if resolved in targets:
                raise _fail("accepted official archive normalizes to duplicate paths")
            targets.add(resolved)
            if info.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            copied = 0
            with package.open(info) as source, target.open("xb") as destination:
                while True:
                    chunk = source.read(4 * 1024 * 1024)
                    if not chunk:
                        break
                    copied += len(chunk)
                    if copied > info.file_size:
                        raise _fail("archive member exceeded its declared size")
                    destination.write(chunk)
                destination.flush()
                os.fsync(destination.fileno())
            if copied != info.file_size:
                raise _fail("archive member size changed during extraction")


def _one_csv(dataset_root: Path, expected_name: str) -> Path:
    matches = [
        path
        for path in dataset_root.rglob(expected_name)
        if path.is_file() and not path.is_symlink()
    ]
    if len(matches) != 1:
        raise _fail(f"official archive must contain exactly one {expected_name}")
    return matches[0]


def _count_csv_rows(path: Path) -> int:
    with path.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.reader(stream)
        try:
            next(reader)
        except StopIteration as exc:
            raise _fail(f"official CSV {path.name} is empty") from exc
        count = sum(1 for row in reader if row)
    if count < 1:
        raise _fail(f"official CSV {path.name} has no data rows")
    return count


def _parse_official_csv(dataset_root: Path) -> dict[str, np.ndarray]:
    """Reproduce TGB v2.3.0's numeric CSV adapter without its pickle boundary.

    TGB's ``csv_to_thg_data`` maps original integer node IDs in first-seen
    edge order, while ``process_node_type`` applies the companion type table.
    Its official split uses timestamp quantiles at 0.70 and 0.85.  This local
    implementation is pinned to ``TGB_REFERENCE_COMMIT`` and tests those exact
    semantics, but never imports TGB, pandas, ``clint`` or generated pickle files.
    """

    edge_path = _one_csv(dataset_root, "thgl-software_edgelist.csv")
    type_path = _one_csv(dataset_root, "thgl-software_nodetype.csv")
    edge_count = _count_csv_rows(edge_path)
    sources = np.empty(edge_count, dtype=np.int64)
    destinations = np.empty(edge_count, dtype=np.int64)
    timestamps = np.empty(edge_count, dtype=np.int64)
    relations = np.empty(edge_count, dtype=np.int16)
    node_ids: dict[int, int] = {}

    with edge_path.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.reader(stream)
        header = next(reader, None)
        if header is None or len(header) != 4:
            raise _fail("official edge CSV header must contain four columns")
        index = 0
        for line_number, row in enumerate(reader, start=2):
            if not row:
                continue
            if len(row) != 4 or index >= edge_count:
                raise _fail(f"official edge CSV row {line_number} is malformed")
            try:
                timestamp, original_source, original_target, relation = (
                    int(value) for value in row
                )
            except ValueError as exc:
                raise _fail(f"official edge CSV row {line_number} is not integral") from exc
            for original in (original_source, original_target):
                if original not in node_ids:
                    node_ids[original] = len(node_ids)
            if relation < np.iinfo(np.int16).min or relation > np.iinfo(np.int16).max:
                raise _fail("official relation ID exceeds int16")
            timestamps[index] = timestamp
            sources[index] = node_ids[original_source]
            destinations[index] = node_ids[original_target]
            relations[index] = relation
            index += 1
    if index != edge_count:
        raise _fail("official edge CSV row count changed while reading")

    node_types = np.full(len(node_ids), -1, dtype=np.int16)
    seen_types: set[int] = set()
    with type_path.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.reader(stream)
        header = next(reader, None)
        if header is None or len(header) != 2:
            raise _fail("official node-type CSV header must contain two columns")
        for line_number, row in enumerate(reader, start=2):
            if not row:
                continue
            if len(row) != 2:
                raise _fail(f"official node-type CSV row {line_number} is malformed")
            try:
                original, node_type = (int(value) for value in row)
            except ValueError as exc:
                raise _fail(
                    f"official node-type CSV row {line_number} is not integral"
                ) from exc
            local = node_ids.get(original)
            if local is None:
                raise _fail("official node-type CSV references a node absent from events")
            if local in seen_types:
                raise _fail("official node-type CSV contains a duplicate node")
            if node_type < np.iinfo(np.int16).min or node_type > np.iinfo(np.int16).max:
                raise _fail("official node type exceeds int16")
            seen_types.add(local)
            node_types[local] = node_type
    if len(seen_types) != len(node_ids) or bool(np.any(node_types < 0)):
        raise _fail("official node-type CSV does not cover every event node")

    validation_threshold, test_threshold = np.quantile(timestamps, (0.70, 0.85))
    train_mask = timestamps <= validation_threshold
    validation_mask = (timestamps > validation_threshold) & (
        timestamps <= test_threshold
    )
    test_mask = timestamps > test_threshold
    return _runtime_arrays(
        {
            "full_data": {
                "sources": sources,
                "destinations": destinations,
                "timestamps": timestamps,
                "edge_type": relations,
            },
            "node_type": node_types,
            "train_mask": train_mask,
            "val_mask": validation_mask,
            "test_mask": test_mask,
        }
    )


def _load_official_arrays_isolated(archive: Path, layout: RuntimeLayout) -> dict[str, np.ndarray]:
    layout.temporary.mkdir(parents=True, exist_ok=True)
    isolated_root = Path(
        tempfile.mkdtemp(prefix="thgl-software-", dir=layout.temporary)
    ).resolve()
    expected_parent = layout.temporary.resolve()
    if isolated_root.parent != expected_parent or not isolated_root.name.startswith("thgl-software-"):
        raise _fail("temporary TGB root is outside the runtime tmp directory")
    try:
        _extract_official_archive(archive, isolated_root)
        return _parse_official_csv(isolated_root / "thgl_software")
    finally:
        # This UUID-scoped path is verified immediately before recursive cleanup.
        resolved = isolated_root.resolve()
        if resolved.parent != expected_parent or not resolved.name.startswith("thgl-software-"):
            raise _fail("refusing to clean an unverified temporary TGB root")
        shutil.rmtree(resolved, ignore_errors=False)


def _runtime_arrays(dataset: Any) -> dict[str, np.ndarray]:
    """Extract numeric public API fields, never runtime pickle/cache paths."""

    value = dataset.dataset if hasattr(dataset, "dataset") else dataset
    full = value.get("full_data") if isinstance(value, Mapping) else getattr(value, "full_data", None)
    if not isinstance(full, Mapping):
        raise _fail("TGB adapter did not expose full_data")
    required = {"sources", "destinations", "timestamps", "edge_type"}
    if not required.issubset(full):
        raise _fail("TGB full_data lacks typed temporal edges")
    node_type = value.get("node_type") if isinstance(value, Mapping) else getattr(value, "node_type", None)
    if node_type is None:
        raise _fail("TGB adapter did not expose node_type")
    arrays = {
        "src": _as_numpy(full["sources"], name="sources", dtype=np.dtype(np.int64)),
        "dst": _as_numpy(full["destinations"], name="destinations", dtype=np.dtype(np.int64)),
        "timestamp": _as_numpy(full["timestamps"], name="timestamps", dtype=np.dtype(np.int64)),
        "relation": _as_numpy(full["edge_type"], name="edge_type", dtype=np.dtype(np.int16)),
        "node_type": _as_numpy(node_type, name="node_type", dtype=np.dtype(np.int16)),
    }
    for output_name, attribute in (
        ("train_mask", "train_mask"),
        ("validation_mask", "val_mask"),
        ("test_mask", "test_mask"),
    ):
        value_mask = value.get(attribute) if isinstance(value, Mapping) else getattr(value, attribute, None)
        if value_mask is None:
            value_mask = getattr(dataset, attribute, None)
        if value_mask is None:
            raise _fail(f"official TGB adapter did not expose {attribute}")
        arrays[output_name] = _as_numpy(
            value_mask, name=attribute, dtype=np.dtype(np.bool_)
        )
    edges = arrays["src"].shape[0]
    if any(arrays[name].shape != (edges,) for name in ("dst", "timestamp", "relation")):
        raise _fail("typed temporal edge arrays are misaligned")
    if edges and (arrays["src"].min() < 0 or arrays["dst"].min() < 0):
        raise _fail("TGB graph contains a negative node ID")
    nodes = arrays["node_type"].shape[0]
    if edges and (arrays["src"].max() >= nodes or arrays["dst"].max() >= nodes):
        raise _fail("TGB graph contains an out-of-bounds node ID")
    if len(np.unique(arrays["node_type"])) != EXPECTED_NODE_TYPES:
        raise _fail("TGB graph does not contain the official four node types")
    if len(np.unique(arrays["relation"])) != EXPECTED_RELATIONS:
        raise _fail("TGB graph does not contain the official fourteen relations")
    masks = [arrays[name] for name in ("train_mask", "validation_mask", "test_mask")]
    if any(mask.shape != (edges,) for mask in masks):
        raise _fail("official TGB split masks are misaligned")
    membership = sum(mask.astype(np.int8) for mask in masks)
    if not bool(np.all(membership == 1)):
        raise _fail("official TGB split masks must be disjoint and exhaustive")
    timestamps = arrays["timestamp"]
    train_time = timestamps[arrays["train_mask"]]
    validation_time = timestamps[arrays["validation_mask"]]
    test_time = timestamps[arrays["test_mask"]]
    if any(values.size == 0 for values in (train_time, validation_time, test_time)):
        raise _fail("official TGB split masks must each contain at least one event")
    if int(train_time.max()) >= int(validation_time.min()):
        raise _fail("official TGB train/validation masks violate temporal ordering")
    if int(validation_time.max()) >= int(test_time.min()):
        raise _fail("official TGB validation/test masks violate temporal ordering")
    return arrays


def prepare_thgl_software(
    source: str | Path,
    root: str | Path,
    *,
    dataset_factory: Callable[[Path], Any] | None = None,
    enforce_official_counts: bool = True,
) -> dict[str, Any]:
    """Use a delayed TGB 2.0 adapter, then cross a numeric-only trust boundary."""

    layout = RuntimeLayout.from_root(root)
    raw = layout.raw_thgl_software
    receipt = read_json_object(raw / "fetch-receipt.json")
    archive = Path(source).expanduser().resolve(strict=True)
    if archive != (raw / "thgl-software-2.0.0.zip").resolve() or file_sha256(archive) != receipt.get("archiveSha256"):
        raise _fail("source archive does not match the accepted official receipt")
    if (
        receipt.get("licenseId") != LICENSE_ID
        or receipt.get("tgbReferenceVersion") != TGB_VERSION
        or receipt.get("tgbReferenceCommit") != TGB_REFERENCE_COMMIT
        or receipt.get("datasetRelease") != DATASET_RELEASE
        or not isinstance(receipt.get("formalEligible"), bool)
    ):
        raise _fail("license or TGB release receipt is invalid")
    arrays = (
        _load_official_arrays_isolated(archive, layout)
        if dataset_factory is None
        else _runtime_arrays(dataset_factory(raw))
    )
    edge_count = int(arrays["src"].shape[0])
    node_count = int(arrays["node_type"].shape[0])
    if enforce_official_counts and (edge_count != EXPECTED_EDGE_COUNT or node_count != EXPECTED_NODE_COUNT):
        raise _fail("numeric arrays do not match official node/edge counts")
    order = np.lexsort((arrays["dst"], arrays["src"], arrays["timestamp"]))
    for name in (
        "src",
        "dst",
        "timestamp",
        "relation",
        "train_mask",
        "validation_mask",
        "test_mask",
    ):
        arrays[name] = np.ascontiguousarray(arrays[name][order])
    train_indices = np.flatnonzero(arrays["train_mask"]).astype(np.int64)
    validation_indices = np.flatnonzero(arrays["validation_mask"]).astype(np.int64)
    test_indices = np.flatnonzero(arrays["test_mask"]).astype(np.int64)
    expected_train = np.arange(train_indices.size, dtype=np.int64)
    expected_validation = np.arange(
        train_indices.size,
        train_indices.size + validation_indices.size,
        dtype=np.int64,
    )
    expected_test = np.arange(
        train_indices.size + validation_indices.size,
        edge_count,
        dtype=np.int64,
    )
    if not (
        np.array_equal(train_indices, expected_train)
        and np.array_equal(validation_indices, expected_validation)
        and np.array_equal(test_indices, expected_test)
    ):
        raise _fail("official TGB masks do not form contiguous temporal split blocks")
    train_end = int(arrays["timestamp"][arrays["train_mask"]].max())
    validation_start = int(arrays["timestamp"][arrays["validation_mask"]].min())
    validation_end = int(arrays["timestamp"][arrays["validation_mask"]].max())
    test_start = int(arrays["timestamp"][arrays["test_mask"]].min())
    split_indices = {
        "train": train_indices,
        "validation": validation_indices,
        "test": test_indices,
    }

    output = layout.processed_gfm / CORPUS_ID
    output.mkdir(parents=True, exist_ok=True)
    if (output / "manifest.json").exists():
        return check_thgl_software(root)
    writer = NumericShardWriter(output, prefix="events", rows_per_shard=max(edge_count, 1))
    shard = writer.write(
        {
            name: arrays[name]
            for name in (
                "src",
                "dst",
                "timestamp",
                "relation",
                "train_mask",
                "validation_mask",
                "test_mask",
            )
        }
    )
    node_writer = NumericShardWriter(output, prefix="nodes", rows_per_shard=max(node_count, 1))
    node_shard = node_writer.write({"node_type": arrays["node_type"]})
    role_masks = {
        "train": arrays["train_mask"],
        "validation": arrays["validation_mask"],
        "test": arrays["test_mask"],
        "shadow": np.zeros(edge_count, dtype=np.bool_),
    }
    access_event_shards = {}
    for role in ACCESS_ROLES:
        mask = role_masks[role]
        role_writer = NumericShardWriter(
            output,
            prefix=f"access-events-{role}",
            rows_per_shard=max(int(mask.sum()), 1),
        )
        access_event_shards[role] = role_writer.write(
            {
                name: arrays[name][mask]
                for name in ("src", "dst", "timestamp", "relation")
            }
        )
    split_writer = NumericShardWriter(
        output,
        prefix="split-indices",
        rows_per_shard=max(edge_count, 1),
    )
    split_shards = tuple(
        split_writer.write({f"{name}_edge_index": split_indices[name]})
        for name in ("train", "validation", "test")
    )
    manifest = build_manifest(
        schema_version="gfm.thgl-software-corpus/1.0",
        corpus_id=CORPUS_ID,
        license_id=LICENSE_ID,
        source={
            "uri": SOURCE_URL,
            "archiveSha256": receipt["archiveSha256"],
            "licenseEvidence": LICENSE_URL,
            "tgbReferenceVersion": TGB_VERSION,
            "tgbReferenceCommit": TGB_REFERENCE_COMMIT,
            "tgbReferenceSource": TGB_PREPROCESS_URL,
            "adapter": "pickle-free-direct-csv",
            "datasetRelease": DATASET_RELEASE,
            "archiveSize": receipt["archiveSize"],
            "sourceEtag": receipt["sourceEtag"],
            "formalEligible": receipt["formalEligible"],
        },
        shards=(
            shard,
            node_shard,
            *split_shards,
            *(access_event_shards[role] for role in ACCESS_ROLES),
        ),
        splits={
            "strategy": "official-temporal-70-15-15",
            "trainEndInclusive": train_end,
            "validationStartInclusive": validation_start,
            "validationEndInclusive": validation_end,
            "testStartInclusive": test_start,
            "indexArtifacts": {
                "train": split_shards[0].path,
                "validation": split_shards[1].path,
                "test": split_shards[2].path,
            },
            "counts": {
                "train": int(arrays["train_mask"].sum()),
                "validation": int(arrays["validation_mask"].sum()),
                "test": int(arrays["test_mask"].sum()),
            },
        },
        privacy={
            "containsText": False,
            "publicCheckpointEligible": bool(receipt["formalEligible"])
            and dataset_factory is None,
        },
        extra={
            "domainId": DOMAIN_ID,
            "physicalAccess": {
                "schemaVersion": PHYSICAL_ACCESS_SCHEMA,
                "roles": list(ACCESS_ROLES),
                "roleFamilies": {
                    "events": {
                        role: [access_event_shards[role].path]
                        for role in ACCESS_ROLES
                    }
                },
                "sharedFamilies": {"nodes": [node_shard.path]},
                "mergeOrder": {"events": "timestamp"},
            },
            "nodeCount": node_count,
            "edgeCount": edge_count,
            "nodeTypeCount": EXPECTED_NODE_TYPES,
            "relationCount": EXPECTED_RELATIONS,
            "negativeSampling": "node-type-filtered",
            "evaluationMetric": "mrr",
        },
    )
    atomic_write_json(output / "manifest.json", manifest)
    return check_thgl_software(root)


def check_thgl_software(root: str | Path) -> dict[str, Any]:
    output = RuntimeLayout.from_root(root).processed_gfm / CORPUS_ID
    manifest = read_json_object(output / "manifest.json")
    if manifest.get("schemaVersion") != "gfm.thgl-software-corpus/1.0":
        raise _fail("processed manifest schema is unsupported")
    source = manifest.get("source")
    privacy = manifest.get("privacy")
    if (
        not isinstance(source, dict)
        or not isinstance(source.get("formalEligible"), bool)
        or not isinstance(privacy, dict)
        or privacy.get("publicCheckpointEligible")
        is not source.get("formalEligible")
    ):
        raise _fail("processed formal-eligibility evidence is inconsistent")
    verify_manifest(output, manifest)
    split_evidence = _load_thgl_split_contract(output, manifest)
    _check_thgl_physical_access(output, manifest, split_evidence)
    return manifest


def _check_thgl_physical_access(
    output: Path, manifest: Mapping[str, Any], split_evidence: Mapping[str, Any]
) -> None:
    access = manifest.get("physicalAccess")
    roles = access.get("roleFamilies", {}).get("events") if isinstance(access, dict) else None
    if (
        not isinstance(access, dict)
        or access.get("schemaVersion") != PHYSICAL_ACCESS_SCHEMA
        or access.get("roles") != list(ACCESS_ROLES)
        or access.get("mergeOrder") != {"events": "timestamp"}
        or not isinstance(roles, dict)
        or set(roles) != set(ACCESS_ROLES)
    ):
        raise _fail("TGB physical role-view contract is invalid")
    node_paths = access.get("sharedFamilies", {}).get("nodes")
    if not isinstance(node_paths, list) or len(node_paths) != 1:
        raise _fail("TGB physical shared node view is invalid")
    records = {
        str(item["path"]): item
        for item in manifest["shards"]
        if isinstance(item, dict) and isinstance(item.get("path"), str)
    }
    canonical_record = next(
        item
        for item in manifest["shards"]
        if isinstance(item, dict) and item.get("path") == "events-00000.npz"
    )
    canonical = load_npz_safe(
        resolve_within(output, str(canonical_record["path"])),
        expected={
            str(item["name"]): (str(item["dtype"]), len(item["shape"]))
            for item in canonical_record["arrays"]
        },
    )
    for role, mask_name in (
        ("train", "train_mask"),
        ("validation", "validation_mask"),
        ("test", "test_mask"),
        ("shadow", None),
    ):
        paths = roles[role]
        if paths != [f"access-events-{role}-00000.npz"]:
            raise _fail("TGB physical role shard path is invalid")
        record = records.get(paths[0])
        if record is None:
            raise _fail("TGB physical role shard is undeclared")
        expected_schema = {
            "src": (np.dtype(np.int64).str, 1),
            "dst": (np.dtype(np.int64).str, 1),
            "timestamp": (np.dtype(np.int64).str, 1),
            "relation": (np.dtype(np.int16).str, 1),
        }
        actual = load_npz_safe(resolve_within(output, paths[0]), expected=expected_schema)
        mask = (
            np.zeros(canonical["timestamp"].shape, dtype=np.bool_)
            if mask_name is None
            else canonical[mask_name]
        )
        if any(
            not np.array_equal(actual[name], canonical[name][mask])
            for name in expected_schema
        ):
            raise _fail(f"TGB physical role shard {role} differs from official mask")
        expected_rows = 0 if role == "shadow" else int(split_evidence["counts"][role])
        if int(record["rows"]) != expected_rows:
            raise _fail(f"TGB physical role shard {role} count differs")


def _load_thgl_split_contract(output: Path, manifest: Mapping[str, Any]) -> dict[str, Any]:
    split_contract = manifest.get("splits")
    if not isinstance(split_contract, dict):
        raise _fail("processed manifest has no split contract")
    paths = split_contract.get("indexArtifacts")
    if not isinstance(paths, dict) or set(paths) != {"train", "validation", "test"}:
        raise _fail("processed manifest has no exact split index artifacts")
    from .common import load_npz_safe, resolve_within

    indices = {
        name: load_npz_safe(
            resolve_within(output, str(paths[name])),
            expected={f"{name}_edge_index": (np.dtype(np.int64).str, 1)},
        )[f"{name}_edge_index"]
        for name in ("train", "validation", "test")
    }
    counts = split_contract.get("counts")
    if not isinstance(counts, dict) or any(
        indices[name].shape != (int(counts.get(name, -1)),)
        for name in ("train", "validation", "test")
    ):
        raise _fail("exact split indices do not match manifest counts")
    if any(
        np.any(indices[name][1:] <= indices[name][:-1])
        for name in ("train", "validation", "test")
    ):
        raise _fail("exact split indices must be strictly increasing")
    event_records = [
        item
        for item in manifest.get("shards", [])
        if isinstance(item, dict)
        and {"timestamp", "train_mask", "validation_mask", "test_mask"}.issubset(
            {
                array.get("name")
                for array in item.get("arrays", [])
                if isinstance(array, dict)
            }
        )
    ]
    if len(event_records) != 1:
        raise _fail("processed manifest must declare exactly one split-bearing event shard")
    event_record = event_records[0]
    event_arrays = event_record["arrays"]
    events = load_npz_safe(
        resolve_within(output, str(event_record["path"])),
        expected={
            str(item["name"]): (str(item["dtype"]), len(item["shape"]))
            for item in event_arrays
            if isinstance(item, dict)
        },
    )
    timestamps = events["timestamp"]
    for name, mask_name in (
        ("train", "train_mask"),
        ("validation", "validation_mask"),
        ("test", "test_mask"),
    ):
        actual = np.flatnonzero(events[mask_name]).astype(np.int64)
        if not np.array_equal(indices[name], actual):
            raise _fail(f"exact {name} indices do not match the official mask")
    actual_bounds = {
        "trainEndInclusive": int(timestamps[events["train_mask"]].max()),
        "validationStartInclusive": int(timestamps[events["validation_mask"]].min()),
        "validationEndInclusive": int(timestamps[events["validation_mask"]].max()),
        "testStartInclusive": int(timestamps[events["test_mask"]].min()),
    }
    if any(int(split_contract.get(name, -1)) != value for name, value in actual_bounds.items()):
        raise _fail("exact split timestamp bounds do not match the official masks")
    if not (
        actual_bounds["trainEndInclusive"] < actual_bounds["validationStartInclusive"]
        and actual_bounds["validationEndInclusive"] < actual_bounds["testStartInclusive"]
    ):
        raise _fail("exact split timestamp bounds overlap")
    return {
        "indices": indices,
        "bounds": actual_bounds,
        "counts": {name: int(counts[name]) for name in ("train", "validation", "test")},
    }


def load_thgl_software_splits(root: str | Path) -> dict[str, Any]:
    """Load official edge indices and exact timestamp bounds for workflow use."""

    output = RuntimeLayout.from_root(root).processed_gfm / CORPUS_ID
    manifest = check_thgl_software(root)
    return _load_thgl_split_contract(output, manifest)
