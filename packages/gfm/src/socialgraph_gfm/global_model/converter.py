"""Explicitly trusted, offline conversion of official Global pickle artifacts.

The training and serving paths never import this module. Pickle conversion remains
an offline operation: callers must acknowledge the trusted source, globals are
statically inventoried, and a restricted unpickler is used before numeric arrays
are published. Resource exhaustion is still possible, so production conversion
should run in a resource-limited worker.
"""

from __future__ import annotations

import gc
import hashlib
import importlib
import json
import math
import os
import pickle
import pickletools
import shutil
import subprocess
import sys
import tempfile
from collections import deque
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from numbers import Real
from pathlib import Path
from typing import Any, cast

import numpy as np

from socialgraph_gfm.canonical import canonical_sha256, file_sha256
from socialgraph_gfm.errors import ContractViolation
from socialgraph_gfm.gfm.corpus.common import atomic_write_json

from .contracts import (
    COUNTRY_IDS,
    GRAPH_STAT_NAMES,
    TRACE_ARRAY_TOKENS,
    TRACE_NAMES,
    CountryId,
    GlobalArrayDescriptor,
    GlobalCorpusEntry,
    GlobalCorpusManifest,
    GlobalCountryManifest,
    GlobalSplitDescriptor,
    atomic_write_contract,
)

OFFICIAL_REGIMES = ("full", "0.5", "0.75", "0.9", "0.95", "0.99", "0.999")
OFFICIAL_SOURCE_KEYS = frozenset({"graph", *TRACE_NAMES, "labels", "splits"})
MAX_PICKLE_BYTES = 2 * 1024 * 1024 * 1024
MAX_TEXT_TENSOR_BYTES = 16 * 1024 * 1024 * 1024
MINIMUM_CONVERSION_FREE_BYTES = 20 * 1024**3
WORKER_CONTRACT_SCHEMA = "socialgraph-fm.global-model-conversion-worker/1.0"
WORKER_RECEIPT_SCHEMA = "socialgraph-fm.global-model-conversion-receipt/1.0"
ALLOWED_PICKLE_GLOBALS = frozenset(
    {
        ("networkx.classes.graph", "Graph"),
        ("networkx.classes.reportviews", "DegreeView"),
        ("numpy.core.multiarray", "_reconstruct"),
        ("numpy._core.multiarray", "_reconstruct"),
        ("numpy", "ndarray"),
        ("numpy", "dtype"),
    }
)
_ALLOWED_MEMO_SYMBOLS = frozenset(
    symbol for global_name in ALLOWED_PICKLE_GLOBALS for symbol in global_name
)
_STRING_OPS = frozenset({"UNICODE", "BINUNICODE", "SHORT_BINUNICODE", "BINUNICODE8"})
_GET_OPS = frozenset({"GET", "BINGET", "LONG_BINGET"})
_PUT_OPS = frozenset({"PUT", "BINPUT", "LONG_BINPUT"})
_FORBIDDEN_OPS = frozenset(
    {"PERSID", "BINPERSID", "EXT1", "EXT2", "EXT4", "NEXT_BUFFER", "READONLY_BUFFER"}
)


def _fail(message: str) -> ContractViolation:
    return ContractViolation(f"SocialGraph-FM Global converter: {message}")


@dataclass(frozen=True)
class PickleInspection:
    path: Path
    byte_length: int
    sha256: str
    protocol: int
    globals: tuple[tuple[str, str], ...]
    opcode_count: int


@dataclass(frozen=True)
class DiskSpaceInspection:
    volume_path: Path
    free_bytes: int
    required_bytes: int


@dataclass(frozen=True)
class ValidatedWorkerRequest:
    country_id: CountryId
    source_root: Path
    destination_root: Path
    destination: Path
    pickle_sources: Mapping[str, Path]
    text_tensor_path: Path
    contract_hash: str


@dataclass(frozen=True)
class WorkerConversionReceipt:
    country_id: CountryId
    contract_hash: str
    manifest_path: Path
    manifest_hash: str
    receipt_hash: str


def require_conversion_disk_space(
    destination: Path,
    *,
    minimum_free_bytes: int = MINIMUM_CONVERSION_FREE_BYTES,
) -> DiskSpaceInspection:
    """Fail before publication unless the destination volume has 20 GiB free."""

    if minimum_free_bytes < 1:
        raise ValueError("minimum_free_bytes must be positive")
    probe = destination.expanduser().resolve()
    while not probe.exists() and probe != probe.parent:
        probe = probe.parent
    if not probe.exists() or not probe.is_dir():
        raise _fail("cannot identify an existing destination volume")
    usage = shutil.disk_usage(probe)
    inspection = DiskSpaceInspection(
        volume_path=probe,
        free_bytes=int(usage.free),
        required_bytes=minimum_free_bytes,
    )
    if inspection.free_bytes < inspection.required_bytes:
        raise _fail(
            "destination volume has insufficient free space: "
            f"required={inspection.required_bytes}, observed={inspection.free_bytes}"
        )
    return inspection


def _resolved_absolute_path(value: object, *, label: str, must_exist: bool) -> Path:
    if not isinstance(value, str) or not value or "\x00" in value:
        raise _fail(f"worker {label} must be a nonempty absolute path")
    path = Path(value)
    if not path.is_absolute():
        raise _fail(f"worker {label} must be an absolute path")
    try:
        resolved = path.resolve(strict=must_exist)
    except OSError as exc:
        raise _fail(f"worker {label} cannot be resolved") from exc
    if path != resolved:
        raise _fail(f"worker {label} must already use its resolved absolute spelling")
    return resolved


def _require_within(path: Path, root: Path, *, label: str) -> None:
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise _fail(f"worker {label} escapes its declared root") from exc


def _source_contract_record(
    path: Path,
    *,
    root: Path,
    label: str,
    maximum_bytes: int,
) -> dict[str, object]:
    source = path.expanduser()
    if source.is_symlink():
        raise _fail(f"worker {label} must not be a symbolic link")
    try:
        resolved = source.resolve(strict=True)
    except OSError as exc:
        raise _fail(f"worker {label} is absent") from exc
    _require_within(resolved, root, label=label)
    if not resolved.is_file():
        raise _fail(f"worker {label} is not a regular file")
    byte_length = resolved.stat().st_size
    if byte_length <= 0 or byte_length > maximum_bytes:
        raise _fail(f"worker {label} is empty or exceeds its byte limit")
    return {
        "path": str(resolved),
        "sha256": file_sha256(resolved),
        "byteLength": byte_length,
    }


def build_worker_contract(
    *,
    country_id: CountryId,
    source_root: Path,
    destination_root: Path,
    pickle_sources: Mapping[str, Path],
    text_tensor_path: Path,
    destination: Path,
) -> dict[str, object]:
    """Bind one worker invocation to resolved paths, byte lengths, and source hashes."""

    if country_id not in COUNTRY_IDS:
        raise ValueError(f"unsupported Global country {country_id!r}")
    regimes = tuple(pickle_sources)
    if (
        "full" not in pickle_sources
        or any(regime not in OFFICIAL_REGIMES for regime in regimes)
        or len(regimes) != len(set(regimes))
    ):
        raise ValueError("worker pickle sources must be a unique official regime set with full")
    source_base = source_root.expanduser()
    destination_base = destination_root.expanduser()
    if source_base.is_symlink() or destination_base.is_symlink():
        raise _fail("worker roots must not be symbolic links")
    source_base = source_base.resolve(strict=True)
    destination_base = destination_base.resolve(strict=True)
    if not source_base.is_dir() or not destination_base.is_dir():
        raise _fail("worker roots must be existing directories")
    raw_source_country = source_base / country_id
    if raw_source_country.is_symlink():
        raise _fail("worker country source must not be a symbolic link")
    source_country = raw_source_country.resolve(strict=True)
    _require_within(source_country, source_base, label="country source directory")
    if not source_country.is_dir() or source_country.is_symlink():
        raise _fail("worker country source must be a regular directory")
    output = destination.expanduser().resolve(strict=False)
    expected_output = (destination_base / "countries" / country_id).resolve(strict=False)
    if output != expected_output:
        raise _fail("worker destination must be destinationRoot/countries/countryId")
    if output.exists():
        raise _fail("immutable conversion destination already exists")
    ordered_regimes = tuple(regime for regime in OFFICIAL_REGIMES if regime in pickle_sources)
    pickle_records = {
        regime: _source_contract_record(
            pickle_sources[regime],
            root=source_country,
            label=f"pickle source {regime}",
            maximum_bytes=MAX_PICKLE_BYTES,
        )
        for regime in ordered_regimes
    }
    text_record = _source_contract_record(
        text_tensor_path,
        root=source_country,
        label="text tensor",
        maximum_bytes=MAX_TEXT_TENSOR_BYTES,
    )
    payload: dict[str, object] = {
        "schemaVersion": WORKER_CONTRACT_SCHEMA,
        "countryId": country_id,
        "sourceRoot": str(source_base),
        "destinationRoot": str(destination_base),
        "destination": str(output),
        "pickleSources": pickle_records,
        "textTensor": text_record,
    }
    payload["contractHash"] = canonical_sha256(payload)
    return payload


def _validated_source_record(
    value: object,
    *,
    root: Path,
    label: str,
    maximum_bytes: int,
) -> Path:
    if not isinstance(value, dict) or set(value) != {"path", "sha256", "byteLength"}:
        raise _fail(f"worker {label} has an invalid source record")
    path = _resolved_absolute_path(value["path"], label=label, must_exist=True)
    _require_within(path, root, label=label)
    if path.is_symlink() or not path.is_file():
        raise _fail(f"worker {label} must be a regular non-symlink file")
    byte_length = value["byteLength"]
    digest = value["sha256"]
    if (
        isinstance(byte_length, bool)
        or not isinstance(byte_length, int)
        or byte_length <= 0
        or byte_length > maximum_bytes
        or path.stat().st_size != byte_length
    ):
        raise _fail(f"worker {label} byte length is invalid")
    if not isinstance(digest, str) or len(digest) != 64 or file_sha256(path) != digest:
        raise _fail(f"worker {label} SHA-256 does not match")
    return path


def validate_worker_contract(value: Mapping[str, object]) -> ValidatedWorkerRequest:
    """Validate the exact parent/worker JSON boundary and reverify all source bytes."""

    expected_keys = {
        "schemaVersion",
        "countryId",
        "sourceRoot",
        "destinationRoot",
        "destination",
        "pickleSources",
        "textTensor",
        "contractHash",
    }
    if set(value) != expected_keys:
        raise _fail("worker contract has unknown or missing keys")
    if value["schemaVersion"] != WORKER_CONTRACT_SCHEMA:
        raise _fail("worker contract schema is unsupported")
    country_id = value["countryId"]
    if not isinstance(country_id, str) or country_id not in COUNTRY_IDS:
        raise _fail("worker contract has an unknown countryId")
    source_root = _resolved_absolute_path(
        value["sourceRoot"], label="sourceRoot", must_exist=True
    )
    destination_root = _resolved_absolute_path(
        value["destinationRoot"], label="destinationRoot", must_exist=True
    )
    if (
        source_root.is_symlink()
        or destination_root.is_symlink()
        or not source_root.is_dir()
        or not destination_root.is_dir()
    ):
        raise _fail("worker roots must be regular non-symlink directories")
    raw_source_country = source_root / country_id
    if raw_source_country.is_symlink():
        raise _fail("worker country source must not be a symbolic link")
    source_country = raw_source_country.resolve(strict=True)
    _require_within(source_country, source_root, label="country source directory")
    destination = _resolved_absolute_path(
        value["destination"], label="destination", must_exist=False
    )
    if destination != (destination_root / "countries" / country_id).resolve(strict=False):
        raise _fail("worker destination must be destinationRoot/countries/countryId")
    if destination.exists():
        raise _fail("immutable conversion destination already exists")
    pickle_values = value["pickleSources"]
    if not isinstance(pickle_values, dict):
        raise _fail("worker pickleSources must be an object")
    if "full" not in pickle_values or any(
        regime not in OFFICIAL_REGIMES for regime in pickle_values
    ):
        raise _fail("worker pickleSources contains an unknown regime or omits full")
    ordered_regimes = tuple(regime for regime in OFFICIAL_REGIMES if regime in pickle_values)
    pickle_sources = {
        regime: _validated_source_record(
            pickle_values[regime],
            root=source_country,
            label=f"pickle source {regime}",
            maximum_bytes=MAX_PICKLE_BYTES,
        )
        for regime in ordered_regimes
    }
    text_tensor_path = _validated_source_record(
        value["textTensor"],
        root=source_country,
        label="text tensor",
        maximum_bytes=MAX_TEXT_TENSOR_BYTES,
    )
    contract_hash = value["contractHash"]
    logical = {key: item for key, item in value.items() if key != "contractHash"}
    if not isinstance(contract_hash, str) or contract_hash != canonical_sha256(logical):
        raise _fail("worker contractHash does not match its payload")
    return ValidatedWorkerRequest(
        country_id=cast(CountryId, country_id),
        source_root=source_root,
        destination_root=destination_root,
        destination=destination,
        pickle_sources=pickle_sources,
        text_tensor_path=text_tensor_path,
        contract_hash=contract_hash,
    )


def _parse_worker_receipt(
    stdout: str,
    *,
    request: Mapping[str, object],
) -> WorkerConversionReceipt:
    try:
        value = json.loads(stdout)
    except (json.JSONDecodeError, TypeError) as exc:
        raise _fail("conversion worker did not return one JSON receipt") from exc
    expected_keys = {
        "schemaVersion",
        "countryId",
        "contractHash",
        "manifestPath",
        "manifestHash",
        "receiptHash",
    }
    if not isinstance(value, dict) or set(value) != expected_keys:
        raise _fail("conversion worker receipt has unknown or missing keys")
    logical = {key: item for key, item in value.items() if key != "receiptHash"}
    if (
        value["schemaVersion"] != WORKER_RECEIPT_SCHEMA
        or value["countryId"] != request["countryId"]
        or value["contractHash"] != request["contractHash"]
        or value["receiptHash"] != canonical_sha256(logical)
    ):
        raise _fail("conversion worker receipt identity is invalid")
    manifest_path = _resolved_absolute_path(
        value["manifestPath"], label="receipt manifestPath", must_exist=True
    )
    expected_manifest = Path(cast(str, request["destination"])) / "manifest.json"
    if manifest_path != expected_manifest or manifest_path.is_symlink():
        raise _fail("conversion worker receipt points at an unexpected manifest")
    from .contracts import read_country_manifest

    manifest = read_country_manifest(manifest_path)
    manifest_hash = value["manifestHash"]
    if (
        not isinstance(manifest_hash, str)
        or manifest.country_id != request["countryId"]
        or manifest.content_hash != manifest_hash
    ):
        raise _fail("conversion worker manifest does not match its receipt")
    return WorkerConversionReceipt(
        country_id=manifest.country_id,
        contract_hash=cast(str, value["contractHash"]),
        manifest_path=manifest_path,
        manifest_hash=manifest_hash,
        receipt_hash=cast(str, value["receiptHash"]),
    )


def convert_country_in_worker(
    *,
    country_id: CountryId,
    source_root: Path,
    destination_root: Path,
    pickle_sources: Mapping[str, Path],
    text_tensor_path: Path,
    destination: Path,
    trusted_source: bool = False,
    timeout_seconds: float = 7200.0,
) -> WorkerConversionReceipt:
    """Convert one country in a fail-closed subprocess with no stdin."""

    if not trusted_source:
        raise _fail("pickle conversion requires trusted_source=True")
    if not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be a positive finite number")
    output = destination.expanduser().resolve(strict=False)
    require_conversion_disk_space(output)
    destination_base = destination_root.expanduser().resolve(strict=False)
    destination_base.mkdir(parents=True, exist_ok=True)
    contract = build_worker_contract(
        country_id=country_id,
        source_root=source_root,
        destination_root=destination_base,
        pickle_sources=pickle_sources,
        text_tensor_path=text_tensor_path,
        destination=output,
    )
    with tempfile.TemporaryDirectory(prefix="global-model-worker-contract-") as raw_temporary:
        contract_path = Path(raw_temporary) / "contract.json"
        atomic_write_json(contract_path, contract)
        command = (
            sys.executable,
            "-m",
            "socialgraph_gfm.global_model.converter_worker",
            "--contract",
            str(contract_path),
        )
        creation_flags = getattr(subprocess, "CREATE_NO_WINDOW", 0) if os.name == "nt" else 0
        try:
            completed = subprocess.run(
                command,
                stdin=subprocess.DEVNULL,
                capture_output=True,
                text=True,
                timeout=timeout_seconds,
                check=False,
                creationflags=creation_flags,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise _fail(f"conversion worker could not complete: {exc}") from exc
    if completed.returncode != 0:
        detail = completed.stderr.strip()[-2000:] or "no diagnostic output"
        raise _fail(f"conversion worker failed with code {completed.returncode}: {detail}")
    return _parse_worker_receipt(completed.stdout, request=contract)


def _memo_reference(
    operation: tuple[pickletools.OpcodeInfo, Any, int],
    memo_strings: Mapping[int, str],
) -> str | None:
    opcode, argument, _ = operation
    if opcode.name in _STRING_OPS:
        return str(argument)
    if opcode.name in _GET_OPS:
        try:
            return memo_strings[int(argument)]
        except (KeyError, TypeError, ValueError):
            return None
    return None


def _previous_symbol(
    history: Sequence[tuple[pickletools.OpcodeInfo, Any, int]],
    memo_strings: Mapping[int, str],
    *,
    before: int | None = None,
) -> tuple[str, int] | None:
    index = len(history) - 1 if before is None else before - 1
    while index >= 0 and history[index][0].name == "MEMOIZE":
        index -= 1
    if index < 0:
        return None
    value = _memo_reference(history[index], memo_strings)
    return (value, index) if value is not None else None


def inspect_global_model_pickle(
    path: Path,
    *,
    maximum_bytes: int = MAX_PICKLE_BYTES,
) -> PickleInspection:
    """Inventory executable pickle globals without instantiating the payload."""

    source = path.expanduser().resolve()
    if (
        not source.is_file()
        or source.is_symlink()
        or source.stat().st_size <= 0
        or source.stat().st_size > maximum_bytes
    ):
        raise _fail("pickle source is absent, unsafe, empty, or exceeds its byte limit")
    protocol: int | None = None
    globals_seen: set[tuple[str, str]] = set()
    history: deque[tuple[pickletools.OpcodeInfo, Any, int]] = deque(maxlen=8)
    memo_strings: dict[int, str] = {}
    next_memo = 0
    opcode_count = 0
    try:
        with source.open("rb") as stream:
            for opcode, argument, position in pickletools.genops(stream):
                if position is None:
                    raise _fail("pickle opcode has no byte position")
                opcode_count += 1
                name = opcode.name
                if name == "PROTO":
                    if not isinstance(argument, int):
                        raise _fail(f"PROTO has a non-integer argument at byte {position}")
                    protocol = argument
                elif name in _FORBIDDEN_OPS:
                    raise _fail(f"pickle uses forbidden opcode {name} at byte {position}")
                elif name == "GLOBAL":
                    parts = str(argument).split(" ", 1)
                    if len(parts) != 2:
                        raise _fail(f"malformed GLOBAL opcode at byte {position}")
                    globals_seen.add((parts[0], parts[1]))
                elif name == "STACK_GLOBAL":
                    items = list(history)
                    symbol = _previous_symbol(items, memo_strings)
                    module = (
                        _previous_symbol(items, memo_strings, before=symbol[1])
                        if symbol is not None
                        else None
                    )
                    if symbol is None or module is None:
                        raise _fail(f"cannot statically resolve STACK_GLOBAL at byte {position}")
                    globals_seen.add((module[0], symbol[0]))

                if name == "MEMOIZE":
                    prior = _previous_symbol(list(history), memo_strings)
                    if prior is not None and prior[0] in _ALLOWED_MEMO_SYMBOLS:
                        memo_strings[next_memo] = prior[0]
                    next_memo += 1
                elif name in _PUT_OPS:
                    if not isinstance(argument, (int, str)):
                        raise _fail(f"{name} has an invalid memo index at byte {position}")
                    memo_index = int(argument)
                    prior = _previous_symbol(list(history), memo_strings)
                    if prior is not None and prior[0] in _ALLOWED_MEMO_SYMBOLS:
                        memo_strings[memo_index] = prior[0]
                    next_memo = max(next_memo, memo_index + 1)
                history.append((opcode, cast(Any, argument), position))
    except (OSError, pickle.UnpicklingError, ValueError) as exc:
        if isinstance(exc, ContractViolation):
            raise
        raise _fail(f"pickle opcode stream is invalid: {exc}") from exc
    if protocol != 4:
        raise _fail(f"official SocialGraph-FM Global conversion requires pickle protocol 4, observed {protocol}")
    forbidden_globals = globals_seen - ALLOWED_PICKLE_GLOBALS
    if forbidden_globals:
        formatted = ", ".join(f"{module}.{name}" for module, name in sorted(forbidden_globals))
        raise _fail(f"pickle references globals outside the exact allowlist: {formatted}")
    return PickleInspection(
        path=source,
        byte_length=source.stat().st_size,
        sha256=file_sha256(source),
        protocol=protocol,
        globals=tuple(sorted(globals_seen)),
        opcode_count=opcode_count,
    )


class _RestrictedGlobalUnpickler(pickle.Unpickler):
    def find_class(self, module: str, name: str) -> Any:
        if (module, name) not in ALLOWED_PICKLE_GLOBALS:
            raise pickle.UnpicklingError(f"forbidden global {module}.{name}")
        return getattr(importlib.import_module(module), name)

    def persistent_load(self, pid: object) -> Any:
        raise pickle.UnpicklingError(f"persistent pickle IDs are forbidden: {pid!r}")


def load_trusted_global_model_pickle(
    path: Path,
    *,
    trusted_source: bool = False,
) -> dict[str, Any]:
    """Load one inspected official source, requiring an explicit trust acknowledgement."""

    if not trusted_source:
        raise _fail("pickle loading requires trusted_source=True in an offline worker")
    inspection = inspect_global_model_pickle(path)
    try:
        with inspection.path.open("rb") as stream:
            value = _RestrictedGlobalUnpickler(stream).load()
    except (OSError, ImportError, AttributeError, pickle.UnpicklingError) as exc:
        raise _fail(f"restricted pickle load failed: {exc}") from exc
    if not isinstance(value, dict) or set(value) != OFFICIAL_SOURCE_KEYS:
        raise _fail("pickle payload does not have the exact official Global key inventory")
    return cast(dict[str, Any], value)


def load_text_tensor(path: Path) -> np.ndarray:
    """Load the official tensor with PyTorch's weights-only boundary and return float32 NumPy."""

    source = path.expanduser().resolve()
    if not source.is_file() or source.is_symlink() or source.stat().st_size <= 0:
        raise _fail("text tensor source is absent, unsafe, or empty")
    try:
        import torch

        tensor = torch.load(source, map_location="cpu", weights_only=True)
    except (ImportError, OSError, RuntimeError, ValueError) as exc:
        raise _fail(f"weights-only text tensor load failed: {exc}") from exc
    if not isinstance(tensor, torch.Tensor):
        raise _fail("text tensor source must contain exactly one Tensor")
    if tensor.dtype != torch.float32 or tensor.ndim != 2 or tensor.shape[1] != 768:
        raise _fail("text tensor must have shape [N,768] and dtype float32")
    if not bool(torch.isfinite(tensor).all()):
        raise _fail("text tensor contains NaN or Infinity")
    return tensor.detach().contiguous().numpy()


def _validate_graph_nodes(graph: Any) -> int:
    try:
        nodes = list(graph.nodes())
        directed = bool(graph.is_directed())
        multigraph = bool(graph.is_multigraph())
    except (AttributeError, TypeError) as exc:
        raise _fail("graph value is not a NetworkX-style graph") from exc
    if directed or multigraph:
        raise _fail("official Global graph must be an undirected simple graph")
    if any(isinstance(node, bool) or not isinstance(node, int) for node in nodes):
        raise _fail("graph node IDs must be integers")
    if sorted(nodes) != list(range(len(nodes))):
        raise _fail("graph node IDs must be contiguous from zero")
    return len(nodes)


def graph_to_arrays(
    graph: Any,
    *,
    structural_buckets: int = 128,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Remove self-loops, preserve real edges, and encode degree without inventing topology."""

    if structural_buckets != 128:
        raise ValueError("SocialGraph-FM Global requires 128 degree buckets")
    node_count = _validate_graph_nodes(graph)
    edge_count = int(graph.number_of_edges())
    self_loop_count = 0
    degrees = np.zeros(node_count, dtype=np.int64)
    for raw_source, raw_target in graph.edges():
        source, target = int(raw_source), int(raw_target)
        if source == target:
            self_loop_count += 1
            continue
        degrees[source] += 1
        degrees[target] += 1
    structure_missing = np.ascontiguousarray(degrees == 0, dtype=np.bool_)
    undirected_count = edge_count - self_loop_count
    edge_index = np.empty((2, undirected_count * 2), dtype=np.int64)
    cursor = 0
    for raw_source, raw_target in graph.edges():
        source, target = int(raw_source), int(raw_target)
        if source == target:
            continue
        edge_index[:, cursor] = (source, target)
        edge_index[:, cursor + 1] = (target, source)
        cursor += 2
    if cursor != edge_index.shape[1]:
        raise _fail("graph edge inventory changed during conversion")
    if edge_index.shape[1]:
        ordering = np.lexsort((edge_index[1], edge_index[0]))
        edge_index = np.ascontiguousarray(edge_index[:, ordering])
    percentiles = np.percentile(degrees, np.linspace(0, 100, structural_buckets))
    degree_bucket = np.searchsorted(percentiles, degrees, side="right") - 1
    degree_bucket = np.clip(degree_bucket, 0, structural_buckets - 1).astype(np.uint8)
    return edge_index, degree_bucket, structure_missing


def _trace_membership(payload: Mapping[str, Any], node_count: int) -> np.ndarray:
    membership = np.zeros((node_count, len(TRACE_NAMES)), dtype=np.bool_)
    for column, trace_name in enumerate(TRACE_NAMES):
        trace = payload[trace_name]
        try:
            if bool(trace.is_directed()) or bool(trace.is_multigraph()):
                raise _fail(f"trace graph {trace_name!r} must be undirected and simple")
            nodes = np.asarray(list(trace.nodes()), dtype=np.int64)
        except (AttributeError, TypeError, ValueError) as exc:
            raise _fail(f"trace graph {trace_name!r} is invalid") from exc
        if nodes.size and (int(nodes.min()) < 0 or int(nodes.max()) >= node_count):
            raise _fail(f"trace graph {trace_name!r} contains an out-of-range node")
        membership[nodes, column] = True
    return membership


def _edge_index_to_csr(edge_index: np.ndarray, node_count: int) -> tuple[np.ndarray, np.ndarray]:
    counts = np.bincount(edge_index[0], minlength=node_count)
    indptr = np.empty(node_count + 1, dtype=np.int64)
    indptr[0] = 0
    np.cumsum(counts, out=indptr[1:])
    return indptr, np.ascontiguousarray(edge_index[1])


def relation_graph_to_csr(
    graph: Any,
    *,
    node_count: int,
    trace_name: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    try:
        if bool(graph.is_directed()) or bool(graph.is_multigraph()):
            raise _fail(f"trace graph {trace_name!r} must be undirected and simple")
        nodes = list(graph.nodes())
        edge_count = int(graph.number_of_edges())
    except (AttributeError, TypeError, ValueError) as exc:
        raise _fail(f"trace graph {trace_name!r} is invalid") from exc
    if any(
        isinstance(node, bool)
        or not isinstance(node, int)
        or node < 0
        or node >= node_count
        for node in nodes
    ):
        raise _fail(f"trace graph {trace_name!r} contains an invalid node ID")
    self_loop_count = sum(1 for source, target in graph.edges() if source == target)
    directed_count = (edge_count - self_loop_count) * 2
    sources = np.empty(directed_count, dtype=np.int64)
    indices = np.empty(directed_count, dtype=np.int64)
    weights = np.empty(directed_count, dtype=np.float64)
    cursor = 0
    for raw_source, raw_target, attributes in graph.edges(data=True):
        source, target = int(raw_source), int(raw_target)
        if source == target:
            continue
        weight = attributes.get("weight")
        if isinstance(weight, bool) or not isinstance(weight, Real):
            raise _fail(f"trace graph {trace_name!r} edge is missing a finite numeric weight")
        numeric_weight = float(weight)
        if not math.isfinite(numeric_weight):
            raise _fail(f"trace graph {trace_name!r} edge is missing a finite numeric weight")
        sources[cursor : cursor + 2] = (source, target)
        indices[cursor : cursor + 2] = (target, source)
        weights[cursor : cursor + 2] = numeric_weight
        cursor += 2
    if cursor != directed_count:
        raise _fail(f"trace graph {trace_name!r} changed during conversion")
    if directed_count:
        ordering = np.lexsort((indices, sources))
        sources = sources[ordering]
        indices = indices[ordering]
        weights = weights[ordering]
    counts = np.bincount(sources, minlength=node_count)
    indptr = np.empty(node_count + 1, dtype=np.int64)
    indptr[0] = 0
    np.cumsum(counts, out=indptr[1:])
    return indptr, np.ascontiguousarray(indices), np.ascontiguousarray(weights)


def build_unlabeled_graph_stats(
    graph: Any,
    *,
    fused_indptr: np.ndarray,
    relation_edge_counts: Mapping[str, int],
) -> np.ndarray:
    import networkx as nx

    if set(relation_edge_counts) != set(TRACE_NAMES) or any(
        value < 0 for value in relation_edge_counts.values()
    ):
        raise ValueError("relation_edge_counts must bind all five factual relations")
    node_count = fused_indptr.shape[0] - 1
    degrees = np.diff(fused_indptr).astype(np.float64, copy=False)
    directed_edges = int(fused_indptr[-1])
    density = (
        directed_edges / (node_count * (node_count - 1)) if node_count > 1 else 0.0
    )
    components = nx.number_connected_components(graph)
    degree_quantiles = np.percentile(degrees, (25, 50, 90))
    if node_count > 1:
        _, degree_counts = np.unique(degrees, return_counts=True)
        probabilities = degree_counts.astype(np.float64) / node_count
        degree_entropy = float(
            -np.sum(probabilities * np.log(probabilities)) / np.log(node_count)
        )
    else:
        degree_entropy = 0.0
    relation_total = sum(relation_edge_counts.values())
    relation_proportions = (
        tuple(relation_edge_counts[name] / relation_total for name in TRACE_NAMES)
        if relation_total
        else (0.0,) * len(TRACE_NAMES)
    )
    values = (
        np.log1p(node_count),
        density,
        components / node_count,
        float(np.mean(degrees == 0)),
        *(float(np.log1p(value)) for value in degree_quantiles),
        degree_entropy,
        *relation_proportions,
    )
    if len(values) != len(GRAPH_STAT_NAMES):
        raise AssertionError("graph statistic inventory changed")
    return np.asarray(values, dtype=np.float32)


def _labels(payload: Mapping[str, Any], node_count: int) -> np.ndarray:
    labels = np.asarray(payload["labels"])
    if labels.shape != (node_count,) or labels.dtype.hasobject or not np.isin(labels, (0, 1)).all():
        raise _fail("labels must be a length-N binary numeric array")
    return np.ascontiguousarray(labels, dtype=np.uint8)


def _split_masks(
    payload: Mapping[str, Any],
    *,
    node_count: int,
    regime: str,
) -> list[tuple[GlobalSplitDescriptor, dict[str, np.ndarray]]]:
    raw_splits = payload["splits"]
    if isinstance(raw_splits, dict):
        if set(raw_splits) != set(range(len(raw_splits))):
            raise _fail("split mapping keys must be contiguous integer fold IDs")
        split_items = tuple(raw_splits[fold] for fold in range(len(raw_splits)))
    elif isinstance(raw_splits, (list, tuple)):
        split_items = tuple(raw_splits)
    else:
        raise _fail("splits must be a nonempty sequence or integer-keyed mapping")
    if not split_items:
        raise _fail("splits must not be empty")
    if len(split_items) != 5:
        raise _fail("official SocialGraph-FM Global requires exactly five folds per regime")
    converted = []
    regime_token = regime.replace(".", "p")
    for fold, split in enumerate(split_items):
        if not isinstance(split, dict) or set(split) != {"train", "val", "test"}:
            raise _fail("each official split must contain exactly train/val/test")
        arrays: dict[str, np.ndarray] = {}
        names: dict[str, str] = {}
        for source_role, role in (("train", "train"), ("val", "validation"), ("test", "test")):
            mask = np.asarray(split[source_role])
            if mask.shape != (node_count,) or mask.dtype.hasobject:
                raise _fail(f"split {fold} {source_role} mask has the wrong shape or dtype")
            if not np.isin(mask, (False, True, 0, 1)).all():
                raise _fail(f"split {fold} {source_role} mask is not binary")
            name = f"split_{regime_token}_{fold}_{role}"
            names[role] = name
            arrays[name] = np.ascontiguousarray(mask, dtype=np.bool_)
        if not all(bool(mask.any()) for mask in arrays.values()):
            raise _fail(f"split {fold} contains an empty role")
        train, validation, test = (arrays[names[role]] for role in ("train", "validation", "test"))
        if np.logical_or(
            np.logical_and(train, validation),
            np.logical_or(np.logical_and(train, test), np.logical_and(validation, test)),
        ).any():
            raise _fail(f"split {fold} role masks overlap")
        split_id = f"{regime}-fold-{fold}"
        descriptor = GlobalSplitDescriptor.create(
            split_id=split_id,
            regime=regime,
            fold=fold,
            train_array=names["train"],
            validation_array=names["validation"],
            test_array=names["test"],
        )
        converted.append((descriptor, arrays))
    return converted


def _validate_variant_split_records(
    full_records: Sequence[tuple[GlobalSplitDescriptor, dict[str, np.ndarray]]],
    variant_records: Sequence[tuple[GlobalSplitDescriptor, dict[str, np.ndarray]]],
    *,
    labels: np.ndarray,
    regime: str,
) -> str:
    if len(full_records) != 5 or len(variant_records) != 5 or regime == "full":
        raise _fail("undersampling validation requires two aligned five-fold inventories")
    removal_fraction = float(regime)
    stratified_folds: list[bool] = []
    global_folds: list[bool] = []
    for fold, ((_, full), (_, variant)) in enumerate(
        zip(full_records, variant_records, strict=True)
    ):
        full_validation = _split_role_array(full, "validation")
        variant_validation = _split_role_array(variant, "validation")
        full_test = _split_role_array(full, "test")
        variant_test = _split_role_array(variant, "test")
        full_train = _split_role_array(full, "train")
        variant_train = _split_role_array(variant, "train")
        if not np.array_equal(variant_validation, full_validation):
            raise _fail(f"regime {regime!r} fold {fold} changed validation membership")
        if not np.array_equal(variant_test, full_test):
            raise _fail(f"regime {regime!r} fold {fold} changed test membership")
        if bool(np.logical_and(variant_train, ~full_train).any()):
            raise _fail(f"regime {regime!r} fold {fold} train is not a full-train subset")
        if np.array_equal(variant_train, full_train):
            raise _fail(f"regime {regime!r} fold {fold} train is not a strict subset")
        full_counts = _binary_class_counts(full_train, labels)
        observed_counts = _binary_class_counts(variant_train, labels)
        expected_stratified = tuple(
            count - int(count * removal_fraction) for count in full_counts
        )
        expected_global = sum(full_counts) - int(
            sum(full_counts) * removal_fraction
        )
        is_stratified = observed_counts == expected_stratified
        is_global = sum(observed_counts) == expected_global
        if not is_stratified and not is_global:
            raise _fail(
                f"regime {regime!r} fold {fold} retained class counts "
                f"{observed_counts}; expected per-class {expected_stratified} or "
                f"legacy-global total {expected_global}"
            )
        stratified_folds.append(is_stratified)
        global_folds.append(is_global)
    if all(stratified_folds):
        return "stratified-per-class-floor-removal"
    if all(global_folds):
        return "legacy-global-floor-removal"
    raise _fail(f"regime {regime!r} mixes incompatible undersampling semantics across folds")


def _split_role_array(arrays: Mapping[str, np.ndarray], role: str) -> np.ndarray:
    suffix = f"_{role}"
    matches = [array for name, array in arrays.items() if name.endswith(suffix)]
    if len(matches) != 1:
        raise AssertionError(f"split record does not contain exactly one {role!r} array")
    return matches[0]


def _binary_class_counts(mask: np.ndarray, labels: np.ndarray) -> tuple[int, int]:
    return (
        int(np.logical_and(mask, labels == 0).sum()),
        int(np.logical_and(mask, labels == 1).sum()),
    )


def validate_undersampling_inventory(
    records: Sequence[tuple[GlobalSplitDescriptor, dict[str, np.ndarray]]],
    *,
    labels: np.ndarray,
) -> None:
    """Require retained class counts to decrease monotonically with removal strength."""

    by_regime: dict[str, dict[int, dict[str, np.ndarray]]] = {}
    for descriptor, arrays in records:
        folds = by_regime.setdefault(descriptor.regime, {})
        if descriptor.fold in folds:
            raise _fail(
                f"duplicate split inventory for {descriptor.regime!r} fold {descriptor.fold}"
            )
        folds[descriptor.fold] = arrays
    if "full" not in by_regime or any(
        set(folds) != set(range(5)) for folds in by_regime.values()
    ):
        raise _fail("undersampling inventory must contain complete five-fold regimes and full")
    ordered_regimes = tuple(regime for regime in OFFICIAL_REGIMES if regime in by_regime)
    for fold in range(5):
        previous_counts: tuple[int, int] | None = None
        previous_regime = ""
        for regime in ordered_regimes:
            train = _split_role_array(by_regime[regime][fold], "train")
            counts = _binary_class_counts(train, labels)
            if previous_counts is not None and (
                any(current > previous for current, previous in zip(counts, previous_counts))
                or sum(counts) >= sum(previous_counts)
            ):
                raise _fail(
                    f"regime {regime!r} fold {fold} is not monotonically stronger than "
                    f"{previous_regime!r}: observed {counts} after {previous_counts}"
                )
            previous_counts = counts
            previous_regime = regime


def validate_undersampling_splits(
    full_payload: Mapping[str, Any],
    variant_payload: Mapping[str, Any],
    *,
    node_count: int,
    regime: str,
) -> list[tuple[GlobalSplitDescriptor, dict[str, np.ndarray]]]:
    full_records = _split_masks(full_payload, node_count=node_count, regime="full")
    variant_records = _split_masks(variant_payload, node_count=node_count, regime=regime)
    labels = _labels(full_payload, node_count)
    if not np.array_equal(_labels(variant_payload, node_count), labels):
        raise _fail(f"regime {regime!r} labels differ from full")
    _validate_variant_split_records(
        full_records, variant_records, labels=labels, regime=regime
    )
    return variant_records


def _array_content_identity(array: np.ndarray) -> dict[str, Any]:
    value = np.ascontiguousarray(array)
    return {
        "dtype": value.dtype.str,
        "shape": list(value.shape),
        "sha256": hashlib.sha256(memoryview(value).cast("B")).hexdigest(),
    }


def _factual_array_fingerprint(
    edge_index: np.ndarray,
    relation_arrays: Mapping[str, np.ndarray],
    trace_membership: np.ndarray,
) -> str:
    inventory = {
        "edge_index": _array_content_identity(edge_index),
        "trace_membership": _array_content_identity(trace_membership),
        **{
            name: _array_content_identity(relation_arrays[name])
            for name in sorted(relation_arrays)
        },
    }
    return canonical_sha256(
        {
            "schemaVersion": "socialgraph-fm.global-model-factual-graph/1.0",
            "arrays": inventory,
        }
    )


def factual_graph_fingerprint(
    payload: Mapping[str, Any],
    *,
    node_count: int,
) -> str:
    """Hash canonical fused topology and all five factual weighted relations."""

    edge_index, degree_bucket, _ = graph_to_arrays(payload["graph"])
    if degree_bucket.shape[0] != node_count:
        raise _fail("factual graph node count differs from full")
    inventory = {
        "edge_index": _array_content_identity(edge_index),
        "trace_membership": _array_content_identity(
            _trace_membership(payload, node_count)
        ),
    }
    for trace_name in TRACE_NAMES:
        token = TRACE_ARRAY_TOKENS[trace_name]
        indptr, indices, weights = relation_graph_to_csr(
            payload[trace_name], node_count=node_count, trace_name=trace_name
        )
        inventory[f"relation_{token}_indptr"] = _array_content_identity(indptr)
        inventory[f"relation_{token}_indices"] = _array_content_identity(indices)
        inventory[f"relation_{token}_weights"] = _array_content_identity(weights)
        del indptr, indices, weights
    return canonical_sha256(
        {
            "schemaVersion": "socialgraph-fm.global-model-factual-graph/1.0",
            "arrays": inventory,
        }
    )


def validate_variant_factual_graph(
    payload: Mapping[str, Any],
    *,
    node_count: int,
    expected_fingerprint: str,
    regime: str,
) -> str:
    observed = factual_graph_fingerprint(payload, node_count=node_count)
    if observed != expected_fingerprint:
        raise _fail(
            f"regime {regime!r} factual fused topology or weighted relations differ from full"
        )
    return observed


def write_npy_atomic(path: Path, array: np.ndarray) -> None:
    """Write a fixed numeric array using fsync and same-directory atomic replace."""

    value = np.ascontiguousarray(array)
    if value.dtype.hasobject or value.dtype.kind in {"O", "S", "U", "V"}:
        raise _fail("only fixed numeric/bool arrays may cross the safe corpus boundary")
    if value.dtype.kind == "f" and not bool(np.isfinite(value).all()):
        raise _fail("numeric artifacts must not contain NaN or Infinity")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("xb") as stream:
            np.save(stream, value, allow_pickle=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def write_array_artifact(
    root: Path,
    *,
    name: str,
    array: np.ndarray,
) -> GlobalArrayDescriptor:
    relative = f"arrays/{name}.npy"
    path = root / "arrays" / f"{name}.npy"
    write_npy_atomic(path, array)
    return GlobalArrayDescriptor(
        name=name,
        path=relative,
        sha256=file_sha256(path),
        dtype=np.asarray(array).dtype.str,
        shape=tuple(np.asarray(array).shape),
        byteLength=path.stat().st_size,
    )


def convert_trusted_country(
    *,
    country_id: CountryId,
    pickle_sources: Mapping[str, Path],
    text_tensor_path: Path,
    destination: Path,
    trusted_source: bool = False,
) -> GlobalCountryManifest:
    """Convert one country's official sources into immutable safe arrays.

    ``pickle_sources`` must include ``full`` and may include the six official
    undersampling regimes. The destination must not already exist.
    """

    if country_id not in COUNTRY_IDS:
        raise ValueError(f"unsupported Global country {country_id!r}")
    regimes = tuple(pickle_sources)
    if "full" not in pickle_sources or any(regime not in OFFICIAL_REGIMES for regime in regimes):
        raise ValueError(f"pickle_sources must use regimes from {OFFICIAL_REGIMES!r} and include full")
    ordered_regimes = tuple(regime for regime in OFFICIAL_REGIMES if regime in pickle_sources)
    output = destination.expanduser().resolve()
    if output.exists():
        raise _fail("immutable conversion destination already exists")
    require_conversion_disk_space(output)
    output.parent.mkdir(parents=True, exist_ok=True)

    full = load_trusted_global_model_pickle(
        pickle_sources["full"], trusted_source=trusted_source
    )
    node_count = _validate_graph_nodes(full["graph"])
    text_features = load_text_tensor(text_tensor_path)
    if text_features.shape[0] != node_count:
        raise _fail("text tensor row count does not match the graph")
    edge_index, degree_bucket, structure_missing = graph_to_arrays(full["graph"])
    trace_membership = _trace_membership(full, node_count)
    fused_indptr, fused_indices = _edge_index_to_csr(edge_index, node_count)
    relation_arrays: dict[str, np.ndarray] = {}
    relation_edge_counts: dict[str, int] = {}
    for trace_name in TRACE_NAMES:
        token = TRACE_ARRAY_TOKENS[trace_name]
        indptr, indices, weights = relation_graph_to_csr(
            full[trace_name], node_count=node_count, trace_name=trace_name
        )
        relation_arrays[f"relation_{token}_indptr"] = indptr
        relation_arrays[f"relation_{token}_indices"] = indices
        relation_arrays[f"relation_{token}_weights"] = weights
        relation_edge_counts[trace_name] = indices.shape[0]
    fixed_arrays = {
        "edge_index": edge_index,
        "fused_indptr": fused_indptr,
        "fused_indices": fused_indices,
        "text_features": text_features,
        "degree_bucket": degree_bucket,
        "structure_missing": structure_missing,
        "graph_stats": build_unlabeled_graph_stats(
            full["graph"],
            fused_indptr=fused_indptr,
            relation_edge_counts=relation_edge_counts,
        ),
        "labels": _labels(full, node_count),
        "trace_membership": trace_membership,
        **relation_arrays,
    }
    factual_fingerprint = _factual_array_fingerprint(
        edge_index, relation_arrays, trace_membership
    )

    full_split_records = _split_masks(full, node_count=node_count, regime="full")
    split_records = list(full_split_records)
    del full
    gc.collect()
    for regime in ordered_regimes:
        if regime == "full":
            continue
        payload = load_trusted_global_model_pickle(
            pickle_sources[regime], trusted_source=trusted_source
        )
        if _validate_graph_nodes(payload["graph"]) != node_count:
            raise _fail(f"regime {regime!r} graph node count differs from full")
        if not np.array_equal(_labels(payload, node_count), fixed_arrays["labels"]):
            raise _fail(f"regime {regime!r} labels differ from full")
        validate_variant_factual_graph(
            payload,
            node_count=node_count,
            expected_fingerprint=factual_fingerprint,
            regime=regime,
        )
        variant_split_records = _split_masks(
            payload, node_count=node_count, regime=regime
        )
        _validate_variant_split_records(
            full_split_records,
            variant_split_records,
            labels=fixed_arrays["labels"],
            regime=regime,
        )
        split_records.extend(variant_split_records)
        del payload
        gc.collect()
    validate_undersampling_inventory(split_records, labels=fixed_arrays["labels"])

    source_hashes = {
        **{
            f"pickle:{regime}": file_sha256(pickle_sources[regime].expanduser().resolve())
            for regime in ordered_regimes
        },
        "textTensor": file_sha256(text_tensor_path.expanduser().resolve()),
    }
    preprocessing = {
        "edgeCanonicalization": "undirected-simple-to-sorted-bidirectional",
        "isolateRepair": "none; isolated nodes retained",
        "degreeEncoding": "128-global-percentile-buckets-right-inclusive",
        "textTensorLoad": "torch-weights-only-float32",
        "traceMembership": "node-presence-fixed-five-trace-order",
        "relationCSR": "factual-bidirectional-self-loop-free-float64-raw-weight",
        "factualGraphHash": factual_fingerprint,
        "graphStats": ",".join(GRAPH_STAT_NAMES),
    }
    with tempfile.TemporaryDirectory(prefix=f".{country_id}.staging-", dir=output.parent) as raw_stage:
        stage = Path(raw_stage)
        descriptors = [
            write_array_artifact(stage, name=name, array=array)
            for name, array in fixed_arrays.items()
        ]
        split_descriptors = []
        for split_descriptor, masks in split_records:
            split_descriptors.append(split_descriptor)
            descriptors.extend(
                write_array_artifact(stage, name=name, array=array)
                for name, array in masks.items()
            )
        manifest = GlobalCountryManifest.create(
            country_id=country_id,
            node_count=node_count,
            edge_count=edge_index.shape[1],
            arrays=descriptors,
            splits=split_descriptors,
            source_hashes=source_hashes,
            relation_edge_counts=relation_edge_counts,
            preprocessing=preprocessing,
        )
        atomic_write_contract(stage / "manifest.json", manifest)
        os.replace(stage, output)
    return manifest


def publish_corpus_manifest(
    root: Path,
    *,
    country_manifest_paths: Mapping[CountryId, Path],
) -> GlobalCorpusManifest:
    """Bind six already-converted country manifests into one root index."""

    corpus_root = root.expanduser().resolve()
    if tuple(country_manifest_paths) != COUNTRY_IDS:
        raise ValueError(f"country_manifest_paths must use fixed order {COUNTRY_IDS!r}")
    entries = []
    for country_id in COUNTRY_IDS:
        manifest_path = country_manifest_paths[country_id].expanduser().resolve()
        try:
            relative = manifest_path.relative_to(corpus_root).as_posix()
        except ValueError as exc:
            raise _fail(f"country manifest {country_id!r} is outside the corpus root") from exc
        from .contracts import read_country_manifest

        country_manifest = read_country_manifest(manifest_path)
        if country_manifest.country_id != country_id:
            raise _fail(f"country manifest identity mismatch for {country_id!r}")
        entries.append(
            GlobalCorpusEntry.from_country_manifest(
                country_manifest,
                manifest_path=relative,
            )
        )
    manifest = GlobalCorpusManifest.create(entries)
    atomic_write_contract(corpus_root / "manifest.json", manifest)
    return manifest


__all__ = [
    "ALLOWED_PICKLE_GLOBALS",
    "MAX_PICKLE_BYTES",
    "MAX_TEXT_TENSOR_BYTES",
    "MINIMUM_CONVERSION_FREE_BYTES",
    "OFFICIAL_REGIMES",
    "WORKER_CONTRACT_SCHEMA",
    "WORKER_RECEIPT_SCHEMA",
    "DiskSpaceInspection",
    "PickleInspection",
    "ValidatedWorkerRequest",
    "WorkerConversionReceipt",
    "build_unlabeled_graph_stats",
    "build_worker_contract",
    "convert_country_in_worker",
    "convert_trusted_country",
    "factual_graph_fingerprint",
    "graph_to_arrays",
    "inspect_global_model_pickle",
    "load_text_tensor",
    "load_trusted_global_model_pickle",
    "publish_corpus_manifest",
    "relation_graph_to_csr",
    "require_conversion_disk_space",
    "validate_undersampling_inventory",
    "validate_undersampling_splits",
    "validate_variant_factual_graph",
    "validate_worker_contract",
    "write_array_artifact",
    "write_npy_atomic",
]
