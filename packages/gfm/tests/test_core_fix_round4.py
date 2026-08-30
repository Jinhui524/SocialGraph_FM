from __future__ import annotations

import copy
import hashlib
import http.client
import json
import os
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any
from urllib.parse import unquote_to_bytes

import pytest
from pydantic import BaseModel, ValidationError

from socialgraph_gfm.canonical import canonical_bytes, canonical_sha256
from socialgraph_gfm.core import inference_contracts as inference_contracts_module
from socialgraph_gfm.core.inference_cli import _parser
from socialgraph_gfm.core.inference_contracts import (
    GfmCapabilities,
    GfmRunResult,
    GfmRunStatus,
    InternalCreateRunReceipt,
    InternalCreateRunRequest,
    InternalErrorEnvelope,
    RunExecutionSnapshot,
)
from socialgraph_gfm.core.governance import GovernanceFinding
from socialgraph_gfm.core.inference_service import InferenceRuntime, RunStore
from socialgraph_gfm.core.serving_control import ServingControlStore

from _core_inference_test_support import (
    _make_test_internal_create_request,
    _make_test_only_run_store,
)
from test_core_inference_fix_round1 import _catalog, _serving_registry


_SCHEMA_ANNOTATIONS = frozenset({"description", "examples", "title"})
_SCHEMA_SET_LIKE_ARRAYS = frozenset({"allOf", "anyOf", "enum", "oneOf", "required", "type"})
_SCHEMA_ARRAY_KEYWORDS = frozenset({"allOf", "anyOf", "oneOf", "prefixItems"})
_SCHEMA_MAP_KEYWORDS = frozenset({"$defs", "dependentSchemas", "patternProperties", "properties"})
_SCHEMA_SINGLE_KEYWORDS = frozenset(
    {
        "additionalProperties",
        "contains",
        "contentSchema",
        "else",
        "if",
        "items",
        "not",
        "propertyNames",
        "then",
        "unevaluatedItems",
        "unevaluatedProperties",
    }
)
_BOUNDARY_ROOTS: dict[str, type[BaseModel]] = {
    "capabilities": GfmCapabilities,
    "error": InternalErrorEnvelope,
    "finding": GovernanceFinding,
    "result": GfmRunResult,
    "status": GfmRunStatus,
}


def _decode_json_pointer_token(token: str) -> str:
    decoded: list[str] = []
    index = 0
    while index < len(token):
        character = token[index]
        if character != "~":
            decoded.append(character)
            index += 1
            continue
        if index + 1 >= len(token) or token[index + 1] not in {"0", "1"}:
            raise ValueError(f"invalid JSON Pointer escape in {token!r}")
        decoded.append("~" if token[index + 1] == "0" else "/")
        index += 2
    return "".join(decoded)


def _decode_uri_fragment(fragment: str) -> str:
    hexadecimal = frozenset("0123456789abcdefABCDEF")
    index = 0
    while index < len(fragment):
        if fragment[index] != "%":
            index += 1
            continue
        if index + 2 >= len(fragment) or any(
            character not in hexadecimal for character in fragment[index + 1 : index + 3]
        ):
            raise ValueError(f"malformed percent escape in URI fragment: {fragment!r}")
        index += 3
    try:
        return unquote_to_bytes(fragment).decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError(f"invalid UTF-8 URI fragment: {fragment!r}") from exc


def normalize_boundary_schema(schema: dict[str, Any]) -> dict[str, Any]:
    """Inline local refs and retain all JSON Schema behavior, minus three annotations."""

    document = copy.deepcopy(schema)

    def normalize_instance(value: Any) -> Any:
        if isinstance(value, dict):
            return {key: normalize_instance(value[key]) for key in sorted(value)}
        if isinstance(value, list):
            return [normalize_instance(item) for item in value]
        return value

    def resolve_pointer(reference: str) -> tuple[str, Any]:
        if not reference.startswith("#"):
            raise ValueError(f"external $ref is forbidden: {reference}")
        fragment = _decode_uri_fragment(reference[1:])
        if fragment and not fragment.startswith("/"):
            raise ValueError(f"local $ref must be a JSON Pointer: {reference}")
        current: Any = document
        for raw_token in fragment.split("/")[1:] if fragment else ():
            token = _decode_json_pointer_token(raw_token)
            if isinstance(current, dict) and token in current:
                current = current[token]
            elif isinstance(current, list) and token.isdecimal() and int(token) < len(current):
                current = current[int(token)]
            else:
                raise ValueError(f"missing JSON Pointer target: {reference}")
        return fragment, current

    def normalize_schema(value: Any, *, active: tuple[str, ...] = ()) -> Any:
        if not isinstance(value, dict):
            return normalize_instance(value)

        if "$ref" in value:
            reference = value["$ref"]
            if not isinstance(reference, str):
                raise ValueError("$ref must be a string")
            pointer, target_source = resolve_pointer(reference)
            if pointer in active:
                chain = " -> ".join((*active, pointer))
                raise ValueError(f"cyclic local $ref: {chain}")
            target = normalize_schema(target_source, active=(*active, pointer))
            sibling_source = {
                key: item
                for key, item in value.items()
                if key not in {"$ref", "$defs", *_SCHEMA_ANNOTATIONS}
            }
            if not sibling_source:
                return target
            outer = normalize_schema(sibling_source, active=active)
            if not isinstance(outer, dict):
                raise TypeError("$ref siblings must normalize to an object")
            existing = outer.get("allOf", [])
            if not isinstance(existing, list):
                raise ValueError("allOf must be an array")
            outer["allOf"] = sorted([*existing, target], key=canonical_bytes)
            return {key: outer[key] for key in sorted(outer)}

        normalized: dict[str, Any] = {}
        for key in sorted(value):
            if key in _SCHEMA_ANNOTATIONS or key == "$defs":
                continue
            item = value[key]
            if key in _SCHEMA_MAP_KEYWORDS and isinstance(item, dict):
                normalized[key] = {
                    name: normalize_schema(item[name], active=active) for name in sorted(item)
                }
            elif key in _SCHEMA_SINGLE_KEYWORDS:
                normalized[key] = normalize_schema(item, active=active)
            elif key in _SCHEMA_ARRAY_KEYWORDS and isinstance(item, list):
                children = [normalize_schema(child, active=active) for child in item]
                if key in _SCHEMA_SET_LIKE_ARRAYS:
                    children.sort(key=canonical_bytes)
                normalized[key] = children
            elif key == "dependentRequired" and isinstance(item, dict):
                dependencies: dict[str, Any] = {}
                for name in sorted(item):
                    names = normalize_instance(item[name])
                    if isinstance(names, list):
                        names.sort(key=canonical_bytes)
                    dependencies[name] = names
                normalized[key] = dependencies
            elif key in {"const", "default"}:
                normalized[key] = normalize_instance(item)
            elif key == "enum" and isinstance(item, list):
                members = [normalize_instance(member) for member in item]
                members.sort(key=canonical_bytes)
                normalized[key] = members
            elif key in {"required", "type"} and isinstance(item, list):
                members = [normalize_instance(member) for member in item]
                members.sort(key=canonical_bytes)
                normalized[key] = members
            else:
                normalized[key] = normalize_instance(item)
        return normalized

    normalized = normalize_schema(document)
    if not isinstance(normalized, dict):
        raise ValueError("root JSON Schema must normalize to an object")
    return normalized


def _boundary_manifest() -> dict[str, Any]:
    return {
        "normalizationPolicy": {
            "annotationsStripped": sorted(_SCHEMA_ANNOTATIONS),
            "instanceValues": "preserved recursively without schema-key reinterpretation",
            "localRefs": "inlined as allOf branches; siblings remain at outer schema scope",
            "mappingKeys": "Unicode code-point order",
            "omittedAfterInlining": ["$defs"],
            "orderSensitiveArrays": "preserved, including raw instance arrays and prefixItems",
            "setLikeArraysSorted": sorted({*_SCHEMA_SET_LIKE_ARRAYS, "dependentRequired values"}),
        },
        "roots": {
            name: normalize_boundary_schema(model.model_json_schema(by_alias=True))
            for name, model in sorted(_BOUNDARY_ROOTS.items())
        },
        "schemaVersion": "socialgraph-fm.core-boundary-manifest/2.0",
    }


def _raw_property(schema: dict[str, Any], name: str) -> dict[str, Any]:
    matches: list[dict[str, Any]] = []

    def visit(value: Any) -> None:
        if isinstance(value, dict):
            properties = value.get("properties")
            if isinstance(properties, dict) and isinstance(properties.get(name), dict):
                matches.append(properties[name])
            for child in value.values():
                visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)

    visit(schema)
    assert len(matches) == 1, f"expected one {name!r} property, found {len(matches)}"
    return matches[0]


def test_complete_neutral_behavioral_boundary_manifest_is_canonical() -> None:
    artifact_bytes = (
        Path(__file__).parents[3] / "contracts" / "core-inference-boundaries.json"
    ).read_bytes()
    artifact = json.loads(artifact_bytes)
    actual = _boundary_manifest()

    assert actual == artifact
    assert canonical_bytes(actual) == artifact_bytes


def test_boundary_normalizer_detects_additional_properties_drift() -> None:
    raw = GfmRunStatus.model_json_schema(by_alias=True)
    mutated = copy.deepcopy(raw)
    assert mutated["additionalProperties"] is False
    mutated["additionalProperties"] = True

    assert normalize_boundary_schema(mutated) != normalize_boundary_schema(raw)


def test_boundary_normalizer_detects_created_at_datetime_format_drift() -> None:
    raw = GfmRunStatus.model_json_schema(by_alias=True)
    mutated = copy.deepcopy(raw)
    created_at = _raw_property(mutated, "createdAt")
    assert created_at.pop("format") == "date-time"

    assert normalize_boundary_schema(mutated) != normalize_boundary_schema(raw)


@pytest.mark.parametrize("drift", ["absent", "value"])
def test_boundary_normalizer_detects_error_code_default_presence_and_value_drift(
    drift: str,
) -> None:
    raw = GfmRunStatus.model_json_schema(by_alias=True)
    mutated = copy.deepcopy(raw)
    error_code = _raw_property(mutated, "errorCode")
    assert "default" in error_code and error_code["default"] is None
    if drift == "absent":
        del error_code["default"]
    else:
        error_code["default"] = "DRIFT"

    assert normalize_boundary_schema(mutated) != normalize_boundary_schema(raw)


def test_boundary_normalizer_detects_edge_identity_explicit_null_union_drift() -> None:
    raw = GovernanceFinding.model_json_schema(by_alias=True)
    mutated = copy.deepcopy(raw)
    edge_identity = _raw_property(mutated, "edgeIdentity")
    variants = edge_identity["anyOf"]
    assert isinstance(variants, list)
    assert any(isinstance(item, dict) and item.get("type") == "null" for item in variants)
    edge_identity["anyOf"] = [
        item for item in variants if not (isinstance(item, dict) and item.get("type") == "null")
    ]

    assert normalize_boundary_schema(mutated) != normalize_boundary_schema(raw)


def test_boundary_normalizer_decodes_pointer_and_preserves_ref_sibling_conjunctively() -> None:
    raw = {
        "$defs": {
            "named/schema~v2": {"minLength": 1, "title": "implementation name", "type": "string"}
        },
        "properties": {
            "value": {"$ref": "#/$defs/named~1schema~0v2", "maxLength": 8},
        },
        "type": "object",
    }
    normalized = normalize_boundary_schema(raw)
    mutated = copy.deepcopy(raw)
    _raw_property(mutated, "value")["maxLength"] = 9

    assert normalized == {
        "properties": {
            "value": {
                "allOf": [{"minLength": 1, "type": "string"}],
                "maxLength": 8,
            }
        },
        "type": "object",
    }
    assert "$defs" not in normalized
    assert normalize_boundary_schema(mutated) != normalized


def test_boundary_normalizer_preserves_annotation_named_properties() -> None:
    raw = {
        "description": "schema annotation",
        "properties": {
            "description": {"description": "field annotation", "type": "integer"},
            "examples": {"items": {"type": "string"}, "type": "array"},
            "title": {"title": "field annotation", "type": "string"},
        },
        "required": ["title", "examples", "description"],
        "title": "schema annotation",
        "type": "object",
    }

    assert normalize_boundary_schema(raw) == {
        "properties": {
            "description": {"type": "integer"},
            "examples": {"items": {"type": "string"}, "type": "array"},
            "title": {"type": "string"},
        },
        "required": ["description", "examples", "title"],
        "type": "object",
    }


@pytest.mark.parametrize("keyword", ["const", "default", "enum"])
def test_boundary_normalizer_preserves_object_instance_values_recursively(keyword: str) -> None:
    instance = {
        "description": "kept",
        "examples": ["z", "a"],
        "nested": [{"enum": ["z", "a"], "required": ["right", "left"]}],
        "required": ["z", "a"],
        "title": "kept",
    }
    raw: dict[str, Any] = {keyword: [instance] if keyword == "enum" else instance}
    expected = copy.deepcopy(raw)
    mutated = copy.deepcopy(raw)
    mutated_instance = mutated[keyword][0] if keyword == "enum" else mutated[keyword]
    assert isinstance(mutated_instance, dict)
    mutated_instance["nested"][0]["required"][0] = "changed"

    assert normalize_boundary_schema(raw) == expected
    assert normalize_boundary_schema(mutated) != normalize_boundary_schema(raw)


def test_boundary_normalizer_keeps_ref_siblings_outer_for_unevaluated_semantics() -> None:
    target = {
        "properties": {"value": {"type": "string"}},
        "required": ["value"],
        "type": "object",
    }
    raw = {
        "$defs": {"record": target},
        "$ref": "#/$defs/record",
        "allOf": [{"properties": {"tag": {"type": "string"}}}],
        "unevaluatedProperties": False,
    }

    normalized = normalize_boundary_schema(raw)

    assert normalized["unevaluatedProperties"] is False
    assert normalized["allOf"] == sorted(
        [target, {"properties": {"tag": {"type": "string"}}}], key=canonical_bytes
    )


def test_boundary_normalizer_percent_decodes_utf8_fragment_before_pointer_tokens() -> None:
    raw = {
        "$defs": {"a b": {"const": "decoded"}},
        "$ref": "#/$defs/a%20b",
    }

    assert normalize_boundary_schema(raw) == {"const": "decoded"}


@pytest.mark.parametrize("reference", ["#/$defs/bad%", "#/$defs/bad%GG", "#/$defs/%FF"])
def test_boundary_normalizer_rejects_malformed_percent_or_utf8_fragments(reference: str) -> None:
    with pytest.raises(ValueError, match="URI fragment"):
        normalize_boundary_schema({"$defs": {}, "$ref": reference})


def test_boundary_normalizer_sorts_dependent_required_but_not_raw_arrays() -> None:
    first: dict[str, Any] = {
        "default": {
            "enum": ["z", "a"],
            "nested": [{"required": ["right", "left"]}],
            "required": ["z", "a"],
        },
        "dependentRequired": {"account": ["email", "name"]},
        "type": "object",
    }
    reordered = copy.deepcopy(first)
    reordered["dependentRequired"]["account"] = ["name", "email"]

    normalized = normalize_boundary_schema(first)
    assert normalized == normalize_boundary_schema(reordered)
    assert normalized["default"] == first["default"]


@pytest.mark.parametrize(
    ("raw", "message"),
    [
        ({"$ref": "other.json#/$defs/Value"}, "external"),
        ({"$ref": "#/$defs/Missing"}, "missing JSON Pointer"),
        (
            {"$defs": {"Loop": {"$ref": "#/$defs/Loop"}}, "$ref": "#/$defs/Loop"},
            "cyclic local",
        ),
    ],
)
def test_boundary_normalizer_fails_explicitly_for_unsafe_refs(
    raw: dict[str, Any], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        normalize_boundary_schema(raw)


def _publish_control(
    path: Path,
    *,
    root: Path,
    registry_path: Path,
    catalog_path: Path,
) -> None:
    registry_bytes = registry_path.read_bytes()
    catalog_bytes = catalog_path.read_bytes()
    registry = json.loads(registry_bytes)
    catalog = json.loads(catalog_bytes)
    versions = root / "control-versions"
    versions.mkdir()
    versioned_registry = versions / "registry.json"
    versioned_catalog = versions / "catalog.json"
    versioned_registry.write_bytes(registry_bytes)
    versioned_catalog.write_bytes(catalog_bytes)
    payload: dict[str, object] = {
        "schemaVersion": "socialgraph-fm.core-serving-control/1.0",
        "generation": 1,
        "registry": {
            "relativePath": versioned_registry.relative_to(root).as_posix(),
            "sha256": hashlib.sha256(registry_bytes).hexdigest(),
            "semanticHash": canonical_sha256(registry),
            "generation": registry["generation"],
        },
        "catalog": {
            "relativePath": versioned_catalog.relative_to(root).as_posix(),
            "sha256": hashlib.sha256(catalog_bytes).hexdigest(),
            "semanticHash": canonical_sha256(catalog),
            "generation": catalog["generation"],
        },
    }
    payload["controlHash"] = canonical_sha256(payload)
    path.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")


def _control(
    tmp_path: Path, registry, catalog, *, control_root: Path | None = None
) -> ServingControlStore:
    root = control_root or tmp_path
    versions = root / "fixture-control-inputs"
    versions.mkdir(exist_ok=True)
    registry_path = versions / "registry.json"
    catalog_path = versions / "catalog.json"
    registry_path.write_bytes(registry.path.read_bytes())
    catalog_path.write_bytes(catalog.path.read_bytes())
    path = root / "serving-control.json"
    _publish_control(path, root=root, registry_path=registry_path, catalog_path=catalog_path)
    return ServingControlStore.load(path, high_water_root=tmp_path / "high-water")


def _old_create_body(reference) -> dict[str, object]:
    return {
        "schemaVersion": "socialgraph-fm.core-internal-create-run/2.0",
        "request": {
            "schemaVersion": "socialgraph-fm.core-run-request/2.0",
            "graphVersionId": "graph-v1",
            "taskId": "core.risk_and_trust_review",
            "targetScope": {"kind": "risk-review", "nodeIds": ["a"], "edgeIds": []},
            "modelVersionId": "socialgraph-fm-core/review",
            "parameters": {"kind": "risk-and-trust", "topKSimilarCases": 0},
        },
        "graphReference": reference.model_dump(mode="json", by_alias=True),
    }


def test_historical_request_decoder_accepts_exact_early_and_round3_shapes(
    tmp_path: Path,
) -> None:
    _catalog_value, reference, _bundle = _catalog(tmp_path)
    early = _old_create_body(reference)
    round3 = copy.deepcopy(early)
    round3["expectedServingControl"] = None
    early_decoded = inference_contracts_module._decode_persisted_create_run_request(
        json.dumps(early).encode("utf-8")
    )
    round3_decoded = inference_contracts_module._decode_persisted_create_run_request(
        json.dumps(round3).encode("utf-8")
    )
    assert early_decoded.schema_version == "socialgraph-fm.core-internal-create-run/2.0"
    assert round3_decoded.schema_version == "socialgraph-fm.core-internal-create-run/2.0"
    assert early_decoded.request_hash == canonical_sha256(early)
    assert round3_decoded.request_hash == canonical_sha256(round3)

    nonnull = copy.deepcopy(round3)
    nonnull["expectedServingControl"] = {}
    with pytest.raises(ValidationError):
        inference_contracts_module._decode_persisted_create_run_request(
            json.dumps(nonnull).encode("utf-8")
        )


def _open_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind(("127.0.0.1", 0))
        return int(listener.getsockname()[1])


def _post_after_server_start(
    *, port: int, token: str, body: dict[str, object]
) -> tuple[int, dict[str, object]]:
    deadline = time.monotonic() + 5
    while True:
        try:
            connection = http.client.HTTPConnection("127.0.0.1", port, timeout=2)
            connection.request(
                "POST",
                "/internal/core/runs",
                body=json.dumps(body),
                headers={
                    "Authorization": f"Bearer {token}",
                    "Content-Type": "application/json",
                },
            )
            response = connection.getresponse()
            return response.status, json.loads(response.read())
        except ConnectionRefusedError:
            assert time.monotonic() < deadline
            time.sleep(0.02)


def test_cli_rejects_authenticated_legacy_create_before_run_allocation(tmp_path: Path) -> None:
    catalog, reference, _ = _catalog(tmp_path)
    registry = _serving_registry(tmp_path, catalog=catalog)
    runtime_root = registry.runtime_root
    control = _control(tmp_path, registry, catalog, control_root=runtime_root)
    token_file = runtime_root / "session.token"
    port = _open_port()
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(Path(__file__).parents[1] / "src")
    process = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "socialgraph_gfm.core.inference_cli",
            "--runtime-root",
            str(runtime_root),
            "--serving-control",
            str(control.path),
            "--artifact-root",
            str(catalog.artifact_root),
            "--token-file",
            str(token_file),
            "--port",
            str(port),
        ],
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        deadline = time.monotonic() + 15
        while not token_file.is_file() and process.poll() is None:
            assert time.monotonic() < deadline
            time.sleep(0.02)
        assert process.poll() is None, process.stderr.read()
        token = token_file.read_text(encoding="utf-8")
        status, response = _post_after_server_start(
            port=port, token=token, body=_old_create_body(reference)
        )
        assert status == 422
        assert response == {"error": {"code": "GFM_CORE_REQUEST_INVALID"}}
        assert list((runtime_root / "inference" / "runs").iterdir()) == []
    finally:
        process.terminate()
        process.wait(timeout=10)


def test_production_construction_and_input_schemas_have_no_control_fallback(tmp_path: Path) -> None:
    catalog, reference, _ = _catalog(tmp_path)
    registry = _serving_registry(tmp_path, catalog=catalog)
    control = _control(tmp_path, registry, catalog)

    with pytest.raises(TypeError):
        RunStore(tmp_path / "inference", registry=registry, artifact_catalog=catalog)
    store = RunStore(
        tmp_path / "inference",
        registry=registry,
        artifact_catalog=catalog,
        serving_control=control,
    )
    with pytest.raises(TypeError):
        InferenceRuntime(store, registry)
    for invalid_control in (None, object()):
        invalid_root = tmp_path / f"invalid-control-{type(invalid_control).__name__}"
        with pytest.raises(TypeError, match="ServingControlStore"):
            RunStore(
                invalid_root,
                registry=registry,
                artifact_catalog=catalog,
                serving_control=invalid_control,  # type: ignore[arg-type]
            )
        assert not invalid_root.exists()
        with pytest.raises(TypeError, match="ServingControlStore"):
            InferenceRuntime(
                store,
                registry,
                invalid_control,  # type: ignore[arg-type]
            )
    with pytest.raises(TypeError):
        RunStore(
            tmp_path / "other",
            registry=registry,
            artifact_catalog=catalog,
            serving_control=control,
            test_executor=lambda *_args: [],
        )
    valid_request = _make_test_internal_create_request(reference, control).model_dump(
        mode="json", by_alias=True
    )
    for field in ("registry", "artifactCatalog"):
        with pytest.raises(ValidationError):
            InternalCreateRunRequest.model_validate(valid_request | {field: {}})
    cli_arguments = [
        "--runtime-root",
        "runtime",
        "--serving-control",
        "control.json",
        "--artifact-root",
        "artifacts",
        "--token-file",
        "runtime/token",
        "--host",
        "127.0.0.1",
        "--port",
        "8766",
    ]
    for flag in ("--registry", "--artifact-catalog"):
        with pytest.raises(SystemExit):
            _parser().parse_args([*cli_arguments, flag, "fallback.json"])


_LEASE_IDENTITY_FIELDS = (
    "runId",
    "requestHash",
    "controlSourceSha256",
    "controlHash",
    "controlGeneration",
    "registrySourceSha256",
    "registryHash",
    "registryGeneration",
    "artifactCatalogSha256",
    "artifactCatalogHash",
    "artifactCatalogGeneration",
    "modelVersionId",
    "modelVersionHash",
    "checkpointSha256",
    "servingManifestSha256",
    "adapterSchemaHash",
    "calibrationIdentities",
    "calibrationSetHash",
    "taskId",
    "graphVersionId",
    "sourceGraphFactHash",
    "graphVersionHash",
    "artifactId",
    "artifactHash",
    "bundleSha256",
    "graphSchemaVersion",
    "featureContractHash",
    "nodeCount",
    "edgeCount",
    "createdAt",
)


def _lease_projection(snapshot: dict[str, object]) -> dict[str, object]:
    return {
        "schemaVersion": "socialgraph-fm.core-run-lease-identity/2.2",
        **{field: snapshot[field] for field in _LEASE_IDENTITY_FIELDS if field in snapshot},
    }


def _complete_receipt(tmp_path: Path):
    catalog, reference, _ = _catalog(tmp_path)
    registry = _serving_registry(tmp_path, catalog=catalog)
    control = _control(tmp_path, registry, catalog)
    envelope = _make_test_internal_create_request(reference, control)
    store = _make_test_only_run_store(
        tmp_path / "receipt-runtime",
        registry=registry,
        artifact_catalog=catalog,
        serving_control=control,
        executor=lambda *_args: [],
    )
    receipt = store.create(envelope)
    assert isinstance(receipt, InternalCreateRunReceipt)
    return receipt, envelope, reference, registry, catalog, control


def test_receipt_contains_complete_sorted_lease_identity(tmp_path: Path) -> None:
    receipt, envelope, reference, registry, catalog, control = _complete_receipt(tmp_path)
    snapshot = receipt.execution_snapshot.model_dump(mode="json", by_alias=True)
    model = registry._document.models[0]
    calibrations = sorted(
        model.task_head(envelope.request.task_id).calibrations,
        key=lambda item: item.entity_type,
    )
    expected_calibrations = [
        {
            "entityType": binding.entity_type,
            "calibrationVersion": binding.calibration_version,
            "method": binding.calibration_method,
            "calibrationArtifactHash": binding.calibration_artifact_hash,
            "calibrationProtocolHash": binding.calibration_protocol_hash,
            "confidenceKind": binding.confidence_kind,
            "adapterDomain": binding.adapter_domain,
            "adapterSchemaHash": binding.adapter_schema_hash,
            "adapterStateHash": binding.adapter_state_hash,
            "featureContractHash": binding.graph_feature_contract_hash,
            "sha256": binding.calibration_sha256,
        }
        for binding in calibrations
    ]
    expected_identity = {
        "controlSourceSha256": hashlib.sha256(control.path.read_bytes()).hexdigest(),
        "controlHash": envelope.expected_serving_control.control_hash,
        "controlGeneration": envelope.expected_serving_control.control_generation,
        "registrySourceSha256": hashlib.sha256(registry.path.read_bytes()).hexdigest(),
        "registryHash": envelope.expected_serving_control.registry_hash,
        "registryGeneration": envelope.expected_serving_control.registry_generation,
        "artifactCatalogSha256": hashlib.sha256(catalog.path.read_bytes()).hexdigest(),
        "artifactCatalogHash": envelope.expected_serving_control.catalog_hash,
        "artifactCatalogGeneration": envelope.expected_serving_control.catalog_generation,
        "modelVersionId": model.model_version_id,
        "modelVersionHash": model.model_version_hash,
        "checkpointSha256": model.checkpoint.sha256,
        "servingManifestSha256": model.checkpoint.serving_manifest_sha256,
        "adapterSchemaHash": snapshot["adapterSchemaHash"],
        "calibrationIdentities": expected_calibrations,
        "calibrationSetHash": canonical_sha256(expected_calibrations),
        "taskId": envelope.request.task_id,
        "graphVersionId": reference.graph_version_id,
        "sourceGraphFactHash": reference.source_graph_fact_hash,
        "graphVersionHash": reference.graph_version_hash,
        "artifactId": reference.artifact_id,
        "artifactHash": reference.artifact_hash,
        "bundleSha256": reference.bundle_sha256,
        "graphSchemaVersion": reference.graph_schema_version,
        "featureContractHash": reference.feature_contract_hash,
        "nodeCount": reference.node_count,
        "edgeCount": reference.edge_count,
    }
    assert snapshot["schemaVersion"] == "socialgraph-fm.core-run-execution-snapshot/2.2"
    assert {key: snapshot[key] for key in expected_identity} == expected_identity
    assert [item["entityType"] for item in snapshot["calibrationIdentities"]] == [
        "edge",
        "node",
    ]
    calibration_associations = {
        (
            item["calibrationVersion"],
            item["calibrationArtifactHash"],
            item["calibrationProtocolHash"],
            item["sha256"],
        )
        for item in snapshot["calibrationIdentities"]
    }
    assert len(calibration_associations) == 2
    assert receipt.lease_identity_hash == canonical_sha256(_lease_projection(snapshot))


@pytest.mark.parametrize(
    "path",
    [
        ("controlSourceSha256",),
        ("controlHash",),
        ("controlGeneration",),
        ("registrySourceSha256",),
        ("registryHash",),
        ("registryGeneration",),
        ("artifactCatalogSha256",),
        ("artifactCatalogHash",),
        ("artifactCatalogGeneration",),
        ("modelVersionId",),
        ("modelVersionHash",),
        ("checkpointSha256",),
        ("servingManifestSha256",),
        ("adapterSchemaHash",),
        ("calibrationIdentities", 0, "entityType"),
        ("calibrationIdentities", 0, "calibrationVersion"),
        ("calibrationIdentities", 0, "method"),
        ("calibrationIdentities", 0, "calibrationArtifactHash"),
        ("calibrationIdentities", 0, "calibrationProtocolHash"),
        ("calibrationIdentities", 0, "sha256"),
        ("calibrationSetHash",),
        ("taskId",),
        ("graphVersionId",),
        ("sourceGraphFactHash",),
        ("graphVersionHash",),
        ("artifactId",),
        ("artifactHash",),
        ("bundleSha256",),
        ("graphSchemaVersion",),
        ("featureContractHash",),
        ("nodeCount",),
        ("edgeCount",),
        ("createdAt",),
    ],
)
def test_receipt_rejects_every_missing_lease_identity(
    tmp_path: Path, path: tuple[str | int, ...]
) -> None:
    receipt, *_ = _complete_receipt(tmp_path)
    payload = receipt.model_dump(mode="json", by_alias=True)
    snapshot = payload["executionSnapshot"]
    assert isinstance(snapshot, dict)
    assert path[0] in snapshot
    target: object = snapshot
    for part in path[:-1]:
        assert isinstance(target, (dict, list))
        target = target[part]  # type: ignore[index]
    assert isinstance(target, dict)
    assert path[-1] in target
    target.pop(path[-1])  # type: ignore[union-attr]
    with pytest.raises(ValidationError):
        InternalCreateRunReceipt.model_validate(payload)


def _rehash_receipt(payload: dict[str, object], *, lease: bool = True) -> None:
    snapshot = payload["executionSnapshot"]
    assert isinstance(snapshot, dict)
    snapshot["snapshotHash"] = canonical_sha256(
        {key: value for key, value in snapshot.items() if key != "snapshotHash"}
    )
    if lease:
        payload["leaseIdentityHash"] = canonical_sha256(_lease_projection(snapshot))
    payload["receiptHash"] = canonical_sha256(
        {key: value for key, value in payload.items() if key != "receiptHash"}
    )


@pytest.mark.parametrize("case", ["request", "run", "self-consistent-snapshot"])
def test_receipt_rejects_cross_identity_and_rehashed_snapshot_substitution(
    tmp_path: Path, case: str
) -> None:
    receipt, *_ = _complete_receipt(tmp_path)
    payload = copy.deepcopy(receipt.model_dump(mode="json", by_alias=True))
    snapshot = payload["executionSnapshot"]
    assert isinstance(snapshot, dict)
    if case == "request":
        snapshot["requestHash"] = "f" * 64
        _rehash_receipt(payload)
    elif case == "run":
        snapshot["runId"] = "00000000-0000-0000-0000-000000000001"
        _rehash_receipt(payload)
    else:
        snapshot["edgeCount"] = int(snapshot["edgeCount"]) + 1
        _rehash_receipt(payload, lease=False)
    with pytest.raises(ValidationError):
        InternalCreateRunReceipt.model_validate(payload)


def test_receipt_rejects_historical_snapshot_schema(tmp_path: Path) -> None:
    receipt, *_ = _complete_receipt(tmp_path)
    snapshot = receipt.execution_snapshot.model_dump(mode="json", by_alias=True)
    assert isinstance(snapshot, dict)
    snapshot["schemaVersion"] = "socialgraph-fm.core-run-execution-snapshot/2.0"
    snapshot["snapshotHash"] = canonical_sha256(
        {key: value for key, value in snapshot.items() if key != "snapshotHash"}
    )
    with pytest.raises(ValidationError):
        RunExecutionSnapshot.model_validate(snapshot)


def _write_hashed_document(path: Path, payload: dict[str, object], hash_field: str) -> None:
    payload[hash_field] = canonical_sha256(
        {key: value for key, value in payload.items() if key != hash_field}
    )
    path.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")


def _rewrite_persisted_request_and_hash_bindings(run_dir: Path, request: dict[str, object]) -> None:
    request_hash = canonical_sha256(request)
    (run_dir / "request.json").write_text(
        json.dumps(request, separators=(",", ":")), encoding="utf-8"
    )
    manifest_path = run_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_bytes())
    manifest["requestHash"] = request_hash
    _write_hashed_document(manifest_path, manifest, "snapshotHash")
    state_path = run_dir / "state.json"
    state = json.loads(state_path.read_bytes())
    state["requestHash"] = request_hash
    _write_hashed_document(state_path, state, "stateHash")
    result_path = run_dir / "result.json"
    result = json.loads(result_path.read_bytes())
    result["requestHash"] = request_hash
    _write_hashed_document(result_path, result, "resultHash")
    marker_path = run_dir / "success.json"
    marker = json.loads(marker_path.read_bytes())
    marker["requestHash"] = request_hash
    marker["snapshotHash"] = manifest["snapshotHash"]
    marker["resultHash"] = result["resultHash"]
    _write_hashed_document(marker_path, marker, "markerHash")


def _rewrite_run_as_historical(runtime_root: Path, run_id: str, historical_format: str) -> None:
    run_dir = runtime_root / "runs" / run_id
    request_path = run_dir / "request.json"
    request = json.loads(request_path.read_bytes())
    if historical_format in {"round1-2.0", "round2-2.0"}:
        request["schemaVersion"] = "socialgraph-fm.core-internal-create-run/2.0"
        request.pop("expectedServingControl")
    elif historical_format == "round3-2.0":
        request["schemaVersion"] = "socialgraph-fm.core-internal-create-run/2.0"
        request["expectedServingControl"] = None
    else:
        assert historical_format == "round3-2.1"
    request_hash = canonical_sha256(request)
    request_path.write_text(json.dumps(request, separators=(",", ":")), encoding="utf-8")

    manifest_path = run_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_bytes())
    calibrations = manifest.pop("calibrationIdentities")
    manifest.pop("controlSourceSha256")
    manifest["schemaVersion"] = (
        "socialgraph-fm.core-run-execution-snapshot/2.1"
        if historical_format == "round3-2.1"
        else "socialgraph-fm.core-run-execution-snapshot/2.0"
    )
    manifest["requestHash"] = request_hash
    calibration_by_entity = {item["entityType"]: item for item in calibrations}
    manifest["calibrationSetHash"] = canonical_sha256(
        [
            {
                "entityType": entity_type,
                "sha256": calibration_by_entity[entity_type]["sha256"],
            }
            for entity_type in ("node", "edge")
        ]
    )
    if historical_format == "round1-2.0":
        for field in (
            "registrySourceSha256",
            "adapterSchemaHash",
            "calibrationSetHash",
            "artifactCatalogSha256",
            "artifactCatalogHash",
            "artifactCatalogGeneration",
            "controlHash",
            "controlGeneration",
        ):
            manifest.pop(field)
    elif historical_format == "round2-2.0":
        for field in ("controlHash", "controlGeneration", "artifactCatalogHash"):
            manifest.pop(field)
    elif historical_format == "round3-2.0":
        manifest["controlHash"] = None
        manifest["controlGeneration"] = None
        manifest["artifactCatalogHash"] = None
    _write_hashed_document(manifest_path, manifest, "snapshotHash")

    state_path = run_dir / "state.json"
    state = json.loads(state_path.read_bytes())
    state["requestHash"] = request_hash
    _write_hashed_document(state_path, state, "stateHash")

    result_path = run_dir / "result.json"
    if result_path.is_file():
        result = json.loads(result_path.read_bytes())
        result["requestHash"] = request_hash
        _write_hashed_document(result_path, result, "resultHash")

    marker_path = run_dir / "success.json"
    if marker_path.is_file():
        marker = json.loads(marker_path.read_bytes())
        marker["requestHash"] = request_hash
        marker["snapshotHash"] = manifest["snapshotHash"]
        if result_path.is_file():
            marker["resultHash"] = json.loads(result_path.read_bytes())["resultHash"]
        _write_hashed_document(marker_path, marker, "markerHash")


def test_status_read_holds_the_transition_lock_through_its_snapshot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    catalog, reference, _ = _catalog(tmp_path)
    registry = _serving_registry(tmp_path, catalog=catalog)
    control = _control(tmp_path, registry, catalog)
    executor_entered = threading.Event()
    release_executor = threading.Event()

    def gated_executor(*_args: object) -> list[GovernanceFinding]:
        executor_entered.set()
        assert release_executor.wait(5)
        return []

    store = _make_test_only_run_store(
        tmp_path / "serialized-status-read",
        registry=registry,
        artifact_catalog=catalog,
        serving_control=control,
        executor=gated_executor,
    )
    receipt = store.create(_make_test_internal_create_request(reference, control))
    run_id = receipt.status.run_id
    assert executor_entered.wait(5)

    reader_entered = threading.Event()
    release_reader = threading.Event()
    reader_errors: list[BaseException] = []
    observed: list[GfmRunStatus] = []
    original_load_core = store._load_core

    def blocking_load_core(target_run_id: str):
        if threading.current_thread().name == "gfm-status-reader":
            reader_entered.set()
            assert release_reader.wait(5)
        return original_load_core(target_run_id)

    monkeypatch.setattr(store, "_load_core", blocking_load_core)

    def read_status() -> None:
        try:
            observed.append(store.get(run_id))
        except BaseException as error:  # thread failure is asserted below
            reader_errors.append(error)

    reader = threading.Thread(target=read_status, name="gfm-status-reader")
    acquired_transition_lock = False
    try:
        reader.start()
        assert reader_entered.wait(5)
        acquired_transition_lock = store._lock.acquire(blocking=False)
        if acquired_transition_lock:
            store._lock.release()
    finally:
        release_reader.set()
        reader.join(timeout=5)
        release_executor.set()

    assert not reader.is_alive()
    assert not reader_errors
    assert observed and observed[0].status == "running"
    assert acquired_transition_lock is False

    deadline = time.monotonic() + 5
    while store.get(run_id).status not in {"succeeded", "failed"}:
        assert time.monotonic() < deadline
        time.sleep(0.01)
    assert store.get(run_id).status == "succeeded"


@pytest.mark.parametrize(
    "field",
    [
        "controlHash",
        "controlGeneration",
        "registryHash",
        "registryGeneration",
        "catalogHash",
        "catalogGeneration",
        "modelVersionHash",
    ],
)
def test_restart_rejects_coherently_rehashed_control_expectation_substitution(
    tmp_path: Path, field: str
) -> None:
    catalog, reference, _ = _catalog(tmp_path)
    registry = _serving_registry(tmp_path, catalog=catalog)
    control = _control(tmp_path, registry, catalog)
    runtime_root = tmp_path / "expectation-substitution"
    store = _make_test_only_run_store(
        runtime_root,
        registry=registry,
        artifact_catalog=catalog,
        serving_control=control,
        executor=lambda *_args: [],
    )
    receipt = store.create(_make_test_internal_create_request(reference, control))
    run_id = receipt.status.run_id
    deadline = time.monotonic() + 5
    while store.get(run_id).status not in {"succeeded", "failed"}:
        assert time.monotonic() < deadline
        time.sleep(0.01)
    assert store.get(run_id).status == "succeeded"
    run_dir = runtime_root / "runs" / run_id
    request = json.loads((run_dir / "request.json").read_bytes())
    expectation = request["expectedServingControl"]
    assert isinstance(expectation, dict)
    current = expectation[field]
    expectation[field] = int(current) + 1 if isinstance(current, int) else "f" * 64
    _rewrite_persisted_request_and_hash_bindings(run_dir, request)
    substituted_bytes = {
        path.name: path.read_bytes() for path in run_dir.iterdir() if path.is_file()
    }

    restarted = _make_test_only_run_store(
        runtime_root,
        registry=registry,
        artifact_catalog=catalog,
        serving_control=control,
        executor=lambda *_args: pytest.fail("substituted run must not replay"),
    )

    assert restarted.recovery_diagnostics() == (
        {"runId": run_id, "code": "GFM_CORE_RUN_RECOVERY_INVALID"},
    )
    assert {
        path.name: path.read_bytes() for path in run_dir.iterdir() if path.is_file()
    } == substituted_bytes


@pytest.mark.parametrize(
    "historical_format",
    ["round1-2.0", "round2-2.0", "round3-2.0", "round3-2.1"],
)
def test_restart_recovers_exact_historical_terminal_run_snapshots(
    tmp_path: Path, historical_format: str
) -> None:
    catalog, reference, _ = _catalog(tmp_path)
    registry = _serving_registry(tmp_path, catalog=catalog)
    control = _control(tmp_path, registry, catalog)
    runtime_root = tmp_path / "historical-terminal"
    store = _make_test_only_run_store(
        runtime_root,
        registry=registry,
        artifact_catalog=catalog,
        serving_control=control,
        executor=lambda *_args: [],
    )
    receipt = store.create(_make_test_internal_create_request(reference, control))
    run_id = receipt.status.run_id
    deadline = time.monotonic() + 5
    while store.get(run_id).status not in {"succeeded", "failed"}:
        assert time.monotonic() < deadline
        time.sleep(0.01)
    assert store.get(run_id).status == "succeeded"

    _rewrite_run_as_historical(runtime_root, run_id, historical_format)
    run_dir = runtime_root / "runs" / run_id
    committed_bytes = {path.name: path.read_bytes() for path in run_dir.iterdir() if path.is_file()}
    restarted = _make_test_only_run_store(
        runtime_root,
        registry=registry,
        artifact_catalog=catalog,
        serving_control=control,
        executor=lambda *_args: pytest.fail("terminal historical run must not replay"),
    )

    assert restarted.recovery_diagnostics() == ()
    assert restarted.get(run_id).status == "succeeded"
    assert restarted.get_result(run_id).run_id == run_id
    assert {
        path.name: path.read_bytes() for path in run_dir.iterdir() if path.is_file()
    } == committed_bytes


@pytest.mark.parametrize(
    "historical_format",
    ["round1-2.0", "round2-2.0", "round3-2.0", "round3-2.1"],
)
def test_restart_rolls_forward_historical_committed_result_over_failed_state(
    tmp_path: Path, historical_format: str
) -> None:
    catalog, reference, _ = _catalog(tmp_path)
    registry = _serving_registry(tmp_path, catalog=catalog)
    control = _control(tmp_path, registry, catalog)
    runtime_root = tmp_path / "historical-committed"
    store = _make_test_only_run_store(
        runtime_root,
        registry=registry,
        artifact_catalog=catalog,
        serving_control=control,
        executor=lambda *_args: [],
    )
    receipt = store.create(_make_test_internal_create_request(reference, control))
    run_id = receipt.status.run_id
    deadline = time.monotonic() + 5
    while store.get(run_id).status not in {"succeeded", "failed"}:
        assert time.monotonic() < deadline
        time.sleep(0.01)
    assert store.get(run_id).status == "succeeded"
    _rewrite_run_as_historical(runtime_root, run_id, historical_format)
    run_dir = runtime_root / "runs" / run_id
    result_bytes = (run_dir / "result.json").read_bytes()
    marker_bytes = (run_dir / "success.json").read_bytes()
    state_path = run_dir / "state.json"
    state = json.loads(state_path.read_bytes())
    state.update(status="failed", progress=100, errorCode="GFM_CORE_EXECUTION_FAILED")
    _write_hashed_document(state_path, state, "stateHash")

    restarted = _make_test_only_run_store(
        runtime_root,
        registry=registry,
        artifact_catalog=catalog,
        serving_control=control,
        executor=lambda *_args: pytest.fail("committed historical run must not replay"),
    )

    assert restarted.recovery_diagnostics() == ()
    assert restarted.get(run_id).status == "succeeded"
    assert restarted.get_result(run_id).run_id == run_id
    assert (run_dir / "result.json").read_bytes() == result_bytes
    assert (run_dir / "success.json").read_bytes() == marker_bytes


@pytest.mark.parametrize(
    "historical_format",
    ["round1-2.0", "round2-2.0", "round3-2.0", "round3-2.1"],
)
def test_restart_fails_exact_historical_nonterminal_run_without_replay(
    tmp_path: Path, historical_format: str
) -> None:
    catalog, reference, _ = _catalog(tmp_path)
    registry = _serving_registry(tmp_path, catalog=catalog)
    control = _control(tmp_path, registry, catalog)
    runtime_root = tmp_path / "historical-interrupted"
    gate = threading.Event()
    store = _make_test_only_run_store(
        runtime_root,
        registry=registry,
        artifact_catalog=catalog,
        serving_control=control,
        executor=lambda *_args: (gate.wait(5), [])[1],
    )
    receipt = store.create(_make_test_internal_create_request(reference, control))
    run_id = receipt.status.run_id
    deadline = time.monotonic() + 5
    while store.get(run_id).status != "running":
        assert time.monotonic() < deadline
        time.sleep(0.01)

    _rewrite_run_as_historical(runtime_root, run_id, historical_format)
    try:
        restarted = _make_test_only_run_store(
            runtime_root,
            registry=registry,
            artifact_catalog=catalog,
            serving_control=control,
            executor=lambda *_args: pytest.fail("historical run must not replay"),
        )

        assert restarted.recovery_diagnostics() == ()
        recovered = restarted.get(run_id)
        assert recovered.status == "failed"
        assert recovered.error_code == "GFM_CORE_RUN_INTERRUPTED"
    finally:
        gate.set()


@pytest.mark.parametrize("case", ["request-snapshot-version", "hybrid-current", "controlled-null"])
def test_restart_quarantines_hybrid_or_mismatched_historical_snapshots(
    tmp_path: Path, case: str
) -> None:
    catalog, reference, _ = _catalog(tmp_path)
    registry = _serving_registry(tmp_path, catalog=catalog)
    control = _control(tmp_path, registry, catalog)
    runtime_root = tmp_path / "historical-invalid"
    store = _make_test_only_run_store(
        runtime_root,
        registry=registry,
        artifact_catalog=catalog,
        serving_control=control,
        executor=lambda *_args: [],
    )
    receipt = store.create(_make_test_internal_create_request(reference, control))
    run_id = receipt.status.run_id
    deadline = time.monotonic() + 5
    while store.get(run_id).status not in {"succeeded", "failed"}:
        assert time.monotonic() < deadline
        time.sleep(0.01)
    assert store.get(run_id).status == "succeeded"
    run_dir = runtime_root / "runs" / run_id

    if case == "request-snapshot-version":
        current_request = (run_dir / "request.json").read_bytes()
        _rewrite_run_as_historical(runtime_root, run_id, "round3-2.0")
        (run_dir / "request.json").write_bytes(current_request)
    elif case == "controlled-null":
        _rewrite_run_as_historical(runtime_root, run_id, "round3-2.1")
        manifest_path = run_dir / "manifest.json"
        manifest = json.loads(manifest_path.read_bytes())
        manifest["controlHash"] = None
        _write_hashed_document(manifest_path, manifest, "snapshotHash")
    else:
        manifest_path = run_dir / "manifest.json"
        manifest = json.loads(manifest_path.read_bytes())
        manifest.pop("controlSourceSha256")
        _write_hashed_document(manifest_path, manifest, "snapshotHash")
    invalid_bytes = {path.name: path.read_bytes() for path in run_dir.iterdir() if path.is_file()}

    restarted = _make_test_only_run_store(
        runtime_root,
        registry=registry,
        artifact_catalog=catalog,
        serving_control=control,
        executor=lambda *_args: pytest.fail("invalid historical run must not replay"),
    )

    assert restarted.recovery_diagnostics() == (
        {"runId": run_id, "code": "GFM_CORE_RUN_RECOVERY_INVALID"},
    )
    assert {
        path.name: path.read_bytes() for path in run_dir.iterdir() if path.is_file()
    } == invalid_bytes
