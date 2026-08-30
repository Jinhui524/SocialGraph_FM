from __future__ import annotations

import copy
import ctypes
import hashlib
import importlib
import json
import os
import stat
import threading
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import unquote_to_bytes

import httpx
import pytest
from pydantic import BaseModel

from app.config import Settings
from app.dataset_imports import DatasetImportService
from app.gfm_client import CoreGateway, GfmProxyError, CoreRunBindingStore
from app.gfm_hashing import canonical_json, canonical_sha256
from app.gfm_core_schemas import (
    CoreRunBinding,
    CoreRunBindingAnchor,
    CoreCapabilitiesResponse,
    CoreRunResult,
    CoreRunStatus,
    CoreFinding,
    CoreInternalCreateRunReceipt,
    CoreInternalErrorEnvelope,
)
from app.main import create_app

from .test_atomic_handoff import _request as graph_handoff_request

HASHES = {letter: letter * 64 for letter in "123456789abcdef"}

_IGNORED_SCHEMA_ANNOTATIONS = {"title", "description", "examples"}
_UNORDERED_SCHEMA_LISTS = {"required", "enum", "type", "allOf", "anyOf", "oneOf"}
_SCHEMA_CHILD_LISTS = {"allOf", "anyOf", "oneOf", "prefixItems"}
_SCHEMA_CHILD_MAPS = {"$defs", "dependentSchemas", "patternProperties", "properties"}
_SCHEMA_CHILD_VALUES = {
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
_PUBLIC_BOUNDARY_MODELS: dict[str, type[BaseModel]] = {
    "status": CoreRunStatus,
    "result": CoreRunResult,
    "finding": CoreFinding,
    "error": CoreInternalErrorEnvelope,
    "capabilities": CoreCapabilitiesResponse,
}


def _pointer_component(encoded: str) -> str:
    position = 0
    while position < len(encoded):
        if encoded[position] != "~":
            position += 1
            continue
        if position + 1 == len(encoded) or encoded[position + 1] not in "01":
            raise ValueError(f"invalid JSON Pointer escape in {encoded!r}")
        position += 2
    return encoded.replace("~1", "/").replace("~0", "~")


def _uri_fragment_text(encoded: str) -> str:
    hexdigits = "0123456789abcdefABCDEF"
    cursor = 0
    while cursor < len(encoded):
        if encoded[cursor] != "%":
            cursor += 1
            continue
        pair = encoded[cursor + 1 : cursor + 3]
        if len(pair) != 2 or any(character not in hexdigits for character in pair):
            raise ValueError(f"malformed percent escape in URI fragment: {encoded!r}")
        cursor += 3
    try:
        return unquote_to_bytes(encoded).decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError(f"invalid UTF-8 URI fragment: {encoded!r}") from exc


def normalize_boundary_schema(schema: dict[str, Any]) -> dict[str, Any]:
    """Return a complete, reference-free behavioral schema for neutral comparison."""

    source = copy.deepcopy(schema)

    def copy_instance(value: Any) -> Any:
        if isinstance(value, list):
            return [copy_instance(item) for item in value]
        if isinstance(value, dict):
            copied: dict[str, Any] = {}
            for name in sorted(value):
                copied[name] = copy_instance(value[name])
            return copied
        return value

    def referenced_value(reference: str) -> tuple[str, Any]:
        if not reference.startswith("#"):
            raise ValueError(f"external $ref is forbidden: {reference}")
        fragment = _uri_fragment_text(reference[1:])
        if fragment and not fragment.startswith("/"):
            raise ValueError(f"local $ref must be a JSON Pointer: {reference}")
        value: Any = source
        components = fragment[1:].split("/") if fragment.startswith("/") else []
        for component in components:
            key = _pointer_component(component)
            if isinstance(value, dict):
                if key not in value:
                    raise ValueError(f"missing JSON Pointer target: {reference}")
                value = value[key]
                continue
            if isinstance(value, list) and key.isdigit() and int(key) < len(value):
                value = value[int(key)]
                continue
            raise ValueError(f"missing JSON Pointer target: {reference}")
        return fragment, value

    def visit_schema(node: Any, chain: tuple[str, ...] = ()) -> Any:
        if not isinstance(node, dict):
            return copy_instance(node)

        reference = node.get("$ref")
        if reference is not None:
            if not isinstance(reference, str):
                raise ValueError("$ref must be a string")
            pointer, target_node = referenced_value(reference)
            if pointer in chain:
                raise ValueError(f"cyclic local $ref: {' -> '.join((*chain, pointer))}")
            referenced = visit_schema(target_node, chain=(*chain, pointer))
            sibling_node = {
                name: child
                for name, child in node.items()
                if name not in _IGNORED_SCHEMA_ANNOTATIONS | {"$defs", "$ref"}
            }
            if not sibling_node:
                return referenced
            outer = visit_schema(sibling_node, chain=chain)
            if not isinstance(outer, dict):
                raise TypeError("$ref siblings must normalize to an object")
            prior_branches = outer.get("allOf", [])
            if not isinstance(prior_branches, list):
                raise ValueError("allOf must be an array")
            outer["allOf"] = sorted([*prior_branches, referenced], key=canonical_json)
            return {name: outer[name] for name in sorted(outer)}

        normalized: dict[str, Any] = {}
        for name in sorted(node):
            if name in _IGNORED_SCHEMA_ANNOTATIONS or name == "$defs":
                continue
            child = node[name]
            if name in _SCHEMA_CHILD_MAPS and isinstance(child, dict):
                child_map: dict[str, Any] = {}
                for field_name in sorted(child):
                    child_map[field_name] = visit_schema(child[field_name], chain=chain)
                normalized[name] = child_map
            elif name in _SCHEMA_CHILD_VALUES:
                normalized[name] = visit_schema(child, chain=chain)
            elif name in _SCHEMA_CHILD_LISTS and isinstance(child, list):
                schemas = [visit_schema(item, chain=chain) for item in child]
                if name in _UNORDERED_SCHEMA_LISTS:
                    schemas.sort(key=canonical_json)
                normalized[name] = schemas
            elif name == "dependentRequired" and isinstance(child, dict):
                requirements: dict[str, Any] = {}
                for property_name in sorted(child):
                    property_requirements = copy_instance(child[property_name])
                    if isinstance(property_requirements, list):
                        property_requirements.sort(key=canonical_json)
                    requirements[property_name] = property_requirements
                normalized[name] = requirements
            elif name in {"const", "default"}:
                normalized[name] = copy_instance(child)
            elif name == "enum" and isinstance(child, list):
                choices = [copy_instance(choice) for choice in child]
                choices.sort(key=canonical_json)
                normalized[name] = choices
            elif name in {"required", "type"} and isinstance(child, list):
                values = [copy_instance(value) for value in child]
                values.sort(key=canonical_json)
                normalized[name] = values
            else:
                normalized[name] = copy_instance(child)
        return normalized

    result = visit_schema(source)
    if not isinstance(result, dict):
        raise TypeError("root JSON Schema must normalize to an object")
    return result


def _normalized_boundary_manifest() -> dict[str, Any]:
    roots: dict[str, Any] = {}
    for root_name in sorted(_PUBLIC_BOUNDARY_MODELS):
        raw_schema = _PUBLIC_BOUNDARY_MODELS[root_name].model_json_schema(by_alias=True)
        roots[root_name] = normalize_boundary_schema(raw_schema)
    return {
        "schemaVersion": "socialgraph-fm.core-boundary-manifest/2.0",
        "normalizationPolicy": {
            "annotationsStripped": sorted(_IGNORED_SCHEMA_ANNOTATIONS),
            "instanceValues": "preserved recursively without schema-key reinterpretation",
            "localRefs": "inlined as allOf branches; siblings remain at outer schema scope",
            "mappingKeys": "Unicode code-point order",
            "omittedAfterInlining": ["$defs"],
            "orderSensitiveArrays": "preserved, including raw instance arrays and prefixItems",
            "setLikeArraysSorted": sorted(
                {*_UNORDERED_SCHEMA_LISTS, "dependentRequired values"}
            ),
        },
        "roots": roots,
    }


def _locate_schema_property(schema: dict[str, Any], field_name: str) -> dict[str, Any]:
    found: list[dict[str, Any]] = []
    pending: list[Any] = [schema]
    while pending:
        candidate = pending.pop()
        if isinstance(candidate, list):
            pending.extend(candidate)
            continue
        if not isinstance(candidate, dict):
            continue
        properties = candidate.get("properties")
        if isinstance(properties, dict) and isinstance(properties.get(field_name), dict):
            found.append(properties[field_name])
        pending.extend(candidate.values())
    assert len(found) == 1, f"expected one {field_name!r} property, found {len(found)}"
    return found[0]


def test_api_complete_neutral_behavioral_boundary_manifest_is_canonical() -> None:
    artifact_bytes = (
        Path(__file__).parents[3] / "contracts" / "core-inference-boundaries.json"
    ).read_bytes()
    expected = json.loads(artifact_bytes)
    actual = _normalized_boundary_manifest()

    assert actual == expected
    assert canonical_json(actual).encode("utf-8") == artifact_bytes


def test_api_boundary_normalizer_detects_extras_policy_drift() -> None:
    original = CoreRunStatus.model_json_schema(by_alias=True)
    changed = copy.deepcopy(original)
    assert changed["additionalProperties"] is False
    changed["additionalProperties"] = True

    assert normalize_boundary_schema(changed) != normalize_boundary_schema(original)


def test_api_boundary_normalizer_detects_created_at_format_drift() -> None:
    original = CoreRunStatus.model_json_schema(by_alias=True)
    changed = copy.deepcopy(original)
    created_at = _locate_schema_property(changed, "createdAt")
    assert created_at.pop("format") == "date-time"

    assert normalize_boundary_schema(changed) != normalize_boundary_schema(original)


@pytest.mark.parametrize("replacement", [pytest.param("remove", id="absent"), None, "DRIFT"])
def test_api_boundary_normalizer_distinguishes_error_code_default_states(
    replacement: str | None,
) -> None:
    original = CoreRunStatus.model_json_schema(by_alias=True)
    changed = copy.deepcopy(original)
    error_code = _locate_schema_property(changed, "errorCode")
    assert "default" in error_code and error_code["default"] is None
    if replacement == "remove":
        error_code.pop("default")
    else:
        error_code["default"] = replacement

    expected_equal = replacement is None
    assert (normalize_boundary_schema(changed) == normalize_boundary_schema(original)) is expected_equal


def test_api_boundary_normalizer_detects_edge_identity_null_union_drift() -> None:
    original = CoreFinding.model_json_schema(by_alias=True)
    changed = copy.deepcopy(original)
    edge_identity = _locate_schema_property(changed, "edgeIdentity")
    options = edge_identity["anyOf"]
    assert isinstance(options, list)
    without_null = [
        option
        for option in options
        if not (isinstance(option, dict) and option.get("type") == "null")
    ]
    assert len(without_null) + 1 == len(options)
    edge_identity["anyOf"] = without_null

    assert normalize_boundary_schema(changed) != normalize_boundary_schema(original)


def test_api_boundary_normalizer_keeps_ref_sibling_and_decodes_pointer_tokens() -> None:
    original = {
        "type": "object",
        "$defs": {"named/schema~v2": {"type": "string", "minLength": 1, "examples": ["x"]}},
        "properties": {
            "value": {"maxLength": 8, "$ref": "#/$defs/named~1schema~0v2"},
        },
    }
    normalized = normalize_boundary_schema(original)
    changed = copy.deepcopy(original)
    _locate_schema_property(changed, "value")["maxLength"] = 9

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
    assert normalize_boundary_schema(changed) != normalized


def test_api_boundary_normalizer_keeps_annotation_named_property_keys() -> None:
    schema = {
        "title": "root annotation",
        "description": "root annotation",
        "type": "object",
        "required": ["examples", "description", "title"],
        "properties": {
            "title": {"type": "string", "title": "field annotation"},
            "description": {"type": "integer", "description": "field annotation"},
            "examples": {"type": "array", "items": {"type": "string"}},
        },
    }

    assert normalize_boundary_schema(schema) == {
        "properties": {
            "description": {"type": "integer"},
            "examples": {"items": {"type": "string"}, "type": "array"},
            "title": {"type": "string"},
        },
        "required": ["description", "examples", "title"],
        "type": "object",
    }


@pytest.mark.parametrize("value_keyword", ["default", "const", "enum"])
def test_api_boundary_normalizer_does_not_reinterpret_raw_instance_objects(
    value_keyword: str,
) -> None:
    instance = {
        "title": "kept",
        "description": "kept",
        "examples": ["z", "a"],
        "required": ["z", "a"],
        "nested": [{"enum": ["z", "a"], "required": ["right", "left"]}],
    }
    schema: dict[str, Any] = {
        value_keyword: [instance] if value_keyword == "enum" else instance
    }
    expected = copy.deepcopy(schema)
    changed = copy.deepcopy(schema)
    changed_instance = (
        changed[value_keyword][0] if value_keyword == "enum" else changed[value_keyword]
    )
    assert isinstance(changed_instance, dict)
    changed_instance["nested"][0]["enum"][0] = "changed"

    assert normalize_boundary_schema(schema) == expected
    assert normalize_boundary_schema(changed) != normalize_boundary_schema(schema)


def test_api_boundary_normalizer_keeps_ref_siblings_at_outer_schema_scope() -> None:
    referenced = {
        "type": "object",
        "properties": {"value": {"type": "string"}},
        "required": ["value"],
    }
    schema = {
        "$ref": "#/$defs/record",
        "$defs": {"record": referenced},
        "allOf": [{"properties": {"tag": {"type": "string"}}}],
        "unevaluatedProperties": False,
    }

    normalized = normalize_boundary_schema(schema)

    assert normalized["unevaluatedProperties"] is False
    assert normalized["allOf"] == sorted(
        [referenced, {"properties": {"tag": {"type": "string"}}}], key=canonical_json
    )


def test_api_boundary_normalizer_percent_decodes_local_ref_fragment() -> None:
    schema = {"$ref": "#/$defs/a%20b", "$defs": {"a b": {"const": "decoded"}}}

    assert normalize_boundary_schema(schema) == {"const": "decoded"}


@pytest.mark.parametrize("reference", ["#/$defs/bad%", "#/$defs/bad%Q1", "#/$defs/%FF"])
def test_api_boundary_normalizer_rejects_bad_percent_or_utf8_fragment(reference: str) -> None:
    with pytest.raises(ValueError, match="URI fragment"):
        normalize_boundary_schema({"$ref": reference, "$defs": {}})


def test_api_boundary_normalizer_contextually_sorts_dependent_required_only() -> None:
    original: dict[str, Any] = {
        "type": "object",
        "dependentRequired": {"account": ["name", "email"]},
        "default": {
            "required": ["z", "a"],
            "enum": ["z", "a"],
            "nested": [{"required": ["right", "left"]}],
        },
    }
    reordered = copy.deepcopy(original)
    reordered["dependentRequired"]["account"] = ["email", "name"]

    normalized = normalize_boundary_schema(original)
    assert normalized == normalize_boundary_schema(reordered)
    assert normalized["default"] == original["default"]


@pytest.mark.parametrize(
    ("schema", "expected_message"),
    [
        ({"$ref": "https://example.invalid/schema.json"}, "external"),
        ({"$ref": "#/$defs/not-there"}, "missing JSON Pointer"),
        (
            {"$ref": "#/$defs/Loop", "$defs": {"Loop": {"$ref": "#/$defs/Loop"}}},
            "cyclic local",
        ),
    ],
)
def test_api_boundary_normalizer_rejects_unresolvable_refs(
    schema: dict[str, Any], expected_message: str
) -> None:
    with pytest.raises(ValueError, match=expected_message):
        normalize_boundary_schema(schema)


class RecordingGfmClient:
    def __init__(self, capabilities: dict[str, Any]) -> None:
        self.capability_payload = capabilities
        self.created: dict[str, Any] | None = None
        self.create_response: dict[str, Any] | Callable[[dict[str, Any]], dict[str, Any]] = {}
        self.status_response: dict[str, Any] = {}
        self.result_response: dict[str, Any] = {}

    async def core_capabilities(self) -> dict[str, Any]:
        return copy.deepcopy(self.capability_payload)

    async def create_core_run(self, payload: dict[str, Any]) -> dict[str, Any]:
        self.created = payload
        response = (
            self.create_response(payload)
            if callable(self.create_response)
            else self.create_response
        )
        return copy.deepcopy(response)

    async def get_core_run(self, _run_id: str) -> dict[str, Any]:
        return copy.deepcopy(self.status_response)

    async def get_core_result(self, _run_id: str) -> dict[str, Any]:
        return copy.deepcopy(self.result_response)


@dataclass
class ServingFixture:
    settings: Settings
    root: Path
    control_path: Path
    registry_path: Path
    catalog_path: Path
    manifest_path: Path
    model: dict[str, Any]
    catalog: dict[str, Any]
    manifest: dict[str, Any]
    capabilities: dict[str, Any]

    def publish(
        self,
        *,
        control_generation: int = 1,
        registry_generation: int = 1,
        catalog_generation: int = 1,
        recompute_model_hash: bool = True,
    ) -> None:
        if recompute_model_hash:
            self.model["modelVersionHash"] = canonical_sha256(
                {
                    key: value
                    for key, value in self.model.items()
                    if key not in {"modelVersionHash", "state"}
                }
            )
        self.manifest_path.write_text(
            json.dumps(self.manifest, separators=(",", ":")), encoding="utf-8"
        )
        self.model["checkpoint"]["servingManifestSha256"] = hashlib.sha256(
            self.manifest_path.read_bytes()
        ).hexdigest()
        if recompute_model_hash:
            self.model["modelVersionHash"] = canonical_sha256(
                {
                    key: value
                    for key, value in self.model.items()
                    if key not in {"modelVersionHash", "state"}
                }
            )
        registry = {
            "schemaVersion": "socialgraph-fm.core-serving-registry/2.0",
            "generation": registry_generation,
            "models": [self.model],
        }
        self.catalog["generation"] = catalog_generation
        self.registry_path.write_text(
            json.dumps(registry, separators=(",", ":")), encoding="utf-8"
        )
        self.catalog_path.write_text(
            json.dumps(self.catalog, separators=(",", ":")), encoding="utf-8"
        )
        registry_bytes = self.registry_path.read_bytes()
        catalog_bytes = self.catalog_path.read_bytes()
        control: dict[str, Any] = {
            "schemaVersion": "socialgraph-fm.core-serving-control/1.0",
            "generation": control_generation,
            "registry": {
                "relativePath": self.registry_path.name,
                "sha256": hashlib.sha256(registry_bytes).hexdigest(),
                "semanticHash": canonical_sha256(registry),
                "generation": registry_generation,
            },
            "catalog": {
                "relativePath": self.catalog_path.name,
                "sha256": hashlib.sha256(catalog_bytes).hexdigest(),
                "semanticHash": canonical_sha256(self.catalog),
                "generation": catalog_generation,
            },
        }
        control["controlHash"] = canonical_sha256(control)
        replacement = self.control_path.with_suffix(".replacement")
        replacement.write_text(
            json.dumps(control, separators=(",", ":")), encoding="utf-8"
        )
        os.replace(replacement, self.control_path)
        self.capabilities.clear()
        self.capabilities.update(self.expected_capabilities(control, registry))

    def expected_capabilities(
        self,
        control: dict[str, Any] | None = None,
        registry: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        control = control or json.loads(self.control_path.read_bytes())
        registry = registry or json.loads(self.registry_path.read_bytes())
        models = [
            {
                "modelVersionId": item["modelVersionId"],
                "modelVersionHash": item["modelVersionHash"],
                "state": item["state"],
                "tasks": item["tasks"],
                "graphSchemaVersions": item["graphSchemaVersions"],
                "graphFeatureContractHash": item["graphFeatureContractHash"],
                "taskBindings": [
                    {
                        "taskId": head["taskId"],
                        "entityType": binding["entityType"],
                        "confidenceKind": binding["confidenceKind"],
                        "calibrationVersion": binding["calibrationVersion"],
                        "method": binding["calibrationMethod"],
                        "calibrationArtifactHash": binding[
                            "calibrationArtifactHash"
                        ],
                        "calibrationProtocolHash": binding[
                            "calibrationProtocolHash"
                        ],
                        "adapterDomain": binding["adapterDomain"],
                        "adapterSchemaHash": binding["adapterSchemaHash"],
                        "adapterStateHash": binding["adapterStateHash"],
                        "featureContractHash": binding[
                            "graphFeatureContractHash"
                        ],
                    }
                    for head in item["taskHeads"]
                    for binding in head["calibrations"]
                ],
                "maxNodes": item["maxNodes"],
                "maxEdges": item["maxEdges"],
            }
            for item in registry["models"]
        ]
        accepted = [
            item for item in models if item["state"] in {"accepted", "servingReady"}
        ]
        serving = [item for item in accepted if item["state"] == "servingReady"]
        return {
            "schemaVersion": "socialgraph-fm.core-capabilities/2.0",
            "controlHash": control["controlHash"],
            "controlGeneration": control["generation"],
            "registryHash": control["registry"]["semanticHash"],
            "registryGeneration": control["registry"]["generation"],
            "catalogHash": control["catalog"]["semanticHash"],
            "catalogGeneration": control["catalog"]["generation"],
            "servingReady": bool(serving),
            "models": models,
            "tasks": sorted({task for item in accepted for task in item["tasks"]}),
            "readiness": {
                "modelValidated": bool(accepted),
                "coreServingReady": bool(serving),
            },
        }


@pytest.fixture
def serving_fixture(unconfigured_settings: Settings) -> ServingFixture:
    service = DatasetImportService(unconfigured_settings)
    request, _, _ = graph_handoff_request(service, "graph-v1")
    service.commit_graph_handoff(request)
    binding = service.store.resolve_graph_version_binding("graph-v1")
    assert binding is not None
    artifact = service.store.get_artifact(binding.artifact_id)
    assert artifact is not None

    root = Path(unconfigured_settings.dataset_storage_root)
    feature_contract = {
        "schemaVersion": "socialgraph-fm.core-graph-feature-contract/2.0",
        "nodeFeatures": [],
        "structuralFeatureNames": [],
    }
    catalog = {
        "schemaVersion": "socialgraph-fm.core-serving-graph-catalog/1.0",
        "generation": 1,
        "artifacts": [
            {
                "artifactId": binding.artifact_id,
                "artifactHash": artifact.content_hash,
                "bundleSha256": HASHES["d"],
                "relativePath": "bundles/graph-v1.json",
                "graphVersionId": binding.graph_version_id,
                "sourceGraphFactHash": binding.graph_fact_hash,
                "graphVersionHash": HASHES["e"],
                "graphSchemaVersion": "socialgraph-fm.core-graph-bundle/2.0",
                "featureContract": feature_contract,
                "featureContractHash": canonical_sha256(feature_contract),
                "nodeCount": artifact.profile.node_count or 0,
                "edgeCount": artifact.profile.edge_count or 0,
            }
        ],
    }
    task_heads = [
        {
            "taskId": "core.risk_and_trust_review",
            "kind": "risk-and-trust",
            "nodeOutputIndex": 1,
            "calibrations": [
                {
                    "entityType": "node",
                    "confidenceKind": "binary-calibration",
                    "calibrationVersion": "risk-node-calibration/1",
                    "calibrationMethod": "sigmoid",
                    "calibrationArtifactHash": HASHES["7"],
                    "calibrationRelativePath": "calibration/risk-node.json",
                    "calibrationSha256": HASHES["8"],
                    "calibrationProtocolHash": HASHES["9"],
                    "adapterDomain": "risk-node",
                    "adapterSchemaHash": HASHES["e"],
                    "adapterStateHash": HASHES["f"],
                    "graphFeatureContractHash": canonical_sha256(feature_contract),
                },
                {
                    "entityType": "edge",
                    "confidenceKind": "binary-calibration",
                    "calibrationVersion": "risk-edge-calibration/1",
                    "calibrationMethod": "sigmoid",
                    "calibrationArtifactHash": HASHES["4"],
                    "calibrationRelativePath": "calibration/risk-edge.json",
                    "calibrationSha256": HASHES["5"],
                    "calibrationProtocolHash": HASHES["6"],
                    "adapterDomain": "risk-edge",
                    "adapterSchemaHash": HASHES["d"],
                    "adapterStateHash": HASHES["c"],
                    "graphFeatureContractHash": canonical_sha256(feature_contract),
                },
            ],
        }
    ]
    manifest = {
        "schemaVersion": "socialgraph-fm.core-serving-checkpoint-manifest/1.1",
        "task4CheckpointSha256": HASHES["a"],
        "accepted": True,
        "promotable": True,
        "modelStateHash": HASHES["b"],
        "adapterStateHash": HASHES["c"],
        "adapterSchemaHash": HASHES["d"],
        "adapterDomain": "risk-edge",
        "nodeClasses": 2,
        "multiHotBuckets": 32,
        "adapterBindings": [
            {
                "adapterDomain": "risk-edge",
                "adapterSchemaHash": HASHES["d"],
                "adapterStateHash": HASHES["c"],
                "multiHotBuckets": 32,
            },
            {
                "adapterDomain": "risk-node",
                "adapterSchemaHash": HASHES["e"],
                "adapterStateHash": HASHES["f"],
                "multiHotBuckets": 32,
            },
        ],
        "taskHeads": copy.deepcopy(task_heads),
    }
    model: dict[str, Any] = {
        "modelVersionId": "socialgraph-fm-core/review",
        "modelVersionHash": HASHES["1"],
        "state": "servingReady",
        "checkpoint": {
            "relativePath": "checkpoints/model.pt",
            "sha256": HASHES["a"],
            "servingManifestRelativePath": "model.serving.json",
            "servingManifestSha256": HASHES["1"],
            "bindings": {
                "configHash": HASHES["1"],
                "dataHash": HASHES["2"],
                "codeHash": HASHES["3"],
                "environmentHash": HASHES["4"],
            },
            "adapterDomain": "risk-edge",
            "nodeClasses": 2,
            "multiHotBuckets": 32,
        },
        "taskHeads": task_heads,
        "tasks": ["core.risk_and_trust_review"],
        "graphSchemaVersions": ["socialgraph-fm.core-graph-bundle/2.0"],
        "graphFeatureContractHash": canonical_sha256(
            [
                {
                    "taskId": head["taskId"],
                    "entityType": binding["entityType"],
                    "featureContractHash": binding["graphFeatureContractHash"],
                }
                for head in task_heads
                for binding in head["calibrations"]
            ]
        ),
        "maxNodes": 1000,
        "maxEdges": 5000,
    }
    binding_root = root / "api-run-bindings"
    settings = unconfigured_settings.model_copy(
        update={
            "gfm_core_serving_control_file": str(root / "serving-control.json"),
            "gfm_core_run_binding_root": str(binding_root),
        }
    )
    fixture = ServingFixture(
        settings=settings,
        root=root,
        control_path=root / "serving-control.json",
        registry_path=root / "serving-registry.json",
        catalog_path=root / "serving-catalog.json",
        manifest_path=root / "model.serving.json",
        model=model,
        catalog=catalog,
        manifest=manifest,
        capabilities={},
    )
    fixture.publish()
    return fixture


def _public_request() -> dict[str, Any]:
    return {
        "schemaVersion": "socialgraph-fm.core-run-request/2.0",
        "graphVersionId": "graph-v1",
        "taskId": "core.risk_and_trust_review",
        "targetScope": {"kind": "risk-review", "nodeIds": ["a"], "edgeIds": []},
        "modelVersionId": "socialgraph-fm-core/review",
        "parameters": {"kind": "risk-and-trust", "topKSimilarCases": 3},
    }


async def _responses(
    fixture: ServingFixture, fake: RecordingGfmClient
) -> tuple[httpx.Response, httpx.Response]:
    app = create_app(fixture.settings, gfm_client=fake)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(
        transport=transport, base_url="http://testserver"
    ) as client:
        capabilities = await client.get("/api/v1/gfm/capabilities")
        created = await client.post("/api/v1/gfm/runs", json=_public_request())
    return capabilities, created


def _rewrite_bound_document(
    fixture: ServingFixture,
    kind: str,
    mutate: Callable[[dict[str, Any]], None],
) -> None:
    path = fixture.registry_path if kind == "registry" else fixture.catalog_path
    payload = json.loads(path.read_bytes())
    mutate(payload)
    path.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")
    control = json.loads(fixture.control_path.read_bytes())
    content = path.read_bytes()
    control[kind]["sha256"] = hashlib.sha256(content).hexdigest()
    control[kind]["semanticHash"] = canonical_sha256(payload)
    control[kind]["generation"] = payload["generation"]
    control["controlHash"] = canonical_sha256(
        {key: value for key, value in control.items() if key != "controlHash"}
    )
    fixture.control_path.write_text(
        json.dumps(control, separators=(",", ":")), encoding="utf-8"
    )
    fixture.capabilities.clear()
    fixture.capabilities.update(
        fixture.expected_capabilities(
            control,
            payload if kind == "registry" else None,
        )
    )


@pytest.mark.anyio
@pytest.mark.parametrize(
    "case",
    [
        "generic-registry",
        "duplicate-model",
        "duplicate-artifact",
        "duplicate-feature",
        "raw-registry-hash",
        "semantic-registry-hash",
        "registry-generation",
        "raw-catalog-hash",
        "semantic-catalog-hash",
        "catalog-generation",
        "manifest-identity",
    ],
)
async def test_strict_local_metadata_failures_reject_capabilities_and_create(
    serving_fixture: ServingFixture, case: str
) -> None:
    if case == "generic-registry":
        _rewrite_bound_document(
            serving_fixture, "registry", lambda value: value.update(untrusted={})
        )
    elif case == "duplicate-model":
        _rewrite_bound_document(
            serving_fixture,
            "registry",
            lambda value: value["models"].append(copy.deepcopy(value["models"][0])),
        )
    elif case == "duplicate-artifact":
        _rewrite_bound_document(
            serving_fixture,
            "catalog",
            lambda value: value["artifacts"].append(
                copy.deepcopy(value["artifacts"][0])
            ),
        )
    elif case == "duplicate-feature":

        def duplicate_feature(value: dict[str, Any]) -> None:
            contract = value["artifacts"][0]["featureContract"]
            contract["nodeFeatures"] = [
                {"kind": "numeric", "name": "duplicate"},
                {"kind": "categorical", "name": "duplicate"},
            ]
            value["artifacts"][0]["featureContractHash"] = canonical_sha256(contract)

        _rewrite_bound_document(serving_fixture, "catalog", duplicate_feature)
    elif case in {
        "raw-registry-hash",
        "semantic-registry-hash",
        "registry-generation",
        "raw-catalog-hash",
        "semantic-catalog-hash",
        "catalog-generation",
    }:
        control = json.loads(serving_fixture.control_path.read_bytes())
        target, field, replacement = {
            "raw-registry-hash": ("registry", "sha256", HASHES["f"]),
            "semantic-registry-hash": ("registry", "semanticHash", HASHES["f"]),
            "registry-generation": ("registry", "generation", 2),
            "raw-catalog-hash": ("catalog", "sha256", HASHES["f"]),
            "semantic-catalog-hash": ("catalog", "semanticHash", HASHES["f"]),
            "catalog-generation": ("catalog", "generation", 2),
        }[case]
        control[target][field] = replacement
        control["controlHash"] = canonical_sha256(
            {key: value for key, value in control.items() if key != "controlHash"}
        )
        serving_fixture.control_path.write_text(
            json.dumps(control, separators=(",", ":")), encoding="utf-8"
        )
        serving_fixture.capabilities.clear()
        serving_fixture.capabilities.update(
            serving_fixture.expected_capabilities(control)
        )
    else:
        serving_fixture.manifest["task4CheckpointSha256"] = HASHES["f"]
        serving_fixture.publish()

    fake = RecordingGfmClient(serving_fixture.capabilities)
    capabilities, created = await _responses(serving_fixture, fake)

    assert capabilities.status_code == 503
    assert created.status_code == 503
    assert fake.created is None


@pytest.mark.anyio
@pytest.mark.parametrize("case", ["identity", "model-projection"])
async def test_local_and_remote_capability_mismatch_rejects_both_public_paths(
    serving_fixture: ServingFixture, case: str
) -> None:
    remote = copy.deepcopy(serving_fixture.capabilities)
    if case == "identity":
        remote["registryHash"] = HASHES["f"]
    else:
        remote["models"][0]["maxNodes"] += 1
    fake = RecordingGfmClient(remote)

    capabilities, created = await _responses(serving_fixture, fake)

    assert capabilities.status_code == 503
    assert created.status_code == 503
    assert fake.created is None


@pytest.mark.anyio
async def test_api_high_water_rejects_rollback_after_restart(
    serving_fixture: ServingFixture,
) -> None:
    serving_fixture.publish(
        control_generation=2, registry_generation=2, catalog_generation=2
    )
    first_fake = RecordingGfmClient(serving_fixture.capabilities)
    first_app = create_app(serving_fixture.settings, gfm_client=first_fake)
    first_transport = httpx.ASGITransport(app=first_app)
    async with httpx.AsyncClient(
        transport=first_transport, base_url="http://testserver"
    ) as client:
        accepted = await client.get("/api/v1/gfm/capabilities")
    assert accepted.status_code == 200

    serving_fixture.publish(
        control_generation=1, registry_generation=1, catalog_generation=1
    )
    restarted_fake = RecordingGfmClient(serving_fixture.capabilities)
    capabilities, created = await _responses(serving_fixture, restarted_fake)

    assert capabilities.status_code == 503
    assert created.status_code == 503
    assert restarted_fake.created is None


@pytest.mark.anyio
async def test_api_high_water_rejects_same_generation_fork(
    serving_fixture: ServingFixture,
) -> None:
    first_fake = RecordingGfmClient(serving_fixture.capabilities)
    first_app = create_app(serving_fixture.settings, gfm_client=first_fake)
    first_transport = httpx.ASGITransport(app=first_app)
    async with httpx.AsyncClient(
        transport=first_transport, base_url="http://testserver"
    ) as client:
        accepted = await client.get("/api/v1/gfm/capabilities")
    assert accepted.status_code == 200

    serving_fixture.model["maxNodes"] += 1
    serving_fixture.publish()
    fork_fake = RecordingGfmClient(serving_fixture.capabilities)
    capabilities, created = await _responses(serving_fixture, fork_fake)

    assert capabilities.status_code == 503
    assert created.status_code == 503
    assert fork_fake.created is None


@pytest.mark.anyio
async def test_api_control_capture_rejects_coordinated_aba_replacement(
    serving_fixture: ServingFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = RecordingGfmClient(serving_fixture.capabilities)
    app = create_app(serving_fixture.settings, gfm_client=fake)
    assert hasattr(app.state, "core_serving_control_store")
    module = importlib.import_module("app.gfm_core_serving_control")
    original = serving_fixture.control_path.read_bytes()

    def replace_with_aba(_stage: str) -> None:
        alternate = json.loads(original)
        alternate["generation"] += 1
        alternate["controlHash"] = canonical_sha256(
            {key: value for key, value in alternate.items() if key != "controlHash"}
        )
        for payload in (alternate, json.loads(original)):
            replacement = serving_fixture.control_path.with_suffix(".aba")
            replacement.write_text(
                json.dumps(payload, separators=(",", ":")), encoding="utf-8"
            )
            os.replace(replacement, serving_fixture.control_path)

    monkeypatch.setattr(module, "_CORE_CONTROL_CAPTURE_SEAM", replace_with_aba)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(
        transport=transport, base_url="http://testserver"
    ) as client:
        capabilities = await client.get("/api/v1/gfm/capabilities")
        created = await client.post("/api/v1/gfm/runs", json=_public_request())

    assert capabilities.status_code == 503
    assert created.status_code == 503
    assert fake.created is None


@pytest.mark.anyio
@pytest.mark.parametrize("field", ["tasks", "graphSchemaVersions"])
async def test_duplicate_model_descriptor_lists_fail_closed(
    serving_fixture: ServingFixture, field: str
) -> None:
    serving_fixture.model[field].append(serving_fixture.model[field][0])
    serving_fixture.publish()
    fake = RecordingGfmClient(serving_fixture.capabilities)
    capabilities, created = await _responses(serving_fixture, fake)
    assert capabilities.status_code == 503
    assert created.status_code == 503
    assert fake.created is None


@pytest.mark.anyio
async def test_create_reacquire_rejects_control_changed_after_capability_acceptance(
    serving_fixture: ServingFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    remote = copy.deepcopy(serving_fixture.capabilities)
    fake = RecordingGfmClient(remote)
    app = create_app(serving_fixture.settings, gfm_client=fake)
    store = app.state.core_serving_control_store
    original = store.acquire
    calls = 0

    def acquire(required_model_id: str | None = None):
        nonlocal calls
        calls += 1
        if calls == 2:
            serving_fixture.model["maxNodes"] += 1
            serving_fixture.publish(
                control_generation=2, registry_generation=2, catalog_generation=2
            )
        return original(required_model_id)

    monkeypatch.setattr(store, "acquire", acquire)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(
        transport=transport, base_url="http://testserver"
    ) as client:
        created = await client.post("/api/v1/gfm/runs", json=_public_request())
    assert created.status_code == 503
    assert fake.created is None


@pytest.mark.anyio
@pytest.mark.parametrize(
    "case",
    [
        "checkpoint-sha",
        "manifest-sha",
        "bindings",
        "adapter-domain",
        "node-classes",
        "multi-hot-buckets",
        "task-heads",
        "calibrations",
        "adapter-schema",
    ],
)
async def test_model_and_manifest_identity_mutations_fail_closed(
    serving_fixture: ServingFixture, case: str
) -> None:
    if case == "checkpoint-sha":
        serving_fixture.model["checkpoint"]["sha256"] = HASHES["f"]
        serving_fixture.publish()
    elif case == "manifest-sha":
        serving_fixture.publish()
        serving_fixture.manifest_path.write_bytes(
            serving_fixture.manifest_path.read_bytes() + b" "
        )
    elif case == "bindings":
        serving_fixture.model["checkpoint"]["bindings"]["dataHash"] = HASHES["f"]
        serving_fixture.publish(recompute_model_hash=False)
    elif case in {"adapter-domain", "node-classes", "multi-hot-buckets"}:
        key = {
            "adapter-domain": "adapterDomain",
            "node-classes": "nodeClasses",
            "multi-hot-buckets": "multiHotBuckets",
        }[case]
        serving_fixture.model["checkpoint"][key] = (
            "changed"
            if case == "adapter-domain"
            else 3
            if case == "node-classes"
            else 64
        )
        serving_fixture.publish()
    elif case == "task-heads":
        serving_fixture.manifest["taskHeads"][0]["nodeOutputIndex"] = 0
        serving_fixture.publish()
    elif case == "calibrations":
        serving_fixture.manifest["taskHeads"][0]["calibrations"][0][
            "calibrationVersion"
        ] = "changed/2"
        serving_fixture.publish()
    else:
        serving_fixture.manifest["adapterSchemaHash"] = HASHES["f"]
        serving_fixture.manifest_path.write_text(
            json.dumps(serving_fixture.manifest, separators=(",", ":")),
            encoding="utf-8",
        )
    fake = RecordingGfmClient(serving_fixture.capabilities)
    capabilities, created = await _responses(serving_fixture, fake)
    assert capabilities.status_code == 503
    assert created.status_code == 503
    assert fake.created is None


@pytest.mark.anyio
@pytest.mark.parametrize("limit_name", ["MAX_MODELS", "MAX_TOTAL_MANIFEST_BYTES"])
async def test_api_snapshot_rejects_aggregate_work_over_limit(
    serving_fixture: ServingFixture,
    monkeypatch: pytest.MonkeyPatch,
    limit_name: str,
) -> None:
    module = importlib.import_module("app.gfm_core_serving_control")
    assert hasattr(module, limit_name)
    monkeypatch.setattr(module, limit_name, 0)
    fake = RecordingGfmClient(serving_fixture.capabilities)
    capabilities, created = await _responses(serving_fixture, fake)
    assert capabilities.status_code == 503
    assert created.status_code == 503
    assert fake.created is None


@pytest.mark.parametrize("value", ["", ".", "./", "./."])
def test_api_safe_relative_rejects_empty_or_dot_normalization(value: str) -> None:
    module = importlib.import_module("app.gfm_core_serving_control")
    with pytest.raises(ValueError, match="safe and relative"):
        module._safe_relative(value)


def _publish_dot_path(
    fixture: ServingFixture, case: str
) -> tuple[type[Any], dict[str, Any]]:
    schemas = importlib.import_module("app.gfm_core_schemas")
    if case.startswith("control-"):
        control = json.loads(fixture.control_path.read_bytes())
        control[case.removeprefix("control-")]["relativePath"] = "."
        control["controlHash"] = canonical_sha256(
            {key: value for key, value in control.items() if key != "controlHash"}
        )
        fixture.control_path.write_text(
            json.dumps(control, separators=(",", ":")), encoding="utf-8"
        )
        fixture.capabilities.clear()
        fixture.capabilities.update(fixture.expected_capabilities(control))
        return schemas.CoreServingControl, control
    if case == "registry-checkpoint":
        fixture.model["checkpoint"]["relativePath"] = "."
        fixture.publish()
        return schemas.CoreServingRegistry, json.loads(fixture.registry_path.read_bytes())
    if case == "registry-manifest":
        fixture.model["checkpoint"]["servingManifestRelativePath"] = "."
        fixture.publish()
        return schemas.CoreServingRegistry, json.loads(fixture.registry_path.read_bytes())
    if case == "catalog-bundle":
        fixture.catalog["artifacts"][0]["relativePath"] = "."
        fixture.publish()
        return schemas.CoreServingGraphCatalog, json.loads(fixture.catalog_path.read_bytes())
    fixture.manifest["taskHeads"][0]["calibrations"][0][
        "calibrationRelativePath"
    ] = "."
    fixture.publish()
    return schemas.CoreServingCheckpointManifest, json.loads(
        fixture.manifest_path.read_bytes()
    )


@pytest.mark.anyio
@pytest.mark.parametrize(
    "case",
    [
        "control-registry",
        "control-catalog",
        "registry-checkpoint",
        "registry-manifest",
        "catalog-bundle",
        "manifest-calibration",
    ],
)
async def test_dot_normalized_metadata_paths_fail_closed_without_client_create(
    serving_fixture: ServingFixture, case: str
) -> None:
    model, payload = _publish_dot_path(serving_fixture, case)
    with pytest.raises(ValueError):
        model.model_validate(payload)
    fake = RecordingGfmClient(serving_fixture.capabilities)
    capabilities, created = await _responses(serving_fixture, fake)
    assert capabilities.status_code == 503
    assert created.status_code == 503
    assert fake.created is None


@pytest.mark.anyio
@pytest.mark.parametrize("target", ["parent", "final"])
async def test_confined_reader_rejects_component_swap(
    serving_fixture: ServingFixture,
    monkeypatch: pytest.MonkeyPatch,
    target: str,
) -> None:
    module = importlib.import_module("app.gfm_core_serving_control")
    assert hasattr(module, "_CORE_PATH_WALK_SEAM")
    nested = serving_fixture.root / "metadata"
    nested.mkdir()
    moved = nested / "model.serving.json"
    os.replace(serving_fixture.manifest_path, moved)
    serving_fixture.manifest_path = moved
    serving_fixture.model["checkpoint"]["servingManifestRelativePath"] = (
        "metadata/model.serving.json"
    )
    serving_fixture.publish()
    fired = False

    def swap(stage: str, path: Path) -> None:
        nonlocal fired
        if fired or stage != "after-open" or path.name != "model.serving.json":
            return
        fired = True
        if target == "final":
            replacement = path.with_suffix(".swap")
            replacement.write_bytes(path.read_bytes())
            os.replace(replacement, path)
        else:
            replacement_parent = path.parent.with_name("metadata.swap")
            replacement_parent.mkdir()
            (replacement_parent / path.name).write_bytes(path.read_bytes())
            os.replace(path.parent, path.parent.with_name("metadata.old"))
            os.replace(replacement_parent, path.parent)

    monkeypatch.setattr(module, "_CORE_PATH_WALK_SEAM", swap)
    fake = RecordingGfmClient(serving_fixture.capabilities)
    app = create_app(serving_fixture.settings, gfm_client=fake)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(
        transport=transport, base_url="http://testserver"
    ) as client:
        created = await client.post("/api/v1/gfm/runs", json=_public_request())
    assert created.status_code == 503
    assert fake.created is None


@pytest.mark.anyio
async def test_windows_held_reader_allows_a_real_concurrent_reader(
    serving_fixture: ServingFixture,
) -> None:
    if os.name != "nt":
        pytest.skip("Windows sharing modes are not available")
    fake = RecordingGfmClient(serving_fixture.capabilities)
    app = create_app(serving_fixture.settings, gfm_client=fake)
    transport = httpx.ASGITransport(app=app)
    with serving_fixture.manifest_path.open("rb") as concurrent_reader:
        async with httpx.AsyncClient(
            transport=transport, base_url="http://testserver"
        ) as client:
            capabilities = await client.get("/api/v1/gfm/capabilities")
        assert concurrent_reader.read(1)
    assert capabilities.status_code == 200


@pytest.mark.anyio
@pytest.mark.parametrize(
    "operation", ["file-fsync", "atomic-replace", "directory-flush"]
)
async def test_high_water_depends_on_each_real_durability_operation(
    serving_fixture: ServingFixture,
    monkeypatch: pytest.MonkeyPatch,
    operation: str,
) -> None:
    fake = RecordingGfmClient(serving_fixture.capabilities)
    app = create_app(serving_fixture.settings, gfm_client=fake)
    store = app.state.core_serving_control_store
    module = importlib.import_module("app.gfm_core_serving_control")
    invoked = 0

    if operation == "file-fsync":
        real_fsync = module.os.fsync

        def fail_file_fsync(descriptor: int) -> None:
            nonlocal invoked
            if stat.S_ISREG(os.fstat(descriptor).st_mode):
                invoked += 1
                raise OSError("injected real file fsync failure")
            real_fsync(descriptor)

        monkeypatch.setattr(module.os, "fsync", fail_file_fsync)
    elif operation == "atomic-replace":
        real_replace = module.os.replace

        def fail_atomic_replace(source: str | Path, destination: str | Path) -> None:
            nonlocal invoked
            if Path(destination) == store.high_water_path:
                invoked += 1
                raise OSError("injected real atomic replace failure")
            real_replace(source, destination)

        monkeypatch.setattr(module.os, "replace", fail_atomic_replace)
    elif os.name == "nt":
        assert hasattr(module, "_FlushFileBuffers")

        def fail_directory_flush(_handle: int) -> int:
            nonlocal invoked
            invoked += 1
            ctypes.set_last_error(5)
            return 0

        monkeypatch.setattr(module, "_FlushFileBuffers", fail_directory_flush)
    else:
        real_fsync = module.os.fsync

        def fail_directory_fsync(descriptor: int) -> None:
            nonlocal invoked
            if stat.S_ISDIR(os.fstat(descriptor).st_mode):
                invoked += 1
                raise OSError("injected real directory fsync failure")
            real_fsync(descriptor)

        monkeypatch.setattr(module.os, "fsync", fail_directory_fsync)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(
        transport=transport, base_url="http://testserver"
    ) as client:
        created = await client.post("/api/v1/gfm/runs", json=_public_request())

    assert invoked == 1
    assert created.status_code == 503
    assert fake.created is None


@pytest.mark.anyio
async def test_identical_high_water_retry_reflushes_after_post_replace_flush_failure(
    serving_fixture: ServingFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = importlib.import_module("app.gfm_core_serving_control")
    real_flush = module._flush_parent_directory
    flush_attempts = 0

    def fail_first_flush(path: Path) -> None:
        nonlocal flush_attempts
        flush_attempts += 1
        if flush_attempts == 1:
            raise OSError("injected post-replace directory flush failure")
        real_flush(path)

    monkeypatch.setattr(module, "_flush_parent_directory", fail_first_flush)
    fake = RecordingGfmClient(serving_fixture.capabilities)
    app = create_app(serving_fixture.settings, gfm_client=fake)
    store = app.state.core_serving_control_store
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(
        transport=transport, base_url="http://testserver"
    ) as client:
        failed = await client.get("/api/v1/gfm/capabilities")
        assert failed.status_code == 503
        assert store.high_water_path.is_file()

        retried = await client.get("/api/v1/gfm/capabilities")

    assert retried.status_code == 200
    assert flush_attempts == 2


@pytest.mark.anyio
@pytest.mark.parametrize("mutation", ["noncanonical-current", "post-flush-mutated"])
async def test_identical_high_water_retry_rejects_unverified_persisted_bytes(
    serving_fixture: ServingFixture,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
) -> None:
    module = importlib.import_module("app.gfm_core_serving_control")
    fake = RecordingGfmClient(serving_fixture.capabilities)
    app = create_app(serving_fixture.settings, gfm_client=fake)
    store = app.state.core_serving_control_store
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(
        transport=transport, base_url="http://testserver"
    ) as client:
        accepted = await client.get("/api/v1/gfm/capabilities")
        assert accepted.status_code == 200
        payload = json.loads(store.high_water_path.read_bytes())

        if mutation == "noncanonical-current":
            store.high_water_path.write_text(
                json.dumps(payload, indent=2), encoding="utf-8"
            )
        else:
            fired = False

            def mutate_after_flush(stage: str, path: Path) -> None:
                nonlocal fired
                if fired or stage != "before-persisted-reread":
                    return
                fired = True
                changed = json.loads(path.read_bytes())
                changed["controlHash"] = HASHES["f"]
                changed["recordHash"] = canonical_sha256(
                    {
                        key: value
                        for key, value in changed.items()
                        if key != "recordHash"
                    }
                )
                path.write_bytes((canonical_json(changed) + "\n").encode("utf-8"))

            monkeypatch.setattr(module, "_CORE_HIGH_WATER_SEAM", mutate_after_flush)

        rejected = await client.get("/api/v1/gfm/capabilities")

    assert rejected.status_code == 503
    assert rejected.json() == {"detail": {"code": "GFM_CORE_SERVING_CONTROL_INVALID"}}


@pytest.mark.anyio
@pytest.mark.parametrize("mutation", ["record-hash", "canonical-bytes", "safe-reread"])
async def test_api_snapshot_verifies_persisted_high_water_bytes(
    serving_fixture: ServingFixture,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
) -> None:
    module = importlib.import_module("app.gfm_core_serving_control")
    assert hasattr(module, "_CORE_HIGH_WATER_SEAM")

    def mutate(stage: str, path: Path) -> None:
        if stage != "before-persisted-reread":
            return
        payload = json.loads(path.read_bytes())
        if mutation == "record-hash":
            payload["recordHash"] = HASHES["f"]
            path.write_text(
                json.dumps(payload, separators=(",", ":")), encoding="utf-8"
            )
        elif mutation == "canonical-bytes":
            path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        else:
            path.unlink()

    monkeypatch.setattr(module, "_CORE_HIGH_WATER_SEAM", mutate)
    fake = RecordingGfmClient(serving_fixture.capabilities)
    app = create_app(serving_fixture.settings, gfm_client=fake)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(
        transport=transport, base_url="http://testserver"
    ) as client:
        created = await client.post("/api/v1/gfm/runs", json=_public_request())
    assert created.status_code == 503
    assert fake.created is None


@pytest.mark.anyio
async def test_native_reparse_manifest_is_rejected_when_supported(
    serving_fixture: ServingFixture,
) -> None:
    target = serving_fixture.root / "manifest-target.json"
    target.write_bytes(serving_fixture.manifest_path.read_bytes())
    serving_fixture.manifest_path.unlink()
    try:
        serving_fixture.manifest_path.symlink_to(target)
    except OSError:
        pytest.skip("native symlink/reparse creation is unavailable")
    fake = RecordingGfmClient(serving_fixture.capabilities)
    capabilities, created = await _responses(serving_fixture, fake)
    assert capabilities.status_code == 503
    assert created.status_code == 503
    assert fake.created is None


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


def _lease_projection(snapshot: dict[str, Any]) -> dict[str, Any]:
    return {
        "schemaVersion": "socialgraph-fm.core-run-lease-identity/2.2",
        **{field: snapshot[field] for field in _LEASE_IDENTITY_FIELDS if field in snapshot},
    }


def _rehash_receipt(receipt: dict[str, Any]) -> None:
    snapshot = receipt["executionSnapshot"]
    snapshot["snapshotHash"] = canonical_sha256(
        {key: value for key, value in snapshot.items() if key != "snapshotHash"}
    )
    receipt["leaseIdentityHash"] = canonical_sha256(_lease_projection(snapshot))
    receipt["receiptHash"] = canonical_sha256(
        {key: value for key, value in receipt.items() if key != "receiptHash"}
    )


def _complete_api_receipt(
    fixture: ServingFixture,
    envelope: dict[str, Any],
    *,
    run_id: str = "00000000-0000-0000-0000-000000000001",
) -> dict[str, Any]:
    control = json.loads(fixture.control_path.read_bytes())
    registry = json.loads(fixture.registry_path.read_bytes())
    catalog = json.loads(fixture.catalog_path.read_bytes())
    model = registry["models"][0]
    task_head = next(
        head for head in model["taskHeads"] if head["taskId"] == envelope["request"]["taskId"]
    )
    calibrations = [
        {
            "entityType": item["entityType"],
            "confidenceKind": item["confidenceKind"],
            "calibrationVersion": item["calibrationVersion"],
            "method": item["calibrationMethod"],
            "calibrationArtifactHash": item["calibrationArtifactHash"],
            "calibrationProtocolHash": item["calibrationProtocolHash"],
            "adapterDomain": item["adapterDomain"],
            "adapterSchemaHash": item["adapterSchemaHash"],
            "adapterStateHash": item["adapterStateHash"],
            "featureContractHash": item["graphFeatureContractHash"],
            "sha256": item["calibrationSha256"],
        }
        for item in sorted(task_head["calibrations"], key=lambda value: value["entityType"])
    ]
    target_scope = envelope["request"]["targetScope"]
    selected_entity_type = {
        "community": "community",
        "node-pairs": "node-pair",
    }.get(target_scope["kind"])
    if selected_entity_type is None:
        selected_entity_type = "node" if target_scope["nodeIds"] else "edge"
    selected = next(
        item for item in calibrations if item["entityType"] == selected_entity_type
    )
    created_at = "2026-08-15T00:00:00.000000Z"
    status: dict[str, Any] = {
        "schemaVersion": "socialgraph-fm.core-run-status/2.0",
        "runId": run_id,
        "requestHash": canonical_sha256(envelope),
        "status": "queued",
        "progress": 0,
        "createdAt": created_at,
        "updatedAt": created_at,
        "errorCode": None,
    }
    status["stateHash"] = canonical_sha256(status)
    graph = envelope["graphReference"]
    snapshot: dict[str, Any] = {
        "schemaVersion": "socialgraph-fm.core-run-execution-snapshot/2.2",
        "runId": run_id,
        "requestHash": canonical_sha256(envelope),
        "controlSourceSha256": hashlib.sha256(fixture.control_path.read_bytes()).hexdigest(),
        "controlHash": control["controlHash"],
        "controlGeneration": control["generation"],
        "registrySourceSha256": hashlib.sha256(fixture.registry_path.read_bytes()).hexdigest(),
        "registryHash": control["registry"]["semanticHash"],
        "registryGeneration": registry["generation"],
        "artifactCatalogSha256": hashlib.sha256(fixture.catalog_path.read_bytes()).hexdigest(),
        "artifactCatalogHash": control["catalog"]["semanticHash"],
        "artifactCatalogGeneration": catalog["generation"],
        "modelVersionId": model["modelVersionId"],
        "modelVersionHash": model["modelVersionHash"],
        "checkpointSha256": model["checkpoint"]["sha256"],
        "servingManifestSha256": hashlib.sha256(fixture.manifest_path.read_bytes()).hexdigest(),
        "adapterSchemaHash": selected["adapterSchemaHash"],
        "calibrationIdentities": calibrations,
        "calibrationSetHash": canonical_sha256(calibrations),
        "taskId": envelope["request"]["taskId"],
        "graphVersionId": graph["graphVersionId"],
        "sourceGraphFactHash": graph["sourceGraphFactHash"],
        "graphVersionHash": graph["graphVersionHash"],
        "artifactId": graph["artifactId"],
        "artifactHash": graph["artifactHash"],
        "bundleSha256": graph["bundleSha256"],
        "graphSchemaVersion": graph["graphSchemaVersion"],
        "featureContractHash": graph["featureContractHash"],
        "nodeCount": graph["nodeCount"],
        "edgeCount": graph["edgeCount"],
        "createdAt": created_at,
    }
    snapshot["snapshotHash"] = canonical_sha256(snapshot)
    receipt = {
        "schemaVersion": "socialgraph-fm.core-internal-create-run-receipt/2.0",
        "status": status,
        "executionSnapshot": snapshot,
        "leaseIdentityHash": canonical_sha256(_lease_projection(snapshot)),
    }
    receipt["receiptHash"] = canonical_sha256(receipt)
    return receipt


def _success_status_and_result(receipt: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    queued = receipt["status"]
    snapshot = receipt["executionSnapshot"]
    status = {
        **queued,
        "status": "succeeded",
        "progress": 100,
        "updatedAt": "2026-08-15T00:00:01.000000Z",
    }
    status["stateHash"] = canonical_sha256(
        {key: value for key, value in status.items() if key != "stateHash"}
    )
    result: dict[str, Any] = {
        "schemaVersion": "socialgraph-fm.core-run-result/2.0",
        "runId": snapshot["runId"],
        "requestHash": snapshot["requestHash"],
        "taskId": snapshot["taskId"],
        "graphVersionId": snapshot["graphVersionId"],
        "graphVersionHash": snapshot["graphVersionHash"],
        "modelVersionId": snapshot["modelVersionId"],
        "modelVersionHash": snapshot["modelVersionHash"],
        "findings": [],
        "completedAt": "2026-08-15T00:00:01.000000Z",
    }
    result["resultHash"] = canonical_sha256(result)
    return status, result


def _configure_community_serving(fixture: ServingFixture) -> None:
    feature_hash = fixture.catalog["artifacts"][0]["featureContractHash"]
    calibration = {
        "entityType": "community",
        "confidenceKind": "regression-interval",
        "calibrationVersion": "community-interval/1",
        "calibrationMethod": "validation-residual-interval",
        "calibrationArtifactHash": HASHES["7"],
        "calibrationRelativePath": "calibration/community.json",
        "calibrationSha256": HASHES["8"],
        "calibrationProtocolHash": HASHES["9"],
        "adapterDomain": "community",
        "adapterSchemaHash": HASHES["d"],
        "adapterStateHash": HASHES["c"],
        "graphFeatureContractHash": feature_hash,
    }
    task_head = {
        "taskId": "core.community_resilience_review",
        "kind": "community-resilience",
        "nodeOutputIndex": None,
        "calibrations": [calibration],
    }
    fixture.model["taskHeads"] = [task_head]
    fixture.model["tasks"] = ["core.community_resilience_review"]
    fixture.model["graphFeatureContractHash"] = canonical_sha256(
        [
            {
                "taskId": task_head["taskId"],
                "entityType": calibration["entityType"],
                "featureContractHash": feature_hash,
            }
        ]
    )
    fixture.model["checkpoint"]["adapterDomain"] = "community"
    fixture.manifest.update(
        {
            "adapterDomain": "community",
            "adapterSchemaHash": HASHES["d"],
            "adapterStateHash": HASHES["c"],
            "adapterBindings": [
                {
                    "adapterDomain": "community",
                    "adapterSchemaHash": HASHES["d"],
                    "adapterStateHash": HASHES["c"],
                    "multiHotBuckets": 32,
                }
            ],
            "taskHeads": [copy.deepcopy(task_head)],
        }
    )
    fixture.publish()


def _community_request() -> dict[str, Any]:
    return {
        "schemaVersion": "socialgraph-fm.core-run-request/2.0",
        "graphVersionId": "graph-v1",
        "taskId": "core.community_resilience_review",
        "targetScope": {"kind": "community", "communityIds": ["community-a"]},
        "modelVersionId": "socialgraph-fm-core/review",
        "parameters": {"kind": "community-resilience", "topKSimilarCases": 3},
    }


def _finding_for_snapshot(
    snapshot: dict[str, Any], entity_type: str
) -> dict[str, Any]:
    identity = next(
        item
        for item in snapshot["calibrationIdentities"]
        if item["entityType"] == entity_type
    )
    entity_ids = ["community-a"] if entity_type == "community" else ["node-a"]
    score: dict[str, Any] = {
        "schemaVersion": "socialgraph-fm.core-model-score/2.0",
        "taskId": snapshot["taskId"],
        "entityType": entity_type,
        "entityIds": entity_ids,
        "score": 0.25,
        "graphVersionHash": snapshot["graphVersionHash"],
        "modelVersion": snapshot["modelVersionId"],
        "modelVersionHash": snapshot["modelVersionHash"],
        "edgeIdentity": None,
    }
    score["scoreHash"] = canonical_sha256(score)
    confidence: dict[str, Any]
    if entity_type == "community":
        confidence = {
            "schemaVersion": "socialgraph-fm.core-regression-confidence-interval/1.0",
            "pointEstimate": score["score"],
            "lowerBound": 0.1,
            "upperBound": 0.4,
            "coverage": 0.9,
            "validationCount": 32,
            "scoreHash": score["scoreHash"],
            "taskId": score["taskId"],
            "entityType": entity_type,
            "entityIds": entity_ids,
            "graphVersionHash": score["graphVersionHash"],
            "modelVersion": score["modelVersion"],
            "modelVersionHash": score["modelVersionHash"],
            "confidenceVersion": identity["calibrationVersion"],
            "method": identity["method"],
            "confidenceArtifactHash": identity["calibrationArtifactHash"],
            "confidenceProtocolHash": identity["calibrationProtocolHash"],
        }
    else:
        confidence = {
            "schemaVersion": "socialgraph-fm.core-calibrated-confidence/2.0",
            "value": 0.6,
            "scoreHash": score["scoreHash"],
            "taskId": score["taskId"],
            "entityType": entity_type,
            "entityIds": entity_ids,
            "graphVersionHash": score["graphVersionHash"],
            "modelVersion": score["modelVersion"],
            "modelVersionHash": score["modelVersionHash"],
            "calibrationVersion": identity["calibrationVersion"],
            "method": identity["method"],
            "calibrationArtifactHash": identity["calibrationArtifactHash"],
            "calibrationProtocolHash": identity["calibrationProtocolHash"],
        }
    confidence["confidenceHash"] = canonical_sha256(confidence)
    evidence: dict[str, Any] = {
        "schemaVersion": "socialgraph-fm.core-evidence/2.0",
        "metric": "registered_model.score-reference",
        "valueCanonicalJson": "{}",
        "graphVersionHash": score["graphVersionHash"],
        "sourceType": "registered-model-output",
        "nodeIds": entity_ids,
        "edgeIds": [],
        "algorithmConfigHash": None,
        "modelVersionHash": score["modelVersionHash"],
        "modelVersion": score["modelVersion"],
        "modelScoreHash": score["scoreHash"],
        "modelTaskId": score["taskId"],
        "modelEntityType": entity_type,
        "modelEntityIds": entity_ids,
        "limitations": [
            "The score is a registered model output, not a graph fact or decision."
        ],
    }
    evidence["evidenceHash"] = canonical_sha256(evidence)
    finding: dict[str, Any] = {
        "schemaVersion": "socialgraph-fm.core-finding/2.0",
        "taskId": score["taskId"],
        "findingType": (
            "community-resilience-candidate"
            if entity_type == "community"
            else "node-risk-candidate"
        ),
        "subjectIds": entity_ids,
        "score": score,
        "calibratedConfidence": confidence,
        "evidence": [evidence],
        "similarCases": [],
        "graphVersionHash": score["graphVersionHash"],
        "modelVersion": score["modelVersion"],
        "modelVersionHash": score["modelVersionHash"],
        "limitations": [
            "Manual human review is required; no automatic sanction or action is authorized.",
            "This finding is non-causal and does not predict future events.",
            *(
                [
                    "The resilience interval reports validation residual coverage, not a probability."
                ]
                if entity_type == "community"
                else []
            ),
        ],
        "reviewStatus": "pending-human-review",
    }
    finding["findingHash"] = canonical_sha256(finding)
    return finding


def _mutate_result_confidence(
    result: dict[str, Any], snapshot: dict[str, Any], case: str
) -> None:
    finding = result["findings"][0]
    confidence = finding["calibratedConfidence"]
    if case == "swap-node-edge":
        replacement = next(
            item
            for item in snapshot["calibrationIdentities"]
            if item["entityType"] == "edge"
        )
        confidence.update(
            {
                "calibrationVersion": replacement["calibrationVersion"],
                "method": replacement["method"],
                "calibrationArtifactHash": replacement["calibrationArtifactHash"],
                "calibrationProtocolHash": replacement["calibrationProtocolHash"],
            }
        )
    elif case == "method":
        confidence["method"] = "temperature-scaling"
    else:
        regression = confidence["schemaVersion"].endswith(
            "regression-confidence-interval/1.0"
        )
        field = {
            "version": "confidenceVersion" if regression else "calibrationVersion",
            "artifact": (
                "confidenceArtifactHash" if regression else "calibrationArtifactHash"
            ),
            "protocol": (
                "confidenceProtocolHash" if regression else "calibrationProtocolHash"
            ),
        }[case]
        confidence[field] = "changed/2" if case == "version" else HASHES["f"]
    confidence["confidenceHash"] = canonical_sha256(
        {key: value for key, value in confidence.items() if key != "confidenceHash"}
    )
    finding["findingHash"] = canonical_sha256(
        {key: value for key, value in finding.items() if key != "findingHash"}
    )
    result["resultHash"] = canonical_sha256(
        {key: value for key, value in result.items() if key != "resultHash"}
    )


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("confidence_kind", "case"),
    [
        ("binary", "swap-node-edge"),
        ("binary", "version"),
        ("binary", "method"),
        ("binary", "artifact"),
        ("binary", "protocol"),
        ("regression", "version"),
        ("regression", "artifact"),
        ("regression", "protocol"),
    ],
)
async def test_restart_rejects_rehashed_result_confidence_lease_substitution(
    serving_fixture: ServingFixture, confidence_kind: str, case: str
) -> None:
    if confidence_kind == "regression":
        _configure_community_serving(serving_fixture)
    request = _community_request() if confidence_kind == "regression" else _public_request()
    fake = RecordingGfmClient(serving_fixture.capabilities)
    fake.create_response = lambda envelope: _complete_api_receipt(
        serving_fixture, envelope
    )
    app = create_app(serving_fixture.settings, gfm_client=fake)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(
        transport=transport, base_url="http://testserver"
    ) as client:
        created = await client.post("/api/v1/gfm/runs", json=request)
    assert created.status_code == 202
    assert fake.created is not None
    receipt = _complete_api_receipt(serving_fixture, fake.created)
    fake.status_response, fake.result_response = _success_status_and_result(receipt)
    snapshot = receipt["executionSnapshot"]
    entity_type = "community" if confidence_kind == "regression" else "node"
    fake.result_response["findings"] = [
        _finding_for_snapshot(snapshot, entity_type)
    ]
    fake.result_response["resultHash"] = canonical_sha256(
        {
            key: value
            for key, value in fake.result_response.items()
            if key != "resultHash"
        }
    )
    run_id = snapshot["runId"]
    restarted = CoreGateway(
        fake,
        binding_store=CoreRunBindingStore(
            serving_fixture.root / "api-run-bindings"
        ),
    )

    assert (await restarted.get_run(run_id)).status == "succeeded"
    assert (await restarted.get_result(run_id)).findings[0].score.entity_type == (
        entity_type
    )
    _mutate_result_confidence(fake.result_response, snapshot, case)
    with pytest.raises(GfmProxyError) as rejected:
        await restarted.get_result(run_id)
    assert rejected.value.code == "GFM_CORE_RESULT_BINDING_INVALID"


@pytest.mark.anyio
async def test_api_accepts_complete_receipt_and_persists_full_binding(
    serving_fixture: ServingFixture,
) -> None:
    fake = RecordingGfmClient(serving_fixture.capabilities)
    fake.create_response = lambda envelope: _complete_api_receipt(
        serving_fixture, envelope
    )
    app = create_app(serving_fixture.settings, gfm_client=fake)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(
        transport=transport, base_url="http://testserver"
    ) as client:
        created = await client.post("/api/v1/gfm/runs", json=_public_request())

    assert created.status_code == 202, (
        created.json(),
        fake.created,
        list((serving_fixture.root / "api-run-bindings").glob("*.json")),
    )
    assert fake.created is not None
    receipt = _complete_api_receipt(serving_fixture, fake.created)
    calibration_associations = {
        (
            item["calibrationVersion"],
            item["calibrationArtifactHash"],
            item["calibrationProtocolHash"],
            item["sha256"],
        )
        for item in receipt["executionSnapshot"]["calibrationIdentities"]
    }
    assert len(calibration_associations) == 2
    run_id = receipt["status"]["runId"]
    binding_path = (
        serving_fixture.root / "api-run-bindings" / f"{run_id}.json"
    )
    binding = json.loads(binding_path.read_bytes())
    assert binding["schemaVersion"] == "socialgraph-fm.core-api-run-binding/2.2"
    assert binding["receipt"] == CoreInternalCreateRunReceipt.model_validate_json(
        json.dumps(receipt)
    ).model_dump(mode="json", by_alias=True)
    assert binding["expectation"]["createRequest"] == fake.created
    assert binding["bindingHash"] == canonical_sha256(
        {key: value for key, value in binding.items() if key != "bindingHash"}
    )


_RECEIPT_SUBSTITUTION_CASES = (
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
    "calibration-entityType",
    "calibration-calibrationVersion",
    "calibration-method",
    "calibration-calibrationArtifactHash",
    "calibration-calibrationProtocolHash",
    "calibration-sha256",
    "calibration-association-swap",
    "calibration-association-reuse",
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
    "snapshotHash",
    "leaseIdentityHash",
    "receiptHash",
    "missing-controlSourceSha256",
    "missing-calibration-sha256",
)


def _substitute_receipt_identity(receipt: dict[str, Any], case: str) -> None:
    snapshot = receipt["executionSnapshot"]
    status = receipt["status"]
    if case.startswith("calibration-association-"):
        fields = (
            "calibrationVersion",
            "method",
            "calibrationArtifactHash",
            "calibrationProtocolHash",
            "sha256",
        )
        edge, node = snapshot["calibrationIdentities"]
        edge_values = {field: edge[field] for field in fields}
        node_values = {field: node[field] for field in fields}
        if case.endswith("swap"):
            edge.update(node_values)
            node.update(edge_values)
        else:
            node.update(edge_values)
        snapshot["calibrationSetHash"] = canonical_sha256(
            snapshot["calibrationIdentities"]
        )
        _rehash_receipt(receipt)
        return
    if case.startswith("missing-"):
        field = case.removeprefix("missing-")
        if field.startswith("calibration-"):
            snapshot["calibrationIdentities"][0].pop(field.removeprefix("calibration-"))
            snapshot["calibrationSetHash"] = canonical_sha256(
                snapshot["calibrationIdentities"]
            )
        else:
            snapshot.pop(field)
        _rehash_receipt(receipt)
        return
    if case.startswith("calibration-"):
        field = case.removeprefix("calibration-")
        replacement: Any = "node" if field == "entityType" else "changed/2"
        if field == "method":
            replacement = "isotonic"
        elif field in {
            "calibrationArtifactHash",
            "calibrationProtocolHash",
            "sha256",
        }:
            replacement = HASHES["f"]
        snapshot["calibrationIdentities"][0][field] = replacement
        snapshot["calibrationSetHash"] = canonical_sha256(
            snapshot["calibrationIdentities"]
        )
        _rehash_receipt(receipt)
        return
    if case == "runId":
        value: Any = "00000000-0000-0000-0000-000000000099"
        snapshot[case] = value
    elif case == "requestHash":
        value = HASHES["f"]
        snapshot[case] = value
        status[case] = value
        status["stateHash"] = canonical_sha256(
            {key: item for key, item in status.items() if key != "stateHash"}
        )
    elif case in {
        "controlGeneration",
        "registryGeneration",
        "artifactCatalogGeneration",
        "nodeCount",
        "edgeCount",
    }:
        snapshot[case] += 1
    elif case == "modelVersionId":
        snapshot[case] = "changed-model/2"
    elif case == "taskId":
        snapshot[case] = "core.collaboration_completion"
        snapshot["calibrationIdentities"] = [
            {**snapshot["calibrationIdentities"][0], "entityType": "node-pair"}
        ]
        snapshot["calibrationSetHash"] = canonical_sha256(
            snapshot["calibrationIdentities"]
        )
    elif case == "graphVersionId":
        snapshot[case] = "changed-graph-v2"
    elif case == "artifactId":
        snapshot[case] = "changed-artifact-v2"
    elif case == "graphSchemaVersion":
        snapshot[case] = "socialgraph-fm.core-graph-bundle/9.9"
    elif case == "createdAt":
        snapshot[case] = "2026-08-15T00:00:02.000000Z"
    elif case in {"snapshotHash", "leaseIdentityHash", "receiptHash"}:
        pass
    else:
        snapshot[case] = HASHES["f"]
    _rehash_receipt(receipt)
    if case == "snapshotHash":
        snapshot["snapshotHash"] = HASHES["f"]
        receipt["receiptHash"] = canonical_sha256(
            {key: value for key, value in receipt.items() if key != "receiptHash"}
        )
    elif case == "leaseIdentityHash":
        receipt["leaseIdentityHash"] = HASHES["f"]
        receipt["receiptHash"] = canonical_sha256(
            {key: value for key, value in receipt.items() if key != "receiptHash"}
        )
    elif case == "receiptHash":
        receipt["receiptHash"] = HASHES["f"]


@pytest.mark.anyio
@pytest.mark.parametrize("case", _RECEIPT_SUBSTITUTION_CASES)
async def test_api_rejects_rehashed_receipt_identity_substitution(
    serving_fixture: ServingFixture, case: str
) -> None:
    control_fake = RecordingGfmClient(serving_fixture.capabilities)
    control_fake.create_response = lambda envelope: _complete_api_receipt(
        serving_fixture, envelope
    )
    control_app = create_app(serving_fixture.settings, gfm_client=control_fake)
    control_transport = httpx.ASGITransport(app=control_app)
    async with httpx.AsyncClient(
        transport=control_transport, base_url="http://testserver"
    ) as client:
        control = await client.post("/api/v1/gfm/runs", json=_public_request())
    assert control.status_code == 202

    attacked_fake = RecordingGfmClient(serving_fixture.capabilities)

    def attacked(envelope: dict[str, Any]) -> dict[str, Any]:
        receipt = _complete_api_receipt(
            serving_fixture,
            envelope,
            run_id="00000000-0000-0000-0000-000000000002",
        )
        _substitute_receipt_identity(receipt, case)
        return receipt

    attacked_fake.create_response = attacked
    attacked_app = create_app(serving_fixture.settings, gfm_client=attacked_fake)
    attacked_transport = httpx.ASGITransport(app=attacked_app)
    async with httpx.AsyncClient(
        transport=attacked_transport, base_url="http://testserver"
    ) as client:
        response = await client.post("/api/v1/gfm/runs", json=_public_request())
    assert response.status_code == 502
    assert response.json()["detail"]["code"] == "GFM_CORE_RUN_BINDING_INVALID"


@pytest.mark.anyio
async def test_later_reads_revalidate_complete_nested_binding_and_receipt(
    serving_fixture: ServingFixture,
) -> None:
    fake = RecordingGfmClient(serving_fixture.capabilities)
    fake.create_response = lambda envelope: _complete_api_receipt(
        serving_fixture, envelope
    )
    app = create_app(serving_fixture.settings, gfm_client=fake)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(
        transport=transport, base_url="http://testserver"
    ) as client:
        created = await client.post("/api/v1/gfm/runs", json=_public_request())
        assert created.status_code == 202
        assert fake.created is not None
        receipt = _complete_api_receipt(serving_fixture, fake.created)
        fake.status_response, fake.result_response = _success_status_and_result(receipt)
        run_id = receipt["status"]["runId"]
        assert (await client.get(f"/api/v1/gfm/runs/{run_id}")).status_code == 200
        assert (
            await client.get(f"/api/v1/gfm/runs/{run_id}/result")
        ).status_code == 200

        binding_path = (
            serving_fixture.root / "api-run-bindings" / f"{run_id}.json"
        )
        binding = json.loads(binding_path.read_bytes())
        binding["receipt"]["executionSnapshot"]["checkpointSha256"] = HASHES["f"]
        _rehash_receipt(binding["receipt"])
        binding["bindingHash"] = canonical_sha256(
            {key: value for key, value in binding.items() if key != "bindingHash"}
        )
        binding_path.write_text(
            json.dumps(binding, separators=(",", ":")), encoding="utf-8"
        )

        status = await client.get(f"/api/v1/gfm/runs/{run_id}")
        result = await client.get(f"/api/v1/gfm/runs/{run_id}/result")
    assert status.status_code == 502
    assert result.status_code == 502


def _coherently_substitute_complete_binding(
    binding: dict[str, Any],
) -> dict[str, Any]:
    changed = copy.deepcopy(binding)
    receipt = CoreInternalCreateRunReceipt.model_validate_json(
        json.dumps(changed["receipt"])
    ).model_dump(mode="python", by_alias=True)
    receipt["executionSnapshot"]["checkpointSha256"] = HASHES["f"]
    changed["expectation"]["checkpointSha256"] = HASHES["f"]
    _rehash_receipt(receipt)
    changed["receipt"] = CoreInternalCreateRunReceipt.model_validate(receipt).model_dump(
        mode="json", by_alias=True
    )
    changed["bindingHash"] = canonical_sha256(
        {key: value for key, value in changed.items() if key != "bindingHash"}
    )
    return changed


async def _create_persisted_complete_binding(
    serving_fixture: ServingFixture,
) -> tuple[RecordingGfmClient, dict[str, Any], Path]:
    fake = RecordingGfmClient(serving_fixture.capabilities)
    fake.create_response = lambda envelope: _complete_api_receipt(
        serving_fixture, envelope
    )
    app = create_app(serving_fixture.settings, gfm_client=fake)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(
        transport=transport, base_url="http://testserver"
    ) as client:
        created = await client.post("/api/v1/gfm/runs", json=_public_request())
    assert created.status_code == 202
    assert fake.created is not None
    receipt = _complete_api_receipt(serving_fixture, fake.created)
    run_id = receipt["status"]["runId"]
    binding_path = serving_fixture.root / "api-run-bindings" / f"{run_id}.json"
    return fake, receipt, binding_path


@pytest.mark.anyio
async def test_restart_rejects_coherent_binding_only_rewrite(
    serving_fixture: ServingFixture,
) -> None:
    fake, receipt, binding_path = await _create_persisted_complete_binding(
        serving_fixture
    )
    binding = json.loads(binding_path.read_bytes())
    changed = _coherently_substitute_complete_binding(binding)
    CoreRunBinding.model_validate_json(json.dumps(changed))
    binding_path.write_text(
        json.dumps(changed, separators=(",", ":")), encoding="utf-8"
    )
    fake.status_response, fake.result_response = _success_status_and_result(receipt)
    restarted = CoreGateway(
        fake,
        binding_store=CoreRunBindingStore(binding_path.parent),
    )
    run_id = receipt["status"]["runId"]

    with pytest.raises(GfmProxyError) as status_error:
        await restarted.get_run(run_id)
    with pytest.raises(GfmProxyError) as result_error:
        await restarted.get_result(run_id)
    assert status_error.value.code == "GFM_CORE_RUN_BINDING_INVALID"
    assert result_error.value.code == "GFM_CORE_RUN_BINDING_INVALID"


@pytest.mark.anyio
async def test_binding_before_anchor_crash_is_unreadable_and_exact_retry_repairs(
    serving_fixture: ServingFixture,
) -> None:
    _fake, receipt, binding_path = await _create_persisted_complete_binding(
        serving_fixture
    )
    original = CoreRunBinding.model_validate_json(binding_path.read_bytes())
    anchor_path = binding_path.with_name(f"{receipt['status']['runId']}.anchor.json")
    anchor_path.unlink(missing_ok=True)
    restarted = CoreRunBindingStore(binding_path.parent)

    with pytest.raises(GfmProxyError) as unreadable:
        restarted.get(receipt["status"]["runId"])
    assert unreadable.value.code == "GFM_CORE_RUN_BINDING_INVALID"

    restarted.save(original)
    assert anchor_path.is_file()
    assert restarted.get(receipt["status"]["runId"]) == original


@pytest.mark.anyio
async def test_binding_before_anchor_crash_rejects_mismatched_retry(
    serving_fixture: ServingFixture,
) -> None:
    _fake, receipt, binding_path = await _create_persisted_complete_binding(
        serving_fixture
    )
    original = CoreRunBinding.model_validate_json(binding_path.read_bytes())
    changed_payload = _coherently_substitute_complete_binding(
        original.model_dump(mode="json", by_alias=True)
    )
    changed = CoreRunBinding.model_validate_json(json.dumps(changed_payload))
    anchor_path = binding_path.with_name(f"{receipt['status']['runId']}.anchor.json")
    anchor_path.unlink(missing_ok=True)
    restarted = CoreRunBindingStore(binding_path.parent)

    with pytest.raises(GfmProxyError) as unreadable:
        restarted.get(receipt["status"]["runId"])
    assert unreadable.value.code == "GFM_CORE_RUN_BINDING_INVALID"
    with pytest.raises(GfmProxyError) as mismatch:
        restarted.save(changed)
    assert mismatch.value.code == "GFM_CORE_RUN_BINDING_INVALID"
    assert not anchor_path.exists()
    assert CoreRunBinding.model_validate_json(binding_path.read_bytes()) == original


def _expected_canonical_record_bytes(
    record: CoreRunBinding | CoreRunBindingAnchor,
) -> bytes:
    payload = record.model_dump(mode="json", by_alias=True)
    return (
        json.dumps(
            payload,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")


def _binding_anchor(binding: CoreRunBinding) -> CoreRunBindingAnchor:
    payload = {
        "schemaVersion": "socialgraph-fm.core-api-run-binding-anchor/1.0",
        "runId": binding.run_id,
        "bindingHash": binding.binding_hash,
    }
    payload["anchorHash"] = canonical_sha256(payload)
    return CoreRunBindingAnchor.model_validate(payload)


async def _publication_candidate(
    serving_fixture: ServingFixture,
) -> CoreRunBinding:
    _fake, _receipt, binding_path = await _create_persisted_complete_binding(
        serving_fixture
    )
    return CoreRunBinding.model_validate_json(binding_path.read_bytes())


@pytest.mark.anyio
async def test_binding_publication_uses_private_exclusive_temps_and_canonical_bytes(
    serving_fixture: ServingFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    binding = await _publication_candidate(serving_fixture)
    module = importlib.import_module("app.gfm_client")
    real_open = module.os.open
    temp_opens: list[tuple[int, int, int | None]] = []

    def observed_open(
        path: str | bytes | Path,
        flags: int,
        mode: int = 0o777,
        *,
        dir_fd: int | None = None,
    ) -> int:
        if str(path).endswith(".tmp"):
            temp_opens.append((flags, mode, dir_fd))
        if dir_fd is None:
            return real_open(path, flags, mode)
        return real_open(path, flags, mode, dir_fd=dir_fd)

    monkeypatch.setattr(module.os, "open", observed_open)
    store = CoreRunBindingStore(serving_fixture.root / "private-publication")
    store.save(binding)

    creation_opens = [entry for entry in temp_opens if entry[0] & os.O_CREAT]
    safe_reopens = [entry for entry in temp_opens if not entry[0] & os.O_CREAT]
    assert len(creation_opens) == 2
    assert all(flags & os.O_EXCL for flags, _mode, _dir_fd in creation_opens)
    assert all(mode & 0o777 == 0o600 for _flags, mode, _dir_fd in creation_opens)
    if os.name == "nt":
        assert not safe_reopens
    else:
        assert len(safe_reopens) == 2
        assert all(dir_fd is not None for _flags, _mode, dir_fd in safe_reopens)
        assert all(flags & os.O_NOFOLLOW for flags, _mode, _dir_fd in safe_reopens)
    binding_path = store.root / f"{binding.run_id}.json"
    anchor_path = store.root / f"{binding.run_id}.anchor.json"
    assert binding_path.read_bytes() == _expected_canonical_record_bytes(binding)
    assert anchor_path.read_bytes() == _expected_canonical_record_bytes(
        _binding_anchor(binding)
    )


@pytest.mark.anyio
async def test_binding_publication_depends_on_real_temp_file_fsync(
    serving_fixture: ServingFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    binding = await _publication_candidate(serving_fixture)
    module = importlib.import_module("app.gfm_client")
    real_fsync = module.os.fsync
    invoked = 0

    def fail_regular_file_fsync(descriptor: int) -> None:
        nonlocal invoked
        if stat.S_ISREG(os.fstat(descriptor).st_mode):
            invoked += 1
            raise OSError("injected binding temp fsync failure")
        real_fsync(descriptor)

    monkeypatch.setattr(module.os, "fsync", fail_regular_file_fsync)
    store = CoreRunBindingStore(serving_fixture.root / "temp-fsync-failure")

    with pytest.raises(GfmProxyError):
        store.save(binding)
    assert invoked == 1
    assert not (store.root / f"{binding.run_id}.json").exists()
    assert not (store.root / f"{binding.run_id}.anchor.json").exists()
    assert not list(store.root.glob("*.tmp"))


@pytest.mark.anyio
async def test_binding_publication_depends_on_real_parent_directory_flush(
    serving_fixture: ServingFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    binding = await _publication_candidate(serving_fixture)
    module = importlib.import_module("app.gfm_core_serving_control")
    invoked = 0
    if os.name == "nt":
        assert hasattr(module, "_FlushFileBuffers")

        def fail_directory_flush(_handle: int) -> int:
            nonlocal invoked
            invoked += 1
            ctypes.set_last_error(5)
            return 0

        monkeypatch.setattr(module, "_FlushFileBuffers", fail_directory_flush)
    else:
        real_fsync = module.os.fsync

        def fail_directory_fsync(descriptor: int) -> None:
            nonlocal invoked
            if stat.S_ISDIR(os.fstat(descriptor).st_mode):
                invoked += 1
                raise OSError("injected binding directory fsync failure")
            real_fsync(descriptor)

        monkeypatch.setattr(module.os, "fsync", fail_directory_fsync)
    store = CoreRunBindingStore(serving_fixture.root / "directory-flush-failure")

    with pytest.raises(GfmProxyError):
        store.save(binding)
    assert invoked == 1
    assert not (store.root / f"{binding.run_id}.anchor.json").exists()


@pytest.mark.anyio
async def test_binding_publication_calls_the_real_held_safe_reopen(
    serving_fixture: ServingFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    binding = await _publication_candidate(serving_fixture)
    module = importlib.import_module("app.gfm_core_serving_control")
    final_name = f"{binding.run_id}.json"
    invoked = 0
    if os.name == "nt":
        real_create_file = module._CreateFileW

        def fail_final_open(
            path: str,
            access: int,
            share: int,
            security: Any,
            creation: int,
            flags: int,
            template: Any,
        ) -> int:
            nonlocal invoked
            if str(path).endswith(final_name) and access & 0x80000000:
                invoked += 1
                ctypes.set_last_error(5)
                return module._INVALID
            return real_create_file(
                path, access, share, security, creation, flags, template
            )

        monkeypatch.setattr(module, "_CreateFileW", fail_final_open)
    else:
        real_open = module.os.open

        def fail_final_open(
            path: str | bytes | Path,
            flags: int,
            mode: int = 0o777,
            *,
            dir_fd: int | None = None,
        ) -> int:
            nonlocal invoked
            if os.fspath(path) == final_name and dir_fd is not None:
                invoked += 1
                raise OSError("injected held final open failure")
            if dir_fd is None:
                return real_open(path, flags, mode)
            return real_open(path, flags, mode, dir_fd=dir_fd)

        monkeypatch.setattr(module.os, "open", fail_final_open)
    store = CoreRunBindingStore(serving_fixture.root / "safe-reopen-failure")

    with pytest.raises(GfmProxyError):
        store.save(binding)
    assert invoked == 1
    assert not (store.root / f"{binding.run_id}.anchor.json").exists()


@pytest.mark.anyio
@pytest.mark.parametrize("record_kind", ["binding", "anchor"])
async def test_binding_publication_rejects_post_publish_replacement_or_bad_hash(
    serving_fixture: ServingFixture,
    monkeypatch: pytest.MonkeyPatch,
    record_kind: str,
) -> None:
    binding = await _publication_candidate(serving_fixture)
    module = importlib.import_module("app.gfm_client")
    real_link = module.os.link
    replaced = 0

    def replace_after_link(source: str | Path, destination: str | Path, **kwargs: Any) -> None:
        nonlocal replaced
        real_link(source, destination, **kwargs)
        destination_path = Path(destination)
        is_anchor = destination_path.name.endswith(".anchor.json")
        if (record_kind == "anchor") == is_anchor:
            replaced += 1
            if record_kind == "binding":
                payload = binding.model_dump(mode="json", by_alias=True)
                payload["bindingHash"] = HASHES["f"]
                hostile = (
                    json.dumps(payload, separators=(",", ":"), sort_keys=True) + "\n"
                ).encode("utf-8")
            else:
                hostile = b"{}\n"
            replacement = destination_path.with_name(
                f".{destination_path.name}.replacement"
            )
            replacement.write_bytes(hostile)
            os.replace(replacement, destination_path)

    monkeypatch.setattr(module.os, "link", replace_after_link)
    store = CoreRunBindingStore(serving_fixture.root / f"replace-{record_kind}")

    with pytest.raises(GfmProxyError):
        store.save(binding)
    assert replaced == 1


@pytest.mark.anyio
async def test_save_reverifies_binding_after_anchor_publication(
    serving_fixture: ServingFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    binding = await _publication_candidate(serving_fixture)
    module = importlib.import_module("app.gfm_client")
    real_link = module.os.link
    replaced = 0

    def replace_binding_after_anchor(
        source: str | Path, destination: str | Path, **kwargs: Any
    ) -> None:
        nonlocal replaced
        real_link(source, destination, **kwargs)
        destination_path = Path(destination)
        if destination_path.name.endswith(".anchor.json"):
            replaced += 1
            binding_path = destination_path.with_name(f"{binding.run_id}.json")
            replacement = destination_path.with_name(".late-binding-replacement")
            replacement.write_bytes(b"{}\n")
            os.replace(replacement, binding_path)

    monkeypatch.setattr(module.os, "link", replace_binding_after_anchor)
    store = CoreRunBindingStore(serving_fixture.root / "late-binding-replacement")

    with pytest.raises(GfmProxyError):
        store.save(binding)
    assert replaced == 1


@pytest.mark.anyio
@pytest.mark.parametrize("record_kind", ["binding", "anchor"])
async def test_exact_retry_rejects_noncanonical_existing_record_bytes(
    serving_fixture: ServingFixture,
    record_kind: str,
) -> None:
    binding = await _publication_candidate(serving_fixture)
    store = CoreRunBindingStore(serving_fixture.root / f"noncanonical-{record_kind}")
    binding_path = store.root / f"{binding.run_id}.json"
    anchor_path = store.root / f"{binding.run_id}.anchor.json"
    if record_kind == "binding":
        binding_path.write_text(
            json.dumps(binding.model_dump(mode="json", by_alias=True), indent=2),
            encoding="utf-8",
        )
    else:
        binding_path.write_bytes(_expected_canonical_record_bytes(binding))
        anchor = _binding_anchor(binding)
        anchor_path.write_text(
            json.dumps(anchor.model_dump(mode="json", by_alias=True), indent=2),
            encoding="utf-8",
        )

    with pytest.raises(GfmProxyError):
        store.save(binding)


@pytest.mark.anyio
async def test_different_concurrent_publishers_use_real_no_overwrite_conflict(
    serving_fixture: ServingFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = await _publication_candidate(serving_fixture)
    changed = CoreRunBinding.model_validate_json(
        json.dumps(
            _coherently_substitute_complete_binding(
                original.model_dump(mode="json", by_alias=True)
            )
        )
    )
    store = CoreRunBindingStore(serving_fixture.root / "different-concurrent")
    module = importlib.import_module("app.gfm_client")
    real_link = module.os.link
    barrier = threading.Barrier(2)
    binding_link_calls = 0

    def synchronized_link(source: str | Path, destination: str | Path, **kwargs: Any) -> None:
        nonlocal binding_link_calls
        if not str(destination).endswith(".anchor.json"):
            binding_link_calls += 1
            barrier.wait(timeout=5)
        real_link(source, destination, **kwargs)

    monkeypatch.setattr(module.os, "link", synchronized_link)

    def publish(candidate: CoreRunBinding) -> CoreRunBinding | GfmProxyError:
        try:
            store.save(candidate)
            return candidate
        except GfmProxyError as error:
            return error

    with ThreadPoolExecutor(max_workers=2) as executor:
        outcomes = list(executor.map(publish, (original, changed)))

    winners = [value for value in outcomes if isinstance(value, CoreRunBinding)]
    failures = [value for value in outcomes if isinstance(value, GfmProxyError)]
    assert binding_link_calls == 2
    assert len(winners) == 1
    assert len(failures) == 1
    assert failures[0].code == "GFM_CORE_RUN_BINDING_INVALID"
    assert store.get(original.run_id) == winners[0]


@pytest.mark.anyio
async def test_post_publish_native_reparse_is_rejected_when_supported(
    serving_fixture: ServingFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    binding = await _publication_candidate(serving_fixture)
    module = importlib.import_module("app.gfm_client")
    real_link = module.os.link
    target = serving_fixture.root / "binding-target.json"
    target.write_bytes(_expected_canonical_record_bytes(binding))
    installed = False

    def install_reparse(source: str | Path, destination: str | Path, **kwargs: Any) -> None:
        nonlocal installed
        real_link(source, destination, **kwargs)
        destination_path = Path(destination)
        if not destination_path.name.endswith(".anchor.json"):
            destination_path.unlink()
            try:
                destination_path.symlink_to(target)
            except OSError:
                pytest.skip("native symlink/reparse creation is unavailable")
            installed = True

    monkeypatch.setattr(module.os, "link", install_reparse)
    store = CoreRunBindingStore(serving_fixture.root / "post-link-reparse")

    with pytest.raises(GfmProxyError):
        store.save(binding)
    assert installed


@pytest.mark.anyio
async def test_public_create_returns_no_run_id_and_counts_one_orphan_on_reopen_failure(
    serving_fixture: ServingFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = RecordingGfmClient(serving_fixture.capabilities)
    fake.create_response = lambda envelope: _complete_api_receipt(
        serving_fixture, envelope
    )
    module = importlib.import_module("app.gfm_client")
    real_link = module.os.link

    def remove_binding_after_link(
        source: str | Path, destination: str | Path, **kwargs: Any
    ) -> None:
        real_link(source, destination, **kwargs)
        destination_path = Path(destination)
        if not destination_path.name.endswith(".anchor.json"):
            destination_path.unlink()

    monkeypatch.setattr(module.os, "link", remove_binding_after_link)
    app = create_app(serving_fixture.settings, gfm_client=fake)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(
        transport=transport, base_url="http://testserver"
    ) as client:
        response = await client.post("/api/v1/gfm/runs", json=_public_request())

    assert response.status_code == 502
    assert response.json()["detail"]["code"] == "GFM_CORE_CREATE_RECEIPT_PERSIST_FAILED"
    assert "runId" not in response.json()
    assert app.state.core_gateway.diagnostics() == {
        "code": "GFM_CORE_INTERNAL_ORPHANED_CREATE_COUNT",
        "count": 1,
    }


def _replace_binding_with_coherent_substitution(binding_path: Path) -> None:
    changed = _coherently_substitute_complete_binding(
        json.loads(binding_path.read_bytes())
    )
    replacement = CoreRunBinding.model_validate_json(json.dumps(changed))
    replacement_path = binding_path.with_name(f".{binding_path.name}.late")
    replacement_path.write_bytes(_expected_canonical_record_bytes(replacement))
    os.replace(replacement_path, binding_path)


@pytest.mark.anyio
async def test_save_rejects_binding_replaced_during_final_anchor_reread(
    serving_fixture: ServingFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = RecordingGfmClient(serving_fixture.capabilities)
    fake.create_response = lambda envelope: _complete_api_receipt(
        serving_fixture, envelope
    )
    module = importlib.import_module("app.gfm_core_serving_control")
    anchor_reads = 0

    def replace_after_old_binding_reread(stage: str, path: Path) -> None:
        nonlocal anchor_reads
        if stage != "after-open" or not path.name.endswith(".anchor.json"):
            return
        anchor_reads += 1
        if anchor_reads == 2:
            binding_path = path.with_name(path.name.removesuffix(".anchor.json") + ".json")
            _replace_binding_with_coherent_substitution(binding_path)

    monkeypatch.setattr(
        module, "_CORE_PATH_WALK_SEAM", replace_after_old_binding_reread
    )
    app = create_app(serving_fixture.settings, gfm_client=fake)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(
        transport=transport, base_url="http://testserver"
    ) as client:
        response = await client.post("/api/v1/gfm/runs", json=_public_request())

    # The third capture proves the failed save still owns the exact anchor
    # identity before removing its uncommitted publication.
    assert anchor_reads == 3
    assert response.status_code == 502
    assert response.json()["detail"]["code"] == "GFM_CORE_CREATE_RECEIPT_PERSIST_FAILED"
    assert "runId" not in response.json()
    assert app.state.core_gateway.diagnostics() == {
        "code": "GFM_CORE_INTERNAL_ORPHANED_CREATE_COUNT",
        "count": 1,
    }


@pytest.mark.anyio
async def test_get_rejects_binding_replaced_during_anchor_reread(
    serving_fixture: ServingFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _fake, receipt, binding_path = await _create_persisted_complete_binding(
        serving_fixture
    )
    module = importlib.import_module("app.gfm_core_serving_control")
    anchor_reads = 0

    def replace_after_old_binding_read(stage: str, path: Path) -> None:
        nonlocal anchor_reads
        if stage == "after-open" and path.name.endswith(".anchor.json"):
            anchor_reads += 1
            _replace_binding_with_coherent_substitution(binding_path)

    monkeypatch.setattr(module, "_CORE_PATH_WALK_SEAM", replace_after_old_binding_read)
    store = CoreRunBindingStore(binding_path.parent)

    with pytest.raises(GfmProxyError) as invalid:
        store.get(receipt["status"]["runId"])
    assert anchor_reads == 1
    assert invalid.value.code == "GFM_CORE_RUN_BINDING_INVALID"


@pytest.mark.anyio
@pytest.mark.parametrize("retry_kind", ["exact", "mismatched"])
async def test_anchor_without_binding_retry_is_rejected_without_filesystem_mutation(
    serving_fixture: ServingFixture,
    retry_kind: str,
) -> None:
    original = await _publication_candidate(serving_fixture)
    changed = CoreRunBinding.model_validate_json(
        json.dumps(
            _coherently_substitute_complete_binding(
                original.model_dump(mode="json", by_alias=True)
            )
        )
    )
    candidate = original if retry_kind == "exact" else changed
    store = CoreRunBindingStore(serving_fixture.root / f"reverse-state-{retry_kind}")
    anchor = _binding_anchor(original)
    anchor_path = store.root / f"{original.run_id}.anchor.json"
    anchor_path.write_bytes(_expected_canonical_record_bytes(anchor))
    before = {
        path.name: path.read_bytes()
        for path in sorted(store.root.iterdir(), key=lambda item: item.name)
    }

    with pytest.raises(GfmProxyError) as invalid:
        store.save(candidate)

    after = {
        path.name: path.read_bytes()
        for path in sorted(store.root.iterdir(), key=lambda item: item.name)
    }
    assert invalid.value.code == "GFM_CORE_RUN_BINDING_INVALID"
    assert after == before


@pytest.mark.anyio
@pytest.mark.parametrize("retry_kind", ["exact", "mismatched"])
async def test_retry_does_not_recreate_binding_deleted_after_successful_preflight(
    serving_fixture: ServingFixture,
    monkeypatch: pytest.MonkeyPatch,
    retry_kind: str,
) -> None:
    initial_fake = RecordingGfmClient(serving_fixture.capabilities)
    initial_fake.create_response = lambda envelope: _complete_api_receipt(
        serving_fixture, envelope
    )
    initial_app = create_app(serving_fixture.settings, gfm_client=initial_fake)
    initial_transport = httpx.ASGITransport(app=initial_app)
    async with httpx.AsyncClient(
        transport=initial_transport, base_url="http://testserver"
    ) as client:
        initial = await client.post("/api/v1/gfm/runs", json=_public_request())
    assert initial.status_code == 202
    run_id = initial.json()["runId"]
    binding_root = serving_fixture.root / "api-run-bindings"
    binding_path = binding_root / f"{run_id}.json"
    anchor_path = binding_root / f"{run_id}.anchor.json"
    assert binding_path.is_file()
    assert anchor_path.is_file()

    retry_fake = RecordingGfmClient(serving_fixture.capabilities)
    retry_fake.create_response = lambda envelope: _complete_api_receipt(
        serving_fixture, envelope, run_id=run_id
    )
    retry_app = create_app(serving_fixture.settings, gfm_client=retry_fake)
    real_exists = Path.exists
    binding_exists_calls = 0
    state_after_deletion: dict[str, bytes] | None = None

    def delete_after_binding_preflight(path: Path) -> bool:
        nonlocal binding_exists_calls, state_after_deletion
        exists = real_exists(path)
        if path == binding_path and exists:
            binding_exists_calls += 1
            if binding_exists_calls == 2:
                path.unlink()
                state_after_deletion = {
                    item.name: item.read_bytes()
                    for item in sorted(
                        binding_root.iterdir(), key=lambda value: value.name
                    )
                }
        return exists

    monkeypatch.setattr(Path, "exists", delete_after_binding_preflight)
    request = _public_request()
    if retry_kind == "mismatched":
        request["parameters"]["topKSimilarCases"] = 4
    retry_transport = httpx.ASGITransport(app=retry_app)
    async with httpx.AsyncClient(
        transport=retry_transport, base_url="http://testserver"
    ) as client:
        response = await client.post("/api/v1/gfm/runs", json=request)

    final_state = {
        item.name: item.read_bytes()
        for item in sorted(binding_root.iterdir(), key=lambda value: value.name)
    }
    assert state_after_deletion is not None
    assert state_after_deletion == {anchor_path.name: anchor_path.read_bytes()}
    assert response.status_code == 502
    assert response.json()["detail"]["code"] == "GFM_CORE_CREATE_RECEIPT_PERSIST_FAILED"
    assert "runId" not in response.json()
    assert retry_app.state.core_gateway.diagnostics() == {
        "code": "GFM_CORE_INTERNAL_ORPHANED_CREATE_COUNT",
        "count": 1,
    }
    assert final_state == state_after_deletion


@pytest.mark.anyio
@pytest.mark.parametrize("retry_kind", ["exact", "mismatched"])
async def test_binding_only_retry_does_not_republish_binding_deleted_after_preflight(
    serving_fixture: ServingFixture,
    monkeypatch: pytest.MonkeyPatch,
    retry_kind: str,
) -> None:
    initial_fake = RecordingGfmClient(serving_fixture.capabilities)
    initial_fake.create_response = lambda envelope: _complete_api_receipt(
        serving_fixture, envelope
    )
    initial_app = create_app(serving_fixture.settings, gfm_client=initial_fake)
    initial_transport = httpx.ASGITransport(app=initial_app)
    async with httpx.AsyncClient(
        transport=initial_transport, base_url="http://testserver"
    ) as client:
        initial = await client.post("/api/v1/gfm/runs", json=_public_request())
    assert initial.status_code == 202
    run_id = initial.json()["runId"]
    binding_root = serving_fixture.root / "api-run-bindings"
    binding_path = binding_root / f"{run_id}.json"
    anchor_path = binding_root / f"{run_id}.anchor.json"
    anchor_path.unlink()
    assert binding_path.is_file()
    assert not anchor_path.exists()

    retry_fake = RecordingGfmClient(serving_fixture.capabilities)
    retry_fake.create_response = lambda envelope: _complete_api_receipt(
        serving_fixture, envelope, run_id=run_id
    )
    retry_app = create_app(serving_fixture.settings, gfm_client=retry_fake)
    real_exists = Path.exists
    binding_exists_calls = 0
    state_after_deletion: dict[str, bytes] | None = None

    def delete_after_binding_only_preflight(path: Path) -> bool:
        nonlocal binding_exists_calls, state_after_deletion
        exists = real_exists(path)
        if path == binding_path and exists:
            binding_exists_calls += 1
            if binding_exists_calls == 2:
                path.unlink()
                state_after_deletion = {
                    item.name: item.read_bytes()
                    for item in sorted(
                        binding_root.iterdir(), key=lambda value: value.name
                    )
                }
        return exists

    monkeypatch.setattr(Path, "exists", delete_after_binding_only_preflight)
    request = _public_request()
    if retry_kind == "mismatched":
        request["parameters"]["topKSimilarCases"] = 4
    retry_transport = httpx.ASGITransport(app=retry_app)
    async with httpx.AsyncClient(
        transport=retry_transport, base_url="http://testserver"
    ) as client:
        response = await client.post("/api/v1/gfm/runs", json=request)

    final_state = {
        item.name: item.read_bytes()
        for item in sorted(binding_root.iterdir(), key=lambda value: value.name)
    }
    assert state_after_deletion == {}
    assert response.status_code == 502
    assert response.json()["detail"]["code"] == "GFM_CORE_CREATE_RECEIPT_PERSIST_FAILED"
    assert "runId" not in response.json()
    assert retry_app.state.core_gateway.diagnostics() == {
        "code": "GFM_CORE_INTERNAL_ORPHANED_CREATE_COUNT",
        "count": 1,
    }
    assert final_state == state_after_deletion


@pytest.mark.anyio
async def test_concurrent_exact_binding_only_repairs_reach_real_anchor_link_conflict(
    serving_fixture: ServingFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    binding = await _publication_candidate(serving_fixture)
    store = CoreRunBindingStore(serving_fixture.root / "concurrent-anchor-repair")
    binding_path = store.root / f"{binding.run_id}.json"
    binding_path.write_bytes(_expected_canonical_record_bytes(binding))
    module = importlib.import_module("app.gfm_client")
    real_link = module.os.link
    barrier = threading.Barrier(2)
    anchor_link_calls = 0

    def synchronized_anchor_link(
        source: str | Path, destination: str | Path, **kwargs: Any
    ) -> None:
        nonlocal anchor_link_calls
        if str(destination).endswith(".anchor.json"):
            anchor_link_calls += 1
            barrier.wait(timeout=5)
        real_link(source, destination, **kwargs)

    monkeypatch.setattr(module.os, "link", synchronized_anchor_link)

    def repair() -> GfmProxyError | None:
        try:
            store.save(binding)
        except GfmProxyError as error:
            return error
        return None

    with ThreadPoolExecutor(max_workers=2) as executor:
        outcomes = list(executor.map(lambda _index: repair(), range(2)))

    assert outcomes == [None, None]
    assert anchor_link_calls == 2
    assert store.get(binding.run_id) == binding


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("candidate_kind", "top_k"),
    [("exact", 3), ("mismatched", 4)],
)
async def test_owned_anchor_is_removed_when_binding_disappears_before_real_link(
    serving_fixture: ServingFixture,
    monkeypatch: pytest.MonkeyPatch,
    candidate_kind: str,
    top_k: int,
) -> None:
    request = _public_request()
    request["parameters"]["topKSimilarCases"] = top_k
    run_id = "00000000-0000-0000-0000-000000000001"
    seed_root = serving_fixture.root / f"handoff-seed-{candidate_kind}"
    seed_settings = serving_fixture.settings.model_copy(
        update={"gfm_core_run_binding_root": str(seed_root)}
    )
    seed_fake = RecordingGfmClient(serving_fixture.capabilities)
    seed_fake.create_response = lambda envelope: _complete_api_receipt(
        serving_fixture, envelope, run_id=run_id
    )
    seed_app = create_app(seed_settings, gfm_client=seed_fake)
    seed_transport = httpx.ASGITransport(app=seed_app)
    async with httpx.AsyncClient(
        transport=seed_transport, base_url="http://testserver"
    ) as client:
        seeded = await client.post("/api/v1/gfm/runs", json=request)
    assert seeded.status_code == 202

    binding_root = serving_fixture.root / f"handoff-under-test-{candidate_kind}"
    binding_root.mkdir()
    binding_path = binding_root / f"{run_id}.json"
    anchor_path = binding_root / f"{run_id}.anchor.json"
    binding_path.write_bytes((seed_root / binding_path.name).read_bytes())
    retry_settings = serving_fixture.settings.model_copy(
        update={"gfm_core_run_binding_root": str(binding_root)}
    )
    retry_fake = RecordingGfmClient(serving_fixture.capabilities)
    retry_fake.create_response = lambda envelope: _complete_api_receipt(
        serving_fixture, envelope, run_id=run_id
    )
    retry_app = create_app(retry_settings, gfm_client=retry_fake)
    module = importlib.import_module("app.gfm_client")
    real_link = module.os.link
    real_flush = module._flush_parent_directory
    anchor_link_calls = 0
    cleanup_flushes = 0
    state_after_deletion: dict[str, bytes] | None = None

    def delete_binding_before_anchor_link(
        source: str | Path, destination: str | Path, **kwargs: Any
    ) -> None:
        nonlocal anchor_link_calls, state_after_deletion
        destination_path = Path(destination)
        if destination_path == anchor_path:
            anchor_link_calls += 1
            binding_path.unlink()
            state_after_deletion = {
                item.name: item.read_bytes()
                for item in sorted(binding_root.iterdir(), key=lambda value: value.name)
                if not item.name.endswith(".tmp")
            }
        real_link(source, destination, **kwargs)

    def observe_cleanup_flush(path: Path) -> None:
        nonlocal cleanup_flushes
        real_flush(path)
        if state_after_deletion is not None and not anchor_path.exists():
            cleanup_flushes += 1

    monkeypatch.setattr(module.os, "link", delete_binding_before_anchor_link)
    monkeypatch.setattr(module, "_flush_parent_directory", observe_cleanup_flush)
    retry_transport = httpx.ASGITransport(app=retry_app)
    async with httpx.AsyncClient(
        transport=retry_transport, base_url="http://testserver"
    ) as client:
        response = await client.post("/api/v1/gfm/runs", json=request)

    final_state = {
        item.name: item.read_bytes()
        for item in sorted(binding_root.iterdir(), key=lambda value: value.name)
    }
    assert anchor_link_calls == 1
    assert cleanup_flushes == 1
    assert state_after_deletion == {}
    assert response.status_code == 502
    assert response.json()["detail"]["code"] == "GFM_CORE_CREATE_RECEIPT_PERSIST_FAILED"
    assert "runId" not in response.json()
    assert retry_app.state.core_gateway.diagnostics() == {
        "code": "GFM_CORE_INTERNAL_ORPHANED_CREATE_COUNT",
        "count": 1,
    }
    assert final_state == state_after_deletion


@pytest.mark.anyio
async def test_failed_handoff_never_removes_concurrent_anchor_winner(
    serving_fixture: ServingFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    binding = await _publication_candidate(serving_fixture)
    store = CoreRunBindingStore(serving_fixture.root / "concurrent-anchor-handoff")
    binding_path = store.root / f"{binding.run_id}.json"
    anchor_path = store.root / f"{binding.run_id}.anchor.json"
    binding_path.write_bytes(_expected_canonical_record_bytes(binding))
    concurrent_source = store.root / ".concurrent-anchor"
    concurrent_source.write_bytes(
        _expected_canonical_record_bytes(_binding_anchor(binding))
    )
    module = importlib.import_module("app.gfm_client")
    real_link = module.os.link
    state_after_deletion: dict[str, bytes] | None = None

    def publish_concurrent_winner_first(
        source: str | Path, destination: str | Path, **kwargs: Any
    ) -> None:
        nonlocal state_after_deletion
        destination_path = Path(destination)
        if destination_path == anchor_path:
            real_link(concurrent_source, destination_path)
            binding_path.unlink()
            state_after_deletion = {anchor_path.name: anchor_path.read_bytes()}
        real_link(source, destination, **kwargs)

    monkeypatch.setattr(module.os, "link", publish_concurrent_winner_first)

    with pytest.raises(GfmProxyError) as invalid:
        store.save(binding)

    assert invalid.value.code == "GFM_CORE_RUN_BINDING_INVALID"
    assert state_after_deletion is not None
    assert {anchor_path.name: anchor_path.read_bytes()} == state_after_deletion
    assert {item.name for item in store.root.iterdir()} == {
        anchor_path.name,
        concurrent_source.name,
    }
    assert not [
        item
        for item in store.root.iterdir()
        if item.name.endswith((".tmp", ".cleanup"))
    ]


@pytest.mark.anyio
@pytest.mark.parametrize("substitution_kind", ["same-bytes-new-identity", "different-bytes"])
async def test_owned_anchor_cleanup_never_removes_a_substituted_record(
    serving_fixture: ServingFixture,
    monkeypatch: pytest.MonkeyPatch,
    substitution_kind: str,
) -> None:
    binding = await _publication_candidate(serving_fixture)
    store = CoreRunBindingStore(
        serving_fixture.root / f"substituted-anchor-cleanup-{substitution_kind}"
    )
    binding_path = store.root / f"{binding.run_id}.json"
    anchor_path = store.root / f"{binding.run_id}.anchor.json"
    binding_path.write_bytes(_expected_canonical_record_bytes(binding))
    module = importlib.import_module("app.gfm_core_serving_control")
    binding_reads = 0
    substituted_state: dict[str, bytes] | None = None

    def substitute_anchor_before_final_binding_open(stage: str, path: Path) -> None:
        nonlocal binding_reads, substituted_state
        if stage != "before-open" or path != binding_path:
            return
        binding_reads += 1
        if binding_reads != 2:
            return
        replacement = store.root / ".substituted-anchor"
        replacement.write_bytes(
            _expected_canonical_record_bytes(_binding_anchor(binding))
            if substitution_kind == "same-bytes-new-identity"
            else b"{}\n"
        )
        os.replace(replacement, anchor_path)
        binding_path.unlink()
        substituted_state = {anchor_path.name: anchor_path.read_bytes()}

    monkeypatch.setattr(
        module, "_CORE_PATH_WALK_SEAM", substitute_anchor_before_final_binding_open
    )

    with pytest.raises(GfmProxyError) as invalid:
        store.save(binding)

    assert invalid.value.code == "GFM_CORE_RUN_BINDING_INVALID"
    assert binding_reads == 2
    assert substituted_state is not None
    assert {
        item.name: item.read_bytes()
        for item in store.root.iterdir()
    } == substituted_state


@pytest.mark.anyio
@pytest.mark.parametrize("substitution_kind", ["new-identity", "same-identity-bytes"])
async def test_owned_anchor_cleanup_rechecks_identity_after_cleanup_proof(
    serving_fixture: ServingFixture,
    monkeypatch: pytest.MonkeyPatch,
    substitution_kind: str,
) -> None:
    binding = await _publication_candidate(serving_fixture)
    store = CoreRunBindingStore(serving_fixture.root / "cleanup-proof-substitution")
    binding_path = store.root / f"{binding.run_id}.json"
    anchor_path = store.root / f"{binding.run_id}.anchor.json"
    binding_path.write_bytes(_expected_canonical_record_bytes(binding))
    module = importlib.import_module("app.gfm_client")
    real_link = module.os.link
    real_require = store._require_same_record
    anchor_proofs = 0
    substituted_state: dict[str, bytes] | None = None

    def delete_binding_before_anchor_link(
        source: str | Path, destination: str | Path, **kwargs: Any
    ) -> None:
        destination_path = Path(destination)
        if destination_path == anchor_path:
            binding_path.unlink()
        real_link(source, destination, **kwargs)

    def substitute_after_cleanup_proof(
        path: Path,
        record: CoreRunBinding | CoreRunBindingAnchor,
        *,
        expected_bytes: bytes | None = None,
    ) -> tuple[int, int]:
        nonlocal anchor_proofs, substituted_state
        identity = real_require(path, record, expected_bytes=expected_bytes)
        if path == anchor_path:
            anchor_proofs += 1
            if anchor_proofs == 3:
                if substitution_kind == "new-identity":
                    replacement = store.root / ".after-proof-substitution"
                    replacement.write_bytes(b"{}\n")
                    os.replace(replacement, anchor_path)
                else:
                    anchor_path.write_bytes(b"{}\n")
                substituted_state = {anchor_path.name: anchor_path.read_bytes()}
        return identity

    monkeypatch.setattr(module.os, "link", delete_binding_before_anchor_link)
    monkeypatch.setattr(store, "_require_same_record", substitute_after_cleanup_proof)

    with pytest.raises(GfmProxyError) as invalid:
        store.save(binding)

    assert invalid.value.code == "GFM_CORE_RUN_BINDING_INVALID"
    assert anchor_proofs == 3
    assert substituted_state is not None
    assert {
        item.name: item.read_bytes()
        for item in store.root.iterdir()
    } == substituted_state


@pytest.mark.anyio
@pytest.mark.skipif(os.name != "nt", reason="Windows held-delete behavior")
async def test_windows_cleanup_holds_identity_through_delete_disposition(
    serving_fixture: ServingFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    binding = await _publication_candidate(serving_fixture)
    store = CoreRunBindingStore(serving_fixture.root / "held-delete-disposition")
    binding_path = store.root / f"{binding.run_id}.json"
    anchor_path = store.root / f"{binding.run_id}.anchor.json"
    binding_path.write_bytes(_expected_canonical_record_bytes(binding))
    replacement = store.root / ".delete-disposition-substitution"
    replacement.write_bytes(b"{}\n")
    client_module = importlib.import_module("app.gfm_client")
    serving_module = importlib.import_module("app.gfm_core_serving_control")
    real_link = client_module.os.link
    real_set_information = serving_module._kernel32.SetFileInformationByHandle
    disposition_calls = 0
    substitution_blocked = False

    def delete_binding_before_anchor_link(
        source: str | Path, destination: str | Path, **kwargs: Any
    ) -> None:
        destination_path = Path(destination)
        if destination_path == anchor_path:
            binding_path.unlink()
        real_link(source, destination, **kwargs)

    def attempt_substitution_at_delete_disposition(
        handle: int,
        information_class: int,
        information: Any,
        size: int,
    ) -> int:
        nonlocal disposition_calls, substitution_blocked
        disposition_calls += 1
        try:
            os.replace(replacement, anchor_path)
        except PermissionError:
            substitution_blocked = True
        return real_set_information(handle, information_class, information, size)

    monkeypatch.setattr(client_module.os, "link", delete_binding_before_anchor_link)
    monkeypatch.setattr(
        serving_module._kernel32,
        "SetFileInformationByHandle",
        attempt_substitution_at_delete_disposition,
    )

    with pytest.raises(GfmProxyError) as invalid:
        store.save(binding)

    assert invalid.value.code == "GFM_CORE_RUN_BINDING_INVALID"
    assert disposition_calls == 1
    assert substitution_blocked
    assert not anchor_path.exists()
    assert replacement.read_bytes() == b"{}\n"
    assert {item.name for item in store.root.iterdir()} == {replacement.name}
