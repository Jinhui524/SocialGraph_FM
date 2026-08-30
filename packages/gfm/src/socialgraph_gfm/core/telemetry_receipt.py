"""Operator-authenticated receipts for persisted formal resource telemetry.

The existing resource telemetry hash chain detects accidental mutation, but it is
not an authenticity proof because an artifact producer can recompute its hashes.
This module adds a canonical HMAC-SHA256 receipt issued by a controlled runner and
verified offline against an explicitly provisioned operator capability.

Threat boundary: the receipt prevents an artifact/API caller without the operator
secret from forging or rebinding telemetry.  It does not protect against a process
that can read the capability's memory, an operator-secret disclosure, or a host
administrator.  Secret persistence, rotation, and filesystem/secret-store ACLs are
deployment responsibilities; artifact bytes never provide a trust root.
"""

from __future__ import annotations

import hashlib
import hmac
import re
from dataclasses import dataclass
from typing import Any, Literal, SupportsIndex

from pydantic import BaseModel, ConfigDict, Field, model_validator

from socialgraph_gfm.canonical import canonical_bytes, canonical_sha256

from .resource_telemetry import (
    ResourceTelemetryRecord,
    VerifiedResourceTelemetry,
    verify_resource_telemetry,
)


_SCHEMA = "socialgraph-fm.core-telemetry-receipt/1.0"
_TRUST_SCHEMA = "socialgraph-fm.core-telemetry-operator-trust/1.0"
_SIGNATURE_DOMAIN = "socialgraph-fm.core-telemetry-receipt:hmac-sha256:v1"
_MAC_ALGORITHM = "HMAC-SHA256"
_HASH = r"^[0-9a-f]{64}$"
_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:/-]{0,199}$")
_MIN_SECRET_BYTES = 32
_MIN_FORMAL_STEPS = 2_000
_MAX_FORMAL_STEPS = 10_000
_MAX_ELAPSED_NS = 21_600 * 1_000_000_000
_MAX_CUDA_BYTES = int(6.5 * 1024**3)


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, populate_by_name=True, strict=True)


def _require_identifier(value: object, *, name: str) -> str:
    if type(value) is not str or _IDENTIFIER.fullmatch(value) is None:
        raise ValueError(f"{name} has invalid syntax")
    return value


def _record_hash(record: ResourceTelemetryRecord) -> str:
    return canonical_sha256(record.model_dump(mode="python", by_alias=True))


class OperatorTelemetryCapability:
    """Non-serializable operator capability shared only with controlled code.

    Creating the capability is the sole raw-secret boundary.  Signing and verification
    APIs accept this object, never caller-provided secret bytes.  The copied secret is
    intentionally absent from repr, schemas, receipts, and public properties.
    """

    __slots__ = ("_key_id", "_secret", "_trust_root_hash")

    def __init__(self, *, key_id: str, secret: bytes) -> None:
        if type(secret) is not bytes or len(secret) < _MIN_SECRET_BYTES:
            raise ValueError("operator secret must contain at least 32 bytes")
        self._key_id = _require_identifier(key_id, name="keyId")
        self._secret = bytearray(secret)
        commitment = self._mac(
            canonical_bytes(
                {
                    "schemaVersion": _TRUST_SCHEMA,
                    "role": "operator-secret-commitment",
                    "algorithm": _MAC_ALGORITHM,
                    "keyId": self._key_id,
                }
            )
        )
        self._trust_root_hash = canonical_sha256(
            {
                "schemaVersion": _TRUST_SCHEMA,
                "algorithm": _MAC_ALGORITHM,
                "keyId": self._key_id,
                "keyCommitment": commitment,
            }
        )

    @classmethod
    def from_secret(
        cls, *, key_id: str, secret: bytes | bytearray | memoryview | object
    ) -> OperatorTelemetryCapability:
        if not isinstance(secret, (bytes, bytearray, memoryview)):
            raise TypeError("operator secret must contain at least 32 bytes")
        copied = bytes(secret)
        if len(copied) < _MIN_SECRET_BYTES:
            raise ValueError("operator secret must contain at least 32 bytes")
        return cls(key_id=key_id, secret=copied)

    @property
    def key_id(self) -> str:
        return self._key_id

    @property
    def trust_root_hash(self) -> str:
        return self._trust_root_hash

    def _mac(self, message: bytes) -> str:
        return hmac.new(self._secret, message, hashlib.sha256).hexdigest()

    def __repr__(self) -> str:
        return (
            "OperatorTelemetryCapability("
            f"key_id={self._key_id!r}, trust_root_hash={self._trust_root_hash!r}, secret=<redacted>)"
        )

    def __reduce__(self) -> str | tuple[Any, ...]:
        raise TypeError("operator telemetry capability must not be serialized")

    def __reduce_ex__(self, protocol: SupportsIndex) -> str | tuple[Any, ...]:
        del protocol
        raise TypeError("operator telemetry capability must not be serialized")


class TelemetryReceipt(_StrictModel):
    """Persisted signed identity of one exact formal telemetry record."""

    schema_version: Literal["socialgraph-fm.core-telemetry-receipt/1.0"] = Field(
        alias="schemaVersion"
    )
    mac_algorithm: Literal["HMAC-SHA256"] = Field(alias="macAlgorithm")
    key_id: str = Field(alias="keyId", pattern=r"^[A-Za-z0-9][A-Za-z0-9._:/-]{0,199}$")
    trust_root_hash: str = Field(alias="trustRootHash", pattern=_HASH)
    cell_id: str = Field(alias="cellId", pattern=_HASH)
    fold_id: str = Field(alias="foldId", pattern=r"^[A-Za-z0-9][A-Za-z0-9._:/-]{0,199}$")
    run_id: str = Field(alias="runId", pattern=r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,199}$")
    phase: Literal["formal"]
    config_hash: str = Field(alias="configHash", pattern=_HASH)
    data_hash: str = Field(alias="dataHash", pattern=_HASH)
    code_hash: str = Field(alias="codeHash", pattern=_HASH)
    environment_hash: str = Field(alias="environmentHash", pattern=_HASH)
    telemetry_hash: str = Field(alias="telemetryHash", pattern=_HASH)
    telemetry_record_hash: str = Field(alias="telemetryRecordHash", pattern=_HASH)
    latest_checkpoint_semantic_hash: str = Field(
        alias="latestCheckpointSemanticHash", pattern=_HASH
    )
    best_checkpoint_semantic_hash: str = Field(alias="bestCheckpointSemanticHash", pattern=_HASH)
    final_model_state_hash: str = Field(alias="finalModelStateHash", pattern=_HASH)
    final_fit_state_hash: str = Field(alias="finalFitStateHash", pattern=_HASH)
    composite_state_hash: str = Field(alias="compositeStateHash", pattern=_HASH)
    recovery_state_hash: str = Field(alias="recoveryStateHash", pattern=_HASH)
    final_optimizer_step: int = Field(alias="finalOptimizerStep", ge=1, le=_MAX_FORMAL_STEPS)
    elapsed_ns: int = Field(alias="elapsedNs", gt=0)
    data_wait_ns: int = Field(alias="dataWaitNs", ge=0)
    peak_cuda_bytes: int = Field(alias="peakCudaBytes", ge=0)
    mac: str = Field(pattern=_HASH)
    receipt_hash: str = Field(alias="receiptHash", pattern=_HASH)

    @model_validator(mode="after")
    def validate_receipt_hash(self) -> TelemetryReceipt:
        expected = canonical_sha256(
            self.model_dump(mode="python", by_alias=True, exclude={"receipt_hash"})
        )
        if self.receipt_hash != expected:
            raise ValueError("receiptHash does not match the telemetry receipt")
        return self


class TelemetryReceiptExpectations(_StrictModel):
    """Acceptance-owned identities; none are taken from receipt trust claims."""

    cell_id: str = Field(alias="cellId", pattern=_HASH)
    fold_id: str = Field(alias="foldId", pattern=r"^[A-Za-z0-9][A-Za-z0-9._:/-]{0,199}$")
    run_id: str = Field(alias="runId", pattern=r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,199}$")
    config_hash: str = Field(alias="configHash", pattern=_HASH)
    data_hash: str = Field(alias="dataHash", pattern=_HASH)
    code_hash: str = Field(alias="codeHash", pattern=_HASH)
    environment_hash: str = Field(alias="environmentHash", pattern=_HASH)
    telemetry_record_hash: str = Field(alias="telemetryRecordHash", pattern=_HASH)
    latest_checkpoint_semantic_hash: str = Field(
        alias="latestCheckpointSemanticHash", pattern=_HASH
    )
    best_checkpoint_semantic_hash: str = Field(alias="bestCheckpointSemanticHash", pattern=_HASH)
    final_model_state_hash: str = Field(alias="finalModelStateHash", pattern=_HASH)
    final_fit_state_hash: str = Field(alias="finalFitStateHash", pattern=_HASH)
    composite_state_hash: str = Field(alias="compositeStateHash", pattern=_HASH)
    recovery_state_hash: str = Field(alias="recoveryStateHash", pattern=_HASH)


@dataclass(frozen=True)
class TrustedTelemetryVerification:
    """Successful offline signature, identity, and fixed-resource-gate result."""

    record_hash: str
    receipt_hash: str
    key_id: str
    trust_root_hash: str
    cell_id: str
    fold_id: str
    run_id: str
    optimizer_steps: int
    elapsed_ns: int
    data_wait_ns: int
    peak_cuda_bytes: int


def _receipt_payload(
    *,
    record: ResourceTelemetryRecord,
    fold_id: str,
    capability: OperatorTelemetryCapability,
    composite_state_hash: str,
    recovery_state_hash: str,
) -> dict[str, Any]:
    return {
        "schemaVersion": _SCHEMA,
        "macAlgorithm": _MAC_ALGORITHM,
        "keyId": capability.key_id,
        "trustRootHash": capability.trust_root_hash,
        "cellId": record.cell_id,
        "foldId": fold_id,
        "runId": record.run_id,
        "phase": record.phase,
        "configHash": record.config_hash,
        "dataHash": record.data_hash,
        "codeHash": record.code_hash,
        "environmentHash": record.environment_hash,
        "telemetryHash": record.telemetry_hash,
        "telemetryRecordHash": _record_hash(record),
        "latestCheckpointSemanticHash": record.latest_checkpoint_semantic_hash,
        "bestCheckpointSemanticHash": record.best_checkpoint_semantic_hash,
        "finalModelStateHash": record.final_model_state_hash,
        "finalFitStateHash": record.final_fit_state_hash,
        "compositeStateHash": composite_state_hash,
        "recoveryStateHash": recovery_state_hash,
        "finalOptimizerStep": record.final_optimizer_step,
        "elapsedNs": record.elapsed_ns,
        "dataWaitNs": record.data_wait_ns,
        "peakCudaBytes": record.peak_cuda_bytes,
    }


def _signing_bytes(payload: dict[str, Any]) -> bytes:
    return canonical_bytes({"signatureDomain": _SIGNATURE_DOMAIN, "receipt": payload})


class TelemetryReceiptSigner:
    """Controlled-runner signer; raw secret bytes are never accepted by issue()."""

    __slots__ = ("_capability",)

    def __init__(self, capability: OperatorTelemetryCapability) -> None:
        if type(capability) is not OperatorTelemetryCapability:
            raise TypeError("telemetry signer requires an exact operator capability")
        self._capability = capability

    def issue(
        self,
        *,
        telemetry: VerifiedResourceTelemetry,
        fold_id: str,
        composite_state_hash: str,
        recovery_state_hash: str,
    ) -> TelemetryReceipt:
        record = verify_resource_telemetry(telemetry)
        if record.phase != "formal":
            raise ValueError("operator receipt signing requires formal telemetry")
        normalized_fold_id = _require_identifier(fold_id, name="foldId")
        if (
            re.fullmatch(_HASH, composite_state_hash) is None
            or re.fullmatch(_HASH, recovery_state_hash) is None
        ):
            raise ValueError("telemetry receipt state identities must be lowercase SHA-256")
        payload = _receipt_payload(
            record=record,
            fold_id=normalized_fold_id,
            capability=self._capability,
            composite_state_hash=composite_state_hash,
            recovery_state_hash=recovery_state_hash,
        )
        payload["mac"] = self._capability._mac(_signing_bytes(payload))
        payload["receiptHash"] = canonical_sha256(payload)
        return TelemetryReceipt.model_validate(payload)


class TrustedTelemetryPolicy:
    """Offline verifier pinned to one explicitly provisioned operator capability."""

    __slots__ = ("_capability",)

    def __init__(self, capability: OperatorTelemetryCapability) -> None:
        if type(capability) is not OperatorTelemetryCapability:
            raise TypeError("trusted telemetry policy requires an exact operator capability")
        self._capability = capability

    @staticmethod
    def _validate_exact_models(
        *,
        record: ResourceTelemetryRecord,
        receipt: TelemetryReceipt,
        expected: TelemetryReceiptExpectations,
    ) -> None:
        if type(record) is not ResourceTelemetryRecord:
            raise TypeError("telemetry verification requires an exact ResourceTelemetryRecord")
        if type(receipt) is not TelemetryReceipt:
            raise TypeError("telemetry verification requires an exact TelemetryReceipt")
        if type(expected) is not TelemetryReceiptExpectations:
            raise TypeError("telemetry verification requires exact acceptance expectations")
        ResourceTelemetryRecord.model_validate(record.model_dump(mode="python", by_alias=True))
        TelemetryReceipt.model_validate(receipt.model_dump(mode="python", by_alias=True))
        TelemetryReceiptExpectations.model_validate(
            expected.model_dump(mode="python", by_alias=True)
        )

    @staticmethod
    def _apply_fixed_resource_gates(record: ResourceTelemetryRecord) -> None:
        if not _MIN_FORMAL_STEPS <= record.final_optimizer_step <= _MAX_FORMAL_STEPS:
            raise ValueError("formal optimizer steps must be between 2000 and 10000")
        if record.elapsed_ns > _MAX_ELAPSED_NS:
            raise ValueError("formal telemetry exceeds the fixed six-hour limit")
        if record.peak_cuda_bytes > _MAX_CUDA_BYTES:
            raise ValueError("formal telemetry exceeds the fixed 6.5 GiB CUDA limit")
        if record.data_wait_ns * 5 >= record.elapsed_ns:
            raise ValueError("formal telemetry data-wait ratio must remain below 20%")

    def verify(
        self,
        *,
        record: ResourceTelemetryRecord,
        receipt: TelemetryReceipt,
        expected: TelemetryReceiptExpectations,
    ) -> TrustedTelemetryVerification:
        self._validate_exact_models(record=record, receipt=receipt, expected=expected)
        if (
            receipt.key_id != self._capability.key_id
            or receipt.trust_root_hash != self._capability.trust_root_hash
        ):
            raise ValueError("telemetry receipt is not bound to the trusted operator key")

        unsigned = receipt.model_dump(mode="python", by_alias=True, exclude={"mac", "receipt_hash"})
        observed_mac = self._capability._mac(_signing_bytes(unsigned))
        if not hmac.compare_digest(receipt.mac, observed_mac):
            raise ValueError("telemetry receipt operator MAC is invalid")

        expected_payload = _receipt_payload(
            record=record,
            fold_id=receipt.fold_id,
            capability=self._capability,
            composite_state_hash=receipt.composite_state_hash,
            recovery_state_hash=receipt.recovery_state_hash,
        )
        if unsigned != expected_payload:
            raise ValueError("telemetry receipt does not bind the exact telemetry record")

        if (
            receipt.cell_id != expected.cell_id
            or receipt.fold_id != expected.fold_id
            or receipt.run_id != expected.run_id
            or receipt.config_hash != expected.config_hash
            or receipt.data_hash != expected.data_hash
            or receipt.code_hash != expected.code_hash
            or receipt.environment_hash != expected.environment_hash
            or receipt.telemetry_record_hash != expected.telemetry_record_hash
        ):
            raise ValueError("telemetry receipt identity differs from acceptance evidence")
        if (
            receipt.latest_checkpoint_semantic_hash != expected.latest_checkpoint_semantic_hash
            or receipt.best_checkpoint_semantic_hash != expected.best_checkpoint_semantic_hash
            or receipt.final_model_state_hash != expected.final_model_state_hash
            or receipt.final_fit_state_hash != expected.final_fit_state_hash
            or receipt.composite_state_hash != expected.composite_state_hash
            or receipt.recovery_state_hash != expected.recovery_state_hash
        ):
            raise ValueError(
                "telemetry receipt checkpoint identity differs from acceptance evidence"
            )

        self._apply_fixed_resource_gates(record)
        return TrustedTelemetryVerification(
            record_hash=receipt.telemetry_record_hash,
            receipt_hash=receipt.receipt_hash,
            key_id=receipt.key_id,
            trust_root_hash=receipt.trust_root_hash,
            cell_id=receipt.cell_id,
            fold_id=receipt.fold_id,
            run_id=receipt.run_id,
            optimizer_steps=record.final_optimizer_step,
            elapsed_ns=record.elapsed_ns,
            data_wait_ns=record.data_wait_ns,
            peak_cuda_bytes=record.peak_cuda_bytes,
        )


__all__ = [
    "OperatorTelemetryCapability",
    "TelemetryReceipt",
    "TelemetryReceiptExpectations",
    "TelemetryReceiptSigner",
    "TrustedTelemetryPolicy",
    "TrustedTelemetryVerification",
]
