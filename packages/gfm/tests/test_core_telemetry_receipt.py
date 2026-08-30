from __future__ import annotations

import pickle
from collections.abc import Iterator, Mapping
from typing import Any

import pytest
import torch

from socialgraph_gfm.canonical import canonical_sha256
from socialgraph_gfm.core import resource_telemetry as telemetry_module
from socialgraph_gfm.core.resource_telemetry import (
    ResourceTelemetryRecord,
    ResourceTelemetryRecorder,
    VerifiedResourceTelemetry,
    verify_resource_telemetry,
)
from socialgraph_gfm.core.telemetry_receipt import (
    OperatorTelemetryCapability,
    TelemetryReceipt,
    TelemetryReceiptExpectations,
    TelemetryReceiptSigner,
    TrustedTelemetryPolicy,
)
from socialgraph_gfm.tensor_digest import canonical_tensor_digest


CELL_ID = "1" * 64
CONFIG_HASH = "2" * 64
DATA_HASH = "3" * 64
CODE_HASH = "4" * 64
ENVIRONMENT_HASH = "5" * 64
LATEST_CHECKPOINT_HASH = "6" * 64
BEST_CHECKPOINT_HASH = "7" * 64
FOLD_ID = "tolokers::official-00"
SECRET = bytes(range(32))
COMPOSITE_STATE_HASH = "8" * 64
RECOVERY_STATE_HASH = "9" * 64


def _model_state(step: int) -> dict[str, torch.Tensor]:
    return {"encoder.weight": torch.tensor([[float(step)]], dtype=torch.float32)}


def _fit_state(step: int, state: Mapping[str, torch.Tensor]) -> dict[str, Any]:
    model_hash = canonical_sha256(
        {name: canonical_tensor_digest(value) for name, value in sorted(state.items())}
    )
    payload: dict[str, Any] = {
        "schemaVersion": "socialgraph-fm.test-fit-state/1.0",
        "optimizerStep": step,
        "checkpointModelStateHash": model_hash,
    }
    payload["stateHash"] = canonical_sha256(payload)
    return payload


def _verified_telemetry(
    *,
    final_step: int = 2_000,
    run_id: str = "formal-run-20260821",
    clock: Iterator[int] | None = None,
    wait: bool = False,
    cuda_peak_bytes: int | None = None,
    monkeypatch: pytest.MonkeyPatch | None = None,
) -> VerifiedResourceTelemetry:
    if clock is not None:
        assert monkeypatch is not None
        monkeypatch.setattr(telemetry_module.time, "monotonic_ns", lambda: next(clock))
    recorder = ResourceTelemetryRecorder(
        cell_id=CELL_ID,
        run_id=run_id,
        phase="formal",
        config_hash=CONFIG_HASH,
        data_hash=DATA_HASH,
        code_hash=CODE_HASH,
        environment_hash=ENVIRONMENT_HASH,
    )
    if cuda_peak_bytes is not None:
        assert monkeypatch is not None
        monkeypatch.setattr(recorder, "_cuda_peak", lambda: cuda_peak_bytes)
    state = _model_state(0)
    recorder.record_start(model_state=state, fit_state=_fit_state(0, state))
    if wait:
        with recorder.measure_data_wait():
            pass
    for step in range(250, final_step, 250):
        state = _model_state(step)
        recorder.record_checkpoint(
            optimizer_step=step,
            model_state=state,
            fit_state=_fit_state(step, state),
        )
    state = _model_state(final_step)
    return recorder.finish(
        final_optimizer_step=final_step,
        model_state=state,
        fit_state=_fit_state(final_step, state),
        latest_checkpoint_semantic_hash=LATEST_CHECKPOINT_HASH,
        best_checkpoint_semantic_hash=BEST_CHECKPOINT_HASH,
    )


def _expectations(record: ResourceTelemetryRecord) -> TelemetryReceiptExpectations:
    return TelemetryReceiptExpectations(
        cellId=record.cell_id,
        foldId=FOLD_ID,
        runId=record.run_id,
        configHash=record.config_hash,
        dataHash=record.data_hash,
        codeHash=record.code_hash,
        environmentHash=record.environment_hash,
        telemetryRecordHash=canonical_sha256(record.model_dump(mode="python", by_alias=True)),
        latestCheckpointSemanticHash=record.latest_checkpoint_semantic_hash,
        bestCheckpointSemanticHash=record.best_checkpoint_semantic_hash,
        finalModelStateHash=record.final_model_state_hash,
        finalFitStateHash=record.final_fit_state_hash,
        compositeStateHash=COMPOSITE_STATE_HASH,
        recoveryStateHash=RECOVERY_STATE_HASH,
    )


def _trusted_pair() -> tuple[TelemetryReceiptSigner, TrustedTelemetryPolicy]:
    capability = OperatorTelemetryCapability.from_secret(
        key_id="formal-runner-2026-08", secret=SECRET
    )
    return (
        TelemetryReceiptSigner(capability),
        TrustedTelemetryPolicy(capability),
    )


def _issue(signer: TelemetryReceiptSigner, verified: VerifiedResourceTelemetry) -> TelemetryReceipt:
    return signer.issue(
        telemetry=verified,
        fold_id=FOLD_ID,
        composite_state_hash=COMPOSITE_STATE_HASH,
        recovery_state_hash=RECOVERY_STATE_HASH,
    )


def _rehash_receipt(payload: dict[str, Any]) -> TelemetryReceipt:
    payload["receiptHash"] = canonical_sha256(
        {key: value for key, value in payload.items() if key != "receiptHash"}
    )
    return TelemetryReceipt.model_validate(payload)


def test_operator_signed_exact_formal_telemetry_passes_offline_policy() -> None:
    verified = _verified_telemetry()
    record = verify_resource_telemetry(verified)
    signer, policy = _trusted_pair()

    receipt = _issue(signer, verified)
    result = policy.verify(record=record, receipt=receipt, expected=_expectations(record))

    assert result.record_hash == canonical_sha256(record.model_dump(mode="python", by_alias=True))
    assert result.receipt_hash == receipt.receipt_hash
    assert result.cell_id == CELL_ID
    assert result.fold_id == FOLD_ID
    assert result.optimizer_steps == 2_000


def test_self_hashed_artifact_and_bogus_mac_are_not_operator_proof() -> None:
    verified = _verified_telemetry()
    signer, policy = _trusted_pair()
    receipt = _issue(signer, verified)
    forged_record = verify_resource_telemetry(_verified_telemetry(run_id="caller-forged-self-hash"))
    forged_payload = receipt.model_dump(mode="python", by_alias=True)
    forged_payload.update(
        {
            "runId": forged_record.run_id,
            "telemetryHash": forged_record.telemetry_hash,
            "telemetryRecordHash": canonical_sha256(
                forged_record.model_dump(mode="python", by_alias=True)
            ),
            "mac": "0" * 64,
        }
    )
    forged_receipt = _rehash_receipt(forged_payload)

    with pytest.raises(ValueError, match="operator MAC"):
        policy.verify(
            record=forged_record,
            receipt=forged_receipt,
            expected=_expectations(forged_record),
        )


def test_wrong_operator_key_is_rejected() -> None:
    verified = _verified_telemetry()
    record = verify_resource_telemetry(verified)
    signer, _ = _trusted_pair()
    wrong_capability = OperatorTelemetryCapability.from_secret(
        key_id="formal-runner-2026-08", secret=b"z" * 32
    )
    wrong_policy = TrustedTelemetryPolicy(wrong_capability)

    with pytest.raises(ValueError, match="trusted operator key"):
        wrong_policy.verify(
            record=record,
            receipt=_issue(signer, verified),
            expected=_expectations(record),
        )


def test_receipt_cannot_be_replayed_for_a_different_record() -> None:
    verified = _verified_telemetry()
    signer, policy = _trusted_pair()
    receipt = _issue(signer, verified)
    different_record = verify_resource_telemetry(_verified_telemetry(run_id="different-formal-run"))

    with pytest.raises(ValueError, match="exact telemetry record"):
        policy.verify(
            record=different_record,
            receipt=receipt,
            expected=_expectations(different_record),
        )


@pytest.mark.parametrize(
    "field",
    [
        "latest_checkpoint_semantic_hash",
        "composite_state_hash",
        "recovery_state_hash",
    ],
)
def test_wrong_checkpoint_or_recovery_expectation_is_rejected(field: str) -> None:
    verified = _verified_telemetry()
    record = verify_resource_telemetry(verified)
    signer, policy = _trusted_pair()
    wrong = _expectations(record).model_copy(update={field: "a" * 64})

    with pytest.raises(ValueError, match="checkpoint identity"):
        policy.verify(
            record=record,
            receipt=_issue(signer, verified),
            expected=wrong,
        )


@pytest.mark.parametrize(
    ("verified_factory", "message"),
    [
        (
            lambda monkeypatch: _verified_telemetry(
                clock=iter(index * 2_700_125_000_000 for index in range(9)),
                monkeypatch=monkeypatch,
            ),
            "six-hour",
        ),
        (
            lambda monkeypatch: _verified_telemetry(
                cuda_peak_bytes=int(6.5 * 1024**3) + 1,
                monkeypatch=monkeypatch,
            ),
            "6.5 GiB",
        ),
        (
            lambda monkeypatch: _verified_telemetry(
                clock=iter([0, 100, 400, 500, 600, 700, 800, 900, 1_000, 1_100, 1_200]),
                wait=True,
                monkeypatch=monkeypatch,
            ),
            "data-wait",
        ),
        (
            lambda monkeypatch: _verified_telemetry(final_step=1_750),
            "formal optimizer",
        ),
    ],
    ids=["elapsed", "cuda", "wait-ratio", "formal-min-steps"],
)
def test_operator_signature_does_not_bypass_fixed_resource_gates(
    verified_factory: Any,
    message: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    verified = verified_factory(monkeypatch)
    record = verify_resource_telemetry(verified)
    signer, policy = _trusted_pair()

    with pytest.raises(ValueError, match=message):
        policy.verify(
            record=record,
            receipt=_issue(signer, verified),
            expected=_expectations(record),
        )


def test_operator_secret_capability_is_not_serializable_or_emitted() -> None:
    capability = OperatorTelemetryCapability.from_secret(
        key_id="formal-runner-2026-08", secret=SECRET
    )
    signer = TelemetryReceiptSigner(capability)
    receipt = _issue(signer, _verified_telemetry())

    with pytest.raises(TypeError, match="must not be serialized"):
        pickle.dumps(capability)
    assert SECRET.hex() not in repr(capability)
    assert "secret" not in str(receipt.model_dump(mode="json", by_alias=True)).lower()


@pytest.mark.parametrize("secret", [b"", b"short", "not-bytes"])
def test_operator_capability_rejects_weak_or_non_binary_secrets(secret: object) -> None:
    with pytest.raises((TypeError, ValueError), match="32 bytes"):
        OperatorTelemetryCapability.from_secret(key_id="formal-runner", secret=secret)


def test_direct_capability_constructor_cannot_bypass_minimum_secret_strength() -> None:
    with pytest.raises(ValueError, match="32 bytes"):
        OperatorTelemetryCapability(key_id="formal-runner", secret=b"x")
