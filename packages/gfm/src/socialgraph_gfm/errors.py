"""Stable machine-readable exceptions for the infrastructure boundary."""


class GfmError(RuntimeError):
    """Base error carrying a stable public error code."""

    code = "GFM_ERROR"

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message

    def as_dict(self) -> dict[str, str]:
        return {"code": self.code, "message": self.message}


class ArtifactRootNotConfigured(GfmError):
    code = "GFM_ARTIFACT_ROOT_NOT_CONFIGURED"


class MissingRuntimeDependency(GfmError):
    code = "GFM_RUNTIME_DEPENDENCY_MISSING"


class RuntimeVersionMismatch(GfmError):
    code = "GFM_RUNTIME_VERSION_MISMATCH"


class ContractViolation(GfmError):
    code = "GFM_CONTRACT_INVALID"


class CheckpointIntegrityError(GfmError):
    code = "GFM_CHECKPOINT_INTEGRITY_ERROR"


class RegistrationRejected(GfmError):
    code = "GFM_MODEL_REGISTRATION_REJECTED"


class RunCancelled(GfmError):
    code = "GFM_RUN_CANCELLED"


class GfmCorpusError(GfmError):
    """A formal multi-domain corpus failed a safety or provenance check."""

    code = "GFM_DOMAIN_CORPUS_INVALID"


class LicenseNotAccepted(GfmCorpusError):
    code = "GFM_CORPUS_LICENSE_NOT_ACCEPTED"


class ExternalDataUnavailable(GfmCorpusError):
    code = "GFM_EXTERNAL_DATA_UNAVAILABLE"


class GfmTrainingError(GfmError):
    code = "GFM_TRAINING_FAILED"


class GfmAcceptanceRejected(GfmError):
    code = "GFM_ACCEPTANCE_REJECTED"
