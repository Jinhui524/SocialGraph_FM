"""Lightweight SocialGraph-FM Governance service errors (safe without ML dependencies)."""


class GovernanceServiceError(Exception):
    status = 409
    code = "GFM_GOVERNANCE_CONFLICT"


class GovernanceUnavailable(GovernanceServiceError):
    status = 503
    code = "GFM_GOVERNANCE_MODEL_NOT_INSTALLED"


class GovernanceInvalid(GovernanceServiceError):
    status = 422
    code = "GFM_GOVERNANCE_REQUEST_INVALID"


class GovernanceNotFound(GovernanceServiceError):
    status = 404
    code = "GFM_GOVERNANCE_NOT_FOUND"


class GovernanceNotReady(GovernanceServiceError):
    status = 409
    code = "GFM_GOVERNANCE_RESULT_NOT_READY"


class GovernanceAdaptationPolicyNotReady(GovernanceServiceError):
    status = 409
    code = "GFM_GOVERNANCE_ADAPTATION_POLICY_NOT_READY"


__all__ = [
    "GovernanceAdaptationPolicyNotReady",
    "GovernanceInvalid",
    "GovernanceNotFound",
    "GovernanceNotReady",
    "GovernanceServiceError",
    "GovernanceUnavailable",
]
