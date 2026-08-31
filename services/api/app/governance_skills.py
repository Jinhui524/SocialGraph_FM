"""Compatibility exports for the split SocialGraph-FM Governance skill API runtime."""

from .governance_skill_runtime.gateway import (
    GovernanceSkillsClientProtocol,
    GovernanceSkillsGateway,
)

__all__ = ["GovernanceSkillsClientProtocol", "GovernanceSkillsGateway"]
