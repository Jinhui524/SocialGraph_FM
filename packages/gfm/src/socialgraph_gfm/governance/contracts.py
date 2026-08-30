"""Strict public input contract for SocialGraph-FM Governance online inference bundles."""

from __future__ import annotations

import re
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

SCHEMA_VERSION = "socialgraph-fm.gfm-governance/2.0"
INPUT_SCHEMA_VERSION = "socialgraph-fm.governance-input/2.0"
MODALITIES = ("coRT", "coURL", "hashSeq", "fastRT", "tweetSim")
MAX_NODES = 10_000
MAX_RELATION_ROWS = 500_000
MAX_EVIDENCE_NODES = 300
MAX_EVIDENCE_EDGES = 1_000
MAX_PREVIEW_NODES = 3_000
MAX_PREVIEW_EDGES = 12_000
SHA256_PATTERN = r"^[0-9a-f]{64}$"
ARTIFACT_ID_PATTERN = re.compile(r"^governance-artifact-[0-9a-f]{32}$")
RUN_ID_PATTERN = re.compile(r"^governance-[0-9a-f]{32}$")


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class InputFileDescriptor(_StrictModel):
    sha256: Annotated[str, Field(pattern=SHA256_PATTERN)]
    bytes: Annotated[int, Field(ge=1, le=512 * 1024 * 1024)]


class GovernanceInputManifest(_StrictModel):
    schemaVersion: Literal["socialgraph-fm.governance-input/2.0"]
    datasetId: Annotated[str, Field(pattern=r"^[A-Za-z0-9._:-]{1,100}$")]
    displayName: Annotated[str, Field(min_length=1, max_length=200)]
    nodeCount: Annotated[int, Field(ge=1, le=MAX_NODES)]
    relationRowCount: Annotated[int, Field(ge=1, le=MAX_RELATION_ROWS)]
    featureDimension: Literal[768]
    modalities: tuple[Literal["coRT", "coURL", "hashSeq", "fastRT", "tweetSim"], ...]
    files: dict[str, InputFileDescriptor]
    license: Annotated[str, Field(min_length=1, max_length=200)] | None = None
    sourceUri: Annotated[str, Field(min_length=1, max_length=2048)] | None = None

    @model_validator(mode="after")
    def validate_inventory(self) -> GovernanceInputManifest:
        if set(self.files) != {"nodes.csv", "relations.csv", "features.npz"}:
            raise ValueError("files must bind nodes.csv, relations.csv, and features.npz")
        if not self.modalities or len(self.modalities) != len(set(self.modalities)):
            raise ValueError("modalities must be nonempty and unique")
        if tuple(sorted(self.modalities, key=MODALITIES.index)) != self.modalities:
            raise ValueError("modalities must follow the fixed Governance order")
        if any(ord(character) < 32 for character in self.displayName):
            raise ValueError("displayName contains a control character")
        return self


__all__ = [
    "ARTIFACT_ID_PATTERN",
    "INPUT_SCHEMA_VERSION",
    "MAX_EVIDENCE_EDGES",
    "MAX_EVIDENCE_NODES",
    "MAX_NODES",
    "MAX_PREVIEW_EDGES",
    "MAX_PREVIEW_NODES",
    "MAX_RELATION_ROWS",
    "MODALITIES",
    "RUN_ID_PATTERN",
    "SCHEMA_VERSION",
    "GovernanceInputManifest",
    "InputFileDescriptor",
]
