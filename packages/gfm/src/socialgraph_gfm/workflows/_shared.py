"""Shared dependencies and invariant identifiers for staged GFM workflows.

Concrete orchestration bodies live in the neighboring responsibility modules.
They import this namespace before the package aggregator wires cross-stage
helpers, preserving the former fail-closed workflow behavior without a single
implementation module.
"""

# ruff: noqa: F401
from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import uuid
from collections import OrderedDict, defaultdict
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

import numpy as np

from ..canonical import canonical_sha256, file_sha256
from ..checkpoint import restore_rng_state
from ..errors import (
    ContractViolation,
    GfmAcceptanceRejected,
    GfmTrainingError,
    RegistrationRejected,
)
from ..gfm.checkpoint import (
    load_gfm_checkpoint,
    read_gfm_checkpoint_manifest,
    save_gfm_checkpoint,
)
from ..gfm.configuration import (
    apply_exploratory_overrides,
    load_core_config,
    load_openalex_spec,
)
from ..gfm.contracts import (
    GfmDomainCorpusManifest,
    GfmEvaluationReport,
    GfmPretrainConfig,
    GfmRunManifest,
    GfmTaskProtocolManifest,
)
from ..gfm.corpus import (
    EmbeddingConfig,
    OpenAlexConfig,
    build_bge_m3_embeddings,
    check_all_gfm_corpora,
    check_openalex_newcomers,
    fetch_openalex,
    fetch_thgl_software,
    fetch_wikimedia,
    load_domain,
    load_domain_view,
    load_openalex_newcomers_view,
    newcomer_overlay_status,
    open_embedding_artifact,
    open_embedding_artifact_view,
    prepare_openalex,
    prepare_thgl_software,
    prepare_wikimedia,
    verify_openalex_newcomers,
)
from ..gfm.corpus.common import (
    atomic_write_json,
    atomic_write_npz,
    exclusive_file_lock,
    load_npz_safe,
    portable_id_hash,
    read_json_object,
)
from ..gfm.lodo_execution import (
    HEARTBEAT_EVERY_OPTIMIZER_STEPS,
    LodoCellIdentity,
    bind_lodo_role_views,
    bind_lodo_selected_indices,
    commit_lodo_progress,
    complete_lodo_stage,
    create_lodo_run_state,
    exclusive_lodo_execution_lock,
    load_lodo_resume_checkpoint,
    mark_lodo_succeeded,
    persist_lodo_run_state,
    record_lodo_heartbeat,
    validate_lodo_run_state,
)
from ..gfm.registry import GfmRegistry
from ..identity import code_identity_hash
from ..runtime import (
    RuntimeLayout,
    prepare_runtime_layout,
    require_gfm_optional_runtime,
    require_ml_runtime,
    runtime_report,
    set_seed,
)

DomainAlias = Literal["openalex", "thgl-software", "wikimedia-talk"]
TrainingPhase = Literal["dev", "formal"]
ProductTask = Literal["collaboration", "newcomer"]
EvaluationProtocol = Literal["lodo", "product", "shadow"]
NewcomerOverlayMode = Literal["skip", "require"]
ValidationScope = Literal["pretraining", "full"]

DOMAIN_IDS: dict[DomainAlias, str] = {
    "openalex": "openalex-graph-ai",
    "thgl-software": "thgl-software-2.0.0",
    "wikimedia-talk": "wikimedia-talk-article-2011-2015",
}
DOMAIN_ALIASES = {value: key for key, value in DOMAIN_IDS.items()}
DOMAIN_RELATION_OFFSETS = {
    "openalex-graph-ai": 0,
    "thgl-software-2.0.0": 5,
    "wikimedia-talk-article-2011-2015": 19,
}
TOTAL_RELATIONS = 20
TEXT_DOMAINS = (
    "openalex-graph-ai",
    "wikimedia-talk-article-2011-2015",
)
REGISTRY_NAME = "gfm-registry.sqlite3"
COLLABORATION_TASK = "governance.collaboration_recommendation"
NEWCOMER_TASK = "core.newcomer_support"

# Workflow stage modules intentionally need private helpers and imported runtime
# dependencies.  An explicit runtime export list keeps star-import behavior
# deterministic while the compatibility aggregator wires cross-stage symbols.
__all__ = [name for name in globals() if not name.startswith("__")]
