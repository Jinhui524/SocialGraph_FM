"""Deterministic build/source and smoke-recipe identities."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .canonical import canonical_sha256, file_sha256
from .contracts import TRANSFORM_RECIPE_VERSION
from .materialize import MATERIALIZATION_RECIPE_VERSION

SMOKE_SCHEMA_VERSION = "gfm.smoke/2.0"
SMOKE_CONFIG_VERSION = "gfm.smoke-config/2.0"
DEFAULT_SMOKE_SEED = 20260812


def code_identity_hash() -> str:
    """Hash build-relevant bytes without checkout paths, mtimes or dirty-state metadata."""

    package = Path(__file__).resolve().parent
    project = package.parents[1]
    entries: dict[str, str] = {}
    for path in sorted(package.rglob("*"), key=lambda item: item.as_posix()):
        if path.is_file() and path.suffix in {".py", ".json"} and "__pycache__" not in path.parts:
            entries[f"package/{path.relative_to(package).as_posix()}"] = file_sha256(path)
    checkout_files = (
        "pyproject.toml",
        "runtime-profiles.json",
        "locks/runtime-lock-manifest.json",
        "contracts/public-contracts.schema.json",
        "contracts/public-contracts.full.schema.json",
        "configs/ogbl-collab-baseline.json",
        "configs/socialgraph-core.json",
        "configs/openalex-graph-ai.json",
    )
    for relative in checkout_files:
        path = project / relative
        if path.is_file():
            entries[f"checkout/{relative}"] = file_sha256(path)
    return canonical_sha256(
        {"schemaVersion": "gfm.code-identity/1.0", "files": entries}
    )


def smoke_config(
    *, fixture: str, seed: int, device: str, input_dim: int, hidden_dim: int = 8
) -> dict[str, Any]:
    return {
        "schemaVersion": SMOKE_CONFIG_VERSION,
        "smokeSchemaVersion": SMOKE_SCHEMA_VERSION,
        "materializationRecipeVersion": MATERIALIZATION_RECIPE_VERSION,
        "transformRecipeVersion": TRANSFORM_RECIPE_VERSION,
        "fixture": fixture,
        "seed": seed,
        "device": device,
        "hiddenDim": hidden_dim,
        "inputDim": input_dim,
    }
