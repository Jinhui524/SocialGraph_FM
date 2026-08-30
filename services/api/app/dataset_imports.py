"""Backward-compatible import surface for the modular dataset importer.

New code should import from :mod:`app.dataset_import`. Existing routes, tools,
tests, and downstream integrations may continue importing this module.
"""

from .dataset_import import *  # noqa: F401,F403
from .dataset_import import (
    DatasetImportService,
    FewShotJsonNpzAdapter,
    GeomGcnTextDirectoryAdapter,
    GraphVersionTargetDomainAdapter,
    LegacyPlanetoidPickleDetector,
    SafeGraphNpzAdapter,
    SocialGraphDatasetPackageAdapter,
    StrictSplitNpzAdapter,
    TorchPygArchiveDetector,
)

__all__ = [
    "DatasetImportService",
    "FewShotJsonNpzAdapter",
    "GeomGcnTextDirectoryAdapter",
    "GraphVersionTargetDomainAdapter",
    "LegacyPlanetoidPickleDetector",
    "SafeGraphNpzAdapter",
    "SocialGraphDatasetPackageAdapter",
    "StrictSplitNpzAdapter",
    "TorchPygArchiveDetector",
]
