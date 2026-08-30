from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from socialgraph_gfm.errors import ContractViolation
from socialgraph_gfm.gfm.corpus.common import (
    NumericShardWriter,
    atomic_write_json,
    build_manifest,
)
from socialgraph_gfm.gfm.corpus.domains import (
    ACCESS_ROLES,
    PHYSICAL_ACCESS_SCHEMA,
    load_domain_view,
)


def _role_corpus(root: Path) -> tuple[Path, dict[str, object]]:
    output = root / "datasets/processed/gfm/openalex-graph-ai"
    output.mkdir(parents=True)
    event_records = {}
    target_records = {}
    for index, role in enumerate(ACCESS_ROLES):
        event_records[role] = NumericShardWriter(
            output, prefix=f"rv-e-{role}", rows_per_shard=2
        ).write(
            {
                "src": np.asarray([index], dtype=np.int64),
                "dst": np.asarray([index + 1], dtype=np.int64),
                "timestamp": np.asarray([100 + index], dtype=np.int64),
                "relation": np.asarray([0], dtype=np.int16),
            }
        )
        target_records[role] = NumericShardWriter(
            output, prefix=f"rv-t-{role}", rows_per_shard=2
        ).write(
            {
                "src": np.asarray([index], dtype=np.int64),
                "dst": np.asarray([index + 1], dtype=np.int64),
                "timestamp": np.asarray([200 + index], dtype=np.int64),
                "first_collaboration": np.asarray([True], dtype=np.bool_),
            }
        )
    records = tuple(event_records.values()) + tuple(target_records.values())
    manifest = build_manifest(
        schema_version="gfm.openalex-corpus/1.0",
        corpus_id="openalex-graph-ai",
        license_id="CC0-1.0",
        source={"fixture": True},
        shards=records,
        splits={"fixture": True},
        privacy={},
        extra={
            "domainId": "openalex-graph-ai",
            "physicalAccess": {
                "schemaVersion": PHYSICAL_ACCESS_SCHEMA,
                "roles": list(ACCESS_ROLES),
                "roleFamilies": {
                    "events": {
                        role: [event_records[role].path] for role in ACCESS_ROLES
                    },
                    "targets": {
                        role: [target_records[role].path] for role in ACCESS_ROLES
                    },
                },
                "sharedFamilies": {},
                "mergeOrder": {"events": "timestamp", "targets": "timestamp"},
            },
        },
    )
    atomic_write_json(output / "manifest.json", manifest)
    return output, manifest


def test_validation_view_never_opens_test_or_shadow_bytes(tmp_path: Path) -> None:
    output, manifest = _role_corpus(tmp_path)
    test_path = Path(manifest["physicalAccess"]["roleFamilies"]["events"]["test"][0])
    (output / test_path).write_bytes(b"deliberately corrupt future artifact")

    loaded = load_domain_view(
        tmp_path,
        "openalex-graph-ai",
        maximum_role="validation",
        families=("events",),
    )
    assert loaded["arrays"]["timestamp"].tolist() == [100, 101]
    assert str(test_path) in loaded["accessAudit"]["excludedRestrictedPaths"]
    assert loaded["accessAudit"]["testArtifactsOpened"] is False

    with pytest.raises(ContractViolation, match="selected role artifact hash mismatch"):
        load_domain_view(
            tmp_path,
            "openalex-graph-ai",
            maximum_role="test",
            families=("events",),
        )

    validation_path = Path(
        manifest["physicalAccess"]["roleFamilies"]["events"]["validation"][0]
    )
    (output / validation_path).write_bytes(b"deliberately corrupt validation artifact")
    with pytest.raises(ContractViolation, match="selected role artifact hash mismatch"):
        load_domain_view(
            tmp_path,
            "openalex-graph-ai",
            maximum_role="validation",
            families=("events",),
        )


def test_family_selection_does_not_open_product_labels(tmp_path: Path) -> None:
    output, manifest = _role_corpus(tmp_path)
    target_path = Path(
        manifest["physicalAccess"]["roleFamilies"]["targets"]["train"][0]
    )
    (output / target_path).write_bytes(b"corrupt label bytes")

    event_only = load_domain_view(
        tmp_path,
        "openalex-graph-ai",
        maximum_role="train",
    )
    assert "targets.first_collaboration" not in event_only["arrays"]
    with pytest.raises(ContractViolation, match="selected role artifact hash mismatch"):
        load_domain_view(
            tmp_path,
            "openalex-graph-ai",
            maximum_role="train",
            families=("events", "targets"),
        )
