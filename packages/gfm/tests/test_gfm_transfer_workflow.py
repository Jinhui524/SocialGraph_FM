from __future__ import annotations

import numpy as np
import pytest

from socialgraph_gfm.gfm.transfer_workflow import (
    DOMAIN_FAMILY_BY_ID,
    LodoIsolationAudit,
    assert_lodo_isolation,
    few_shot_indices,
    load_lodo_shared_backbone,
    select_core_variant,
    select_formal_checkpoints,
)


def _hash(character: str) -> str:
    return character * 64


def _audit() -> LodoIsolationAudit:
    return LodoIsolationAudit(
        held_out_family="academic-collaboration",
        source_domain_ids=(
            "thgl-software-2.0.0",
            "wikimedia-talk-article-2011-2015",
        ),
        target_domain_ids=("openalex-graph-ai",),
        source_corpus_hashes=(_hash("a"), _hash("b")),
        target_corpus_hashes=(_hash("c"),),
        verified_corpus_hashes=(_hash("a"), _hash("b"), _hash("c")),
        adapter_statistic_hashes=(_hash("d"), _hash("e")),
        excluded_academic_sibling_ids=("ogbl-collab", "ogbn-arxiv"),
        academic_sibling_access_count=0,
        academic_sibling_exclusion_evidence_hash=_hash("f"),
        target_adapter_initialized_after_pretraining=True,
    )


def test_lodo_isolation_rejects_same_family_source() -> None:
    assert len(assert_lodo_isolation(_audit())) == 64
    invalid = LodoIsolationAudit(
        **{**_audit().__dict__, "source_domain_ids": ("openalex-graph-ai",)}
    )
    with pytest.raises(ValueError, match="held-out|isolation|source domains"):
        assert_lodo_isolation(invalid)


def test_lodo_isolation_requires_verified_disjoint_hashes_and_sibling_evidence() -> None:
    reordered = LodoIsolationAudit(
        **{
            **_audit().__dict__,
            "source_domain_ids": tuple(reversed(_audit().source_domain_ids)),
            "source_corpus_hashes": tuple(reversed(_audit().source_corpus_hashes)),
            "verified_corpus_hashes": tuple(reversed(_audit().verified_corpus_hashes)),
        }
    )
    assert assert_lodo_isolation(reordered) == assert_lodo_isolation(_audit())
    with pytest.raises(ValueError, match="verified corpus inventory"):
        assert_lodo_isolation(
            LodoIsolationAudit(
                **{
                    **_audit().__dict__,
                    "verified_corpus_hashes": (_hash("a"), _hash("b"), _hash("9")),
                }
            )
        )
    with pytest.raises(ValueError, match="access count"):
        assert_lodo_isolation(
            LodoIsolationAudit(**{**_audit().__dict__, "academic_sibling_access_count": 1})
        )
    with pytest.raises(ValueError, match="independent identity"):
        assert_lodo_isolation(
            LodoIsolationAudit(
                **{
                    **_audit().__dict__,
                    "academic_sibling_exclusion_evidence_hash": _hash("a"),
                }
            )
        )
    with pytest.raises(ValueError, match="lowercase SHA-256"):
        assert_lodo_isolation(
            LodoIsolationAudit(
                **{**_audit().__dict__, "source_corpus_hashes": (_hash("A"), _hash("b"))}
            )
        )


def test_few_shot_selection_is_stratified_and_deterministic() -> None:
    labels = np.asarray([0] * 100 + [1] * 100 + [2] * 20)
    roles = np.asarray(["train"] * 200 + ["validation"] * 10 + ["test"] * 10)
    times = np.asarray([100] * 200 + [101] * 20)
    hashes = np.asarray([_hash("a")] * labels.size)
    arguments = dict(
        fraction=0.05,
        seed=20260821,
        split_roles=roles,
        event_times=times,
        cutoff_time=100,
        sample_corpus_hashes=hashes,
        expected_corpus_hash=_hash("a"),
    )
    first = few_shot_indices(labels, **arguments)
    second = few_shot_indices(labels, **arguments)
    assert np.array_equal(first, second)
    assert first.size == 10
    assert np.bincount(labels[first]).tolist() == [5, 5]
    assert bool(np.all(roles[first] == "train"))


def test_few_shot_selection_rejects_future_train_and_foreign_corpus() -> None:
    arguments = dict(
        labels=np.asarray([0, 0, 1, 1]),
        fraction=0.05,
        seed=1,
        split_roles=np.asarray(["train", "train", "validation", "test"]),
        event_times=np.asarray([10, 11, 20, 30]),
        cutoff_time=10,
        sample_corpus_hashes=np.asarray([_hash("a")] * 4),
        expected_corpus_hash=_hash("a"),
    )
    with pytest.raises(ValueError, match="after the cutoff"):
        few_shot_indices(**arguments)
    arguments["event_times"] = np.asarray([10, 10, 20, 30])
    arguments["sample_corpus_hashes"] = np.asarray([_hash("a"), _hash("b"), _hash("a"), _hash("a")])
    with pytest.raises(ValueError, match="expected corpus"):
        few_shot_indices(**arguments)
    arguments["sample_corpus_hashes"] = np.asarray([_hash("a")] * 4)
    arguments["seed"] = 1.5
    with pytest.raises(ValueError, match="seed"):
        few_shot_indices(**arguments)


def test_moe_promotion_requires_both_fixed_gates() -> None:
    domains = tuple(DOMAIN_FAMILY_BY_ID)
    base = dict.fromkeys(domains, 0.5)
    promoted = select_core_variant(
        base_by_domain=base,
        moe_by_domain=dict.fromkeys(domains, 0.515),
    )
    assert promoted.selected == "core-moe"
    rejected = select_core_variant(
        base_by_domain=base,
        moe_by_domain={domains[0]: 0.54, domains[1]: 0.54, domains[2]: 0.49},
    )
    assert rejected.selected == "core-base"
    assert rejected.maximum_domain_regression > 0.01
    with pytest.raises(ValueError, match="fixed domain IDs"):
        select_core_variant(
            base_by_domain={"a": 0.5, "b": 0.5, "c": 0.5},
            moe_by_domain={"a": 0.6, "b": 0.6, "c": 0.6},
        )


def test_formal_checkpoint_selection_requires_three_verified_seeds() -> None:
    config_hash, code_hash, environment_hash = _hash("a"), _hash("b"), _hash("c")
    corpus_hashes = (_hash("d"), _hash("e"), _hash("f"))
    records = [
        {
            "variant": "core-base",
            "seed": seed,
            "checkpointId": f"run-best-{seed}",
            "freshProcessDigest": str(index) * 64,
            "freshProcessVerified": True,
            "checkpointRole": "best",
            "phase": "formal",
            "configHash": config_hash,
            "codeHash": code_hash,
            "environmentHash": environment_hash,
            "corpusHashes": corpus_hashes,
        }
        for index, seed in enumerate((20260821, 20260822, 20260823), start=1)
    ]
    assert select_formal_checkpoints(
        records,
        selected_variant="core-base",
        expected_config_hash=config_hash,
        expected_code_hash=code_hash,
        expected_environment_hash=environment_hash,
        expected_corpus_hashes=corpus_hashes,
    ) == tuple(f"run-best-{seed}" for seed in (20260821, 20260822, 20260823))

    invalid = [dict(value) for value in records]
    invalid[0]["checkpointRole"] = "latest"
    with pytest.raises(ValueError, match="verified best-state provenance"):
        select_formal_checkpoints(
            invalid,
            selected_variant="core-base",
            expected_config_hash=config_hash,
            expected_code_hash=code_hash,
            expected_environment_hash=environment_hash,
            expected_corpus_hashes=corpus_hashes,
        )
    invalid = [dict(value) for value in records]
    invalid[0]["freshProcessDigest"] = "A" * 64
    with pytest.raises(ValueError, match="verified best-state provenance"):
        select_formal_checkpoints(
            invalid,
            selected_variant="core-base",
            expected_config_hash=config_hash,
            expected_code_hash=code_hash,
            expected_environment_hash=environment_hash,
            expected_corpus_hashes=corpus_hashes,
        )


def test_lodo_backbone_keeps_target_adapter_new() -> None:
    torch = pytest.importorskip("torch")
    from socialgraph_gfm.gfm.model import SocialGraphFMCore
    from socialgraph_gfm.gfm.types import CoreModelConfig

    common = dict(
        modality_dims={"numeric": 3},
        num_relations=2,
        variant="base",
        hidden_channels=16,
        time_channels=8,
        relation_bases=8,
    )
    torch.manual_seed(1)
    source = SocialGraphFMCore(CoreModelConfig(domains=("source",), **common))
    torch.manual_seed(2)
    target = SocialGraphFMCore(CoreModelConfig(domains=("target",), **common))
    adapter_before = {
        name: value.clone()
        for name, value in target.state_dict().items()
        if "domain_adapter" in name
    }
    loaded = load_lodo_shared_backbone(target, source.state_dict())
    assert loaded
    assert all("domain_adapter" not in name for name in loaded)
    assert all(
        torch.equal(target.state_dict()[name], value) for name, value in adapter_before.items()
    )
