from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from socialgraph_gfm.cli import build_parser, main
from socialgraph_gfm.errors import ContractViolation, GfmTrainingError
from socialgraph_gfm.gfm_workflow import (
    _DomainStream,
    _PreparedProductBatch,
    _build_lodo_eligible_ordinals,
    _core_batch,
    _formal_corpus_contract,
    _product_task_asset_evidence,
    _lodo_cached_eligible_ordinals,
    _lodo_few_shot_selection,
    _product_audit_counters,
    _recent_causal_edges,
    _stream_role_indices,
    _temporal_audit_counters,
    _visible_edges_between_local_nodes,
    check_gfm_task_assets,
    fetch_gfm_openalex,
    prepare_gfm_corpus,
)
from socialgraph_gfm.runtime import RuntimeLayout


def test_all_public_gfm_commands_have_strict_parsers() -> None:
    parser = build_parser()
    cases = (
        [
            "gfm-corpus-fetch-openalex",
            "--spec",
            "graph-ai",
            "--api-key-env",
            "OPENALEX_API_KEY",
        ],
        ["gfm-corpus-fetch-thgl-software", "--accept-license", "CC-BY-4.0"],
        [
            "gfm-corpus-fetch-wikimedia-talk",
            "--years",
            "2011:2015",
            "--namespace",
            "article",
            "--accept-license",
            "CC0",
        ],
        [
            "gfm-corpus-prepare",
            "--domain",
            "openalex",
            "--newcomer-overlay",
            "skip",
        ],
        ["gfm-task-assets", "--task", "collaboration"],
        [
            "gfm-text-embed",
            "--encoder",
            "BAAI/bge-m3",
            "--domain",
            "openalex",
        ],
        [
            "gfm-pretrain",
            "--phase",
            "dev",
            "--config",
            "socialgraph-core.json",
        ],
        ["gfm-adapt", "--task", "collaboration", "--experiment-id", "experiment"],
        ["gfm-evaluate", "--protocol", "lodo", "--experiment-id", "experiment"],
        ["gfm-resume", "--run-id", "run"],
        ["gfm-validate", "--experiment-id", "experiment"],
        ["gfm-export", "--experiment-id", "experiment"],
    )
    assert [parser.parse_args(value).command for value in cases] == [
        value[0] for value in cases
    ]


def test_cli_dispatch_keeps_pretrain_and_task_scoped_evaluation_arguments_separate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import socialgraph_gfm.gfm_workflow as workflow

    calls: dict[str, dict[str, object]] = {}

    def fake_pretrain(**kwargs):
        calls["pretrain"] = kwargs
        return {"ok": True}

    def fake_evaluate(**kwargs):
        calls["evaluate"] = kwargs
        return {"ok": True}

    monkeypatch.setattr(workflow, "pretrain_gfm", fake_pretrain)
    monkeypatch.setattr(workflow, "evaluate_gfm", fake_evaluate)

    assert main(
        [
            "gfm-pretrain",
            "--phase",
            "dev",
            "--config",
            "socialgraph-core.json",
            "--variant",
            "core-base",
            "--seed",
            "20260820",
        ]
    ) == 0
    assert "task" not in calls["pretrain"]

    assert main(
        [
            "gfm-evaluate",
            "--protocol",
            "product",
            "--task",
            "collaboration",
            "--experiment-id",
            "experiment",
        ]
    ) == 0
    assert calls["evaluate"]["task"] == "collaboration"


def test_corpus_workflow_import_does_not_load_torch() -> None:
    environment = dict(os.environ)
    source = str((Path(__file__).parents[1] / "src").resolve())
    environment["PYTHONPATH"] = os.pathsep.join(
        value for value in (source, environment.get("PYTHONPATH", "")) if value
    )
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import socialgraph_gfm.cli; "
            "import socialgraph_gfm.gfm_workflow; "
            "assert 'torch' not in sys.modules",
        ],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )
    assert result.returncode == 0, result.stderr


def test_formal_corpus_contract_consumes_source_eligibility(tmp_path: Path) -> None:
    manifest = {
        "nodeCounts": {"author": 2, "work": 1},
        "edgeCount": 1,
        "source": {
            "formalEligible": False,
            "licenseEvidence": "official receipt",
            "uri": "https://example.test/openalex",
        },
        "privacy": {"publicCheckpointEligible": True},
        "splits": {"train": "2017-2021"},
        "logicalHash": "1" * 64,
        "licenseId": "CC0",
    }
    contract = _formal_corpus_contract(
        RuntimeLayout(tmp_path), "openalex", manifest
    )
    assert not contract.public_checkpoint_eligible
    assert contract.task_ids == ("governance.collaboration_recommendation",)


def test_openalex_fetch_refuses_before_creating_runtime_without_key(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.delenv("OPENALEX_API_KEY", raising=False)
    root = tmp_path / "runtime"
    with pytest.raises(ContractViolation, match="absent"):
        fetch_gfm_openalex(root=root)
    assert not root.exists()


def test_cli_dispatches_gfm_prepare_as_sorted_json(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    import socialgraph_gfm.gfm_workflow as workflow

    seen: dict[str, object] = {}

    def fake_prepare(
        *, root: str | None, domain: str, newcomer_overlay: str
    ) -> dict[str, object]:
        seen.update(
            {
                "root": root,
                "domain": domain,
                "newcomer_overlay": newcomer_overlay,
            }
        )
        return {"z": 1, "a": "ready"}

    monkeypatch.setattr(workflow, "prepare_gfm_corpus", fake_prepare)
    result = main(
        [
            "gfm-corpus-prepare",
            "--domain",
            "openalex",
            "--newcomer-overlay",
            "skip",
            "--root",
            str(tmp_path),
            "--json",
        ]
    )
    assert result == 0
    assert capsys.readouterr().out.strip() == '{"a": "ready", "z": 1}'
    assert seen == {
        "root": str(tmp_path),
        "domain": "openalex",
        "newcomer_overlay": "skip",
    }


def _task_asset_corpora() -> tuple[SimpleNamespace, ...]:
    return (
        SimpleNamespace(
            domain_id="openalex-graph-ai", logical_hash="a" * 64
        ),
        SimpleNamespace(
            domain_id="thgl-software-2.0.0", logical_hash="b" * 64
        ),
        SimpleNamespace(
            domain_id="wikimedia-talk-article-2011-2015",
            logical_hash="c" * 64,
        ),
    )


def test_collaboration_task_assets_do_not_touch_newcomer_overlay(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import socialgraph_gfm.gfm_workflow as workflow

    def unexpected_overlay_read(_: Path) -> dict[str, object]:
        raise AssertionError("collaboration must not read the newcomer overlay")

    monkeypatch.setattr(
        workflow, "check_openalex_newcomers", unexpected_overlay_read
    )
    evidence = _product_task_asset_evidence(
        RuntimeLayout(tmp_path),
        task="collaboration",
        corpora=_task_asset_corpora(),  # type: ignore[arg-type]
    )

    assert evidence["task"] == "collaboration"
    assert evidence["newcomerOverlay"] is None
    assert evidence["baseCorpusHashes"] == {
        "openalex-graph-ai": "a" * 64,
        "thgl-software-2.0.0": "b" * 64,
        "wikimedia-talk-article-2011-2015": "c" * 64,
    }


def test_newcomer_task_assets_fail_closed_when_overlay_is_missing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import socialgraph_gfm.gfm_workflow as workflow

    def missing_overlay(_: Path) -> dict[str, object]:
        raise ContractViolation("newcomer overlay is absent")

    monkeypatch.setattr(workflow, "check_openalex_newcomers", missing_overlay)
    with pytest.raises(ContractViolation, match="overlay is absent"):
        _product_task_asset_evidence(
            RuntimeLayout(tmp_path),
            task="newcomer",
            corpora=_task_asset_corpora(),  # type: ignore[arg-type]
        )


def test_newcomer_task_assets_bind_overlay_and_base_hashes(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import socialgraph_gfm.gfm_workflow as workflow

    overlay_hash = "d" * 64
    base_source_hash = "e" * 64
    monkeypatch.setattr(
        workflow,
        "check_openalex_newcomers",
        lambda _: {
            "logicalHash": overlay_hash,
            "authorCount": 17,
            "verifiedCount": 17,
            "historyQueryProtocol": "openalex-global-history-v1",
            "source": {
                "baseCorpusId": "openalex-graph-ai",
                "baseCorpusLogicalHash": "a" * 64,
                "baseCorpusSourceHash": base_source_hash,
            },
        },
    )

    evidence = _product_task_asset_evidence(
        RuntimeLayout(tmp_path),
        task="newcomer",
        corpora=_task_asset_corpora(),  # type: ignore[arg-type]
    )

    assert evidence["newcomerOverlay"] == {
        "corpusId": "openalex-graph-ai",
        "baseCorpusLogicalHash": "a" * 64,
        "baseCorpusSourceHash": base_source_hash,
        "overlayLogicalHash": overlay_hash,
        "verifiedCount": 17,
        "historyQueryProtocol": "openalex-global-history-v1",
    }


def test_prepare_openalex_skip_succeeds_without_secret_or_overlay_read(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import socialgraph_gfm.gfm_workflow as workflow

    logical_hash = "f" * 64
    monkeypatch.delenv("OPENALEX_API_KEY", raising=False)
    monkeypatch.setattr(
        workflow, "prepare_runtime_layout", lambda *_args, **_kwargs: RuntimeLayout(tmp_path)
    )
    monkeypatch.setattr(
        workflow, "prepare_openalex", lambda *_args, **_kwargs: {"logicalHash": logical_hash}
    )
    monkeypatch.setattr(
        workflow,
        "load_domain",
        lambda *_args, **_kwargs: {"manifest": {"logicalHash": logical_hash}},
    )
    contract = SimpleNamespace(
        corpus_id="openalex-graph-ai",
        domain_id="academic-collaboration",
        logical_hash="1" * 64,
    )
    monkeypatch.setattr(workflow, "_formal_corpus_contract", lambda *_: contract)
    monkeypatch.setattr(workflow, "_write_contract", lambda *_: None)
    monkeypatch.setattr(
        workflow,
        "_registry",
        lambda _layout: SimpleNamespace(record_corpus=lambda _contract: None),
    )

    def unexpected_overlay_access(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("skip mode must not access or resume the newcomer overlay")

    monkeypatch.setattr(workflow, "verify_openalex_newcomers", unexpected_overlay_access)
    monkeypatch.setattr(workflow, "check_openalex_newcomers", unexpected_overlay_access)

    result = prepare_gfm_corpus(
        root=tmp_path, domain="openalex", newcomer_overlay="skip"
    )

    assert result["ok"] is True
    assert result["newcomerOverlay"]["required"] is False
    assert result["newcomerOverlay"]["ready"] is False
    assert result["newcomerOverlay"]["deferred"] is True
    assert result["newcomerOverlay"]["state"] == "absent"


def test_task_asset_check_can_pass_collaboration_and_defer_newcomer(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import socialgraph_gfm.gfm_workflow as workflow

    monkeypatch.setattr(
        workflow, "prepare_runtime_layout", lambda *_args, **_kwargs: RuntimeLayout(tmp_path)
    )
    monkeypatch.setattr(workflow, "_load_corpus_contracts", lambda _layout: _task_asset_corpora())
    monkeypatch.setattr(
        workflow,
        "check_openalex_newcomers",
        lambda _root: (_ for _ in ()).throw(ContractViolation("overlay absent")),
    )

    result = check_gfm_task_assets(root=tmp_path)

    assert result["ok"] is False
    assert result["tasks"]["collaboration"]["ready"] is True
    assert result["tasks"]["collaboration"]["evidenceHash"]
    assert result["tasks"]["newcomer"]["ready"] is False
    assert "overlay absent" in result["tasks"]["newcomer"]["reason"]


def test_task_asset_cli_returns_success_for_selected_collaboration(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    import socialgraph_gfm.gfm_workflow as workflow

    monkeypatch.setattr(
        workflow,
        "check_gfm_task_assets",
        lambda **_: {"ok": True, "tasks": {"collaboration": {"ready": True}}},
    )
    result = main(["gfm-task-assets", "--task", "collaboration", "--json"])
    assert result == 0
    assert json.loads(capsys.readouterr().out)["tasks"]["collaboration"]["ready"]


def test_text_embed_cli_dispatches_selected_domain(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    import socialgraph_gfm.gfm_workflow as workflow

    seen: dict[str, object] = {}

    def fake_embed(**values: object) -> dict[str, object]:
        seen.update(values)
        return {"ok": True}

    monkeypatch.setattr(workflow, "embed_gfm_text", fake_embed)
    result = main(
        [
            "gfm-text-embed",
            "--encoder",
            "BAAI/bge-m3",
            "--domain",
            "openalex",
            "--json",
        ]
    )
    assert result == 0
    assert seen["domain"] == "openalex"
    assert json.loads(capsys.readouterr().out)["ok"] is True


def test_validate_cli_returns_nonzero_for_failed_derived_gates(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    import socialgraph_gfm.gfm_workflow as workflow

    monkeypatch.setattr(
        workflow,
        "validate_gfm",
        lambda **_: {
            "schemaVersion": "gfm.workflow-validate/1.0",
            "accepted": False,
            "gates": {"product_metrics": False},
        },
    )
    result = main(["gfm-validate", "--experiment-id", "experiment", "--json"])
    assert result == 7
    assert json.loads(capsys.readouterr().out)["accepted"] is False


def test_validate_cli_forwards_independent_pretraining_scope(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    import socialgraph_gfm.gfm_workflow as workflow

    captured: dict[str, object] = {}

    def fake_validate(**values: object) -> dict[str, object]:
        captured.update(values)
        return {
            "schemaVersion": "gfm.workflow-validate/1.0",
            "scope": "pretraining",
            "accepted": True,
            "gates": {"formal_pretrain_matrix": True},
        }

    monkeypatch.setattr(workflow, "validate_gfm", fake_validate)
    result = main(
        [
            "gfm-validate",
            "--experiment-id",
            "experiment",
            "--scope",
            "pretraining",
            "--json",
        ]
    )

    assert result == 0
    assert captured["scope"] == "pretraining"
    assert json.loads(capsys.readouterr().out)["scope"] == "pretraining"


def test_pretraining_validation_never_builds_product_suite(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import socialgraph_gfm.gfm_workflow as workflow

    acceptance = SimpleNamespace(
        accepted=True,
        selected_variant="core-base",
        selected_checkpoint_ids=("a", "b", "c"),
        gates={"formal_pretrain_matrix": True},
        reasons=(),
        report_hash="a" * 64,
        corpus_hashes=("d" * 64,),
    )

    class Registry:
        def build_pretraining_acceptance(self, *, experiment_id: str):
            assert experiment_id == "experiment"
            return acceptance

        def record_pretraining_acceptance(self, value: object) -> None:
            assert value is acceptance

    monkeypatch.setattr(
        workflow,
        "prepare_runtime_layout",
        lambda *_args, **_kwargs: SimpleNamespace(
            root=tmp_path, gfm_reports=tmp_path
        ),
    )
    monkeypatch.setattr(workflow, "_registry", lambda _layout: Registry())
    monkeypatch.setattr(workflow, "_require_experiment_runs", lambda *_: (object(),))
    monkeypatch.setattr(workflow, "check_all_gfm_corpora", lambda *_: {"ready": True})
    monkeypatch.setattr(
        workflow,
        "_load_corpus_contracts",
        lambda *_args, **_kwargs: (SimpleNamespace(logical_hash="d" * 64),),
    )
    monkeypatch.setattr(workflow, "_write_contract", lambda *_: None)
    monkeypatch.setattr(
        workflow,
        "_build_product_suite_checkpoint",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("pretraining validation must not touch product assets")
        ),
    )

    result = workflow.validate_gfm(
        root=tmp_path, experiment_id="experiment", scope="pretraining"
    )

    assert result["accepted"] is True
    assert result["scope"] == "pretraining"


def test_synthetic_domain_batch_is_causal_exact_and_finite() -> None:
    torch = pytest.importorskip("torch")
    stream = _DomainStream(
        domain_id="thgl-software-2.0.0",
        manifest={"logicalHash": "a" * 64},
        src=np.asarray([1, 2, 3, 4, 5, 6, 7, 8, 9, 1, 0, 2], dtype=np.int64),
        dst=np.asarray([8, 3, 4, 5, 6, 7, 8, 9, 1, 2, 8, 9], dtype=np.int64),
        timestamp=np.arange(1, 13, dtype=np.int64),
        relation=np.zeros(12, dtype=np.int64),
        node_type=np.zeros(10, dtype=np.int64),
        node_count=10,
        train_end=11,
        validation_end=12,
        relation_offset=5,
        text_embedding=None,
        text_id_hash=None,
        text_timestamp=None,
        text_node_offset=None,
        work_id_hash=None,
        work_publication_timestamp=None,
        work_cluster=None,
        cursor=10,
    )
    batch = _core_batch(
        stream,
        batch_size=1,
        fanout=(15, 10),
        seed=20260820,
        allow_negative_fallback=True,
    )
    batch.validate()
    assert bool(torch.all(batch.edge_time <= float(batch.cutoff_time)))
    assert batch.positive_edge_index is not None
    assert batch.negative_edge_index is not None
    assert batch.negative_edge_index.shape[1] == 4
    assert not torch.equal(batch.positive_edge_index, batch.negative_edge_index)
    message_pairs = set(map(tuple, batch.edge_index.t().tolist()))
    assert all((target, source) in message_pairs for source, target in message_pairs)
    positive = batch.positive_edge_index[:, 0].tolist()
    assert positive == [0, 5]
    assert positive[::-1] not in batch.positive_edge_index.t().tolist()
    assert bool(torch.isfinite(batch.modalities["numeric"]).all())
    assert stream.cursor == 11


def test_validation_candidates_do_not_depend_on_training_epoch() -> None:
    stream = _DomainStream(
        domain_id="thgl-software-2.0.0",
        manifest={"logicalHash": "9" * 64},
        src=np.asarray([1, 2, 3, 4, 5, 6, 7, 8, 9, 1, 0, 2], dtype=np.int64),
        dst=np.asarray([8, 3, 4, 5, 6, 7, 8, 9, 1, 2, 8, 9], dtype=np.int64),
        timestamp=np.arange(1, 13, dtype=np.int64),
        relation=np.zeros(12, dtype=np.int64),
        node_type=np.zeros(10, dtype=np.int64),
        node_count=10,
        train_end=10,
        validation_end=12,
        relation_offset=5,
        text_embedding=None,
        text_id_hash=None,
        text_timestamp=None,
        text_node_offset=None,
        work_id_hash=None,
        work_publication_timestamp=None,
        work_cluster=None,
        cursor=9,
    )
    first = _core_batch(
        stream,
        batch_size=1,
        fanout=(15, 10),
        seed=20260820,
        cursor=0,
        upper_index=2,
        advance=False,
        split_role=1,
    )
    assert first.negative_edge_index is not None
    first_negative = first.negative_edge_index.clone()
    first_audit = stream.negative_sampling_audit["lastBatch"].copy()
    stream.epoch = 99
    second = _core_batch(
        stream,
        batch_size=1,
        fanout=(15, 10),
        seed=20260820,
        cursor=0,
        upper_index=2,
        advance=False,
        split_role=1,
    )
    assert second.negative_edge_index is not None
    assert first_negative.equal(second.negative_edge_index)
    assert stream.negative_sampling_audit["lastBatch"] == first_audit


def _lodo_fixture_stream(
    src: np.ndarray,
    dst: np.ndarray,
    *,
    node_count: int,
    relation: np.ndarray | None = None,
    cache_path: Path | None = None,
) -> _DomainStream:
    edge_count = int(src.size)
    return _DomainStream(
        domain_id="thgl-software-2.0.0",
        manifest={"logicalHash": "8" * 64},
        src=src,
        dst=dst,
        timestamp=np.arange(edge_count, dtype=np.int64),
        relation=(
            np.zeros(edge_count, dtype=np.int64) if relation is None else relation
        ),
        node_type=np.zeros(node_count, dtype=np.int64),
        node_count=node_count,
        train_end=edge_count,
        validation_end=edge_count,
        relation_offset=5,
        text_embedding=None,
        text_id_hash=None,
        text_timestamp=None,
        text_node_offset=None,
        work_id_hash=None,
        work_publication_timestamp=None,
        work_cluster=None,
        cursor=1,
        maximum_access_role="validation",
        access_audit={"maximumRole": "validation", "testArtifactsOpened": False},
        lodo_eligibility_cache_path=cache_path,
    )


def test_lodo_eligibility_rejects_global_but_not_local_negative_pool() -> None:
    disconnected_src = np.arange(100, 300, 2, dtype=np.int64)
    disconnected_dst = disconnected_src + 1
    hub_src = np.zeros(200, dtype=np.int64)
    hub_dst = np.arange(1, 201, dtype=np.int64)
    stream = _lodo_fixture_stream(
        np.concatenate((disconnected_src, hub_src)),
        np.concatenate((disconnected_dst, hub_dst)),
        node_count=320,
    )

    eligible = _build_lodo_eligible_ordinals(
        stream, fanout=(15, 10), negatives_per_positive=4
    )

    # Globally hundreds of same-type nodes are visible, but a hub target's
    # bounded local graph contains only its already-forbidden neighbours.
    assert 250 not in eligible
    with pytest.raises(GfmTrainingError, match="exact typed negative"):
        _core_batch(
            stream,
            batch_size=1,
            fanout=(15, 10),
            seed=7,
            cursor=250,
            upper_index=300,
            advance=False,
            split_role=0,
        )


def test_lodo_selected_pool_is_exact_fraction_and_every_row_builds() -> None:
    rng = np.random.default_rng(7)
    node_count, edge_count = 80, 800
    src = rng.integers(0, node_count, edge_count, dtype=np.int64)
    dst = rng.integers(0, node_count, edge_count, dtype=np.int64)
    dst = np.where(dst == src, (dst + 1) % node_count, dst)
    relation = rng.integers(0, 3, edge_count, dtype=np.int64)
    stream = _lodo_fixture_stream(
        src, dst, node_count=node_count, relation=relation
    )

    selection = _lodo_few_shot_selection(stream, seed=20260821, fraction=0.01)

    assert len(selection.event_indices) == max(
        1, int(np.floor(selection.eligible_pool_count * selection.fraction))
    )
    for ordinal in selection.ordinals:
        batch = _core_batch(
            stream,
            batch_size=1,
            fanout=(15, 10),
            seed=20260821,
            cursor=int(ordinal),
            upper_index=edge_count,
            advance=False,
            split_role=0,
        )
        assert batch.negative_edge_index is not None
        assert batch.negative_edge_index.shape[1] == 4


def test_lodo_eligibility_cache_reuses_and_recovers_interrupted_publication(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import socialgraph_gfm.gfm_workflow as workflow

    src = np.asarray([1, 2, 3, 4, 5, 6, 7, 8, 9, 1, 0, 2], dtype=np.int64)
    dst = np.asarray([8, 3, 4, 5, 6, 7, 8, 9, 1, 2, 8, 9], dtype=np.int64)
    cache = tmp_path / "eligibility"
    calls = 0
    original = workflow._build_lodo_eligible_ordinals

    def counted(*args: object, **kwargs: object) -> np.ndarray:
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(workflow, "_build_lodo_eligible_ordinals", counted)
    first = _lodo_fixture_stream(src, dst, node_count=10, cache_path=cache)
    first_values = _lodo_cached_eligible_ordinals(
        first, fanout=(15, 10), negatives_per_positive=4
    )
    second = _lodo_fixture_stream(src, dst, node_count=10, cache_path=cache)
    second_values = _lodo_cached_eligible_ordinals(
        second, fanout=(15, 10), negatives_per_positive=4
    )
    assert np.array_equal(first_values, second_values)
    assert calls == 1

    manifest = next(cache.glob("*.json"))
    artifact = next(cache.glob("*.npz"))
    manifest.unlink()
    third = _lodo_fixture_stream(src, dst, node_count=10, cache_path=cache)
    rebuilt = _lodo_cached_eligible_ordinals(
        third, fanout=(15, 10), negatives_per_positive=4
    )
    assert np.array_equal(first_values, rebuilt)
    assert artifact.is_file() and next(cache.glob("*.json")).is_file()
    assert calls == 2


def test_core_batch_retains_cold_start_positives_and_audits_actual_mix() -> None:
    # Every target is type compatible, but the visible graph has no genuine
    # 2/3-hop hard pool.  Formal batching must keep all positives and relabel
    # the exact uniform replacements instead of rebuilding per query and
    # silently discarding those rows.
    stream = _DomainStream(
        domain_id="thgl-software-2.0.0",
        manifest={"logicalHash": "f" * 64},
        src=np.asarray(
            [20, 2, 2, 2, 2, 2, 2, 21, 10, 10, 10, 10, 10, 10, 20, 21, 2, 10]
            + [0, 1],
            dtype=np.int64,
        ),
        dst=np.asarray(
            [2, 3, 4, 5, 6, 7, 8, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 22]
            + [20, 21],
            dtype=np.int64,
        ),
        timestamp=np.arange(1, 21, dtype=np.int64),
        relation=np.zeros(20, dtype=np.int64),
        node_type=np.zeros(30, dtype=np.int64),
        node_count=30,
        train_end=20,
        validation_end=20,
        relation_offset=5,
        text_embedding=None,
        text_id_hash=None,
        text_timestamp=None,
        text_node_offset=None,
        work_id_hash=None,
        work_publication_timestamp=None,
        work_cluster=None,
        cursor=18,
    )
    batch = _core_batch(
        stream,
        batch_size=2,
        fanout=(15, 10),
        seed=20260820,
    )
    assert batch.positive_edge_index is not None
    assert batch.positive_edge_index.shape[1] == 2
    audit = stream.negative_sampling_audit
    assert audit["requestedPositiveCount"] == 2
    assert audit["retainedPositiveCount"] == 2
    assert audit["effectivePositiveRatio"] == 1.0
    assert audit["requestedComponentCounts"] == {
        "degree_matched": 2,
        "hard": 4,
        "uniform": 2,
    }
    assert audit["negativeCount"] == 8
    assert sum(audit["actualComponentCounts"].values()) == 8
    assert audit["actualComponentCounts"]["hard_fallback_uniform"] == 4
    assert audit["fallbackDrawCount"] == 4
    assert audit["fallbackQueryCount"] == 2
    assert audit["actualMix"]["hard_fallback_uniform"] == 0.5
    assert audit["exactNoFalseNegative"] is True
    assert audit["typed"] is True
    assert audit["causal"] is True
    assert audit["cutoffVisibleCandidatesOnly"] is True
    assert audit["futureUnseenCandidateCount"] == 0
    assert audit["lastBatch"]["futureUnseenCandidateCount"] == 0
    assert audit["queryLocalUnique"] is True
    replay = stream.state_dict()
    restored = _DomainStream(
        domain_id=stream.domain_id,
        manifest=stream.manifest,
        src=stream.src,
        dst=stream.dst,
        timestamp=stream.timestamp,
        relation=stream.relation,
        node_type=stream.node_type,
        node_count=stream.node_count,
        train_end=stream.train_end,
        validation_end=stream.validation_end,
        relation_offset=stream.relation_offset,
        text_embedding=None,
        text_id_hash=None,
        text_timestamp=None,
        text_node_offset=None,
        work_id_hash=None,
        work_publication_timestamp=None,
        work_cluster=None,
        cursor=1,
    )
    restored.load_state_dict(replay)
    assert restored.negative_sampling_audit == audit


def test_incident_causal_index_matches_naive_reverse_scan_without_future_rows() -> None:
    rng = np.random.default_rng(20260812)
    src = rng.integers(0, 17, size=200, dtype=np.int64)
    dst = rng.integers(0, 17, size=200, dtype=np.int64)
    stream = _DomainStream(
        domain_id="thgl-software-2.0.0",
        manifest={"logicalHash": "b" * 64},
        src=src,
        dst=dst,
        timestamp=np.arange(200, dtype=np.int64),
        relation=np.zeros(200, dtype=np.int64),
        node_type=np.zeros(17, dtype=np.int64),
        node_count=17,
        train_end=150,
        validation_end=180,
        relation_offset=5,
        text_embedding=None,
        text_id_hash=None,
        text_timestamp=None,
        text_node_offset=None,
        work_id_hash=None,
        work_publication_timestamp=None,
        work_cluster=None,
        cursor=20,
    )

    def naive(end: int, seeds: set[int], fanout: tuple[int, int]) -> np.ndarray:
        frontier = set(seeds)
        selected: set[int] = set()
        for limit in fanout:
            counts = {node: 0 for node in frontier}
            layer: set[int] = set()
            for index in range(end - 1, -1, -1):
                source, target = int(src[index]), int(dst[index])
                touched = [
                    node
                    for node in {source, target}
                    if node in counts and counts[node] < limit
                ]
                if not touched:
                    continue
                selected.add(index)
                layer.update((source, target))
                for node in touched:
                    counts[node] += 1
                if counts and all(value >= limit for value in counts.values()):
                    break
            frontier = layer.difference(frontier)
            if not frontier:
                break
        return np.asarray(sorted(selected), dtype=np.int64)

    for end, seeds in ((15, {1}), (80, {2, 7}), (151, {0, 8, 16})):
        indexed = _recent_causal_edges(
            stream, end=end, seeds=seeds, fanout=(15, 10)
        )
        assert np.array_equal(indexed, naive(end, seeds, (15, 10)))
        assert bool(np.all(indexed < end))


def test_exact_negative_inventory_includes_history_omitted_by_fanout() -> None:
    stream = _DomainStream(
        domain_id="thgl-software-2.0.0",
        manifest={"logicalHash": "c" * 64},
        src=np.asarray([0, 0, 0, 2, 3], dtype=np.int64),
        dst=np.asarray([1, 2, 3, 3, 1], dtype=np.int64),
        timestamp=np.arange(1, 6, dtype=np.int64),
        relation=np.zeros(5, dtype=np.int64),
        node_type=np.zeros(4, dtype=np.int64),
        node_count=4,
        train_end=4,
        validation_end=5,
        relation_offset=5,
        text_embedding=None,
        text_id_hash=None,
        text_timestamp=None,
        text_node_offset=None,
        work_id_hash=None,
        work_publication_timestamp=None,
        work_cluster=None,
        cursor=4,
    )
    recent = _recent_causal_edges(stream, end=4, seeds={0}, fanout=(1, 1))
    assert 0 not in recent
    visible = _visible_edges_between_local_nodes(
        stream, end=4, local_nodes=np.asarray([0, 1, 2, 3], dtype=np.int64)
    )
    assert np.array_equal(visible, np.asarray([0, 1, 2, 3]))


def test_wikimedia_explicit_page_split_drives_targets_and_causal_visibility() -> None:
    stream = _DomainStream(
        domain_id="wikimedia-talk-article-2011-2015",
        manifest={"logicalHash": "d" * 64},
        src=np.asarray(
            [9, 8, 9, 10, 11, 3, 3, 3, 3, 3, 3, 2, 5], dtype=np.int64
        ),
        dst=np.asarray(
            [0, 12, 13, 14, 15, 8, 7, 9, 10, 11, 7, 7, 0], dtype=np.int64
        ),
        timestamp=np.arange(1, 14, dtype=np.int64),
        relation=np.zeros(13, dtype=np.int64),
        node_type=np.zeros(16, dtype=np.int64),
        node_count=16,
        train_end=10,
        validation_end=12,
        relation_offset=19,
        text_embedding=None,
        text_id_hash=None,
        text_timestamp=None,
        text_node_offset=8,
        work_id_hash=None,
        work_publication_timestamp=None,
        work_cluster=None,
        cursor=1,
        # A future test-page history row deliberately precedes train/validation
        # rows in global time.  It must never enter either message graph.
        event_split=np.asarray(
            [2, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 2], dtype=np.int8
        ),
    )
    assert np.array_equal(_stream_role_indices(stream, 0), [1, 2, 3, 4, 5, 7, 8, 9])
    assert np.array_equal(_stream_role_indices(stream, 1), [10, 11])
    assert np.array_equal(_stream_role_indices(stream, 2), [12])
    selected = _recent_causal_edges(
        stream,
        end=10,
        seeds=set(range(16)),
        fanout=(15, 10),
        maximum_split_role=1,
    )
    assert 0 not in selected
    assert 6 in selected  # older history of a validation-assigned page is context
    batch = _core_batch(
        stream,
        batch_size=1,
        fanout=(15, 10),
        seed=20260820,
        cursor=0,
        upper_index=2,
        advance=False,
        allow_negative_fallback=True,
        split_role=1,
    )
    batch.validate()
    assert batch.positive_edge_index is not None
    # Split role 1 ordinal zero is global event row 10 (3 -> 7); supervision
    # stays in the original direction while only the message graph is reversed.
    assert batch.positive_edge_index[:, 0].tolist() == [0, 1]
    assert float(batch.cutoff_time) == pytest.approx(10.0 / 86_400.0)


def test_wikimedia_temporal_audit_uses_page_roles_not_calendar_overlap() -> None:
    stream = _DomainStream(
        domain_id="wikimedia-talk-article-2011-2015",
        manifest={"logicalHash": "e" * 64},
        src=np.asarray([0, 1, 0, 2, 3, 2], dtype=np.int64),
        dst=np.asarray([4, 5, 4, 6, 5, 6], dtype=np.int64),
        timestamp=np.asarray([1, 2, 3, 4, 5, 6], dtype=np.int64),
        relation=np.zeros(6, dtype=np.int64),
        node_type=np.zeros(7, dtype=np.int64),
        node_count=7,
        train_end=3,
        validation_end=5,
        relation_offset=19,
        text_embedding=None,
        text_id_hash=None,
        text_timestamp=None,
        text_node_offset=4,
        work_id_hash=None,
        work_publication_timestamp=None,
        work_cluster=None,
        cursor=1,
        event_split=np.asarray([0, 1, 0, 2, 1, 2], dtype=np.int8),
    )
    assert _temporal_audit_counters(streams=(stream,)) == {
        "future_edge_access_count": 0,
        "cutoff_violation_count": 0,
        "split_overlap_count": 0,
    }


def test_product_audit_treats_candidate_rows_as_one_ranking_query() -> None:
    torch = pytest.importorskip("torch")
    from socialgraph_gfm.gfm.product_training import ProductAdaptBatch, SampleProvenance
    from socialgraph_gfm.gfm.types import CoreBatch, CoreSampleProvenance

    provenance = SampleProvenance(
        domain_id="openalex-graph-ai",
        graph_version="a" * 64,
        cutoff=10.0,
        horizon=365.0,
        task_id="collaboration",
        source_corpus_hash="a" * 64,
    )
    core = CoreBatch(
        domain_id=provenance.domain_id,
        modalities={"numeric": torch.zeros((2, 3))},
        modality_masks={"numeric": torch.ones(2, dtype=torch.bool)},
        edge_index=torch.tensor([[0, 1], [1, 0]]),
        edge_type=torch.zeros(2, dtype=torch.long),
        edge_time=torch.tensor([9.0, 9.0]),
        cutoff_time=10.0,
        provenance=CoreSampleProvenance(
            domain_id=provenance.domain_id,
            graph_version=provenance.graph_version,
            cutoff=provenance.cutoff,
            horizon=provenance.horizon,
            task_id=provenance.task_id,
            source_corpus_hash=provenance.source_corpus_hash,
        ),
    )
    candidate = torch.zeros((2, 100), dtype=torch.long)
    batch = ProductAdaptBatch(
        core_batch=core,
        candidate_edge_index=candidate,
        pair_features=torch.zeros((100, 8)),
        pair_labels=torch.cat((torch.ones(1), torch.zeros(99))),
        query_ids=torch.zeros(100, dtype=torch.long),
        provenance=provenance,
    )
    prepared = _PreparedProductBatch(
        batch=batch,
        raw_features=np.zeros((100, 8), dtype=np.float32),
        baseline_scores=np.zeros(100, dtype=np.float32),
    )
    assert _product_audit_counters([prepared]) == {
        "future_edge_access_count": 0,
        "cutoff_violation_count": 0,
        "split_overlap_count": 0,
    }
