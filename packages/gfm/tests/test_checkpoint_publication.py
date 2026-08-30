from pathlib import Path

import pytest

import socialgraph_gfm.gfm.checkpoint as checkpoint_module
import socialgraph_gfm.gfm_workflow as workflow
from socialgraph_gfm.gfm.checkpoint import save_gfm_checkpoint


def test_recovery_checkpoint_has_one_manifest_publication(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    run_id = "run-1"
    checkpoint_directory = tmp_path / "checkpoints"
    save_gfm_checkpoint(
        checkpoint_directory,
        checkpoint_id=f"{run_id}-latest-1",
        run_id=run_id,
        epoch=1,
        step=1,
        components={"encoder": {"weight": torch.tensor([1.0])}},
        optimizer_state={"state": {}, "param_groups": []},
        scheduler_state=None,
        scaler_state=None,
        sampler_state={"cursor": 1},
        best_state={"metric": 1.0},
        config={"model": "core-base"},
        corpus_hashes=("a" * 64,),
    )

    primary_publications = 0
    original_atomic_text = checkpoint_module._atomic_text

    def count_primary_publication(path: Path, value: str) -> None:
        nonlocal primary_publications
        if path.name.endswith(".manifest.json"):
            primary_publications += 1
        original_atomic_text(path, value)

    caller_publications = 0
    original_write_contract = workflow._write_contract

    def count_caller_publication(path: Path, value: object) -> None:
        nonlocal caller_publications
        if path.name.endswith(".manifest.json"):
            caller_publications += 1
        original_write_contract(path, value)

    monkeypatch.setattr(checkpoint_module, "_atomic_text", count_primary_publication)
    monkeypatch.setattr(workflow, "_write_contract", count_caller_publication)

    workflow._rotate_latest_to_recovery(checkpoint_directory, run_id=run_id)

    assert primary_publications == 1
    assert caller_publications == 0
    assert (checkpoint_directory / f"{run_id}-recovery-1.manifest.json").is_file()
