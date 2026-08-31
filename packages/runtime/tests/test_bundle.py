from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import socialgraph_fm_runtime.bundle as runtime_bundle
from socialgraph_fm_runtime.bundle import (
    RuntimeBundle,
    file_sha256,
    install_seed,
    materialize_target_examples,
    run_checkpoint_forward_probe,
    verify_target_examples,
)
from socialgraph_fm_runtime.layout import RuntimeLayout


def test_empty_legacy_seed_directory_is_replaced_atomically(tmp_path: Path) -> None:
    layout = RuntimeLayout(tmp_path)
    relative = (
        "examples/governance/target-domain/.governance-target-catalog/"
        + "a" * 64
        + "/target-domain-a-zero.sgtask.zip"
    )
    source = tmp_path.joinpath(*relative.split("/"))
    source.parent.mkdir(parents=True)
    source.write_bytes(b"target-task-fixture")
    destination = layout.target_input_root
    destination.mkdir(parents=True)
    asset = {
        "path": relative,
        "role": "target_domain_example",
        "bytes": source.stat().st_size,
        "sha256": file_sha256(source),
    }
    bundle = RuntimeBundle(
        {"bundleVersion": "fixture"}, (asset,), "0" * 64
    )

    install_seed(
        layout,
        bundle,
        role="target_domain_example",
        prefix="examples/governance/target-domain",
        destination=destination,
    )

    installed = destination / ".governance-target-catalog" / ("a" * 64)
    assert (installed / "target-domain-a-zero.sgtask.zip").read_bytes() == (
        b"target-task-fixture"
    )


def _canonical_sha256(document: dict[str, object]) -> str:
    value = json.dumps(
        document, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(value).hexdigest()


def test_target_examples_are_visible_copies_bound_to_the_catalog(tmp_path: Path) -> None:
    layout = RuntimeLayout(tmp_path)
    source_root = tmp_path / "examples" / "governance" / "target-domain"
    generation = "a" * 64
    hidden = source_root / ".governance-target-catalog" / generation
    hidden.mkdir(parents=True)
    declarations = []
    assets = []
    for role, name, value in (
        ("zero_shot", "target-domain-a-zero.sgtask.zip", b"zero"),
        ("few_shot", "target-domain-b-few.sgtask.zip", b"few"),
    ):
        source = hidden / name
        source.write_bytes(value)
        relative = f".governance-target-catalog/{generation}/{name}"
        digest = hashlib.sha256(value).hexdigest()
        declarations.append(
            {"role": role, "path": relative, "bytes": len(value), "sha256": digest}
        )
        assets.append(
            {
                "path": f"examples/governance/target-domain/{relative}",
                "role": "target_domain_example",
                "bytes": len(value),
                "sha256": digest,
            }
        )
    logical: dict[str, object] = {
        "schemaVersion": "socialgraph-fm.governance-target-catalog/1.0",
        "generationId": generation,
        "targets": declarations,
    }
    catalog = {**logical, "catalogHash": _canonical_sha256(logical)}
    catalog_path = source_root / "governance-target-tasks.catalog.json"
    catalog_path.write_text(json.dumps(catalog), encoding="utf-8")
    assets.append(
        {
            "path": "examples/governance/target-domain/governance-target-tasks.catalog.json",
            "role": "target_domain_example",
            "bytes": catalog_path.stat().st_size,
            "sha256": file_sha256(catalog_path),
        }
    )
    bundle = RuntimeBundle({"bundleVersion": "fixture"}, tuple(assets), "0" * 64)

    report = materialize_target_examples(layout, bundle)

    assert Path(report["zeroShot"]["path"]).read_bytes() == b"zero"
    assert Path(report["fewShot"]["path"]).read_bytes() == b"few"
    assert verify_target_examples(layout, bundle) == report
    Path(report["zeroShot"]["path"]).write_bytes(b"tampered")
    with pytest.raises(RuntimeError, match="hash differs"):
        verify_target_examples(layout, bundle)


def test_checkpoint_forward_probe_returns_a_path_free_four_protocol_summary(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    layout = RuntimeLayout(tmp_path)
    layout.gfm_package.mkdir(parents=True)
    layout.model_root.mkdir(parents=True)
    verified = RuntimeBundle({"bundleVersion": "fixture"}, (), "0" * 64)
    monkeypatch.setattr(runtime_bundle, "load_and_verify_bundle", lambda _layout: verified)
    monkeypatch.setattr(
        runtime_bundle, "verify_installed_runtime_bundle", lambda _layout, _bundle: {}
    )
    protocols = []
    for index, protocol in enumerate(
        ("global", "in_domain", "low_label", "cross_domain"), start=1
    ):
        protocols.append(
            {
                "protocol": protocol,
                "checkpoint": {"sha256": str(index) * 64},
                "model": {
                    "modelVersionHash": chr(96 + index) * 64,
                    "modelStateHash": str(index + 4) * 64,
                },
                "router": {"routesAllowed": True, "weightsValid": True},
                "shape": {"logits": [8]},
                "finite": True,
                "modelStateUnchanged": True,
                "modalityContributionsValid": True,
                "allowedExpertMask": [True, False],
                "outputHash": "f" * 64,
            }
        )
    payload = {
        "ok": True,
        "command": "forward-smoke",
        "root": str(layout.model_root),
        "schemaVersion": "socialgraph-fm.global-model-forward-smoke/1.0",
        "passed": True,
        "readOnly": True,
        "device": "cpu",
        "deviceName": "cpu",
        "torchVersion": "2.8.0",
        "torchGeometricVersion": "2.6.1",
        "exportHash": "9" * 64,
        "corpus": {"country": "russia"},
        "batch": {"batchHash": "8" * 64},
        "protocolCount": 4,
        "protocols": protocols,
        "reportHash": "7" * 64,
    }
    monkeypatch.setattr(
        runtime_bundle,
        "run_clean_python",
        lambda *_args, **_kwargs: SimpleNamespace(
            returncode=0, stdout=json.dumps(payload), stderr=""
        ),
    )

    report = run_checkpoint_forward_probe(layout, Path("python"))

    assert [item["protocol"] for item in report["protocols"]] == [
        "global",
        "in_domain",
        "low_label",
        "cross_domain",
    ]
    assert "root" not in report
    assert str(tmp_path) not in json.dumps(report)


def test_runtime_state_probe_uses_the_public_global_runtime_name() -> None:
    assert "GlobalServingRuntime" in runtime_bundle._RUNTIME_STATE_PROBE_SOURCE
    assert "GlobalModelServingRuntime" not in runtime_bundle._RUNTIME_STATE_PROBE_SOURCE
