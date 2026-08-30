import json
import platform
from types import SimpleNamespace

import pytest

from socialgraph_gfm.cli import main
from socialgraph_gfm.errors import RunCancelled
from socialgraph_gfm.preflight import preflight_report
from socialgraph_gfm.runtime import RunContext, runtime_report
import socialgraph_gfm.runtime as runtime_module


def test_doctor_is_machine_readable_and_runtime_status_is_honest(capsys, tmp_path):
    result = main(["doctor", "--device", "cpu", "--root", str(tmp_path), "--json"])
    payload = json.loads(capsys.readouterr().out)
    assert payload["schemaVersion"] == "gfm.doctor/1.0"
    assert isinstance(payload["missing"], list)
    assert payload["artifactRoot"] == str(tmp_path.resolve())
    assert result == (0 if payload["runtimeReady"] else 2)


def test_preflight_keeps_independent_readiness_gates(tmp_path):
    report = preflight_report(device="cpu", root=tmp_path)
    assert report["readiness"]["WorkbenchInputReady"] is True
    assert report["readiness"]["CorpusReady"] is False
    assert report["readiness"]["GfmCorpusReady"] is False
    assert report["readiness"]["NewcomerOverlayReady"] is False
    assert report["readiness"]["GfmPretrainingValidated"] is False
    assert report["readiness"]["GfmProductValidated"] is False
    assert report["readiness"]["ModelValidated"] is False
    assert report["readiness"]["GfmServingReady"] is False
    assert report["readiness"]["LargeGraphUiReleaseReady"] is False
    assert report["models"] == []
    assert report["smokeCoverage"] == []


def test_preflight_can_validate_pretraining_while_product_and_model_stay_false(
    tmp_path, monkeypatch
):
    import socialgraph_gfm.preflight as preflight

    hashes = {
        "openalex-graph-ai": "a" * 64,
        "thgl-software-2.0.0": "b" * 64,
        "wikimedia-talk-article-2011-2015": "c" * 64,
    }
    monkeypatch.setattr(preflight, "artifact_root", lambda _root: tmp_path)
    monkeypatch.setattr(
        preflight,
        "runtime_report",
        lambda _device: {"runtimeReady": False, "environmentHash": "d" * 64},
    )
    monkeypatch.setattr(
        preflight, "verify_lock_manifest", lambda: {"releaseLocksReady": False}
    )
    monkeypatch.setattr(preflight, "get_fixture", lambda name: name)
    monkeypatch.setattr(
        preflight,
        "check_compatibility",
        lambda _fixture: SimpleNamespace(
            model_dump=lambda **_kwargs: {"compatible": True}
        ),
    )
    monkeypatch.setattr(
        preflight, "_root_report", lambda _root: {"anchorWritable": False}
    )
    monkeypatch.setattr(
        preflight,
        "_formal_corpus_evidence",
        lambda _root: ({"ready": False}, None),
    )
    monkeypatch.setattr(
        preflight, "_baseline_evidence", lambda *_args: {"ready": False}
    )
    monkeypatch.setattr(
        preflight,
        "_gfm_corpus_evidence",
        lambda _root: ({"ready": True, "domainManifestHashes": hashes}, tuple(hashes.values())),
    )
    monkeypatch.setattr(
        preflight,
        "_gfm_task_asset_evidence",
        lambda *_args, **_kwargs: {"newcomerOverlay": {"ready": False}},
    )
    monkeypatch.setattr(
        preflight,
        "_gfm_acceptance_evidence",
        lambda *_args, **_kwargs: {
            "ready": False,
            "productValidated": False,
        },
    )
    monkeypatch.setattr(
        preflight,
        "_gfm_pretraining_acceptance_evidence",
        lambda *_args, **_kwargs: {"ready": True},
    )
    monkeypatch.setattr(preflight, "gfm_optional_runtime_report", lambda: {})
    monkeypatch.setattr(preflight, "_storage_evidence", lambda *_args: {})

    readiness = preflight.preflight_report(root=tmp_path)["readiness"]

    assert readiness["GfmCorpusReady"] is True
    assert readiness["GfmPretrainingValidated"] is True
    assert readiness["GfmProductValidated"] is False
    assert readiness["ModelValidated"] is False


def test_materialize_fails_clearly_when_runtime_is_absent(capsys, tmp_path):
    report = runtime_report("cpu")
    if report["runtimeReady"]:
        return
    result = main(
        [
            "materialize", "--fixture", "actor", "--output", str(tmp_path / "out"),
            "--device", "cpu", "--json",
        ]
    )
    payload = json.loads(capsys.readouterr().out)
    assert result == 4
    assert payload["error"]["code"] in {
        "GFM_RUNTIME_DEPENDENCY_MISSING",
        "GFM_RUNTIME_VERSION_MISMATCH",
    }
    assert not (tmp_path / "out").exists()


def test_run_context_honours_cancellation_marker(tmp_path):
    context = RunContext(run_id="cancelled", root=tmp_path)
    context.prepare()
    context.cancel_path.write_text("cancel", encoding="utf-8")
    with pytest.raises(RunCancelled):
        context.check_cancelled()


@pytest.mark.parametrize("wrong_tag", ["2.12.0+cpu", "2.12.0+cu128"])
def test_cuda_doctor_rejects_wrong_torch_local_wheel_tag(monkeypatch, wrong_tag):
    pytest.importorskip("torch")
    monkeypatch.setattr(runtime_module.platform, "system", lambda: "Windows")
    monkeypatch.setattr(
        runtime_module,
        "_version",
        lambda name: {
            "pydantic": "2.13.4",
            "numpy": "2.3.3",
            "torch": wrong_tag,
            "torch_geometric": "2.8.0.post1",
            "ogb": "1.3.6",
            "pyg_lib": "0.7.0+pt212cu130",
        }.get(name),
    )
    report = runtime_report("cuda")
    assert any(item["package"] == "torch" for item in report["mismatches"])


def test_windows_cu130_wheel_can_execute_cpu_without_switching_to_cpu_profile():
    if platform.system() != "Windows":
        pytest.skip("Windows profile selection test")
    cuda_report = runtime_report("cuda")
    if cuda_report["versions"].get("torch") != "2.12.0+cu130":
        pytest.skip("exact cu130 wheel is not installed")
    cpu_report = runtime_report("cpu")
    assert cpu_report["selectedProfile"] == "windows-cu130"
    assert cpu_report["runtimeReady"] is True
    assert cpu_report["cuda"]["requested"] is False


def test_windows_cpu_wheel_selects_the_dedicated_profile():
    if platform.system() != "Windows":
        pytest.skip("Windows profile selection test")
    report = runtime_report("cpu")
    if report["versions"].get("torch") != "2.8.0+cpu":
        pytest.skip("exact Windows CPU wheel is not installed")
    assert report["selectedProfile"] == "windows-cpu"
    assert report["runtimeReady"] is True
    assert report["versions"]["pyg_lib"] == "0.6.0+pt28cpu"


def test_macos_arm64_cpu_selects_the_verified_platform_profile(monkeypatch):
    monkeypatch.setattr(runtime_module.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(runtime_module.platform, "machine", lambda: "arm64")
    monkeypatch.setattr(runtime_module.platform, "libc_ver", lambda: ("", ""))
    monkeypatch.setattr(
        runtime_module,
        "_version",
        lambda name: {
            "pydantic": "2.13.4",
            "numpy": "2.3.3",
            "torch": "2.8.0",
            "torch_geometric": "2.8.0.post1",
            "ogb": "1.3.6",
            "pyg_lib": "0.6.0+pt28",
        }.get(name),
    )

    report = runtime_report("cpu")

    assert report["selectedProfile"] == "macos-arm64-cpu"
    assert report["installProfile"] == "macos-arm64-cpu-pt28"
    assert report["runtimeReady"] is True


@pytest.mark.parametrize(
    ("system", "machine", "device"),
    [
        ("Darwin", "x86_64", "cpu"),
        ("Linux", "aarch64", "cpu"),
        ("Windows", "ARM64", "cpu"),
        ("Darwin", "arm64", "cuda"),
    ],
)
def test_runtime_report_rejects_unpublished_platform_profiles(
    monkeypatch, system, machine, device
):
    monkeypatch.setattr(runtime_module.platform, "system", lambda: system)
    monkeypatch.setattr(runtime_module.platform, "machine", lambda: machine)
    monkeypatch.setattr(
        runtime_module.platform,
        "libc_ver",
        lambda: ("glibc", "2.39") if system == "Linux" else ("", ""),
    )
    monkeypatch.setattr(runtime_module, "_version", lambda _name: None)

    report = runtime_report(device)

    assert report["selectedProfile"] is None
    assert report["installProfile"] is None
    assert report["runtimeReady"] is False
    assert any(item["package"] == "platform" for item in report["mismatches"])
