from __future__ import annotations

from pathlib import Path

from app.config import Settings


def test_root_launcher_can_inject_every_mutable_api_path(monkeypatch, tmp_path: Path) -> None:
    project_var = tmp_path / "var"
    values = {
        "DATASET_STORAGE_ROOT": project_var / "api" / "dataset-store",
        "GFM_SESSION_TOKEN_FILE": (
            project_var / "gfm" / "core-runtime" / "serving" / "session.token"
        ),
        "GFM_CORE_SERVING_CONTROL_FILE": (
            project_var
            / "gfm"
            / "core-runtime"
            / "serving"
            / "serving-control.json"
        ),
        "GFM_CORE_RUN_BINDING_ROOT": project_var / "api" / "gfm-run-bindings",
        "GFM_CORE_SERVING_HIGH_WATER_ROOT": project_var / "api" / "gfm-serving-control-high-water",
        "TRUSTED_DATA_ROOTS": project_var / "research" / "incoming",
    }
    monkeypatch.setenv("GFM_SERVICE_URL", "http://127.0.0.1:8766")
    for name, value in values.items():
        monkeypatch.setenv(name, str(value))

    settings = Settings(_env_file=None)

    assert settings.dataset_storage_root == str(values["DATASET_STORAGE_ROOT"])
    assert settings.gfm_session_token_file == str(values["GFM_SESSION_TOKEN_FILE"])
    assert settings.gfm_core_serving_control_file == str(values["GFM_CORE_SERVING_CONTROL_FILE"])
    assert settings.gfm_core_run_binding_root == str(values["GFM_CORE_RUN_BINDING_ROOT"])
    assert settings.gfm_core_serving_high_water_root == str(values["GFM_CORE_SERVING_HIGH_WATER_ROOT"])
    assert settings.trusted_roots == [values["TRUSTED_DATA_ROOTS"]]
