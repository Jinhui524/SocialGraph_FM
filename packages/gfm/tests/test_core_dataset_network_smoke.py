from __future__ import annotations

import pytest

from socialgraph_gfm.errors import GfmError
from socialgraph_gfm.core.datasets import network_smoke
from socialgraph_gfm.core.datasets.network_smoke import main


def test_network_smoke_is_explicitly_opt_in() -> None:
    with pytest.raises(SystemExit) as error:
        main([])

    assert error.value.code == 2


def test_network_smoke_requires_configured_socialgraph_home(monkeypatch) -> None:
    monkeypatch.delenv("SOCIALGRAPH_FM_HOME", raising=False)

    with pytest.raises(GfmError, match="SOCIALGRAPH_FM_HOME"):
        main(["--network"])


def test_network_smoke_derives_and_confines_runtime_root(monkeypatch, tmp_path) -> None:
    home = tmp_path / "var" / "gfm"
    expected = (home / "core-runtime").resolve()
    monkeypatch.setenv("SOCIALGRAPH_FM_HOME", str(home))
    monkeypatch.setattr(network_smoke, "load_dataset_recipes", lambda: {})
    monkeypatch.setattr(
        network_smoke,
        "materialize_email_eu_core",
        lambda *, runtime_root: runtime_root / "materialized" / "email",
    )

    assert main(["--network", "--materialize-email"]) == 0

    with pytest.raises(ValueError, match="authorized only"):
        main(
            [
                "--network",
                "--materialize-email",
                "--runtime-root",
                str(tmp_path / "outside"),
            ]
        )

    assert expected == (home / "core-runtime").resolve()
