import importlib.util
import sys
from pathlib import Path

import pytest
from pydantic import ValidationError
from socialgraph_gfm import public_contracts
from socialgraph_gfm.tasks import CORE_TASKS

PUBLIC_NAMES = (
    "TimeRange",
    "FeatureManifest",
    "GraphSnapshotRef",
    "CorpusManifest",
    "CoreTaskManifest",
    "TrainingRunManifest",
    "CheckpointManifest",
    "ModelCapability",
    "FindingEvidence",
    "CoreFinding",
    "CompatibilityMapping",
    "CompatibilityIssue",
    "GfmCompatibilityReport",
    "GfmReadiness",
    "CoreCapabilitiesResponse",
    "GfmTasksResponse",
    "CoreRunRequest",
)


def load_api_contract_module():
    module_path = (
        Path(__file__).resolve().parents[3]
        / "services"
        / "api"
        / "app"
        / "gfm_schemas.py"
    )
    if not module_path.is_file():
        pytest.skip("sibling API checkout is not available; checked schema artifact still applies")
    spec = importlib.util.spec_from_file_location("api_gfm_schemas_for_parity", module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("name", PUBLIC_NAMES)
def test_public_pydantic_json_schemas_exactly_match_api(name):
    api = load_api_contract_module()
    offline_model = getattr(public_contracts, name)
    api_model = getattr(api, name)
    assert offline_model.model_json_schema(by_alias=True) == api_model.model_json_schema(by_alias=True)


def test_public_task_payloads_exactly_match_api():
    api = load_api_contract_module()
    offline = [task.model_dump(mode="json", by_alias=True) for task in CORE_TASKS]
    online = [task.model_dump(mode="json", by_alias=True) for task in api.CORE_TASKS]
    assert offline == online


def test_internal_contracts_do_not_redeclare_public_names():
    from socialgraph_gfm import contracts

    overlap = set(PUBLIC_NAMES).intersection(vars(contracts))
    assert overlap == set()


def test_time_range_behavior_matches_api_for_naive_datetimes():
    api = load_api_contract_module()
    for model in (public_contracts.TimeRange, api.TimeRange):
        with pytest.raises(ValidationError, match="时区"):
            model(start="2026-01-01T00:00:00", end="2026-01-02T00:00:00")
