import json
from pathlib import Path

from pydantic.json_schema import models_json_schema
from socialgraph_gfm.canonical import canonical_sha256
from socialgraph_gfm.public_contracts import PUBLIC_CONTRACTS, public_contract_schemas


def test_public_schema_registry_hashes_every_complete_model_schema():
    artifact = json.loads(
        (
            Path(__file__).resolve().parents[1]
            / "contracts"
            / "public-contracts.schema.json"
        ).read_text(encoding="utf-8")
    )
    assert artifact["$schema"] == "https://json-schema.org/draft/2020-12/schema"
    actual = {
        name: canonical_sha256(schema)
        for name, schema in public_contract_schemas().items()
    }
    assert artifact["x-modelSchemaSha256"] == actual
    assert set(artifact["properties"]["contractName"]["enum"]) == set(actual)


def test_complete_public_schema_bundle_matches_models():
    path = (
        Path(__file__).resolve().parents[1]
        / "contracts"
        / "public-contracts.full.schema.json"
    )
    checked = json.loads(path.read_text(encoding="utf-8"))
    _, generated = models_json_schema(
        [(model, "validation") for model in PUBLIC_CONTRACTS],
        by_alias=True,
        title="SocialGraph-FM Public Contracts",
    )
    generated["$schema"] = "https://json-schema.org/draft/2020-12/schema"
    generated["$id"] = "https://socialgraph-fm.local/contracts/public-contracts/1.0"
    generated["anyOf"] = [
        {"$ref": f"#/$defs/{model.__name__}"} for model in PUBLIC_CONTRACTS
    ]
    assert checked == generated
