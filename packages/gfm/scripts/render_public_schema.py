"""Render or verify the complete public contract JSON Schema without writing files."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from pydantic.json_schema import models_json_schema

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from socialgraph_gfm.public_contracts import PUBLIC_CONTRACTS


def render() -> dict:
    _, schema = models_json_schema(
        [(model, "validation") for model in PUBLIC_CONTRACTS],
        by_alias=True,
        title="SocialGraph-FM Public Contracts",
    )
    schema["$schema"] = "https://json-schema.org/draft/2020-12/schema"
    schema["$id"] = "https://socialgraph-fm.local/contracts/public-contracts/1.0"
    schema["anyOf"] = [
        {"$ref": f"#/$defs/{model.__name__}"} for model in PUBLIC_CONTRACTS
    ]
    return schema


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    parser.add_argument(
        "--artifact",
        default=str(
            Path(__file__).resolve().parents[1]
            / "contracts"
            / "public-contracts.full.schema.json"
        ),
    )
    args = parser.parse_args()
    rendered = render()
    if args.check:
        checked = json.loads(Path(args.artifact).read_text(encoding="utf-8"))
        if checked != rendered:
            print("public contract schema artifact is stale")
            return 1
        print("public contract schema artifact is current")
        return 0
    print(json.dumps(rendered, ensure_ascii=False, separators=(",", ":"), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
