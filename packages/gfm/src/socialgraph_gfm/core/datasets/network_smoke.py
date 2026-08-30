"""Opt-in publisher endpoint smoke and Email-Eu-core sanity materialization."""

from __future__ import annotations

import argparse
import json
import urllib.parse
import urllib.request
from pathlib import Path

from socialgraph_gfm.runtime import core_runtime_root

from .materialize import materialize_email_eu_core
from .recipes import SourceRecipe, load_dataset_recipes


def _probe(source: SourceRecipe, *, timeout_seconds: float) -> dict[str, object]:
    request = urllib.request.Request(
        source.url,
        method="HEAD",
        headers={"User-Agent": "socialgraph-gfm-network-smoke/0.1"},
    )
    with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
        final_url = response.geturl()
        if urllib.parse.urlsplit(final_url).scheme != "https":
            raise ValueError("network smoke redirect target is not HTTPS")
        if final_url != source.url:
            raise ValueError("network smoke redirect target is outside the recipe allowlist")
        content_length_header = response.headers.get("Content-Length")
        content_length = int(content_length_header) if content_length_header else None
        if content_length is not None and content_length > source.max_bytes:
            raise ValueError("network smoke response exceeds recipe maximum")
        return {
            "sourceId": source.source_id,
            "url": source.url,
            "status": int(response.status),
            "contentLength": content_length,
        }


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--network",
        action="store_true",
        required=True,
        help="explicitly authorize publisher network requests",
    )
    parser.add_argument("--timeout-seconds", type=float, default=30.0)
    parser.add_argument(
        "--runtime-root",
        type=Path,
        default=None,
    )
    parser.add_argument("--materialize-email", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    authorized_root = core_runtime_root()
    requested_root = (args.runtime_root or authorized_root).resolve()
    if requested_root != authorized_root:
        raise ValueError(
            "core runtime access is authorized only at "
            f"{authorized_root} derived from SOCIALGRAPH_FM_HOME"
        )
    recipes = load_dataset_recipes()
    probes = []
    for recipe_id, recipe in sorted(recipes.items()):
        for source in recipe.sources:
            result = _probe(source, timeout_seconds=args.timeout_seconds)
            probes.append({"recipeId": recipe_id, **result})
            print(json.dumps({"event": "source-ok", "recipeId": recipe_id, **result}))

    materialized = None
    if args.materialize_email:
        materialized = str(materialize_email_eu_core(runtime_root=requested_root))
        print(json.dumps({"event": "email-materialized", "path": materialized}))
    print(
        json.dumps(
            {
                "event": "network-smoke-complete",
                "sources": len(probes),
                "emailMaterialized": materialized,
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
