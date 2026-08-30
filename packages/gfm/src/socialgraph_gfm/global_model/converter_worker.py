"""Private subprocess entry for trusted Global pickle conversion."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from socialgraph_gfm.canonical import canonical_json, canonical_sha256
from socialgraph_gfm.gfm.corpus.common import read_json_object

from .converter import (
    WORKER_RECEIPT_SCHEMA,
    convert_trusted_country,
    validate_worker_contract,
)

WORKER_ADDRESS_SPACE_BYTES = 32 * 1024**3


def _apply_posix_address_space_limit() -> None:
    if sys.platform == "win32":
        return
    try:
        import resource

        soft, hard = resource.getrlimit(resource.RLIMIT_AS)
        ceiling = (
            WORKER_ADDRESS_SPACE_BYTES
            if hard == resource.RLIM_INFINITY
            else min(WORKER_ADDRESS_SPACE_BYTES, hard)
        )
        selected_soft = ceiling if soft == resource.RLIM_INFINITY else min(soft, ceiling)
        resource.setrlimit(resource.RLIMIT_AS, (selected_soft, hard))
    except (ImportError, OSError, ValueError):
        # Source byte caps and sequential country conversion remain mandatory.
        return


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--contract", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    request = validate_worker_contract(read_json_object(args.contract))
    _apply_posix_address_space_limit()
    manifest = convert_trusted_country(
        country_id=request.country_id,
        pickle_sources=request.pickle_sources,
        text_tensor_path=request.text_tensor_path,
        destination=request.destination,
        trusted_source=True,
    )
    receipt = {
        "schemaVersion": WORKER_RECEIPT_SCHEMA,
        "countryId": request.country_id,
        "contractHash": request.contract_hash,
        "manifestPath": str(request.destination / "manifest.json"),
        "manifestHash": manifest.content_hash,
    }
    receipt["receiptHash"] = canonical_sha256(receipt)
    sys.stdout.write(canonical_json(receipt) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
