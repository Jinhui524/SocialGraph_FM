"""Fresh-process checkpoint verifier used by the smoke command."""

import argparse
import json

from .smoke import verify_checkpoint_output


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--fixture", required=True, choices=("actor", "hetero"))
    parser.add_argument("--device", required=True, choices=("cpu", "cuda"))
    args = parser.parse_args()
    print(json.dumps(verify_checkpoint_output(args.manifest, args.fixture, args.device), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
