#!/usr/bin/env python3
"""Source-checkout entry point for the packaged SocialGraph-FM runtime CLI."""

from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUNTIME_SOURCE = PROJECT_ROOT / "packages" / "runtime" / "src"
sys.path.insert(0, str(RUNTIME_SOURCE))
# A checkout-local entry point must always operate on the checkout that owns
# this script. Cross-checkout targeting is available only through --project-root.
os.environ["SOCIALGRAPH_PROJECT_ROOT"] = str(PROJECT_ROOT)

main = importlib.import_module("socialgraph_fm_runtime.cli").main


if __name__ == "__main__":
    raise SystemExit(main())
