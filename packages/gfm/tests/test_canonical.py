import json
import math
import random
import shutil
import subprocess
from datetime import datetime
from pathlib import Path

import pytest

from socialgraph_gfm.canonical import canonical_json, canonical_sha256
from socialgraph_gfm.errors import ContractViolation


def test_golden_vectors_match_code_point_order_and_hash():
    fixture = json.loads(
        (Path(__file__).parent / "golden" / "canonical-vectors.json").read_text(encoding="utf-8")
    )
    for vector in fixture["vectors"]:
        assert canonical_json(vector["value"]) == vector["canonical"], vector["name"]
        assert canonical_sha256(vector["value"]) == vector["sha256"], vector["name"]


@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf])
def test_nonfinite_numbers_are_rejected(value):
    with pytest.raises(ContractViolation, match="NaN and Infinity"):
        canonical_json({"value": value})


def test_naive_datetime_is_rejected():
    with pytest.raises(ContractViolation, match="timezone"):
        canonical_json({"value": datetime(2026, 1, 1)})  # noqa: DTZ001 - deliberately naive


def test_hash_is_independent_of_insertion_order():
    assert canonical_sha256({"中": 1, "a": 2}) == canonical_sha256({"a": 2, "中": 1})


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (-0.0, "0"),
        (1.0, "1"),
        (0.001, "0.001"),
        (0.00000123, "0.00000123"),
        (1e-6, "0.000001"),
        (1e-7, "1e-7"),
        (1e20, "100000000000000000000"),
        (1e21, "1e+21"),
    ],
)
def test_number_rendering_matches_ecmascript_json_stringify(value, expected):
    assert canonical_json(value) == expected


def test_finite_number_rendering_differential_against_node():
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js is not available for the optional cross-runtime differential")
    random_values = [random.Random(20260812).uniform(-1e22, 1e22) for _ in range(128)]
    values = [-0.0, 0.001, 0.00000123, 1e-6, 1e-7, 1e20, 1e21, *random_values]
    completed = subprocess.run(
        [
            node,
            "-e",
            "const xs=JSON.parse(process.argv[1]);process.stdout.write(JSON.stringify(xs.map(JSON.stringify)))",
            json.dumps(values),
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert [canonical_json(value) for value in values] == json.loads(completed.stdout)
