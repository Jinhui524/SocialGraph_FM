from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
NAMES = (
    "core-serving-control.json",
    "core-serving-graph-catalog.json",
    "core-serving-registry.json",
)


def test_serving_contract_mirrors_match_canonical_source() -> None:
    canonical = PROJECT_ROOT / "contracts" / "core" / "serving"
    mirrors = (
        PROJECT_ROOT / "services" / "api" / "app" / "contracts",
        PROJECT_ROOT / "packages" / "gfm" / "contracts",
    )
    for name in NAMES:
        expected = (canonical / name).read_bytes()
        for mirror in mirrors:
            assert (mirror / name).read_bytes() == expected
