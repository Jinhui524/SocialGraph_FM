import shutil
import subprocess
import sys
import tomllib
import zipfile
from pathlib import Path


PROJECT = Path(__file__).resolve().parents[1]


def _copy_effective_context(destination: Path, dockerfile: str) -> None:
    instructions = dockerfile.splitlines()
    wheel_build = next(
        index for index, line in enumerate(instructions) if "python -m pip install" in line
    )
    for line in instructions[:wheel_build]:
        if not line.startswith("COPY "):
            continue
        sources, target = line.split()[1:-1], line.split()[-1]
        for source in sources:
            origin = PROJECT / source
            if origin.is_dir():
                shutil.copytree(origin, destination / target, dirs_exist_ok=True)
            else:
                copied = destination / target / Path(source).name if target.endswith("/") else destination / target
                copied.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(origin, copied)


def test_effective_docker_context_builds_a_wheel_with_every_force_included_resource(tmp_path):
    context = tmp_path / "context"
    context.mkdir()
    dockerfile = (PROJECT / "Dockerfile").read_text(encoding="utf-8")
    _copy_effective_context(context, dockerfile)
    wheel_directory = tmp_path / "wheel"
    result = subprocess.run(
        [sys.executable, "-m", "pip", "wheel", ".", "--no-deps", "--wheel-dir", str(wheel_directory)],
        cwd=context,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr

    pyproject = tomllib.loads((PROJECT / "pyproject.toml").read_text(encoding="utf-8"))
    force_include = pyproject["tool"]["hatch"]["build"]["targets"]["wheel"]["force-include"]
    wheel = next(wheel_directory.glob("*.whl"))
    with zipfile.ZipFile(wheel) as archive:
        names = set(archive.namelist())
    expected_resources = set()
    for source, target in force_include.items():
        origin = PROJECT / source
        if origin.is_dir():
            expected_resources.update(
                str(Path(target) / path.relative_to(origin)).replace("\\", "/")
                for path in origin.rglob("*")
                if path.is_file()
            )
        else:
            expected_resources.add(target)
    assert expected_resources.issubset(names)


def test_context_without_config_copy_cannot_build_the_wheel(tmp_path):
    dockerfile = (PROJECT / "Dockerfile").read_text(encoding="utf-8")
    context = tmp_path / "context-without-configs"
    context.mkdir()
    _copy_effective_context(context, dockerfile.replace("COPY configs ./configs\n", ""))

    result = subprocess.run(
        [sys.executable, "-m", "pip", "wheel", ".", "--no-deps", "--wheel-dir", str(tmp_path / "wheel")],
        cwd=context,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode != 0


def test_container_entrypoint_keeps_the_authenticated_runtime_on_loopback():
    dockerfile = (PROJECT / "Dockerfile").read_text(encoding="utf-8")
    assert 'ENTRYPOINT ["python", "-m", "socialgraph_gfm.core.inference_cli"]' in dockerfile
    assert '"--global-model-root", "/var/lib/socialgraph-fm/socialgraph-global"' in dockerfile
    assert '"--host", "127.0.0.1"' in dockerfile
    assert "EXPOSE 8766" not in dockerfile
    for name in (
        "core-serving-control.json",
        "core-serving-registry.json",
        "core-serving-graph-catalog.json",
    ):
        assert f"cp contracts/{name}" in dockerfile


def test_docker_context_excludes_local_build_and_tool_caches():
    ignored = set(
        (PROJECT / ".dockerignore").read_text(encoding="utf-8").splitlines()
    )
    assert {
        ".mypy_cache",
        ".ruff_cache",
        ".venv",
        "venv",
        "build",
        "dist",
        "*.egg-info",
    }.issubset(ignored)
