"""Portable repository and runtime paths."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


PROJECT_MARKERS = ("apps/web", "services/api", "packages/gfm", "bundles/runtime-manifest.json")


def discover_project_root(explicit: str | Path | None = None) -> Path:
    """Find the checkout without relying on an installed package location."""

    if explicit is not None:
        root = Path(explicit).expanduser().resolve()
        _require_project(root)
        return root
    configured = os.environ.get("SOCIALGRAPH_PROJECT_ROOT", "").strip()
    if configured:
        root = Path(configured).expanduser().resolve()
        _require_project(root)
        return root
    candidates = [Path.cwd(), Path(__file__).resolve()]
    for candidate in candidates:
        for parent in (candidate, *candidate.parents):
            if all((parent / marker).exists() for marker in PROJECT_MARKERS):
                return parent.resolve()
    raise RuntimeError(
        "Could not locate the SocialGraph-FM checkout. Run inside the repository or set "
        "SOCIALGRAPH_PROJECT_ROOT."
    )


def _require_project(root: Path) -> None:
    missing = [marker for marker in PROJECT_MARKERS if not (root / marker).exists()]
    if missing:
        raise RuntimeError(f"Not a SocialGraph-FM checkout ({', '.join(missing)} missing): {root}")


def environment_python(root: Path) -> Path:
    return root / ("Scripts/python.exe" if os.name == "nt" else "bin/python")


def _is_link_or_reparse(path: Path) -> bool:
    if path.is_symlink():
        return True
    try:
        attributes = getattr(path.lstat(), "st_file_attributes", 0)
    except OSError:
        return False
    return bool(attributes & 0x400)  # FILE_ATTRIBUTE_REPARSE_POINT


@dataclass(frozen=True)
class RuntimeLayout:
    project_root: Path

    @classmethod
    def discover(cls, explicit: str | Path | None = None) -> "RuntimeLayout":
        return cls(discover_project_root(explicit))

    @property
    def var_root(self) -> Path:
        return self.project_root / "var"

    @property
    def config_root(self) -> Path:
        return self.var_root / "config"

    @property
    def profile_file(self) -> Path:
        return self.config_root / "runtime-profile.json"

    @property
    def llm_config_file(self) -> Path:
        return self.config_root / "socialgraph-api.env"

    @property
    def log_root(self) -> Path:
        return self.var_root / "deploy" / "logs"

    @property
    def setup_log_file(self) -> Path:
        return self.log_root / "setup.log"

    @property
    def pid_root(self) -> Path:
        return self.var_root / "deploy" / "pids"

    @property
    def temp_root(self) -> Path:
        return self.var_root / "tmp"

    @property
    def cache_root(self) -> Path:
        return self.var_root / "gfm" / "cache"

    @property
    def api_root(self) -> Path:
        return self.project_root / "services" / "api"

    @property
    def gfm_package(self) -> Path:
        return self.project_root / "packages" / "gfm"

    @property
    def runtime_package(self) -> Path:
        return self.project_root / "packages" / "runtime"

    @property
    def web_root(self) -> Path:
        return self.project_root / "apps" / "web"

    @property
    def runtime_environment(self) -> Path:
        """The only public Python environment used by both managed processes."""

        return self.var_root / "runtime"

    @property
    def web_bundle_root(self) -> Path:
        return self.project_root / "bundles" / "web"

    @property
    def web_bundle_manifest(self) -> Path:
        return self.web_bundle_root / "manifest.json"

    @property
    def web_bundle_archive(self) -> Path:
        return self.web_bundle_root / "client.zip"

    @property
    def web_client_root(self) -> Path:
        return self.var_root / "web" / "client"

    @property
    def managed_environment_root(self) -> Path:
        # Legacy split-environment root retained only so onboarding can remove it.
        return self.var_root / "e"

    @property
    def legacy_managed_environment_root(self) -> Path:
        return self.var_root / "envs" / "managed"

    @property
    def gfm_home(self) -> Path:
        return self.var_root / "gfm"

    @property
    def core_runtime(self) -> Path:
        return self.gfm_home / "core-runtime"

    @property
    def serving_root(self) -> Path:
        return self.core_runtime / "serving"

    @property
    def serving_control(self) -> Path:
        return self.serving_root / "core-serving-control.json"

    @property
    def serving_token(self) -> Path:
        return self.serving_root / "session.token"

    @property
    def serving_artifacts(self) -> Path:
        return self.core_runtime / "serving-graphs"

    @property
    def dataset_store(self) -> Path:
        return self.core_runtime / "api" / "dataset-store"

    @property
    def bindings_root(self) -> Path:
        return self.core_runtime / "api" / "gfm-run-bindings"

    @property
    def research_bindings_root(self) -> Path:
        return self.core_runtime / "api" / "gfm-research-run-bindings"

    @property
    def global_model_bindings_root(self) -> Path:
        return self.core_runtime / "api" / "gfm-global-model-run-bindings"

    @property
    def global_model_reviews_root(self) -> Path:
        return self.core_runtime / "api" / "gfm-global-model-reviews"

    @property
    def serving_high_water_root(self) -> Path:
        return self.core_runtime / "api" / "serving-control"

    @property
    def research_root(self) -> Path:
        return self.gfm_home / "research"

    @property
    def governance_root(self) -> Path:
        return self.gfm_home / "governance"

    @property
    def model_root(self) -> Path:
        return self.var_root / "models" / "socialgraph-global"

    @property
    def target_input_root(self) -> Path:
        return self.var_root / "governance" / "adaptation-inputs"

    @property
    def target_examples_root(self) -> Path:
        return self.var_root / "examples" / "target-domain"

    @property
    def bundle_manifest(self) -> Path:
        return self.project_root / "bundles" / "runtime-manifest.json"

    @property
    def install_profiles(self) -> Path:
        return self.gfm_package / "install-profiles.json"

    def initialize_directories(self) -> None:
        directories = (
            self.config_root,
            self.log_root,
            self.pid_root,
            self.temp_root,
            self.cache_root / "pip",
            self.cache_root / "uv",
            self.cache_root / "hf",
            self.cache_root / "torch",
            self.cache_root / "torchinductor",
            self.cache_root / "wandb",
            self.serving_root,
            self.serving_artifacts,
            self.dataset_store,
            self.bindings_root,
            self.research_bindings_root,
            self.global_model_bindings_root,
            self.global_model_reviews_root,
            self.serving_high_water_root,
            self.research_root,
            self.governance_root / "incoming",
            self.governance_root / "artifacts",
            self.governance_root / "runs",
            self.governance_root / "samples",
        )
        for directory in directories:
            self.assert_safe_var_path(directory)
            directory.mkdir(parents=True, exist_ok=True)
            self.assert_safe_var_path(directory)

    def initialize_config_directory(self) -> None:
        self.assert_safe_var_path(self.config_root)
        self.config_root.mkdir(parents=True, exist_ok=True)
        self.assert_safe_var_path(self.config_root)

    def assert_safe_var_path(self, path: Path) -> None:
        """Reject runtime paths that escape ignored var through links/reparse points."""

        var_root = Path(os.path.abspath(self.var_root))
        selected = Path(os.path.abspath(path))
        try:
            relative = selected.relative_to(var_root)
        except ValueError as error:
            raise RuntimeError(f"Runtime state path is outside var: {selected}") from error
        current = var_root
        if _is_link_or_reparse(current):
            raise RuntimeError(f"Runtime state path cannot contain links: {current}")
        for part in relative.parts:
            current = current / part
            if _is_link_or_reparse(current):
                raise RuntimeError(f"Runtime state path cannot contain links: {current}")

    def initialize_serving_contracts(self) -> None:
        source_root = self.gfm_package / "contracts"
        for name in (
            "core-serving-control.json",
            "core-serving-registry.json",
            "core-serving-graph-catalog.json",
        ):
            source = source_root / name
            destination = self.serving_root / name
            self.assert_safe_var_path(destination)
            if not source.is_file():
                raise RuntimeError(f"Missing GFM serving contract: {source}")
            if not destination.exists():
                destination.write_bytes(source.read_bytes())
