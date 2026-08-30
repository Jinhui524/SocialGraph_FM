# SocialGraph-FM Reference

This document is the complete technical and operational reference for the public
SocialGraph-FM runtime. The root README remains the quick-start entry point.

## Runtime architecture

SocialGraph-FM runs three isolated loopback processes:

1. `apps/web` provides graph import, visualization, governance, adaptation, retrieval,
   review, and reporting interfaces.
2. `services/api` validates public requests, stores local workflow state, manages
   confirmation tickets, and optionally calls the configured model API. It must not
   import Torch.
3. `packages/gfm` loads PyTorch/PyG, SocialGraph-FM Global, Governance, Core, and
   Research modules. It communicates with the API through authenticated internal HTTP.

The browser never receives provider credentials or direct GFM filesystem access. The
API validates public contracts independently from the stricter internal GFM contracts.
Model, graph, threshold, task, and execution-environment identities are recorded with
each governed run.

## Supported platforms and wheels

| Platform | CPU wheel | CUDA wheel |
| --- | --- | --- |
| Windows x86-64 | PyTorch 2.8 / PyG 2.8 | PyTorch 2.12 / CUDA 13.0 |
| glibc Linux x86-64 | PyTorch 2.8 / PyG 2.8 | Profile available; real NVIDIA release gate required |
| macOS 15 Apple silicon | PyTorch 2.8 / PyG 2.8 | Not supported |

The install catalog is `packages/gfm/install-profiles.json`. It fixes Python, platform,
architecture, PyTorch, PyG, extension wheels, package indexes, and hash-locked
requirements. Arbitrary wheel URLs and silent source builds are rejected. CUDA requires
a matching wheel backend and driver; a local CUDA Toolkit or `nvcc` is not required.

### Environment selection

```console
python scripts/socialgraph.py setup \
  --wheel-profile auto|cpu|cuda|PROFILE_ID \
  --device-policy auto|cpu|cuda-required \
  --env-mode auto|reuse|managed
```

- `auto` checks explicit interpreters, the last verified profile, the active venv or
  Conda environment, and an existing managed generation before creating a new one.
- `reuse` is strictly read-only. It runs imports, exact version checks, `pip check`, PyG
  ABI checks, NeighborLoader, checkpoint loading, and real forwards without invoking
  pip writes.
- `managed` creates separate API and GFM generations under `var/e/`, verifies the new
  generation, and atomically switches runtime-profile v3.

Explicit interpreters are supported:

```console
python scripts/socialgraph.py setup --wheel-profile cpu --env-mode reuse \
  --api-python /absolute/path/to/api/python \
  --gfm-python /absolute/path/to/gfm/python
```

The API interpreter must contain FastAPI, HTTPX, NumPy, Pydantic, multipart, Uvicorn,
and related runtime dependencies, and must not contain Torch. The GFM interpreter must
match one exact verified wheel profile. Parent `PYTHONPATH`, user-site packages, pip
configuration, and ambient model-provider credentials are excluded from probes.

### Execution device

The default device policy is `auto`:

```python
device = requested_device or torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)
```

A CPU wheel always resolves to CPU. A CUDA wheel resolves to CUDA when availability,
tensor operations, and a real model forward pass; otherwise `auto` uses the verified
CPU fallback. If CUDA is detected but the real CUDA forward fails, setup and startup
fail closed. `cuda-required` rejects a missing or unusable GPU.

Dynamic GPU state is not part of the software fingerprint. Each service start resolves
the device again and records an execution-environment hash; a running process never
migrates devices.

## Onboarding and lifecycle

Recommended interactive flow:

```console
python scripts/socialgraph.py onboard
python scripts/socialgraph.py start --llm-mode required
python scripts/socialgraph.py stop
```

Non-interactive onboarding requires explicit wheel, provider, model, and stdin key
arguments. For example, with a trusted OpenAI-compatible relay:

```console
printf '%s\n' "$MODEL_API_KEY" | python scripts/socialgraph.py onboard \
  --wheel-profile cpu \
  --device-policy auto \
  --env-mode managed \
  --preset custom \
  --api-base https://relay.example/v1 \
  --model exact-model-id \
  --api-mode responses \
  --auth-scheme bearer \
  --api-key-stdin
```

Use `--llm-mode optional|required|disabled` at startup. `required` rejects missing or
partial configuration; `optional` uses deterministic fallback when configuration is
absent; `disabled` never loads the private configuration. Startup never installs or
changes dependencies.

Diagnostics:

```console
python scripts/socialgraph.py doctor
python scripts/socialgraph.py doctor --full --test-llm --json
```

Full diagnostics validate runtime fingerprints, wheel backend, GPU driver, bundle
hashes, Russia 01–04, all four checkpoints, CPU fallback, target task copies, process
state, and the optional model API. Output never contains keys or environment values.

## Model API channels

The provider wizard separates channel, base URL, protocol, model, timeout,
authentication, and key. Supported protocols are:

- OpenAI Chat Completions
- OpenAI Responses
- Anthropic Messages with `anthropic-version: 2023-06-01`

OpenAI-compatible requests default to Bearer authentication. Anthropic-compatible
requests default to `x-api-key`; a relay may explicitly select Bearer. The final endpoint
is derived from a root URL, `/v1` base, or complete `/chat/completions`, `/responses`, or
`/v1/messages` endpoint.

Configuration is written atomically to `var/config/socialgraph-api.env`. Windows uses a
restricted ACL. POSIX uses a `0700` directory and `0600` file. The launcher clears
ambient OpenAI, Anthropic, DeepSeek, GLM, Gemini, and other common provider variables,
then injects only the private `LLM_*` allowlist into the API child.

Provider requests disable environment proxies and redirects, cap responses at 2 MiB,
and classify failures as authentication, endpoint/protocol, request/model, rate limit,
upstream, timeout, network/TLS, or invalid response. A structured-output retry is
allowed only for an explicit 400/422 unsupported-field response.

## Models and data

`bundles/models/socialgraph-global/` contains four immutable protocol checkpoints:

| Protocol | Purpose | Runtime status |
| --- | --- | --- |
| Global | Multi-domain primary model | Online serving |
| In-domain | Fully supervised target-domain reference | Frozen comparison |
| Low-label | Label-limited target-domain reference | Frozen comparison |
| Cross-domain | Source-only transfer reference | Frozen comparison |

Checkpoint envelopes bind the release, coordination-risk task, protocol, model version,
model-state hash, expert mask, model configuration, and tensor state. Rebranding changes
packaging identities but not tensor-state hashes. The model card records intended use,
limitations, metrics, licensing, and research provenance.

`examples/governance/russia/` contains Russia 01–04 and one full input. The target-domain
directory contains zero-shot and few-shot task archives bound by catalog and receipt
hashes. Setup verifies the tracked runtime manifest and installs model, knowledge,
reviewed cases, examples, and target upload copies without modifying tracked files.

The public runtime does not include the full six-country training corpus, training runs,
or preliminary Research runtime artifacts. Training and Research source code remains
available, but the public release promises the complete user workflow rather than full
training reproduction.

## Governance Skills

`skills/governance/catalog.json` is the only source of truth for the eight public
Skills. API models, GFM commands, generated TypeScript types, JSON Schemas, examples,
and tests must match its exact order and permissions.

`run_governance_analysis` and persistence through `draft_review_report` require a
short-lived confirmation ticket. Tickets bind the command, parameters, graph version,
model version, model state, and relevant artifact hashes. Read-only Skills cannot write
review or report state.

Core Skills under `skills/core/` have a separate namespace and are never merged into the
Governance catalog. These are product contracts, not agent configuration files.

## Public and internal APIs

The public surface contains exactly 96 method/path pairs. SocialGraph-FM Global uses
`/api/v1/gfm/global-model/*`; Governance uses `/api/v2/gfm/governance/*`; Research uses
`/api/v1/gfm/research/*`; general Core orchestration remains below `/api/v1/gfm/*`.
Branded predecessor routes are intentionally absent.

The GFM process exposes authenticated internal Global, Governance, Core, and Research
routes only on loopback. Public and internal payload limits, archive safety checks, and
Pydantic validation are independent.

## Runtime state

All mutable state is ignored below `var/`:

```text
var/config/                 Private configuration and runtime profile
var/e/                      Managed Python generations
var/models/socialgraph-global/
var/gfm/governance/        Uploads, runs, knowledge, and reviewed cases
var/gfm/core-runtime/      Core serving and API stores
var/gfm/research/          Research runtime state
var/governance/            Target adaptation inputs
var/deploy/                Logs and PID records
```

PID records bind process ID, start time, executable, and command identity. Windows uses
independent process groups; POSIX uses sessions and process groups with TERM followed by
bounded KILL. Stop refuses to terminate a reused PID whose identity no longer matches.

## Troubleshooting

- **Python rejected:** use CPython 3.12 and rerun setup with an explicit bootstrap
  interpreter if necessary.
- **API environment rejected:** remove Torch from the API environment or let managed
  mode create a separate environment.
- **GFM environment rejected:** choose a catalog wheel profile that exactly matches
  Python, platform, Torch, PyG, and compiled extensions.
- **CUDA wheel falls back to CPU:** run `doctor --full`; inspect driver availability,
  `torch.version.cuda`, `torch.cuda.is_available()`, and the real-forward result.
- **Startup reports profile drift:** the recorded environment changed; rerun setup.
- **Model installation rejected:** do not edit bundled files. Restore the checkout and
  rerun setup from an empty ignored model destination.
- **Model API test fails:** edit only the reported base, model, protocol, authentication,
  or key field and retry. Response bodies and credentials are deliberately hidden.
- **Ports unavailable:** stop the owning process or configure the supported loopback
  port environment variables before startup.

## Development and release checks

The repository CI builds platform-independent Runtime, API, and GFM wheels and runs:

- Runtime, API, and GFM pytest, Ruff, and mypy suites
- Web typecheck, unit tests, production build, and Playwright
- 96-route and Skill contract parity
- recursive brand, secret, personal-path, binary, license, and Markdown-link scans
- CPU managed/reuse clean-clone gates on Windows, Linux, and macOS
- real CUDA model gates on configured NVIDIA runners
- Russia 01–04, target adaptation, retrieval, review, report, and LLM-assisted E2E

Common local commands are documented in `CONTRIBUTING.md`. Publication artifacts are
created through `python scripts/socialgraph.py export-github`; the tool produces a clean
one-commit repository and a ZIP containing the same tracked bytes without `.git`.
