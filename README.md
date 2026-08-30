# SocialGraph-FM

[简体中文](README.zh-CN.md) · [Technical reference](docs/REFERENCE.md) ·
[Skills reference](skills/README.md)

SocialGraph-FM is a local, graph-foundation-model workbench for social-network
governance. It joins deterministic graph inspection, real Global-model inference,
target-domain adaptation, evidence retrieval, controlled Skills, human review, and
report generation without turning model scores into automated decisions.

The public repository contains the complete user runtime: Global, In-domain,
Low-label, and Cross-domain checkpoints; Russia 01–04 and the full Russia input;
zero-shot and few-shot target tasks; the governance knowledge index; and 68 reviewed
cases. Training corpora, training runs, caches, credentials, and local user state are
not published.

> Model output is a prioritization signal for human review. It is not proof of
> identity, intent, wrongdoing, or a basis for automated sanctions.

## Why this project

SocialGraph-FM keeps five kinds of information separate and traceable: imported graph
facts, model predictions, deterministic derived clues, retrieved material, and human
conclusions. This makes the system useful for governance research, coordinated-behavior
analysis, anomaly/community investigation, teaching demonstrations, and platform-risk
prototyping while preserving an auditable human decision boundary.

Ordinary CSV, JSON, GraphML, and GEXF files support mapping, visualization, and
deterministic structural analysis. Only a validated, hash-bound Global inference
package can enter the model path, so ordinary topology is never presented as a model
prediction.

## Architecture

```text
Browser
  │ loopback HTTP; no provider key or direct model filesystem access
  ▼
apps/web        React governance and research workbench
  │
  ▼
services/api    Torch-free FastAPI validation, state, confirmation, and LLM boundary
  │ authenticated internal loopback HTTP
  ▼
packages/gfm    Isolated PyTorch/PyG inference, adaptation, and retrieval process
```

All three services bind to loopback addresses. The configured LLM key is stored below
ignored `var/` state and injected only into the API process; it is not passed to the
browser or GFM process.

## Requirements and support

- CPython 3.12
- Node.js 24.x and npm 11.x
- An OpenAI-compatible or Anthropic-compatible model API for LLM-assisted workflows

| Platform | Runtime profile | Release status |
| --- | --- | --- |
| Windows x86-64 | CPU, PyTorch 2.8 / PyG 2.8 | Required CI path |
| Windows x86-64 with NVIDIA GPU | CUDA 13.0, PyTorch 2.12 / PyG 2.8 | Validated on a temporary self-hosted GPU runner for releases |
| Ubuntu glibc x86-64 | CPU, PyTorch 2.8 / PyG 2.8 | Required CI path |
| macOS ARM64 | CPU, PyTorch 2.8 / PyG 2.8 | Best-effort, non-blocking |

Linux CUDA, Intel macOS, MPS, ROCm, musl, and other architectures are not current
release commitments. Git is needed to clone or maintain the repository, but a GitHub
Download ZIP can be onboarded without Git.

The default wheel profile is CPU. CUDA wheels are installed only when the user passes
`--wheel-profile cuda`. Wheel selection and runtime device policy are independent:
`--device-policy auto` uses verified CUDA when available and otherwise the verified CPU
fallback, `cpu` forces CPU execution, and `cuda-required` rejects a host without working
CUDA.

## Quick start

Run these commands from the repository root (`python3` may replace `python` on POSIX):

```console
python scripts/socialgraph.py onboard
python scripts/socialgraph.py start --llm-mode required
python scripts/socialgraph.py stop
```

After `start`, open `http://127.0.0.1:5173`; run `stop` when the session is finished.
`onboard` checks Python and platform compatibility, creates or safely reuses isolated
API/GFM environments, installs the default CPU profile, verifies bundled assets, and
guides model-API configuration. To opt into the Windows CUDA profile, run
`python scripts/socialgraph.py onboard --wheel-profile cuda --device-policy auto`.

Windows users may use `scripts/onboard.ps1`, `scripts/start.ps1`, and
`scripts/stop.ps1` as equivalent wrappers.

## Model API configuration

The onboarding wizard supports one active channel at a time:

| Channel | Protocol | Default authentication |
| --- | --- | --- |
| OpenAI | Responses | Bearer |
| DeepSeek | Chat Completions | Bearer |
| GLM | Chat Completions | Bearer |
| Anthropic | Messages | `x-api-key` |
| Custom OpenAI-compatible relay | Chat Completions or Responses | Bearer |
| Custom Anthropic-compatible relay | Messages | `x-api-key` or Bearer |

Supply the exact model ID and endpoint requested by the provider. The key is entered
through hidden input and stored only in ignored `var/config/socialgraph-api.env` with
restricted permissions. Remote plaintext HTTP, embedded credentials, URL query strings
and fragments, redirects, inherited proxies, malformed endpoints, and oversized
responses are rejected. ChatGPT subscriptions, Codex client login, and Claude Code
login are not API credentials.

Use `--llm-mode required` for the complete assisted workflow, `optional` to allow the
deterministic fallback when no API is configured, or `disabled` to prevent private LLM
configuration from being loaded.

## Complete user workflows

- Import ordinary CSV, JSON, GraphML, and GEXF graphs; map fields; validate quality;
  create immutable GraphVersions; restore sessions; and explore topology.
- Load a compatible governance input and execute a real Global model forward on CPU or
  validated CUDA; preserve scores, representations, and expert routing with source
  hashes.
- Compare Global, In-domain, Low-label, and Cross-domain protocols without changing
  their immutable checkpoints.
- Register zero-shot and few-shot target tasks and complete governed target-domain
  adaptation.
- Rank candidate nodes and relations; discover coordination groups; inspect bounded
  two-hop evidence; and retrieve knowledge and similar reviewed cases.
- Create cases, record append-only human review events, restore governance sessions,
  and export JSON, Markdown, or HTML reports.
- Use the optional LLM only through a closed intent/Skill boundary with explicit
  confirmation for model execution and persisted report drafts.

Examples are under `examples/governance/`. Onboarding installs visible target-task
copies below ignored `var/examples/target-domain/` for native file pickers.

## Skills

`skills/governance/catalog.json` is the sole machine-readable source for the eight
public Governance Skills in namespace `socialgraph-fm.product-skills.governance`.

| Skill | Access |
| --- | --- |
| `inspect_graph` | Read-only |
| `run_governance_analysis` | Explicit confirmation before model execution |
| `get_evidence_subgraph` | Read-only |
| `discover_coordination_groups` | Read-only |
| `rank_coordination_relations` | Read-only |
| `retrieve_similar_cases` | Read-only |
| `get_model_dataset_cards` | Read-only |
| `draft_review_report` | Explicit confirmation before persistence |

The four experimental Core Skills remain isolated in a different namespace under
`skills/core/`; they are not aliases for Governance Skills and are not added to the
Governance API catalog. Parameter schemas, routes, implementation mapping, hash
provenance, confirmation behavior, and failure boundaries are documented in the
[Skills reference](skills/README.md) and [Chinese Skills reference](skills/README.zh-CN.md).

## Repository layout

```text
apps/web/                 React governance and research workbench
services/api/             Torch-free public API and orchestration boundary
packages/gfm/             Global, Governance, Core, and Research model code
packages/runtime/         Cross-platform setup and lifecycle manager
bundles/models/           Hash-bound model assets
bundles/governance/       Knowledge and reviewed-case indexes
examples/governance/      Russia and target-domain inputs
contracts/core/           Experimental Core serving contracts
skills/governance/        Eight governed product Skills and public schemas
skills/core/              Four isolated experimental Core Skills
scripts/                  Setup, lifecycle, export, and verification tools
docs/status/readiness.json  Experimental Core research-gate status only
var/                      Ignored credentials, environments, logs, and user state
```

`docs/status/readiness.json` reports only the formal research and serving gates of the
experimental Core milestone. Its false gates do not mean that the complete Governance
user runtime or Global model workflow is unavailable.

See the [technical reference](docs/REFERENCE.md) for environment reuse, provider
protocols, model/data identity, runtime state, troubleshooting, and release checks.

## Governance and responsibility boundary

SocialGraph-FM is a local research and decision-support system, not a hosted monitoring
or enforcement service. It does not establish identity or intent, automatically block
accounts, impose penalties, or replace contextual investigation. Operators remain
responsible for data rights, lawful use, target-domain validation, interpretation,
human review, and any downstream decision.

Processed examples contain anonymized node identifiers, graph structure, and
precomputed features rather than original usernames, posts, or URLs. New-domain scores
retain Global calibration and must be treated as an unvalidated ranking reference.
Retrieved documents and reviewed cases cannot rewrite graph facts or model scores.

## Publication and license

Maintainers can validate a checkout with the component commands in the
[technical reference](docs/REFERENCE.md) and create a clean GitHub/ZIP artifact with:

```console
python scripts/socialgraph.py export-github --repository ../SocialGraph_FM-github --zip ../SocialGraph_FM-github.zip
```

Source code is licensed under [Apache-2.0](LICENSE). Attribution, redistribution, and
research provenance are recorded in [NOTICE](NOTICE),
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md), and [CITATION.cff](CITATION.cff).
