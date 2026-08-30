# SocialGraph-FM

SocialGraph-FM is a cross-platform research system for social-graph analysis and
governance. It combines deterministic graph inspection, graph foundation model
inference, target-domain adaptation, evidence retrieval, controlled Skills, human
review, and report generation in one local workbench.

The public repository includes the complete user runtime: SocialGraph-FM Global,
In-domain, Low-label, and Cross-domain checkpoints; Russia 01–04 and the full Russia
input; zero-shot and few-shot target tasks; the governance knowledge index; and 68
reviewed cases. Training corpora, training runs, caches, and local user state are not
included.

> Model outputs are prioritization signals for human review. They are not proof of
> identity, intent, wrongdoing, or grounds for automated sanctions.

## Architecture

```text
apps/web      React governance workbench; never receives model API credentials
services/api  Torch-free FastAPI validation and orchestration process
packages/gfm  Isolated PyTorch/PyG model, adaptation, and retrieval process
```

All three services bind to loopback addresses. The LLM key is injected only into the
API process. Web and GFM child processes receive neither the configured key nor ambient
provider credentials from the parent shell.

## Requirements

- Python 3.12
- Node.js 24.x and npm 11.x
- Windows x86-64, glibc Linux x86-64, or macOS 15 on Apple silicon
- An OpenAI-compatible or Anthropic-compatible model API

Windows and Linux provide verified CPU wheel profiles. Windows also provides a
validated CUDA 13.0 profile; Linux CUDA remains gated by a real self-hosted NVIDIA
runner. macOS is CPU-only. Intel macOS, MPS, ROCm, musl, and ARM Linux/Windows are not
release targets.

Git is required only for cloning or maintaining the repository. A GitHub Download ZIP
runs without Git.

## Quick start

Run from the repository root (`python3` may replace `python` on POSIX systems):

```console
python scripts/socialgraph.py onboard
python scripts/socialgraph.py start --llm-mode required
```

Open `http://127.0.0.1:5173`. Stop the system with:

```console
python scripts/socialgraph.py stop
```

`onboard` prepares the small Torch-free API environment first, guides the user through
model API configuration and a real compatibility check, and only then offers a verified
CPU or CUDA wheel profile. A compatible external environment is reused read-only;
otherwise a lock-bound environment is created below ignored `var/`.

Wheel selection and execution device are independent. CPU wheels always execute on
CPU. With the default `auto` policy, a CUDA wheel executes on CUDA when a real model
forward succeeds and otherwise uses its verified CPU fallback. Use
`--device-policy cuda-required` to reject a host without working CUDA.

Windows users may run the equivalent PowerShell wrapper:

```powershell
.\scripts\onboard.ps1
.\scripts\start.ps1 -LlmMode Required
```

## Model API configuration

The terminal wizard supports one active channel at a time:

| Channel | Protocol | Default authentication |
| --- | --- | --- |
| OpenAI | Responses | Bearer |
| DeepSeek | Chat Completions | Bearer |
| GLM | Chat Completions | Bearer |
| Anthropic | Messages | `x-api-key` |
| Custom OpenAI-compatible relay | Chat Completions or Responses | Bearer |
| Custom Anthropic-compatible relay | Messages | `x-api-key` or Bearer |

The user supplies the exact model ID. Relay configuration accepts a root URL, a `/v1`
base, or a complete supported endpoint. Remote HTTP, embedded credentials, query
strings, fragments, malformed ports, redirects, proxy inheritance, and oversized
responses are rejected. The key is entered through hidden input and stored only in
ignored `var/config/socialgraph-api.env` with restricted platform permissions.

Claude Code, Codex client login, and ChatGPT subscriptions are not API credentials.

## Included workflows

- Import ordinary CSV, JSON, GraphML, GEXF, and supported graph packages.
- Run deterministic structural analysis without a model API or GFM inference.
- Run SocialGraph-FM Global on Russia 01–04 or a compatible governance input.
- Compare In-domain, Low-label, Cross-domain, and Global model protocols.
- Register zero-shot or few-shot target tasks and complete governed adaptation.
- Discover coordination groups, rank relations, inspect evidence subgraphs, retrieve
  similar reviewed cases, record human review, and generate reports.
- Use model-bound document retrieval and an optional LLM-assisted governance session.

Example inputs are under `examples/governance/`. Setup creates visible copies of target
task archives under `var/examples/target-domain/` for file pickers.

## Governance Skills

The machine-readable source is `skills/governance/catalog.json` with namespace
`socialgraph-fm.product-skills.governance`.

| Skill | Access |
| --- | --- |
| `inspect_graph` | Read-only |
| `run_governance_analysis` | Explicit confirmation required |
| `get_evidence_subgraph` | Read-only |
| `discover_coordination_groups` | Read-only |
| `rank_coordination_relations` | Read-only |
| `retrieve_similar_cases` | Read-only |
| `get_model_dataset_cards` | Read-only |
| `draft_review_report` | Explicit confirmation before persistence |

The four experimental Core Skills remain isolated under `skills/core/` and use a
separate namespace.

## Repository layout

```text
apps/web/                 Governance workbench
services/api/             Torch-free API gateway
packages/gfm/             Global, Governance, Core, and Research model code
packages/runtime/         Cross-platform setup and lifecycle manager
bundles/models/            Hash-bound model assets
bundles/governance/       Knowledge and reviewed-case indexes
examples/governance/      Russia and target-domain inputs
contracts/core/           Core serving contracts
skills/governance/        Eight governed product Skills
skills/core/              Four isolated Core Skills
scripts/                  Setup, lifecycle, export, and verification tools
var/                      Ignored local runtime state
```

Complete architecture, operations, model, data, Skill, troubleshooting, and developer
reference material is in [docs/REFERENCE.md](docs/REFERENCE.md).

## GitHub export

Maintainers can create a clean one-commit repository and a Download ZIP without
configuring or pushing a remote:

```console
python scripts/socialgraph.py export-github \
  --repository ../SocialGraph_FM-github \
  --zip ../SocialGraph_FM-github.zip
```

The export rejects runtime state, unknown binaries, credentials, personal paths,
unmanifested model files, stale contracts, and a ZIP above the release size budget.

## License and responsibility

Source code is licensed under [Apache-2.0](LICENSE). Third-party licenses and research
provenance are recorded in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) and in the
machine-readable model card. See [SECURITY.md](SECURITY.md),
[CONTRIBUTING.md](CONTRIBUTING.md), and [CITATION.cff](CITATION.cff).
