# Contributing to SocialGraph-FM

Thank you for helping improve this research prototype. Contributions should preserve the
fail-closed model boundary, reproducible contracts, and explicit human-review semantics.

## Before opening a change

1. Read the [technical reference](docs/REFERENCE.md), including the architecture,
   runtime, contract, and release sections relevant to the change.
2. Open an issue for a large behavior or contract change so its research and governance
   implications can be reviewed before implementation.
3. Keep the change focused. Do not combine generated artifacts, directory migration, and
   behavior changes unless they are inseparable.

## Local setup

```console
python scripts/socialgraph.py setup --profile offline --env-mode auto
python scripts/socialgraph.py dev --llm-mode disabled
```

Use `--profile cpu` only for model component work that needs the locked CPU environment.
CUDA-specific changes also require a local CUDA doctor/smoke run, but CUDA is not a
prerequisite for ordinary contributions.

## Design requirements

- Keep React → Torch-free FastAPI → isolated GFM as the process boundary.
- Preserve existing public URLs and versioned wire contracts unless the change follows a
  documented compatibility and migration process.
- Treat `skills/governance/catalog.json` as the only Governance catalog source.
  Generated API/Web/GFM artifacts must remain in parity with it.
- Keep Governance and Core Skills in separate namespaces.
- Never fabricate a model result, readiness flag, probability, checkpoint, citation, or
  dataset authorization record.
- State-changing product Skills require explicit, short-lived, single-use confirmation.
- Store local state only under ignored `var/` paths. Code and docs must not contain a
  developer-specific absolute path.
- Update the English README and technical reference when public capabilities, setup,
  or safety boundaries change.

## Data, secrets, and examples

Do not commit:

- `.env` files or API keys;
- private keys, bearer tokens, credentials, or provider responses;
- model/data assets outside the exact paths, sizes, and SHA-256 values in
  `bundles/runtime-manifest.json`;
- unapproved user/platform data, identifiers, review notes, or restricted datasets;
- caches, environments, logs, screenshots containing real data, or files below `var/`.

Tests should use the smallest deterministic fixture that preserves the exercised
contract. The tracked Russia, target-domain, model, knowledge, and reviewed-case assets
are intentional runtime-bundle inputs; changes require regenerating and reviewing the
manifest rather than bypassing it.

## Checks

Before submitting:

```console
python scripts/socialgraph.py doctor --full
python -m pytest packages/runtime/tests
python -m pytest services/api/tests
python -m pytest packages/gfm/tests
npm --prefix apps/web run typecheck
npm --prefix apps/web test -- --run
npm --prefix apps/web run build
```

Install API and GFM development extras into separate contributor environments before
running their lint, type, or test suites; do not add them to the runtime environments.
On Windows, `scripts/verify.ps1` remains a convenience aggregator when those contributor
dependencies are already present. Run `pwsh -NoProfile -File scripts/secret-scan.ps1`
before publication; PowerShell 7 is a contributor-tool dependency for that check.

Component entry points are `npm --prefix apps/web ...` for Web,
`python -m app` from `services/api`, `python -m socialgraph_gfm` from `packages/gfm`,
and `python scripts/socialgraph.py` for repository lifecycle commands. Build each Python
wheel from its component directory and keep API and GFM contributor environments
separate.

At minimum, run the checks for each touched component. Documentation changes must have
valid relative links and contain no personal absolute paths. Contract changes require
catalog/API/GFM/Web parity tests and regenerated checked artifacts through the supported
generator—not manual edits to generated files.

## Pull requests

Describe:

- the problem and chosen behavior;
- public API, contract, data, safety, and compatibility impact;
- tests run and relevant environment profile; and
- any limitation or follow-up that remains.

Do not include secrets or private data in issues, pull requests, logs, or attachments.
Security vulnerabilities follow [SECURITY.md](SECURITY.md), not a public issue.

Unless stated otherwise, submitted contributions are licensed under Apache-2.0, the same
license as this repository. You must have the right to submit every contribution.
