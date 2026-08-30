# Security policy

## Supported code

Security fixes target the current public `main` branch. This research prototype does not
publish supported version tags or promise long-term maintenance for historical snapshots.

## Reporting a vulnerability

Please do not open a public issue for a suspected vulnerability. Use GitHub's private
vulnerability-reporting or private security-advisory feature for this repository when it
is available. If that feature is unavailable, contact the repository maintainers through
the private contact method shown on the repository owner profile and ask for a secure
reporting channel before sending technical details.

Include only what is necessary:

- affected component and revision;
- impact and prerequisites;
- minimal reproduction steps using public fixtures or a generated minimal graph;
- whether a secret or private artifact may have been exposed; and
- suggested mitigation, if known.

Do not send real API keys, personal data, private graph data, or unrelated large binary
attachments. Replace sensitive values with clearly marked placeholders. If a live credential
may be exposed, revoke or rotate it immediately before continuing the report.

## Security scope

Examples of in-scope issues include:

- archive traversal, unsafe deserialization, link/reparse-point escape, or resource
  exhaustion in import/model-package paths;
- leakage of LLM keys, loopback tokens, private paths, graph facts, prompts, or model data;
- bypass of Skill confirmation, graph/model identity checks, authorization, or readiness;
- unintended non-loopback GFM exposure, redirect/proxy credential forwarding, or unsafe
  provider URL handling; and
- integrity failures that permit an unverified checkpoint, result, case, or audit record.

The absence of production authentication, tenant isolation, centralized monitoring, and
high availability is a documented prototype limitation, not by itself a vulnerability.
However, a code path that claims or bypasses those boundaries is in scope.

## Optional CUDA dependency exception

The optional CUDA 13.0 profile currently follows the upstream PyTorch 2.12 wheel's
`setuptools<82` dependency constraint. That prevents adopting the `setuptools>=83`
release that resolves CVE-2026-59890 in this profile. The public Offline and Windows CPU
profiles do not carry this exception: they use a current, audited setuptools release.
CUDA remains an explicit workstation-only profile and is not presented as a hardened
hosted deployment. This exception must be removed when
an otherwise compatible upstream CUDA wheel relaxes the pin; it does not permit ignoring
any other vulnerability.

## Disclosure

Maintainers will validate the report, coordinate a fix and tests, and agree on disclosure
timing based on severity and user impact. Do not publish exploit details before a fix or
explicit coordination. No bounty program is promised by this policy.
