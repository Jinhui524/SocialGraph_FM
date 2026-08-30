# SocialGraph-FM Skills Reference

[简体中文](README.zh-CN.md) · [Project README](../README.md)

This directory contains two deliberately separate Skill contracts:

- Governance: eight public product Skills in
  `socialgraph-fm.product-skills.governance`.
- Core: four experimental research Skills in
  `socialgraph-fm.product-skills.core`.

The catalogs are contracts for the local SocialGraph-FM runtime. They are not generic
agent configuration files, and the Core names do not extend or alias the Governance
catalog.

## Governance catalog

[governance/catalog.json](governance/catalog.json) is the sole machine-readable source
of names, order, permissions, confirmation actions, parameter-Schema locations, and
internal commands. The following rows preserve its exact order.

| Skill | Purpose | Access and confirmation | Parameter Schema | Failure boundary |
| --- | --- | --- | --- | --- |
| `inspect_graph` | Return bounded graph counts and modality coverage, optionally for a canonical node scope or existing run. | Read-only; no confirmation. | [`inspect_graph`](governance/schemas/public/parameters/inspect_graph.schema.json) | Rejects extra fields, an invalid run identity, more than 100 scope nodes, or an unavailable graph; never returns a model prediction. |
| `run_governance_analysis` | Prepare a Global governance run and its bounded candidate limit. | State-changing; execution requires a short-lived explicit confirmation. | [`run_governance_analysis`](governance/schemas/public/parameters/run_governance_analysis.schema.json) | The first call creates no run. It fails closed on non-Global protocol, invalid `topK`, graph/model drift, invalid or expired confirmation, or model failure. |
| `get_evidence_subgraph` | Trace a bounded, hash-bound evidence subgraph for one node in a completed run. | Read-only; no confirmation. | [`get_evidence_subgraph`](governance/schemas/public/parameters/get_evidence_subgraph.schema.json) | Rejects an unknown run/node or identity mismatch; evidence remains a bounded projection and not proof of causality. |
| `discover_coordination_groups` | Page through deterministic coordination-group summaries from a completed run. | Read-only; no confirmation. | [`discover_coordination_groups`](governance/schemas/public/parameters/discover_coordination_groups.schema.json) | Rejects unknown runs and pagination outside the Schema bounds; does not create or modify cases. |
| `rank_coordination_relations` | Page through factual-relation rankings or potential-clue rankings. | Read-only; no confirmation. | [`rank_coordination_relations`](governance/schemas/public/parameters/rank_coordination_relations.schema.json) | Rejects unknown runs, invalid pagination/modalities, and modalities on `potential` clues; potential relations remain clues, not graph facts. |
| `retrieve_similar_cases` | Retrieve successfully indexed, concluded review cases for one case or bounded run targets. | Read-only; no confirmation. | [`retrieve_similar_cases`](governance/schemas/public/parameters/retrieve_similar_cases.schema.json) | Requires exactly `caseId` or `runId` plus `kindEntries`; rejects unavailable/unindexed cases and cross-model or source-identity drift. |
| `get_model_dataset_cards` | Return the registered model, dataset, and input-contract cards. | Read-only; no confirmation. | [`get_model_dataset_cards`](governance/schemas/public/parameters/get_model_dataset_cards.schema.json) | Accepts no parameters and fails if registered cards or their bound identities cannot be validated. |
| `draft_review_report` | Create a deterministic Markdown or JSON case-review draft for controlled saving. | State-changing; persistence requires a short-lived explicit confirmation. | [`draft_review_report`](governance/schemas/public/parameters/draft_review_report.schema.json) | The first call does not persist the draft. It rejects unknown cases, unsupported formats, changed case context, or invalid/expired confirmation. |

Migration note: the private predecessor capability formerly named `run_iohunter` maps to the sole public canonical name `run_governance_analysis`; no compatibility alias is exposed.

## Public Governance API

The stable base is `/api/v2/gfm/governance`:

| Method and path | Contract |
| --- | --- |
| `GET /skills` | Return the ordered catalog, resolved parameter Schemas, permissions, and canonical `catalogHash`. |
| `POST /skills/execute` | Execute a request whose body contains the Skill name and full graph/model context. |
| `POST /skills/{skill}/execute` | Execute the named Skill with the same strict context and parameter validation. |
| `POST /skills/confirm` | Consume a single-use confirmation token for a previously prepared state-changing action. |

The full request Schema is
[governance/schemas/public/skill-request.schema.json](governance/schemas/public/skill-request.schema.json).
Every request binds `artifactId`, `datasetContentHash`, `graphVersionHash`,
`modelVersionId`, and `modelStateHash`; arbitrary extra properties are rejected. The
catalog schema and deterministic positive/negative vectors are under
[`governance/schemas`](governance/schemas) and
[`governance/vectors`](governance/vectors).

## Governance implementation and provenance

The contract flows through four checked layers:

```text
skills/governance/catalog.json + public JSON Schemas
  → services/api/app/governance_skill_runtime/  validation, confirmation, audit
  → packages/gfm/src/socialgraph_gfm/governance/skill_executor.py  execution
  → apps/web/src/generated/governanceSkillsContract.ts  generated client contract
```

Do not edit the generated Web contract by hand. Catalog/API/GFM/Web order, permissions,
commands, Schemas, and vectors are enforced by parity tests.

`GET /skills` exposes a canonical SHA-256 `catalogHash`. Skill requests bind dataset,
GraphVersion, and model-state hashes; isolated GFM results include a canonical
`provenance.inputHash` and implementation version. Confirmation tickets additionally
bind the action and request digest and are short-lived and single-use. Audit records
store request/response hashes rather than accepting caller-supplied provenance.

All layers fail closed on an unknown Skill, malformed or oversized parameters, catalog
drift, graph/model/source mismatch, invalid GFM result, expired or reused confirmation,
or unavailable runtime artifact. Read-only Skills cannot write review or report state.
An LLM may select only allowlisted read-only Skills automatically; it cannot bypass a
confirmation gate, replace graph facts, or alter model scores.

## Experimental Core catalog

[core/catalog.json](core/catalog.json) is the isolated Core contract and preserves this
order. Core requests and responses use strict versioned Pydantic models implemented in
[`packages/gfm/src/socialgraph_gfm/core/skills.py`](../packages/gfm/src/socialgraph_gfm/core/skills.py).
They operate only on registered graph/finding/knowledge records, return data without
persisting Governance state, and do not use the Governance confirmation routes.

| Skill | Purpose | Access and confirmation | Schema source | Failure boundary |
| --- | --- | --- | --- | --- |
| `generate_report` | Render a deterministic Markdown or JSON report from registered finding hashes. | Read-only registry operation; no Governance confirmation. | Request/response version IDs in [`core/catalog.json`](core/catalog.json); strict models in the Core implementation. | Rejects duplicate or unknown finding hashes and unsupported formats; output is explicitly generated without an LLM. |
| `inspect_graph` | Count registered nodes and edges for a graph hash and optional canonical scope. | Read-only registry operation; no Governance confirmation. | Same source. | Rejects unknown graph hashes and duplicate or unknown scoped nodes; returns static facts only. |
| `retrieve_evidence` | Search registered FTS knowledge and optional structural records. | Read-only registry operation; no Governance confirmation. | Same source. | Rejects invalid queries, limits, or structural hashes; retrieval scores are non-causal and not labels. |
| `run_core_task` | Return registered finding hashes for one Core task, graph, and scope. | Read-only registry operation; no Governance confirmation. | Same source. | Rejects unknown graph/scope/task contracts and never manufactures findings or executes a natural-language plan. |

The machine status in `docs/status/readiness.json` applies only to formal experimental
Core research/serving gates. It does not determine availability of the Governance
catalog or complete Global-model user workflow.

## Change discipline

Make Governance contract changes in `governance/catalog.json` and its source Schemas,
then regenerate checked mirrors with the repository generator and run catalog parity
tests. A name, order, permission, confirmation action, Schema, or internal-command
change is a public contract change. Never merge the Core and Governance namespaces or
weaken hash/confirmation checks for compatibility.
