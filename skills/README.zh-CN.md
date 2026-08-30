# SocialGraph-FM Skills 说明

[English](README.md) · [项目主页](../README.zh-CN.md)

本目录包含两套有意隔离的 Skill 合同：

- Governance：namespace 为 `socialgraph-fm.product-skills.governance` 的 8 个正式产品 Skills。
- Core：namespace 为 `socialgraph-fm.product-skills.core` 的 4 个实验研究 Skills。

这些 catalog 是 SocialGraph-FM 本机运行时合同，不是通用 Agent 配置文件。Core 名称不会扩展或
别名化 Governance catalog。

## Governance 正式目录

[`governance/catalog.json`](governance/catalog.json) 是名称、顺序、权限、确认动作、参数 Schema
位置和内部命令的唯一机器源。下表严格保持 catalog 顺序。

| Skill | 用途 | 权限与确认 | 参数 Schema | 失败边界 |
| --- | --- | --- | --- | --- |
| `inspect_graph` | 返回有界图计数和模态覆盖，可限定规范节点 scope 或已有 run。 | 只读；无需确认。 | [`inspect_graph`](governance/schemas/public/parameters/inspect_graph.schema.json) | 拒绝额外字段、无效 run 身份、超过 100 个 scope 节点或不可用图；不会返回模型预测。 |
| `run_governance_analysis` | 准备 Global 治理运行及有界候选数量。 | 会改变状态；模型执行前必须使用短时显式确认。 | [`run_governance_analysis`](governance/schemas/public/parameters/run_governance_analysis.schema.json) | 首次调用不会创建 run；非 Global 协议、非法 `topK`、图/模型漂移、无效或过期确认及模型失败均 fail closed。 |
| `get_evidence_subgraph` | 为已完成 run 中的一个节点追踪有界、受哈希绑定的证据子图。 | 只读；无需确认。 | [`get_evidence_subgraph`](governance/schemas/public/parameters/get_evidence_subgraph.schema.json) | 拒绝未知 run/node 或身份不一致；证据只是有界投影，不代表因果证明。 |
| `discover_coordination_groups` | 分页读取已完成 run 的确定性协同群组摘要。 | 只读；无需确认。 | [`discover_coordination_groups`](governance/schemas/public/parameters/discover_coordination_groups.schema.json) | 拒绝未知 run 和越界分页；不会创建或修改案件。 |
| `rank_coordination_relations` | 分页读取事实关系优先级或潜在线索优先级。 | 只读；无需确认。 | [`rank_coordination_relations`](governance/schemas/public/parameters/rank_coordination_relations.schema.json) | 拒绝未知 run、非法分页/模态，以及为 `potential` 线索指定模态；潜在关系仍是线索而非图事实。 |
| `retrieve_similar_cases` | 按一个案件或有界 run 对象检索已成功索引的结案案例。 | 只读；无需确认。 | [`retrieve_similar_cases`](governance/schemas/public/parameters/retrieve_similar_cases.schema.json) | 必须二选一：`caseId`，或 `runId` 加 `kindEntries`；拒绝不可用/未索引案例及跨模型或来源身份漂移。 |
| `get_model_dataset_cards` | 返回已登记的模型卡、数据卡和输入合同卡。 | 只读；无需确认。 | [`get_model_dataset_cards`](governance/schemas/public/parameters/get_model_dataset_cards.schema.json) | 不接受参数；已登记卡片或其绑定身份无法验证时失败。 |
| `draft_review_report` | 生成确定性的 Markdown 或 JSON 案件复核草稿，并受控保存。 | 会改变状态；持久化前必须使用短时显式确认。 | [`draft_review_report`](governance/schemas/public/parameters/draft_review_report.schema.json) | 首次调用不持久化草稿；拒绝未知案件、不支持的格式、已变化案件上下文或无效/过期确认。 |

旧版私有分析能力仅迁移为唯一规范公开名称 `run_governance_analysis`，不提供兼容别名；一次性名称
映射记录见 [English reference](README.md)。

## Governance 公开 API

稳定 base 为 `/api/v2/gfm/governance`：

| Method 与 path | 合同 |
| --- | --- |
| `GET /skills` | 返回有序 catalog、已解析参数 Schema、权限和规范 `catalogHash`。 |
| `POST /skills/execute` | 请求 body 携带 Skill 名称及完整图/模型上下文。 |
| `POST /skills/{skill}/execute` | 对 path 指定的 Skill 执行相同的严格上下文和参数校验。 |
| `POST /skills/confirm` | 消费先前已准备状态变更动作的一次性确认 token。 |

完整请求 Schema 为
[`governance/schemas/public/skill-request.schema.json`](governance/schemas/public/skill-request.schema.json)。
每个请求都绑定 `artifactId`、`datasetContentHash`、`graphVersionHash`、`modelVersionId` 和
`modelStateHash`，额外属性会被拒绝。Catalog Schema 与确定性正/负向 vectors 位于
[`governance/schemas`](governance/schemas) 和 [`governance/vectors`](governance/vectors)。

## Governance 实现与来源哈希

合同通过四层受检链路：

```text
skills/governance/catalog.json + public JSON Schemas
  → services/api/app/governance_skill_runtime/  校验、确认、审计
  → packages/gfm/src/socialgraph_gfm/governance/skill_executor.py  执行
  → apps/web/src/generated/governanceSkillsContract.ts  生成的客户端合同
```

不要手工编辑生成的 Web 合同。Catalog/API/GFM/Web 的顺序、权限、命令、Schemas 与 vectors 由
parity tests 约束。

`GET /skills` 暴露规范 SHA-256 `catalogHash`。Skill 请求绑定数据集、GraphVersion 和模型状态
哈希；隔离 GFM 结果包含规范 `provenance.inputHash` 与实现版本。确认票据还绑定 action 和 request
digest，并且短时、一次性有效。审计记录保存 request/response hashes，不接受调用方自报来源。

未知 Skill、参数格式或大小非法、catalog 漂移、图/模型/来源不一致、GFM 返回无效、确认过期或
重复使用、运行制品不可用时，所有层都会 fail closed。只读 Skills 不能写入复核或报告状态。LLM
只能自动选择 allowlist 内的只读 Skills，不能绕过确认、替换图事实或改写模型分数。

## 实验 Core 目录

[`core/catalog.json`](core/catalog.json) 是隔离的 Core 合同，并保持下列顺序。Core 请求/响应使用
严格版本化 Pydantic models，实现位于
[`packages/gfm/src/socialgraph_gfm/core/skills.py`](../packages/gfm/src/socialgraph_gfm/core/skills.py)。
它们只处理已登记的图、finding 和知识记录，返回数据但不持久化 Governance 状态，也不使用
Governance 确认路由。

| Skill | 用途 | 权限与确认 | Schema 来源 | 失败边界 |
| --- | --- | --- | --- | --- |
| `generate_report` | 根据已登记 finding hashes 输出确定性 Markdown 或 JSON 报告。 | 只读 registry 操作；无 Governance 确认。 | [`core/catalog.json`](core/catalog.json) 中的请求/响应版本 ID，以及 Core 实现中的严格 models。 | 拒绝重复或未知 finding hashes 和不支持的格式；输出明确标记为不使用 LLM。 |
| `inspect_graph` | 对一个 graph hash 和可选规范 scope 统计已登记节点/边。 | 只读 registry 操作；无 Governance 确认。 | 同上。 | 拒绝未知 graph hash，以及重复或未知 scope 节点；只返回静态事实。 |
| `retrieve_evidence` | 检索已登记 FTS 知识和可选结构记录。 | 只读 registry 操作；无 Governance 确认。 | 同上。 | 拒绝非法 query、limit 或结构 hash；检索分数非因果，也不是标签。 |
| `run_core_task` | 针对一个 Core task、graph 和 scope 返回已登记 finding hashes。 | 只读 registry 操作；无 Governance 确认。 | 同上。 | 拒绝未知 graph/scope/task 合同，不生成虚构 finding，也不执行自然语言 plan。 |

`docs/status/readiness.json` 中的机器状态只适用于实验 Core 的正式研究/serving gates，不决定
Governance catalog 或完整 Global 模型用户流程是否可用。

## 修改纪律

Governance 合同变更应首先修改 `governance/catalog.json` 及其源 Schemas，再使用仓库 generator
重建受检镜像并运行 catalog parity tests。名称、顺序、权限、确认动作、Schema 或内部命令变化
均属于公开合同变化。不得合并 Core/Governance namespace，也不得为了兼容而削弱哈希或确认校验。
