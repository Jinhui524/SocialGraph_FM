# SocialGraph-FM Skills 索引

本目录把界面中的大模型研判能力与底层治理操作分成两个独立 namespace：

- `socialgraph-fm.product-skills.assistant`：6 个只读、必须调用 LLM 的研判 Skills。
- `socialgraph-fm.product-skills.governance`：8 个受合同约束的图治理 Skills。

Assistant Skills 只能编排允许的只读 Governance Skills，不得执行模型、写入案件或绕过确认。机器合同分别以 [`assistant/catalog.json`](assistant/catalog.json) 和 [`governance/catalog.json`](governance/catalog.json) 为准。

## Assistant Skills 与界面对应

| Skill | 用户看到的位置 | 底层 Governance Skills | 失败行为 |
| --- | --- | --- | --- |
| `answer_governance_question` | 对话研究、研判助手自由追问 | 按问题选择只读 Skills | LLM 或证据调用失败时返回 502/503，不生成替代回答 |
| `summarize_node_evidence` | 证据档案 → 智能证据研判 | `inspect_graph`、`get_evidence_subgraph`、`rank_coordination_relations` | 缺少运行或账号上下文时拒绝执行 |
| `generate_global_situation_report` | 研判助手 → 全局态势报告；分析完成报告 | `inspect_graph`、`discover_coordination_groups`、两类关系排序 | 缺少成功运行时拒绝执行 |
| `generate_account_evidence_report` | 研判助手 → 当前账号证据报告 | `inspect_graph`、`get_evidence_subgraph`、两类关系排序 | 未选择账号时拒绝执行 |
| `generate_coordination_report` | 研判助手 → 群组与关系研判报告 | `inspect_graph`、`discover_coordination_groups`、两类关系排序 | 缺少成功运行时拒绝执行 |
| `generate_case_review_draft` | 研判助手 → 人工研判草稿 | 当前案件、证据、群组、关系 | 只返回预览；缺少案件或案件版本过期时拒绝执行 |

统一接口：

```text
GET  /api/v2/gfm/governance/assistant/skills
POST /api/v2/gfm/governance/assistant/execute
```

请求明确携带 Assistant Skill ID、问题、图/模型身份及受限上下文；响应包含答案、只读调用 trace、证据引用、引用哈希和审计哈希。Assistant 响应没有 `intent`、回答模式或本地 fallback 字段。

## Governance Skills

以下顺序与 [`governance/catalog.json`](governance/catalog.json) 完全一致。

| Skill | 作用 | 只读 | 确认 | 参数 Schema | 说明 |
| --- | --- | --- | --- | --- | --- |
| `inspect_graph` | 返回图规模与关系模态覆盖 | 是 | 否 | [`inspect_graph.schema.json`](governance/schemas/public/parameters/inspect_graph.schema.json) | [SKILL.md](governance/inspect_graph/SKILL.md) |
| `run_governance_analysis` | 准备 Global 治理分析 | 否 | 执行前 | [`run_governance_analysis.schema.json`](governance/schemas/public/parameters/run_governance_analysis.schema.json) | [SKILL.md](governance/run_governance_analysis/SKILL.md) |
| `get_evidence_subgraph` | 获取账号绑定证据子图 | 是 | 否 | [`get_evidence_subgraph.schema.json`](governance/schemas/public/parameters/get_evidence_subgraph.schema.json) | [SKILL.md](governance/get_evidence_subgraph/SKILL.md) |
| `discover_coordination_groups` | 分页读取协同群组 | 是 | 否 | [`discover_coordination_groups.schema.json`](governance/schemas/public/parameters/discover_coordination_groups.schema.json) | [SKILL.md](governance/discover_coordination_groups/SKILL.md) |
| `rank_coordination_relations` | 分页排序事实关系或潜在线索 | 是 | 否 | [`rank_coordination_relations.schema.json`](governance/schemas/public/parameters/rank_coordination_relations.schema.json) | [SKILL.md](governance/rank_coordination_relations/SKILL.md) |
| `retrieve_similar_cases` | 检索已成功索引的审结案例 | 是 | 否 | [`retrieve_similar_cases.schema.json`](governance/schemas/public/parameters/retrieve_similar_cases.schema.json) | [SKILL.md](governance/retrieve_similar_cases/SKILL.md) |
| `get_model_dataset_cards` | 返回绑定的模型、数据与输入合同 | 是 | 否 | [`get_model_dataset_cards.schema.json`](governance/schemas/public/parameters/get_model_dataset_cards.schema.json) | [SKILL.md](governance/get_model_dataset_cards/SKILL.md) |
| `draft_review_report` | 创建待保存的确定性案件草稿 | 否 | 保存前 | [`draft_review_report.schema.json`](governance/schemas/public/parameters/draft_review_report.schema.json) | [SKILL.md](governance/draft_review_report/SKILL.md) |

统一底层接口：

```text
GET  /api/v2/gfm/governance/skills
POST /api/v2/gfm/governance/skills/execute
POST /api/v2/gfm/governance/skills/{skill}/execute
POST /api/v2/gfm/governance/skills/confirm
```

API 在 `services/api/app/governance_skill_runtime/` 校验外部合同、图/模型身份、确认票据和审计记录；GFM 在 `packages/gfm/src/socialgraph_gfm/governance/skill_executor.py` 执行内部命令。参数 Schema、Web 生成合同、API models、GFM catalog 与测试向量必须保持同名、同序和同权限。

所有调用 fail closed：未知 Skill、越界参数、过期图或案件、模型身份不一致、缺少运行制品、确认票据过期/重放以及来源哈希不一致均不会降级为其他 Skill。`draft_review_report` 生成的是经确认保存的确定性案件草稿；`generate_case_review_draft` 仅生成 LLM 预览，二者不能互换。

## 实验 Core Skills

`skills/core/` 中 4 个 Core Skills 属于独立实验 namespace，不是 Governance Skills 的别名，也不进入公开治理 API。`docs/status/readiness.json` 仅记录这部分研究门禁。

| Skill | 实验用途 |
| --- | --- |
| `generate_report` | 根据已登记 finding hashes 生成确定性 Markdown 或 JSON 报告 |
| `inspect_graph` | 按 graph hash 和可选 scope 统计已登记节点与边 |
| `retrieve_evidence` | 检索实验 Core 的知识和结构证据记录 |
| `run_core_task` | 对已登记图和 scope 执行实验 Core task 合同 |

Migration note: the private predecessor capability formerly named `run_iohunter` maps to the sole public canonical name `run_governance_analysis`; no compatibility alias is exposed.
