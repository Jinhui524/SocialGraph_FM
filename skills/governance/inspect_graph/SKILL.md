# `inspect_graph`：图谱概况检查

用于对话研究、全局态势报告和其他只读研判的基础上下文。

- 输入：可选 `scopeNodeIds`、`runId` 和 `candidateLimit`，精确定义见 [`../schemas/public/parameters/inspect_graph.schema.json`](../schemas/public/parameters/inspect_graph.schema.json)。
- 输出：受限的节点/关系计数、模态覆盖和候选概况，不返回整图或私有原文。
- 权限：只读；不需要确认，不创建运行或案件记录。
- API：`POST /api/v2/gfm/governance/skills/execute`，Skill 固定为 `inspect_graph`。
- 实现：API gateway 转发到 `GovernanceSkillExecutor` 的同名内部命令。
- 失败：图/模型身份过期、范围节点未知或运行绑定不一致时 fail closed。
- 来源：[`../catalog.json`](../catalog.json) 同名条目；catalog、Schema、实现和响应来源哈希由 parity tests 校验。
