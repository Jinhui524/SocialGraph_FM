# 全局态势报告

- 界面：研判助手中的“全局态势报告”
- 输入：已完成的治理运行
- 接口：`POST /api/v2/gfm/governance/assistant/execute`，要求 `runId`
- 输出：图概况、候选账号、协同群组、事实关系、潜在线索与人工核验提示
- 调用：`inspect_graph`、`discover_coordination_groups`、两次 `rank_coordination_relations`
- 权限：只读；不需要确认；不生成新的模型运行
- 实现：`services/api/app/governance_skill_runtime/gateway.py`
- 来源：`skills/assistant/catalog.json` 是唯一机器源；响应携带证据与审计哈希
- 失败：任何必要证据或 LLM 输出不可用时显式返回错误
