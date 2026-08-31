# 智能证据研判

- 界面：治理应用中的“智能证据研判”
- 输入：已完成运行和当前选中的账号
- 接口：`POST /api/v2/gfm/governance/assistant/execute`，要求 `runId` 和 node `selectedTarget`
- 输出：证据子图、事实关系与潜在线索的区分说明
- 调用：`inspect_graph`、`get_evidence_subgraph`、`rank_coordination_relations`
- 权限：只读；不需要确认；不修改图、模型结果或研判单
- 实现：`services/api/app/governance_skill_runtime/gateway.py`
- 来源：`skills/assistant/catalog.json` 是唯一机器源；响应携带证据与审计哈希
- 失败：上下文不完整、绑定不匹配、LLM 或底层 Skill 失败时显式返回错误
