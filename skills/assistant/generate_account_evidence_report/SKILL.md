# 当前账号证据报告

- 界面：研判助手中的“当前账号证据报告”
- 输入：已完成运行和当前选中的账号
- 接口：`POST /api/v2/gfm/governance/assistant/execute`，要求 `runId` 和 node `selectedTarget`
- 输出：账号证据子图、事实关系、潜在线索及仍需人工核验的缺口
- 调用：`inspect_graph`、`get_evidence_subgraph`、`rank_coordination_relations`
- 权限：只读；不需要确认；不写入研判单
- 实现：`services/api/app/governance_skill_runtime/gateway.py`
- 来源：`skills/assistant/catalog.json` 是唯一机器源；响应携带证据与审计哈希
- 失败：账号不属于调用上下文、证据绑定失败或 LLM 不可用时显式返回错误
