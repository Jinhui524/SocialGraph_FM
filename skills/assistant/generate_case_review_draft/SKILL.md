# 人工研判草稿

- 界面：研判助手中的“人工研判草稿”
- 输入：当前研判单及其最新 `caseHash`，可附带当前目标
- 接口：`POST /api/v2/gfm/governance/assistant/execute`，要求 `caseId` 与最新 `caseHash`
- 输出：基于现有证据和复核进度的大模型只读预览
- 调用：`inspect_graph`、`get_evidence_subgraph`、`discover_coordination_groups`、`rank_coordination_relations`
- 权限：只读；不需要确认；不落库
- 实现：`services/api/app/governance_skill_runtime/gateway.py`
- 来源：`skills/assistant/catalog.json` 是唯一机器源；响应携带证据与审计哈希
- 失败：研判单哈希过期、证据绑定失败或 LLM 不可用时显式返回错误

需要保存确定性案件草稿时，仍使用低层 `draft_review_report` 并完成明确确认。
