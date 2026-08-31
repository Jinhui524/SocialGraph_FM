# 治理问题问答

- 界面：对话研究与研判助手
- 输入：分析员问题，以及可选的运行、研判单和当前目标上下文
- 接口：`POST /api/v2/gfm/governance/assistant/execute`，参数由 Assistant catalog 约束
- 输出：大模型生成的有依据回答、只读 Skill 调用轨迹和来源哈希
- 调用：大模型仅可从六个只读 Governance Skills 中选择，最多执行四次检索
- 权限：只读；不需要确认；不会创建运行、保存报告或提交复核
- 实现：`services/api/app/governance_skill_runtime/gateway.py`
- 来源：`skills/assistant/catalog.json` 是唯一机器源；响应携带证据与审计哈希
- 失败：LLM 未配置、调用失败、输出不合法或证据调用失败时显式返回 502/503
