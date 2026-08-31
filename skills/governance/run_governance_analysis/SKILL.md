# `run_governance_analysis`：运行治理分析

这是 8 个 Governance Skills 中负责真实模型执行的确认型 Skill。

- 输入：`protocol` 固定为 `global`，可选 `topK`；见 [`../schemas/public/parameters/run_governance_analysis.schema.json`](../schemas/public/parameters/run_governance_analysis.schema.json)。
- 输出：首次执行只返回绑定请求摘要与一次性确认票据；确认成功后返回 Governance run。
- 权限：写入运行状态；必须显式确认 `run_governance_analysis`。
- API：先调用 `/skills/execute` 或 `/skills/run_governance_analysis/execute`，再调用 `/skills/confirm`。
- 实现：API 负责票据、幂等与审计；GFM 执行 hash-bound Global checkpoint forward。
- 失败：票据过期或重放、图/模型身份变化、输入不兼容或真实 forward 失败时不创建伪结果。
- 来源：[`../catalog.json`](../catalog.json) 同名条目及模型/图来源哈希；确认前不得启动运行。
