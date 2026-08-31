# `get_evidence_subgraph`：获取证据子图

- 输入：成功运行的 `runId` 与其中的 `nodeId`；见 [`../schemas/public/parameters/get_evidence_subgraph.schema.json`](../schemas/public/parameters/get_evidence_subgraph.schema.json)。
- 输出：绑定账号、邻居、事实关系、结构信号和证据哈希组成的有界子图。
- 权限：只读；不需要确认，不改变选中对象或人工结论。
- API：`POST /api/v2/gfm/governance/skills/execute`，Skill 固定为 `get_evidence_subgraph`。
- 实现：API 校验身份后由 GFM `GovernanceSkillExecutor` 读取运行制品。
- 失败：运行未成功、账号不属于运行、运行与当前图/模型不一致或证据超界时 fail closed。
- 来源：[`../catalog.json`](../catalog.json) 同名条目；结果保留运行、图版本、模型状态与证据哈希。
