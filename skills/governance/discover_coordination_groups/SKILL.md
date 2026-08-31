# `discover_coordination_groups`：发现协同群组

- 输入：`runId` 及可选 `offset`、`limit`；见 [`../schemas/public/parameters/discover_coordination_groups.schema.json`](../schemas/public/parameters/discover_coordination_groups.schema.json)。
- 输出：有界群组列表、成员计数、关系模态与派生来源，不将群组解释为共同意图。
- 权限：只读；不需要确认。
- API：`POST /api/v2/gfm/governance/skills/execute`，Skill 固定为 `discover_coordination_groups`。
- 实现：GFM 从绑定运行的群组派生结果稳定分页，API 保留审计哈希。
- 失败：运行未成功、分页越界或来源身份漂移时 fail closed。
- 来源：[`../catalog.json`](../catalog.json) 同名条目；群组派生哈希随结果返回。
