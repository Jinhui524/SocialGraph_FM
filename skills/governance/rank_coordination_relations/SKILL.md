# `rank_coordination_relations`：排序协同关系

- 输入：`runId`、分页参数，以及可选 `relationKind` 和关系模态过滤；见 [`../schemas/public/parameters/rank_coordination_relations.schema.json`](../schemas/public/parameters/rank_coordination_relations.schema.json)。
- 输出：稳定排序的关系摘要、两端账号、模态、分值与事实/潜在线索标记。
- 权限：只读；不需要确认。
- API：`POST /api/v2/gfm/governance/skills/execute`，Skill 固定为 `rank_coordination_relations`。
- 实现：GFM 从运行制品分页，事实关系与潜在线索保持两条不可混淆的结果轨道。
- 失败：未知关系类型、模态越界、运行不匹配或分页无效时 fail closed。
- 来源：[`../catalog.json`](../catalog.json) 同名条目；响应绑定结果与运行来源哈希。
