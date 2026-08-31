# `retrieve_similar_cases`：检索相似案例

- 输入：`caseId`，或 `runId` 加对象类型/ID；两种查询互斥，见 [`../schemas/public/parameters/retrieve_similar_cases.schema.json`](../schemas/public/parameters/retrieve_similar_cases.schema.json)。
- 输出：相似案件、语义/结构/关系构成分量、审结时间及记录来源哈希。
- 权限：只读；不需要确认，不复制或改写历史案件。
- API：`POST /api/v2/gfm/governance/skills/execute`；界面也通过受约束的相似案例搜索接口调用同一语义。
- 实现：API 校验案件状态和模型身份，GFM 读取已成功索引的审结记录。
- 失败：案件未审结、索引未就绪、查询轨道混用或身份不一致时 fail closed。
- 来源：[`../catalog.json`](../catalog.json) 同名条目；响应包含案例、索引和审计哈希。
