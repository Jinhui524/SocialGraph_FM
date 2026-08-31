# `get_model_dataset_cards`：获取模型与数据卡

- 输入：空对象；见 [`../schemas/public/parameters/get_model_dataset_cards.schema.json`](../schemas/public/parameters/get_model_dataset_cards.schema.json)。
- 输出：有界模型、数据、协议和输入合同卡片及其内容哈希。
- 权限：只读；不需要确认，不读取 checkpoint 张量或训练语料。
- API：`POST /api/v2/gfm/governance/skills/execute`，Skill 固定为 `get_model_dataset_cards`。
- 实现：GFM 返回随发布物登记且经过 manifest 校验的卡片，API 保留来源链。
- 失败：卡片缺失、哈希不一致或模型身份不受支持时 fail closed。
- 来源：[`../catalog.json`](../catalog.json) 同名条目及发布 manifest。
