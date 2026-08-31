# LLM Assistant Skills

本目录保存面向界面功能的六个只读大模型 Skills。唯一机器源是
[`catalog.json`](catalog.json)，详细说明位于各 Skill 的 `SKILL.md`。

| Skill | 对应界面 | 底层 Governance Skills |
| --- | --- | --- |
| `answer_governance_question` | 对话研究与研判助手 | 按问题选择只读 Skills |
| `summarize_node_evidence` | 智能证据研判 | 图概况、证据子图、关系排名 |
| `generate_global_situation_report` | 全局态势报告 | 图概况、群组、事实关系、潜在线索 |
| `generate_account_evidence_report` | 当前账号证据报告 | 图概况、证据子图、关系排名 |
| `generate_coordination_report` | 群组与关系研判报告 | 图概况、群组、关系排名 |
| `generate_case_review_draft` | 人工研判草稿预览 | 研判单、证据、群组、关系 |

发现接口为 `GET /api/v2/gfm/governance/assistant/skills`，执行接口为
`POST /api/v2/gfm/governance/assistant/execute`。所有执行都要求大模型已经验证成功；
LLM 或证据调用失败时返回明确错误，不生成确定性替代回答。
