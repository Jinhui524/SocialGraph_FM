# `draft_review_report`：起草并保存复核报告

- 输入：`caseId` 和 `format`（`markdown` 或 `json`）；见 [`../schemas/public/parameters/draft_review_report.schema.json`](../schemas/public/parameters/draft_review_report.schema.json)。
- 输出：首次执行返回绑定草稿摘要和一次性确认票据；确认后保存确定性案件草稿。
- 权限：写入报告状态；必须显式确认 `save_draft_report`。
- API：先调用 `/skills/execute` 或 `/skills/draft_review_report/execute`，再调用 `/skills/confirm`。
- 实现：API 绑定案件 revision、票据和原子写入；GFM 只组装登记事实、模型发现、派生线索和人工事件。
- 失败：案件为空或过期、格式无效、票据重放、来源哈希变化或写入失败时不保存部分草稿。
- 来源：[`../catalog.json`](../catalog.json) 同名条目及案件/图/模型哈希。

LLM 生成的“人工研判草稿”只使用 `generate_case_review_draft` 预览；它不能代替本 Skill 的确认保存流程。
