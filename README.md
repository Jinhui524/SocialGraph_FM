# SocialGraph-FM

面向社交网络治理的本地图基础模型工作台：把真实 GFM 推理、关系证据、协同行为线索、案例检索和人工复核组织在同一套可追溯流程中。

![SocialGraph-FM 界面](apps/web/public/assets/socialgraph-atlas-light.webp)

> 模型分数只用于安排人工复核顺序，不证明账号身份、意图或违规事实，也不能作为自动处置依据。

## 主要能力

- 导入 CSV、JSON、GraphML 和 GEXF，完成字段映射、质量校验、图谱浏览与会话恢复。
- 运行随仓库发布的 Global 图基础模型，并比较 In-domain、Low-label 和 Cross-domain 协议。
- 完成 zero-shot / few-shot 目标域适配，保留 checkpoint、图版本和来源哈希。
- 查看重点账号、协同群组、事实关系、潜在线索、证据子图和相似复核案例。
- 使用大模型生成受证据约束的问答、摘要和研判报告，再由人员记录结论并导出报告。

## 环境要求

- Windows x64 或 Ubuntu x64
- CPython 3.12
- 可访问的 OpenAI-compatible Chat Completions API

公开运行版固定使用 CPU。普通用户不需要 CUDA、Node.js、npm、Conda，也不需要自行安装训练或研究依赖。首次安装仍会下载 PyTorch CPU、PyG 和 `pyg-lib`，它们用于运行真实图模型与固定邻居采样，因此不能删除。

## 三步启动

在仓库根目录执行（Ubuntu 可将 `python` 换成 `python3`）：

```console
python scripts/socialgraph.py onboard
python scripts/socialgraph.py start
python scripts/socialgraph.py stop
```

`onboard` 自动创建唯一的 `var/runtime` Python 环境、安装当前平台的 CPU 锁定依赖、校验模型和预构建 Web，并引导配置大模型。启动后访问：

```text
http://127.0.0.1:5173
```

也可以单独重新配置或检查：

```console
python scripts/socialgraph.py configure-llm
python scripts/socialgraph.py doctor
```

## 配置大模型

只需提供三项：

```text
大模型 API 地址
模型 ID
API Key
```

系统固定使用 OpenAI-compatible Chat Completions、Bearer 鉴权、15 秒超时、温度 0 和最多 700 tokens。根地址、`/v1` 地址及完整 `/chat/completions` 地址会自动规范化。

远程服务必须使用 HTTPS；本机回环地址可使用 HTTP。配置向导会先进行真实连通验证，验证失败不会完成保存。API Key 写入 Git 忽略的私有配置，只注入 API 进程，不进入浏览器、GFM 进程、日志或 Git。

大模型是完整系统的必需组件。缺少配置、认证失败、模型不存在、限流、超时或返回非法结构时，相关问答与报告会明确失败，不会生成本地替代回答。

## 研判 Assistant Skills

这些只读 Skills 必须调用已配置的大模型；它们不会直接修改案件或保存人工结论。

| Skill | 界面功能 | 使用的治理能力 |
| --- | --- | --- |
| `answer_governance_question` | 对话研究与研判助手自由追问 | 按问题选择允许的只读 Governance Skills |
| `summarize_node_evidence` | 治理应用“智能证据研判” | 图谱概况、证据子图、关系排序 |
| `generate_global_situation_report` | 研判助手“全局态势报告” | 图谱概况、群组、事实关系、潜在线索 |
| `generate_account_evidence_report` | “当前账号证据报告” | 图谱概况、证据子图、事实与潜在关系 |
| `generate_coordination_report` | “群组与关系研判报告” | 图谱概况、协同群组、关系排序 |
| `generate_case_review_draft` | “人工研判草稿”预览 | 当前研判单、证据、群组和关系 |

## Governance Skills

| Skill | 作用 | 权限 |
| --- | --- | --- |
| `inspect_graph` | 查看图规模、关系模态与覆盖情况 | 只读 |
| `run_governance_analysis` | 准备并运行 Global 治理分析 | 执行前确认 |
| `get_evidence_subgraph` | 获取账号绑定的证据子图 | 只读 |
| `discover_coordination_groups` | 分页查看协同群组 | 只读 |
| `rank_coordination_relations` | 排序事实关系与潜在线索 | 只读 |
| `retrieve_similar_cases` | 检索已审结的相似案例 | 只读 |
| `get_model_dataset_cards` | 查看模型、数据与输入合同 | 只读 |
| `draft_review_report` | 生成可保存的确定性案件草稿 | 保存前确认 |

每个 Skill 的输入、输出、界面位置、底层调用、失败行为和实现路径见 [Skills 索引](skills/README.md)。`skills/governance/catalog.json` 仍是 8 个底层 Governance Skills 的唯一机器源。

## 项目结构

```text
apps/web/             React 前端源码（普通用户运行预构建版本）
services/api/         API、LLM 边界、状态与确认
packages/gfm/         图模型推理、适配、检索与治理 Skills
packages/runtime/     单环境安装与两进程生命周期
bundles/              模型、治理索引和预构建 Web
examples/governance/  Russia 与目标域示例
skills/               Assistant、Governance 和实验 Core Skills
docs/                 中文技术参考与实验状态
scripts/              用户入口和发布校验工具
var/                  Git 忽略的环境、凭据、日志和用户状态
```

普通用户只运行 API/Web 和 GFM 两个回环进程：API 在 `127.0.0.1:5173` 提供接口及静态页面，GFM 在内部 `127.0.0.1:8766` 提供隔离推理。开发 Web 源码时才需要 Node/npm。

## 使用边界

SocialGraph-FM 是本地研究和辅助决策系统，不是托管监控或自动执法服务。使用者需要自行保证数据权利、合法用途、目标域验证、语境核验和最终人工决策。普通图的结构统计不会被标记成模型预测，潜在线索也不会被标记成已登记事实关系。

`docs/status/readiness.json` 只描述实验 Core 研究门禁，不代表 Governance 用户运行版不可用。更多安装锁、模型身份、API、研究依赖和排障信息见 [技术参考](docs/REFERENCE.md)。

## 许可证与引用

代码使用 [Apache-2.0](LICENSE)。归属、第三方来源和研究引用见 [NOTICE](NOTICE)、[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) 和 [CITATION.cff](CITATION.cff)。
