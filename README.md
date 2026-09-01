# SocialGraph-FM

面向社交网络治理的本地图基础模型工作台：把真实 GFM 推理、关系证据、协同行为线索、案例检索和人工复核组织在同一套可追溯流程中。

![SocialGraph-FM 项目视觉背景](apps/web/public/assets/socialgraph.webp)

## 🌟主要能力

- 导入 CSV、JSON、GraphML 和 GEXF，完成字段映射、质量校验、图谱浏览与会话恢复。
- 根据输入的图数据，转换成可视化可交互的图谱，增进社交网络理解。
- 完成 zero-shot / few-shot 目标域适配，保留 checkpoint、图版本和来源哈希。
- 查看重点账号、协同群组、事实关系、潜在线索、证据子图和相似复核案例。
- 使用大模型生成受证据约束的问答、摘要和研判报告，再由人员记录结论并导出报告。

## ⚙️环境要求

- Windows x64 或 Ubuntu x64
- 64 位 CPython 3.12.x
- 一个可用的大模型 API 与对应的模型 ID、API Key

公开运行版固定使用 CPU。普通用户不需要 CUDA、Node.js、npm、Conda，也不需要自行安装训练或研究依赖。`onboard` 会创建唯一的项目内隔离环境并安装锁定依赖；其中 PyTorch CPU、PyG 和 `pyg-lib` 用于真实图模型与固定邻居采样。

项目支持任意符合上述平台要求的 CPython 3.12 补丁版本，但**不支持 3.12 以外的主／次版本**。当前安装锁、预编译 wheel、启动器和 CI 均以 3.12 为发布基线。

## 🚀三步启动

1. 克隆仓库，或从 GitHub 选择 **Code → Download ZIP** 并解压：

```console
git clone https://github.com/Jinhui524/SocialGraph_FM.git
cd SocialGraph_FM
```

2. 在仓库根目录完成环境与大模型配置（Ubuntu 可将 `python` 换成 `python3`）：

```console
python scripts/socialgraph.py onboard
```

3. 启动系统：

```console
python scripts/socialgraph.py start
```

浏览器访问 `http://127.0.0.1:5173`。使用结束后停止受管进程：

```console
python scripts/socialgraph.py stop
```

需要重新配置大模型或检查安装时运行：

```console
python scripts/socialgraph.py configure-llm
python scripts/socialgraph.py doctor
```

## 📁配置大模型

向导先让用户选择服务商，再预填对应 API 地址。模型 ID 不设易过期的默认值，始终由用户从服务商控制台复制；API 地址仍可编辑。

| 选择 | 服务商                   | 预填 API 地址                                             |
| ---: | ------------------------ | --------------------------------------------------------- |
|    1 | OpenAI 官方              | `https://api.openai.com/v1`                               |
|    2 | DeepSeek 官方            | `https://api.deepseek.com`                                |
|    3 | 通义千问                 | `https://dashscope.aliyuncs.com/compatible-mode/v1`       |
|    4 | Gemini OpenAI-compatible | `https://generativelanguage.googleapis.com/v1beta/openai` |
|    5 | MiniMax 中国             | `https://api.minimaxi.com/v1`                             |
|    6 | MiniMax 国际             | `https://api.minimax.io/v1`                               |
|    7 | OpenRouter               | `https://openrouter.ai/api/v1`                            |
|    8 | 自定义 OpenAI-compatible | 由用户填写                                                |

无论选择哪一项，最终只需确认三项：

```text
大模型 API 地址
模型 ID
API Key
```

向导隐藏 API Key，并在保存前进行真实连通验证；验证失败不会保存。Qwen 业务空间用户可以把预填地址替换为控制台提供的专属 OpenAI-compatible 地址。现有中转服务继续通过“自定义 OpenAI-compatible”配置。

如果 LLM 连通验证失败，已经验证完成的 CPU 环境、模型和运行资源会继续保留，不需要重新下载。先运行 `python scripts/socialgraph.py doctor` 查看本地环境状态；本地 runtime 正常时，修正地址、模型或密钥后直接运行 `python scripts/socialgraph.py configure-llm`。网络诊断码与恢复方法见[技术参考](docs/REFERENCE.md#llm-连接诊断与恢复)。

OpenAI 官方配置需要从 [OpenAI API Platform](https://platform.openai.com/api-keys) 创建 API Key，并单独开通 API 用量；ChatGPT 或 Codex 的订阅、登录凭据不能替代 API Key。API Key 写入 Git 忽略的私有配置，只注入 API 进程，不进入浏览器、GFM 进程、日志或 Git。

## 🎮研判 Assistant Skills

这些只读 Skills 必须调用已配置的大模型

| Skill                              | 界面功能                   | 使用的治理能力                         |
| ---------------------------------- | -------------------------- | -------------------------------------- |
| `answer_governance_question`       | 对话研究与研判助手自由追问 | 按问题选择允许的只读 Governance Skills |
| `summarize_node_evidence`          | 治理应用“智能证据研判”     | 图谱概况、证据子图、关系排序           |
| `generate_global_situation_report` | 研判助手“全局态势报告”     | 图谱概况、群组、事实关系、潜在线索     |
| `generate_account_evidence_report` | “当前账号证据报告”         | 图谱概况、证据子图、事实与潜在关系     |
| `generate_coordination_report`     | “群组与关系研判报告”       | 图谱概况、协同群组、关系排序           |
| `generate_case_review_draft`       | “人工研判草稿”预览         | 当前研判单、证据、群组和关系           |

## 📝Governance Skills

| Skill                          | 作用                           | 权限       |
| ------------------------------ | ------------------------------ | ---------- |
| `inspect_graph`                | 查看图规模、关系模态与覆盖情况 | 只读       |
| `run_governance_analysis`      | 准备并运行 Global 治理分析     | 执行前确认 |
| `get_evidence_subgraph`        | 获取账号绑定的证据子图         | 只读       |
| `discover_coordination_groups` | 分页查看协同群组               | 只读       |
| `rank_coordination_relations`  | 排序事实关系与潜在线索         | 只读       |
| `retrieve_similar_cases`       | 检索已审结的相似案例           | 只读       |
| `get_model_dataset_cards`      | 查看模型、数据与输入合同       | 只读       |
| `draft_review_report`          | 生成可保存的确定性案件草稿     | 保存前确认 |

每个 Skill 的输入、输出、界面位置、底层调用、失败行为和实现路径见 [Skills 索引](skills/README.md)。`skills/governance/catalog.json` 仍是 8 个底层 Governance Skills 的唯一机器源。

## 🔧项目结构

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

## 📄使用边界

SocialGraph-FM 是本地研究和辅助决策系统，不是托管监控或自动执法服务。使用者需要自行保证数据权利、合法用途、目标域验证、语境核验和最终人工决策。普通图的结构统计不会被标记成模型预测，潜在线索也不会被标记成已登记事实关系。

`docs/status/readiness.json` 只描述实验 Core 研究门禁，不代表 Governance 用户运行版不可用。更多安装锁、模型身份、API、研究依赖和排障信息见 [技术参考](docs/REFERENCE.md)。

## 🤝许可证与引用

本项目采用 MIT 许可证 - 查看 [LICENSE](LICENSE) 文件了解详情。

## 📞 联系方式

如有问题或建议，请通过以下方式联系：

- 项目Issues：[GitHub Issues](https://github.com/Ar1haraNaN7mI/AI-Streamer-Phy/issues)
- 邮箱：请通过GitHub Issues联系
