# SocialGraph-FM

[English](README.md) · [技术参考](docs/REFERENCE.md) ·
[Skills 说明](skills/README.zh-CN.md)

SocialGraph-FM 是一套以图基础模型为核心、面向社交网络治理的本机工作台。它把确定性图分析、
Global 模型真实推理、目标域适配、证据检索、受控 Skills、人工复核和报告生成连成完整闭环，
同时避免把模型分数直接转化为自动决策。

公开仓库包含完整用户运行版：Global、In-domain、Low-label、Cross-domain 四类 checkpoint，
Russia 01—04 与完整 Russia 输入，zero-shot/few-shot 目标任务，治理知识索引，以及 68 个已复核
案例。完整训练语料、训练运行、缓存、凭据和本地用户状态不在公开范围内。

> 模型输出只用于确定人工核查优先级，不构成身份、意图或违规事实证明，也不能作为自动处罚依据。

## 项目价值

SocialGraph-FM 将导入的图事实、模型预测、确定性派生线索、检索资料和人工结论分层保存并绑定
来源。它可用于社交治理研究、协同行为分析、异常社区发现、教学答辩和平台风控原型验证，且保留
可审计的人工决策边界。

普通 CSV、JSON、GraphML 和 GEXF 文件可用于字段映射、可视化和确定性结构分析；只有通过校验、
受哈希约束的 Global 推理包才能进入模型路径，因此系统不会把普通拓扑结果描述成模型预测。

## 系统架构

```text
浏览器
  │ 回环 HTTP；不接触模型 API Key，也不直接读取模型文件
  ▼
apps/web        React 治理与研究工作台
  │
  ▼
services/api    不加载 Torch 的 FastAPI 校验、状态、确认和 LLM 边界
  │ 经认证的内部回环 HTTP
  ▼
packages/gfm    隔离的 PyTorch/PyG 推理、适配与检索进程
```

三个服务均只绑定回环地址。LLM Key 保存在被 Git 忽略的 `var/` 状态中，并且只注入 API 进程；
浏览器和 GFM 进程均不会接收该 Key。

## 环境要求与支持矩阵

- CPython 3.12
- Node.js 24.x 与 npm 11.x
- 使用 LLM 辅助流程时，需要 OpenAI-compatible 或 Anthropic-compatible 模型 API

| 平台 | 运行配置 | 发布状态 |
| --- | --- | --- |
| Windows x86-64 | CPU，PyTorch 2.8 / PyG 2.8 | 必测 CI 路径 |
| Windows x86-64 + NVIDIA GPU | CUDA 13.0，PyTorch 2.12 / PyG 2.8 | 发布时使用临时 self-hosted GPU runner 实机验证 |
| Ubuntu glibc x86-64 | CPU，PyTorch 2.8 / PyG 2.8 | 必测 CI 路径 |
| macOS ARM64 | CPU，PyTorch 2.8 / PyG 2.8 | best-effort、非阻断 |

Linux CUDA、Intel macOS、MPS、ROCm、musl 和其他架构不属于当前发布承诺。克隆或维护仓库时需要
Git；通过 GitHub Download ZIP 获取的项目无需 Git 也可完成 onboarding。

默认 wheel profile 为 CPU；只有显式传入 `--wheel-profile cuda` 才会安装 CUDA wheels。
Wheel 类型与实际执行设备是两个独立概念：`--device-policy auto` 会在 CUDA 真实验证成功时使用
CUDA，否则使用经过验证的 CPU fallback；`cpu` 强制 CPU；`cuda-required` 会拒绝没有可用 CUDA
的主机。

## 三步启动

在仓库根目录依次运行（POSIX 系统可用 `python3` 替代 `python`）：

```console
python scripts/socialgraph.py onboard
python scripts/socialgraph.py start --llm-mode required
python scripts/socialgraph.py stop
```

执行 `start` 后打开 `http://127.0.0.1:5173`，使用完毕后再执行 `stop`。`onboard` 会检查 Python
与平台兼容性，创建或安全复用隔离的 API/GFM 环境，安装默认 CPU profile，验证捆绑制品，并引导
配置模型 API。如需选择 Windows CUDA profile，可运行
`python scripts/socialgraph.py onboard --wheel-profile cuda --device-policy auto`。

Windows 用户也可使用等价的 `scripts/onboard.ps1`、`scripts/start.ps1` 和
`scripts/stop.ps1` 包装脚本。

## 模型 API 配置

引导程序一次启用一个通道：

| 通道 | 协议 | 默认鉴权方式 |
| --- | --- | --- |
| OpenAI | Responses | Bearer |
| DeepSeek | Chat Completions | Bearer |
| GLM | Chat Completions | Bearer |
| Anthropic | Messages | `x-api-key` |
| 自定义 OpenAI-compatible relay | Chat Completions 或 Responses | Bearer |
| 自定义 Anthropic-compatible relay | Messages | `x-api-key` 或 Bearer |

请填写服务商要求的准确模型 ID 和 endpoint。Key 通过隐藏输入读取，只保存在被 Git 忽略的
`var/config/socialgraph-api.env`，并使用受限文件权限。系统拒绝远程明文 HTTP、URL 内嵌凭据、
query/fragment、重定向、继承代理、格式错误的 endpoint 和超大响应。ChatGPT 订阅、Codex 客户端
登录和 Claude Code 登录均不能替代 API 凭据。

完整辅助流程使用 `--llm-mode required`；`optional` 允许在未配置 API 时使用确定性 fallback；
`disabled` 则完全不加载私有 LLM 配置。

## 完整用户功能

- 导入普通 CSV、JSON、GraphML、GEXF；映射字段、检查质量、生成不可变 GraphVersion、恢复会话并
  交互查看图结构。
- 对兼容治理输入执行 CPU 或已验证 CUDA 上的 Global 模型真实前向，并用来源哈希绑定分数、表示与
  expert 路由。
- 比较 Global、In-domain、Low-label、Cross-domain 协议，同时保持 checkpoint 不可变。
- 注册 zero-shot/few-shot 目标任务，并完成受控目标域适配。
- 对候选节点和关系排序，发现协同群组，查看有界两跳证据，并检索知识和相似已复核案例。
- 创建案件、追加人工复核事件、恢复治理会话，并导出 JSON、Markdown 或 HTML 报告。
- 让可选 LLM 仅通过封闭意图/Skill 边界工作；模型执行和报告草稿持久化均要求显式确认。

示例位于 `examples/governance/`。Onboarding 会把目标任务的可见副本安装到被忽略的
`var/examples/target-domain/`，便于原生文件选择器读取。

## Skills

`skills/governance/catalog.json` 是 8 个正式 Governance Skills 的唯一机器源，namespace 为
`socialgraph-fm.product-skills.governance`。

| Skill | 权限 |
| --- | --- |
| `inspect_graph` | 只读 |
| `run_governance_analysis` | 模型执行前要求显式确认 |
| `get_evidence_subgraph` | 只读 |
| `discover_coordination_groups` | 只读 |
| `rank_coordination_relations` | 只读 |
| `retrieve_similar_cases` | 只读 |
| `get_model_dataset_cards` | 只读 |
| `draft_review_report` | 持久化前要求显式确认 |

4 个实验 Core Skills 使用独立 namespace 并保存在 `skills/core/`；它们不是 Governance Skills 的
别名，也不会加入 Governance API catalog。参数 Schema、API 路由、实现映射、来源哈希、确认行为
和失败边界详见 [中文 Skills 说明](skills/README.zh-CN.md) 与
[English Skills reference](skills/README.md)。

## 仓库目录

```text
apps/web/                 React 治理与研究工作台
services/api/             不加载 Torch 的公开 API 与编排边界
packages/gfm/             Global、Governance、Core、Research 模型代码
packages/runtime/         跨平台安装与生命周期管理器
bundles/models/           受哈希约束的模型制品
bundles/governance/       知识索引和已复核案例索引
examples/governance/      Russia 与目标域输入
contracts/core/           实验 Core serving contracts
skills/governance/        8 个正式 Skills 与公开 Schema
skills/core/              4 个隔离的实验 Core Skills
scripts/                  安装、生命周期、导出与验证工具
docs/status/readiness.json  仅描述实验 Core 研究门禁状态
var/                      被忽略的凭据、环境、日志和用户状态
```

`docs/status/readiness.json` 只记录实验 Core milestone 的正式研究与 serving gates。其中为 false 的
门禁不表示完整 Governance 用户运行版或 Global 模型流程不可用。

环境复用、Provider 协议、模型/数据身份、运行状态、排障和发布检查见
[技术参考](docs/REFERENCE.md)。

## 治理与责任边界

SocialGraph-FM 是本机研究与辅助决策系统，不是公网监控或自动执法服务。它不做身份或意图定性，
不自动封禁账号或实施处罚，也不能替代结合语境的调查。使用者需要对数据权利、合法用途、目标域
验证、结果解释、人工复核及任何下游决策负责。

公开 processed 示例仅包含匿名节点标识、图结构和预计算特征，不包含原始用户名、帖子或 URL。
新域分数沿用 Global calibration，只能视作未经目标域验证的排序参考。检索文档和已复核案例不能
改写图事实或模型分数。

## 发布与许可

维护者可使用[技术参考](docs/REFERENCE.md)中的组件命令验证 checkout，并通过下列命令创建干净的
GitHub 仓库与 ZIP：

```console
python scripts/socialgraph.py export-github --repository ../SocialGraph_FM-github --zip ../SocialGraph_FM-github.zip
```

源代码使用 [Apache-2.0](LICENSE) 许可。署名、再分发信息与研究来源记录在 [NOTICE](NOTICE)、
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) 和 [CITATION.cff](CITATION.cff) 中。
