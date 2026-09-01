# SocialGraph-FM 技术参考

[根 README](../README.md) 是普通用户入口；本文面向开发、排障和发布验证。

## 运行架构

```text
浏览器
  └─ http://127.0.0.1:5173
       API + 预构建 Web（不加载 Torch，唯一持有 LLM Key 的进程）
         └─ 带会话令牌的内部回环 HTTP
              GFM 127.0.0.1:8766（PyTorch/PyG 推理、适配和检索）
```

公开运行版只有一个受管 Python 环境 `var/runtime`，解释器固定为 64 位 CPython 3.12.x；API/Web 与 GFM 仍是两个隔离进程。API Key 仅加入 API 子进程环境；浏览器、GFM、命令输出和日志均不得获得密钥。

API 在所有 `/api/` 路由之后挂载预构建静态站点和 SPA fallback。普通用户不会启动 Vite，也不会创建 `node_modules`。

项目不会读取仓库根目录、Web 或 API 源码目录中的 `.env` 文件。LLM 配置只能通过 `onboard` 或 `configure-llm` 写入受保护的私有配置；服务路径、会话令牌和其他高级变量由统一 launcher 注入，普通用户不应手工维护。

## 平台与依赖

| 平台 | 执行设备 | 发布状态 |
| --- | --- | --- |
| Windows x64 | CPU | 必测 |
| Ubuntu glibc x64 | CPU | 必测 |

公开运行版不提供 CUDA、MPS、ROCm、macOS、musl 或其他架构 profile，也不做运行设备自动选择。

Python 支持边界是 **64 位 CPython 3.12.x**：任意 3.12 补丁版本均可，但不声明从某个更早版本开始的开放范围。原因不是功能上偏好某个补丁版本，而是用户安装锁、PyTorch/PyG 预编译 wheel、启动探测和 Windows/Ubuntu CI 都共同绑定 3.12。宽泛地声明向上兼容会让用户在一个从未经过发布验证、且可能没有匹配 wheel 的解释器上开始安装。项目也不会在后台下载另一套解释器来掩盖版本不兼容。

用户环境保留：

- PyTorch 2.8 CPU、PyG 2.8、`pyg-lib` 0.6；
- NumPy、Pydantic、NetworkX；
- FastAPI、HTTPX、Pydantic Settings、python-multipart、基础 Uvicorn；
- SocialGraph-FM API、GFM 与 runtime 包。

PyG/`pyg-lib` 用于在线 Governance 的固定 fanout `NeighborLoader`，删除会改变真实 checkpoint 推理路径。Torch wheel 安装后只裁剪版本锁定 allowlist 中的编译头、CMake 元数据和链接库；每个平台都必须在裁剪后通过真实 checkpoint forward。

OGB、Pandas、SciPy、scikit-learn、FlagEmbedding、Transformers、pytest、ruff、mypy 和构建工具不进入用户 runtime。研究人员可在独立开发环境安装：

```console
python -m pip install -e "packages/gfm[research,dev]"
```

研究环境不受 runtime 管理，也不能作为发布运行环境复用。

普通用户安装只使用以下两个平台锁，`onboard` 会自动选择，不提供 profile 选项：

```text
packages/gfm/locks/install-windows-x86_64-cpu-pt28.requirements.txt
packages/gfm/locks/install-linux-x86_64-cpu-pt28.requirements.txt
```

`packages/gfm/locks/windows-cpu.requirements.txt` 和 `cpu-ci.requirements.txt` 用于各平台的完整 GFM CI／研究验证，包含用户运行不需要的测试或研究依赖，不会被公开 onboarding 安装。

## Onboarding 与生命周期

```console
python scripts/socialgraph.py onboard
python scripts/socialgraph.py start
python scripts/socialgraph.py stop
python scripts/socialgraph.py doctor
python scripts/socialgraph.py configure-llm
```

`onboard` 自动完成：

1. 验证 Windows/Ubuntu x64 与 64 位 CPython 3.12.x；
2. 按操作系统选择唯一 CPU lock；
3. 使用 `pip --no-compile --no-cache-dir` 在临时目录构建单一环境；
4. 裁剪锁定版本允许删除的 Torch 编译资产与可由同目录源码重建的 `.pyc`；
5. 运行 `pip check`、`NeighborLoader` 和四个 Russia 真实 forward；复用旧环境时执行同等的裁剪后复核；
6. 新环境通过后原子切换到 `var/runtime`，安装并校验模型、数据、Governance 索引、案例、示例和预构建 Web；
7. 完成四 checkpoint forward，通过服务商菜单收集三字段并验证 LLM，写入 runtime profile 后才删除旧环境备份、旧 API/GFM 环境和 `node_modules`。

安装或锁文件变化时不得就地改写当前可用环境。新 generation 验证失败时保留旧环境和用户的案件、模型、数据及私有配置。

受管 Python 命令统一使用 `-B`，避免服务运行时重新生成 bytecode。复用版本和锁哈希均匹配的现有 runtime 时不会重新安装；服务停止后只对 `site-packages` 中可安全重建的 `.pyc` 做幂等裁剪。purelib 本身越界或包含链接时会 fail closed；单个无对应源码、格式异常、链接或 reparse point 候选会被保留并排除在删除清单外。该过程不会触碰模型、配置、案件或旧环境。

如需为本地开发临时覆盖端口，必须在调用 launcher 的进程环境中显式设置 `SOCIALGRAPH_PUBLIC_PORT` 和 `SOCIALGRAPH_GFM_PORT`；项目不会从 `.env` 自动加载这些值。例如：

```powershell
$env:SOCIALGRAPH_PUBLIC_PORT = "15173"
$env:SOCIALGRAPH_GFM_PORT = "18766"
python scripts/socialgraph.py start  # 使用上述临时端口
```

```bash
SOCIALGRAPH_PUBLIC_PORT=15173 SOCIALGRAPH_GFM_PORT=18766 python3 scripts/socialgraph.py start
```

`start` 不安装依赖，并在启动前再次验证 LLM。API/Web 或 GFM 任一进程未就绪时整体启动失败；`stop` 使用绑定 PID、启动时间、可执行文件和命令身份的记录终止受管进程。

## 大模型合同

私有配置只包含：

```text
LLM_API_BASE
LLM_MODEL
LLM_API_KEY
```

服务商菜单只是三字段配置的快捷入口，不会把 `provider`、timeout、鉴权方式或其他高级值写入私有文件：

| 服务商预设 | 预填 Base URL | 官方兼容文档 |
| --- | --- | --- |
| OpenAI 官方 | `https://api.openai.com/v1` | [Chat Completions](https://developers.openai.com/api/reference/resources/chat) |
| DeepSeek 官方 | `https://api.deepseek.com` | [DeepSeek Chat Completions](https://api-docs.deepseek.com/guides/multi_round_chat/) |
| 通义千问 | `https://dashscope.aliyuncs.com/compatible-mode/v1` | [百炼三字段配置](https://help.aliyun.com/zh/model-studio/get-api-key) |
| Gemini OpenAI-compatible | `https://generativelanguage.googleapis.com/v1beta/openai` | [Gemini OpenAI compatibility](https://ai.google.dev/gemini-api/docs/openai) |
| MiniMax 中国 | `https://api.minimaxi.com/v1` | [MiniMax 中国兼容接口](https://platform.minimaxi.com/docs/api-reference/text-chat-openai) |
| MiniMax 国际 | `https://api.minimax.io/v1` | [MiniMax 国际兼容接口](https://platform.minimax.io/docs/api-reference/text-chat-openai) |
| OpenRouter | `https://openrouter.ai/api/v1` | [OpenRouter API](https://openrouter.ai/docs/api/reference/overview) |
| 自定义 OpenAI-compatible | 用户输入 | 由服务提供方说明 |

选择预设后仍可编辑地址，模型 ID 必须从当前账户的服务商控制台复制，API Key 使用隐藏输入。Qwen 业务空间可替换为控制台提供的专属 OpenAI-compatible 地址。新配置没有默认服务商；连续三次输入无效选择或终端 EOF 都会安全取消。非交互自动化继续明确传入 `--api-base`、`--model` 和 `--api-key-stdin`。

配置器接受 API 根地址、版本根地址或完整 `/chat/completions`，规范化后进行真实的最小安全请求。DeepSeek 的根地址、`/v1` 及完整 endpoint 都归一到官方 `/chat/completions`；旧三字段配置在重新验证时自动采用同一规则。验证成功前不保存，`start` 时还会再次验证。

请求协议固定为 OpenAI-compatible Chat Completions，但会根据**精确 hostname**应用最小兼容差异：

- `api.openai.com` 使用 `/v1/chat/completions` 和 `max_completion_tokens: 700`；相似后缀域名不享受该规则；
- `api.deepseek.com` 使用 `max_tokens: 700`，并发送 `thinking: {"type": "disabled"}`，使治理 JSON 保持短而可校验；
- `generativelanguage.googleapis.com` 增加稳定且不含敏感信息的 `x-goog-api-client: socialgraph-fm/1.0.0` 项目标识；
- `api.minimaxi.com` 与 `api.minimax.io` 使用 `max_completion_tokens: 700` 和 `reasoning_split: true`；`MiniMax-M3*` 额外禁用 thinking，其他模型的独立 reasoning 字段不会进入治理正文；
- `openrouter.ai` 按其兼容合同固定使用 `max_tokens: 700`，即使模型 slug 指向 OpenAI 模型；
- 其他兼容地址默认使用 `max_tokens: 700`；模型 ID 最后一段为 `gpt-5*` 或 `o1`–`o9` 时改用 `max_completion_tokens: 700`。

所有服务统一发送 `stream: false`，不发送 `temperature`，让模型采用其服务端默认值。HTTP 固定使用 Bearer 鉴权、15 秒超时和 2 MiB 响应上限；禁止重定向和继承环境代理；远程地址必须是 HTTPS，只有本机回环地址可以使用 HTTP。服务商 hostname 使用严格解析结果匹配，不使用容易被 `api.openai.com.evil.example` 绕过的字符串包含判断。

系统只会剥离位于响应开头且完整闭合的 `<think>...</think>` 前缀，随后仍要求正文通过 JSON/Schema 校验；不完整标签或正文中的标签不会被静默忽略。系统保留一次结构化修复请求；这次修复仍调用同一模型和同一 endpoint，不是备用模型。401/403、404、429、超时、超大响应、非法 JSON 或越界内容都会显式失败，系统不会切换服务商、切换模型、降级协议或生成确定性替代叙述。

公开合同不支持 Responses、Anthropic Messages、Azure 专用 query/header、原生 Gemini 协议、多模型故障转移或供应商专用 SDK。需要这些协议的服务必须另行提供兼容 Chat Completions 的 Base URL。OpenAI 官方密钥从 [API Platform](https://platform.openai.com/api-keys) 创建并独立管理 API 用量；ChatGPT 或 Codex 订阅、登录凭据不能作为本项目的 API Key。

意图理解、构图意图和 Assistant Skills 均要求 LLM。未配置、401/403、404 模型错误、429、超时、非法 JSON 或越界内容会显式失败，不会切换本地规则或生成确定性替代叙述。确定性图算法、输入校验、Skill 结果校验和人工确认仍保留，它们是证据/安全边界，不是 LLM 备选项。

## Skills 与 API

完整映射见 [Skills 索引](../skills/README.md)。

底层 Governance Skills：

```text
GET  /api/v2/gfm/governance/skills
POST /api/v2/gfm/governance/skills/execute
POST /api/v2/gfm/governance/skills/{skill}/execute
POST /api/v2/gfm/governance/skills/confirm
```

LLM Assistant Skills：

```text
GET  /api/v2/gfm/governance/assistant/skills
POST /api/v2/gfm/governance/assistant/execute
```

Assistant 请求使用 `socialgraph-fm.assistant-skill-request/1.0`，明确指定 6 个只读 Skill 之一；响应使用 `socialgraph-fm.assistant-skill-result/1.0`，包含执行 ID、答案、证据结果、底层只读调用 trace、证据引用、引用哈希和审计哈希。旧的 assistant turn/dispatch 组合接口不再属于公开合同。

`run_governance_analysis` 与 `draft_review_report` 仍通过 Governance confirmation ticket 完成模型执行或持久化；Assistant Skills 不得调用这两个写操作。

## 模型、数据与身份

仓库发布 Global、In-domain、Low-label、Cross-domain checkpoint，Russia 01–04/完整输入、zero/few-shot 目标任务、Governance 知识索引和已审结案例。完整训练语料、训练运行、缓存、凭据和本地用户状态不发布。

模型状态、数据内容、图版本、运行结果、目标任务、适配策略、案例 revision、知识块和报告来源均使用稳定哈希绑定。普通 CSV/JSON/GraphML/GEXF 只进入结构分析；只有通过输入合同和来源校验的 Global 推理包可进入模型路径。

普通图结果不得标记为模型预测；潜在线索不得标记为事实边；目标域适配分数保留 Global 校准语义并需要独立验证。

## 运行状态

```text
var/runtime/             唯一受管 Python 环境
var/config/              私有三字段 LLM 配置
var/deploy/pids/         PID、端口和进程身份记录
var/deploy/logs/         setup 与受管进程的脱敏日志
var/gfm/core-runtime/    数据集、服务令牌、运行绑定和模型复核记录
var/gfm/governance/      治理运行、案件、证据、知识与案例索引
var/governance/          用户目标域适配输入
var/models/              展开的 Global 模型资产
var/web/client/          展开的预构建 Web
var/examples/            onboarding 展开的用户示例
```

`var/` 全部被 Git 忽略。Windows 私有配置使用受保护 DACL；POSIX 使用 0600 文件、0700 目录、原子替换和目录 fsync。密钥不得进入异常文本、诊断 JSON、测试 fixture 或环境快照。

## Web 开发

只有修改前端源码的开发者需要 Node/npm：

```console
npm --prefix apps/web ci
npm --prefix apps/web run typecheck
npm --prefix apps/web test -- --run
npm --prefix apps/web run build
npm --prefix apps/web run test:e2e:offline
```

CI 生成确定性的 `bundles/web/client.zip` 及哈希 manifest。Web 源码变化而 bundle 未更新时 publication check 失败。生产构建默认使用同源相对 `/api` URL；本地 Vite 开发可显式设置 API base/proxy。

## 发布校验

Required CI 固定覆盖 repository policy、Web 构建、API/Python、runtime、GFM CPU 和 clean runtime，Windows/Ubuntu 各自从 clean clone 与 GitHub Download ZIP 验证。验收至少包括：

- 单环境中不存在 CUDA、OGB、训练/dev 依赖、Node/npm 或第二个 venv；
- 四类 checkpoint、Russia forward、固定邻居采样、Global smoke 和目标域流程；
- 三项 LLM 配置成功与认证、模型、限流、超时、非法结构失败；
- 6 个 Assistant Skills 与 8 个 Governance Skills 的名称、顺序、权限和调用链一致；
- Web 单测/E2E、API/runtime/GFM 全量测试以及 secret、publication、contract、knowledge、bundle 和 manifest 校验。

`docs/status/readiness.json` 仅表示实验 Core 的研究和 serving gate，不影响 Governance 用户运行版。
