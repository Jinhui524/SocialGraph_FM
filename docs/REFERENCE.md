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

公开运行版只有一个受管 Python 3.12 环境 `var/runtime`，但 API/Web 与 GFM 仍是两个隔离进程。API Key 仅加入 API 子进程环境；浏览器、GFM、命令输出和日志均不得获得密钥。

API 在所有 `/api/` 路由之后挂载预构建静态站点和 SPA fallback。普通用户不会启动 Vite，也不会创建 `node_modules`。

## 平台与依赖

| 平台 | 执行设备 | 发布状态 |
| --- | --- | --- |
| Windows x64 | CPU | 必测 |
| Ubuntu glibc x64 | CPU | 必测 |

公开运行版不提供 CUDA、MPS、ROCm、macOS、musl 或其他架构 profile，也不做运行设备自动选择。

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

## Onboarding 与生命周期

```console
python scripts/socialgraph.py onboard
python scripts/socialgraph.py start
python scripts/socialgraph.py stop
python scripts/socialgraph.py doctor
python scripts/socialgraph.py configure-llm
```

`onboard` 自动完成：

1. 验证 Windows/Ubuntu x64 与 Python 3.12；
2. 按操作系统选择唯一 CPU lock；
3. 在临时目录构建单一环境并运行 import/doctor/smoke；
4. 验证模型、数据、Governance 索引和预构建 Web manifest；
5. 验证三项 LLM 配置；
6. 原子切换到 `var/runtime`，成功后清理旧 API/GFM 环境和 `node_modules`。

安装或锁文件变化时不得就地改写当前可用环境。新 generation 验证失败时保留旧环境和用户的案件、模型、数据及私有配置。

`start` 不安装依赖，并在启动前再次验证 LLM。API/Web 或 GFM 任一进程未就绪时整体启动失败；`stop` 使用绑定 PID、启动时间、可执行文件和命令身份的记录终止受管进程。

## 大模型合同

私有配置只包含：

```text
LLM_API_BASE
LLM_MODEL
LLM_API_KEY
```

固定行为：

- OpenAI-compatible `/chat/completions`；
- Bearer 鉴权；
- 远程 HTTPS，本机回环可 HTTP；
- 15 秒超时、temperature 0、最多 700 tokens；
- 禁止重定向和继承环境代理；
- 响应上限、JSON/Schema 校验和一次结构化修复请求。

配置器接受 API 根地址、`/v1` 或完整 `/chat/completions`，规范化后进行真实的最小安全请求。验证成功前不保存。旧 Chat Completions + Bearer 配置可迁移三字段并重新验证；Responses、Anthropic 或其他协议必须重新填写兼容 API。

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
var/run/                 PID、端口、令牌和生命周期记录
var/logs/                脱敏日志
var/state/               图谱、案件、复核和会话状态
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
