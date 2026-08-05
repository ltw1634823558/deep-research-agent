# 架构决策记录（ADR）

> 记录关键且不易回退的技术选择，供团队对齐与新人 onboarding。格式：背景 → 决策 → 后果。

## ADR-001：编排用 LangGraph StateGraph，而非手写循环
- **背景**：多角色（规划/检索/核验/撰写）之间需要循环与回环，且要可观测、可测试。
- **决策**：用 `StateGraph` 定义节点与条件边；检索循环、补充回环用确定性路由函数实现，迭代上限兜底。
- **后果**：路由逻辑可单测、LangFuse 自动追踪链路；代价是状态 schema 需显式声明，改动节点要同步 state。

## ADR-002：mock-first 设计，所有外部依赖可降级
- **背景**：演示/CI/面试官本地运行都不该依赖 API key 或网络。
- **决策**：`LLM_PROVIDER=mock` 用 `MockChatModel` 跑通整条管线；检索/抓取/Obsidian 在无 key/无路径时返回合成数据。
- **后果**：零门槛可跑、测试稳定；代价是 mock 路由靠关键词匹配，新增节点要同步维护 mock 分支（已在 `mock.py` 注释说明触发词约定）。

## ADR-003：Source 区分来源类型（web / local）
- **背景**：引入 Obsidian 私域知识后，下游读取方式不同——web 来源要 fetch_url，local 来源直接用 snippet。
- **决策**：`Source.source_type` 字段；`researcher_node` 用 `_source_text()` 按类型分发。
- **后果**：扩展新来源（如数据库、飞书）只需新增工具 + 在 researcher 合并，writer 引用逻辑无需改。

## ADR-004：Obsidian 检索默认离线关键词 + 图感知邻居，语义检索可选
- **背景**：用户希望结合 Obsidian 获取"文档联系"，但 embedding 依赖重模型、破坏离线可跑。
- **决策**：默认 TF 关键词打分 + 双链邻居扩展（depth=1）获取关联笔记；`embed_fn` 钩子预留语义检索，默认关闭。
- **后果**：开箱离线可用、双链关系被利用；代价是默认检索精度不如向量库，进阶路线已写明如何接 Chroma+Rerank。

## ADR-005：不引入 yaml 依赖解析 frontmatter
- **背景**：解析 Obsidian frontmatter 最简单是 `import yaml`，但该仓库未装 yaml。
- **决策**：用轻量正则解析 `tags:` 列表与简单 `key: value`，内联标签跳过标题行。
- **后果**：少一个依赖、离线更稳；代价是复杂 frontmatter（嵌套）不被支持——本场景足够。

## ADR-006：质量门禁用 ruff + mypy + pytest
- **背景**：团队要提升工程水平，需统一规范与自动拦截。
- **决策**：`pyproject.toml` 配 ruff/mypy，`.pre-commit-config.yaml` 提交前拦截，CI 跑 `pytest`。
- **后果**：低级错误不进主干；代价是新人需装 dev 依赖，已在 README 说明。

## ADR-007：搜索能力以 MCP 协议接入，而非硬编码 SDK
- **背景**：进阶路线要求 Agent 能调用外部系统（搜索/数据库/飞书/内部 API）；若直接 import 各 SDK，耦合高、难替换。
- **决策**：用 FastMCP 把 Tavily 搜索封装为 MCP 工具（`mcp/server.py`，暴露 `tavily_search`/`tavily_extract`）；Agent 侧 `mcp/client.py` 经 `mcp.client.stdio` 以子进程拉起 Server 并调用，`search.py` 用 `search_provider` 路由（tavily 直连 / mcp）。Server 子进程自带 `load_dotenv()` 读取 `.env`。
- **后果**：搜索后端可无缝替换为任意兼容 MCP 的服务（含官方 `npx -y tavily-mcp`），业务代码零改动；无 key 时自动降级 mock，离线可跑。代价是每次检索需启动子进程（演示/中小规模可接受，高并发可改用常驻 Server + SSE/Streamable HTTP 传输）。

## ADR-008：研究者阶段采用「Chroma 向量召回 + Rerank 重排」的真实 RAG
- **背景**：进阶路线要求把检索升级为生产级 RAG；直接把原始检索结果塞给 LLM 噪声大、易跑题。
- **决策**：候选来源先经 `rag/embeddings.py` 向量化（mock 确定性哈希 / OpenAI Embedding），存入 `rag/store.py` 的 Chroma（ephemeral，不用起服务）做向量召回，再由 `rag/rerank.py` 精排（mock 语义+词面 / Cohere Rerank），只把 top-k 精排结果喂给 LLM。Chroma 不可用时降级内存余弦，保证不中断。
- **后果**：召回质量与可解释性提升（rerank 得分可见），且 mock 默认离线可跑、接真实 key 一键切换；代价是每轮 researcher 临时建库（短期向量库），跨轮持久化知识库需后续接持久化 Chroma/Milvus（进阶路线 #2 已留口）。
- **补充（Chroma 离线行为）**：Chroma 在离线环境会对远端遥测做阻塞式握手。因此 `RAG_BACKEND=auto`（默认）用守护线程 + 超时（6s）探测包裹其初始化：成功走 Chroma，离线/受限则降级内存向量库，保证「零 key 离线可跑、进程正常退出」。可联网环境或显式 `RAG_BACKEND=chroma` 即走真实 Chroma。内存后端与「稠密召回 + 重排」逻辑完全等价。

## ADR-009：评估指标（完成率/引用准确率/幻觉率）作为 LangFuse score 写入
- **背景**：进阶路线要求建立任务完成率、引用准确率、幻觉率等量化评估，支撑回归与对比。
- **决策**：`evaluation.py` 的 `evaluate(state)` 计算三项 [0,1] 启发式代理指标（完成度、引用精度、未接地占比），CLI 末尾打印、API 返回 `metrics` 字段；配置 LangFuse 后以 `score` 写入 `deep-research-eval` trace。
- **后果**：每次研究都有量化反馈，便于在 LangFuse UI 做看板与跨运行对比；代价是指标为启发式代理（尤其 mock 下 LLM 不产出真实引用，数值仅作管线演示），接真实模型后才有真实意义——已在 README §11 注明。

## ADR-010：实时运行状态面板（Dashboard）用 LangGraph 流式 + in-memory JobStore
- **背景**：进阶路线要求「能直观看到 Agent 的运行状态及其其他内容」；用户希望有一个面板实时反映管线阶段、子主题、来源、评估指标与报告。
- **决策**：
  - 新增 `src/jobs.py` 的线程安全 in-memory `JobStore`（演示/单机足够；多副本换 Redis），记录 query/mode/status/阶段进度/子主题/来源/分析/报告/指标/LangFuse 链接/配置快照；阶段顺序 `STAGES` 单一真源，同时驱动进度条与前端步进器。
  - 新增 `POST /research/job`：后台线程用 `graph.stream(stream_mode="updates")` 实时拿节点名做阶段推进，本地累加 `sources/findings`，最终重建 state 调 `evaluate()` 落指标；返回 `job_id` 供前端轮询。
  - 新增 `GET /api/jobs`、`GET /api/jobs/{id}`（轮询）、`GET /` 与 `/dashboard`（返回 `src/templates/dashboard.html` 自包含零依赖面板：暗/亮/系统主题切换 + 玻璃拟态 + 阶段步进器 + 指标仪表，纯前端轮询，离线可用）。
  - **保留** `GET /health` 与 `POST /research`（同步契约不变），并把同步运行也记入 JobStore 做历史，避免破坏已有集成与测试。
- **后果**：开箱即用的实时可观测面板，零外部依赖、可离线演示；阶段与指标单一真源、易扩展。代价是 in-memory 存储重启即丢（演示足够），且流式推进依赖 `stream_mode="updates"` 的节点返回字段约定（每个节点返回全量 `plan`、增量 `sources`/`findings`，已在 agent 节点保持一致）。
- **顺带修复（fetch 守卫）**：dashboard 实测暴露一个既有健壮性 bug——`tools/fetch.py` 仅在「无 Tavily key」时把 `example.com` mock URL 当离线占位返回；若配置了 key 但离线，真实 Tavily 调用失败回退到 `example.com` mock 来源，再进 `fetch_url` 会真的 `requests.get` 该 URL 并挂起（每源 10s 超时）。改为「`example.com` 一律按 mock 处理」，与 key 是否存在解耦，修复了「有 key 离线时 researcher 卡死」的问题。

## ADR-011：长记忆升级为 Chroma 持久化语义召回（SQLite 关键词兜底）
- **背景**：进阶路线 #2 要求把长记忆从 SQLite 关键词召回升级为 Embedding 语义召回（Chroma/Milvus），让研究具备跨任务连贯性、降低重复检索。
- **决策**：`src/memory.py` 新增 `SemanticIndex`（复用 `rag/embeddings.Embedder` 做向量化），后端 `MEMORY_BACKEND=auto`（默认）优先 **Chroma 持久化**（`PersistentClient` 写盘到 `MEMORY_PATH`，跨进程/重启保留），离线/受限自动降级**内存余弦召回**；`MemoryStore.save` 把报告经 `_extract_insights`（确定、无 LLM 依赖，跳过大标题/引用/代码块并去重截断）抽成洞察写入索引（按文本 md5 幂等 upsert），`recall` **语义优先、无命中回退 SQLite 关键词**；researcher 阶段对每个子主题 `recall` 后把历史洞察作为先验知识注入检索提示词。
- **后果**：研究具备跨任务记忆与连贯性，且 mock 默认离线可跑、接 OpenAI Embedding 一键升级真实语义；代价是默认后端在离线环境降级为内存（跨重启不保留），可联网或显式 `chroma` 即持久化——与 RAG 同源的守护线程 + 超时（6s）探测包裹 Chroma 初始化，保证进程正常退出。SQLite 关键词兜底保留了对旧行为的兼容。

## ADR-012：Analyst 真实 LLM 自愈回路（批判-修复循环）
- **背景**：进阶路线 #3 要求用真实 LLM 驱动 analyst 触发补充检索回环，展示自愈式研究；但 mock 模式不应进入昂贵循环，且需保证质量不过关时能自我修正或退回检索。
- **决策**：`src/agents/analyst.py` 仅在 `LLM_PROVIDER=openai` 时进入「自我批判 → 修复」内部循环：每次产出分析后由纯函数 `_critique_analysis` 评估**覆盖度（各发现是否被引用）/置信度（是否含低置信表述）/长度**；不过关则把具体问题回灌模型修订，最多 `ANALYST_SELF_HEAL` 次；仍判缺口则追加补充子主题，触发已有的 `analyst → researcher` 条件回环。`_critique_analysis` 抽为纯函数便于单测；mock 模式保持单轮、行为向后兼容。
- **后果**：真实 LLM 下分析质量有闭环保障（自我修正 + 必要时补检索），面板以「自愈 N 次」徽标展示修复次数；代价是真实 LLM 下每轮分析最多多 `ANALYST_SELF_HEAL` 次调用，需在成本与质量间权衡（默认 2 次）。
