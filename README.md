# Deep Research Multi-Agent（深度研究多智能体）

一个生产级导向的 AI Agent 项目骨架，用于演示 **Agent 开发工程师** 岗位 JD 要求的核心能力：
LangGraph 多智能体编排、联网检索（RAG/Tool Use）、长短期记忆、可观测评估、FastAPI 服务化、Docker 部署。
**无需任何 API key 即可离线跑通**（mock 模式），方便录 demo / 跑 CI。

---

## ⚡ TL;DR（30 秒速览）

> **它是什么**：一个 LangGraph 编排的「深度研究」多智能体 —— 规划 → 检索 → 核验 → 撰写 → 评估，带 RAG 精排、语义长记忆、实时可观测面板。
> **一键跑通**（零 key，离线）：`pip install -r requirements.txt` → `python main.py "你的问题"` → 看终端报告。
> **无本地模型下载**：向量化用项目内置 mock / OpenAI API，Chroma 不再自动下载 `all-MiniLM-L6-v2` 等 ONNX 模型，即装即跑。
> **看面板**：`python -m uvicorn src.server:app --port 8000` → 浏览器开 `http://localhost:8000/`，全程鼠标点。

**核心能力清单**

| 维度 | 能力 | 说明 |
| --- | --- | --- |
| 编排 | LangGraph `StateGraph` 多智能体 | Planner/Researcher/Analyst/Writer，条件边实现检索循环 + 分析自愈回环 |
| 检索 | 联网检索 + 私域知识 | Tavily / MCP Server + Obsidian 双链图；mock 离线可跑 |
| RAG | 向量化 → 召回 → Rerank 重排 | Chroma 持久化，离线自动降级内存 |
| 记忆 | 语义长记忆（跨任务连贯） | Chroma Embedding 语义召回 + SQLite 关键词兜底 |
| 质量 | Analyst 自愈回路 | 真实 LLM 下批判-修复循环，必要时补检索 |
| 评估 | 任务完成率 / 引用准确率 / 幻觉率 | `evaluation.py` 计算并写 LangFuse score |
| 可观测 | 实时 Dashboard 面板 | 阶段步进器 + 指标仪表 + 来源/记忆召回，纯浏览器刷新 |
| 工程 | FastAPI + Docker + 门禁 | ruff / mypy / pytest，mock 模式 CI 稳定 |

**首次体验路径**：① 离线跑 `python main.py`（看三块打印：报告 / 来源 / 评估）→ ② 起 `uvicorn` 开面板点「发起新研究」实时看状态 → ③ 连跑两个相关主题验证「长记忆召回」卡片。

---

## 架构

```
用户问题
  │
  ▼
[Planner]      任务规划：把问题拆成多个可独立检索的子主题
  │
  ▼
[Researcher] ──► 联网检索(Tavily/mock，可经 MCP Server 调用) + Obsidian 私域知识(双链图) + 网页抓取 + 长期记忆召回
              │    ↓ 候选来源经「向量化(Chroma) → 向量召回 → Rerank 重排」精排后喂给 LLM ──┐
  │                                                              │
  │ 还有 pending 子主题？── 是 ──► 回到 Researcher（循环）        │
  ▼                                                              │
[Analyst]      交叉核验、检测矛盾/缺口，必要时追加子主题 ─────────┘
  │
  │ 还有 pending 子主题？── 是 ──► 回到 Researcher（补充检索）
  ▼
[Writer]       综合发现 + 引用，产出 Markdown 报告，落库长期记忆
  │
  ▼
最终报告 + 来源列表
```

- **编排层**：LangGraph `StateGraph`，条件边实现 researcher 循环与 analyst→researcher 回环，迭代上限防死循环。
- **能力层**：`tools/search.py`（检索，按 `search_provider` 路由 Tavily 直连 / MCP Server）、`tools/fetch.py`（抓取）、`mcp/`（MCP Server + Client）、`rag/`（向量化 + Chroma 向量召回 + Rerank 重排）、`memory.py`（长短记忆）。
- **模型层**：`config.get_llm()` 统一抽象，`mock` / `openai` 兼容端点（DeepSeek、通义等）一键切换。
- **可观测**：`observability.py` 接入 LangFuse，自动追踪每个节点链路与 token 消耗。
- **工程层**：`server.py`（FastAPI）、`Dockerfile`（容器化一键起）。

---

## 快速开始

> **环境要求**：Python **≥ 3.10**（`pyproject.toml` 已锁定 `requires-python`）。若本机 `python` 指向旧版本，改用 `python3`（macOS/Linux）或 `py`（Windows）。
> **零配置即可跑**：默认无需任何 API key，LLM / 检索 / 向量化 / 重排 / 长记忆全部走 mock，离线一条龙跑通。

### 1. 离线 mock 模式（无需 key，先跑通再说）
```bash
# ① 建虚拟环境并激活
python -m venv .venv
#   Windows 激活：
.venv\Scripts\activate
#   macOS / Linux 激活：
# source .venv/bin/activate

# ② 装依赖（含 langgraph / fastapi / chromadb / langfuse 等）
pip install -r requirements.txt

# ③ 跑一条研究（默认 mock，约 2 秒出结果）
python main.py "人工智能 Agent 的发展现状与趋势"
```

**你会看到什么**（三块输出）：
```
===== RESEARCH REPORT (mode=web) =====      # 一篇带引用的 Markdown 报告
...
===== SOURCES =====                          # 检索到的来源列表（类型 + 标题 + URL）
- [web] ...
===== EVALUATION (任务完成率/引用准确率/幻觉率) =====
{ "task_completion_rate": 1.0, "citation_accuracy": 1.0, ... }
```

### 2. 接真实模型（傻瓜式填 key 不踩坑）

先把模板复制成自己的配置文件，再用**记事本 / VS Code / 任意纯文本编辑器**打开它：
```bash
cp .env.example .env
# 然后用编辑器打开 .env（Windows 可用：notepad .env  或  code .env）
```

> **🔑 关于「双引号」的灵魂拷问（必看）**
> - `.env` 里一律写成 `KEY=值`，**不要加任何引号、等号两边不要留空格**。
>   - ✅ 正确：`OPENAI_API_KEY=sk-abcd1234`
>   - ❌ 错误：`OPENAI_API_KEY = sk-...`（等号两边有空格，会被当成空值）
>   - ⚠️ `OPENAI_API_KEY="sk-..."` 也能跑——本项目依赖的 `python-dotenv 1.2.2` 会**自动剥掉首尾引号**（已实测验证）。但为避免手滑，**强烈建议干脆不加引号**。
> - `sk-xxx`、`你的key` 这类都是**占位符**，必须整段替换成你真实的 key。留着 `sk-xxx` 或 `你的key` 字面量去跑，会直接报「鉴权失败 / 401」。

下面给两种**可直接抄**的填法，你照着换掉 key 就行。

**场景 A：用 OpenAI 官方**（最常见）
```bash
LLM_PROVIDER=openai
OPENAI_API_KEY=sk-这里换成你的真实key
OPENAI_BASE_URL=https://api.openai.com/v1   # 官方默认，照抄即可
MODEL_NAME=gpt-4o-mini                      # 也可换成 gpt-4o / gpt-4-turbo 等
```
> 注意：`OPENAI_BASE_URL` 是 `OPENAI_BASE_URL`（不是 `OPENAI_API_BASE`），变量名抄错会连不上。

**场景 B：用 DeepSeek / 通义 / 本地 Ollama 等「OpenAI 兼容端点」**
做法一样，只是把 `OPENAI_BASE_URL` 换成厂商地址、`MODEL_NAME` 换成对应模型名，key 仍填到 `OPENAI_API_KEY`：
```bash
LLM_PROVIDER=openai
# DeepSeek 示例：
OPENAI_BASE_URL=https://api.deepseek.com/v1
MODEL_NAME=deepseek-chat
OPENAI_API_KEY=sk-这里换成你的DeepSeek真实key
```
```bash
# 通义千问示例：
OPENAI_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
MODEL_NAME=qwen-plus
OPENAI_API_KEY=sk-这里换成你的通义真实key
```
```bash
# 本地 Ollama 示例（无需 key，本机起服务后填本地地址）：
OPENAI_BASE_URL=http://localhost:11434/v1
MODEL_NAME=llama3
OPENAI_API_KEY=ollama   # Ollama 不校验 key，随便填一个非空字符串即可
```

改完保存，跑：
```bash
python main.py "你的问题" --iterations 3
```
> 想顺便让向量召回 / 重排也用真实模型？把 `.env` 里的 `EMBEDDING_PROVIDER=openai`（并填 `EMBEDDING_API_KEY` / `EMBEDDING_BASE_URL`）和 `RERANK_PROVIDER=cohere`（并填 `COHERE_API_KEY`）打开即可；不填则自动走 mock，不影响主流程。

### 3. 接 LangFuse 可观测（指标自动上分）

「评估指标（任务完成率 / 引用准确率 / 幻觉率）」要可视化，需要先接 LangFuse。key 免费、获取只要 3 步：

1. 打开 **https://cloud.langfuse.com** → 注册 / 登录（GitHub 或邮箱都行）。
2. 新建一个 **Project（项目）**；进项目后点 **Settings → API Keys**（或在项目首页点 **Create API key**）。
3. 你会看到两串 key：
   - `pk-lf-...` 开头的是 **Public Key** → 填到 `LANGFUSE_PUBLIC_KEY=`
   - `sk-lf-...` 开头的是 **Secret Key** → 填到 `LANGFUSE_SECRET_KEY=`
   - `LANGFUSE_HOST` 用默认值 `https://cloud.langfuse.com` 即可（只有自建部署 / 欧盟区才需要改）。

填法（同样**不加引号**）：
```bash
LANGFUSE_PUBLIC_KEY=pk-lf-xxxxxxxxxxxxxxxx
LANGFUSE_SECRET_KEY=sk-lf-xxxxxxxxxxxxxxxx
LANGFUSE_HOST=https://cloud.langfuse.com
```
> 这两个 key 留空 = 关闭可观测，不影响研究运行；但接上后，每次 `python main.py` / 面板发起的研究都会在 LangFuse 项目里出现一条 trace，并自动带三个 score（任务完成率 / 引用准确率 / 幻觉率），可在 LangFuse 网页做看板与跨运行对比。

**怎么确认填对了**：跑完 `python main.py "问题"`，终端末尾会打印 `===== EVALUATION =====` 的 JSON（这就是算出来的指标）；同时打开 LangFuse 项目页，能看到刚生成的 trace 和那三个分数，即代表打通。

### 4. 启动 API 服务 + 实时面板
```bash
python -m uvicorn src.server:app --host 0.0.0.0 --port 8000
```
启动后浏览器打开 **http://localhost:8000/** 即可看到实时运行状态面板（Dashboard）。
**全程浏览器操作、无需 curl**：面板内有「发起新研究」表单，填问题、选模式、点一下就能发起，并实时看到各阶段推进（详见 §12）。

### 5. Docker
```bash
docker build -t deep-research-agent .
docker run -p 8000:8000 --env-file .env deep-research-agent
```

### 6. 测试（全程离线，约 5~8 秒）
```bash
python -m pytest -q                          # 全部用例：mock 管线 / Obsidian / MCP / RAG / 评估 / 面板
python -m pytest tests/test_mock_run.py -q   # 单独验证管线与路由
```

### 7. 接 Obsidian 私域知识库（获取文档联系）

把你的 Obsidian 仓库接进来，让 Agent 在检索时同时利用本地笔记与双链关系图：

```bash
# .env
OBSIDIAN_VAULT_PATH=D:/path/to/your/vault   # 仓库根目录绝对路径，留空则用内置合成仓库
RESEARCH_MODE=hybrid                        # web=仅联网 | local=仅 Obsidian | hybrid=两者混合
```

```bash
# CLI 指定模式
python main.py "Agent 工程化难点" --mode hybrid
# 或只查私域知识
python main.py "可观测性怎么做" --mode local
```

- `tools/obsidian.py` 会解析每篇笔记的 **frontmatter 标签、内联标签、双链(wikilinks)**，并构建笔记关系图；
- 检索命中后自动**扩展双链邻居（depth=1）**，把关联笔记一并带入上下文——这就是「文档联系」；
- 未配置仓库路径时自动降级为内置合成仓库，离线也能演示 `hybrid`/`local` 模式。

### 7. 团队工程门禁（CI / 代码质量）

本项目内置 senior 级质量门禁，团队成员 clone 后一键对齐规范：

```bash
pip install -r requirements-dev.txt
python -m ruff check src tests     # lint + 格式化检查
python -m ruff format src tests    # 自动格式化
python -m mypy src                 # 静态类型检查
python -m pytest -q                # 测试
pre-commit install                 # 提交前自动拦截低级问题
```

- `pyproject.toml`：ruff（E/F/I/UP/SIM/B）+ mypy 配置，团队统一；
- `.pre-commit-config.yaml`：提交前自动 ruff/格式/尾随空白/EOF 校验；
- `docs/CODE_REVIEW_CHECKLIST.md`：PR 合并前的 senior 审查清单（正确性/可观测/类型/依赖/安全/可维护）；
- `docs/ARCHITECTURE_DECISIONS.md`：关键架构决策记录（ADR），新人 onboarding 直接看。

---

### 8. MCP Server：用 MCP 协议接入 Tavily 搜索

进阶路线里的「接入 MCP Server」已落地。我们把 Tavily 搜索封装成一个标准 MCP 服务，
Agent 不再直接 import 第三方 SDK，而是通过 MCP 协议（stdio 拉起子进程）消费搜索能力——
搜索后端可被任意兼容 MCP 的服务无缝替换（官方 `npx -y tavily-mcp`、内部 API、数据库等）。

**获取 Tavily Key 并接入**
1. 前往 Tavily 官网 https://tavily.com 注册账号；
2. 在 https://app.tavily.com 的 **API Keys** 页面创建 Key；
3. 把 Key 填入 `.env` 的 `TAVILY_API_KEY`，并把检索后端切到 MCP：

```bash
# .env
TAVILY_API_KEY=你的key          # 来自 https://app.tavily.com
SEARCH_PROVIDER=mcp             # 让检索经 MCP Server 调用（默认 tavily 直连也保留）
```

**两种使用方式**
- 作为 Agent 的检索后端（默认）：`SEARCH_PROVIDER=mcp` 时，researcher 经 `src/mcp/client.py`
  以 stdio 拉起 `src/mcp/server.py` 并调用其 `tavily_search` 工具；
- 独立 MCP 服务，可被任意 MCP Client 连接（Claude Desktop、Cursor 等）：

```bash
python -m src.mcp.server        # stdio 模式启动，暴露 tavily_search / tavily_extract 两个工具
```

**替换为官方 Tavily MCP（需本机有 Node.js）**：编辑 `.env`
```bash
MCP_SERVER_COMMAND=npx -y tavily-mcp
```
本项目 Agent 会通过 stdio 连接该官方服务，业务代码（researcher）零改动。

- `src/mcp/server.py`：FastMCP 服务，`@mcp.tool` 暴露 `tavily_search`（联网检索）、`tavily_extract`（正文提取）。
  **行为**：未配置 `TAVILY_API_KEY` 时降级为 mock（离线可跑）；**已配置 key 则必须真实联网，失败直接抛异常，不再静默返回 mock**。
- `src/mcp/client.py`：经 `mcp.client.stdio` 拉起 Server 子进程并调用工具，结果转成 `Source` 列表；失败直接向上抛异常，确保问题可见。

### 9. 模块说明（每个模块的作用与使用）

> 顶层入口
- `main.py`：CLI 入口。解析参数（`query` / `--iterations` / `--mode`），构建图并运行，打印报告与来源列表。
- `src/server.py`：FastAPI 服务。提供 `GET /health`、`POST /research`（含 `mode`，向后兼容）；新增 `POST /research/job`（异步后台研究，返回 `job_id`）、`GET /api/jobs` 与 `GET /api/jobs/{id}`（实时状态轮询）、`GET /` 与 `/dashboard`（实时运行状态面板）。
- `src/jobs.py`：运行状态注册表 `JobStore`（线程安全 in-memory）。记录每次研究的 query/mode/阶段进度/子主题/来源/分析/报告/评估指标/LangFuse 链接/配置快照，供 dashboard 展示。
- `src/templates/dashboard.html`：零依赖自包含面板（内联 CSS/JS，暗/亮/系统主题切换），轮询 `/api/jobs/{id}` 实时刷新。
- `Dockerfile`：容器化构建，配合 `.env` 一键部署。

> 编排与配置
- `src/graph.py`：LangGraph 编排。定义 `Planner→Researcher(循环)→Analyst(可回环)→Writer` 状态机与条件路由；迭代上限防死循环。
- `src/config.py`：全局配置。从环境变量加载 LLM 后端、`search_provider`、Obsidian 路径/模式、LangFuse、流程参数；`get_llm()` 统一返回模型实例。
- `src/state.py`：共享状态与数据模型。`ResearchState`（节点间传递）+ `Subtopic`/`Source`/`Finding` 业务模型（`Source.source_type` 区分 web/local）。
- `src/mock.py`：离线 mock LLM（`MockChatModel`）。无 key 跑通整条管线；按提示关键词路由各节点的 mock 输出。

> 智能体节点（`src/agents/`）
- `planner.py`：规划节点，把问题拆成多个可独立检索的子主题（`Subtopic`）。
- `researcher.py`：检索节点，按 `mode` 聚合 web（经 `search()`）/ Obsidian / 长记忆，产出带引用的摘要 `Finding`。
- `analyst.py`：核验节点，交叉验证、检测矛盾/缺口，必要时追加子主题触发补充回环。
- `writer.py`：撰写节点，综合发现与引用，产出 Markdown 报告并落库长期记忆。

> 工具层（`src/tools/`、`src/mcp/`、`src/rag/`）
- `tools/search.py`：检索统一入口 `search()`，按 `settings.search_provider` 路由到 Tavily 直连（`web_search`）或 MCP Server（`search_via_mcp`）。
- `tools/fetch.py`：网页抓取 `fetch_url()`，取 URL 正文供研究节点引用。
- `tools/obsidian.py`：Obsidian 私域知识库检索。解析 frontmatter/标签/双链，构建笔记关系图，关键词 + 双链邻居（图感知）召回。
- `mcp/server.py`：MCP Server，把 Tavily 搜索/提取封装为 MCP 工具。
- `mcp/client.py`：MCP Client，经 stdio 拉起 Server 并调用工具，结果转 `Source` 列表。
- `rag/embeddings.py`：文本向量化。`Embedder` 支持 mock 确定性哈希向量（离线）与 OpenAI 兼容 Embedding（真实）；
  并提供 `ChromaEmbeddingFunction` 阻止 Chroma 自动下载默认 ONNX 模型。
- `rag/store.py`：`ChromaRAGStore` 用 Chroma（ephemeral）做向量召回，不可用时降级内存余弦；`RAGDoc` 为入库文档模型。
- `rag/rerank.py`：`Reranker` 对召回候选做精排（mock 混合打分 / Cohere Rerank），返回带得分的 `RankedDoc`。

> 支撑能力
- `src/memory.py`：短期（messages 窗口）+ 长期（**语义召回层**）记忆。长期记忆用 `SemanticIndex`（Chroma 持久化 + 离线降级内存，复用 `rag/embeddings.Embedder` 与离线防御式初始化）做 Embedding 召回，SQLite 关键词召回兜底；`MemoryStore.save` 抽取报告洞察落库、`recall` 语义优先，跨研究复用历史、降低重复检索。
- `src/observability.py`：LangFuse 可观测，自动追踪各节点链路与 token 消耗（配置 key 后生效）；`get_langfuse_client()` 供评估模块写 score。
- `src/evaluation.py`：研究质量评估 `evaluate(state)`，计算任务完成率 / 引用准确率 / 幻觉率（启发式代理指标），并写入 LangFuse trace。

> 工程与文档
- `pyproject.toml`：ruff + mypy 团队统一配置。
- `.pre-commit-config.yaml`：提交前自动 lint/格式校验。
- `docs/CODE_REVIEW_CHECKLIST.md`：PR 合并前 senior 审查清单。
- `docs/ARCHITECTURE_DECISIONS.md`：关键架构决策记录（ADR）。
- `tests/`：mock 管线、Obsidian 检索/双链、MCP 工具与 stdio 往返、RAG 向量召回/重排、评估指标测试。

---

### 10. 真实 RAG：向量库（Chroma）+ 重排（Rerank）

进阶路线里的「检索升级为向量库 + Rerank」已落地。研究者阶段不再直接把原始检索结果塞给 LLM，
而是先对候选来源做 **稠密向量召回 + 精排**，只把最相关的内容送进生成环节——这就是生产级 RAG 的标准做法。

**召回 + 重排链路**（`src/rag/`）
1. `embeddings.py` 把每条候选来源（标题 + snippet + 抓取正文）向量化：
   - `mock`：确定性哈希向量，**离线可用、无需 key**；
   - `openai`：OpenAI 兼容 Embedding（DeepSeek/通义等端点），失败自动降级 mock。
2. `store.py` 用 **Chroma**（进程内 ephemeral，不用起服务）做向量召回，取 top-k 候选；
   Chroma 不可用时自动降级为内存余弦召回，保证管线不中断。
3. `rerank.py` 对候选做**精排**：
   - `mock`：语义余弦(0.6) + 词面重叠(0.4) 混合打分；
   - `cohere`：Cohere Rerank API（`COHERE_API_KEY`），失败降级 mock。
4. 重排后的精排上下文连同「rerank 得分」一起喂给研究智能体。

**开启 / 配置**（`.env`）
```bash
EMBEDDING_PROVIDER=mock        # mock=离线哈希向量 | openai=真实 Embedding
EMBEDDING_API_KEY=sk-xxx       # openai 模式填 key（或兼容端点 base_url）
EMBEDDING_MODEL=text-embedding-3-small
RERANK_PROVIDER=mock           # mock=离线混合打分 | cohere=Cohere Rerank
COHERE_API_KEY=
RAG_TOP_K=4                    # 每轮召回+重排后返回条数
RAG_BACKEND=auto               # auto=优先 Chroma，离线自动降级内存 | chroma=强制 | memory=纯内存
```
默认即 `mock`，**无需任何 key 就能跑通向量召回 + 重排**；切 `openai`/`cohere` 即为真实语义能力。

> **关于 Chroma 后端**：`RAG_BACKEND=auto`（默认）会优先使用 Chroma 做向量召回；
> 但 Chroma 在离线环境会尝试连接远端遥测而阻塞，因此本项目用守护线程 + 超时探测包裹其初始化，
> 离线/网络受限时**自动降级为内存向量召回**，保证「零 key 离线可跑、进程正常退出」。
> 可联网环境（或显式 `RAG_BACKEND=chroma`）即走真实 Chroma。内存后端与「稠密召回 + 重排」逻辑完全等价。

### 11. 评估指标（任务完成率 / 引用准确率 / 幻觉率）

进阶路线里的「LangFuse 评估指标」已落地。每次研究结束，`src/evaluation.py` 自动算出三项 [0,1] 指标，
配置 LangFuse 后以 **score** 写入 trace，便于在 LangFuse UI 做看板、跨运行对比与回归监控：

| 指标 | 含义 | 计算（启发式代理） |
|---|---|---|
| `task_completion_rate` | 任务完成度 | 子主题全部完成(0/1) 与 报告达最低长度(0/1) 的平均 |
| `citation_accuracy` | 引用准确率 | 报告显式引用的 URL 中，命中检索来源集合的比例（精度） |
| `hallucination_rate` | 幻觉率 | 未被报告引用的检索来源占比（「未接地」代理指标） |
| `citation_recall` | 引用召回率（附带） | 检索来源中被报告提及的比例 |

> 说明：mock 模式下 LLM 不产出真实引用，指标仅作管线演示；接真实模型后即为真实评估值。

**查看方式**
```bash
# CLI：每次运行末尾自动打印 EVALUATION JSON
python main.py "你的问题" --mode hybrid

# API：POST /research 的返回体新增 metrics 字段
curl -X POST http://localhost:8000/research -H "Content-Type: application/json" \
  -d '{"query":"你的问题","mode":"web"}'
# -> { ..., "metrics": { "task_completion_rate": 1.0, "citation_accuracy": 1.0, ... } }
```
接入 LangFuse（填 `LANGFUSE_PUBLIC_KEY` / `LANGFUSE_SECRET_KEY`）后，上述指标会自动出现在 LangFuse 的 trace 评分中。

### 12. 实时运行状态面板（Dashboard）

进阶路线里的「能直观看到 Agent 的运行状态及其其他内容」已落地。项目内置一个**零依赖、可离线**的 Web 面板，
直观展示 Agent 的实时运行状态与研究切面，方便演示、排查与对外展示。

**启动后直接访问**（无需额外构建、无 CDN 依赖，纯前端自包含）：
```bash
python -m uvicorn src.server:app --host 0.0.0.0 --port 8000
# 浏览器打开：http://localhost:8000/   （或 /dashboard）
```

**纯浏览器操作流程（推荐，无需 curl）**
1. 打开 `http://localhost:8000/`；
2. 在顶部「发起新研究」卡片填写**问题**、选择 `mode`（web / local / hybrid），点「开始研究」；
3. 页面自动跳到该任务并**实时**推进：阶段步进器（规划→检索→核验→撰写→评估→完成）逐节点点亮、进度条走动、子主题与来源逐步填充；
4. 跑完后右侧出现三项**评估指标仪表**；左侧「历史任务」列表可随时点回任意一次研究查看详情；
5. 想看跨研究记忆效果：连跑两个相关主题，第二次的「长记忆召回」卡片会显示第一轮沉淀的洞察。

> API 用法（需要 curl / 程序调用时）：见下方「两种使用方式」。

**面板能看到什么**
- **实时运行状态**：阶段步进器（规划 → 检索 → 核验 → 撰写 → 评估 → 完成）+ 进度条 + 状态徽标，后台用 LangGraph 流式（`stream_mode="updates"`）逐节点推进；
- **子主题规划**：每个子主题的完成/待办状态；
- **检索来源**：web / local 来源列表（标题、URL、摘要、类型标签）；
- **评估指标**：任务完成率 / 引用准确率 / 幻觉率三项仪表（运行结束生成）；
- **分析 & 报告**：Analyst 的交叉核验结论与 Writer 的最终 Markdown 报告；
- **运行配置快照 + LangFuse 链接**：当前 LLM / 检索 / Embedding / Rerank / RAG 后端等配置，以及（已接入时）跳转 LangFuse trace。

**两种使用方式**
- **同步（向后兼容）**：`POST /research` 行为不变，仍同步返回 `report/analysis/sources/metrics`，并自动记入面板历史；
- **异步实时追踪（面板主力）**：`POST /research/job` 立即返回 `job_id`，前端轮询 `GET /api/jobs/{job_id}` 实时刷新，直到 `done`。

```bash
# 异步发起并拿到 job_id
curl -X POST http://localhost:8000/research/job \
  -H "Content-Type: application/json" -d '{"query":"你的问题","mode":"web"}'
# -> {"job_id":"723f...", "dashboard_url":"/dashboard?job=723f..."}

# 查看实时状态
curl http://localhost:8000/api/jobs/723f...
# -> {"status":"running", "progress":40, "stage_index":2, "subtopics":[...], ...}
```

> **实现要点（senior 视角）**：面板为纯前端自包含 HTML（内联 CSS/JS、暗/亮/系统主题切换、玻璃拟态），无 CDN / 构建依赖，完全离线可跑；
> 后端 `JobStore` 为 in-memory 单例（演示/单机足够，多副本换 Redis），阶段顺序 `STAGES` 单一真源同时驱动进度条与步进器。
> 面板实测还暴露并修复了一个既有健壮性 bug：`tools/fetch.py` 原先仅在「无 Tavily key」时把 `example.com` mock 来源当离线占位，
> 导致「配置了 key 但离线」时 researcher 会真的去 `requests.get` 该 URL 而卡死（每源 10s 超时）；已改为 `example.com` 一律按 mock 处理。

---

### 13. 语义长记忆 + Analyst 真实 LLM 自愈回路

进阶路线里的「长记忆从 SQLite 升级为语义召回」与「用真实 LLM 驱动 analyst 自愈式研究」两项已落地。

#### 13.1 语义长记忆（跨研究持久化）

每次研究结束，`src/memory.py` 把报告抽取为若干**洞察条目**写入 `SemanticIndex`：
- 向量化复用 `rag/embeddings.Embedder`（mock 离线哈希 / OpenAI 兼容 Embedding）；
- 索引后端 `MEMORY_BACKEND=auto`（默认）优先 **Chroma 持久化**（写盘到 `MEMORY_PATH`，跨进程/重启保留），
  离线/网络受限时自动降级**内存余弦召回**，保证「零 key 离线可跑」；
- 单条洞察按文本 md5 幂等 upsert，重复研究不产生冗余；
- **recall 语义优先**，无命中时回退 SQLite 关键词召回（向后兼容）。

后续研究的 `researcher` 阶段，对每个子主题先做**语义召回**，把历史洞察作为「先验知识」注入检索提示词——减少重复检索、让研究具备跨任务连贯性。面板「长记忆召回」卡片可直观看到召回条数、后端与具体内容。

#### 13.2 Analyst 真实 LLM 自愈回路

`src/agents/analyst.py` 在**真实 LLM（`LLM_PROVIDER=openai`）**下进入「自我批判 → 修复」内部循环：
1. 产出分析后由纯函数 `_critique_analysis` 评估 **覆盖度 / 置信度 / 长度**；
2. 不过关则把具体问题回灌模型修订，最多尝试 `ANALYST_SELF_HEAL` 次；
3. 仍判缺口（或批判未通过）且未超迭代上限，则追加补充子主题，触发已有的 `analyst → researcher` 回环补检索。

mock 模式**不进入循环**（确定性、零成本），仅做单轮核验，行为向后兼容；面板「自愈 N 次」徽标展示真实 LLM 下的修复次数。

**配置（`.env`）**
```bash
# 长记忆
MEMORY_ENABLED=true
MEMORY_BACKEND=auto          # auto=优先 Chroma 持久化，离线降级内存 | chroma=强制 | memory=纯内存
MEMORY_PATH=.memory_store    # 持久化目录
MEMORY_DB_PATH=memory.db     # SQLite 关键词兜底
MEMORY_TOP_K=3

# Analyst 自愈（仅 openai 生效）
ANALYST_SELF_HEAL=2          # 自我批判-修复最大尝试次数
ANALYST_CRITIC=heuristic     # heuristic=确定性启发式 | llm=额外 LLM 批判
```

> **关于 Chroma 持久化后端**：与 RAG 同理，`MEMORY_BACKEND=auto` 优先用 Chroma 持久化，
> 但 Chroma 离线会尝试远端遥测而阻塞，故用守护线程 + 超时探测包裹初始化，离线自动降级内存索引，
> 保证进程正常退出。可联网环境（或显式 `MEMORY_BACKEND=chroma`）即走真实持久化，历史记忆跨重启保留。

---

### 14. 常见问题 / 排错

**Q：控制台出现 `in-memory` / `memory` 后端提示，正常吗？**
正常。Chroma 在离线/网络受限时会尝试连接远端遥测，本项目用「守护线程 + 超时探测」包裹其初始化，
不可达时**自动降级为内存向量召回**，既保证「零 key 离线可跑」，也保证进程正常退出、不卡死。
可联网环境（或显式 `RAG_BACKEND=chroma` / `MEMORY_BACKEND=chroma`）即走真实 Chroma 持久化。

**Q：启动时控制台在下载 `all-MiniLM-L6-v2/onnx.tar.gz`，需要等吗？**
**不需要等，且最新代码已阻止这个下载。** 这是 Chroma 默认 embedding function 在后台去 HuggingFace 拖 ONNX 模型。
本项目已改为显式传入自定义 `ChromaEmbeddingFunction`，复用 `rag/embeddings.Embedder`（mock 离线哈希 / OpenAI API），
因此：
- 新装的仓库不会再触发该下载；
- 如果你之前已经有一个用默认 embedding function 创建的 `.memory_store/long_term_memory`，
  首次启动时会检测并自动删除重建，期间可能短暂读到旧下载残留，直接忽略即可；
- 想立刻清理缓存：删掉 `C:\Users\<用户名>\.cache\chroma\onnx_models` 目录。

**Q：我配了 `TAVILY_API_KEY` 但面板还是 mock 来源？**
从本轮修复起，**只要配置了 `TAVILY_API_KEY`，系统就会强制走真实 Tavily 联网，失败直接报错，不再静默 mock**。看到 mock 只有两种可能：
1. **key 无效/过期**：去 https://app.tavily.com 重新生成一枚有效 Key；
2. **SEARCH_PROVIDER 选错或没配 key**：检查 `.env` 里 `TAVILY_API_KEY` 是否真填了，且前后无空格/无引号。

若你确实想离线跑（纯 mock 演示），把 `TAVILY_API_KEY=` 留空即可。

**Q：我配了 `TAVILY_API_KEY` 但没联网，会卡住吗？**
不会。`tools/fetch.py` 对 `example.com` 这类 mock 来源一律按离线占位处理，
离线回退路径不会真的去 `requests.get` 外网，researcher 不会卡死。

**Q：端口 8000 被占用了？**
换端口：`python -m uvicorn src.server:app --port 8001`，面板访问对应端口即可。

**Q：想看「真实」语义 / 自愈能力，而不是 mock？**
- 向量化 / 重排：`.env` 设 `EMBEDDING_PROVIDER=openai`（+ key / base_url）、`RERANK_PROVIDER=cohere`（+ `COHERE_API_KEY`）；
- 长记忆：真实语义召回需 `MEMORY_BACKEND=chroma` 或联网（默认 `auto` 离线降为内存）；
- 自愈回路：`.env` 设 `LLM_PROVIDER=openai` 才会进入「自我批判-修复」循环，mock 模式不进入（向后兼容）。

**Q：工作目录里多了 `memory.db` / `.memory_store` 是什么？**
长记忆落库文件（SQLite 关键词兜底 + Chroma 向量持久化）。可随时删除，下次运行会重建；
也可用 `.env` 的 `MEMORY_DB_PATH` / `MEMORY_PATH` 改到别处。

**Q：面板里看不到刚发起的任务？**
- 面板默认每 3 秒自动轮询刷新；
- 若从 `curl` 拿到 `job_id`，把浏览器地址改为 `http://localhost:8000/dashboard?job=<job_id>` 即可自动定位；
- 左侧「历史任务」列表点一下也能切换。

---

## 对照 Agent 开发岗 JD 的能力覆盖

| JD 高频要求 | 本项目落点 |
|---|---|
| Python + 后端工程 | `server.py` FastAPI、`main.py` CLI |
| LangGraph / LangChain | `graph.py` 状态机 + 条件路由 |
| RAG / 检索增强 | `rag/`（Chroma 向量召回 + Rerank 重排）、`tools/search.py` + `tools/fetch.py` |
| 私域知识库 / Obsidian | `tools/obsidian.py`：解析 frontmatter/标签/双链，图感知召回，混合检索 |
| Tool Use / Function Calling | 检索、抓取、Obsidian 工具封装，预留 MCP 接入点 |
| MCP 协议接入 | `mcp/server.py` 用 FastMCP 将 Tavily 搜索封装为 MCP 工具，Agent 经 `mcp/client.py`（stdio）调用，可替换为任意兼容 MCP 的服务 |
| Multi-Agent / Workflow | Planner/Researcher/Analyst/Writer 四角色编排 |
| Memory 记忆 | `memory.py` 短期(messages 窗口)+长期(**Chroma 语义召回**，离线降级内存，SQLite 关键词兜底) |
| Prompt 工程 | 各节点结构化 prompt |
| 可观测 / 评估 | `observability.py` LangFuse 链路追踪 + `evaluation.py` 任务完成率/引用准确率/幻觉率指标 |
| 实时可视化面板 | `server.py` + `jobs.py` + `templates/dashboard.html`：实时运行状态/阶段步进/来源/指标/报告面板（零依赖、可离线） |
| 工程门禁 / 代码质量 | `pyproject.toml`(ruff+mypy) + `.pre-commit-config.yaml` + `pytest` |
| 部署运维 | `Dockerfile` + FastAPI |

---

## 简历可写的 bullet（直接用 JD 原词）

> - 基于 **LangGraph** 设计并实现多智能体深度研究管线（规划→检索→核验→撰写），
>   通过条件边实现检索循环与补充回环，迭代上限保障可控收敛；
> - 构建 **RAG/Tool Use** 能力，封装联网检索、网页抓取与 **Obsidian 私域知识库**工具，
>   支持 web/local/**hybrid** 混合检索；
> - 以 **MCP 协议** 接入搜索能力：用 FastMCP 将 Tavily 搜索封装为 MCP 工具（`mcp/server.py`），
>   Agent 通过 `mcp/client.py`（stdio 拉起子进程）消费，可无缝替换为任意兼容 MCP 的搜索/API 服务；
> - 实现 Obsidian **双链关系图**的图感知召回，将关联笔记纳入上下文（文档联系）；
> - 结合长短期 **Memory** 复用历史研究、降低重复检索：以 Chroma 持久化构建**语义长记忆**，每次研究把报告洞察做 Embedding 入向量库，后续研究在检索阶段做语义召回并注入先验知识，离线自动降级内存索引；
> - 在 **Analyst** 节点实现**真实 LLM 自愈回路**：分析产出后经「自我批判-修复」循环修订，质量不过关则追加子主题触发补充检索回环，提升研究鲁棒性与可信度；
> - 落地真实 **RAG** 召回链路：候选来源经 `rag/`（Chroma 向量召回 + Rerank 重排）精排后再生成，
>   支持 mock 离线向量与 OpenAI Embedding / Cohere Rerank 真实语义能力一键切换；
> - 接入 **LangFuse** 全链路可观测，并以 **score** 写入任务完成率 / 引用准确率 / 幻觉率等评估指标，支撑回归对比；
> - 使用 **FastAPI** 服务化并以 **Docker** 容器化部署，支持 mock 模式离线演示与 CI；
> - 构建**零依赖可离线**的实时运行状态面板（暗/亮主题切换 + 阶段步进器 + 指标仪表），基于 LangGraph 流式（`stream_mode="updates"`）逐节点推送 Agent 运行状态、子主题、来源与评估指标，提升可观测性与演示效果；
> - 建立团队 **ruff + mypy + pytest + pre-commit** 工程门禁与 PR 审查清单，统一代码质量。

---

## 进阶路线（面试加分项）

1. ✅ 已落地真实 **RAG**（Chroma 向量召回 + Rerank 重排）：研究者阶段先向量召回再精排生成（见 §10）。
2. ✅ 已落地**语义长记忆**：长记忆升级为 Chroma 持久化 Embedding 语义召回（离线降级内存、SQLite 关键词兜底），researcher 阶段语义召回注入先验知识（见 §13.1）。
3. ✅ 已落地 **Analyst 真实 LLM 自愈回路**：分析层「自我批判-修复」循环，质量不过关追加子主题触发补充检索回环（mock 模式不进入循环，向后兼容）（见 §13.2）。
4. ✅ 已接入 MCP Server（Tavily 搜索工具，FastMCP + stdio），Agent 经 MCP 协议调用，可替换为数据库 / 飞书 / 内部 API 等任意 MCP 服务；下一步可新增更多 MCP 工具。
5. ✅ 已在 LangFuse 上建立任务完成率、引用准确率、幻觉率等评估指标（见 §11）。
6. ✅ 已新增**实时运行状态面板（Dashboard）**：零依赖可离线，基于 LangGraph 流式推送 Agent 运行状态/子主题/来源/指标/报告，并修复了「有 Tavily key 但离线时 researcher 卡死」的健壮性 bug（见 §12）。
