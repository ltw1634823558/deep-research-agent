"""全局配置：pydantic-settings 驱动，从环境变量加载，兼容 OpenAI / 兼容端点 / 离线 mock 模式。

设计要点（相对旧版 @dataclass 的迁移收益）：
- 类型安全：所有字段带类型注解，pydantic 在建表期即校验（字符串/整型/布尔混用直接报错）。
- 健壮的空值兜底：`field_validator` 把 env 里的空串 / 非法值回落到默认值，
  不再出现 `RAG_TOP_K=` 空值导致 `int("")` 抛 ValueError、整包无法 import 的事故。
- 依赖注入就绪：`settings` 仍保留为进程级全局单例（兼容 `from .config import settings` 与
  测试里的 `monkeypatch.setattr(settings, ...)`），同时新增 `resolve_settings(config)`
  （节点优先读取 `RunnableConfig['configurable']['settings']`，支持按请求覆盖）与
  `get_settings()`（FastAPI `Depends` 依赖），满足多租户 / 按请求覆写的可扩展性。
- 运行时读 .env 的语义保持不变：本模块 `env_file=None`，由入口（main.py / server.py）的
  `load_dotenv(override=True)` 把 .env 注入 os.environ 后，Settings() 从 os.environ 取值；
  测试环境不调用 load_dotenv，因此天然离线、零网络依赖。
"""

from __future__ import annotations

import hashlib
import threading
from collections import OrderedDict
from typing import Any, Optional

from pydantic import field_validator, ValidationInfo
from pydantic_settings import BaseSettings, SettingsConfigDict

from .mock import MockChatModel


# —— 数值 / 布尔字段的空值兜底（替代旧版 _int_env / _float_env 手工解析） ——
_INT_DEFAULTS: dict[str, int] = {
    "rag_top_k": 4,
    "memory_top_k": 3,
    "analyst_self_heal": 2,
    "max_research_iterations": 3,
    "research_window": 10,
}
_FLOAT_DEFAULTS: dict[str, float] = {"temperature": 0.2}


def _coerce_to_int(v: Any, default: int) -> int:
    """空串 / None / 非法字符串 → 默认；布尔显式拦掉（避免 True→1 的意外）。"""
    if v is None or isinstance(v, bool):
        return default
    if isinstance(v, (int, float)):
        return int(v)
    if isinstance(v, str):
        s = v.strip()
        if s == "":
            return default
        try:
            return int(s)
        except ValueError:
            return default
    return default


def _coerce_to_float(v: Any, default: float) -> float:
    if v is None or isinstance(v, bool):
        return default
    if isinstance(v, (int, float)):
        return float(v)
    if isinstance(v, str):
        s = v.strip()
        if s == "":
            return default
        try:
            return float(s)
        except ValueError:
            return default
    return default


class Settings(BaseSettings):
    # 关闭自动读 .env：运行时由 load_dotenv() 注入 os.environ，测试不加载则天然离线。
    model_config = SettingsConfigDict(
        env_file=None,
        extra="ignore",
        case_sensitive=False,
    )

    llm_provider: str = "mock"  # mock | openai
    openai_api_key: str = ""
    openai_base_url: str = "https://api.openai.com/v1"
    model_name: str = "gpt-4o-mini"
    temperature: float = 0.2

    tavily_api_key: str = ""

    # 检索后端：tavily（直连，向后兼容）| mcp（经 MCP Server 调用 Tavily 工具）
    search_provider: str = "tavily"

    # 向量化 Embedding：mock（确定性哈希向量，离线可用）| openai（OpenAI 兼容端点）
    embedding_provider: str = "mock"
    embedding_api_key: str = ""
    embedding_base_url: str = ""
    embedding_model: str = "text-embedding-3-small"

    # 重排 Rerank：mock（语义余弦 + 词面重叠，离线可用）| cohere（Cohere Rerank API）
    rerank_provider: str = "mock"
    cohere_api_key: str = ""

    # RAG 召回：每轮 researcher 对候选来源做向量召回 + 重排返回的条数
    rag_top_k: int = 4
    # RAG 向量库后端：auto（默认，优先 Chroma 离线失败降级内存）| chroma（强制）| memory（纯内存）
    rag_backend: str = "auto"

    # 长记忆（语义召回层）：Chroma 持久化 + 离线降级内存，recall 语义优先、关键词兜底
    memory_enabled: bool = True
    # 后端：auto（默认，优先 Chroma 持久化，离线降级内存）| chroma（强制）| memory（纯内存）
    memory_backend: str = "auto"
    memory_path: str = ".memory_store"  # Chroma 持久化目录（跨进程/重启保留）
    memory_db_path: str = "memory.db"  # SQLite 关键词召回落库（兜底）
    memory_top_k: int = 3  # 每次 researcher 语义召回的历史洞察条数

    # Analyst 真实 LLM 自愈回路
    analyst_self_heal: int = 2  # 真实 LLM 下内部「自我批判-修复」最大尝试次数（mock 模式不进入循环）
    analyst_critic: str = "heuristic"  # heuristic（确定性启发式）| llm（额外 LLM 批判）

    # 自定义 MCP Server 启动命令（空格分隔）；留空则用项目自带服务（python -m src.mcp.server）
    mcp_server_command: str = ""

    # Obsidian 私域知识库：仓库根目录绝对路径；留空则使用内置合成仓库
    obsidian_vault_path: str = ""
    # 研究模式：web（仅联网）| local（仅 Obsidian）| hybrid（联网 + 私域知识）
    research_mode: str = "web"

    langfuse_public_key: str = ""
    langfuse_secret_key: str = ""
    langfuse_host: str = "https://cloud.langfuse.com"

    max_research_iterations: int = 3
    research_window: int = 10

    @field_validator(
        "rag_top_k",
        "memory_top_k",
        "analyst_self_heal",
        "max_research_iterations",
        "research_window",
        mode="before",
    )
    @classmethod
    def _coerce_int_fields(cls, v: Any, info: ValidationInfo) -> int:
        name = info.field_name or ""
        return _coerce_to_int(v, _INT_DEFAULTS.get(name, 0))

    @field_validator("temperature", mode="before")
    @classmethod
    def _coerce_temperature(cls, v: Any, info: ValidationInfo) -> float:
        name = info.field_name or "temperature"
        return _coerce_to_float(v, _FLOAT_DEFAULTS.get(name, 0.2))

    @field_validator("memory_enabled", mode="before")
    @classmethod
    def _coerce_memory_enabled(cls, v: Any) -> bool:
        if isinstance(v, bool):
            return v
        if isinstance(v, str):
            return v.strip().lower() in ("1", "true", "yes", "on")
        return bool(v)

    @classmethod
    def from_env(cls) -> "Settings":
        """向后兼容：原 @dataclass 的 `Settings.from_env()` 构造口，等价于 `Settings()`。"""
        return cls()


# 进程级全局配置单例：兼容 `from .config import settings` 与测试里的属性 mutate。
settings = Settings.from_env()


def resolve_settings(config: Any = None) -> "Settings":
    """节点优先使用按请求注入的配置 `RunnableConfig['configurable']['settings']`，
    未提供时回落到全局单例。这样同一份代码既能跑全局默认，也支持多租户 / 按请求覆写。

    `config` 在 LangGraph 运行时是 `RunnableConfig`（TypedDict，即 dict）。
    """
    if isinstance(config, dict):
        configurable = config.get("configurable")
        if isinstance(configurable, dict):
            candidate = configurable.get("settings")
            if isinstance(candidate, Settings):
                return candidate
    return settings


def get_settings() -> "Settings":
    """FastAPI 依赖：返回全局配置单例。

    后续可按需扩展为「按请求从 query/header/租户上下文构造 Settings」——
    所有通过 `Depends(get_settings)` 拿到配置并透传进 `graph.stream(config={"configurable": {"settings": ...}})`
    的调用点，都会自动享受到按请求覆写，无需改动节点内部。
    """
    return settings


# 按「配置指纹」缓存已构建的 LLM 客户端（复用 httpx 连接池）。
# 不能用 lru_cache 直接缓存 Settings 入参：Settings 可变且不可哈希，
# 且测试/运行时会 mutate 全局单例属性，必须按取值构造稳定 key。
#
# 用有界 LRU + 锁：多租户下 key 空间随租户数增长，无界 dict 会把每个租户的
# httpx 连接池永久留在内存里；而 job 线程并发调用 get_llm，普通 dict 的
# 读改写序列虽不至于崩，但会重复建客户端，白白多开连接池。
_LLM_CACHE_MAXSIZE = 32
_LLM_CACHE: "OrderedDict[tuple, Any]" = OrderedDict()
_LLM_CACHE_LOCK = threading.Lock()


def _credential_fingerprint(raw: str) -> str:
    """凭据指纹：进缓存键但不落明文，避免 key 出现在内存快照 / 异常回溯里。"""
    return hashlib.sha256((raw or "").encode("utf-8")).hexdigest()[:16]


def get_llm(s: Optional["Settings"] = None) -> Any:
    """返回 LLM 实例；`s` 为空时回落到全局单例配置。

    支持依赖注入：节点用 `get_llm(resolve_settings(config))` 即可让按请求 / 按任务
    覆写的 provider / model / temperature 真正生效，而不是永远拿全局那一份。
    mock 模式（provider == "mock" 或未配置 api_key）无需任何 key 即可跑通整条管线。
    """
    cfg = s if s is not None else settings
    provider = (cfg.llm_provider or "").strip().lower()
    has_key = bool(cfg.openai_api_key)
    use_mock = provider == "mock" or not has_key

    # 键里必须放「凭据指纹」而不是 has_key 布尔：否则同 model/base_url 的两个租户
    # 会命中同一个客户端，租户 B 的请求实际用租户 A 的 api_key 调上游——计费错记、
    # 配额互打，自建网关下还可能越权。同理，进程内热换 key 也会拿到陈旧客户端。
    key = (
        "mock" if use_mock else provider,
        cfg.model_name,
        cfg.temperature,
        cfg.openai_base_url,
        "" if use_mock else _credential_fingerprint(cfg.openai_api_key),
    )
    with _LLM_CACHE_LOCK:
        cached = _LLM_CACHE.get(key)
        if cached is not None:
            _LLM_CACHE.move_to_end(key)
            return cached

    # 构建放在锁外：ChatOpenAI 初始化可能触发 IO，不该阻塞其它线程读缓存
    if use_mock:
        client: Any = MockChatModel()
    else:
        from langchain_openai import ChatOpenAI

        client = ChatOpenAI(
            model=cfg.model_name,
            temperature=cfg.temperature,
            api_key=cfg.openai_api_key,
            base_url=cfg.openai_base_url,
        )

    with _LLM_CACHE_LOCK:
        # 并发下别的线程可能已建好同 key 客户端，复用它以免连接池翻倍
        existing = _LLM_CACHE.get(key)
        if existing is not None:
            _LLM_CACHE.move_to_end(key)
            return existing
        _LLM_CACHE[key] = client
        while len(_LLM_CACHE) > _LLM_CACHE_MAXSIZE:
            _LLM_CACHE.popitem(last=False)
    return client
