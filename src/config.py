"""全局配置：从环境变量加载，兼容 OpenAI / 兼容端点 / 离线 mock 模式。"""

from __future__ import annotations

import os
from dataclasses import dataclass

from .mock import MockChatModel


@dataclass
class Settings:
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

    @classmethod
    def from_env(cls) -> Settings:
        return cls(
            llm_provider=os.getenv("LLM_PROVIDER", "mock").strip(),
            openai_api_key=os.getenv("OPENAI_API_KEY", "").strip(),
            openai_base_url=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1").strip(),
            model_name=os.getenv("MODEL_NAME", "gpt-4o-mini").strip(),
            tavily_api_key=os.getenv("TAVILY_API_KEY", "").strip(),
            search_provider=os.getenv("SEARCH_PROVIDER", "tavily").strip(),
            embedding_provider=os.getenv("EMBEDDING_PROVIDER", "mock").strip(),
            embedding_api_key=os.getenv("EMBEDDING_API_KEY", "").strip(),
            embedding_base_url=os.getenv("EMBEDDING_BASE_URL", "").strip(),
            embedding_model=os.getenv("EMBEDDING_MODEL", "text-embedding-3-small").strip(),
            rerank_provider=os.getenv("RERANK_PROVIDER", "mock").strip(),
            cohere_api_key=os.getenv("COHERE_API_KEY", "").strip(),
            rag_top_k=int(os.getenv("RAG_TOP_K", "4")),
            rag_backend=os.getenv("RAG_BACKEND", "auto"),
            memory_enabled=os.getenv("MEMORY_ENABLED", "true").lower() in ("1", "true", "yes", "on"),
            memory_backend=os.getenv("MEMORY_BACKEND", "auto"),
            memory_path=os.getenv("MEMORY_PATH", ".memory_store"),
            memory_db_path=os.getenv("MEMORY_DB_PATH", "memory.db"),
            memory_top_k=int(os.getenv("MEMORY_TOP_K", "3")),
            analyst_self_heal=int(os.getenv("ANALYST_SELF_HEAL", "2")),
            analyst_critic=os.getenv("ANALYST_CRITIC", "heuristic"),
            mcp_server_command=os.getenv("MCP_SERVER_COMMAND", ""),
            obsidian_vault_path=os.getenv("OBSIDIAN_VAULT_PATH", ""),
            research_mode=os.getenv("RESEARCH_MODE", "web"),
            langfuse_public_key=os.getenv("LANGFUSE_PUBLIC_KEY", ""),
            langfuse_secret_key=os.getenv("LANGFUSE_SECRET_KEY", ""),
            langfuse_host=os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com"),
            max_research_iterations=int(os.getenv("MAX_RESEARCH_ITERATIONS", "3")),
            research_window=int(os.getenv("RESEARCH_WINDOW", "10")),
        )


settings = Settings.from_env()


def get_llm():
    """返回 LLM 实例。mock 模式无需任何 key 即可跑通整条管线。"""
    if settings.llm_provider == "mock":
        return MockChatModel()
    from langchain_openai import ChatOpenAI

    return ChatOpenAI(
        model=settings.model_name,
        temperature=settings.temperature,
        api_key=settings.openai_api_key,
        base_url=settings.openai_base_url,
    )
