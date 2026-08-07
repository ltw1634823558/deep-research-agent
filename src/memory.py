"""长短记忆：

- 短期记忆：由 LangGraph 的 messages 通道承载，滑动窗口见 state._add_messages_windowed
  （窗口大小 = config.research_window，<=0 表示不限制）。
- 长期记忆：
  - SQLite 持久化历史研究报告（关键词召回，向后兼容）；
  - 叠加【语义长记忆】层——把研究洞察做 Embedding 存入 Chroma（持久化，跨进程/重启保留），
    researcher 阶段做语义召回，writer 阶段落库。复用 rag.embeddings.Embedder 与离线防御式初始化
    （守护线程 + 超时探测），离线/网络受限时自动降级内存余弦召回，保证「零 key 离线可跑」。

语义召回优先；当语义层无可召回内容（首次运行 / 降级内存且无命中）时，回退到 SQLite 关键词召回，
保持对旧行为的兼容。
"""

from __future__ import annotations

import datetime
import hashlib
import logging
import os
import sqlite3
import threading
from typing import Any

from .config import Settings, settings
from .rag.embeddings import ChromaEmbeddingFunction, Embedder, cosine

logger = logging.getLogger(__name__)

# 离线防御：Chroma 初始化用守护线程 + 超时探测包裹（与 rag/store.py 同源思路，独立探针缓存）
_MEM_PROBE: bool | None = None
_MEM_TIMEOUT = 6.0


class SemanticIndex:
    """语义向量索引：Chroma 持久化优先，离线降级内存余弦。

    设计要点（senior 视角）：
    - 离线环境下 Chroma 会尝试远端遥测握手而阻塞，故用 daemon 线程 + 超时包裹初始化，
      离线/受限时安全放弃，绝不卡死进程退出；
    - PersistentClient 写盘到 memory_path，使历史洞察跨进程重启仍可用；
    - 单条洞察按文本 md5 做幂等 id（upsert），重复研究不产生冗余。
    """

    def __init__(
        self,
        embedder: Embedder | None = None,
        backend: str | None = None,
        path: str | None = None,
    ) -> None:
        self.embedder = embedder or Embedder()
        self.backend_wanted = backend or settings.memory_backend
        self.path = path or settings.memory_path
        self._mem: dict[str, tuple[list[float], str, dict[str, Any]]] = {}
        self._col: Any = None
        self._client: Any = None
        self._use_chroma = False
        if self.backend_wanted != "memory":
            self._use_chroma = self._try_init_chroma(force=(self.backend_wanted == "chroma"))

    @property
    def backend(self) -> str:
        return "chromadb" if self._use_chroma else "in-memory"

    def _try_init_chroma(self, force: bool) -> bool:
        global _MEM_PROBE
        if _MEM_PROBE is False:
            return False  # 已知离线/不可用，直接降级，避免重复阻塞
        box: dict[str, Any] = {}

        def _init() -> None:
            try:
                import chromadb
                from chromadb.config import Settings

                client = chromadb.PersistentClient(
                    path=self.path,
                    settings=Settings(anonymized_telemetry=False, allow_reset=True),
                )
                ef = ChromaEmbeddingFunction.create(self.embedder)

                # 若已存在旧 collection 且使用的是 Chroma 默认 ONNX embedding function，
                # 直接删掉重建，避免触发 all-MiniLM-L6-v2/onnx.tar.gz 下载。
                for existing in client.list_collections():
                    if getattr(existing, "name", existing) == "long_term_memory":
                        cfg = existing.configuration_json
                        ef_cfg = cfg.get("embedding_function", {}) if isinstance(cfg, dict) else {}
                        if ef_cfg.get("name") == "default":
                            client.delete_collection("long_term_memory")
                        break

                try:
                    col = client.get_or_create_collection(
                        name="long_term_memory",
                        metadata={"hnsw:space": "cosine"},
                        embedding_function=ef,
                    )
                except ValueError as ve:
                    # 兜底：万一版本差异导致校验没走上面分支，仍按冲突处理
                    if "embedding function conflict" in str(ve).lower():
                        client.delete_collection("long_term_memory")
                        col = client.create_collection(
                            name="long_term_memory",
                            metadata={"hnsw:space": "cosine"},
                            embedding_function=ef,
                        )
                    else:
                        raise
                # 探针：确认离线环境下不会卡死
                col.add(
                    ids=["__probe__"],
                    documents=["probe"],
                    metadatas=[{"id": "__probe__"}],
                )
                col.query(query_texts=["probe"], n_results=1)
                col.delete(ids=["__probe__"])
                box["col"] = col
                box["client"] = client
            except Exception as exc:
                logger.warning("Chroma 长记忆初始化失败，将降级内存索引：%s", exc)

        t = threading.Thread(target=_init, daemon=True)
        t.start()
        t.join(_MEM_TIMEOUT)
        if "col" in box:
            self._col = box["col"]
            self._client = box.get("client")
            _MEM_PROBE = True
            return True
        _MEM_PROBE = False
        if force:
            import sys

            print(
                "[memory] 警告：Chroma 初始化失败（可能离线/网络受限），长记忆降级内存索引。",
                file=sys.stderr,
            )
        return False

    def add(self, text: str, metadata: dict[str, Any] | None = None) -> None:
        if not text or not text.strip():
            return
        _id = "m_" + hashlib.md5(text.encode("utf-8")).hexdigest()[:16]
        if self._use_chroma:
            try:
                self._col.upsert(
                    ids=[_id],
                    documents=[text],
                    metadatas=[{**(metadata or {}), "id": _id}],
                )
                return
            except Exception as exc:
                # Chroma 运行期失败则降级内存
                logger.warning("Chroma 写入失败，降级内存索引：%s", exc)
        # 内存余弦兜底
        self._mem[_id] = (self.embedder.embed(text), text, metadata or {})

    def search(self, query: str, top_n: int = 3) -> list[str]:
        if not query or not query.strip():
            return []
        if self._use_chroma:
            try:
                res = self._col.query(query_texts=[query], n_results=top_n)
                docs = (res.get("documents") or [[]])[0]
                return [d for d in docs if d]
            except Exception as exc:
                logger.warning("Chroma 检索失败，降级内存余弦召回：%s", exc)
        q = self.embedder.embed(query)
        ranked = sorted(
            self._mem.values(),
            key=lambda t: cosine(q, t[0]),
            reverse=True,
        )
        return [t[1] for t in ranked[:top_n]]

    def close(self) -> None:
        """释放 Chroma 连接（PersistentClient 持有的文件锁），降级为内存态。

        供 MemoryStore.close() 调用，确保 Windows 下 .memory_store 目录能被正常清理（L-C）。
        """
        if self._col is not None:
            try:
                self._col = None
                # 释放 chromadb 类级共享客户端缓存，真正解除文件锁（L-C）
                if self._client is not None:
                    clear = getattr(self._client, "clear_system_cache", None)
                    if callable(clear):
                        clear()
            except Exception as exc:
                logger.debug("Chroma 长记忆清理失败（可忽略）：%s", exc)
        self._client = None
        self._use_chroma = False


def _extract_insights(query: str, report: str) -> list[str]:
    """从报告中抽取可用于长期记忆的洞察条目（确定、离线、无 LLM 依赖）。

    以非空要点行（bullet/短句）为主，跳过大标题/引用/代码块；去重并截断，避免索引膨胀。
    """
    lines = [ln.strip() for ln in report.splitlines() if ln.strip()]
    out: list[str] = []
    seen: set[str] = set()
    for ln in lines:
        if ln.startswith(("#", ">", "`", "|")):
            continue
        if len(ln) < 6:
            continue
        if ln not in seen:
            seen.add(ln)
            out.append(ln)
    # 始终保留一条主题锚点，便于跨研究关联
    anchor = f"研究主题：{query}"
    if anchor not in seen:
        out.insert(0, anchor)
    return out[:12]


class MemoryStore:
    """长期记忆：SQLite 落库 + 语义向量索引（SemanticIndex）。

    - save：落库报告并抽取洞察写入语义索引（幂等）；
    - recall：语义召回优先，无命中时回退关键词召回（向后兼容）。

    语义索引惰性构建：`SemanticIndex()` 内含最长 6s 的 Chroma 探测，若在 __init__ 里建，
    则 `import src.server` 等任何一次导入都要付这份成本；改为首次 save/recall 访问时才建。
    """

    def __init__(
        self,
        path: str | None = None,
        semantic: SemanticIndex | None = None,
    ) -> None:
        self.path = path or settings.memory_db_path
        # 不在 import 时固化开关/连接：enabled 在每次调用时重新读取（支持 per-request 覆盖），
        # sqlite 连接延迟到首次 save/recall 才建立（惰性 + 并发安全）。
        self._conn: sqlite3.Connection | None = None
        self._lock = threading.Lock()
        # index 构建可能触发最长 6s 的 Chroma 探测，独立锁避免阻塞 SQLite 读写（L-3）
        self._index_lock = threading.Lock()
        self._index: SemanticIndex | None = semantic

    @property
    def index(self) -> SemanticIndex:
        """惰性语义索引：首次访问时才构建（含 Chroma 探测），之后复用。

        加锁双检：全局 memory_store 可能被并发任务共享，避免重复构建 SemanticIndex。
        """
        if self._index is None:
            with self._index_lock:
                if self._index is None:
                    self._index = SemanticIndex()
        return self._index

    @index.setter
    def index(self, value: SemanticIndex) -> None:
        self._index = value

    def _ensure_conn(self) -> None:
        """惰性建立（且仅一次）sqlite 连接：带 timeout 与跨线程开关，建表幂等。

        调用方须持有 self._lock（避免在已持锁的 save/recall 中重复加锁导致死锁）。
        """
        if self._conn is None:
            if os.path.dirname(self.path):
                os.makedirs(os.path.dirname(self.path), exist_ok=True)
            con = sqlite3.connect(
                self.path, timeout=30.0, check_same_thread=False
            )
            con.execute(
                "CREATE TABLE IF NOT EXISTS reports ("
                "id INTEGER PRIMARY KEY, query TEXT, report TEXT, ts TEXT)"
            )
            self._conn = con

    def save(self, query: str, report: str, cfg: "Settings | None" = None) -> int:
        """落库报告并写入语义索引，返回写入的洞察条数。

        cfg 透传按请求配置，使 per-request memory_enabled 真正生效（path/backend 为进程级全局，不随请求变）。
        """
        s = cfg or settings
        if not s.memory_enabled:
            return 0
        with self._lock:
            self._ensure_conn()
            if self._conn is None:
                return 0
            with self._conn:
                self._conn.execute(
                    "INSERT INTO reports (query, report, ts) VALUES (?, ?, ?)",
                    (query, report, datetime.datetime.now().isoformat()),
                )

        insights = _extract_insights(query, report)
        for ins in insights:
            self.index.add(ins, {"query": query, "kind": "insight"})
        return len(insights)

    def recall(self, query: str, k: int | None = None, cfg: "Settings | None" = None) -> list[str]:
        """召回历史记忆：语义优先，无命中回退关键词。返回可注入上下文的字符串列表。

        cfg 透传按请求配置，使 per-request memory_enabled / memory_top_k 真正生效。
        """
        s = cfg or settings
        if not s.memory_enabled:
            return []
        k = k or s.memory_top_k

        hits = self.index.search(query, top_n=k)
        if hits:
            # 两条召回路径都做长度上限，避免过长内容灌入 prompt（L-B）
            return [h[:2000] for h in hits[:k]]

        # 兜底：SQLite 关键词重叠（向后兼容原有行为）。加 LIMIT 避免全表扫描撑爆内存（L-2）
        with self._lock:
            self._ensure_conn()
            if self._conn is None:
                return []
            with self._conn:
                rows = self._conn.execute(
                    "SELECT query, report FROM reports ORDER BY id DESC LIMIT 500"
                ).fetchall()
        q_words = set(query.lower().split())
        scored = [
            (len(q_words & set(q.lower().split())), report) for q, report in rows
        ]
        scored = [s for s in scored if s[0] > 0]
        scored.sort(reverse=True)
        # 单条内容截断，避免长报告全文灌入 prompt（L-2）
        return [report[:2000] for _, report in scored[:k]]

    def close(self) -> None:
        """释放 SQLite 连接与语义索引，便于测试与进程退出时的确定性清理。

        连接为惰性长连接（见 _ensure_conn），不主动关闭会导致 Windows 下
        持有文件锁、临时目录/文件无法删除。
        """
        with self._index_lock:
            idx = self._index
            self._index = None
            if idx is not None:
                idx.close()
        with self._lock:
            con = self._conn
            self._conn = None
            if con is not None:
                con.close()


memory_store = MemoryStore()
