"""长短记忆：

- 短期记忆：由 LangGraph 的 messages 通道（带窗口）承载，见 config.research_window。
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
import os
import sqlite3
import threading
from typing import Any

from .config import settings
from .rag.embeddings import ChromaEmbeddingFunction, Embedder, cosine

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
            except Exception:
                pass

        t = threading.Thread(target=_init, daemon=True)
        t.start()
        t.join(_MEM_TIMEOUT)
        if "col" in box:
            self._col = box["col"]
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
            except Exception:
                pass  # Chroma 运行期失败则降级内存
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
            except Exception:
                pass  # 降级内存
        q = self.embedder.embed(query)
        ranked = sorted(
            self._mem.values(),
            key=lambda t: cosine(q, t[0]),
            reverse=True,
        )
        return [t[1] for t in ranked[:top_n]]


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
    """

    def __init__(
        self,
        path: str | None = None,
        semantic: SemanticIndex | None = None,
    ) -> None:
        self.path = path or settings.memory_db_path
        self.enabled = settings.memory_enabled
        self._init_db()
        self.index = semantic or SemanticIndex()
        self.last_recall: list[str] = []

    def _init_db(self) -> None:
        if os.path.dirname(self.path):
            os.makedirs(os.path.dirname(self.path), exist_ok=True)
        con = sqlite3.connect(self.path)
        con.execute(
            "CREATE TABLE IF NOT EXISTS reports ("
            "id INTEGER PRIMARY KEY, query TEXT, report TEXT, ts TEXT)"
        )
        con.commit()
        con.close()

    def save(self, query: str, report: str) -> int:
        """落库报告并写入语义索引，返回写入的洞察条数。"""
        if not self.enabled:
            return 0
        con = sqlite3.connect(self.path)
        con.execute(
            "INSERT INTO reports (query, report, ts) VALUES (?, ?, ?)",
            (query, report, datetime.datetime.now().isoformat()),
        )
        con.commit()
        con.close()

        insights = _extract_insights(query, report)
        for ins in insights:
            self.index.add(ins, {"query": query, "kind": "insight"})
        return len(insights)

    def recall(self, query: str, k: int | None = None) -> list[str]:
        """召回历史记忆：语义优先，无命中回退关键词。返回可注入上下文的字符串列表。"""
        if not self.enabled:
            return []
        k = k or settings.memory_top_k

        hits = self.index.search(query, top_n=k)
        if hits:
            self.last_recall = hits
            return hits

        # 兜底：SQLite 关键词重叠（向后兼容原有行为）
        con = sqlite3.connect(self.path)
        rows = con.execute("SELECT query, report FROM reports").fetchall()
        con.close()
        q_words = set(query.lower().split())
        scored = [
            (len(q_words & set(q.lower().split())), report) for q, report in rows
        ]
        scored = [s for s in scored if s[0] > 0]
        scored.sort(reverse=True)
        self.last_recall = [report for _, report in scored[:k]]
        return self.last_recall


memory_store = MemoryStore()
