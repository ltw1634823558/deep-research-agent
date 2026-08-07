"""Obsidian 私域知识库检索工具（RAG / Tool Use 的本地落地）。

加载 Obsidian 仓库（.md 笔记），解析 frontmatter、标签、双链(wikilinks)，
构建笔记关系图，提供两种检索能力：
- 关键词检索（离线，无需任何外部服务/API key）
- 图感知召回：命中笔记后扩展其双链邻居，获取「文档联系」（这正是与 Obsidian 结合的核心价值）

可选语义检索：传入 embed_fn 即可开启 Embedding 召回（进阶路线，默认关闭以保证离线可跑）。
未配置仓库路径或路径无效时，自动降级为内置合成仓库，保证离线演示与 CI 通过。
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from ..config import Settings, settings
from ..state import Source

# ---------- 轻量解析（不依赖 yaml，避免引入重依赖）----------

_WIKILINK_RE = re.compile(r"\[\[([^\]]+)\]\]")
_HEADING_RE = re.compile(r"^#{1,6}\s+(.+)$", re.MULTILINE)
_INLINE_TAG_RE = re.compile(r"#([A-Za-z0-9_\u4e00-\u9fff/-]+)")


def _tokenize(text: str) -> list[str]:
    """中英文混合分词：按字母/数字/汉字切，转小写。简单但够用且零依赖。"""
    return re.findall(r"[a-zA-Z0-9\u4e00-\u9fff]+", text.lower())


@dataclass
class Note:
    name: str
    path: str  # 仓库相对路径，用于引用
    content: str
    tags: set[str] = field(default_factory=set)
    links: set[str] = field(default_factory=set)  # 出链（指向的笔记名，仅保留存在的目标）
    headings: list[str] = field(default_factory=list)


class ObsidianVault:
    def __init__(
        self,
        root: str | None = None,
        embed_fn: Callable[[str], list[float]] | None = None,
    ) -> None:
        self.root = root
        self.embed_fn = embed_fn
        self.notes: dict[str, Note] = {}
        self._index: dict[str, dict[str, int]] = {}
        if root and Path(root).is_dir():
            self._load(root)
        else:
            self._load_synthetic()

    # ---- 加载 ----

    def _load(self, root: str) -> None:
        for p in Path(root).rglob("*.md"):
            try:
                text = p.read_text(encoding="utf-8", errors="ignore")
            except Exception:
                continue
            rel = str(Path(p).relative_to(root))
            note = self._parse(p.stem, rel, text)
            self.notes[note.name] = note
        self._build_index()
        self._build_links()

    def _parse(self, name: str, rel: str, text: str) -> Note:
        fm_tags: set[str] = set()
        body = text
        if text.startswith("---"):
            end = text.find("\n---", 3)
            if end != -1:
                fm = text[3:end]
                body = text[end + 4 :]
                for line in fm.splitlines():
                    m = re.match(r"\s*tags:\s*\[(.*)\]\s*$", line)
                    if m:
                        fm_tags.update(
                            t.strip().lstrip("#") for t in m.group(1).split(",") if t.strip()
                        )
                    m2 = re.match(r"\s*-\s*#?([\w\u4e00-\u9fff/-]+)\s*$", line)
                    if m2:
                        fm_tags.add(m2.group(1).lstrip("#"))
        tags = set(fm_tags)
        # 内联标签：跳过标题行（以 # 开头的行），避免把标题误判为标签
        for line in body.splitlines():
            if line.lstrip().startswith("#"):
                continue
            tags.update(_INLINE_TAG_RE.findall(line))
        links = {self._link_target(x) for x in _WIKILINK_RE.findall(body)}
        headings = _HEADING_RE.findall(body)
        return Note(name=name, path=rel, content=body, tags=tags, links=links, headings=headings)

    @staticmethod
    def _link_target(raw: str) -> str:
        """[[Note]] / [[Note|alias]] / [[Note#Heading]] -> Note"""
        return raw.split("|")[0].split("#")[0].strip()

    def _build_index(self) -> None:
        for n in self.notes.values():
            tf: dict[str, int] = {}
            toks = _tokenize(n.name.replace("_", " ")) + _tokenize(n.content)
            for tok in toks:
                tf[tok] = tf.get(tok, 0) + 1
            self._index[n.name] = tf

    def _build_links(self) -> None:
        for n in self.notes.values():
            n.links = {t for t in n.links if t in self.notes}

    # ---- 检索 ----

    def retrieve(self, query: str, top_k: int = 5, use_graph: bool = True) -> list[Source]:
        q_terms = _tokenize(query)
        if not q_terms:
            return []
        scored: list[tuple[int, str]] = []
        for name, tf in self._index.items():
            score = sum(tf.get(t, 0) for t in q_terms)
            if any(t in name.lower() for t in q_terms):
                score += 3  # 标题命中加权
            if any(t in self.notes[name].tags for t in q_terms):
                score += 2  # 标签命中加权
            if score > 0:
                scored.append((score, name))
        scored.sort(reverse=True)
        hits = [self.notes[name] for _, name in scored[:top_k]]

        # 防御性兜底：关键词无命中时，返回连接度最高的若干笔记，
        # 保证 hybrid 模式下私域知识总能被纳入（也避免检索完全空窗）。
        if not hits:
            ranked = sorted(self.notes.values(), key=lambda n: len(n.links), reverse=True)
            hits = ranked[:top_k]

        sources: list[Source] = []
        seen: set[str] = set()
        for n in hits:
            sources.append(self._to_source(n))
            seen.add(n.name)
            if use_graph:  # 图感知：把命中笔记的双链邻居也带上，体现「文档联系」
                for nb in self.neighbors(n.name, depth=1):
                    if nb.name not in seen:
                        sources.append(self._to_source(nb, via=n.name))
                        seen.add(nb.name)
        return sources

    def neighbors(self, name: str, depth: int = 1) -> list[Note]:
        """按双链关系做广度优先扩展，返回关联笔记（不含自身）。"""
        if name not in self.notes or depth <= 0:
            return []
        visited = {name}
        frontier = {name}
        result: list[Note] = []
        for _ in range(depth):
            nxt: set[str] = set()
            for cur in frontier:
                for link in self.notes[cur].links:
                    if link not in visited and link in self.notes:
                        visited.add(link)
                        result.append(self.notes[link])
                        nxt.add(link)
            frontier = nxt
        return result

    def _to_source(self, n: Note, via: str | None = None) -> Source:
        snippet = n.content.strip()[:400].replace("\n", " ")
        title = n.name + (f" ← 关联自「{via}」" if via else "")
        if self.root:
            ref = "obsidian://open?path=" + self.root.replace("\\", "/") + "/" + n.path
        else:
            ref = f"vault:{n.path}"
        return Source(url=ref, title=title, snippet=snippet, source_type="local")

    # ---- 合成仓库（离线降级）----

    def _load_synthetic(self) -> None:
        synthetic = {
            "Agent工程化": "# Agent 工程化\n#agent #engineering\nAgent 的工程化需要关注编排、可观测与评估。\n[[多智能体编排]] 与 [[可观测性]] 是核心。",
            "多智能体编排": "# 多智能体编排\n#agent #langgraph\n使用 LangGraph 做状态机编排，支持循环与回环。参见 [[Agent工程化]]。",
            "可观测性": "# 可观测性\n#observability #langfuse\n用 LangFuse 追踪每个节点与 token 消耗。参见 [[Agent工程化]]。",
            "RAG检索增强": "# RAG 检索增强\n#rag\n检索增强生成需要向量库与重排。参见 [[多智能体编排]]。",
            "Obsidian知识库": "# Obsidian 知识库\n#obsidian #knowledge\n用本地笔记作为 Agent 的私域知识源，支持双链关系图。",
            # 覆盖常见研究子主题，保证离线/hybrid 演示时私域知识总能被命中
            "背景与定义": "# 背景与定义\n#overview\nAgent 的基本概念与边界。关联 [[Agent工程化]]。",
            "核心技术架构": "# 核心技术架构\n#architecture\n编排、检索与记忆的协作。关联 [[多智能体编排]] 与 [[RAG检索增强]]。",
            "行业应用与案例": "# 行业应用与案例\n#case\n私域知识在研发提效中的落地。关联 [[Obsidian知识库]]。",
            "挑战与未来趋势": "# 挑战与未来趋势\n#trend\n可靠性、成本与评估仍是难点。关联 [[可观测性]]。",
        }
        for name, body in synthetic.items():
            n = self._parse(name, f"{name}.md", body)
            self.notes[n.name] = n
        self._build_index()
        self._build_links()


# 模块级惰性缓存：按仓库路径缓存 ObsidianVault 单例（首次真正检索时才扫描用户仓库，
# 仓库无效则降级合成仓库）。per-request 的 obsidian_vault_path 变化时命中不同缓存项。
_vaults: dict[str, ObsidianVault] = {}


def _get_vault(settings_obj: "Settings | None" = None) -> ObsidianVault:
    s = settings_obj or settings
    path = s.obsidian_vault_path or None
    if path not in _vaults:
        _vaults[path] = ObsidianVault(path)
    return _vaults[path]


def __getattr__(name: str) -> object:
    """PEP 562：让 `from ...obsidian import obsidian_vault` 仍可用，但按需构建。"""
    if name == "obsidian_vault":
        return _get_vault()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def obsidian_search(
    query: str, top_k: int = 5, use_graph: bool = True, settings_obj: "Settings | None" = None
) -> list[Source]:
    """对上层透明的 Obsidian 检索入口；无仓库时返回合成结果，保证离线可跑、可测。

    settings_obj 透传按请求配置（per-request obsidian_vault_path），未传回落全局。
    """
    return _get_vault(settings_obj).retrieve(query, top_k=top_k, use_graph=use_graph)
