"""Obsidian 工具测试：合成仓库检索、双链邻居、真实仓库加载、以及混合模式端到端。"""

import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

os.environ["LLM_PROVIDER"] = "mock"
os.environ["RAG_BACKEND"] = "memory"  # 离线 CI 用内存向量库，避免 Chroma 远端握手阻塞

from src.graph import build_graph  # noqa: E402
from src.state import initial_state  # noqa: E402
from src.tools.obsidian import ObsidianVault, obsidian_search, obsidian_vault  # noqa: E402


def test_synthetic_retrieval_returns_local_sources():
    results = obsidian_search("Agent 工程化", top_k=3)
    assert results, "合成仓库应返回结果"
    assert all(s.source_type == "local" for s in results)
    assert any("Agent工程化" in s.title for s in results)


def test_graph_expansion_includes_linked_notes():
    # "Agent工程化" 双链指向「多智能体编排」「可观测性」，图感知应把它们带上
    results = obsidian_search("Agent 工程化", top_k=1, use_graph=True)
    titles = [s.title for s in results]
    assert any("多智能体编排" in t for t in titles), "应扩展出双链邻居「多智能体编排」"
    assert any("关联自" in t for t in titles), "邻居来源应标注关联来源"


def test_neighbors_breadth():
    vault = obsidian_vault
    nbs = vault.neighbors("Agent工程化", depth=1)
    names = {n.name for n in nbs}
    assert "多智能体编排" in names
    assert "可观测性" in names


def test_real_vault_load_parses_frontmatter_and_wikilinks():
    with tempfile.TemporaryDirectory() as d:
        vault_dir = Path(d) / "vault"
        vault_dir.mkdir()
        (vault_dir / "A.md").write_text(
            "---\ntags: [project, agent]\n---\n# A\n正文关于 [[B]] 的关联。#inline-tag\n",
            encoding="utf-8",
        )
        (vault_dir / "B.md").write_text("# B\nB 的内容，回链见 [[A]]。\n", encoding="utf-8")
        vault = ObsidianVault(str(vault_dir))
        assert "A" in vault.notes and "B" in vault.notes
        assert "agent" in vault.notes["A"].tags, "frontmatter 标签应被解析"
        assert "inline-tag" in vault.notes["A"].tags, "内联标签应被解析（非标题）"
        assert "B" in vault.notes["A"].links, "双链目标应被解析"
        assert vault.neighbors("A", depth=1)[0].name == "B"


def test_hybrid_mode_end_to_end_includes_local_source():
    graph = build_graph()
    result = graph.invoke(initial_state("Agent 工程化与可观测", max_iterations=2, mode="hybrid"))
    assert result.get("report"), "hybrid 模式应产出报告"
    types = {getattr(s, "source_type", "web") for s in result.get("sources", [])}
    assert "local" in types, "hybrid 模式应引入 Obsidian 私域来源"
    assert "web" in types, "hybrid 模式应保留联网来源"


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("OK:", name)
