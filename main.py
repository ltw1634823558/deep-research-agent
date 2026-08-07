"""CLI 入口：python main.py "你的问题"  [--iterations N]

默认 mock 模式即可离线跑通整条管线；配置 .env 后切换真实模型。
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

try:
    from dotenv import load_dotenv

    load_dotenv(override=True)  # .env 优先于已存在的 Shell 环境变量（避免残留 key 遮蔽）
except Exception:
    pass

from src.config import settings  # noqa: E402
from src.evaluation import evaluate  # noqa: E402
from src.graph import build_graph  # noqa: E402
from src.observability import get_callbacks  # noqa: E402
from src.state import initial_state  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Deep Research Multi-Agent")
    parser.add_argument("query", nargs="?", default="人工智能 Agent 的发展现状与趋势")
    parser.add_argument("--iterations", type=int, default=settings.max_research_iterations)
    parser.add_argument(
        "--mode",
        choices=["web", "local", "hybrid"],
        default=settings.research_mode,
        help="web=仅联网, local=仅 Obsidian 私域知识, hybrid=两者混合",
    )
    args = parser.parse_args()

    graph = build_graph()
    # 与 server 侧行为对齐：CLI 同样注入 settings 与 LangFuse 回调，
    # 否则配置了 LangFuse 也拿不到任何 CLI 运行的 trace。
    result = graph.invoke(
        initial_state(args.query, args.iterations, mode=args.mode),
        config={"configurable": {"settings": settings}, "callbacks": get_callbacks()},
    )

    print(f"\n===== RESEARCH REPORT (mode={args.mode}) =====\n")
    print(result.get("report", ""))
    print("\n===== SOURCES =====\n")
    for s in result.get("sources", []):
        print(f"- [{getattr(s, 'source_type', 'web')}] {s.title}: {s.url}")

    print("\n===== EVALUATION (任务完成率/引用准确率/幻觉率) =====\n")
    print(json.dumps(evaluate(result, settings).to_dict(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
