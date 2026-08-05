"""Mock 冒烟测试：无 API key 跑通整条多智能体管线，验证节点与路由正确。"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

os.environ["LLM_PROVIDER"] = "mock"
os.environ["RAG_BACKEND"] = "memory"  # 离线 CI 用内存向量库，避免 Chroma 远端握手阻塞

from src.graph import build_graph  # noqa: E402
from src.state import initial_state  # noqa: E402


def test_mock_run():
    graph = build_graph()
    result = graph.invoke(initial_state("测试主题：Agent 工程化", max_iterations=2))

    assert result.get("report"), "writer 应产出报告"
    assert result.get("findings"), "researcher 应产出发现"
    assert result.get("plan"), "planner 应产出计划"
    assert all(s.status == "done" for s in result["plan"]), "所有子主题应被标记完成"
    print("\n[mock run] report length:", len(result["report"]))
    print("[mock run] findings:", len(result["findings"]))
    print("[mock run] OK")


if __name__ == "__main__":
    test_mock_run()
