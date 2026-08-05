"""pytest 全局配置：离线 CI 下固定使用内存向量后端，并强制检索走 mock。

- `RAG_BACKEND=memory`：避免 Chroma 远端遥测握手阻塞（与「稠密召回 + 重排」逻辑等价）；
- `MEMORY_BACKEND=memory`：长记忆语义层走内存索引，避免 Chroma 初始化开销，测试快速确定；
- `MEMORY_DB_PATH` 指向临时文件：避免测试在仓库根目录生成 memory.db；
- 清空 `TAVILY_API_KEY`：本机若已配置真实 Tavily key，离线环境会触发真实联网调用并等待超时，
  清空后检索直接走 mock，保证测试快速、确定、零网络依赖。

注意：运行时入口（main.py / server.py）使用 `load_dotenv(override=True)`，会让 `.env` 里的
真实 key 在 import 阶段重新注入 os.environ。为避免测试受其污染，这里用 autouse fixture
在每个测试前**强制回空** TAVILY_API_KEY，保证检索稳定走 mock 分支，与运行环境完全解耦。
"""
import os
import tempfile

os.environ.setdefault("LLM_PROVIDER", "mock")
os.environ["RAG_BACKEND"] = "memory"
os.environ["MEMORY_BACKEND"] = "memory"
os.environ["MEMORY_DB_PATH"] = os.path.join(tempfile.gettempdir(), "dra_test_memory.db")
os.environ["TAVILY_API_KEY"] = ""

import pytest

import src.config as _cfg

_cfg.settings.tavily_api_key = ""


@pytest.fixture(autouse=True)
def _isolate_tavily_key(monkeypatch):
    """每个测试前强制清空 TAVILY_API_KEY，挡住 import 阶段 .env 重新注入的污染。"""
    monkeypatch.setenv("TAVILY_API_KEY", "")
    _cfg.settings.tavily_api_key = ""
