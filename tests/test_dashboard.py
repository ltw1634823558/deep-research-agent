"""Dashboard 面板与实时作业追踪测试（全程离线，conftest 已强制 RAG_BACKEND=memory）。"""

from __future__ import annotations

import time

from fastapi.testclient import TestClient

from src.server import app

client = TestClient(app)


def test_dashboard_page():
    r = client.get("/")
    assert r.status_code == 200
    body = r.text
    assert "Deep Research" in body
    assert "发起新研究" in body
    assert "运行状态" in body


def test_api_jobs_list():
    r = client.get("/api/jobs")
    assert r.status_code == 200
    assert isinstance(r.json(), list)


def test_unknown_job_404():
    r = client.get("/api/jobs/does-not-exist")
    assert r.status_code == 404


def test_research_job_runs_to_done():
    r = client.post("/research/job", json={"query": "测试 Agent 运行状态面板", "mode": "web"})
    assert r.status_code == 200
    data = r.json()
    assert "job_id" in data
    job_id = data["job_id"]

    # 轮询直到完成（mock 模式秒级完成）
    detail = None
    for _ in range(40):
        rr = client.get(f"/api/jobs/{job_id}")
        assert rr.status_code == 200
        detail = rr.json()
        if detail["status"] in ("done", "error"):
            break
        time.sleep(0.5)

    assert detail["status"] == "done", detail.get("error")
    assert detail["report"]
    assert isinstance(detail["sources"], list)
    assert detail["metrics"].get("task_completion_rate") is not None
    # 阶段应推进到完成（STAGES 末尾下标 = 5）
    assert detail["stage_index"] == 5
    assert detail["progress"] == 100


def test_sync_research_also_records_job(monkeypatch):
    # 隔离 store：用独立 JobStore 替换 server 模块级单例，避免依赖全局作业计数
    # （防止其他测试写入或 LRU 淘汰干扰前后差值断言），保证测试密闭可重复。
    from src.jobs import JobStore

    isolated = JobStore()
    monkeypatch.setattr("src.server.store", isolated)
    before = len(isolated.list_snapshots())
    r = client.post("/research", json={"query": "同步接口也应记入面板", "mode": "web"})
    assert r.status_code == 200
    assert r.json()["metrics"]
    after = len(isolated.list_snapshots())
    assert after == before + 1


def test_job_queue_overflow_returns_429(monkeypatch):
    """L-J 回归：在途任务超上限必须返回 429，而不是往无界队列里无限堆。

    ThreadPoolExecutor 队列无界，堆积的任务对应的 Job 记录会被 JobStore 的
    LRU（上限 50）淘汰，结果无处可查——对调用方表现为静默丢失。
    """
    from src import server

    monkeypatch.setattr(server, "_JOB_QUEUE_LIMIT", 1)
    monkeypatch.setattr(server, "_inflight_jobs", 0)
    # 占满唯一名额，且不真正跑图
    monkeypatch.setattr(server._JOB_EXECUTOR, "submit", lambda *a, **k: _DummyFuture())

    ok = client.post("/research/job", json={"query": "q1", "mode": "web"})
    assert ok.status_code == 200

    blocked = client.post("/research/job", json={"query": "q2", "mode": "web"})
    assert blocked.status_code == 429
    assert "排队已满" in blocked.json()["detail"]


class _DummyFuture:
    """占位 Future：不执行任务，也不触发 done_callback（模拟任务仍在途）。"""

    def add_done_callback(self, fn):
        return None


def test_job_error_is_sanitized_before_exposure():
    """L-H 回归：异常原文不得经未鉴权的 /api/jobs 回显。"""
    from src.server import _public_error

    msg = _public_error(RuntimeError("/abs/path/secret.py: invalid api key sk-abcdef123456"))
    assert "sk-abcdef123456" not in msg
    assert "/abs/path" not in msg
    assert "RuntimeError" in msg


def test_shutdown_releases_resources_in_order(monkeypatch):
    """M-C 回归：应用关闭必须按序收在途任务 + flush 追踪 + 关长记忆。

    langfuse 的 OTel 批处理器攒批异步上报，不 flush 就丢最后一批 trace；
    线程池不 shutdown 会拖到编排层 SIGKILL；memory 句柄不关 Windows 下锁文件。
    此前应用无任何 shutdown 钩子，属于「只定义 close 却从不触发」。
    """
    from src import server

    calls = []
    monkeypatch.setattr(server, "shutdown_all_clients", lambda: calls.append("langfuse"))
    monkeypatch.setattr(server.memory_store, "close", lambda: calls.append("memory"))
    monkeypatch.setattr(
        server._JOB_EXECUTOR, "shutdown", lambda **k: calls.append(("pool", k))
    )

    server._shutdown_resources()

    kinds = [c[0] if isinstance(c, tuple) else c for c in calls]
    # 先收在途任务（pool），再 flush 追踪（langfuse），最后关记忆（memory）
    assert kinds.index("pool") < kinds.index("langfuse") < kinds.index("memory")
    # 第一次 shutdown 必须 cancel 掉排队任务，避免收到 SIGTERM 后还在起新活
    assert any(isinstance(c, tuple) and c[1].get("cancel_futures") for c in calls)


def test_shutdown_continues_when_a_step_fails(monkeypatch):
    """单步失败不得阻断后续释放：langfuse flush 抛错，memory 仍要被关闭。"""
    from src import server

    def _boom():
        raise RuntimeError("langfuse 上报炸了")

    closed = []
    monkeypatch.setattr(server, "shutdown_all_clients", _boom)
    monkeypatch.setattr(server.memory_store, "close", lambda: closed.append("memory"))
    monkeypatch.setattr(server._JOB_EXECUTOR, "shutdown", lambda **k: None)

    server._shutdown_resources()  # 不应抛异常
    assert closed == ["memory"]


def test_lifespan_is_wired_to_shutdown(monkeypatch):
    """确保钩子真的挂在 app 上：退出 TestClient 上下文即触发释放。"""
    from src import server

    fired = []
    monkeypatch.setattr(server, "_shutdown_resources", lambda: fired.append(True))
    with TestClient(server.app):
        pass
    assert fired == [True]


def test_shutdown_all_clients_is_idempotent():
    """observability.shutdown_all_clients 幂等：清空缓存后再调无副作用。"""
    from src import observability as obs

    class _FakeClient:
        def __init__(self):
            self.flushed = 0

        def flush(self):
            self.flushed += 1

        def shutdown(self):
            pass

    fake = _FakeClient()
    with obs._LF_LOCK:
        obs._LF_CACHE.clear()
        obs._LF_CACHE[("pk", "sk", "")] = fake

    obs.shutdown_all_clients()
    assert fake.flushed == 1
    assert len(obs._LF_CACHE) == 0
    # 再次调用：缓存已空，不应再触碰任何客户端
    obs.shutdown_all_clients()
    assert fake.flushed == 1
