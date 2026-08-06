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
