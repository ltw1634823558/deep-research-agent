"""FastAPI 服务：把研究 Agent 暴露为 HTTP 接口（生产级落地的工程化一环）。

路由总览：
- GET  /health            健康检查（provider / mode 快照）
- POST /research          同步研究，返回 报告/分析/来源/指标（向后兼容，无中断）
- POST /research/job      异步研究：后台线程流式运行，返回 job_id 供 dashboard 实时追踪
- GET  /api/jobs          任务列表
- GET  /api/jobs/{id}     单任务详情（dashboard 轮询）
- GET  /  , /dashboard    实时运行状态面板（零依赖自包含 HTML）
"""
from __future__ import annotations

import concurrent.futures
from pathlib import Path
from typing import Literal, cast

try:
    from dotenv import load_dotenv

    load_dotenv(override=True)  # .env 优先于已存在的 Shell 环境变量（避免残留 key 遮蔽）
except Exception:
    pass

from fastapi import Depends, FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from .config import Settings, get_settings, settings
from .evaluation import evaluate
from .graph import build_graph
from .jobs import STAGES, JobStatus, _to_sources, _to_subtopics, store
from .memory import memory_store
from .observability import get_callbacks
from .state import ResearchState, _add_sources, initial_state

app = FastAPI(title="Deep Research Agent API", version="1.0.0")
graph = build_graph()

# 后台任务并发上限：避免每个请求裸起线程压垮进程 / 打爆下游 API 配额。
_JOB_EXECUTOR = concurrent.futures.ThreadPoolExecutor(
    max_workers=4, thread_name_prefix="research-job"
)

_DASHBOARD_HTML = (
    Path(__file__).parent / "templates" / "dashboard.html"
).read_text(encoding="utf-8")


class ResearchRequest(BaseModel):
    query: str = Field(max_length=2000)
    mode: Literal["web", "local", "hybrid"] = "web"


class ResearchResponse(BaseModel):
    report: str
    analysis: str
    mode: str
    sources: list[dict]
    metrics: dict = {}


# 节点 -> 阶段下标（见 jobs.STAGES）
_NODE_TO_STAGE = {"planner": 1, "researcher": 1, "analyst": 2, "writer": 3}


def _config_snapshot() -> dict:
    return {
        "llm_provider": settings.llm_provider,
        "search_provider": settings.search_provider,
        "embedding_provider": settings.embedding_provider,
        "rerank_provider": settings.rerank_provider,
        "rag_backend": settings.rag_backend,
        "rag_top_k": settings.rag_top_k,
        "research_mode": settings.research_mode,
        "memory_enabled": settings.memory_enabled,
        "memory_backend": memory_store.index.backend,
        "memory_top_k": settings.memory_top_k,
        "analyst_self_heal": settings.analyst_self_heal,
        "langfuse": bool(settings.langfuse_public_key and settings.langfuse_secret_key),
    }


def _finalize_job(
    job_id: str,
    query: str,
    mode: str,
    sources,
    plan,
    analysis,
    report,
    memory_recall: list[str] | None = None,
    memory_writes: int = 0,
    analyst_self_heal: int = 0,
) -> None:
    """计算评估指标并落库，标记任务完成。"""
    state = cast(
        ResearchState,
        {
            "query": query,
            "plan": plan,
            "findings": [],
            "analysis": analysis,
            "report": report,
            "sources": sources,
            "iteration": 0,
            "max_iterations": settings.max_research_iterations,
            "mode": mode,
            "messages": [],
            "memory_recall": memory_recall or [],
            "analyst_self_heal": analyst_self_heal,
            "memory_writes": memory_writes,
        },
    )
    metrics = evaluate(state).to_dict()
    langfuse_url = settings.langfuse_host if (settings.langfuse_public_key and settings.langfuse_secret_key) else ""
    store.update(
        job_id,
        status=JobStatus.DONE.value,
        stage_index=len(STAGES) - 1,
        progress=100,
        subtopics=_to_subtopics(plan),
        sources=_to_sources(sources),
        analysis=analysis,
        report=report,
        metrics=metrics,
        memory_recall=memory_recall or [],
        memory_backend=memory_store.index.backend,
        memory_writes=memory_writes,
        analyst_self_heal=analyst_self_heal,
        langfuse_url=langfuse_url,
    )


def _run_job(job_id: str, query: str, mode: str, cfg: Settings) -> None:
    """后台线程：用 graph.stream 实时推进 JobStore 状态。

    `cfg` 为按请求注入的配置（经 FastAPI `Depends(get_settings)` 取得），透传进
    `graph.stream` 的 `configurable['settings']`，节点内通过 `resolve_settings(config)` 取用，
    从而支持同一份代码按请求覆写参数（多租户 / A/B 实验）。
    """
    try:
        store.update(job_id, status=JobStatus.RUNNING.value, stage_index=0)
        inputs = initial_state(query, cfg.max_research_iterations, mode=mode)

        # 本地累加器：'updates' 模式只给增量，需要自己汇总
        cur_plan: list = []
        cur_sources: list = []
        cur_analysis = ""
        cur_report = ""
        cur_memory_recall: list[str] = []
        cur_self_heal = 0
        cur_memory_writes = 0

        # 每个任务取一份独立回调，避免并发线程共享 handler 造成 trace 串扰
        job_callbacks = get_callbacks()

        for chunk in graph.stream(
            inputs,
            stream_mode="updates",
            config={"configurable": {"settings": cfg}, "callbacks": job_callbacks},
        ):
            node = next(iter(chunk.keys()))
            patch = chunk[node] or {}
            if node in _NODE_TO_STAGE:
                store.set_stage(job_id, _NODE_TO_STAGE[node])
            if isinstance(patch, dict):
                if "plan" in patch:
                    cur_plan = patch["plan"]
                    store.update(job_id, subtopics=_to_subtopics(cur_plan))
                if "sources" in patch:
                    # 必须复用 state 的 _add_sources reducer：直接 + 拼接会绕过按 URL 去重，
                    # 导致 dashboard 来源重复膨胀、评估分母失真。
                    cur_sources = _add_sources(cur_sources, list(patch["sources"]))
                    store.update(job_id, sources=_to_sources(cur_sources))
                if "memory_recall" in patch:
                    cur_memory_recall = cur_memory_recall + list(patch["memory_recall"])
                    store.update(job_id, memory_recall=cur_memory_recall)
                if "analyst_self_heal" in patch:
                    cur_self_heal = patch["analyst_self_heal"]
                    store.update(job_id, analyst_self_heal=cur_self_heal)
                if "memory_writes" in patch:
                    cur_memory_writes = patch["memory_writes"]
                    store.update(job_id, memory_writes=cur_memory_writes)
                if "analysis" in patch:
                    cur_analysis = patch["analysis"]
                if "report" in patch:
                    cur_report = patch["report"]

        store.set_stage(job_id, 4)  # evaluating
        _finalize_job(
            job_id,
            query,
            mode,
            cur_sources,
            cur_plan,
            cur_analysis,
            cur_report,
            memory_recall=cur_memory_recall,
            memory_writes=cur_memory_writes,
            analyst_self_heal=cur_self_heal,
        )
    except Exception as exc:  # 不吞掉，记录到任务以便前端展示
        store.update(job_id, status=JobStatus.ERROR.value, error=str(exc)[:500])


@app.get("/health")
def health():
    return {"status": "ok", "provider": settings.llm_provider, "mode": settings.research_mode}


@app.post("/research", response_model=ResearchResponse)
def research(req: ResearchRequest, cfg: Settings = Depends(get_settings)):
    state = initial_state(req.query, cfg.max_research_iterations, mode=req.mode)
    result = graph.invoke(state, config={"configurable": {"settings": cfg}})
    metrics = evaluate(result).to_dict()

    # 同步运行也记入 JobStore，供 dashboard 看历史
    job = store.create(req.query, req.mode, _config_snapshot())
    langfuse_url = settings.langfuse_host if (settings.langfuse_public_key and settings.langfuse_secret_key) else ""
    store.update(
        job.id,
        status=JobStatus.DONE.value,
        stage_index=len(STAGES) - 1,
        progress=100,
        subtopics=_to_subtopics(result.get("plan")),
        sources=_to_sources(result.get("sources")),
        analysis=result.get("analysis", ""),
        report=result.get("report", ""),
        metrics=metrics,
        memory_recall=result.get("memory_recall", []) or [],
        memory_backend=memory_store.index.backend,
        memory_writes=result.get("memory_writes", 0) or 0,
        analyst_self_heal=result.get("analyst_self_heal", 0) or 0,
        langfuse_url=langfuse_url,
    )

    return ResearchResponse(
        report=result.get("report", ""),
        analysis=result.get("analysis", ""),
        mode=req.mode,
        sources=[s.model_dump() for s in result.get("sources", [])],
        metrics=metrics,
    )


@app.post("/research/job")
def research_job(req: ResearchRequest, cfg: Settings = Depends(get_settings)):
    job = store.create(req.query, req.mode, _config_snapshot())
    _JOB_EXECUTOR.submit(_run_job, job.id, req.query, req.mode, cfg)
    return {"job_id": job.id, "dashboard_url": f"/dashboard?job={job.id}"}


@app.get("/api/jobs")
def list_jobs():
    # 轻量 DTO：dashboard 高频轮询，不回传 report/analysis/sources 等大字段。
    # 序列化在 store 内部持锁完成（list_snapshots），避免后台线程写入时的撕裂读。
    return [
        {
            "id": j["id"],
            "query": j["query"],
            "mode": j["mode"],
            "status": j["status"],
            "stage_index": j["stage_index"],
            "progress": j["progress"],
            "created_at": j["created_at"],
            "updated_at": j["updated_at"],
            "error": j["error"],
            "source_count": len(j["sources"]),
            "metrics": {
                k: j["metrics"].get(k)
                for k in (
                    "task_completion_rate",
                    "citation_accuracy",
                    "hallucination_rate",
                    "citation_recall",
                    "sources_total",
                )
                if k in j["metrics"]
            },
        }
        for j in store.list_snapshots()
    ]


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str):
    snap = store.snapshot(job_id)
    if snap is None:
        raise HTTPException(status_code=404, detail="job not found")
    return snap


@app.get("/", response_class=HTMLResponse)
@app.get("/dashboard", response_class=HTMLResponse)
def dashboard():
    return HTMLResponse(_DASHBOARD_HTML)
