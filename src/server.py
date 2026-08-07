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
import logging
import threading
from contextlib import asynccontextmanager
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
from .observability import get_callbacks, shutdown_all_clients
from .state import ResearchState, _add_sources, initial_state

logger = logging.getLogger(__name__)

# 收到 SIGTERM 后留给在途研究任务的收尾时间。设得比编排层的
# terminationGracePeriod（K8s 默认 30s）短，确保我们能在被 SIGKILL 前
# 主动 flush 追踪数据，而不是把宽限期全耗在等 job 上。
_SHUTDOWN_GRACE = 20.0


def _shutdown_resources() -> None:
    """进程退出前有序释放资源。任一步失败都不阻断后续步骤。

    顺序不可调换：先让在途 job 收尾（它们仍在产生 span），再 flush 追踪，
    最后才关长记忆句柄——反过来会让最后一批 trace 和记忆写入一起丢。
    """
    # 1) 停止接新活并取消排队任务，给在途 job 一个「有界」的完成窗口。
    #    ThreadPoolExecutor.shutdown 不支持 timeout，故用 daemon 线程 join 限时；
    #    否则一个跑几分钟的 job 会拖到编排层 SIGKILL，数据丢得更彻底。
    try:
        _JOB_EXECUTOR.shutdown(wait=False, cancel_futures=True)
        waiter = threading.Thread(
            target=_JOB_EXECUTOR.shutdown, kwargs={"wait": True}, daemon=True
        )
        waiter.start()
        waiter.join(_SHUTDOWN_GRACE)
        if waiter.is_alive():
            logger.warning(
                "后台研究任务未在 %.0fs 内收尾，继续关闭其余资源", _SHUTDOWN_GRACE
            )
    except Exception:
        logger.exception("关闭后台任务线程池失败")

    # 2) flush 追踪：LangFuse 的 OTel 批处理器攒批异步上报，不 flush 就会丢最后一批
    try:
        shutdown_all_clients()
    except Exception:
        logger.exception("关闭 LangFuse 客户端失败")

    # 3) 收长记忆：SQLite 连接与 Chroma 索引句柄（Windows 下不关会锁住文件）
    try:
        memory_store.close()
    except Exception:
        logger.exception("关闭长记忆存储失败")


@asynccontextmanager
async def _lifespan(_app: FastAPI):
    yield
    _shutdown_resources()


app = FastAPI(title="Deep Research Agent API", version="1.0.0", lifespan=_lifespan)
graph = build_graph()

# 启动预热：首次访问长记忆索引会触发最长 6s 的 Chroma 探测，提前在导入期完成，
# 避免首个请求线程被阻塞（使用独立 index 锁，不阻塞 SQLite 读写，见 L-3/L-4）。
try:
    _ = memory_store.index.backend
except Exception:
    pass

# 后台任务并发上限：避免每个请求裸起线程压垮进程 / 打爆下游 API 配额。
_JOB_WORKERS = 4
_JOB_EXECUTOR = concurrent.futures.ThreadPoolExecutor(
    max_workers=_JOB_WORKERS, thread_name_prefix="research-job"
)
# ThreadPoolExecutor 的队列是无界的：密集 POST /research/job 会让排队闭包无限堆积，
# 而 JobStore 只留最近 50 条，早期 job 记录已被 LRU 淘汰、结果无处可查（静默丢失）。
# 这里显式限制在途任务数，超出直接 429，让调用方可感知并退避。
_JOB_QUEUE_LIMIT = 32
_inflight_jobs = 0
_inflight_lock = threading.Lock()


def _release_job_slot(_fut=None) -> None:
    global _inflight_jobs
    with _inflight_lock:
        _inflight_jobs = max(0, _inflight_jobs - 1)


def _acquire_job_slot() -> bool:
    global _inflight_jobs
    with _inflight_lock:
        if _inflight_jobs >= _JOB_QUEUE_LIMIT:
            return False
        _inflight_jobs += 1
        return True

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


def _public_error(exc: BaseException) -> str:
    """对外错误摘要：只回异常类型 + 短句，绝不回原始文本。

    `GET /api/jobs/{id}` 目前无鉴权，异常原文常带绝对路径、上游 API 的
    key 相关提示、内网地址等，直接回显等于信息泄露。全文只进服务端日志。
    """
    name = type(exc).__name__
    known = {
        "RuntimeError": "研究流程执行失败",
        "TimeoutError": "外部依赖响应超时",
        "ValueError": "输入或配置不合法",
        "ConnectionError": "外部服务连接失败",
    }
    return f"{known.get(name, '研究任务执行异常')}（{name}），详情见服务端日志"


def _config_snapshot(cfg: "Settings | None" = None) -> dict:
    s = cfg or settings
    return {
        "llm_provider": s.llm_provider,
        "search_provider": s.search_provider,
        "embedding_provider": s.embedding_provider,
        "rerank_provider": s.rerank_provider,
        "rag_backend": s.rag_backend,
        "rag_top_k": s.rag_top_k,
        "research_mode": s.research_mode,
        "memory_enabled": s.memory_enabled,
        "memory_backend": memory_store.index.backend,
        "memory_top_k": s.memory_top_k,
        "analyst_self_heal": s.analyst_self_heal,
        "langfuse": bool(s.langfuse_public_key and s.langfuse_secret_key),
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
    cfg: "Settings | None" = None,
) -> None:
    """计算评估指标并落库，标记任务完成。

    `cfg` 为按请求注入的配置；缺省回落到全局 `settings`，保证与 `_run_job` 中
    `graph.stream` 使用的是同一份参数（否则异步 job 的 max_iterations / langfuse_url
    会读到全局值，多租户覆写失效）。
    """
    s = cfg or settings
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
            "max_iterations": s.max_research_iterations,
            "mode": mode,
            "messages": [],
            "memory_recall": memory_recall or [],
            "analyst_self_heal": analyst_self_heal,
            "memory_writes": memory_writes,
        },
    )
    metrics = evaluate(state, s).to_dict()
    langfuse_url = s.langfuse_host if (s.langfuse_public_key and s.langfuse_secret_key) else ""
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
        job_callbacks = get_callbacks(cfg)

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
            cfg=cfg,
        )
    except Exception as exc:  # 不吞掉：全文进服务端日志，对外只暴露脱敏摘要
        logger.exception("研究任务 %s 执行失败", job_id)
        store.update(job_id, status=JobStatus.ERROR.value, error=_public_error(exc))


@app.get("/health")
def health():
    return {"status": "ok", "provider": settings.llm_provider, "mode": settings.research_mode}


@app.post("/research", response_model=ResearchResponse)
def research(req: ResearchRequest, cfg: Settings = Depends(get_settings)):
    state = initial_state(req.query, cfg.max_research_iterations, mode=req.mode)
    # 同步运行也取独立回调，保证 trace 不串扰（L-F）
    result = graph.invoke(state, config={"configurable": {"settings": cfg}, "callbacks": get_callbacks(cfg)})
    metrics = evaluate(result, cfg).to_dict()

    # 同步运行也记入 JobStore，供 dashboard 看历史
    job = store.create(req.query, req.mode, _config_snapshot(cfg))
    langfuse_url = cfg.langfuse_host if (cfg.langfuse_public_key and cfg.langfuse_secret_key) else ""
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
    if not _acquire_job_slot():
        raise HTTPException(
            status_code=429,
            detail=f"研究任务排队已满（在途上限 {_JOB_QUEUE_LIMIT}），请稍后重试",
        )
    try:
        job = store.create(req.query, req.mode, _config_snapshot(cfg))
        future = _JOB_EXECUTOR.submit(_run_job, job.id, req.query, req.mode, cfg)
    except Exception:
        _release_job_slot()
        raise
    future.add_done_callback(_release_job_slot)
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
