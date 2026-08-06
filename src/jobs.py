"""运行状态注册表：把每次研究运行登记为 Job，供 dashboard 实时展示。

设计要点（senior 视角）：
- 线程安全的 in-memory 存储，演示/单机部署足够；多副本需换 Redis/数据库。
- Job 记录完整的可观测切面：query/mode/status/阶段进度/子主题/来源/分析/报告/
  评估指标/错误/创建-更新时间/LangFuse 链接/配置快照。
- 阶段顺序 STAGES 同时驱动进度条与前端步进器，单一真源。
"""

from __future__ import annotations

import threading
import time
import uuid
from collections import OrderedDict
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any

# 阶段顺序：驱动进度（0-100）与前端步进器，单一真源。
STAGES = ["planning", "researching", "analyzing", "writing", "evaluating", "done"]

# 内存上限：只保留最近 N 个任务，超出时按插入顺序淘汰最旧的（防止无界增长）。
MAX_JOBS = 50


class JobStatus(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    DONE = "done"
    ERROR = "error"


@dataclass
class SubtopicView:
    id: str
    question: str
    status: str


@dataclass
class SourceView:
    url: str
    title: str
    snippet: str
    source_type: str


@dataclass
class Job:
    id: str
    query: str
    mode: str
    status: str = JobStatus.QUEUED.value
    stage_index: int = 0  # 对应 STAGES 下标
    progress: int = 0  # 0-100
    subtopics: list[SubtopicView] = field(default_factory=list)
    sources: list[SourceView] = field(default_factory=list)
    analysis: str = ""
    report: str = ""
    metrics: dict = field(default_factory=dict)
    error: str = ""
    langfuse_url: str = ""
    config: dict = field(default_factory=dict)
    # 长记忆切面
    memory_recall: list[str] = field(default_factory=list)
    memory_backend: str = ""
    memory_writes: int = 0
    # Analyst 自愈回路实际尝试次数
    analyst_self_heal: int = 0
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return asdict(self)


def _to_subtopics(plan: Any) -> list[SubtopicView]:
    out: list[SubtopicView] = []
    for s in plan or []:
        if isinstance(s, dict):
            out.append(
                SubtopicView(
                    id=str(s.get("id", "")),
                    question=str(s.get("question", "")),
                    status=str(s.get("status", "pending")),
                )
            )
        else:
            out.append(
                SubtopicView(
                    id=str(getattr(s, "id", "")),
                    question=str(getattr(s, "question", "")),
                    status=str(getattr(s, "status", "pending")),
                )
            )
    return out


def _to_sources(sources: Any) -> list[SourceView]:
    out: list[SourceView] = []
    for s in sources or []:
        if isinstance(s, dict):
            out.append(
                SourceView(
                    url=str(s.get("url", "")),
                    title=str(s.get("title", "")),
                    snippet=str(s.get("snippet", "")),
                    source_type=str(s.get("source_type", "web")),
                )
            )
        else:
            out.append(
                SourceView(
                    url=str(getattr(s, "url", "")),
                    title=str(getattr(s, "title", "")),
                    snippet=str(getattr(s, "snippet", "")),
                    source_type=str(getattr(s, "source_type", "web")),
                )
            )
    return out


class JobStore:
    """线程安全的作业注册表（进程内单例，容量受限的 LRU）。"""

    def __init__(self, max_jobs: int = MAX_JOBS) -> None:
        self._jobs: OrderedDict[str, Job] = OrderedDict()
        self._max_jobs = max_jobs
        self._lock = threading.Lock()

    def _evict_locked(self) -> None:
        """在持锁状态下淘汰最旧的任务，保持容量不超过 _max_jobs。"""
        while len(self._jobs) > self._max_jobs:
            self._jobs.popitem(last=False)

    def create(self, query: str, mode: str, config: dict | None = None) -> Job:
        job = Job(id=uuid.uuid4().hex[:12], query=query, mode=mode, config=config or {})
        with self._lock:
            self._jobs[job.id] = job
            self._jobs.move_to_end(job.id)
            self._evict_locked()
        return job

    def get(self, job_id: str) -> Job | None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is not None:
                self._jobs.move_to_end(job_id)
            return job

    def snapshot(self, job_id: str) -> dict | None:
        """持锁生成任务快照，避免读到写线程的中间状态。"""
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return None
            self._jobs.move_to_end(job_id)
            return job.to_dict()

    def list(self) -> list[Job]:
        with self._lock:
            return sorted(self._jobs.values(), key=lambda j: j.created_at, reverse=True)

    def update(self, job_id: str, **fields: Any) -> None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return
            for k, v in fields.items():
                setattr(job, k, v)
            job.updated_at = time.time()
            self._jobs.move_to_end(job_id)

    def set_stage(self, job_id: str, stage_index: int) -> None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return
            job.stage_index = stage_index
            total = len(STAGES) - 1
            job.progress = int(round(stage_index / total * 100))
            job.updated_at = time.time()
            self._jobs.move_to_end(job_id)


# 进程内单例
store = JobStore()
