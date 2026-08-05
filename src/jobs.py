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
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any

# 阶段顺序：驱动进度（0-100）与前端步进器，单一真源。
STAGES = ["planning", "researching", "analyzing", "writing", "evaluating", "done"]


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
    """线程安全的作业注册表（进程内单例）。"""

    def __init__(self) -> None:
        self._jobs: dict[str, Job] = {}
        self._lock = threading.Lock()

    def create(self, query: str, mode: str, config: dict | None = None) -> Job:
        job = Job(id=uuid.uuid4().hex[:12], query=query, mode=mode, config=config or {})
        with self._lock:
            self._jobs[job.id] = job
        return job

    def get(self, job_id: str) -> Job | None:
        with self._lock:
            return self._jobs.get(job_id)

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

    def set_stage(self, job_id: str, stage_index: int) -> None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return
            job.stage_index = stage_index
            total = len(STAGES) - 1
            job.progress = int(round(stage_index / total * 100))
            job.updated_at = time.time()


# 进程内单例
store = JobStore()
