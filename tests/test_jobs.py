"""JobStore 单元测试：LRU 容量淘汰、快照锁内序列化、与全局单例解耦。

使用独立的 JobStore 实例（不碰进程级 `store` 单例），保证测试密闭、可重复，
不依赖全局作业计数，也不受其他测试写入的影响。
"""
from __future__ import annotations

from src.jobs import JobStore


def test_create_and_snapshot_roundtrip():
    store = JobStore(max_jobs=10)
    job = store.create("hello", "web")
    snap = store.snapshot(job.id)
    assert snap is not None
    assert snap["id"] == job.id
    assert snap["query"] == "hello"
    assert snap["status"] == "queued"


def test_lru_evicts_oldest_when_over_capacity():
    store = JobStore(max_jobs=3)
    j1 = store.create("q1", "web")
    j2 = store.create("q2", "web")
    j3 = store.create("q3", "web")
    assert len(store.list_snapshots()) == 3

    # 第 4 个超过容量，最旧（j1）应被淘汰
    j4 = store.create("q4", "web")
    assert len(store.list_snapshots()) == 3
    assert store.snapshot(j1.id) is None  # 最旧已淘汰
    assert store.snapshot(j4.id) is not None


def test_recently_used_moves_to_end_and_survives():
    store = JobStore(max_jobs=2)
    j1 = store.create("q1", "web")
    j2 = store.create("q2", "web")
    # 访问 j1，把它移到末尾（最近使用），再 create 时淘汰的是 j2
    store.get(j1.id)
    j3 = store.create("q3", "web")
    assert store.snapshot(j1.id) is not None  # j1 存活
    assert store.snapshot(j2.id) is None      # j2 被淘汰


def test_list_snapshots_is_locked_serialization():
    store = JobStore()
    for i in range(7):
        store.create(f"q{i}", "web")
    snaps = store.list_snapshots()
    assert len(snaps) == 7
    # 按 created_at 倒序
    times = [s["created_at"] for s in snaps]
    assert times == sorted(times, reverse=True)
    assert all("id" in s and "config" in s for s in snaps)


def test_update_sets_fields_and_moves_to_end():
    store = JobStore(max_jobs=5)
    j = store.create("q", "web")
    store.update(j.id, status="done", progress=100, report="最终报告")
    snap = store.snapshot(j.id)
    assert snap["status"] == "done"
    assert snap["progress"] == 100
    assert snap["report"] == "最终报告"
