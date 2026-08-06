"""Analyst：交叉核验各子主题发现，检测矛盾 / 缺口，必要时追加子主题进入下一轮检索。

增强（附录自「真实 LLM 分析层自愈回路」）：
- 仅当使用真实 LLM（`LLM_PROVIDER=openai`）时，进入「自我批判 → 修复」内部循环：
  每次产出分析后由 `_critique_analysis` 评估覆盖度 / 置信度 / 长度，不过关则把问题回灌模型修订，
  最多尝试 `analyst_self_heal` 次；仍判缺口则追加补充子主题，触发已有的 analyst→researcher 回环。
- mock 模式不进入循环（保持确定性、零成本），仅做单轮核验，行为向后兼容。
"""

from __future__ import annotations

import re

from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableConfig

from ..config import get_llm, resolve_settings
from ..state import Finding, ResearchState, Subtopic

# 低置信 / 不确定表述标记（命中即视为分析不可靠，需要修补或补充检索）
# 中文按子串匹配（无词边界概念）；英文必须整词匹配，否则 "Singapore" 会被 "gap" 子串误判。
_LOW_CONF_CN = ["不确定", "不清楚", "缺乏", "不足", "无法判断", "需要更多", "数据不足"]
_GAP_RE = re.compile(r"\b(gap|gaps|uncertain|lacks|incomplete)\b", re.IGNORECASE)
_EN_LOW_CONF_RE = re.compile(r"\b(unknown|insufficient)\b", re.IGNORECASE)


def _critique_analysis(findings: list[Finding], analysis: str) -> tuple[bool, list[str]]:
    """纯函数：评估分析质量，返回 (是否通过, 问题列表)。

    - 覆盖度：每个发现都应在分析中被引用（按 subtopic_id 或摘要片段匹配）；
    - 置信度：不得含低置信/不确定表述；
    - 长度：过短视为未充分核验。
    """
    issues: list[str] = []
    if findings:
        covered = 0
        for f in findings:
            if f.subtopic_id and f.subtopic_id in analysis:
                covered += 1
                continue
            frag = f.summary[:12]
            if frag and frag in analysis:
                covered += 1
        cov = covered / len(findings)
        if cov < 0.8:
            issues.append(f"覆盖不足：仅 {covered}/{len(findings)} 个子主题发现被分析引用")
    low = [w for w in _LOW_CONF_CN if w in analysis]
    low += [m.group(0).lower() for m in _GAP_RE.finditer(analysis)]
    low += [m.group(0).lower() for m in _EN_LOW_CONF_RE.finditer(analysis)]
    low = list(dict.fromkeys(low))
    if low:
        issues.append("分析含低置信/不确定表述：" + "/".join(low))
    if len(analysis) < 40:
        issues.append("分析过短，可能未充分核验")
    return (len(issues) == 0, issues)


def analyst_node(state: ResearchState, config: RunnableConfig) -> dict:
    cfg = resolve_settings(config)
    llm = get_llm(cfg)
    findings_text = "\n".join(f"[{f.subtopic_id}] {f.summary}" for f in state["findings"])
    base_prompt = (
        "你是一个研究核验智能体。请分析以下各子主题的发现，"
        "判断是否一致、是否有明显缺口需要补充检索。\n"
        f"发现：\n{findings_text}"
    )
    # 直接透传节点自身的 RunnableConfig：其中已带本次任务专属的 callbacks，
    # 不能再用模块级全局 callbacks 覆盖，否则并发任务的 trace 会串到同一个 handler。
    analysis = llm.invoke(base_prompt, config=config).content
    self_heal = 0
    ok = True  # 最近一次批判结论（mock 模式不批判，视为通过）

    # 仅真实 LLM 进入自愈循环：自我批判 → 修订，最多 analyst_self_heal 次
    if cfg.llm_provider == "openai":
        max_attempts = max(1, cfg.analyst_self_heal)
        for attempt in range(1, max_attempts + 1):
            ok, issues = _critique_analysis(state["findings"], analysis)
            if ok:
                break
            self_heal = attempt
            critique = "你的分析存在以下问题，请修订：\n- " + "\n- ".join(issues)
            try:
                analysis = llm.invoke(
                    base_prompt + "\n\n" + critique,
                    config=config,
                ).content
            except Exception:
                break  # 调用失败则保留上一版，不中断管线

    # 是否触发补充检索回环：缺口标记（向后兼容）或 真实 LLM 下批判未通过
    needs_more = ("缺口" in analysis) or bool(_GAP_RE.search(analysis))
    if cfg.llm_provider == "openai":
        # 复用循环内已算出的批判结论，无需重复评估
        if not ok and state["iteration"] < state["max_iterations"]:
            needs_more = True

    extra: list[Subtopic] = []
    if needs_more and state["iteration"] < state["max_iterations"]:
        extra = [
            Subtopic(
                id=f"g{state['iteration'] + 1}",
                question=f"针对「{state['query']}」的缺口做补充检索",
            )
        ]

    return {
        "analysis": analysis,
        "plan": state["plan"] + extra,
        "iteration": state["iteration"] + 1,
        "analyst_self_heal": self_heal,
        "messages": [
            AIMessage(
                content="[Analyst] 完成交叉核验"
                + (f"（自愈 {self_heal} 次）" if self_heal else "")
            )
        ],
    }
