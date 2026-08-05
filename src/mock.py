"""离线 Mock 大模型：无需任何 API key 即可跑通多智能体管线，用于演示 / 测试 / CI。

真实接入时把 .env 里的 LLM_PROVIDER 改成 openai 并填 key 即可，无需改动业务代码。
"""

from __future__ import annotations

import json

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult


class MockChatModel(BaseChatModel):
    @property
    def _llm_type(self) -> str:
        return "mock"

    def _generate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:
        text = " ".join(getattr(m, "content", "") or "" for m in messages)
        return ChatResult(
            generations=[ChatGeneration(message=AIMessage(content=self._respond(text)))]
        )

    def _respond(self, text: str) -> str:
        t = text.lower()
        # 规划阶段：返回子主题列表（JSON 数组）。用"规划"等专属词，
        # 避免被 researcher/writer 摘要里的"子主题"误命中
        if "规划" in text or "subtopic" in t or ("plan" in t and "research" in t):
            return json.dumps(
                ["背景与定义", "核心技术架构", "行业应用与案例", "挑战与未来趋势"],
                ensure_ascii=False,
            )
        # 分析 / 核验阶段（用具体词，避免被 writer 的"研究发现"误命中）
        if "核验" in text or "缺口" in text or "contradiction" in t or "gap" in t:
            return (
                "分析：现有发现内部一致，未发现明显矛盾。"
                "证据覆盖背景、技术、应用、挑战四个维度，可进入撰写阶段。"
            )
        # 撰写阶段
        if "报告" in text or "report" in t or "撰写" in text:
            return (
                "# 深度研究报告（Mock 模式）\n\n"
                "本报告由多智能体管线自动生成，当前为离线 Mock 模式，未调用真实大模型。\n\n"
                "## 核心要点\n"
                "- 主题背景已梳理，概念边界清晰\n"
                "- 技术架构与关键组件已拆解\n"
                "- 已有行业落地案例支撑论点\n"
                "- 主要挑战集中在可靠性、成本与评估体系\n\n"
                "> 配置 OPENAI_API_KEY 并设 LLM_PROVIDER=openai 后，本产出来自真实模型。\n"
            )
        # 默认：单子主题的研究摘要
        return (
            "基于检索结果，该子主题的关键信息如下（mock 摘要）："
            "相关技术正在快速演进，已有多个生产级落地案例，"
            "当前主要瓶颈在可靠性保障与推理成本控制。"
        )
