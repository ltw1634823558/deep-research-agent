"""MCP 子包：把外部搜索能力以 MCP 协议暴露/消费。

- server.py：MCP Server，把 Tavily 搜索封装为 MCP 工具（供 Agent 或其他 MCP Client 调用）
- client.py：MCP Client，经 stdio 拉起 Server 并调用其工具，结果转成项目 Source 列表
"""
