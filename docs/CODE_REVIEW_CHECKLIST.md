# 代码审查清单（团队 Code Review 标准）

> 资深开发视角：PR 合并前必须逐条过一遍。这不是形式主义，而是把"能跑"和"能维护、能交接"区分开。
> 每条都对应我们项目里真实踩过或容易踩的坑。

## 1. 正确性（Blocking）
- [ ] **不变量成立**：LangGraph 节点的返回值只包含 state 中声明的键；带 reducer 的字段（findings/sources/messages）是"追加"语义，不要用整体赋值覆盖。
- [ ] **路由确定可测**：条件边的路由函数（route_after_research 等）是纯函数，不依赖隐藏状态，有单测覆盖。
- [ ] **终止有界**：任何循环/回环都有迭代上限兜底（max_iterations），不存在死循环路径。
- [ ] **降级路径验证过**：工具调用（检索/抓取/Obsidian）在无 key、无网络、路径无效时都有 mock/合成降级，且降级分支被测试覆盖。

## 2. 可观测与排错（Blocking）
- [ ] 每个节点调用 LLM 都注入了 `config={"callbacks": callbacks}`，LangFuse 能追踪到。
- [ ] 新增外部调用（API/文件 IO）都有 try/except 降级，且失败不抛未捕获异常阻断整条管线。
- [ ] 日志/消息能定位到具体子主题与来源类型（web vs local）。

## 3. 类型与接口契约
- [ ] 所有跨模块数据结构用 Pydantic（Source/Finding/Subtopic），不在 dict 里塞魔法字符串。
- [ ] 新增 Source 必须带 `source_type`（`web`/`local`），下游据此决定读取方式（fetch_url vs 直接读 snippet）。
- [ ] 公共函数有类型注解；`mypy` 门禁通过。

## 4. 依赖与可复现
- [ ] 任何 `import` 的第三方包都写进了 `requirements.txt`（血的教训：本项目曾漏写 `requests`，靠传递依赖侥幸能跑）。
- [ ] 不引入重依赖做轻量活：解析 frontmatter 用正则而非拉 yaml；离线场景零外部服务依赖。
- [ ] `LLM_PROVIDER=mock` 下 `python tests/test_mock_run.py` 全绿，保证 CI 不依赖密钥。

## 5. 安全（Blocking）
- [ ] 不把 API key 提交进仓库；统一走 `.env` + `.env.example` 模板。
- [ ] 读取本地文件路径（如 Obsidian 仓库）仅限配置显式指定的目录，不做任意路径遍历。
- [ ] 提示词不拼接未转义的外部输入到可执行上下文（本项目的 prompt 仅做文本引用，安全）。

## 6. 可维护
- [ ] 函数单一职责，单个节点不超过 ~60 行；超长的拆 helper。
- [ ] 命名表意：节点用 `*_node`，工具用动词短语（web_search / obsidian_search）。
- [ ] 关键设计决策写进 `docs/ARCHITECTURE_DECISIONS.md`，新人能看懂"为什么这样写"。

## Review 礼仪（团队文化）
- 先肯定再提问题；对事不对人。
- "为什么不用 X？" 优于 "你这写错了"。
-  reviewer 必须本地跑一遍 mock 测试再 approve。
