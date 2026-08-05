"""RAG 子包：向量化（embeddings）+ 向量库（store）+ 重排（rerank）。

设计原则：离线可跑。所有组件都有 mock 实现，无 API key 也能演示「向量召回 + 重排」全链路；
配置真实 key 后无缝切换到 OpenAI Embeddings / Cohere Rerank。
"""
