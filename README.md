# KnowAgent 基于LangGraph的RAG‑Agent Demo
> ⚠️ 本仓库为学习演示项目，**并非开箱即用可部署版本**。
> 仅存放核心业务逻辑代码，剥离本地环境配置、密钥、完整web服务。
> 完整系统在本地开发环境完成验证

## 技术栈
Python、LangGraph、LangChain、Milvus、Qdrant、MCP协议

## 主要实现
1. 基于StateGraph构建双Agent工作流，实现文档导入、知识库问答
2. TypedDict状态管理 + Reducer处理并发状态覆盖
3. 向量检索 + HyDE + MCP联网多路召回，RRF融合排序
4. 多轮对话实体消歧，条件边动态路由

## 关键模块
- graph/: Agent状态机、节点定义
- retriever/: 多路召回、RRF融合实现
