from langgraph.graph import StateGraph
from app.query_process.agent.state import QueryGraphState
# 导入所有节点函数
from app.query_process.agent.nodes.node_answer_output import node_answer_output
from app.query_process.agent.nodes.node_item_name_confirm import node_item_name_confirm
# from app.query_process.agent.nodes.node_query_kg import node_query_kg
from app.query_process.agent.nodes.node_rerank import node_rerank
from app.query_process.agent.nodes.node_rrf import node_rrf
from app.query_process.agent.nodes.node_search_embedding import node_search_embedding
from app.query_process.agent.nodes.node_search_embedding_hyde import node_search_embedding_hyde
from app.query_process.agent.nodes.node_web_search_mcp import node_web_search_mcp

# 初始化状态图
builder = StateGraph(QueryGraphState)
builder.add_node("node_item_name_confirm", node_item_name_confirm)
builder.add_node("node_search_embedding", node_search_embedding)
builder.add_node("node_search_embedding_hyde", node_search_embedding_hyde)
builder.add_node("node_rrf", node_rrf)
builder.add_node("node_rerank", node_rerank)
builder.add_node("node_answer_output", node_answer_output)
builder.add_node("node_web_search_mcp", node_web_search_mcp)

# 添加边
builder.set_entry_point("node_item_name_confirm")
# node_item_name_confirm可能没有明确的主体 item_name 要提前结束 -》END
# node_item_name_confirm ->(answer: str # 最终生成的答案)->答案生成 给前端反馈 || 多路召回
# 条件边
# 路由
def route_after_node_item_name_confirm(state: QueryGraphState):
    if state['answer']:
        return "node_answer_output"
    return "node_search_embedding", "node_search_embedding_hyde", "node_web_search_mcp"

builder.add_conditional_edges("node_item_name_confirm",
                              route_after_node_item_name_confirm,
                              path_map={
                                  "node_search_embedding":"node_search_embedding",
                                  "node_search_embedding_hyde":"node_search_embedding_hyde",
                                  "node_web_search_mcp":"node_web_search_mcp"
                              })

builder.add_edge("node_search_embedding", "node_rrf")
builder.add_edge("node_search_embedding_hyde", "node_rrf")
builder.add_edge("node_web_search_mcp", "node_rrf")
builder.add_edge("node_rrf", "node_rerank")
builder.add_edge("node_rerank", "node_answer_output")
builder.set_finish_point("node_answer_output")
query_app = builder.compile()
