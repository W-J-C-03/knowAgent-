from typing_extensions import TypedDict
from typing import List
import copy
from app.core.logger import logger


class QueryGraphState(TypedDict):
    """
    QueryGraphState 定义了整个查询流程中流转的数据结构。
    TypedDict 让我们在代码中能有自动补全和类型检查。
    使用字典式访问（如 state["session_id"]、state.get("embedding_chunks")）
    """
    session_id: str          # 会话唯一标识
    original_query: str      # 用户原始问题

    # --- 检索过程中的中间数据 ---
    embedding_chunks: list        # 普通向量检索回来的切片
    hyde_embedding_chunks: list   # HyDE 检索回来的切片
    kg_chunks: list               # 图谱检索回来的切片
    web_search_docs: list         # 网络搜索回来的文档

    # --- 排序过程中的数据 ---
    rrf_chunks: list              # RRF 融合排序后的切片
    reranked_docs: list           # 重排序后的最终 Top-K 文档

    # --- 生成过程中的数据 ---
    prompt: str                   # 组装好的 Prompt
    answer: str                   # 最终生成的答案

    # --- 辅助信息 ---
    item_names: List[str]         # 提取出的商品名称
    rewritten_query: str          # 改写后的问题
    history: list                 # 历史对话记录
    is_stream: bool               # 是否流式输出标记


# 定义图状态的默认初始值
# 建议定一个初始化对象，方便后续使用
graph_default_state: QueryGraphState = {
    "session_id": "",
    "original_query": "",

    # 检索中间数据：默认空列表，避免节点里 state["embedding_chunks"].append 报 NoneType
    "embedding_chunks": [],
    "hyde_embedding_chunks": [],
    "kg_chunks": [],
    "web_search_docs": [],

    # 排序数据
    "rrf_chunks": [],
    "reranked_docs": [],

    # 生成数据
    "prompt": "",
    "answer": "",

    # 辅助信息
    "item_names": [],
    "rewritten_query": "",
    "history": [],
    "is_stream": False,   # 默认非流式；接口层按需覆盖为 True
}


def create_default_state(**overrides) -> QueryGraphState:
    """
    创建默认状态，支持覆盖

    Args:
        **overrides: 要覆盖的字段（关键字参数解包）

    Returns:
        新的状态实例

    Examples:
        state = create_default_state(
            session_id="sess_001",
            original_query="万用表怎么测电压？",
            is_stream=True
        )
    """
    # 默认状态（深拷贝，避免多个 state 共享同一个列表对象）
    state = copy.deepcopy(graph_default_state)
    # 用 overrides 覆盖默认值
    state.update(overrides)
    # 返回创建好的状态字典实例
    return state


def get_default_state() -> QueryGraphState:
    """
    返回一个新的状态实例，避免全局变量污染
    """
    return copy.deepcopy(graph_default_state)


if __name__ == "__main__":
    """
    测试
    """
    # 1. 创建默认状态
    state = create_default_state(
        session_id="sess_test_001",
        original_query="万用表怎么测电压？"
    )
    logger.info(state)

    # 2. 验证深拷贝：修改一个 state 的列表，不影响默认值
    state["item_names"].append("万用表")
    assert graph_default_state["item_names"] == [], "深拷贝失效：默认状态被污染"
    logger.info(f"深拷贝校验通过，默认 item_names 仍为：{graph_default_state['item_names']}")