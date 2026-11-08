import time
import sys

from app.clients.mongo_history_utils import get_recent_messages, save_chat_message
from app.utils.item_name_confirm_utils import step3_llm_item_name_and_rewritten_query, step4_query_milvus_item_names, \
    step5_confirmed_and_optional_item_name, step6_deal_list
from app.utils.task_utils import add_running_task, add_done_task

def node_item_name_confirm(state):
    """
    节点功能：确认用户问题中的核心商品名称。
    核心目标：1.提取item_name（大模型从历史对话+本次提问提取 -》item_name -》向量库搜索 -》 打分）
            2.利用模型重写用户的问题，确保后续查询召回率更高
    核心参数：state['original_query' -》用户的原问题] || session_id
    响应数据：item_names: List[str] # 提取出的商品名称
            rewritten_query: str # 改写后的问题
            history: list # 历史对话记录
    步骤：
        1.获取历史条件记录（作为依据）
        2.保存当前次的聊天记录
        3.利用模型 -》提取item_names 和 重写提问内容
        4.进行item_name的向量数据库查询
        5.对item_name结果进行打分分类处理 【确认集合】 【可选集合】（这是指分数没有达到标准的结果）
        6.处理确认和可选集合。有确认 -》继续下一个节点执行 有可选或没有item_names -》answer赋值结果
        7.补充state状态item_names rewritten_query history
    """
    print(f"---node_item_name_confirm---开始处理")
    # 记录任务开始
    add_running_task(state["session_id"], sys._getframe().f_code.co_name,state["is_stream"])

    # 1.获取历史条件记录（作为依据）
    history_chats = get_recent_messages(session_id=state['session_id'], limit=10)
    # 3.利用模型 -》提取item_names和重写提问内容
    # 参数：state['original_query'] || history_chats
    # 响应：{item_names:[], rewritten_query: str}
    # 为啥问题要重写
    """
        1.消除指代歧义 他 Ta 它 不明确，明确查询主体item_name
        2.补全上下文，需要有历史记录支撑
        3.去掉口语和冗余
        4.润色问题增加召回率，模型查询的时候也会更精准
    """
    item_names_and_rewritten_query = step3_llm_item_name_and_rewritten_query(state['original_query'], history_chats)
    item_names = item_names_and_rewritten_query.get("item_names", [])
    rewritten_query = item_names_and_rewritten_query.get("rewritten_query", "")

    if len(item_names) > 0:
        # 向量数据库查询item_name
        # 4.milvus向量查询item_names -》模型提取 不一定跟向量数据库的完全相同
        # 参数：item_names = []
        # 返回：1 -》向量数据库中item_names 2 -》向量数据库中item_names（向量查询）
        #   [ {extracted: （模型提取的item_name），matches：[{item_name:name,score:0.8},{item_name:name,score:0.8}] },
    #         {extracted: （模型提取的item_name），matches：[{item_name:name,score:0.8},{item_name:name,score:0.8}] }
    #   ]
        query_milvus_results = step4_query_milvus_item_names(item_names)
        # 5.查询结果进行处理 区分 确定的item_name以及可选的item_name -》没有对应的item_name
        # 参数：query_milvus_results
        # 返回：{确定item_name:[x,x,x], 可选的item_name:[x,x,x]}
        item_results = step5_confirmed_and_optional_item_name(query_milvus_results)
        # 6.根据item_name确定的集合进行用户反馈结果的处理 -》answer赋值结果
        # 参数：item_results || 修改聊天记录对应item_names history_chats
        state = step6_deal_list(state, item_results, history_chats, rewritten_query)
        # 7.记录本次的聊天对话（answer回答）-》放到最后一个节点
        # if state.get("answer"):
        #     save_chat_message(
        #         session_id=state["session_id"],
        #         role="assistant",
        #         text=state["answer"],
        #         item_names=[],
        #         image_urls=[],
        #         rewritten_query=rewritten_query
        #     )
    # 2.保存当前次的聊天记录
    # message_id = save_chat_message(
    #     session_id=state['session_id'],
    #     role="user",
    #     text=state["original_query"],
    #     rewritten_query=state.get("rewritten_query", ""),
    #     item_names=state.get("item_names", []),
    #     image_urls=state.get("image_urls", []),
    # )
    # 记录任务结束
    add_done_task(state["session_id"], sys._getframe().f_code.co_name,state["is_stream"])

    print(f"---node_item_name_confirm---处理结束")

    return {"item_names": ["示例商品"]}