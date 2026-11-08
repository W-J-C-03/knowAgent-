import json
import sys

from langchain_core.messages import HumanMessage

from app.clients.milvus_utils import get_milvus_client, create_hybrid_search_requests, hybrid_search
from app.clients.mongo_history_utils import get_recent_messages
from app.core.load_prompt import load_prompt
from app.core.logger import logger
from app.lm.embedding_utils import generate_embeddings
from app.lm.lm_utils import get_llm_client
from app.utils.task_utils import add_running_task, add_done_task


def step3_llm_item_name_and_rewritten_query(original_query: dict, history_chats: list):
    """
    根据历史记录 -》识别item_names 和重写问题
    :param original_query: 用户原有问题
    :param history_chats: 聊天记录
    :return: {item_names = [], rewritten_query :问题}
    """
    # 1.准备提示词
    history_text = ""
    for chat in history_chats:
        history_text +=f"聊天角色：{chat['role']}，回答内容：{chat['text']}，重写问题：{chat['rewritten_query']}，关联主体：{'.'.join(chat.get('item_names'))}\n"

    prompt = load_prompt("rewritten_query_and_itemnames", history_text=history_text, query=original_query)
    # 2.模型调用
    lm_client = get_llm_client(json_mode=True)
    # system ->模型的角色边界 -> 不变 【角色，规则，格式】
    # user -> 每次任务提示 -> 多条动态调整 【提问/聊天】
    # 事实上，可以把模型的角色和边界写到user中，功能一样
    messages = [
        HumanMessage(content=prompt)
    ]
    response = lm_client.invoke(messages)
    # 怎么确保模型一定能返回格式化数据 json 1.设置json格式化 2.提示词中明确要求输出为特定格式 3.一定要给模型示例 4.做好返回结果的校验
    # 3.结果解析
    content = response.content
    # 格式返回结果是json格式，但可能有``` json
    if content.startswith("```json"):
        content = content.replace("```json", "").replace("```", "")
    dict_content = json.loads(content)
    if "item_names" not in dict_content:
        dict_content["item_names"] =[]
    if "rewritten_query" not in dict_content:
        dict_content["rewritten_query"] = original_query
    # 4.封装返回
    logger.info(f"已经完成问题的重写和item_name的提取，结果为：{dict_content}")
    return dict_content

def step4_query_milvus_item_names(item_names: list):
    """
    查询向量数据库，进行item_name的确定
    :param item_names: 模型提取的item_names可能不准
    :return:
            [{extracted:模型item_name,matches:[{item_name:xx,score:0.9}]}]
    """
    # 一定要做混合查询 （稠密+稀疏）
    # 1.获取milvus客户端
    milvus_client = get_milvus_client()
    # 2.将item_name转成向量（稠密+稀疏）
    embeddings = generate_embeddings(item_names)
    # 3.混合查询（创建稠密+稀疏的AnnSearchRequest || 设置权重重排 || 进行混合查询）
    final_result = []
    for index, item_name in enumerate(item_names):
        # 1.获取当前item_name对应的向量
        dense_vector = embeddings["dense"][index]
        sparse_vector = embeddings["sparse"][index]
        # 2.拼对应的AnnSearchRequest
        reqs = create_hybrid_search_requests(
            dense_vector=dense_vector,
            sparse_vector=sparse_vector,
        )
        # 3.定义权重重排
        response = hybrid_search(
            client=milvus_client,
            collection_name=milvus_client.item_collection_name,
            reqs=reqs,
            ranker_weights=(0.5, 0.5),
            norm_score=True
        )
        """
            [
                [
                    {id:xx, distance:0.x, entity:{item_name:xxx}},
                    {id:xx, distance:0.x, entity:{item_name:xxx}}
                ]
            ]
        """
        # 5.结果解析
        matches = [] # 当前item对应的匹配结果
        if response and len(response) > 0:
            for hit in response[0]:
                entity = hit.get("entity", {})
                hit_name = entity.get("item_name")
                score = entity.get("distance", 0)
                if hit_name:
                    matches.append({
                        "item_name": hit_name,
                        "score": score
                    })

        # 4.提取查询结果封装返回的数据格式
        final_result.append({
            "extracted": item_name,
            "matches": matches
        })
    return final_result

def step5_confirmed_and_optional_item_name(query_milvus_results):
    """
    通过向量数据库查询的item_name，根据分数归纳出确定和可选的item_name列表
    :param query_milvus_results: 元数据 [{extracted:item_name,matches:[{item_name:xx,score:0.9}]}]
    :return:
        {
            confirmed_item_names: [确定的item_name] 匹配的分高
            options_item_names: [可选item_name] 分低
        }
    评分规则：
        确定item_name
        0.85【根据权重和数据分析进行调整】
        可选item_name
        0.6
        忽略
    思路：1.循环处理每个item_name列表和分数 2.高分要一个 3.如果没有一个高分，那需要从可选中找 4.不区分extracted: 装到对应的确认或可选集合中
    """
    # 1.准备两个列表 确认 可选的
    confirmed_item_names = []
    options_item_names = []
    # 2.循环处理元数据 query_milvus_results
    for item_name_result in query_milvus_results:
        extracted_name = item_name_result.get("extracted")
        matches = item_name_result.get("matches", [])
        # 3.进行分数排序（倒序） || 列表推导式 提取 0.85 || 0.6
        # matches [{score:xx, item_name=""}]
        matches.sort(key=lambda x: x["score", 0], reverse=True)
        # >= 0.8 优先从high中查找，如果没有high，才去middle中找最高的分数
        high_score_matches = [x for x in matches if x.get("score", 0) >= 0.85]
        middle_score_matches = [x for x in matches if x.get("score", 0) >=0.6]
        # 4.处理高分的列表 只有一个 || 多个【item_name = extracted】 or 获取最高分的一个
        # 4.1只有一个高分，就直接取
        if len(high_score_matches) == 1:
            confirmed_item_names.append(high_score_matches[0]["item_name"])
            continue
        # 4.2 有多个高分
        if len(high_score_matches) > 1:
            # 同一个名 = 分不是1 也可能不是最高，优先考虑名字相同
            same_name_item = None
            for item in high_score_matches:
                if item.get("item_name") == extracted_name:
                    same_name_item = item
                    break
            if not same_name_item:
                same_name_item = high_score_matches[0]
            confirmed_item_names.append(same_name_item.get("item_name"))
            continue
        # 5.处理可选分数列表 给用户返回提示，可以多带几个返回,带俩
        if len(middle_score_matches) > 0:
            for item in middle_score_matches[0:2]:
                options_item_names.append(item.get("item_name"))
            continue
        logger.info(f"没有匹配的item_name，忽略{extracted_name}")
    result = {
        "confirmed_item_names": confirmed_item_names,
        "options_item_names": options_item_names
    }
    logger.info(f"处理结果为：{result}")
    return result

def step6_deal_list(state, item_results, history_chats, rewritten_query):
    """
    根据集合类型中数据，判定是否要赋值answer内容
    :param state:
    :param item_results: confirmed_item_names, options_item_names两个列表
    :param history_chats:
    :param rewritten_query:
    :return:
    """
    # 1.获取集合
    confirmed_item_names = item_results.get("confirmed_item_names", [])
    options_item_names = item_results.get("options_item_names", [])
    # 2.确认集合有数据
    if len(confirmed_item_names) > 0:
        # 2.1 更新聊天记录 -》item_names -》confirm_item_names
        # 2.2 修改和存储state状态
        state['item_names'] = confirmed_item_names
        state['rewritten_query'] = rewritten_query
        state['history'] = history_chats
        if "answer" in state:
            del state["answer"]
        logger.info(f"有确定的item_name：{confirmed_item_names}")
        return state
    if len(options_item_names) > 0:
        option_names = ""
        answer = f"您是想咨询以下哪个商品：{option_names}"
        state['answer'] = answer
        logger.info(f"有可选的item_name：{option_names}")
        return state
    # 4.确认和可选集合都没数据
    answer = "没有匹配的商品名，请重新提问"
    state['answer'] = answer
    logger.info("没有匹配的item_name")
    return state

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
