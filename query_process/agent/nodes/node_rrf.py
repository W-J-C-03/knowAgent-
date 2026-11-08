import time
import sys
from app.utils.task_utils import add_running_task, add_done_task
from app.core.logger import logger

def step3_reciprocal_rank_fusion(source_with_weight, topk: int = 5):
    """
    进行同源数据排名+权重处理
    :param source_with_weight: [(集合，权重),(集合，权重)]
    :param topk:
    :return: [{},{},{}]
    """
    # 1.准备两个容器 记录历史得分
    score_dict = {} # id | chunk_id ->key ||  计算得分 -》value
    # 2.记录chunk片段
    chunk_dict = {} # 记录 chunk_id -》key || chunk {} 不重复
    # 3.循环处理每个集合中的数据，进行积分计算
    # source_with_weight = [
    #     (embedding_chunks, 1.0),
    #     (hyde_embedding_chunks, 1.0)
    # ]
    for source, weight in source_with_weight:
        # source = [{id:实体的主键,distance:得分,entity:{chunks_id,content, title...}},{},{}]
        # weight = 1.0
        # source为两个节点传入的chunks
        # 嵌套遍历，遍历具体路的数据和chunk
        for rank, chunk in enumerate(source, start=1):
            # 计算当前chunk的得分
            # 获取chunk_id
            chunk_id = chunk.get("id") or chunk.get("entity").get("chunk_id")
            # 计算得分rrf权重版本的公式 = (1/(k+rank))*weight
            score_dict[chunk_id] = score_dict.get(chunk_id, 0.0) + (1.0/(60+rank))*weight
            chunk_dict[chunk_id] = chunk # 会覆盖
            chunk_dict.setdefault(chunk_id, chunk) # 没有chunk_id才添加，两种效果无区别，每个chunk保留一边
    # 4.分和chunk的融合+排序
    merged = []
    for chunk_id, score in score_dict.items():
        chunk = chunk_dict.get(chunk_id)
        merged.append((chunk, score))
        # [(chunk, score), (chunk, score)]
    merged = sorted(merged, key=lambda x: x[1], reverse=True)
    # 5.切指定的topk，我就需要这么多
    merged = merged[:topk]
    # 6.获取chunk的排名数据
    rank_chunks = [chunk for chunk, id in merged]
    logger.info(f"完成了rrf排序处理，结果为：{rank_chunks}")
    return rank_chunks





def node_rrf(state):
    """
    节点功能：Reciprocal Rank Fusion
    将多路召回的结果（向量、HyDE、Web、KG）进行加权融合排序。
    """
    print("---RRF---")
    add_running_task(state["session_id"], sys._getframe().f_code.co_name, state.get("is_stream"))

    # 获取同源路的数据
    # 长啥样？？？
    # milvus -> [ [], []] 1.单列查询 data = [向量1，向量2] -》[[向量1],[向量2]]
    #                     2.混合查询 reps = [anns....]
    #                     [向量1] =》【{id:实体的主键,distance:得分,entity:{chunks_id,content, title...}},{},{}】
    embedding_chunks = state.get("embedding_chunks")
    hyde_embedding_chunks = state.get("hyde_embedding_chunks")
    # 2.数据进行整合（捏到一起）
    # 先增加权重，方便后续动态调整，来自重写的问题进行向量数据库查询，和重写的问题＋模型生成的假设性答案进行的向量数据库查询
    source_with_weight = [
        (embedding_chunks, 1.0),
        (hyde_embedding_chunks, 1.0)
    ]
    # 3.rrf算法进行数据排序处理
    rrf_response = step3_reciprocal_rank_fusion(source_with_weight)
    # 4.将排序后的数据添加到state[]
    add_done_task(state['session_id'], sys._getframe().f_code.co_name, state.get("is_stream"))
    return {"rrf_chunks": rrf_response}