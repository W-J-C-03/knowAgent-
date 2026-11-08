import time
import sys

from app.clients.milvus_utils import create_hybrid_search_requests, get_milvus_client, hybrid_search
from app.conf.milvus_config import milvus_config
from app.lm.embedding_utils import generate_embeddings
from app.utils.task_utils import add_running_task, add_done_task

def node_search_embedding(state):
    """
    节点功能：进行向量内容检索
    主要作用：问题 -》查询chunks切片
    达到目标：{embedding_chunks：[chunks]}
    参数：{
        rewritten_query：重写的问题 -》根据它查询
        item_names：【】 -》明确主体
    }
    :param state:
    :return:
    """
    print("---向量内容检索 开始处理---")
    add_running_task(state["session_id"], sys._getframe().f_code.co_name, state.get("is_stream"))

    # 搜索假设型答案
    # 1.先从state获取参数数据
    rewritten_query = state.get("rewritten_query")
    item_names = state.get("item_names")
    # 2.将重写问题生成对应的向量【稠密和稀疏】
    embeddings = generate_embeddings([rewritten_query])
    # 3.进行向量数据库的混合查询
    # 3.1 创建混合查询请求对象AnnSearchRequest
    # 查询条件：1.向量检索 2.item_name一定要在item_names里 混合查询的查询条件item_name（字段名） in [item_name]
    item_name_str = ', '.join(f'"{item}"' for item in item_names)
    hybrid_search_requests = create_hybrid_search_requests(
        dense_vector=embeddings['dense'][0],
        sparse_vector=embeddings['sparse'][0],
        expr=f"item_name in [{item_name_str}]"
    )
    # 3.2进行混合查询
    milvus_client = get_milvus_client()
    resp = hybrid_search(
        client=milvus_client,
        collection_name=milvus_config.chunks_collection,
        reqs=hybrid_search_requests,
        ranker_weights=(0.9,0.1),
        norm_score=True,
        limit=5,
        output_fields=["chunk_id", "content", "file_title", "parent_title", "item_name"]
    )
    """
        [
            [
                id,
                distance,
                entity:
                    {
                    "chunk_id", "content", "file_title", "parent_title", "item_name"
                }
            ]
        ]
    """
    # 4.处理查询结果赋值embedding_chunks属性
    embeddings_chunks = resp[0] if resp else []

    add_done_task(state["session_id"], sys._getframe().f_code.co_name, state.get("is_stream"))
    print("---向量化检索 处理结束")

    return {"embedding_chunks": embeddings_chunks}