import sys

from app.core.logger import logger
from app.import_process.agent.state import ImportGraphState
from app.utils.item_name_recognition_utils import step1_get_chunks, step2_build_context, step3_call_llm, \
    step4_update_chunks_and_state, step5_generate_embedding, step6_save_to_vector_db
from app.utils.task_utils import add_running_task, add_done_task

"""
    注解：这一步调用向量生成模型是为了：让大模型可以识别用户的问题
    目标：
        1.录用文本大模型识别当前chunks对应的item_name：用于区分不同的文档
        2.使用嵌入式模型（本地），将item_name生成向量存储到向量数据库
        3.修改state[chunks] -》chunk{ title parent_title part file_title content item_name -》每个赋值}
    实现步骤：
        1.校验和取值（file_title，chunks）
        2.构建上下文环境 chunks -》top 5 -》拼接成context文本
        3.调用大模型，拼接提示词，识别chunks对应item_name
        4.修改state_chunks -》item_name
        5.item_name生成向量（稠密/稀疏）
        6.存储向量到向量数据库 kb_item_name（id / file_title / item_name / 稠密和稀疏）
"""
def node_item_name_recognition(state: ImportGraphState) -> ImportGraphState:
    """
    节点: 主体识别 (node_item_name_recognition)
    为什么叫这个名字: 识别文档核心描述的物品/商品名称 (Item Name)。
    未来要实现:
    1. 取文档前几段内容。
    2. 调用 LLM 识别这篇文档讲的是什么东西 (如: "Fluke 17B+ 万用表")。
    3. 存入 state["item_name"] 用于后续数据幂等性清理。
    """
    function_name = sys._getframe().f_code.co_name
    logger.info(f">>> [{function_name}]开始执行了。现在的状态为: {state}")
    add_running_task(state['task_id'], function_name)
    logger.info(f">>> [Stub] 执行节点: {sys._getframe().f_code.co_name}")
    try:
        # 1.检验和取值 file_title，chunks
        chunks, file_title = step1_get_chunks(state)
        # 2.构建上下文环境 chunks -》top 5 -》拼接成context文本
        context = step2_build_context(chunks)
        # 3.调用大模型，拼接提示词，识别chunks对应item_name
        item_name = step3_call_llm(context, file_title)
        # 4.修改state_chunks -》item_name
        step4_update_chunks_and_state(state, item_name, chunks)
        # 5.item_name生成向量（稠密 / 稀疏）
        dense_vector, sparse_vector = step5_generate_embedding(item_name)
        # 6.存储向量到向量数据库kb_item_name（id / file_title / item_name / 稠密和稀疏）
        step6_save_to_vector_db(file_title, item_name, dense_vector, sparse_vector)

    except Exception as e:
        # 处理异常
        logger.error(f">>> [{function_name}]使用minerU解析发生异常，异常信息：{str(e)}")
        raise
    finally:
        logger.info(f">>> [{function_name}]成功完成，现在状态为：{state}")
        add_done_task(state['task_id'], function_name)

    return state