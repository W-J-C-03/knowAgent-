import asyncio
import json
import sys

from agents.mcp import MCPServerStreamableHttp

from app.conf.bailian_mcp_config import mcp_config
from app.core.logger import logger
from app.utils.task_utils import add_done_task, add_running_task

DASHSCOPE_BASE_URL_STREAMABLE=mcp_config.DASHSCOPE_BASE_URL_STREAMABLE
DASHSCOPE_API_KEY=mcp_config.api_key
async def mcp_call_streamable(query):
    """
    调用百炼的网络搜索工具
    :param query:
    :return:
    """
    # 1.创建MCPServerStreamableHttp对象
    search_mcp = MCPServerStreamableHttp(
        name="search_mcp",
        params={
            # 核心参数
            "url": DASHSCOPE_BASE_URL_STREAMABLE,
            "headers": {"Authorization": f"Bearer {DASHSCOPE_API_KEY}"},
            "timeout": 10,
        },
        max_retry_attempts=3
    )
    # 2.连接 - 调用 - 关闭
    try:
        # 连接
        await search_mcp.connect()
        # 获取工具
        tools = await search_mcp.list_tools()
        print(f"工具列表：{tools}")
        # 调用
        result = await search_mcp.call_tool(
            tool_name="bailian_web_search",
            arguments={
                "query": query,
                "count": 5
            }
        )
        return result

    finally:
        await search_mcp.cleanup()


def node_web_search_mcp(state):
    """
    节点功能，调用外部搜索引擎补充信息
    :param state:
    :return:
    """
    add_running_task(state["session_id"], sys._getframe().f_code.co_name,state["is_stream"])
    print("---node-web-search-mcp处理---")

    # 1.获取问题（rewritten_query）
    query = state['rewritten_query']
    # 2.调用streamable网络搜索方法
    result = asyncio.run(mcp_call_streamable(query))
    # 3.结果处理即可（根据官网中返回数据来解析数据）
    # {
    #   "content":[{"text":{"page":目标数据}}]
    # }
    web_documents = json.loads(result.content[0].text).get("pages",[])
    logger.info(f"mcp搜索结果为：{web_documents}")
    add_done_task(state["session_id"], sys._getframe().f_code.co_name, state["is_stream"])
    print("---node-web-search-mcp处理结束---")
    return {"web_search_docs": web_documents}