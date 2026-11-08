import re
import sys
import time

import requests

from app.clients.mongo_history_utils import save_chat_message
from app.core.load_prompt import load_prompt
from app.lm.lm_utils import get_llm_client
from app.utils.sse_utils import push_to_session, SSEEvent
from app.utils.task_utils import add_running_task, add_done_task, set_task_result
from app.core.logger import logger

MAX_CONTEXT_CHARS=12000 # 限制prompt长度
def step1_check_answer(state):
    # 判断第一个节点，有没有明确的answer回答
    # 1.获取answer | is_stream
    answer = state.get("answer")
    is_stream = state.get("is_stream", False)
    if answer:
        # 有
        if is_stream:
            # 流式
            # 1.推送到sse
            push_to_session(state['session_id'], SSEEvent.DELTA, {"delta": answer})
        else:
            # 非流式
            # 1.设置任务结果
            set_task_result(state['session_id'], "answer", answer)
        return True
    else:
        return False


def step2_load_prompt(state):
    """
    加载模型润色答案的提示词
    :param state:
    :return:
    """
    # 数据从state取出
    rewritten_query = state.get("rewritten_query") or state.get("original_query")
    reranked_docs = state.get("reranked_docs", [])
    item_names = state.get("item_names", [])
    history = state.get("history", [])
    # 以下处理的内容均来自对应提示词文件中所需要的参数
    # 1.先处理chunk块的内容 -》context
    docs = []
    used_length = 0
    # reranked_docs =>[{text,chunk_id, score, url, title, source: local | web},{}]
    # 拼成：[1][text][source][url][title][score]\n\n
    # [2][text][source][url][title][score]     \n\n
    for i, doc in enumerate(reranked_docs, start=1):
        text = doc.get("text")
        source = doc.get("source")
        title = doc.get("title")
        score = doc.get("score")

        content = f"[{i}][source={source}][title={title}][score={score}]\n\n[{text}]"
        if used_length + len(content) > MAX_CONTEXT_CHARS:
            logger.info(f"本次提示词已超过限制长度")
            break
        docs.append(content)
        used_length += len(content)
    final_context = "\n\n".join(docs)
    # 2.再处理history -》聊天记录的内容
    history_str = ""
    if history and len(history) > 0:
        for i, msg in enumerate(history, start=1):
            role = msg.get("role")
            text = msg.get("text")
            current_history = ""
            if role == "user" and text:
                current_history = f"【用户】：{text}\n"
            elif role == "assistant" and text:
                current_history = f"【助手】：{text}\n"
            history_str += current_history
            used_length += len(current_history)
            if used_length > MAX_CONTEXT_CHARS:
                logger.info("本次提示词已超过限制长度")
                break
    else:
        history_str = "没有历史对话记录"
    # 3.再处理item_name
    item_names_str = ",".join(item_names)
    # 4.处理question问题
    answer_out_prompt = load_prompt(
        "answer_out",
        content=final_context,
        history=history_str,
        item_names=item_names_str,
        question=rewritten_query,
    )
    logger.info(f"已经完成了提示词生成：{answer_out_prompt}")
    return answer_out_prompt


def step3_create_answer(state, prompt):
    """
    使用模型生成最后的答案
    :param state:
    :param prompt:
    :return:
    """

    # 1.获取模型对象和客户端
    model = get_llm_client()
    # 2.获取流式状态【sse | set_result】
    is_stream = state.get("is_stream", False)
    answer = ""
    if is_stream:
        # 3.调用模型进行生成sse stream || set_result invoke
        for chunk in model.stream(prompt):
            # 3.1 推到sse
            delta = chunk.content
            answer += delta # 累加答案
            push_to_session(state['session_id'], SSEEvent.DELTA, {"delta":delta})
    else:
        # 4.最终的答案赋值给state['answer'] = 答案
        response = model.invoke(prompt)
        content = response.content
        answer = content
        set_task_result(state['session_id'], "answer", content)

    # 5.返回结果answer
    state["answer"] = answer
    return answer


def step4_extract_images_url(state):
    """
    这里的图片来自两个节点，一个是minio服务器，一个是web访问的
    从local -> chunk ->text中提取
        {text: ![](url)} ->url
    从web -> url -> 图片提取
        mcp { url:网络搜索 关联地址 || 图片地址}
    :param state:
    :return:
    """
    images = [] # 存储图片
    set_images = set() # 给图片去重
    # 1.定义正则
    image_reg = re.compile(r"!\[.*?\]\((.*?)\)")
    # reranked_docs => [{text, chunk_id, score, url, title, source: local | web},{}]
    # 2.宣传处理切片 -》从高分 -》低分
    reranked_docs = state.get("reranked_docs", [])
    for doc in reranked_docs:
        # {text, chunk_id, score, url, title, source:local | web}
        # 判断url是不是图片
        url = doc.get("url")
        if url:
            if url.endswith(".jpg", ".png", ".jpeg", ".gif", ".webp"):
                if url not in set_images:
                    images.append(url)
                    set_images.add(url)
        text = doc.get("text")
        # text正则提取图片
        if text:
            matches = image_reg.findall(text)
            for image_url in matches:
                if image_url not in set_images:
                    images.append(image_url)
                    set_images.add(image_url)
        # 不存在 -》添加到images

    logger.info(f"已经完成图片提取：数量：{len(images)}，提取内容：{images}")
    return images


def step5_write_history(state):
    """
    将对话存储到mongodb history
    每次对话 对应两条history
    :param state:
    :return:
    """
    session_id = state.get("session_id")
    answer = state.get("answer")
    rewritten_query = state.get("rewritten_query") or state.get("original_query")
    item_names = state.get("item_names", [])
    if rewritten_query:
        # user
        save_chat_message(
            session_id=session_id,
            role="user",
            text=rewritten_query,
            item_names=item_names
        )
        # assistant
        save_chat_message(
            session_id=session_id,
            role="assistant",
            text=answer,
            item_names=item_names,
            rewritten_query=rewritten_query,
        )


def node_answer_output(state):
    """
    宏观：将最终topk -》大模型 -》润色 -》结果 -》【【流式】sse -》前端（push_to_session） 【非流式】set_task_result】
    1.先检查state中是否存在answer回答 【item_name（1.明确【2.不确定 3.没有】 answer -》state）】 有可以直接退出会话
    2.生成对应的润色的提示词prompt
    3.使用模型润色答案 -》结果 -》文本
    4.提取原来topklist中的图片地址，单独返回sse
    5.对话的聊天记录（用户user/助手assistant）
    6.sse-final->返回图片
    节点功能：进行过处理可以是流式输出可以是整体输出
    """
    print("---node_answer_output节点处理开始---")
    add_running_task(state['session_id'], sys._getframe().f_code.co_name, state.get("is_stream"))

    # 1.先检查state中是否存在answer回答 【item_name（1.明确【2.不确定 3.没有】 answer -》state）】 有可以直接退出会话
    answer_exist = step1_check_answer(state)
    if not answer_exist:
        # 2.没有answer 生成对应的润色的提示词prompt
        prompt = step2_load_prompt(state)
        # 3.没有answer 使用模型润色答案 -》结果 -》文本
        answer = step3_create_answer(state, prompt)
        # 4.没有answer 提取原来topklist中的图片地址，单独返回sse
        image_url = step4_extract_images_url(state)
        # 5.sse-final->返回图片
        if image_url:
            # 不管是流式输出还是非流式输出，只要image_url有数据，都得输出
            push_to_session(
                state['session_id'],
                SSEEvent.FINAL,
                {
                    "answer": answer,
                    "status": "complete",
                    "image_url": image_url,
                }
            )
    # 6.对话的聊天记录（用户user/助手assistant）
    step5_write_history(state)
    add_done_task(state['session_id'], sys._getframe().f_code.co_name, state.get("is_stream"))
    print("---node_answer_output节点处理结束---")
    return state