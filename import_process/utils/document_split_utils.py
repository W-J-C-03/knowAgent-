import json
import os
import re

from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.core.logger import logger
from app.import_process.agent.state import ImportGraphState


def step1_get_content_split(state: ImportGraphState):
    # 读取要切片的内容
    md_content = state['md_content']
    if not md_content:
        logger.error(f"[step1_get_content_split]没有有效的md内容，抛异常")
        raise Exception("检查输入文件路径是否正确")
    # 处理md_content中的换行符号 -》因为不同系统的换行可能不一样
    md_content = md_content.replace("\r\n", "\n").replace("\r", "\n")
    file_title = state.get('file_title', 'default_file')
    return md_content, file_title

def step2_split_by_title(md_content, file_title):
    """
    语义切割
    根据标题进行切割
    :param md_content:
    :param file_title:
    :return: [{content, title, file_title}]
    """
    # 判断是否是标题（# - ######），和代码块（注释是#，开头是```或~~~）区分开
    # 1.准备前置工作
    # 1.1正则
    title_pattern = r'^\s*#{1,6}\s+.+'
    # 1.2 md_content切割\n
    lines = md_content.split('\n')
    # 1.3 定义临时存储变量 current_title = str | current_lines = [] | title_count = 0 存储了多少块
    # is_code_block = bool 判断是不是代码块
    current_title = ""
    current_lines = [] # 当前标题行
    title_count = 0
    is_code_block = False
    # 1.4 最终存储的列表 sections = []
    sections = []
    # 2.循环每行的列表
    for line in lines:
        strip_line = line.strip() # 去掉字符串首尾的空白字符。
        # 2.1 判断代码块状态
        if strip_line.startswith('```') or strip_line.startswith('~~~'):
            # 进入代码块 或退出代码块
            # 第一次来一定要进入代码块
            is_code_block = not is_code_block
            # 内容一定不是标题
            current_lines.append(strip_line)
            continue
        # 2.2 判断是不是标题
        is_title = (not is_code_block) and re.match(title_pattern, strip_line)
        if is_title:
            # 先检查（是不是第一次，也就是md文件的第一个标题）只要不是第一个标题，就把current_lines存储起来
            if current_title:
                sections.append({
                    "title": current_title,
                    "content": "\n".join(current_lines),
                    "file_title": file_title
                })

            # 2.3 是标题如何处理
            current_title = strip_line
            current_lines = [current_title]
            title_count += 1
        else:
            # 2.4 不是标题如何处理
            current_lines.append(line)
    if current_title:
        sections.append({
            "title": current_title,
            "content": "\n".join(current_lines),
            "file_title": file_title
        })
    # 3.返回结果 sections
    return sections, title_count, len(lines)


def split_long_section(section, max_len):
    # 将当前chunk超长内容进行二次切割
    # 返回切割后的[{}, {}]
    # 1.content获取到
    content = section.get('content')
    # 2.判断content是否超长了没有，如果没有，直接返回
    if len(content) <= max_len:
        logger.info(f"[split_long_section]:{content}当前chunk长度小于等于{max_len}，不切割")
        return [section]
    # 3.超长则进行切割
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=max_len,
        chunk_overlap=100, # 重叠长度
        separators=['\n\n','\n','。','！','；']
    )
    # title = 标题名 _1 _2 _3 || part 1 2 3 || parent_title = section.title
    sub_sections = []
    for index, chunk in enumerate(splitter.split_text(content), start=1):
        text = chunk.strip() # 去首位空白字符
        title = f"{section.get('title')}_{index}"
        parent_title = section.get("title")
        part = index
        file_title = section.get("file_title")
        sub_sections.append({
            "title": title,
            "content": text,
            "file_title": file_title,
            "parent_title": parent_title,
            "part": part
        })

    # 4.返回切割后的结果
    return sub_sections


def merge_short_sections(final_sections, min_len):
    """
    上次切的碎，需要做合并
        1.content长度要小于min_len
        2.同一个parent_title才能合并
    :param final_sections:
    :param min_len:
    :return:
    """
    merged_sections = []
    pre_section = None
    # 双指针法
    for section in final_sections:
        # section 除了第一次，是第二个指针
        if pre_section is None:
            pre_section = section
            continue
        # 判断pre_section是否小于min_len，小于则和当前section合并
        is_pre_short = len(pre_section.get('content')) < min_len
        # 判断pre_section和当前section是否同一标题下，若是，才能合并
        is_same_title = pre_section.get("parent_title") and pre_section.get("parent_title") == section.get("parent_title")
        if is_pre_short and is_same_title:
            # 合并条件成立
            pre_section +="\n\n" + section.get("content")
            pre_section['part'] = section.get('part')
        else:
            merged_sections.append(pre_section)
            pre_section = section
    if pre_section is not None:
        merged_sections.append(pre_section)
    return merged_sections

def step3_refine_chunks(sections, max_len, min_len):
    """
    做内容精细切割
        1.超过了MIN_CONTENT_LENGTH块，要做切割（parent_title | part）
        2.小于MAX_CONTENT_LENGTH块，并且是同一个parent_title，进行合并
    :param sections:
    :param max_len:
    :param min_len:
    :return:
    """
    # 最后处理的结果：
    final_sections = []

    # 超过的先切碎
    for section in sections:
        # section 每个切块： title content file_title
        sub_sections = split_long_section(section['content'], max_len)
        # 不行[{}]
        final_sections.extend(sub_sections)
    #小于的再合并
    final_sections = merge_short_sections(final_sections, min_len)
    # 补全属性和参数（parent_title title）
    for section in final_sections:
        section['part'] = section.get('part') or 1
        section['parent_title'] = section.get('parent_title') or section.get('title')
    return final_sections

def step4_backup_chunks(state, sections):
    """
    将切割完的碎片进行存储
    :param state: 本地地址：local_dir
    :param sections: 要存储的内容[{}]
    :return:
    """
    local_dir = state.get('local_dir')
    backup_file_path = os.path.join(local_dir, 'chunks.json')
    with open(backup_file_path, 'w', encoding="utf-8") as f:
        json.dump(
            sections, # 要写的数据
            f, # 写出的位置
            ensure_ascii=False, # 是否将中文转成ascii
            indent=4 # 缩进
        )
    logger.info(f"已将内容存入{backup_file_path}")









