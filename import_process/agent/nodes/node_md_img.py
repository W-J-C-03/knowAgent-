import os
import sys

from app.core.logger import logger
from app.import_process.agent.state import ImportGraphState
from app.utils.md_img_utils import step1_get_content, step2_scan_images, step3_generate_img_summeries, \
    step4_upload_images_and_replace_md, step5_replace_md_and_save
from app.utils.task_utils import add_running_task, add_done_task


def node_md_img(state: ImportGraphState) -> ImportGraphState:
    """
    MD文件图片处理核心节点 - 五步法完成图片全流程处理
    核心流程：
    1. 初始化获取MD内容、文件路径、图片文件夹路径
    2. 扫描图片文件夹，筛选MD中实际引用的支持格式图片
    3. 调用多模态大模型为图片生成内容摘要
    4. 将图片上传至MinIO，替换MD中本地图片路径为MinIO访问URL，并填充图片摘要
    5. 备份原MD文件，保存处理后的新MD文件并更新状态
    :param state: 导入流程全局状态对象，包含task_id、md_path、md_content等核心参数
    :return: 更新后的全局状态对象（md_content/md_path为处理后新值）
    """
    function_name = sys._getframe().f_code.co_name
    logger.info(f">>> [{function_name}]开始执行了。现在的状态为: {state}")
    add_running_task(state['task_id'], function_name)
    # 1.校验数并获取本次操作的数据
    # 参数：state -》md_path md_content
    # 响应：1.校验后的md_content 2.md路径对象 3.获取图片的文件夹images
    md_content, md_path_obj, images_dir_obj = step1_get_content(state)
    # 如果没有图片，则直接返回state
    if not images_dir_obj.exists():
        logger.info(f">>> [{function_name}]没有图片，直接返回state")
        return state
    # 2.识别md中使用过的图片
    # 参数：1.md_content 2.images中图片的文件夹地址
    targets = step2_scan_images(md_content, images_dir_obj)
    # 3.调用多模态大模型为图片生成内容摘要
    # 参数：[(图片名，图片地址，（上文，下文）)] || md文件名称
    # 响应：{图片名1：总结，图片名2：总结，...}
    summeries = step3_generate_img_summeries(targets, md_path_obj.stem)
    logger.info(f">>> [Stub] 执行节点: {sys._getframe().f_code.co_name}")
    # 4.上传图片到minio同时替换md中的图片（描述+地址）
    # 参数：minio_client || {图片名：总结} || [图片名，图片地址，（上文，下文）] (minio) || md_content
    # 响应：new_md_content
    # state[md_content] = new_md_content
    """
    为啥要用minio：通过minio来存储md中的图片，使得在替换掉md中的图片url地址，不会丢失图片，并且有统一的
    网络地址，!(图片名)[图片url] -> !(图片名)[图片在minio服务器的网络地址]，图片在minio统一管理
    """
    # 替换后的md内容
    new_md_content = step4_upload_images_and_replace_md(summeries, targets, md_content, md_path_obj.stem)
    # 5.将新内容替换和保存修改
    # 参数：new_md_content，原地址 -》xx.md -》xx_new.md
    # 响应：新的md的地址 new_md_path
    # state[md_path] = new_md_path
    new_md_file_path = step5_replace_md_and_save(new_md_content, md_path_obj)
    state['md_path'] = new_md_file_path
    state['md_content'] = new_md_content
    logger.info(f">>> [{function_name}]开始执行了。现在的状态为: {state}")
    add_done_task(state['task_id'], function_name)
    return state
