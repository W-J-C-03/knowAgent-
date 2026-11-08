import os
import sys

from app.core.logger import logger
from app.import_process.agent.state import ImportGraphState, create_default_state
from app.utils.pdf_to_md_utils import step1_validate_paths, step2_upload_and_poll, step3_download_and_extract
from app.utils.task_utils import add_running_task, add_done_task


def node_pdf_to_md(state: ImportGraphState) -> ImportGraphState:
    """
    节点: PDF转Markdown (node_pdf_to_md)
    为什么叫这个名字: 核心任务是将 PDF 非结构化数据转换为 Markdown 结构化数据。
    未来要实现:
    1.进入的日志和任务状态的配置
    2.进行参数校验（local_dir -》 给与默认值 | local_file_path完成字面意思的校验 -》 深入校验文件是否真的存在）
    3.调用minerU进行pdf的解析（local_file_path）返回一个下载文件的地址xx.zip url地址
    4.下载zip包并解析和提取（local_dir）
    5.把md_path地址进行赋值，读取md的文件内容md_content（文本内容）
    6.结束的日志和任务状态的配置
    """
    # 1.进入结点的日志输出【节点 + 模型】记录任务状态【哪个任务开始了】-》给前端推送信息
    function_name = sys._getframe().f_code.co_name
    logger.info(f">>> [{function_name}]开始执行了。现在的状态为: {state}")
    add_running_task(state['task_id'], function_name)

    try:
        # 2.进行参数校验（local_dir-》）
        # 返回校验完毕，可以直接使用的Path路径对象:文件和输出文件夹
        pdf_path_obj, local_dir_obj = step1_validate_paths(state)
        # 3.调用minerU
        zip_url = step2_upload_and_poll(pdf_path_obj)
        # 4.下载zip：参数：1.要下载的地址 2.local_dir_pbj解压的文件夹 3.文件名
        # 返回值：解压后的md文件的真实路径
        md_path = step3_download_and_extract(zip_url, local_dir_obj, pdf_path_obj.stem)
        # 5.把md_path进行赋值给md_content
        state['md_path'] = md_path
        state['local_dir'] = local_dir_obj
        with open(md_path, 'r', encoding="utf-8") as f:
            state['md_content'] = f.read()

    except Exception as e:
        logger.error(f">>> [{function_name}]使用minerU解析发生了异常，异常信息：{str(e)}")
        raise
    finally:
        # 6.结束日志和任务状态的配置
        logger.info(f">>> [{function_name}]开始执行了。现在的状态为: {state}")
        add_done_task(state['task_id'], function_name)

    return state

if __name__ == "__main__":

    # 单元测试：验证PDF转MD全流程
    logger.info("===== 开始node_pdf_to_md节点单元测试 =====")

    from app.utils.path_util import PROJECT_ROOT
    logger.info(f"测试获取根地址：{PROJECT_ROOT}")

    test_pdf_name = os.path.join("doc", "hak180产品安全手册.pdf")
    test_pdf_path = os.path.join(PROJECT_ROOT, test_pdf_name)

    # 构造测试状态
    test_state = create_default_state(
        task_id="test_pdf2md_task_001",
        pdf_path=test_pdf_path,
        local_dir=os.path.join(PROJECT_ROOT, "output")
    )

    node_pdf_to_md(test_state)

    logger.info("===== 结束node_pdf_to_md节点单元测试 =====")