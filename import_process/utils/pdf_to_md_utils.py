import shutil
import time
import zipfile
from pathlib import Path

import requests

from app.conf.mineru_config import mineru_config
from app.import_process.agent.state import ImportGraphState
from app.core.logger import logger
from app.utils.path_util import PROJECT_ROOT


def step1_validate_paths(state: ImportGraphState):
    """
    进行路径校验！pdf_path失效，直接异常处理
               local_dir没有，给予默认值
    :param state:
    :return:
    """
    logger.info(f">>> [step1_validate_paths]开始执行了。开始进行文件格式校验")
    pdf_path = state['pdf_path']
    local_dir = state['local_dir']
    # 常规的非空校验
    if not pdf_path:
        logger.error("[step1_validate_paths]检查发现没有输入文件，无法继续解析")
        raise ValueError("[step1_validate_paths]检查发现没有输入文件，无法继续解析")
    if not local_dir:
        # 给予一个默认值
        local_dir = str(PROJECT_ROOT / "output")
        logger.info(f">>> [step1_validate_paths]检查发现local_dir没有赋值，给予默认值{local_dir}")

    # 进行文件校验
    pdf_path_obj = Path(pdf_path) # 绝对路径
    local_dir_obj = Path(local_dir)
    if not pdf_path_obj.exists():
        logger.error("[step1_validate_paths]检查发现pdf_path不存在，请检查输入文件格式是否正确")
        raise ValueError("[step1_validate_paths]检查发现pdf_path不存在，请检查输入文件格式是否正确")
    if not local_dir_obj.exists():
        logger.error("[step1_validate_paths]检查发现local_dir不存在，主动创建对应的文件夹")
        local_dir_obj.mkdir(parents=True, exist_ok=True)
    return pdf_path_obj, local_dir_obj
def step2_upload_and_poll(pdf_path_obj) -> str:
    """
    将pdf文件使用minerU解析，并且获取md对应的下载的url地址
    :param pdf_path_obj:上传解析后md文件zip压缩包的下载地址
    :return: str -> url，minerU解析后md文件zip压缩包的下载地址
    """
    # 1.申请上传解析的地址
    # 前置准备和参数：url api | token | 准备固定格式的请求头
    token = mineru_config.api_key
    url = f"{mineru_config.base_url}/file-urls/batch"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {token}"
    }
    data = {
        "files":[
            {"name": f"{pdf_path_obj.name}"}
        ],
        "model_version":"vlm"
    }
    response = requests.post(url, headers=headers, json=data)
    # 结果处理 请求http状态码不是200或者code不是0，则请求失败
    if response.status_code != 200:
        logger.error(f"[step2_upload_and_poll]请求minerU解析接口失败，请检查输入文件路径是否正确")
        raise RuntimeError(f"[step2_upload_and_poll]请求minerU解析接口失败，请检查输入文件路径是否正确")
    upload_url = response.json()["data"]['file_urls'][0] # 在这个地址上传文件
    batch_id = response.json()["data"]['batch_id'] # 处理id，后续根据这个id获取结果
    # 2.将文件上传到对应的解析地址
    # 不能直接使用put上传，因为可能会有代理干扰
    # 处理代理
    http_session = requests.Session()
    http_session.trust_env = False
    try:
        with open(pdf_path_obj, 'rb') as f:
            file_data = f.read()
        upload_response = http_session.put(upload_url, data=file_data)
        if upload_response.status_code != 200:
            logger.error(f"[step2_upload_and_poll]上传文件到minerU失败，请检查输入文件路径是否正确")
            raise RuntimeError(f"[step2_upload_and_poll]上传文件到minerU失败，请检查输入文件路径是否正确")
    except Exception as e:
        logger.error(f"[step2_upload_and_poll]上传文件到minerU失败，请检查输入文件路径是否正确")
        raise RuntimeError(f"[step2_upload_and_poll]上传文件到minerU失败，请检查输入文件路径是否正确")
    finally:
        http_session.close()

    # 3.轮询获取解析结果
    # 循环获取，确保获取到结果
    # 设计一个循环，3秒获取一次，最多等待10分钟600秒 -> 600页pdf
    url = f"{mineru_config.base_url}/extract-results/batch/{batch_id}"
    timeout_seconds = 600
    poll_interval = 3
    state_time = time.time()
    while True:
        # 3.1 超时判断 不能站在第一次角度
        if time.time() - state_time > timeout_seconds:
            logger.error(f"[step2_upload_and_poll]请求minerU解析接口超时，请检查输入文件路径是否正确")
            raise RuntimeError(f"[step2_upload_and_poll]请求minerU解析接口超时，请检查输入文件路径是否正确")
        # 3.2 向指定url地址获取本次解析的结果
        res = requests.get(url, headers=headers)
        # 3.3 解析结果判断和获取zip_url
        if res.status_code != 200:
            # 可以给机会或抛异常，看程度
            if 500 <= res.status_code < 600: # 程度小，给机会
                time.sleep(poll_interval)
                continue
            # 程度大
            raise RuntimeError(f"[step2_upload_and_poll]请求minerU解析接口失败，返回的状态码{res.status_code}")
        json_data = res.json() # 获取本次结果
        if json_data['code'] != 0:
            # 很可能没有token了
            raise RuntimeError(f"[step2_upload_and_poll]请求minerU解析接口失败，返回的错误：{json_data['code']}信息{json_data['msg']}")
        # 判断下解析状态
        extract_result = json_data['data']['extract_result'][0]
        if extract_result['state'] == 'done':
            # 解析完毕可以获取结果
            full_zip_url = extract_result['full_zip_url']
            return full_zip_url
        else:
            time.sleep(poll_interval)


def step3_download_and_extract(zip_url, local_dir_obj, stem):
    """
    下载指定的md.zip文件，并且解压，返回解压后的md文件的地址
    :param zip_url: 要下载的zip所存放的地址
    :param local_dir_obj: 存储的文件夹
    :param stem: pdf的文件名字
    :return: 返回的md文件的地址
    """
    # 1.下载zip文件的response响应体
    response = requests.get(zip_url)
    if response.status_code != 200:
        logger.error(f"[step3_download_and_extract]下载文件失败，请检查输入文件路径是否正确")
        raise RuntimeError(f"[step3_download_and_extract]下载文件失败，请检查输入文件路径是否正确")
    # 2.将响应体的zip文件保存到本地
    zip_save_path = local_dir_obj / f"{stem}_result.zip"
    with open(zip_save_path, 'wb') as f:
        f.write(response.content)
    logger.info(f"[step3_download_and_extract]下载文件成功，保存位置：{zip_save_path}")
    # 3.清空旧目录（上次执行的内容删除）
    extract_target_dir = local_dir_obj / stem
    if extract_target_dir.exists():
        shutil.rmtree(extract_target_dir)
    # 创建一个新目录
    extract_target_dir.mkdir(parents=True, exist_ok=True)
    # 4.进行zip文件的解压工作
    # 利用python zip解压缩模块zipflie进行zip压缩和解压
    with zipfile.ZipFile(zip_save_path, 'r') as zip_ref:
        zip_ref.extractall(extract_target_dir)

    # 5.返回md文件的地址
    # 解压后的文件名可能不叫文件.md，也可能叫full.md，也可能有多个文件
    md_file_list = list(extract_target_dir.rglob('*.md'))
    target_md_file = None # 存储最终md文件
    if len(md_file_list) == 0:
        logger.error(f"[step3_download_and_extract]下载文件失败，请检查输入文件路径是否正确")
        raise RuntimeError(f"[step3_download_and_extract]下载文件失败，请检查输入文件路径是否正确")
    # 检查有没有原文件（名字相同）的md
    for md_file in md_file_list:
        if md_file.stem == stem:
            target_md_file = md_file
            break
    # 检查有没有full.md（上面没有命中，才进行这一步）
    if not target_md_file:
        for md_file in md_file_list:
            if md_file.name.lower() == "full.md":
                target_md_file = md_file
                break
    # 上面都没有，则获取第一个md文件
    if not target_md_file:
        target_md_file = md_file_list[0]

    # 统一改名字-》==stem.md
    if target_md_file.stem != stem:
        target_md_file = target_md_file.rename(target_md_file.with_name(f"{stem}.md"))

    # 最终文件获取绝对路径，并且返回字符串
    final_md_str_path = str(target_md_file.resolve())
    logger.info(f"[step3_download_and_extract]完成md解压，最终存储路径为：{final_md_str_path}")
    return final_md_str_path









