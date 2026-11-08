import base64
import os, re
from collections import deque
from pathlib import Path
from typing import Tuple

from minio.deleteobjects import DeleteObject

from app.clients.minio_utils import get_minio_client
from app.conf.minio_config import minio_config
from app.conf.lm_config import lm_config
from app.core.load_prompt import load_prompt
from app.core.logger import logger

from app.import_process.agent.state import ImportGraphState
from app.lm.lm_utils import get_llm_client
from app.utils.rate_limit_utils import apply_api_rate_limit

# MinIO支持的图片格式集合
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp"}
def is_supported_image(filename:str) -> bool:
    """
    判断是否是MinIO支持的图片格式
    :param filename:
    :return:
    """
    return os.path.splitext(filename)[1].lower() in IMAGE_EXTENSIONS

def step1_get_content(state: ImportGraphState) -> Tuple[str, Path, Path]:
    """
    提取内容
    :param state:
    :return:
    """
    # 1.获取md的地址md_path
    md_file_path = state['md_path']
    if not md_file_path:
        raise ValueError("md_path不能为空")
    md_path_obj = Path(md_file_path)
    if not md_path_obj.exists():
        raise FileNotFoundError(f"md_path:{md_file_path}文件不存在")

    # 读取content （防止在入口阶段直接传入md文件，未经过pdf转换，致使content为空）
    if not state['md_content']:
        # 没有，再读取
        with md_path_obj.open('r', encoding='utf-8') as f:
            state['md_content'] = f.read()
    # 图片文件夹obj
    # 自己传入的md文件的图片文件夹命名必须是images
    images_dir_obj = md_path_obj.parent / 'images'
    return state['md_content'], md_path_obj, images_dir_obj


def find_image_in_md_context(md_content, image_file, context_length: int = 100):
    """
    从md_content识别图片的上下文
    约定上下文长度各为100
    :param context_length:
    :param md_content:
    :param image_file:
    :return:
    """
    # 定义正则表达式
    pattern = re.compile(r"!\[.*?\]\(.*?" + image_file + r".*?\)")
    result = [] # 存储图片多处使用，上下文不同，本次仅取第一个
    # 查询符合位置
    for item in pattern.finditer(md_content):
        start, end = item.span()
        # 截取上文
        pre_text = md_content[max(start-context_length, 0):start]
        # 截取下文
        post_text = md_content[end:min(end+context_length, len(md_content))]
        result.append((pre_text, post_text))
    # 截取位置前后的内容
    if result:
        logger.info(f"图片：{image_file} 上下文：{result[0]}")
        return result[0]
    return None

def step2_scan_images(md_content: str, images_dir_obj: Path):
    """
    进行md中图片识别，并且截取图片对应的上下文环境
    :param md_content:
    :param images_dir_obj:
    :return:
    """
    # 1.先创建一个目标集合
    targets = []
    # 2.循环读取images中的所有图片，判断是否在md中使用
    for image_file in os.listdir(images_dir_obj):
        # 遍历每个文件的名字
        # 检查图片是否可用
        if not is_supported_image(image_file):
            logger.warning(f"当前文件：{image_file}，不是图片格式，无需处理")
            continue
        # 是图片，在md中查询，看是否存在，如果存在，读取对应的上下文即可
        context_data = find_image_in_md_context(md_content, image_file)
        if not context_data:
            logger.warning(f"图片：{image_file}没有在md中使用，上下文为空")
            continue

        targets.append((image_file, str(images_dir_obj / image_file), context_data))
    return targets

def step3_generate_img_summeries(targets, stem):
    """
    获取图片的内容描述！利用视觉模型
    :param targets: [(图片名，图片地址，(上文，下文))]
    :param stem: md文件名
    :return: {图片名1：总结，图片名2：总结}
    """
    # 循环每一张图片，向视觉模型进行请求，获取总结结果
    summeries = {}
    # 解构
    request_times = deque()
    for image_file, image_path, context in targets:
        # 限速访问模型
        apply_api_rate_limit(request_times, max_requests=9)
        # 向视觉模型发送请求
        vm_model = get_llm_client(model=lm_config.lv_model)
        # 准备提示词
        prompt = load_prompt("image_summary", root_folder=stem, image_content=context)
        # import base64 :Base64 是一种把"二进制数据（图片、音频、文件）转成纯文本字符串"的编码方式。
        # 在这个项目里，它的作用是：把本地图片文件变成一串文本，塞进 messages 里发给多模态大模型（VLM），
        # 让模型能"看到"图片。
        with open(image_path, 'rb') as f:
            image_baser64 = base64.b64encode(f.read()).decode("utf-8")
        # 通用格式
        messages = [{
            "role":"user",
            "content":[{
                "type":"image_url",
                "image_url": {
                    "url":f"data:image/jpeg;base64,{image_baser64}"
                }
            },
                {"type": "text", "text": f"{prompt}"}
            ]
        }]
        # 执行获取总结
        response = vm_model.invoke(messages)
        summery = response.content.strip().replace("\n", "")
        summeries[image_file] = summery
    return summeries

def step4_upload_images_and_replace_md(summeries, targets, md_content, stem):
    """
    将我们图片传递到minio服务器
    替换原md中的图片和描述
    :param summeries: {图片名：描述}
    :param targets: （图片名，源地址，（上文，下文））
    :param md_content: 原md内容
    :param stem: 文件名
    :return: 新md
    """
    # minio存储结果：桶/upload-images/文件夹名/图片对象.jpg
    # 获取minio客户端
    minio_client = get_minio_client()
    # 1.删除minio中的对应文件的图片
    # 获取要删除的对象
    # [1:]用于去掉前面的/，因为在minio_img_dir变量前面带有/
    object_list = minio_client.list_objects(minio_config.bucket_name,
                              prefix=f"{minio_config.minio_img_dir[1:]}/{stem}",
                              recursive=True) # recursive用于递归
    # 需要的DeleteObject
    delete_object_list = [DeleteObject(object) for object in object_list]
    """
    部分成功机制：当你一次性传给它 100 个文件去删除时，MinIO 会尽量删。
    如果有 99 个成功，1 个失败，它不会直接抛出一个异常让程序崩溃。

    返回生成器：它会返回一个迭代器（Generator / Iterator）。
    你可以遍历这个返回值，如果遍历出内容了，说明有文件没删掉（包含了失败的文件名和错误原因）；
    如果遍历不出内容，说明全部删除成功。
    """
    errors = minio_client.remove_objects(minio_config.bucket_name, delete_objects=delete_object_list)
    for error in errors:
        logger.error(f"删除对象失败：{error}")
    logger.info(f"已经完成{stem}下的对象清空，本次删除了：{len(delete_object_list)}个对象")
    # 2.上传图片到minio服务器
    # 声明记录图片上传结果的字典
    image_url = {"图片名.xx":"url"}
    for image_file, image_path, _ in targets:
        try:
            minio_client.fput_object(
                bucket_name=minio_config.bucket_name,
                object_name=f"{minio_config.minio_img_dir}/{stem}/{image_file}",
                file_path=image_path,
                content_type="image/jpeg"
            )
            # 上传完毕记录
            # 图片地址 = http://ip:端口/桶名/对象名
            image_url[image_file] = f"https://{minio_config.endpoint}/{minio_config.bucket_name}{minio_config.minio_img_dir}/{stem}/{image_file}"
            logger.info(f"图片{image_file}完成上传，访问地址为：{image_url[image_file]}")
        except Exception as e:
            logger.error(f"图片{image_file}上传失败，失败原因：{str(e)}")
    # 3.md中图片的替换即可
    # summeries = 图片名：描述
    # images_url = 图片名：minio网络地址
    # 汇总：{图片名：（描述，地址）} 便于后续操作
    image_infos = {}
    for image_file, summery in summeries.items():
        if url := image_url.get(image_file):
            image_infos[image_file] = (summery, url)
    # 正事：替换内容
    if image_infos:
        for image_file, (summery, url) in image_infos.items():
            # 正则：![xx](图片地址/image_file) -> ![summery][minio的url]
            rep = re.compile(r"!\[.*?\]\(.*?"+ image_file+r".*?\)")
            md_content = rep.sub(f"![{summery}]({url})", md_content)
        logger.info(f"已经完成对md内容的替换")
    return md_content

def step5_replace_md_and_save(new_md_content, md_path_obj):
    """
    完成新的md的磁盘备份，并且返回新地址
    新的命名 xxx_new.md
    :param new_md_content: 新内容
    :param md_path_obj: 老地址
    :return: 新地址
    """
    # 设置新地址
    # 路径/文件名.md ->路径/文件名_new.md
    new_md_path_str = os.path.splitext(md_path_obj)[0] + "_new.md"
    with open(new_md_path_str, 'w', encoding="utf-8") as f:
        f.write(new_md_content)
    logger.info(f"已经完成了新内容的写入，新的地址为：{new_md_path_str}")
    return new_md_path_str