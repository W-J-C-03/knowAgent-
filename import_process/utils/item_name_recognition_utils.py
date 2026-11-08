import os

from langchain_core.messages import HumanMessage, SystemMessage
from pymilvus import DataType
from sympy.physics.mechanics import system

from app.clients.milvus_utils import get_milvus_client
from app.conf.milvus_config import milvus_config
from app.core.load_prompt import load_prompt
from app.core.logger import logger
from app.lm.embedding_utils import generate_embeddings
from app.lm.lm_utils import get_llm_client

# --- 配置参数 (Configuration) ---
# 大模型识别商品名称的上下文切片数：取前5个切片，避免上下文过长导致大模型输入超限
DEFAULT_ITEM_NAME_CHUNK_K = 5
# 单个切片内容截断长度：防止单切片内容过长，占满大模型上下文
SINGLE_CHUNK_CONTENT_MAX_LEN = 800
# 大模型上下文总字符数上限：适配主流大模型输入限制，默认2500
CONTEXT_TOTAL_MAX_CHARS = 2500
def step1_get_chunks(state):
    """
    获取chunks和file_title，并检验是否为空处理
    :param state:
    :return:
    """
    chunks = state.get('chunks')
    file_title = state.get('file_title')
    if not chunks:
        raise ValueError("chunks没有值，无法继续进行，抛异常")
    if not file_title:
        # 从md_path获取文件名
        file_title = os.path.basename(state.get('md_path'))
    return chunks, file_title

def step2_build_context(chunks):
    """
    根据chunks切面的content内容进行拼接
    截取内容限制：1.最多截取前top个 2.最多字符不能超过CONTEXT_TOTAL_MAX_CHARS
    截取内容处理：
        切片：{1}，标题：{title}，内容：{content} \n\n
        切片：{2}，标题：{title}，内容：{content} \n\n
        切片：{3}，标题：{title}，内容：{content} \n\n
        切片：{4}，标题：{title}，内容：{content} \n\n
        切片：{5}，标题：{title}，内容：{content} \n\n
    :param chunks:
    :return:
    """
    # 前置工作
    parts = [] # 存储处理后的切片：{1}，标题：{title}，内容：{content} \n\n
    total_chars = 0
    # 循环处理content +判断
    for index, chunk in enumerate(chunks[:DEFAULT_ITEM_NAME_CHUNK_K], start=1):
        chunk_title = chunk['title']
        chunk_content = chunk['content']
        data = f"切片：{index}，标题：{chunk_title}，内容：{chunk_content}"
        parts.append(data)
    # 结果的转化：
    context = "\n\n".join(parts)
    final_context = context[:SINGLE_CHUNK_CONTENT_MAX_LEN] # 切片的长度小于最大值
    return final_context

def step3_call_llm(context, file_title):
    """
    模型调用：获取item_name
    使用file_title进行兜底
    :param context:
    :param file_title:
    :return:
    """
    # 1.构建提示词
    human_prompt = load_prompt("item_name_recognition", file_title=file_title, context=context)
    system_prompt = load_prompt("product_recognition_system")
    # 2.获取模型对象
    llm = get_llm_client(json_mode=False)
    # 3.执行调用
    messages = [
        HumanMessage(content=human_prompt),
        SystemMessage(content=system_prompt)
    ]
    response = llm.invoke(messages)
    # 4.阶段判断和兜底
    item_name = response.content
    if not item_name:
        item_name = file_title
    return item_name

def step4_update_chunks_and_state(state, item_name, chunks):
    """
    state[item_name] = item_name
    chunks -> {item_name:item_name}
    :param state:
    :param item_name:
    :param chunks:
    :return:
    """
    state['item_name'] = item_name
    for chunk in chunks:
        chunk['item_name'] = item_name
    state['chunks'] = chunks
    logger.info(f"完成了chunks和state[item_name]的赋值和修改")

def step5_generate_embedding(item_name):
    """
    根据item_name生成向量 -》稠密和稀疏
    :param item_name:
    :return: 稠密向量和稀疏向量
    """
    """
        generate_embeddings 自己封装的嵌入式模型生成向量的函数
            embedding list对应的向量 = model.encode_document(texts)传入的字符串list
            参数：生成向量的字符串 ["1","2","3"]
            返回结果：
                result = {
                    "dense":[1的稠密向量,2,3]
                    "sparse":[1的稀疏向量,2,3]
                } 仅取第一个稠密和第一个稀疏
    """
    result = generate_embeddings([item_name])
    dense_vector, sparse_vector = result['dense'][0], result['sparse'][0]
    return dense_vector, sparse_vector

def step6_save_to_vector_db(file_title, item_name, dense_vector, sparse_vector):
    """
        步骤 6: 将商品名称、文件标题、双向量持久化到Milvus向量数据库
        核心逻辑：
            1. 配置校验：检查Milvus连接地址和集合名配置，缺失则跳过
            2. 客户端获取：获取单例Milvus客户端，连接失败则跳过
            3. 集合初始化：无集合则创建（定义Schema+索引），有集合则直接使用（保留原有配置）
            4. 幂等性处理：删除同名商品数据，避免重复存储
            5. 数据插入：构造符合Schema的数据，非空向量才添加
            6. 集合加载：插入后强制加载集合，确保数据立即可查/Attu可见
        参数：
            state: 流程状态对象，用于最终状态同步
            file_title: 处理后的文件标题
            item_name: 识别后的商品名称（主键去重依据）
            dense_vector: 步骤5生成的稠密向量（1024维列表）
            sparse_vector: 步骤5生成的稀疏向量（字典格式）
    """
    # 1.获取milvus的客户端
    milvus_client = get_milvus_client()
    # 2.判断是否存在集合（表），存在创建集合（表）
    if not milvus_client.has_collection(collection_name=milvus_config.item_name_collection):
        # 没有集合，先创建集合
        # 创建集合对应的列信息
        schema = milvus_client.create_schema(
            auto_id=True, # 主键自增长
            enable_dynamic_field=True # 动态字段
        )
        # 这是在创建集合（表）的列信息：pk（主键），file_title，item_name，dense_vector，sparse_vector
        schema.add_field(field_name="pk",datatype=DataType.INT64, is_primary=True, auto_id=True)
        schema.add_field(field_name="file_title",datatype=DataType.VARCHAR, max_length=65535)
        schema.add_field(field_name="item_name",datatype=DataType.VARCHAR, max_length=65535)
        schema.add_field(field_name="dense_vector",datatype=DataType.FLOAT_VECTOR, dim=1024)
        schema.add_field(field_name="sparse_vector",datatype=DataType.SPARSE_FLOAT_VECTOR)
        # 创建索引
        index_params = milvus_client.prepare_index_params()
        # 仅给稠密图和稀疏图创建索引
        index_params.add_index(
            field_name="dense_vector", # 给特定列创建索引
            index_name="dense_vector_index", # 索引名
            index_type="HNSW", # 配置查找所用的算法，还有一个是ivf
            metric_type="COSINE",
            params = {"M": 16, "efConstruction":200}
        )
        index_params.add_index(
            field_name="sparse_vector",  # 给特定列创建索引
            index_name="sparse_vector_index",  # 索引名
            index_type="SPARSE_INVERTED_INDEX",  # 配置查找所用的算法，还有一个是HNSW
            metric_type="IP",
            params={"inverted_index_algo": "DAAT_MAXSCORE"}
        )
        milvus_client.create_collection(
            collection_name=milvus_config.item_name_collection,
            schema=schema,
            index_params=index_params
        )
    # 3.先删除之前存在的item_name
    # 加载选中集合
    milvus_client.load_collection(collection_name=milvus_config.item_name_collection)
    milvus_client.delete(collection_name=milvus_config.item_name_collection,
                         filter=f"item_name=='{item_name}'")
    # 4.向集合插入最新的item_name数据和对应的向量
    item = {
        "file_title": file_title,
        "item_name": item_name,
        "dense_vector": dense_vector,
        "sparse_vector": sparse_vector
    }
    milvus_client.insert(collection_name=milvus_config.item_name_collection,
                         data=[item])
    milvus_client.load_collection(collection_name=milvus_config.item_name_collection)
    logger.info(f"保存了item_name：{item_name}的数据到向量数据库中")





