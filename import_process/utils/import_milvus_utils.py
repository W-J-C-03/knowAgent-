from pymilvus import DataType

from app.clients.milvus_utils import get_milvus_client
from app.conf.milvus_config import milvus_config
from app.core.logger import logger

def step2_prepare_collection(state):
    # 1.获取milvus的客户端
    milvus_client = get_milvus_client()
    # 2.判断是否存在集合（表），存在创建集合（表）
    if not milvus_client.has_collection(collection_name=milvus_config.item_name_collection):
        # 没有集合，先创建集合
        # 创建集合对应的列信息
        schema = milvus_client.create_schema(
            auto_id=True,  # 主键自增长
            enable_dynamic_field=True  # 动态字段
        )
        # 这是在创建集合（表）的列信息
        schema.add_field(field_name="chunk_id", datatype=DataType.INT64, is_primary=True, auto_id=True)
        schema.add_field(field_name="file_title", datatype=DataType.VARCHAR, max_length=65535)
        schema.add_field(field_name="item_name", datatype=DataType.VARCHAR, max_length=65535)
        schema.add_field(field_name="content", datatype=DataType.VARCHAR, max_length=65535)
        schema.add_field(field_name="title", datatype=DataType.VARCHAR, max_length=65535)
        schema.add_field(field_name="parent_title", datatype=DataType.VARCHAR, max_length=65535)
        schema.add_field(field_name="part", datatype=DataType.INT8)
        schema.add_field(field_name="dense_vector", datatype=DataType.FLOAT_VECTOR, dim=1024)
        schema.add_field(field_name="sparse_vector", datatype=DataType.SPARSE_FLOAT_VECTOR)
        # 创建索引
        index_params = milvus_client.prepare_index_params()
        # 仅给稠密图和稀疏图创建索引
        index_params.add_index(
            field_name="dense_vector",  # 给特定列创建索引
            index_name="dense_vector_index",  # 索引名
            index_type="HNSW",  # 配置查找所用的算法，还有一个是ivf
            metric_type="COSINE",
            params={"M": 32, "efConstruction": 300}
        )
        index_params.add_index(
            field_name="sparse_vector",  # 给特定列创建索引
            index_name="sparse_vector_index",  # 索引名
            index_type="SPARSE_INVERTED_INDEX",  # 配置查找所用的算法，还有一个是HNSW
            metric_type="IP",
            params={"inverted_index_algo": "DAAT_MAXSCORE"}
        )
        milvus_client.create_collection(
            collection_name=milvus_config.chunks_collection,
            schema=schema,
            index_params=index_params
        )
    return milvus_client

def step3_delete_old_data(milvus_client, item_name):
    """
    删除旧数据 根据item_name删除
    :param milvus_client:
    :param item_name:
    :return:
    """
    milvus_client.delete(collection_name=milvus_config.chunks_collection, filter=f"item_name=='{item_name}'")
    milvus_client.load_collection(collection_name=milvus_config.chunks_collection)

def step4_insert_collection(milvus_client, chunks):
    """
    插入数据
    :param milvus_client:
    :param chunks:
    :return: chunks -》主键回显
    """
    insert_result = milvus_client.insert(collection_name=milvus_config.chunks_collection, data=chunks)
    # 成功插入了几条
    insert_count = insert_result.get("insert_count",0)
    logger.info(f"完成了数据插入，成功插入了{insert_count}条数据")

    # 获取回显的ids
    ids = insert_result.get("ids",[])
    if ids and len(ids) == len(chunks):
        for index, chunk in enumerate(chunks):
            chunk['chunk_id'] = ids[index]
    else:
        logger.warning(f"主键回填失败：生成ID数({len(ids)})与切片数({len(chunks)})不一致")
    return chunks