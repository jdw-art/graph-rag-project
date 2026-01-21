""" Milvus 索引构建模块 """
import logging
import time
from typing import List

from accelerate.test_utils.scripts.external_deps.test_ds_alst_ulysses_sp import model_name
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from pymilvus import MilvusClient, CollectionSchema, FieldSchema, DataType

logger = logging.getLogger(__name__)

class MilvusIndexConstructionModule:
    """
    Milvus索引构建模块，负责向量化和Milvus索引构建
    """

    def __init__(self,
                 host: str = "localhost",
                 port: int = 19530,
                 collection_name: str = "cooking_knowledge",
                 dimension: int = 512,
                 model_name: str = "BAAI/bge-small-zh-v1.5"):
        """
        初始化Milvus索引构建模块
        :param host: 服务器地址
        :param port: 服务器端口
        :param collection_name: 集合名称
        :param dimension: 向量唯独
        :param model_name: 嵌入模型
        """
        self.host = host
        self.port = port
        self.collection_name = collection_name
        self.dimension = dimension
        self.model_name = model_name

        self.client = None
        self.embeddings = None
        self.collection_created = None

        self._setup_client()
        self._setup_embeddings()

    def _safe_truncate(self, text: str, max_length: int) -> str:
        """
        安全截取字符串，处理None值
        :param text: 输入文本
        :param max_length: 最大长度
        :return: 截取后的字符串
        """

        if text is None:
            return ""
        return str(text)[:max_length]


    def _setup_client(self):
        """
        初始化Milvus客户端
        :return:
        """
        try:
            self.client = MilvusClient(
                uri=f"http://{self.host}:{self.port}"
            )
            logger.info(f"已连接到Milvus服务器: {self.host}:{self.port}")

            # 测试连接
            collections = self.client.list_collections()
            logger.info(f"连接成功，当前集合: {collections}")
        except Exception as e:
            logger.error(f"连接Mlivus失败: {e}")
            raise

    def _setup_embeddings(self):
        """
        初始化嵌入模型
        :return:
        """
        logger.info(f"正在初始化嵌入模型: {self.model_name}")

        self.embeddings = HuggingFaceEmbeddings(
            model_name=self.model_name,
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )

        logger.info("嵌入模型初始化完成")

    def _create_collection_schema(self) -> CollectionSchema:
        """
        创建集合模式
        :return: 集合模式对象
        """

        # 定义字段
        fields = [
            FieldSchema(name="id", dtype=DataType.VARCHAR, max_length=512, is_primary=True),
            FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=self.dimension),
            FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=15000),
            FieldSchema(name="node_id", dtype=DataType.VARCHAR, max_length=100),
            FieldSchema(name="recipe_name", dtype=DataType.VARCHAR, max_length=300),
            FieldSchema(name="node_type", dtype=DataType.VARCHAR, max_length=100),
            FieldSchema(name="category", dtype=DataType.VARCHAR, max_length=100),
            FieldSchema(name="cuisine_type", dtype=DataType.VARCHAR, max_length=200),
            FieldSchema(name="difficulty", dtype=DataType.INT64),
            FieldSchema(name="doc_type", dtype=DataType.VARCHAR, max_length=50),
            FieldSchema(name="chunk_id", dtype=DataType.VARCHAR, max_length=150),
            FieldSchema(name="parent_id", dtype=DataType.VARCHAR, max_length=100)
        ]

        # 创建集合模式
        schema = CollectionSchema(
            fields=fields,
            description="中式烹饪知识图谱向量集合"
        )

        return schema

    def create_collections(self, force_recreate: bool = False) -> bool:
        """
        创建Milvus集合
        :param force_recreate: 是否强制重新创建集合
        :return: 是否创建成功
        """

        try:
            # 检查集合是否存在
            if self.client.has_collection(self.collection_name):
                if force_recreate:
                    logger.info(f"删除已存在的集合: {self.collection_name}")
                    self.client.drop_collection(self.collection_name)
                else:
                    logger.info(f"集合 {self.collection_name} 已存在")
                    self.collection_created = True
                    return True

            # 创建集合
            schema = self._create_collection_schema()

            self.client.create_collection(
                collection_name=self.collection_name,
                schema=schema,
                metric_type="COSINE",   # 使用余弦相似度
                consistency_level="Strong"
            )

            logger.info(f"成功创建集合 {self.collection_name}")
            self.collection_created = True

            return True

        except Exception as e:
            logger.error(f"集合创建失败 {e}")
            return False

    def build_vector_index(self, chunks: List[Document]) -> bool:
        """
        构建向量索引
        :param chunks: 文档块列表
        :return: 是否构建成功
        """
        logger.info(f"正在构建Milvus向量索引，文档数量: {len(chunks)}...")

        if not chunks:
            raise ValueError("文档块列表不能为空")

        try:
            # 1. 创建集合（如果schema不兼容则强制重新创建）
            if not self.create_collections(force_recreate=True):
                return False
            # 2. 准备数据
            logger.info("正在生成向量embeddings...")
            texts = [chunk.page_content for chunk in chunks]
            vectors = self.embeddings.embed_documents(texts)

            # 3. 准备插入数据
            entities = []
            for i, (chunk, vector) in enumerate(zip(chunks, vectors)):
                entity = {
                    "id": self._safe_truncate(chunk.metadata.get("chunk_id", f"chunk_{i}"), 150),
                    "vector": vector,
                    "text": self._safe_truncate(chunk.page_content, 15000),
                    "node_id": self._safe_truncate(chunk.metadata.get("node_id", ""), 100),
                    "recipe_name": self._safe_truncate(chunk.metadata.get("recipe_name", ""), 300),
                    "node_type": self._safe_truncate(chunk.metadata.get("node_type", ""), 100),
                    "category": self._safe_truncate(chunk.metadata.get("category", ""), 100),
                    "cuisine_type": self._safe_truncate(chunk.metadata.get("cuisine_type", ""), 200),
                    "difficulty": int(chunk.metadata.get("difficulty", 0)),
                    "doc_type": self._safe_truncate(chunk.metadata.get("doc_type", ""), 50),
                    "chunk_id": self._safe_truncate(chunk.metadata.get("chunk_id", f"chunk_{i}"), 150),
                    "parent_id": self._safe_truncate(chunk.metadata.get("parent_id", ""), 100)
                }
                entities.append(entity)

            # 4. 批量写入数据
            logger.info("正在插入向量数据...")
            batch_size = 100
            for i in range(0, len(entities), batch_size):
                batch = entities[i:i + batch_size]
                self.client.insert(
                    collection_name=self.collection_name,
                    data=batch
                )
                logger.info(f"已插入 {min(i + batch_size, len(entities))}/{len(entities)} 条数据")

            # 5. 创建索引
            if not self.create_index():
                return False

            # 6. 加载集合到内存
            self.client.load_collection(self.collection_name)
            logger.info("集合已加载到内存")

            # 7. 等待索引构建完成
            logger.info("等待索引构建完成...")
            time.sleep(2)

            logger.info(f"索引构建完成，包含 {len(chunks)} 个向量")
            return True

        except Exception as e:
            logger.error(f"Milvus向量索引构建失败: {e}")
            return False

    def create_index(self) -> bool:
        """
        创建向量索引
        :return: 是否创建成功
        """
        try:
            if not self.collection_created:
                raise ValueError("请先创建集合")

            # 使用prepare_index_params创建正确的IndexParams对象
            index_params = self.client.prepare_index_params()

            # 添加向量字段索引
            index_params.add_index(
                field_name="vector",
                index_type="HNSW",
                metric_type="COSINE",
                params={
                    "M": 16,
                    "efConstruction": 200
                }
            )

            self.client.create_index(
                collection_name=self.collection_name,
                index_params=index_params
            )

            logger.info("向量索引创建成功")
            return True

        except Exception as e:
            logger.error(f"构建索引失败 {e}")
            return False

    def add_documents(self, new_chunks: List[Document]) -> bool:
        """
        向现有索引添加文档
        :param new_chunks: 新的文本分块
        :return: 是否添加成功
        """
        # todo: