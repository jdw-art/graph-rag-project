"""
图数据库数据准备模块
"""
import logging
from dataclasses import dataclass
from typing import List, Dict, Any

from langchain_core.documents import Document
from neo4j import GraphDatabase
from sklearn.externals.array_api_compat.torch import result_type
from sympy.physics.units import amount

logger = logging.getLogger(__name__)

@dataclass
class GraphNode:
    """图节点数据结构"""
    node_id: str
    labels: List[str]
    name: str
    properties: Dict[str, Any]

class GraphRelation:
    """图关系数据结构"""
    start_node_id: str
    end_node_id: str
    relation_type: str
    properties: Dict[str, Any]

class GraphDataPreparationModule:
    """图数据库数据准备模块，从Neo4j中读取数据并转换为文档"""

    def __init__(self, uri: str, user: str, password: str, database: str = "neo4j"):
        """初始化图数据库连接"""
        self.uri = uri
        self.user = user
        self.password = password
        self.database = database
        self.driver = None
        self.documents: List[Document] = [] # 文档
        self.chunks: List[Document] = []    # 分块
        self.recipes: List[GraphNode] = []  # 菜谱
        self.ingredients: List[GraphNode] = []  # 食材
        self.cooking_step: List[GraphNode] = [] # 烹饪步骤

        self._connect()

    def _connect(self):
        """建立Neo4j连接"""
        try:
            self.driver = GraphDatabase.driver(
                self.uri,
                auth=(self.user, self.password),
                database=self.database
            )
            logger.info(f"已连接到Neo4j数据库：{self.uri}")

            # 测试连接
            with self.driver.session() as session:
                result = session.run("RETURN 1 AS TEST")
                test_result = result.single()
                if test_result:
                    logger.info("Neo4j连接测试成功")

        except Exception as e:
            logger.error(f"连接Neo4j失败：{e}")

    def close(self):
        """关闭数据库连接"""
        if hasattr(self, "driver") and self.driver:
            self.driver.close()
            logger.info("Neo4j连接已关闭")

    def load_graph_data(self) -> Dict[str, Any]:
        """
        从Neo4j中加载图数据
        :return: 包含节点和关系的数据字典
        """

        logger.info("正在从Neo4j中加载图数据")

        with self.driver.session() as session:
            # 加载所有菜谱节点，从category关系中读取分类信息
            # “找出 nodeId ≥ 200000000 的所有菜谱节点，
            # 把它们通过 BELONGS_TO 关系拿到的分类（可能多个）合并成 mainCategory / allCategories 两个字段返回；
            # 如果没有分类关系，就退而用节点自身的 category 属性，连属性也没有就给‘未知’。”
            recipes_query = """
            MATCH (r:Recipe)
            WHERE r.nodeId >= '200000000'
            OPTIONAL MATCH (r)-[:BELONGS_TO]->(c:Category)
            WITH r, collect(c.name) as categories
            RETURN r.nodeId as nodeId, labels(r) as labels, r.name as name,
                    properties(r) as originalProperties
                    CASE WHEN size(categories) > 0
                        THEN categories[0]
                        ELSE COALESCE(r.category, '未知') END as mainCategory,
                    CASE WHEN size(categories) > 0
                        THEN categories
                        ELSE [COALESCE(r.category, '未知')] END as allCategories
            ORDER BY r.nodeId
            """

            result = session.run(recipes_query)
            self.recipes = []

            for record in result:
                # 合并原始数据和新的分类信息
                properties = dict(record["originalProperties"])
                properties["category"] = record["mainCategory"]
                properties["all_categories"] = record["allCategory"]

                node = GraphNode(
                    node_id=record["nodeId"],
                    labels=record["labels"],
                    name=record["name"],
                    properties=properties
                )
                self.recipes.append(node)

            logger.info(f"加载了 {len(self.recipes)} 个菜谱节点")

            # 加载所有食材节点
            ingredients_query = """
            MATCH (i:Ingredient)
            WHERE i.nodeId > '200000000'
            RETURN i.nodeId as nodeId, labels(i) as labels, i.name as name,
                    properties(i) as properties
            ORDER BY i.nodeId
            """

            result = session.run(ingredients_query)
            self.ingredients = []

            for record in result:
                node = GraphNode(
                    node_id=record["nodeId"],
                    labels=record["labels"],
                    name=record["name"],
                    properties=record["properties"]
                )
                self.ingredients.append(node)

            logger.info(f"加载了 {len(self.ingredients)} 个食材节点")

            # 加载所有烹饪步骤节点
            steps_query = """
            MATCH (s:CookingStep)
            WHERE s.nodeId > '200000000'
            RETURN s.nodeId as nodeId, labels(s) as labels, s.name as name,
                    properties(s) as properties
            ORDER BY s.nodeId
            """

            result = session.run(steps_query)
            self.cooking_step = []
            for record in result:
                node = GraphNode(
                    node_id=record["nodeId"],
                    labels=record["labels"],
                    name=record["name"],
                    properties=record["properties"]
                )
                self.cooking_step.append(node)

            logger.info(f"加载了 {len(self.cooking_step)} 个烹饪步骤节点")

        return {
            'recipes': len(self.recipes),
            'ingredients': len(self.ingredients),
            'cooking_step': len(self.cooking_step)
        }


    def build_recipe_documents(self) -> List[Document]:
        """
        构建菜谱文档，集成相关食材和步骤信息
        :return: 结构化的菜谱文档列表
        """

        logger.info("正在构建菜谱文档...")

        documents = []
        with self.driver.session() as session:
            for recipe in self.recipes:
                try:
                    recipe_id = recipe.node_id
                    recipe_name = recipe.name

                    # 获取菜谱相关食材
                    ingredients_query = """
                    MATCH (r:Recipe {nodeId: $recipe_id})-[req:REQUIRES]->(i:Ingredient)
                    RETURN i.name as name, i.category as category,
                           req.amount as amount, req.unit as unit,
                           i.description as description
                    ORDER BY i.name
                    """

                    ingredients_result = session.run(ingredients_query)
                    ingredients_info = []
                    for ing_record in ingredients_result:
                        amount = ing_record.get("amount", "")
                        unit = ing_record.get("unit", "")
                        ingredients_text = f"{ing_record['name']}"
                        if amount and unit:
                            ingredients_text += f"({amount}{unit})"
                        if ing_record.get("description"):
                            ingredients_text += f" - {ing_record['description']}"
                        ingredients_info.append(ingredients_text)

                    # 获取菜谱的烹饪步骤
                    steps_query = """
                    MATCH (r:Recipe {nodeId: $recipe_id})-[c:CONTAINS_STEP]->(s:CookingStep)
                    RETURN s.name as name, s.description as description,
                           s.stepNumber as stepNumber, s.methods as methods,
                           s.tools as tools, s.timeEstimate as timeEstimate,
                           c.stepOrder as stepOrder
                    ORDER BY COALESCE(c.stepOrder, s.stepNumber, 999)
                    """

                    steps_result = session.run(steps_query, {"recipe_id": recipe_id})
                    steps_info = []
                    for step_record in steps_result:
                        step_text = f"步骤：{step_record['name']}"
                        if step_record.get("description"):
                            step_text += f"\n描述: {step_record['description']}"
                        if step_record.get("methods"):
                            step_text += f"\n方法: {step_record['methods']}"
                        if step_record.get("tools"):
                            step_text += f"\n工具: {step_record['tools']}"
                        if step_record.get("timeEstimate"):
                            step_text += f"\n时间: {step_record['timeEstimate']}"
                        steps_info.append(step_text)

                    # 构建完整的菜谱信息
                    content_parts = [f"# {recipe_name}"]

                    # 添加菜谱基本信息
                    if recipe.properties.get("description"):
                        content_parts.append(f"\n## 菜品描述\n{recipe.properties['description']}")
                    if recipe.properties.get("cuisineType"):
                        content_parts.append(f"\n菜系: {recipe.properties['cuisineType']}")
                    if recipe.properties.get("difficulty"):
                        content_parts.append(f"难度: {recipe.properties['difficulty']}是")
                    if recipe.properties.get("prepTime") or recipe.properties.get("cookTime"):
                        time_info = []
                        if recipe.properties.get("prepTime"):
                            time_info.append(f"准备时间: {recipe.properties['prepTime']}")
                        if recipe.properties.get("cookTime"):
                            time_info.append(f"烹饪时间: {recipe.properties['cookTime']}")
                        content_parts.append(f"\n时间信息: {', '.join(time_info)}")
                    if recipe.properties.get("servings"):
                        content_parts.append(f"份量: {recipe.properties['servings']}")

                    # 添加食材信息
                    if ingredients_info:
                        content_parts.append("\n## 所需食材")
                        for i, ingredient in enumerate(ingredients_info, 1):
                            content_parts.append(f"{i}. {ingredient}")

                    # 添加步骤信息
                    if steps_info:
                        content_parts.append(f"\n## 制作步骤")
                        for i, step in enumerate(steps_info, 1):
                            content_parts.append(f"\n## 第{i}步\n{step}")

                    # 添加标签信息
                    if recipe.properties.get("tags"):
                        content_parts.append(f"\n## 标签\n{recipe.properties['tags']}")

                    # 组合成最终内容
                    full_content = "\n".join(content_parts)

                    # 创建文档对象
                    doc = Document(
                        page_content=full_content,
                        metadata={
                            "node_id": recipe_id,
                            "recipe_name": recipe_name,
                            "node_type": "Recipe",
                            "category": recipe.properties.get("category", "未知"),
                            "cuisine_type": recipe.properties.get("cuisineType", "未知"),
                            "difficulty": recipe.properties.get("difficulty", 0),
                            "prep_time": recipe.properties.get("prepTime", ""),
                            "cook_time": recipe.properties.get("cookTime", ""),
                            "servings": recipe.properties.get("servings", ""),
                            "ingredients_count": len(ingredients_info),
                            "steps_count": len(steps_info),
                            "doc_type": "recipe",
                            "content_length": len(full_content)
                        }
                    )

                    documents.append(doc)

                except Exception as e:
                    logger.warning(f"构建菜谱文档失败 {recipe_name} (ID: {recipe_id}): {e}")
                    continue

        self.documents = documents
        logger.info(f"成功构建 {len(self.documents)} 个文档")
        return documents



