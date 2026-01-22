"""
智能查询路由器
根据查询特点自动选择最适合的检索策略：
- 传统混合检索：适合简单的信息查找
- 图RAG检索：适合复杂的关系推理和知识发现
"""
import json
import logging
from dataclasses import dataclass
from enum import Enum
from typing import Tuple, List
from xml.dom.minidom import Document

logger = logging.getLogger(__name__)

class SearchStrategy(Enum):
    """搜索策略枚举"""
    HYBRID_TRADITIONAL = "hybrid_traditional"   # 传统混合检索
    GRAPH_RAG = "graph_rag" # 图RAG检索
    COMBINED = "combined"   # 组合策略

@dataclass
class QueryAnalysis:
    """查询分析结果"""
    query_complexity: float     # 查询复杂度（0 - 1）
    relationship_intensity: float # 关系密集度（0 - 1）
    reasoning_required: bool    # 是否需要推理
    entity_count: int           # 实体数量
    recommended_strategy: SearchStrategy
    confidence: float           # 推荐置信度
    reasoning: str              # 推荐理由

class IntelligentQueryRouterModule:
    """
    智能查询路由器

    核心能力：
    1. 查询复杂度分析：识别简单查找 vs 复杂推理
    2. 关系密集度评估：判断是否需要图结构优势
    3. 策略自动选择：路由到最适合的检索引擎
    4. 结果质量监控：基于反馈优化路由决策
    """

    def __init__(self,
                 traditional_retrieval, # 传统混合检索
                 graph_retrieval,       # 图RAG检索
                 llm_client,
                 config):
        self.traditional_retrieval = traditional_retrieval
        self.graph_retrieval = graph_retrieval
        self.llm_client = llm_client
        self.config = config

        # 路由统计
        self.route_stats = {
            "traditional_count": 0,
            "graph_rag_count": 0,
            "combined_count": 0,
            "total_queries": 0
        }

    def analyze_query(self, query: str) -> QueryAnalysis:
        """
        深度分析查询特征，决定最佳检索策略
        :param query: 查询
        :return: 响应Json数据
        """
        analysis_prompt = f"""
        作为RAG系统的查询分析专家，请深度分析以下查询信息的特征：
        
        查询：{query}
        
        请从以下纬度分析：
        
        1. 查询复杂度 (0-1):
            - 0.0-0.3: 简单信息查找（如：红烧肉怎么做？）
            - 0.4-0.7: 中等复杂度（如：川菜有哪些特色菜？）
            - 0.8-1.0: 高复杂度推理（如：为什么川菜用花椒而不是胡椒？）
            
        2. 关系密集度 (0-1)：
           - 0.0-0.3: 单一实体信息（如：西红柿的营养价值）
           - 0.4-0.7: 实体间关系（如：鸡肉配什么蔬菜？）
           - 0.8-1.0: 复杂关系网络（如：川菜的形成与地理、历史的关系）
           
        3. 推理需求：
           - 是否需要多跳推理？
           - 是否需要因果分析？
           - 是否需要对比分析？
           
        4. 实体识别：
           - 查询中包含多少个明确实体？
           - 实体类型是什么？
           
        基于分析推荐检索策略：
        - hybrid_traditional: 适合简单直接的信息查找
        - graph_rag: 适合复杂关系推理和知识发现
        - combined: 需要两种策略结合
        
        返回JSON格式：
        {{
            "query_complexity": 0.6,
            "relationship_intensity": 0.8,
            "reasoning_required": true,
            "entity_count": 3,
            "recommended_strategy": "graph_rag",
            "confidence": 0.85,
            "reasoning": "该查询涉及多个实体间的复杂关系，需要图结构推理"
        }}
        """

        try:
            response = self.llm_client.chat.completions.create(
                model=self.config.llm_model,
                message=[{"role": "user", "content": query}],
                temperature=0.1,
                max_tokens=800
            )

            result = json.loads(response.choices[0].message.content.strip())

            analysis = QueryAnalysis(
                query_complexity=result.get("query_complexity", 0.5),
                relationship_intensity=result.get("relationship_intensity", 0.5),
                reasoning_required=result.get("reasoning_required", False),
                entity_count=result.get("entity_count", 1),
                recommended_strategy=SearchStrategy(result.get("recommended_strategy", "hybrid_traditional")),
                confidence=result.get("confidence", 0.5),
                reasoning=result.get("reasoning", "默认分析")
            )

            logger.info(f"查询分析完成: {analysis.recommended_strategy.value} (置信度: {analysis.confidence:.2f})")
            return analysis

        except Exception as e:
            logger.error(f"分析查询特征失败: {e}")
            self._rule_based_analysis(query)

    def _rule_based_analysis(self, query: str) -> QueryAnalysis:
        """
        基于规则的降级分析方案
        :param query: 查询
        :return:
        """

        # 简单的规则判断
        complexity_keywords = ["为什么", "如何", "关系", "影响", "原因", "比较", "区别"]
        relation_keywords = ["配", "搭配", "组合", "相关", "联系", "连接"]

        complexity = sum(1 for kw in complexity_keywords if kw in query) / len(complexity_keywords)
        relation_intensity = sum(1 for kw in relation_keywords if kw in query) / len(relation_keywords)

        if complexity > 0.3 or relation_intensity > 0.3:
            strategy = SearchStrategy.GRAPH_RAG
        else:
            strategy = SearchStrategy.HYBRID_TRADITIONAL

        return QueryAnalysis(
            query_complexity=complexity,
            relationship_intensity=relation_intensity,
            reasoning_required=complexity > 0.3,
            entity_count=len(query.split()),
            recommended_strategy=strategy,
            confidence=0.6,
            reasoning="基于规则的简单分析"
        )

    def route_query(self, query: str, top_k: int = 3) -> Tuple[List[Document], QueryAnalysis]:
        """
        智能路由查询到最合适的检索引擎
        :param query: 查询
        :param top_k: 前k个查询
        :return:
        """
        logger.info("开始智能路由...")

        # todo: 智能路由查询