"""
生成集成模块
"""
import logging
import os
import time
from csv import excel
from pickletools import long1
from typing import List

from click import prompt
from langchain_core.documents import Document
from openai import OpenAI
from sympy.polys.polyconfig import query

logger = logging.getLogger(__name__)

class GenerationIntegrationModule:
    """
    生成集成模块，复杂答案生成
    """

    def __init__(self, model_name: str = "kimi-k2-0711-preview", temperature: float = 0.1, max_tokens: int = 1024):
        """初始化生成集成模块"""
        self.model_name = model_name
        self.temperature = temperature
        self.temperature = temperature
        self.max_tokens = max_tokens

        # 统一的LLM客户端配置（支持所有兼容OpenAI格式的供应商）
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("请设置 OPENAI_API_KEY 环境变量")

        self.base_url = os.getenv("OPENAI_BASE_URL", "https://api.moonshot.cn/v1")

        self.client = OpenAI(
            api_key=api_key,
            base_url=self.base_url
        )

        logger.info(f"生成模型初始化完成，模型: {model_name}, API地址: {self.base_url}")

    def _build_prompt(self, question: str, context: str) -> str:
        """
        构建统一的提示词
        :param question:
        :param context:
        :return:
        """
        return f"""
        作为一位专业的烹饪助手，请给予一下信息回答用户的信息
        
        检索到的相关信息
        {context}
        
        用户问题: {question}
        
        请提供准确、使用的回答。根据问题的性质：
        - 如果是询问多个菜品，请提供清晰的列表
        - 如果是询问制作方式，请提供详细的制作步骤
        - 如果是一般性咨询，请提供综合性回答
        
        重要提醒：如果问题涉及之前对话中提到的具体菜谱或食材，请严格给予之前提供的回答，不要添加之前没有提到的食材或调料。
        
        回答：
        """

    def generate_adaptive_answer(self, question: str, documents: List[Document]) -> str:
        """
        智能统一答案生成
        自动适应不同类型的查询，无需预先分类
        :param question:
        :param documents:
        :return:
        """
        # 构建上下文
        context_parts = []

        for doc in documents:
            content = doc.page_content.strip()
            if content:
                # 添加检索层级信息（如果有的话）
                level = doc.metadata.get('retrieval_level', '')
                if level:
                    context_parts.append(f"[{level.upper()}] {content}")
                else:
                    context_parts.append(content)

        # 使用统一的提示词构建方法
        context = "\n\n".join(context_parts)

        prompt = self._build_prompt(question, context)

        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )

            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"智能统一答案生成失败: {e}")
            return f"抱歉，生成回答时出现错误：{str(e)}"

    def generate_adaptive_answer_stream(self, question: str, documents: List[Document], max_retries: int = 3):
        """
        LightRag风格的流式答案生成（带重试机制）
        :param question:
        :param documents:
        :param max_retries: 最大重试次数
        :return:
        """
        context_parts = []

        for doc in documents:
            content = doc.page_content.strip()
            if content:
                level = doc.metadata.get('retrieval_level', '')
                if level:
                    context_parts.append(f"[{level.upper()}] {content}")
                else:
                    context_parts.append(content)

        context = "\n\n".join(context_parts)

        # 使用统一的提示词构建方法
        prompt = self._build_prompt(question, context)

        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "user", "content": prompt}
                    ],
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    stream=True,
                    timeout=60  # 增加超时设置
                )

                if attempt == 0:
                    print("开始流式生成回答...\n")
                else:
                    print(f"第{attempt + 1}次尝试流式生成...\n")

                full_response = ""
                for chunk in response:
                    if chunk.choices[0].delta.content:
                        content = chunk.choices[0].delta.content
                        full_response += content
                        yield content # 使用yield返回流式内容

                # 如果成功完成，退出重试循环
                return
            except Exception as e:
                logger.error(f"流式生成第{attempt + 1}次尝试失败: {e}")

                if attempt < max_retries - 1:
                    # 递增等待时间
                    wait_time = (attempt + 1) * 2
                    print(f"⚠️连接中断，{wait_time}秒后重试...")
                    time.sleep(wait_time)
                    continue
                else:
                    # 所有重试都失败，使用非流失作为降级方案
                    logger.error(f"流失生成完全失败，尝试非流失后备方案")
                    print("⚠️流失生成失败，切换到标准模式...")

                    try:
                        fallback_response = self.generate_adaptive_answer(question, documents)
                        yield fallback_response
                        return
                    except Exception as fallback_error:
                        logger.error(f"后备生成也失败: {fallback_error}")
                        error_msg = f"抱歉，生成回答时出现网络问题，请稍后重试。错误信息: {str(e)}"
                        yield error_msg
                        return
