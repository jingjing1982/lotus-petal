"""
术语保护器 - 支持语境感知的术语保护和替换
"""
import re
from typing import Dict, List, Tuple, Optional
import logging
from postprocessor.adapter import ProcessingAdapter

logger = logging.getLogger(__name__)


class TermProtector:
    def __init__(self, term_database):
        """初始化术语保护器"""

        self.term_database = term_database
        self.placeholder_counter = 0
        self.protection_map = {}

        # 占位符格式
        self.placeholder_formats = [
            "TERM{:03d}",  # 基本格式
            "佛TERM{:03d}",  # 带中文前缀
            "·TERM{:03d}·",  # 带标点符号
        ]

        # 文本语境信息
        self.text_context = None
        self.mixed_strategy = None

    def protect_terms(self, text: str, identified_terms: List[Dict]) -> Tuple[str, Dict]:
        """
        保护文本中的术语
        """
        self.protection_map = {}
        self.placeholder_counter = 0

        # 检测文本语境
        self.text_context = self.term_database.detect_text_context(text)
        self.mixed_strategy = self.term_database.get_mixed_context_strategy(text)

        logger.info(f"Detected context: {self.text_context[0]}, function: {self.text_context[1]}")
        if self.mixed_strategy['is_mixed']:
            logger.info(f"Mixed text detected: primary={self.mixed_strategy['primary_context']}")

        # 按术语长度降序排序
        sorted_terms = sorted(identified_terms, key=lambda x: x['length'], reverse=True)

        protected_text = text
        term_positions = []  # 记录术语位置信息

        for term in sorted_terms:
            term_text = term['text']
            term_start = term.get('start', 0)

            # 创建占位符
            placeholder = self._create_placeholder()

            # 获取术语上下文
            term_context = self._build_term_context(text, term_start, term_text, term_positions)

            # 记录映射关系
            self.protection_map[placeholder] = {
                'tibetan': term_text,
                'type': term.get('type', 'general'),
                'position': term_start,
                'length': term['length'],
                'context': term_context,
                'detected_context': term_context['detected_context'],
                'detected_function': term_context['detected_function']
            }

            # 记录位置信息
            term_positions.append({
                'start': term_start,
                'end': term_start + len(term_text),
                'term': term_text,
                'placeholder': placeholder
            })

            # 替换文本中的术语
            protected_text = protected_text.replace(term_text, placeholder, 1)

        return protected_text, self.protection_map

    def _build_term_context(self, text: str, position: int, term: str,
                            existing_terms: List[Dict]) -> Dict:
        """构建术语的详细上下文信息"""
        # 提取周围文本
        window = 50
        start = max(0, position - window)
        end = min(len(text), position + len(term) + window)

        surrounding_text = text[start:end]

        # 查找周围的其他术语
        surrounding_terms = []
        for other_term in existing_terms:
            if abs(other_term['start'] - position) < 100:  # 100字符范围内
                surrounding_terms.append(other_term['term'])

        # 确定该术语的语境
        # 如果是混合文本，需要更精确的判断
        if self.mixed_strategy and self.mixed_strategy['is_mixed']:
            # 检查局部语境
            local_context, local_function = self.term_database.detect_text_context(surrounding_text)

            # 如果局部语境明确，使用局部语境；否则使用主导语境
            if local_context != 'general':
                detected_context = local_context
                detected_function = local_function
            else:
                detected_context = self.mixed_strategy['primary_context']
                detected_function = self.text_context[1]
        else:
            detected_context = self.text_context[0]
            detected_function = self.text_context[1]

        return {
            'text': text,
            'position': position,
            'detected_context': detected_context,
            'detected_function': detected_function,
            'surrounding_text': surrounding_text,
            'surrounding_terms': surrounding_terms,
            'before': text[start:position],
            'after': text[position + len(term):end]
        }

    def _create_placeholder(self) -> str:
        """创建占位符"""
        placeholder = self.placeholder_formats[0].format(self.placeholder_counter)
        self.placeholder_counter += 1
        return placeholder

    def restore_terms(self, text: str, context) -> str:
        """恢复被保护的术语"""
        """恢复所有术语"""
        # 使用适配器获取术语映射
        term_mappings = ProcessingAdapter.get_term_mappings(context)

        if not term_mappings:
            return text

        restored_text = text

        # 按占位符长度降序排序，避免替换冲突
        sorted_placeholders = sorted(
            term_mappings.keys(),
            key=len,
            reverse=True
        )

        for placeholder in sorted_placeholders:
            if placeholder in restored_text:
                term_info = term_mappings[placeholder]

                # 获取最适合的中文翻译
                chinese_term = self._select_best_translation(
                    restored_text,
                    placeholder,
                    term_info
                )

                # 执行替换
                restored_text = restored_text.replace(placeholder, chinese_term)

                logger.debug(f"Restored {placeholder} -> {chinese_term}")

        return restored_text