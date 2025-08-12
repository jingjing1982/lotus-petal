"""
术语保护器 - 支持语境感知的术语保护和替换
"""
import re
from typing import Dict, List, Tuple, Optional, Any
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
            "{:03d}",  # 基本格式 - 修改为数字格式以匹配翻译结果
            "TERM{:03d}",  # 带前缀格式（用于内部处理）
            "佛TERM{:03d}",  # 带中文前缀
            "·TERM{:03d}·",  # 带标点符号
        ]

        # 文本语境信息
        self.text_context = None
        self.mixed_strategy = None

    def _safe_get(self, obj: Any, key: str, default: Any = None) -> Any:
        """安全获取对象属性或字典值"""
        if obj is None:
            return default

        # 尝试属性访问
        if hasattr(obj, key):
            return getattr(obj, key)

        # 尝试get方法访问
        if hasattr(obj, 'get'):
            return obj.get(key, default)

        # 尝试字典式访问
        try:
            return obj[key]
        except (TypeError, KeyError, IndexError):
            return default

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
        if self.mixed_strategy['is_mixed_context']:
            logger.info(f"Mixed text detected: primary={self.mixed_strategy['primary_context']}")

        # 按术语长度降序排序
        sorted_terms = sorted(identified_terms, key=lambda x: self._safe_get(x, 'length', 0), reverse=True)

        protected_text = text
        term_positions = []  # 记录术语位置信息

        for term in sorted_terms:
            # 使用安全访问方法获取属性
            term_text = self._safe_get(term, 'text', '')
            term_start = self._safe_get(term, 'start', 0)
            term_length = self._safe_get(term, 'length', len(term_text))
            term_type = self._safe_get(term, 'type', 'general')

            # 创建占位符
            placeholder = self._create_placeholder()

            # 获取术语上下文
            term_context = self._build_term_context(text, term_start, term_text, term_positions)

            # 记录映射关系
            self.protection_map[placeholder] = {
                'tibetan': term_text,
                'type': term_type,
                'position': term_start,
                'length': term_length,
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
            other_start = self._safe_get(other_term, 'start', 0)
            if abs(other_start - position) < 100:  # 100字符范围内
                surrounding_terms.append(self._safe_get(other_term, 'term', ''))

        # 确定该术语的语境
        # 如果是混合文本，需要更精确的判断
        if self.mixed_strategy and self.mixed_strategy['is_mixed_context']:
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
        # 使用简单数字格式的占位符，与翻译引擎输出匹配
        placeholder = self.placeholder_formats[0].format(self.placeholder_counter)
        self.placeholder_counter += 1
        return placeholder

    def restore_terms(self, text: str, context) -> str:
        """恢复被保护的术语"""
        # 使用适配器获取术语映射
        term_mappings = ProcessingAdapter.get_term_mappings(context)

        if not term_mappings:
            logger.warning("没有找到术语映射，无法恢复术语")
            return text

        # 添加日志以帮助调试
        logger.debug(f"开始恢复术语，映射表：{term_mappings}")
        logger.debug(f"恢复前文本：{text}")

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
            else:
                # 检查数字格式的占位符
                numeric_placeholder = placeholder.lstrip("TERM")
                if numeric_placeholder in restored_text:
                    term_info = term_mappings[placeholder]
                    chinese_term = self._select_best_translation(
                        restored_text,
                        placeholder,
                        term_info
                    )
                    restored_text = restored_text.replace(numeric_placeholder, chinese_term)
                    logger.debug(f"Restored numeric {numeric_placeholder} -> {chinese_term}")

        logger.debug(f"恢复后文本：{restored_text}")
        return restored_text

    def _select_best_translation(self, text: str, placeholder: str, term_info: Dict) -> str:
        """
        选择最佳的中文翻译

        Args:
            text: 当前文本
            placeholder: 占位符
            term_info: 术语信息

        Returns:
            最佳的中文翻译
        """
        # 获取术语信息
        tibetan = term_info.get('tibetan', '')
        if not tibetan:
            logger.warning(f"术语信息中没有藏文原文: {term_info}")
            return placeholder  # 如果没有藏文，保留占位符

        # 获取术语上下文
        context = term_info.get('context', {})

        # 尝试从数据库获取翻译
        if self.term_database:
            try:
                # 调用数据库获取最佳翻译
                translation, confidence = self.term_database.get_translation(
                    tibetan, context
                )

                if translation and confidence > 0.5:
                    return translation

                # 如果置信度不够，记录一下
                if translation:
                    logger.debug(f"找到翻译 '{translation}' 但置信度不足: {confidence}")

            except Exception as e:
                logger.error(f"获取术语翻译时出错: {e}")

        # 备选方案1: 从术语信息中获取预定义的中文翻译
        chinese = term_info.get('chinese', '')
        if chinese:
            return chinese

        # 备选方案2: 根据术语类型选择默认翻译
        term_type = term_info.get('type', 'general')
        detected_context = term_info.get('detected_context', 'GENERAL')

        # 常见佛教术语的默认翻译
        if term_type == 'buddha_name':
            return '佛陀'
        elif term_type == 'sutra_name':
            return '经'
        elif term_type == 'dharma_term':
            if detected_context == 'MADHYAMIKA':
                return '中观法门'
            elif detected_context == 'YOGACARA':
                return '唯识法门'
            else:
                return '佛法'

        # 实在没有合适的翻译，以藏文音译形式返回
        return f"[{tibetan}]"