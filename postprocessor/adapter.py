"""
适配器 - 确保预处理和后处理之间的兼容性
"""
from typing import Dict, List, Any, Optional


class ProcessingAdapter:
    """处理阶段之间的适配器，确保信息正确传递"""

    @staticmethod
    def prepare_context_for_postprocessing(context: Any, translation_result: str) -> Dict:
        """
        准备用于后处理的上下文对象

        Args:
            context: 预处理生成的上下文对象
            translation_result: 翻译引擎的输出结果

        Returns:
            适合后处理使用的上下文字典
        """
        postprocessing_context = {
            'translated_text': translation_result,
            'original_context': context
        }

        # 复制重要的预处理信息到顶层
        if hasattr(context, 'term_mappings'):
            postprocessing_context['term_mappings'] = context.term_mappings

        if hasattr(context, 'grammatical_info'):
            postprocessing_context['grammatical_info'] = context.grammatical_info
        elif hasattr(context, 'grammatical_analyses'):
            postprocessing_context['grammatical_analyses'] = context.grammatical_analyses

        if hasattr(context, 'sentences'):
            postprocessing_context['sentences'] = context.sentences

        if hasattr(context, 'buddhist_context'):
            postprocessing_context['buddhist_context'] = context.buddhist_context

        if hasattr(context, 'original_grammar_analysis'):
            postprocessing_context['original_grammar_analysis'] = context.original_grammar_analysis

        # 复制其他可能有用的信息
        for attr in ['is_verse', 'has_enumeration', 'has_parallel_structure',
                     'function_type', 'context_confidence']:
            if hasattr(context, attr):
                postprocessing_context[attr] = getattr(context, attr)

        return postprocessing_context

    @staticmethod
    def extract_info_from_context(context: Any, info_key: str, default=None) -> Any:
        """
        从上下文对象中安全地提取信息

        Args:
            context: 上下文对象
            info_key: 信息键名
            default: 默认值

        Returns:
            提取的信息或默认值
        """
        # 直接访问属性
        if hasattr(context, info_key):
            return getattr(context, info_key)

        # 作为字典访问
        if isinstance(context, dict) and info_key in context:
            return context[info_key]

        # 检查原始上下文
        if hasattr(context, 'original_context'):
            return ProcessingAdapter.extract_info_from_context(
                context.original_context, info_key, default
            )

        # 检查嵌套字典
        if isinstance(context, dict) and 'original_context' in context:
            return ProcessingAdapter.extract_info_from_context(
                context['original_context'], info_key, default
            )

        return default

    @staticmethod
    def get_grammatical_analyses(context: Any) -> List[Dict]:
        """获取语法分析信息"""
        # 尝试从grammatical_info获取
        grammatical_info = ProcessingAdapter.extract_info_from_context(context, 'grammatical_info')
        if grammatical_info and 'sentences' in grammatical_info:
            return [s.get('grammatical_analysis', {}) for s in grammatical_info['sentences']]

        # 尝试从grammatical_analyses获取
        analyses = ProcessingAdapter.extract_info_from_context(context, 'grammatical_analyses')
        if analyses:
            return analyses

        return []

    @staticmethod
    def get_case_particles(context: Any, sentence_index: int = 0) -> List[Dict]:
        """获取特定句子的格助词信息"""
        analyses = ProcessingAdapter.get_grammatical_analyses(context)
        if 0 <= sentence_index < len(analyses):
            return analyses[sentence_index].get('case_particles', [])
        return []

    @staticmethod
    def get_tense(context: Any, sentence_index: int = 0) -> Optional[str]:
        """获取特定句子的时态信息"""
        analyses = ProcessingAdapter.get_grammatical_analyses(context)
        if 0 <= sentence_index < len(analyses):
            return analyses[sentence_index].get('tense')
        return None

    @staticmethod
    def get_syntactic_roles(context: Any, sentence_index: int = 0) -> List[Dict]:
        """获取特定句子的句法角色信息"""
        analyses = ProcessingAdapter.get_grammatical_analyses(context)
        if 0 <= sentence_index < len(analyses):
            return analyses[sentence_index].get('syntactic_roles', [])
        return []

    @staticmethod
    def get_sentence_type(context: Any, sentence_index: int = 0) -> str:
        """获取特定句子的类型"""
        analyses = ProcessingAdapter.get_grammatical_analyses(context)
        if 0 <= sentence_index < len(analyses):
            return analyses[sentence_index].get('type', 'declarative')
        return 'declarative'

    @staticmethod
    def get_buddhist_context(context: Any) -> str:
        """获取佛教语境类型"""
        return ProcessingAdapter.extract_info_from_context(
            context, 'buddhist_context', 'GENERAL'
        )

    @staticmethod
    def get_term_mappings(context: Any) -> Dict:
        """获取术语映射"""
        return ProcessingAdapter.extract_info_from_context(
            context, 'term_mappings', {}
        )

    @staticmethod
    def is_verse(context: Any) -> bool:
        """检查文本是否为诗偈格式"""
        return ProcessingAdapter.extract_info_from_context(
            context, 'is_verse', False
        )

    @staticmethod
    def has_enumeration(context: Any) -> bool:
        """检查文本是否包含列举结构"""
        return ProcessingAdapter.extract_info_from_context(
            context, 'has_enumeration', False
        )

    # 在 ProcessingAdapter 类中添加这些方法

    @staticmethod
    def get_original_text(context: Any) -> str:
        """获取原始藏文文本"""
        return ProcessingAdapter.extract_info_from_context(
            context, 'original_text', ''
        )

    @staticmethod
    def get_sentence_boundaries(context: Any) -> List[int]:
        """获取句子边界信息"""
        return ProcessingAdapter.extract_info_from_context(
            context, 'sentence_boundaries', []
        )

    @staticmethod
    def get_confidence_scores(context: Any) -> Dict:
        """获取各项分析的置信度分数"""
        return ProcessingAdapter.extract_info_from_context(
            context, 'confidence_scores', {}
        )

    @staticmethod
    def get_sentence_info(context: Any, sentence_index: int) -> Dict:
        """获取指定句子的完整信息"""
        sentences = ProcessingAdapter.extract_info_from_context(context, 'sentences', [])
        if 0 <= sentence_index < len(sentences):
            return sentences[sentence_index]
        return {}

    @staticmethod
    def get_terms(context: Any) -> List[Dict]:
        """获取识别出的术语列表"""
        return ProcessingAdapter.extract_info_from_context(context, 'terms', [])

    @staticmethod
    def has_parallel_structure(context: Any) -> bool:
        """检查文本是否包含平行结构"""
        return ProcessingAdapter.extract_info_from_context(
            context, 'has_parallel_structure', False
        )