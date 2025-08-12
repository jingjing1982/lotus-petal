"""
翻译管理器 - 协调整个翻译流程
"""
import re
import logging
import math
from typing import List, Dict, Optional, Tuple, Any
from preprocessor import BotokAnalyzer, TermProtector, ContextExtractor
from postprocessor import TermRestorer, GrammarCorrector, QualityController, TextFinalizer
from .nllb_wrapper import NLLBTranslator
from utils.term_database import FlexibleContextDetector, TermDatabase
from utils.helpers import TextUtils
from postprocessor.adapter import ProcessingAdapter

logger = logging.getLogger(__name__)

class TranslationManager:
    """翻译管理器类 - 负责协调整个翻译流程"""

    def __init__(self, config: Optional[Dict] = None):
        """初始化翻译管理器"""
        # 保存配置
        self.config = config or {}

        # 初始化基础配置参数
        self.sentence_split_threshold = self.config.get('sentence_split_threshold', 100)
        self.use_alternatives = self.config.get('use_alternatives', False)

        # 初始化数据库连接 - 先初始化数据库
        db_config = self.config.get('database', {})
        self.term_database = TermDatabase(db_config)

        # 初始化预处理组件
        self.botok_analyzer = BotokAnalyzer()
        self.term_protector = TermProtector(self.term_database)
        self.context_extractor = ContextExtractor()
        self.context_detector = FlexibleContextDetector()

        # 初始化翻译引擎
        self.translator = NLLBTranslator()

        # 初始化后处理组件
        self.term_restorer = TermRestorer(self.term_database)
        self.grammar_corrector = GrammarCorrector()
        self.quality_controller = QualityController(self.term_database)
        self.text_finalizer = TextFinalizer()

        # 将数据库连接传递给需要的组件
        self._setup_database_connections()

        logger.info("Translation Manager initialized")

    def _setup_database_connections(self):
        """设置各组件的数据库连接"""
        # 为需要数据库的组件设置连接
        components_need_db = [
            self.term_protector,
            self.quality_controller
        ]

        for component in components_need_db:
            if hasattr(component, 'set_database'):
                component.set_database(self.term_database)

    def translate(self, text: str, **kwargs) -> Dict:
        """
        完整的翻译流程
        返回包含译文和元数据的字典
        """
        logger.info(f"Starting translation of text (length: {len(text)})")

        try:
            # 1. 预处理阶段
            logger.info("Phase 1: Preprocessing")
            preprocessed = self._preprocess(text)

            # 2. 翻译阶段
            logger.info("Phase 2: Translation")
            raw_translation = self._translate_core(
                preprocessed['protected_text'],
                preprocessed['context']
            )

            # 3. 后处理阶段
            logger.info("Phase 3: Postprocessing")
            final_translation = self._postprocess(
                raw_translation,
                preprocessed['context']
            )

            # 4. 质量评估
            quality_score = self._evaluate_quality(
                text,
                final_translation,
                preprocessed['context']
            )

            result = {
                'translation': final_translation,
                'quality_score': quality_score,
                'metadata': {
                    'source_length': len(text),
                    'translation_length': len(final_translation),
                    'terms_found': len(preprocessed['context'].term_mappings),
                    'sentences': len(preprocessed['context'].sentences),
                    'confidence_scores': preprocessed['context'].confidence_scores,
                    # 添加佛教语境信息
                    'buddhist_context': getattr(preprocessed['context'], 'buddhist_context', 'GENERAL'),
                    'context_confidence': getattr(preprocessed['context'], 'context_confidence', 0.0),
                    'function_type': getattr(preprocessed['context'], 'function_type', 'GENERAL_TERM')
                }
            }
            # 可选：如果希望在结果中包含完整的语境分析
            if hasattr(preprocessed['context'], 'context_analysis'):
                result['context_analysis'] = preprocessed['context'].context_analysis

            logger.info("Translation completed successfully")
            return result

        except Exception as e:
            logger.error(f"Translation failed: {str(e)}")
            return {
                'translation': '',
                'error': str(e),
                'quality_score': 0.0
            }

    def _preprocess(self, text: str) -> Dict:
        """预处理阶段"""
        try:
            # 1. 规范化文本
            normalized_text = TextUtils.normalize_tibetan(text)

            # 2. Botok分析
            botok_analysis = self.botok_analyzer.analyze(normalized_text)
            botok_analysis['original_text'] = normalized_text

            # 3. 提取语法信息
            grammar_info = self.botok_analyzer.get_grammatical_info_for_translation(normalized_text)

            # 4. 术语保护
            identified_terms = botok_analysis.get('terms', [])

            # 转换Token对象为字典（如果需要）
            converted_terms = []
            for term in identified_terms:
                if hasattr(term, '__dict__'):
                    # 如果是Token对象，转换为字典
                    term_dict = {
                        'text': getattr(term, 'text', ''),
                        'start': getattr(term, 'start', 0),
                        'end': getattr(term, 'end', 0),
                        'length': len(getattr(term, 'text', '')),
                        'type': getattr(term, 'pos', 'general'),
                        'pos': getattr(term, 'pos', None)
                    }
                    converted_terms.append(term_dict)
                else:
                    # 如果已经是字典，直接使用
                    converted_terms.append(term)

            protected_text, protection_map = self.term_protector.protect_terms(
                normalized_text, converted_terms
            )

            # 5. 提取上下文
            context = self.context_extractor.extract_context(
                botok_analysis, protection_map
            )

            # 6. 佛教语境检测
            context_analysis = self.context_detector.detect(normalized_text)

            # 7. 将语法信息添加到上下文
            context.grammatical_info = grammar_info

            # 8. 丰富上下文信息
            context.buddhist_context = context_analysis.get('primary_context', 'GENERAL')
            context.context_confidence = context_analysis.get('context_confidence', 0.0)
            context.function_type = context_analysis.get('primary_function', 'GENERAL_TERM')

            return {
                'protected_text': protected_text,
                'context': context,
                'botok_analysis': botok_analysis
            }
        except Exception as e:
            logger.error(f"Error in _preprocess: {str(e)}")
            import traceback
            logger.error(f"Full traceback:\n{traceback.format_exc()}")
            raise

    def _translate_core(self, protected_text: str, context) -> str:
        """核心翻译阶段"""
        # 判断是否需要分句翻译
        if self._should_split_sentences(protected_text, context):
            return self._translate_by_sentences(protected_text, context)
        else:
            # 直接翻译整段
            return self.translator.translate(protected_text)

    def _should_split_sentences(self, text: str, context) -> bool:
        """判断是否需要分句翻译"""
        # 基于文本长度和句子数量
        if len(text) > self.sentence_split_threshold and len(context.sentences) > 1:
            return True

        # 如果置信度较低，也分句翻译
        if context.confidence_scores.get('sentence_segmentation', 0) < 0.5:
            return False  # 句子分割不可靠时不分句

        return False

    def _translate_by_sentences(self, text: str, context) -> str:
        """分句翻译"""
        translations = []

        # 根据句子边界分割文本
        sentences = self._split_text_by_boundaries(text, context.sentence_boundaries)

        # 批量翻译
        if len(sentences) > 1:
            sentence_translations = self.translator.translate_batch(sentences)
        else:
            sentence_translations = [self.translator.translate(sentences[0])]

        # 合并翻译结果
        for i, trans in enumerate(sentence_translations):
            # 应用句子级别的语法信息
            if i < len(context.grammatical_analyses):
                trans = self._apply_sentence_context(
                    trans,
                    context.grammatical_analyses[i]
                )
            translations.append(trans)

        # 智能合并句子
        return self._merge_sentence_translations(translations, context)

    def _split_text_by_boundaries(self, text: str, boundaries: List[int]) -> List[str]:
        """根据边界分割文本"""
        sentences = []
        start = 0

        for end in boundaries:
            if end > start:
                sentences.append(text[start:end].strip())
                start = end

        # 添加最后一部分
        if start < len(text):
            sentences.append(text[start:].strip())

        return sentences

    def _apply_sentence_context(self, translation: str, grammar_analysis: Dict) -> str:
        """应用句子级语法上下文"""
        # 这里可以根据语法分析微调翻译
        # 例如，如果识别出特定的句型，可以调整词序
        return translation

    def _merge_sentence_translations(self, translations: List[str], context) -> str:
        """智能合并句子翻译"""
        # 基本合并
        merged = ' '.join(translations)

        # 处理句子间的连接
        # 中文通常使用句号分隔句子
        merged = merged.replace(' 。', '。')
        merged = merged.replace('。 ', '。')

        # 如果是诗偈格式，保持换行
        if context.is_verse:
            # 按照原始格式恢复换行
            lines_per_verse = len(context.verse_structure.get('lines', [])) // len(translations)
            if lines_per_verse > 0:
                formatted_lines = []
                for i in range(0, len(translations), lines_per_verse):
                    verse_block = '。'.join(translations[i:i + lines_per_verse])
                    formatted_lines.append(verse_block)
                merged = '\n'.join(formatted_lines)

        return merged

    def _postprocess(self, raw_translation: str, context) -> str:
        """后处理阶段"""
        # 使用适配器准备后处理上下文
        postprocessing_context = ProcessingAdapter.prepare_context_for_postprocessing(
            context, raw_translation
        )

        # 1. 恢复术语
        text = self.term_restorer.restore(raw_translation, postprocessing_context)

        # 2. 语法修正
        text = self.grammar_corrector.correct(text, postprocessing_context)

        # 3. 质量控制
        text = self.quality_controller.refine(text, postprocessing_context)

        # 4. 最终调整
        text = self.text_finalizer.finalize(text, postprocessing_context)

        return text

    def _evaluate_quality(self, source: str, translation: str, context) -> float:
        """
        评估翻译质量

        Args:
            source: 源文本
            translation: 翻译文本
            context: 上下文信息

        Returns:
            质量得分 (0.0-1.0)
        """
        scores = []

        # 1. 术语覆盖率
        term_mappings = ProcessingAdapter.get_term_mappings(context)
        if term_mappings:
            term_coverage = self._calculate_term_coverage(translation, term_mappings)
            scores.append(term_coverage * 0.3)  # 30% 权重

        # 2. 句子完整性
        sentence_completeness = self._calculate_sentence_completeness(translation, context)
        scores.append(sentence_completeness * 0.2)  # 20% 权重

        # 3. 语法正确性
        grammar_score = self._calculate_grammar_score(translation, context)
        scores.append(grammar_score * 0.2)  # 20% 权重

        # 4. 流畅度
        fluency_score = self._calculate_fluency_score(translation)
        scores.append(fluency_score * 0.2)  # 20% 权重

        # 5. 长度比例合理性
        length_ratio_score = self._calculate_length_ratio_score(source, translation)
        scores.append(length_ratio_score * 0.1)  # 10% 权重

        # 综合得分
        total_score = sum(scores)

        return min(total_score, 1.0)

    def _calculate_term_coverage(self, translation: str, term_mappings: Dict) -> float:
        """
        计算术语覆盖率

        Args:
            translation: 翻译文本
            term_mappings: 术语映射字典

        Returns:
            覆盖率得分 (0.0-1.0)
        """
        if not term_mappings:
            return 1.0  # 没有术语需要覆盖

        # 统计应该出现的术语数量
        expected_terms = set()

        for placeholder, term_info in term_mappings.items():
            if placeholder not in translation:  # 如果占位符还在，说明术语未被替换
                if 'default_translation' in term_info:
                    expected_terms.add(term_info['default_translation'])
                elif 'chinese' in term_info:
                    expected_terms.add(term_info['chinese'])

        # 统计实际出现的术语数量
        found_terms = 0
        for term in expected_terms:
            if term in translation:
                found_terms += 1

        # 计算覆盖率
        if not expected_terms:
            return 1.0

        return found_terms / len(expected_terms)

    def _calculate_sentence_completeness(self, translation: str, context) -> float:
        """
        计算句子完整性得分

        Args:
            translation: 翻译文本
            context: 上下文信息

        Returns:
            完整性得分 (0.0-1.0)
        """
        # 获取原始句子数量
        original_sentences = ProcessingAdapter.extract_info_from_context(
            context, 'sentences', []
        )

        # 计算翻译文本中的句子数量
        translated_sentences = re.split(r'[。！？]', translation)
        translated_sentences = [s.strip() for s in translated_sentences if s.strip()]

        # 检查句子是否完整（是否有主谓结构）
        complete_sentences = 0
        for sentence in translated_sentences:
            if len(sentence) >= 5:  # 忽略太短的句子
                # 简单检查：长度适中且包含动词
                has_verb = False
                common_verbs = ['是', '有', '来', '去', '做', '说', '看', '想', '知道', '认为']
                for verb in common_verbs:
                    if verb in sentence:
                        has_verb = True
                        break

                if has_verb:
                    complete_sentences += 1

        # 计算完整率
        if not original_sentences:
            expected_sentences = max(1, len(translated_sentences))
        else:
            expected_sentences = len(original_sentences)

        completeness_ratio = complete_sentences / max(1, expected_sentences)

        # 检查是否有未完成的句子
        incomplete_patterns = [
            r'。{2,}',  # 多个句号
            r'[^。，、！？；：""''【】《》（）\s]\s*$',  # 句尾没有标点
            r'^[，、]',  # 以逗号开头
            r'[，、]{2,}',  # 连续的逗号
        ]

        penalty = 0
        for pattern in incomplete_patterns:
            if re.search(pattern, translation):
                penalty += 0.1

        # 检查句子数量是否合理
        if original_sentences:
            source_sentence_count = len(original_sentences)
            trans_sentence_count = len(re.findall(r'[。！？]', translation))

            ratio = trans_sentence_count / source_sentence_count
            if 0.5 <= ratio <= 2.0:
                sentence_ratio_score = 1.0
            else:
                sentence_ratio_score = max(0, 1.0 - abs(ratio - 1.0) * 0.5)

            completeness_ratio = (completeness_ratio + sentence_ratio_score) / 2

        return max(0, min(1.0, completeness_ratio - penalty))

    def _calculate_grammar_score(self, translation: str, context) -> float:
        """
        计算语法正确性得分

        Args:
            translation: 翻译文本
            context: 上下文信息

        Returns:
            语法得分 (0.0-1.0)
        """
        score = 1.0

        # 检查常见语法错误
        grammar_errors = [
            (r'的的', 0.1),  # 重复的"的"
            (r'在在', 0.1),  # 重复的"在"
            (r'是是', 0.1),  # 重复的"是"
            (r'[他她它]们们', 0.1),  # 重复的"们"
            (r'把[^把]{0,10}把', 0.15),  # 连续的"把"字句
            (r'被[^被]{0,10}被', 0.15),  # 连续的"被"字句
        ]

        for pattern, penalty in grammar_errors:
            if re.search(pattern, translation):
                score -= penalty

        # 检查主谓搭配问题
        subject_verb_errors = self._check_subject_verb_errors(translation)
        score -= subject_verb_errors * 0.05

        # 检查量词使用问题
        measure_word_errors = self._check_measure_word_errors(translation)
        score -= measure_word_errors * 0.05

        # 检查成分缺失问题
        missing_component_errors = self._check_missing_components(translation)
        score -= missing_component_errors * 0.05

        # 检查虚词使用问题
        function_word_errors = self._check_function_word_errors(translation)
        score -= function_word_errors * 0.05

        # 检查主谓宾结构完整性
        if hasattr(context, 'grammatical_analyses') and context.grammatical_analyses:
            for analysis in context.grammatical_analyses:
                if analysis.get('syntactic_roles'):
                    # 检查是否有孤立的语法成分
                    roles = [r['role'] for r in analysis['syntactic_roles']]
                    if 'agent' in roles and 'patient' not in roles:
                        score -= 0.05  # 有施事无受事
                    if 'patient' in roles and 'agent' not in roles:
                        score -= 0.05  # 有受事无施事

        return max(0, score)

    def _check_subject_verb_errors(self, text: str) -> int:
        """检查主谓搭配错误"""
        errors = 0

        # 简化实现：检查一些常见的错误搭配
        # 添加捕获组以防止将来出现"no such group"错误
        error_patterns = [
            r'(我们).{0,5}(是).{0,5}(去)',
            r'(他们).{0,5}(是).{0,5}(有)',
            r'(佛陀).{0,5}(我).{0,5}(说)',
            r'(菩萨).{0,5}(你).{0,5}(修)',
            r'(经中).{0,5}(他).{0,5}(讲)'
        ]

        for pattern in error_patterns:
            matches = re.finditer(pattern, text)
            errors += len(list(matches))

        return errors

    def _check_measure_word_errors(self, text: str) -> int:
        """检查量词使用错误"""
        errors = 0

        # 检查数量词后缺少量词的情况
        error_patterns = [
            r'([一二三四五六七八九十百千万亿两])([\u4e00-\u9fff])',
            r'(\d+)([\u4e00-\u9fff])'
        ]

        # 常见量词
        measure_words = ['个', '只', '条', '位', '尊', '座', '部', '篇', '次', '回', '种']

        for pattern in error_patterns:
            matches = re.finditer(pattern, text)
            for match in matches:
                if match.group(2) not in measure_words and not self._is_measure_word(match.group(2)):
                    errors += 1

        return errors

    def _is_measure_word(self, word: str) -> bool:
        """检查是否为量词"""
        common_measure_words = [
            '个', '只', '条', '位', '尊', '座', '部', '篇', '次', '回', '种',
            '片', '幅', '张', '颗', '粒', '件', '台', '辆', '节', '届', '轮',
            '场', '口', '道', '句', '本', '册', '卷', '门', '套', '项', '枚'
        ]

        return word in common_measure_words

    def _check_missing_components(self, text: str) -> int:
        """检查成分缺失问题"""
        errors = 0

        # 检查"把"字句缺少动词的情况
        ba_errors = len(re.findall(r'把.{1,5}了', text))
        errors += ba_errors

        # 检查"被"字句缺少动词的情况
        bei_errors = len(re.findall(r'被.{1,5}了', text))
        errors += bei_errors

        # 检查缺少宾语的情况
        missing_object_errors = len(re.findall(r'(给予|告诉|通知|回答|解释)。', text))
        errors += missing_object_errors

        return errors

    def _check_function_word_errors(self, text: str) -> int:
        """检查虚词使用问题"""
        errors = 0

        # 检查"的"、"地"、"得"混用
        de_errors = len(re.findall(r'[动形]的[动名]', text))  # 应该用"地"
        errors += de_errors

        de2_errors = len(re.findall(r'[动]得[形名]', text))  # 应该用"的"
        errors += de2_errors

        # 检查"了"的错误使用
        le_errors = len(re.findall(r'[了]{2,}', text))
        errors += le_errors

        return errors

    def _calculate_fluency_score(self, translation: str) -> float:
        """
        计算流畅度得分

        Args:
            translation: 翻译文本

        Returns:
            流畅度得分 (0.0-1.0)
        """
        score = 1.0

        # 检查不流畅的模式
        disfluency_patterns = [
            (r'(\S)\1{3,}', 0.2),  # 字符重复3次以上
            (r'[，。]\s*[，。]', 0.1),  # 标点符号连续
            (r'\s{2,}', 0.05),  # 多个空格
            (r'[一二三四五六七八九十]\s+[一二三四五六七八九十]', 0.1),  # 数字断开
        ]

        for pattern, penalty in disfluency_patterns:
            matches = re.findall(pattern, translation)
            score -= penalty * len(matches)

        # 检查句子长度分布
        sentences = re.split(r'[。！？]', translation)
        sentences = [s.strip() for s in sentences if s.strip()]

        if sentences:
            lengths = [len(s) for s in sentences]
            avg_length = sum(lengths) / len(lengths)

            # 理想的中文句子长度在15-30字之间
            if 15 <= avg_length <= 30:
                length_score = 1.0
            elif avg_length < 10 or avg_length > 50:
                length_score = 0.7
            else:
                length_score = 0.85

            score *= length_score

        # 检查冗余重复
        repetition_issues = self._check_repetition(translation)
        score -= repetition_issues * 0.05

        # 检查生硬表达
        awkward_expressions = self._check_awkward_expressions(translation)
        score -= awkward_expressions * 0.05

        # 检查连贯性
        coherence_issues = self._check_coherence(translation)
        score -= coherence_issues * 0.05

        return max(0, score)

    def _check_repetition(self, text: str) -> int:
        """检查冗余重复"""
        issues = 0

        # 检查相邻词语重复
        for word_length in range(2, 5):
            for i in range(len(text) - word_length * 2):
                if text[i:i + word_length] == text[i + word_length:i + word_length * 2]:
                    issues += 1

        # 检查相近位置同义词重复
        synonym_pairs = [
            ('说', '讲'), ('看', '观'), ('想', '思'), ('知道', '了解'),
            ('快乐', '高兴'), ('美好', '美妙'), ('重要', '关键')
        ]

        for word1, word2 in synonym_pairs:
            if word1 in text and word2 in text:
                pos1 = text.find(word1)
                pos2 = text.find(word2)
                if abs(pos1 - pos2) < 15:  # 相近位置
                    issues += 1

        return issues

    def _check_awkward_expressions(self, text: str) -> int:
        """检查生硬表达"""
        issues = 0

        # 检查不自然的词语搭配
        awkward_patterns = [
            r'做出.{0,2}理解', r'进行.{0,2}思考', r'给予.{0,2}思维',
            r'作出.{0,2}学习', r'实施.{0,2}观看', r'做好.{0,2}准备工作'
        ]

        for pattern in awkward_patterns:
            matches = re.finditer(pattern, text)
            issues += len(list(matches))

        # 检查过于直译的表达
        literal_patterns = [
            r'以.{1,5}的方式', r'关于.{1,5}的方面', r'在.{1,5}之上',
            r'由于.{1,5}的原因', r'对于.{1,5}来说', r'如果.{1,5}的话'
        ]

        for pattern in literal_patterns:
            matches = re.finditer(pattern, text)
            issues += len(list(matches))

        return issues

    def _check_coherence(self, text: str) -> int:
        """检查连贯性"""
        issues = 0

        # 检查转折词语使用不当
        coherence_patterns = [
            r'因此.{0,10}但是', r'所以.{0,10}然而', r'因为.{0,10}不过',
            r'虽然.{0,20}因此', r'尽管.{0,20}所以', r'不但.{0,20}不过'
        ]

        for pattern in coherence_patterns:
            matches = re.finditer(pattern, text)
            issues += len(list(matches))

        # 检查指代不明
        reference_patterns = [
            r'这个.{0,5}这个', r'那个.{0,5}那个', r'他.{0,10}他.{0,10}他',
            r'它.{0,10}它.{0,10}它'
        ]

        for pattern in reference_patterns:
            matches = re.finditer(pattern, text)
            issues += len(list(matches))

        return issues

    def _calculate_length_ratio_score(self, source: str, translation: str) -> float:
        """
        计算长度比例合理性得分

        Args:
            source: 源文本
            translation: 翻译文本

        Returns:
            长度比例得分 (0.0-1.0)
        """
        # 藏文到中文的合理长度比例范围
        min_ratio = 0.5  # 翻译长度至少应为原文的50%
        max_ratio = 1.2  # 翻译长度最多为原文的120%

        # 空值处理
        if not source or not translation:
            return 0.0

        # 计算字符比例（不计空格）
        source_len = len(source.replace(' ', ''))
        trans_len = len(translation.replace(' ', ''))

        if source_len == 0:
            return 0.0

        actual_ratio = trans_len / source_len

        # 如果比例在理想范围内，得分为1.0
        if min_ratio <= actual_ratio <= max_ratio:
            return 1.0

        # 如果比例超出理想范围，根据偏离程度扣分
        if actual_ratio < min_ratio:
            # 太短
            return actual_ratio / min_ratio
        else:
            # 太长
            return max_ratio / actual_ratio

    def translate_document(self, document: str, progress_callback=None) -> Dict:
        """
        翻译长文档，支持进度回调

        Args:
            document: 要翻译的文档
            progress_callback: 进度回调函数

        Returns:
            包含翻译结果和元数据的字典
        """
        # 分段处理长文档
        paragraphs = self._split_document(document)
        translations = []
        total_paragraphs = len(paragraphs)

        logger.info(f"Translating document with {total_paragraphs} paragraphs")

        for i, paragraph in enumerate(paragraphs):
            if progress_callback:
                progress_callback(i, total_paragraphs)

            # 翻译段落
            result = self.translate(paragraph)
            translations.append(result['translation'])

            # 记录进度
            logger.debug(f"Translated paragraph {i + 1}/{total_paragraphs}")

        # 合并结果
        final_translation = '\n\n'.join(translations)

        return {
            'translation': final_translation,
            'metadata': {
                'total_paragraphs': total_paragraphs,
                'source_length': len(document),
                'translation_length': len(final_translation)
            }
        }

    def _split_document(self, document: str) -> List[str]:
        """
        智能分割文档

        Args:
            document: 要分割的文档

        Returns:
            段落列表
        """
        # 首先尝试按段落分割（双换行）
        paragraphs = re.split(r'\n\s*\n', document)

        # 处理过长的段落
        max_paragraph_length = 500
        final_paragraphs = []

        for para in paragraphs:
            if len(para) > max_paragraph_length:
                # 按句子分割
                sentences = re.split(r'།\s*', para)
                current_chunk = ""

                for sent in sentences:
                    if len(current_chunk) + len(sent) < max_paragraph_length:
                        current_chunk += sent + "། "
                    else:
                        if current_chunk:
                            final_paragraphs.append(current_chunk.strip())
                        current_chunk = sent + "། "

                if current_chunk:
                    final_paragraphs.append(current_chunk.strip())
            else:
                final_paragraphs.append(para)

        return [p for p in final_paragraphs if p.strip()]