"""
语法修正器 - 根据藏文语法信息修正中文译文
"""
import re
from typing import Dict, List, Optional, Tuple
import logging
from .adapter import ProcessingAdapter

logger = logging.getLogger(__name__)


class GrammarCorrector:
    def __init__(self):
        """初始化语法修正器"""
        # 格助词处理规则
        self.case_particle_rules = {
            'genitive': self._handle_genitive,  # 属格 གི་/ཀྱི་/གྱི་/ཡི་/གི་
            'ergative': self._handle_ergative,  # 作格 གིས་/ཀྱིས་/གྱིས་/ཡིས་/གིས་
            'dative': self._handle_dative,  # 与格 ལ་
            'ablative': self._handle_ablative,  # 从格 ནས་/ལས་
            'locative': self._handle_locative,  # 位格 ན་/ནས་
            'comitative': self._handle_comitative,  # 共格 དང་
            'comparative': self._handle_comparative,  # 比较格 ལས་/བས་
            'terminative': self._handle_terminative,  # 终格 སུ་/ཏུ་/དུ་/རུ་/ར་
        }

        # 时态修正规则
        self.tense_rules = {
            'past': self._apply_past_tense,
            'present': self._apply_present_tense,
            'future': self._apply_future_tense,
            'present_progressive': self._apply_progressive
        }

        # 句型模板
        self.sentence_patterns = {
            'declarative': '{subject}{predicate}。',
            'interrogative': '{subject}{predicate}吗？',
            'imperative': '请{predicate}！',
            'exclamatory': '{content}！'
        }

    def correct(self, text: str, context) -> str:
        """执行语法修正"""
        # 使用适配器获取语法分析信息
        grammar_analyses = ProcessingAdapter.get_grammatical_analyses(context)

        if not grammar_analyses:
            return text

        # 分句处理
        sentences = self._split_into_sentences(text)
        corrected_sentences = []

        for i, sentence in enumerate(sentences):
            if i < len(grammar_analyses):
                # 使用适配器提供的信息进行句子修正
                corrected = self._correct_sentence(sentence, i, context)
                corrected_sentences.append(corrected)
            else:
                corrected_sentences.append(sentence)

        # 合并句子
        result = self._merge_sentences(corrected_sentences)

        # 全局语法检查和修正
        result = self._global_grammar_fixes(result)

        return result

    def _split_into_sentences(self, text: str) -> List[str]:
        """分割句子"""
        # 保留标点符号的分割
        sentences = re.split(r'([。！？；])', text)

        # 重组句子（包含标点）
        result = []
        for i in range(0, len(sentences), 2):
            if i + 1 < len(sentences):
                result.append(sentences[i] + sentences[i + 1])
            else:
                if sentences[i].strip():
                    result.append(sentences[i])

        return result

    def _correct_sentence(self, sentence: str, sentence_index: int, context) -> str:
        """
        修正单个句子

        Args:
            sentence: 需要修正的句子
            sentence_index: 句子在原文中的索引
            context: 上下文信息

        Returns:
            修正后的句子
        """
        # 1. 处理格助词
        case_particles = ProcessingAdapter.get_case_particles(context, sentence_index)
        if case_particles:
            sentence = self._process_case_particles(sentence, case_particles)

        # 2. 处理时态
        tense = ProcessingAdapter.get_tense(context, sentence_index)
        if tense:
            sentence = self._process_tense(sentence, tense)

        # 3. 调整语序
        syntactic_roles = ProcessingAdapter.get_syntactic_roles(context, sentence_index)
        if syntactic_roles:
            sentence = self._adjust_word_order(sentence, syntactic_roles)

        # 4. 处理句型
        sentence_type = ProcessingAdapter.get_sentence_type(context, sentence_index)
        sentence = self._apply_sentence_pattern(sentence, sentence_type)

        # 5. 处理佛教术语
        sentence = self._handle_buddhist_terminology(sentence, context)

        return sentence

    # 重命名此方法以避免重复
    def _correct_sentence_with_analysis(self, sentence: str, analysis: Dict, context) -> str:
        """
        基于分析结果修正单个句子

        Args:
            sentence: 需要修正的句子
            analysis: 句子的语法分析结果
            context: 上下文信息

        Returns:
            修正后的句子
        """
        # 1. 处理格助词
        if analysis.get('case_particles'):
            sentence = self._process_case_particles(sentence, analysis['case_particles'])

        # 2. 处理时态
        if analysis.get('tense'):
            sentence = self._process_tense(sentence, analysis['tense'])

        # 3. 调整语序
        if analysis.get('syntactic_roles'):
            sentence = self._adjust_word_order(sentence, analysis['syntactic_roles'])

        # 4. 处理句型
        sentence_type = analysis.get('type', 'declarative')
        sentence = self._apply_sentence_pattern(sentence, sentence_type)

        return sentence

    def _process_case_particles(self, sentence: str, particles: List[Dict]) -> str:
        """处理格助词，基于语法分析信息，而不是硬编码规则"""
        if not particles:
            return sentence

        # 按照位置排序，从后向前处理，避免位置变化
        sorted_particles = sorted(particles, key=lambda p: p.get('position', 0), reverse=True)

        for particle_info in sorted_particles:
            particle_type = particle_info.get('type')
            attached_to = particle_info.get('attached_to', '')

            if not particle_type or not attached_to or attached_to not in sentence:
                continue

            # 根据格类型和上下文选择合适的处理方式
            sentence = self._apply_case_particle(sentence, attached_to, particle_type)

        return sentence

    def _apply_case_particle(self, sentence: str, word: str, case_type: str) -> str:
        """根据格类型和上下文智能应用中文表达"""
        # 分析句子上下文
        context_features = self._detect_context_features(sentence, word)

        # 根据格类型和上下文特征选择最佳的中文表达
        if case_type == 'genitive':  # 属格
            # 检查是否已有"的"或"之"
            if not any(f"{word}{marker}" in sentence for marker in ['的', '之']):
                # 选择正式程度
                marker = '之' if 'formal' in context_features else '的'
                return sentence.replace(word, f"{word}{marker}")

        elif case_type == 'ergative':  # 作格
            # 检查是否为被动语态
            if 'passive' in context_features:
                if not f"被{word}" in sentence:
                    return sentence.replace(word, f"被{word}")
            # 检查是否需要"把"字句
            elif 'patient_after_verb' in context_features:
                if not f"把{word}" in sentence:
                    return sentence.replace(word, f"把{word}")

        elif case_type == 'dative':  # 与格
            # 根据语境选择合适的介词
            if 'speaking' in context_features:
                marker = '对'
            elif 'giving' in context_features:
                marker = '给'
            elif 'movement' in context_features:
                marker = '向'
            else:
                marker = '为'

            if not any(f"{prep}{word}" in sentence for prep in ['对', '给', '向', '为']):
                return sentence.replace(word, f"{marker}{word}")

        elif case_type == 'ablative':  # 从格
            if not any(f"{prep}{word}" in sentence for prep in ['从', '由', '自']):
                marker = '从' if 'movement' in context_features else '由'
                return sentence.replace(word, f"{marker}{word}")

        elif case_type == 'locative':  # 位格
            if not any(f"{prep}{word}" in sentence for prep in ['在', '于']):
                marker = '于' if 'formal' in context_features else '在'
                return sentence.replace(word, f"{marker}{word}")

        # 其他格助词处理...

        return sentence

    def _detect_context_features(self, sentence: str, word: str) -> set:
        """检测句子中的上下文特征"""
        features = set()

        # 文体特征
        if any(formal in sentence for formal in ['尊者', '世尊', '如是', '者', '乃', '之']):
            features.add('formal')
        else:
            features.add('modern')

        # 语态特征
        if '被' in sentence or '所' in sentence and '为' in sentence:
            features.add('passive')
        else:
            features.add('active')

        # 语义特征
        verb_indicators = ['说', '告诉', '解释', '陈述', '宣说', '开示']
        if any(verb in sentence for verb in verb_indicators):
            features.add('speaking')

        movement_indicators = ['去', '来', '到', '往', '行', '走']
        if any(verb in sentence for verb in movement_indicators):
            features.add('movement')

        giving_indicators = ['给', '赐', '授', '供', '献']
        if any(verb in sentence for verb in giving_indicators):
            features.add('giving')

        # 词序特征
        word_pos = sentence.find(word)
        verb_pos = -1

        # 寻找可能的谓语动词
        common_verbs = ['说', '做', '去', '来', '修', '学', '证', '得', '看']
        for verb in common_verbs:
            if verb in sentence:
                verb_pos = sentence.find(verb)
                break

        # 检查受事者位置
        if word_pos > -1 and verb_pos > -1:
            if word_pos > verb_pos:
                features.add('patient_after_verb')

        return features

    def _handle_genitive(self, sentence: str, particle_info: Dict) -> str:
        """处理属格（的）"""
        # 藏文属格通常翻译为"的"
        # 检查是否已经有"的"，避免重复
        attached_to = particle_info.get('attached_to', '')

        if attached_to and attached_to in sentence:
            # 确保属格标记正确
            pattern = f"{attached_to}\\s*的"
            if not re.search(pattern, sentence):
                # 在词后添加"的"
                sentence = sentence.replace(attached_to, f"{attached_to}的")

        return sentence

    def _handle_ergative(self, sentence: str, particle_info: Dict) -> str:
        """
        处理作格（施事标记），确保句子中的施事者位置正确

        Args:
            sentence: 原始句子
            particle_info: 作格助词信息

        Returns:
            修正后的句子
        """
        attached_to = particle_info.get('attached_to', '')

        if not attached_to or attached_to not in sentence:
            return sentence

        # 分析句子结构
        words = re.findall(r'\w+', sentence)
        agent_pos = None
        verb_pos = None

        # 查找施事者位置
        for i, word in enumerate(words):
            if attached_to in word:
                agent_pos = i
                break

        # 查找动词位置（使用更复杂的动词识别）
        verb_markers = ['了', '着', '过', '在', '要', '会', '能', '被']
        common_buddhist_verbs = [
            '修行', '禅修', '念诵', '供养', '顶礼', '皈依', '受戒', '持戒',
            '忏悔', '回向', '发愿', '证悟', '开示', '传授', '解脱', '度化'
        ]

        for i, word in enumerate(words):
            # 检查时态标记
            if any(marker in word for marker in verb_markers):
                verb_pos = i
                break

            # 检查常见佛教动词
            for verb in common_buddhist_verbs:
                if verb in word:
                    verb_pos = i
                    break

            if verb_pos is not None:
                break

        # 如果找到了施事者和动词，且施事者位于动词之后（不符合中文语序）
        if agent_pos is not None and verb_pos is not None and agent_pos > verb_pos:
            # 重新组织句子成分
            before_verb = ' '.join(words[:verb_pos])
            verb = words[verb_pos]
            between = ' '.join(words[verb_pos + 1:agent_pos])
            agent = words[agent_pos]
            after_agent = ' '.join(words[agent_pos + 1:])

            # 重构句子，将施事者移到动词前
            new_sentence = f"{before_verb} {agent} {verb} {between} {after_agent}"
            # 规范化空格
            new_sentence = re.sub(r'\s+', ' ', new_sentence).strip()

            return new_sentence

        # 检查施事者后是否缺少施事标记（在中文中通常是"被"字）
        if agent_pos is not None and '被' not in sentence:
            # 在某些情况下，藏文作格需要在中文中添加"被"字结构
            # 检查句子是否表达被动含义
            passive_indicators = ['受到', '遭到', '经历', '获得']
            has_passive = any(ind in sentence for ind in passive_indicators)

            if has_passive:
                # 在施事者前添加"被"
                words[agent_pos] = f"被{words[agent_pos]}"
                new_sentence = ' '.join(words)
                return new_sentence

        return sentence

    def _handle_dative(self, sentence: str, particle_info: Dict) -> str:
        """处理与格（向、给、对）"""
        attached_to = particle_info.get('attached_to', '')

        if attached_to and attached_to in sentence:
            # 检查上下文决定使用哪个介词
            if '说' in sentence or '讲' in sentence:
                preposition = '对'
            elif '给予' in sentence or '赐' in sentence:
                preposition = '给'
            else:
                preposition = '向'

            # 确保有适当的介词
            if not any(prep in sentence for prep in ['对', '给', '向', '为']):
                # 在目标词前加介词
                sentence = sentence.replace(attached_to, f"{preposition}{attached_to}")

        return sentence

    def _handle_ablative(self, sentence: str, particle_info: Dict) -> str:
        """处理从格（从、由）"""
        attached_to = particle_info.get('attached_to', '')

        if attached_to and attached_to in sentence:
            # 检查是否需要"从"或"由"
            if not any(prep in sentence for prep in ['从', '由', '自']):
                # 根据语境选择介词
                if '来' in sentence or '到' in sentence:
                    preposition = '从'
                else:
                    preposition = '由'

                sentence = sentence.replace(attached_to, f"{preposition}{attached_to}")

        return sentence

    def _handle_locative(self, sentence: str, particle_info: Dict) -> str:
        """处理位格（在、于）"""
        attached_to = particle_info.get('attached_to', '')

        if attached_to and attached_to in sentence:
            # 确保有位置介词
            if not any(prep in sentence for prep in ['在', '于', '处']):
                sentence = sentence.replace(attached_to, f"在{attached_to}")

        return sentence

    def _handle_comitative(self, sentence: str, particle_info: Dict) -> str:
        """处理共格（和、与、同）"""
        attached_to = particle_info.get('attached_to', '')

        if attached_to and attached_to in sentence:
            # 检查是否已有连词
            if not any(conj in sentence for conj in ['和', '与', '同', '及']):
                # 根据正式程度选择连词
                if any(formal in sentence for formal in ['尊者', '大师', '法师']):
                    conjunction = '与'
                else:
                    conjunction = '和'

                sentence = sentence.replace(attached_to, f"{conjunction}{attached_to}")

        return sentence

    def _handle_comparative(self, sentence: str, particle_info: Dict) -> str:
        """处理比较格（比、较）"""
        attached_to = particle_info.get('attached_to', '')

        if attached_to and attached_to in sentence:
            # 确保有比较结构
            if '比' not in sentence:
                sentence = sentence.replace(attached_to, f"比{attached_to}")

        return sentence

    def _handle_terminative(self, sentence: str, particle_info: Dict) -> str:
        """处理终格（到、至、成为）"""
        attached_to = particle_info.get('attached_to', '')

        if attached_to and attached_to in sentence:
            # 根据动词选择适当的标记
            if '变' in sentence or '成' in sentence:
                preposition = '成为'
            elif '走' in sentence or '去' in sentence:
                preposition = '到'
            else:
                preposition = '至'

            if preposition not in sentence:
                sentence = sentence.replace(attached_to, f"{preposition}{attached_to}")

        return sentence

    def _process_tense(self, sentence: str, tense: str) -> str:
        """
        基于上下文智能处理时态，而非简单规则

        Args:
            sentence: 原始句子
            tense: 时态信息

        Returns:
            处理后的句子
        """
        # 检查句子是否已有时态标记
        has_past_marker = any(marker in sentence for marker in ['了', '过', '曾经', '已经'])
        has_future_marker = any(marker in sentence for marker in ['将', '会', '要', '即将'])
        has_progressive_marker = any(marker in sentence for marker in ['正在', '正', '着'])

        # 分析句子结构
        verbs = self._find_main_verbs(sentence)
        if not verbs:
            return sentence  # 没有找到动词，不处理时态

        # 根据时态选择处理方式
        if tense == 'past':
            return self._apply_past_tense(sentence, verbs, has_past_marker)
        elif tense == 'future':
            return self._apply_future_tense(sentence, verbs, has_future_marker)
        elif tense == 'present_progressive':
            return self._apply_progressive(sentence, verbs, has_progressive_marker)
        elif tense == 'present':
            return self._apply_present_tense(sentence, has_past_marker, has_future_marker)

        return sentence

    def _apply_past_tense(self, sentence: str, verbs: List[str], has_marker: bool) -> str:
        """智能应用过去时标记"""
        if has_marker:
            return sentence  # 已有过去时标记

        # 分析句子，找出主要动词
        main_verb = verbs[0]
        verb_pos = sentence.find(main_verb)

        # 检查动词的上下文，判断应该在动词后加"了"还是在句末加"了"
        if verb_pos >= 0:
            # 动词后已有宾语，在动词后加"了"
            # 例如：昨天我看书 -> 昨天我看了书
            after_verb = sentence[verb_pos + len(main_verb):]
            if after_verb and not after_verb.startswith('了'):
                # 避免影响词组
                if not any(main_verb + next_char in verbs for next_char in after_verb[:1]):
                    return sentence[:verb_pos + len(main_verb)] + '了' + after_verb

            # 句末加"了"
            if not sentence.rstrip().endswith('了'):
                # 移除尾部标点
                clean_sentence = re.sub(r'[。！？；：，]$', '', sentence.rstrip())
                # 检查是否已以"了"结尾
                if not clean_sentence.endswith('了'):
                    punctuation = sentence[len(clean_sentence):]
                    return clean_sentence + '了' + punctuation

        return sentence

    def _apply_future_tense(self, sentence: str, verbs: List[str], has_marker: bool) -> str:
        """智能应用将来时标记"""
        if has_marker:
            return sentence  # 已有将来时标记

        # 分析句子中的时间指示词
        future_time_words = ['明天', '未来', '即将', '后来', '不久']
        has_future_indicator = any(word in sentence for word in future_time_words)

        # 如果已有将来时间指示词，可能不需要额外标记
        if has_future_indicator:
            return sentence

        # 找出主要动词
        main_verb = verbs[0]
        verb_pos = sentence.find(main_verb)

        if verb_pos >= 0:
            # 在动词前添加"将"或"会"
            before_verb = sentence[:verb_pos]
            # 根据句子语气选择"将"或"会"
            marker = "将" if self._is_formal_sentence(sentence) else "会"

            # 避免添加到已有情态动词后
            modal_verbs = ['可以', '能够', '应该', '必须', '想要']
            if any(before_verb.endswith(modal) for modal in modal_verbs):
                return sentence

            return before_verb + marker + sentence[verb_pos:]

        return sentence

    def _apply_progressive(self, sentence: str, verbs: List[str], has_marker: bool) -> str:
        """智能应用进行时标记"""
        if has_marker:
            return sentence  # 已有进行时标记

        # 找出主要动词
        main_verb = verbs[0]
        verb_pos = sentence.find(main_verb)

        if verb_pos >= 0:
            # 分析句子结构，选择合适的进行时标记方式
            before_verb = sentence[:verb_pos]
            after_verb = sentence[verb_pos + len(main_verb):]

            # 方式1：在动词前加"正在"
            if not any(marker in before_verb[-2:] for marker in ['正在', '正']):
                return before_verb + '正在' + main_verb + after_verb

            # 方式2：在动词后加"着"
            if not after_verb.startswith('着'):
                return before_verb + main_verb + '着' + after_verb

        return sentence

    def _apply_present_tense(self, sentence: str, has_past: bool, has_future: bool) -> str:
        """智能应用现在时（移除不当的时态标记）"""
        # 中文现在时通常不需要特殊标记
        # 但要移除不当的时态标记

        # 如果错误地有过去时标记，尝试移除
        if has_past:
            sentence = re.sub(r'了(?=[，。！？])', '', sentence)

        # 如果错误地有将来时标记，尝试移除
        if has_future:
            sentence = re.sub(r'(将|会)(?=[\u4e00-\u9fff]{1,3}[，。！？])', '', sentence)

        return sentence

    def _is_formal_sentence(self, sentence: str) -> bool:
        """判断句子是否为正式/典雅风格"""
        formal_indicators = ['之', '乃', '者', '也', '矣', '焉', '尊者', '如是', '世尊']
        return any(indicator in sentence for indicator in formal_indicators)

    def _find_main_verbs(self, sentence: str) -> List[str]:
        """智能识别句子中的主要动词"""
        found_verbs = []

        # 使用jieba分词进行词性标注
        try:
            import jieba.posseg as pseg
            words = pseg.cut(sentence)

            # 按位置收集所有动词
            verbs_with_position = []
            for word, flag in words:
                if flag.startswith('v'):  # 动词
                    position = sentence.find(word)
                    if position >= 0:  # 确保找到了
                        verbs_with_position.append((word, position))

            # 按位置排序
            verbs_with_position.sort(key=lambda x: x[1])

            # 提取动词
            found_verbs = [v[0] for v in verbs_with_position]

            # 如果找不到动词，使用后备方案
            if not found_verbs:
                found_verbs = self._find_verbs_by_rules(sentence)

        except ImportError:
            # 如果没有jieba，使用规则方法
            found_verbs = self._find_verbs_by_rules(sentence)

        return found_verbs

    def _find_verbs_by_rules(self, sentence: str) -> List[str]:
        """使用规则识别动词（作为后备方案）"""
        found_verbs = []

        # 1. 常见汉语动词和佛教专用动词
        common_verbs = [
            '说', '做', '去', '来', '看', '听', '想', '要', '是', '有',
            '修', '学', '证', '得', '行', '住', '坐', '卧', '生', '死',
            '觉', '悟', '度', '救', '护', '持', '念', '观', '思', '知'
        ]

        buddhist_verbs = [
            '修行', '禅修', '念诵', '供养', '顶礼', '皈依', '受戒', '持戒',
            '忏悔', '回向', '发愿', '证悟', '开示', '传授', '解脱', '度化'
        ]

        # 2. 动词组合
        verb_compounds = [
            '开始', '结束', '继续', '完成', '尝试', '停止', '保持', '增长'
        ]

        # 首先检查多字词动词
        all_longer_verbs = buddhist_verbs + verb_compounds
        for verb in sorted(all_longer_verbs, key=len, reverse=True):
            if verb in sentence:
                found_verbs.append(verb)
                # 标记已找到的部分
                sentence = sentence.replace(verb, '□' * len(verb))

        # 然后检查单字动词
        for verb in common_verbs:
            if verb in sentence:
                found_verbs.append(verb)

        # 使用时态标记识别动词
        tense_patterns = [
            r'(\w+)了(?=[，。！？\s]|$)',
            r'(\w+)着(?=[，。！？\s]|$)',
            r'(\w+)过(?=[，。！？\s]|$)',
            r'正在(\w+)',
            r'将要(\w+)'
        ]

        for pattern in tense_patterns:
            matches = re.finditer(pattern, sentence)
            for match in matches:
                verb = match.group(1)
                if verb and len(verb) <= 3 and verb not in found_verbs:
                    found_verbs.append(verb)

        return found_verbs

    def _adjust_word_order(self, sentence: str, syntactic_roles: List[Dict]) -> str:
        """
        调整语序，将藏文SOV结构转换为中文SVO结构，基于语法分析而非硬编码规则

        Args:
            sentence: 原始句子
            syntactic_roles: 句法角色信息

        Returns:
            调整后的句子
        """
        # 如果没有足够的语法信息，返回原句
        if not syntactic_roles:
            return sentence

        # 句子成分分析
        components = self._identify_sentence_components(sentence, syntactic_roles)

        # 检查是否需要调整语序
        if not self._needs_reordering(components):
            return sentence

        # 执行智能重排
        return self._reorder_components(sentence, components)

    def _identify_sentence_components(self, sentence: str, syntactic_roles: List[Dict]) -> Dict:
        """识别句子主要成分及其位置"""
        components = {
            'agent': None,  # 施事者/主语
            'patient': None,  # 受事者/宾语
            'verb': None,  # 主要动词
            'time': None,  # 时间状语
            'location': None,  # 地点状语
            'instrument': None,  # 工具状语
            'manner': None,  # 方式状语
            'purpose': None  # 目的状语
        }

        # 1. 从句法角色中提取主要成分
        for role in syntactic_roles:
            role_type = role.get('role')
            text = role.get('text')

            if not text or text not in sentence:
                continue

            if role_type == 'agent':
                components['agent'] = {
                    'text': text,
                    'position': sentence.find(text)
                }
            elif role_type == 'patient':
                components['patient'] = {
                    'text': text,
                    'position': sentence.find(text)
                }

        # 2. 识别主要动词
        verbs = self._find_main_verbs(sentence)
        if verbs:
            components['verb'] = {
                'text': verbs[0],
                'position': sentence.find(verbs[0])
            }

        # 3. 识别其他句子成分 (时间、地点等)
        self._identify_adverbial_components(sentence, components)

        return components

    def _identify_adverbial_components(self, sentence: str, components: Dict):
        """识别状语成分"""
        # 时间状语
        time_markers = ['昨天', '今天', '明天', '过去', '现在', '未来', '当时']
        for marker in time_markers:
            if marker in sentence:
                components['time'] = {
                    'text': marker,
                    'position': sentence.find(marker)
                }
                break

        # 地点状语
        location_patterns = [
            r'在([\u4e00-\u9fff]{1,5})(上|下|中|内|外)',
            r'([\u4e00-\u9fff]{1,5})之(上|下|中|内|外)'
        ]

        for pattern in location_patterns:
            matches = re.finditer(pattern, sentence)
            for match in matches:
                if match.group(0):
                    components['location'] = {
                        'text': match.group(0),
                        'position': match.start()
                    }
                    break

        # 方式状语 (通常以"地"结尾)
        manner_pattern = r'([\u4e00-\u9fff]{1,4})地'
        matches = re.finditer(manner_pattern, sentence)
        for match in matches:
            if match.group(0):
                components['manner'] = {
                    'text': match.group(0),
                    'position': match.start()
                }
                break

    def _needs_reordering(self, components: Dict) -> bool:
        """判断是否需要调整词序"""
        # 1. 检查是否有足够的成分可以调整
        has_verb = components['verb'] is not None
        has_subject = components['agent'] is not None
        has_object = components['patient'] is not None

        # 如果没有谓语动词，不需要调整
        if not has_verb:
            return False

        # 2. 检查SOV错误 (中文应该是SVO)
        if has_verb and has_object:
            verb_pos = components['verb']['position']
            obj_pos = components['patient']['position']

            # 如果是SOV结构 (动词在宾语后)
            if verb_pos > obj_pos:
                return True

        # 3. 检查状语位置错误
        adverbial_error = False
        verb_pos = components['verb']['position'] if has_verb else float('inf')

        # 时间状语应在句首或主语之后，动词之前
        if components['time'] and components['time']['position'] > verb_pos:
            adverbial_error = True

        # 地点状语应在动词前(方位)或后(目的地)
        # 这里简化处理：如果不在动词附近，认为是错误的
        if components['location']:
            loc_pos = components['location']['position']
            if abs(loc_pos - verb_pos) > 10:
                adverbial_error = True

        return adverbial_error

    def _reorder_components(self, sentence: str, components: Dict) -> str:
        """根据中文语法重新排列句子成分"""
        # 提取所有已识别的成分及其位置
        identified_parts = []
        for role, info in components.items():
            if info:
                identified_parts.append((
                    info['text'],
                    info['position'],
                    role
                ))

        # 按原始位置排序
        identified_parts.sort(key=lambda x: x[1])

        # 计算理想顺序
        ideal_order = []

        # 1. 主语 (agent)
        agent = components.get('agent')
        if agent:
            ideal_order.append(agent['text'])

        # 2. 时间状语
        time = components.get('time')
        if time:
            ideal_order.append(time['text'])

        # 3. 地点状语 (如果是位置而非目的地)
        location = components.get('location')
        if location and components.get('verb'):
            # 如果地点在动词前，可能是位置状语
            if location['position'] < components['verb']['position']:
                ideal_order.append(location['text'])

        # 4. 方式状语
        manner = components.get('manner')
        if manner:
            ideal_order.append(manner['text'])

        # 5. 谓语动词
        verb = components.get('verb')
        if verb:
            ideal_order.append(verb['text'])

        # 6. 补语
        # 7. 宾语
        patient = components.get('patient')
        if patient:
            ideal_order.append(patient['text'])

        # 8. 地点状语 (如果是目的地)
        if location and components.get('verb'):
            # 如果地点在动词后，可能是目的地
            if location['position'] > components['verb']['position']:
                ideal_order.append(location['text'])

        # 9. 目的状语
        purpose = components.get('purpose')
        if purpose:
            ideal_order.append(purpose['text'])

        # 检查是否与原始顺序不同
        original_order = [part[0] for part in identified_parts]
        if original_order == ideal_order:
            return sentence  # 顺序已经是理想的

        # 构建新句子
        return self._construct_reordered_sentence(sentence, identified_parts, ideal_order)

    def _construct_reordered_sentence(self, original: str, identified_parts: List[Tuple],
                                      ideal_order: List[str]) -> str:
        """构建重排后的句子，保留未识别部分"""
        # 创建标记映射：文本 -> (开始位置, 结束位置)
        text_spans = {}
        for text, pos, _ in identified_parts:
            text_spans[text] = (pos, pos + len(text))

        # 标记已识别的部分
        masked_sentence = list(original)
        for text, (start, end) in text_spans.items():
            for i in range(start, end):
                masked_sentence[i] = '□'  # 使用特殊字符标记

        # 提取未识别的部分及其位置
        unidentified_parts = []
        current_part = ''
        current_start = 0

        for i, char in enumerate(masked_sentence):
            if char == '□':
                if current_part:
                    unidentified_parts.append((current_part, current_start))
                    current_part = ''
            else:
                if not current_part:
                    current_start = i
                current_part += original[i]

        # 添加最后一个未识别部分
        if current_part:
            unidentified_parts.append((current_part, current_start))

        # 智能插入未识别部分
        result = ''
        last_identified_end = 0

        # 按理想顺序添加已识别部分，并在适当位置插入未识别部分
        for component in ideal_order:
            # 添加此组件前的未识别部分
            component_start = text_spans[component][0]
            for upart, ustart in unidentified_parts:
                if last_identified_end <= ustart < component_start:
                    result += upart

            # 添加组件
            result += component
            last_identified_end = text_spans[component][1]

        # 添加最后的未识别部分
        for upart, ustart in unidentified_parts:
            if ustart >= last_identified_end:
                result += upart

        return result

    def _apply_sentence_pattern(self, sentence: str, sentence_type: str) -> str:
        """
        基于句子类型智能应用句型模式，而非硬编码模板

        Args:
            sentence: 原始句子
            sentence_type: 句子类型(declarative, interrogative, imperative, exclamatory)

        Returns:
            应用句型后的句子
        """
        # 删除句尾标点
        clean_sentence = re.sub(r'[。！？；：，]$', '', sentence.strip())

        # 根据句型确定适当的句尾标点和调整
        if sentence_type == 'interrogative':  # 疑问句
            # 检查是否已有疑问标记
            has_question_marker = any(marker in clean_sentence for marker in ['吗', '呢', '吧', '啊'])

            # 分析句子判断是否为特殊疑问句
            is_special_question = any(
                marker in clean_sentence for marker in ['什么', '谁', '哪里', '为何', '如何', '多少'])

            if is_special_question:
                # 特殊疑问句不需要"吗"
                return clean_sentence + '？'
            elif not has_question_marker:
                # 一般疑问句添加"吗"
                return clean_sentence + '吗？'
            else:
                # 已有疑问标记，只添加问号
                return clean_sentence + '？'

        elif sentence_type == 'imperative':  # 祈使句
            # 检查句子是否已经有明显的命令标记
            has_command_marker = any(marker in clean_sentence for marker in ['请', '须', '应', '当', '切'])

            # 检查末尾是否已有感叹号
            if clean_sentence.endswith('！'):
                return clean_sentence

            # 根据语气强度选择标点
            if has_command_marker or any(word in clean_sentence for word in ['必须', '一定要', '切记']):
                return clean_sentence + '！'  # 强命令用感叹号
            else:
                return clean_sentence + '。'  # 柔和命令用句号

        elif sentence_type == 'exclamatory':  # 感叹句
            # 检查是否已有感叹词
            has_exclamation = any(marker in clean_sentence for marker in ['多么', '太', '真', '好', '真是'])

            return clean_sentence + '！'

        else:  # 陈述句 (declarative)
            # 保持简单，使用句号
            return clean_sentence + '。'

    def _merge_sentences(self, sentences: List[str]) -> str:
        """合并句子"""
        # 简单合并，去除多余空格
        text = ''.join(sentences)
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'\s+([。！？，；])', r'\1', text)
        return text.strip()

    def _global_grammar_fixes(self, text: str) -> str:
        """全局语法修正"""
        # 修正常见的语法错误
        fixes = [
            (r'的的+', '的'),
            (r'在在+', '在'),
            (r'把把+', '把'),
            (r'被被+', '被'),
            (r'和和+', '和'),
            (r'与与+', '与'),
            (r'([。！？])\1+', r'\1'),  # 重复的句号
            (r'，\s*，', '，'),  # 重复的逗号
            (r'。\s+', '。'),  # 句号后的空格
            (r'，\s+', '，'),  # 逗号后的空格
        ]

        for pattern, replacement in fixes:
            text = re.sub(pattern, replacement, text)

        return text

    def _handle_buddhist_terminology(self, sentence: str, context) -> str:
        """
        处理佛教术语的特殊语法和表达方式，减少硬编码规则

        Args:
            sentence: 原始句子
            context: 上下文信息

        Returns:
            处理后的句子
        """
        # 从适配器获取佛教语境信息

        buddhist_context = ProcessingAdapter.get_buddhist_context(context)

        # 1. 获取语境相关的处理规则
        patterns_to_apply = self._get_contextual_patterns(buddhist_context)

        # 2. 应用规则，减少硬编码
        for pattern_type, rules in patterns_to_apply.items():
            for pattern, replacement in rules:
                # 可以是正则表达式或简单字符串替换
                if isinstance(pattern, str):
                    sentence = sentence.replace(pattern, replacement)
                else:
                    sentence = re.sub(pattern, replacement, sentence)

        # 3. 智能处理特殊结构
        sentence = self._process_special_buddhist_structures(sentence, buddhist_context)

        return sentence

    def _get_contextual_patterns(self, buddhist_context: str) -> Dict:
        """根据佛教语境获取相应的处理模式"""
        # 基础处理模式，适用于所有佛教文本
        base_patterns = {
            'negation': [
                (r'不是不(\w+)', r'确实\1'),  # 双重否定转为肯定
                (r'无(\w+)非(\w+)', r'有\1是\2'),  # 特殊否定结构
            ],
            'realization': [
                (r'证(\w{1,3})(?!悟)', r'证悟\1'),  # 补充"悟"字，但避免重复
                (r'得(\w{1,3})果(?!位)', r'获得\1果位'),  # 标准化果位表述
            ]
        }

        # 根据特定语境添加额外规则
        if buddhist_context == 'MADHYAMIKA':  # 中观文献
            base_patterns['specific'] = [
                (r'(自性|本性)空', r'自性空性'),  # 标准化术语
                (r'二谛(?!正理)', r'二谛正理'),  # 补充完整术语
            ]
        elif buddhist_context == 'YOGACARA':  # 唯识文献
            base_patterns['specific'] = [
                (r'阿赖耶', r'阿赖耶识'),  # 补充完整术语
                (r'三性(?!三无性)', r'三性三无性'),  # 相关术语配对
            ]
        elif buddhist_context == 'VAJRAYANA':  # 密宗文献
            base_patterns['specific'] = [
                (r'本尊[^法]', r'本尊法'),  # 补充术语
                (r'灌顶(?!仪轨)', r'灌顶仪轨'),  # 补充相关术语
            ]

        return base_patterns

    def _process_special_buddhist_structures(self, sentence: str, buddhist_context: str) -> str:
        """处理佛教文献特有的句式结构"""
        # 1. 处理"如是我闻"开头
        if sentence.startswith('如是我闻') and len(sentence) > 4:
            return '如是我闻：' + sentence[4:]

        # 2. 处理偈颂引用
        if '如偈言' in sentence or '经中说' in sentence:
            # 分割引用部分
            for marker in ['如偈言', '经中说']:
                if marker in sentence:
                    parts = sentence.split(marker)
                    if len(parts) > 1:
                        # 格式化偈颂部分
                        quote = parts[1].strip()
                        # 四句偈处理：每两句一组，中间用顿号，组间用分号
                        if quote.count('，') >= 3:
                            sentences = quote.split('，')
                            if len(sentences) >= 4:
                                formatted = f"{sentences[0]}，{sentences[1]}；{sentences[2]}，{sentences[3]}"
                                sentence = parts[0] + marker + '：' + formatted
                    break

        # 3. 处理特定术语的搭配关系
        pairs = [
            ('空', '性'),
            ('无', '相'),
            ('无', '生'),
            ('真', '如'),
            ('法', '性')
        ]

        for first, second in pairs:
            # 确保术语搭配完整
            pattern = f"({first})(?!{second}\\b)\\b"
            replacement = f"\\1{second}"
            sentence = re.sub(pattern, replacement, sentence)

        return sentence