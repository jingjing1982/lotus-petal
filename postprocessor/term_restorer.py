"""
术语恢复器 - 将占位符替换回正确的中文术语
"""
import re
from typing import List, Dict, Optional, Any, Tuple
import logging

logger = logging.getLogger(__name__)


class TermRestorer:
    def __init__(self):
        """初始化术语恢复器"""
        # 上下文相关的术语选择规则
        self.context_rules = {
            '法': {
                'before_patterns': ['说', '讲', '听', '闻', '学', '修'],
                'after_patterns': ['门', '要', '义', '性', '相'],
                'default': '法'
            },
            '道': {
                'before_patterns': ['修', '行', '证', '入', '得'],
                'after_patterns': ['路', '次第', '果'],
                'default': '道'
            }
        }

    def restore(self, text: str, context) -> str:
        """恢复所有术语"""
        # 使用适配器获取术语映射
        from .adapter import ProcessingAdapter
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

        # 简单文本处理：如果术语较少，直接处理无需构建网络
        if len(term_mappings) <= 3:
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

        # 复杂文本处理：构建术语网络并考虑术语关系
        term_network = self._build_term_network(term_mappings)

        for placeholder in sorted_placeholders:
            if placeholder in restored_text:
                term_info = term_mappings[placeholder]

                # 获取最适合的中文翻译，考虑术语网络
                chinese_term = self._select_best_translation_with_network(
                    restored_text,
                    placeholder,
                    term_info,
                    term_network
                )

                # 执行替换
                restored_text = restored_text.replace(placeholder, chinese_term)

                logger.debug(f"Restored {placeholder} -> {chinese_term}")

        return restored_text

    def _build_term_network(self, term_mappings: Dict) -> Dict:
        """
        构建术语关系网络

        Args:
            term_mappings: 术语映射字典

        Returns:
            术语关系网络
        """
        term_network = {
            'terms': {},
            'relations': [],
            'groups': []
        }

        # 添加所有术语
        for placeholder, term_info in term_mappings.items():
            tibetan_term = term_info.get('tibetan', '')
            if tibetan_term:
                term_network['terms'][tibetan_term] = {
                    'placeholder': placeholder,
                    'info': term_info
                }

        # 查询术语之间的关系 - 只在有多个术语时进行
        if len(term_network['terms']) >= 2:
            for tibetan_term in term_network['terms'].keys():
                try:
                    related_terms = self.term_database.get_related_terms(tibetan_term)

                    for relation in related_terms:
                        related_term = relation['term']

                        # 只添加在当前文本中出现的术语之间的关系
                        if related_term in term_network['terms']:
                            term_network['relations'].append({
                                'source': tibetan_term,
                                'target': related_term,
                                'type': relation['relation_type'],
                                'confidence': relation['confidence']
                            })
                except Exception as e:
                    logger.error(f"获取术语关系失败: {e}")

            # 仅当有关系时才构建术语组
            if term_network['relations']:
                # 识别术语组（具有相同关系类型的术语集合）
                term_network['groups'] = self._identify_term_groups(
                    term_network['terms'],
                    term_network['relations']
                )

        return term_network

    def _identify_term_groups(self, terms: Dict, relations: List[Dict]) -> List[Dict]:
        """
        识别术语组

        Args:
            terms: 术语字典
            relations: 术语关系列表

        Returns:
            术语组列表
        """
        # 如果关系数量较少，使用简单分组
        if len(relations) < 5:
            return self._simple_group_by_type(relations)

        # 对于复杂关系网络，使用图算法
        groups = []

        # 按关系类型分组
        relation_by_type = {}
        for relation in relations:
            rel_type = relation['type']
            if rel_type not in relation_by_type:
                relation_by_type[rel_type] = []
            relation_by_type[rel_type].append(relation)

        # 为每种关系类型构建术语组
        for rel_type, rel_list in relation_by_type.items():
            # 构建关系图
            graph = {}
            for rel in rel_list:
                source = rel['source']
                target = rel['target']

                if source not in graph:
                    graph[source] = []
                if target not in graph:
                    graph[target] = []

                graph[source].append(target)
                if rel.get('bidirectional', False):
                    graph[target].append(source)

            # 使用DFS寻找连通分量（术语组）
            visited = set()

            for term in graph:
                if term not in visited:
                    group = []
                    self._dfs_collect_group(term, graph, visited, group)

                    if len(group) >= 2:  # 至少需要两个术语才构成组
                        groups.append({
                            'type': rel_type,
                            'terms': group
                        })

        return groups

    def _simple_group_by_type(self, relations: List[Dict]) -> List[Dict]:
        """简单的按关系类型分组"""
        groups_by_type = {}

        for relation in relations:
            rel_type = relation['type']
            source = relation['source']
            target = relation['target']

            if rel_type not in groups_by_type:
                groups_by_type[rel_type] = set()

            groups_by_type[rel_type].add(source)
            groups_by_type[rel_type].add(target)

        # 转换为预期格式
        result = []
        for rel_type, terms in groups_by_type.items():
            if len(terms) >= 2:
                result.append({
                    'type': rel_type,
                    'terms': list(terms)
                })

        return result

    def _dfs_collect_group(self, term, graph, visited, group):
        """使用DFS收集术语组"""
        visited.add(term)
        group.append(term)

        for neighbor in graph.get(term, []):
            if neighbor not in visited:
                self._dfs_collect_group(neighbor, graph, visited, group)

    def _select_best_translation_with_network(self, text: str, placeholder: str,
                                              term_info: Dict, term_network: Dict) -> str:
        """
        考虑术语网络选择最佳翻译

        Args:
            text: 包含占位符的文本
            placeholder: 术语占位符
            term_info: 术语信息
            term_network: 术语关系网络

        Returns:
            选择的中文翻译
        """
        tibetan_term = term_info.get('tibetan', '')

        if not tibetan_term:
            return term_info.get('default_translation', placeholder)

        # 1. 首先使用标准方法获取最佳翻译
        best_translation = self._select_best_translation(text, placeholder, term_info)

        # 2. 检查是否需要考虑术语组一致性
        if tibetan_term in term_network['terms']:
            # 查找该术语所在的组
            for group in term_network['groups']:
                if tibetan_term in group['terms']:
                    # 根据组的类型调整翻译
                    adjusted = self._adjust_translation_for_group(
                        best_translation, tibetan_term, group, term_network, text
                    )

                    if adjusted != best_translation:
                        logger.info(f"调整术语翻译以保持一致性: {best_translation} -> {adjusted}")
                        return adjusted

        return best_translation

    def _adjust_translation_for_group(self, translation: str, term: str,
                                      group: Dict, term_network: Dict, text: str) -> str:
        """
        根据术语组调整翻译

        Args:
            translation: 原始翻译
            term: 藏文术语
            group: 术语组信息
            term_network: 术语关系网络
            text: 完整文本

        Returns:
            调整后的翻译
        """
        group_type = group['type']

        # 对不同类型的关系进行特殊处理
        if group_type == 'opposite':
            # 确保对立概念使用对立的词语
            for other_term in group['terms']:
                if other_term != term:
                    # 获取对立术语的占位符和信息
                    other_info = term_network['terms'].get(other_term, {})
                    other_placeholder = other_info.get('placeholder')

                    # 如果对立术语已被翻译
                    if other_placeholder and other_placeholder not in text:
                        # 尝试查找对立术语的翻译
                        try:
                            translations = self.term_database.get_translations_for_term(other_term)
                            if translations:
                                other_translation = translations[0].get('chinese', '')

                                # 确保翻译词对立
                                if self._are_terms_opposite(translation, other_translation):
                                    # 保持对立关系
                                    return translation
                                else:
                                    # 找到更好的对立术语
                                    better_opposite = self._find_better_opposite(translation, other_translation)
                                    if better_opposite:
                                        return better_opposite
                        except Exception as e:
                            logger.error(f"检查对立术语失败: {e}")

        elif group_type == 'includes':
            # 确保上位概念和下位概念的翻译一致
            for relation in term_network['relations']:
                if relation['type'] == 'includes' and (relation['source'] == term or relation['target'] == term):
                    other_term = relation['target'] if relation['source'] == term else relation['source']

                    # 检查是否是上位概念
                    is_superordinate = relation['source'] == term

                    if other_term in term_network['terms']:
                        # 获取其他术语的占位符和信息
                        other_info = term_network['terms'].get(other_term, {})
                        other_placeholder = other_info.get('placeholder')

                        # 如果其他术语已被翻译
                        if other_placeholder and other_placeholder not in text:
                            # 尝试查找其他术语的翻译
                            try:
                                translations = self.term_database.get_translations_for_term(other_term)
                                if translations:
                                    other_translation = translations[0].get('chinese', '')

                                    # 确保包含关系一致
                                    if is_superordinate:
                                        # 当前术语是上位概念
                                        if not self._is_superordinate_term(translation, other_translation):
                                            # 调整翻译以保持上位关系
                                            return self._adjust_for_superordinate(translation, other_translation)
                                    else:
                                        # 当前术语是下位概念
                                        if not self._is_subordinate_term(translation, other_translation):
                                            # 调整翻译以保持下位关系
                                            return self._adjust_for_subordinate(translation, other_translation)
                            except Exception as e:
                                logger.error(f"检查包含术语失败: {e}")

        return translation

    def _are_terms_opposite(self, term1: str, term2: str) -> bool:
        """检查两个术语是否为对立关系"""
        # 常见的对立词对
        opposite_pairs = [
            ('有', '无'), ('是', '非'), ('善', '恶'), ('净', '垢'),
            ('常', '无常'), ('空', '不空'), ('生', '灭'), ('有为', '无为')
        ]

        # 检查两个术语是否构成对立词对
        for t1, t2 in opposite_pairs:
            if (t1 in term1 and t2 in term2) or (t2 in term1 and t1 in term2):
                return True

        return False

    def _find_better_opposite(self, term1: str, term2: str) -> Optional[str]:
        """找到更好的对立术语"""
        # 常见的对立词对替换
        opposite_mappings = {
            '有': '无', '无': '有',
            '是': '非', '非': '是',
            '善': '恶', '恶': '善',
            '净': '垢', '垢': '净',
            '常': '无常', '无常': '常',
            '空': '不空', '不空': '空',
            '生': '灭', '灭': '生'
        }

        # 尝试替换词语的一部分
        for original, opposite in opposite_mappings.items():
            if original in term1:
                return term1.replace(original, opposite)

        return None

    def _is_superordinate_term(self, superordinate: str, subordinate: str) -> bool:
        """检查第一个术语是否为第二个术语的上位概念"""
        # 检查包含关系
        if superordinate in subordinate:
            return True

        # 检查类别关系
        category_words = ['法', '道', '乘', '宗', '门', '藏']
        for word in category_words:
            if superordinate.endswith(word) and not subordinate.endswith(word):
                return True

        return False

    def _adjust_for_superordinate(self, current: str, subordinate: str) -> str:
        """调整翻译以保持上位关系"""
        # 如果下位术语包含当前术语，可能不需要调整
        if current in subordinate:
            return current

        # 尝试添加类别词
        if not any(current.endswith(word) for word in ['法', '道', '乘', '宗', '门', '藏']):
            # 根据下位术语的性质选择合适的类别词
            if '禅' in subordinate or '观' in subordinate:
                return current + '法'
            elif '见' in subordinate:
                return current + '见'
            else:
                return current + '门'

        return current

    def _is_subordinate_term(self, subordinate: str, superordinate: str) -> bool:
        """检查第一个术语是否为第二个术语的下位概念"""
        # 与上位函数相反
        return self._is_superordinate_term(superordinate, subordinate)

    def _adjust_for_subordinate(self, current: str, superordinate: str) -> str:
        """调整翻译以保持下位关系"""
        # 如果当前术语已经包含上位术语，可能不需要调整
        if superordinate in current:
            return current

        # 尝试添加上位术语作为前缀
        if len(superordinate) <= 2:  # 上位术语较短时才添加
            return superordinate + current

        return current

    def _select_best_translation(self, text: str, placeholder: str, term_info: Dict) -> str:
        """
        根据上下文选择最佳翻译

        Args:
            text: 包含占位符的文本
            placeholder: 术语占位符
            term_info: 术语信息

        Returns:
            选择的中文翻译
        """
        # 1. 获取术语信息
        tibetan_term = term_info.get('tibetan', '')

        if not tibetan_term:
            return term_info.get('default_translation', placeholder)

        # 2. 提取术语上下文
        term_context = self._extract_term_context(text, placeholder)

        # 3. 从数据库获取候选翻译
        try:
            translations = self.term_database.get_translations_for_term(tibetan_term)
        except Exception as e:
            logger.error(f"获取术语翻译失败: {e}")
            translations = []

        # 如果没有找到翻译，返回默认翻译
        if not translations:
            return term_info.get('default_translation', tibetan_term)

        # 4. 获取佛教语境信息
        buddhist_context = term_info.get('context', {}).get('buddhist_context', 'GENERAL')

        # 5. 为每个翻译评分
        scored_translations = []

        for translation in translations:
            chinese_term = translation.get('chinese', '')
            if not chinese_term:
                continue

            # 计算翻译分数
            score = self._calculate_translation_score(
                chinese_term, translation, term_context, buddhist_context
            )

            scored_translations.append((chinese_term, score, translation))

        # 6. 选择最高分的翻译
        if not scored_translations:
            return term_info.get('default_translation', tibetan_term)

        scored_translations.sort(key=lambda x: x[1], reverse=True)
        best_translation, best_score, trans_info = scored_translations[0]

        # 7. 应用语境适配（如数量、敬语等）
        adapted_translation = self._adapt_translation_to_context(
            best_translation, trans_info, text, placeholder, term_info
        )

        return adapted_translation

    def _extract_term_context(self, text: str, placeholder: str) -> Dict:
        """
        提取术语的上下文信息

        Args:
            text: 包含占位符的文本
            placeholder: 术语占位符

        Returns:
            上下文信息字典
        """
        # 找到占位符的位置
        pos = text.find(placeholder)
        if pos == -1:
            return {'before': '', 'after': '', 'full_text': text}

        # 提取前后文本
        context_window = 50  # 上下文窗口大小
        before_text = text[max(0, pos - context_window):pos]
        after_text = text[pos + len(placeholder):min(len(text), pos + len(placeholder) + context_window)]

        # 分析句子位置
        sentence_start = text.rfind('。', 0, pos)
        sentence_start = sentence_start + 1 if sentence_start != -1 else 0

        sentence_end = text.find('。', pos)
        sentence_end = sentence_end if sentence_end != -1 else len(text)

        current_sentence = text[sentence_start:sentence_end]

        # 检测术语在句子中的位置
        position_in_sentence = 'middle'
        if pos - sentence_start < 10:
            position_in_sentence = 'beginning'
        elif sentence_end - (pos + len(placeholder)) < 10:
            position_in_sentence = 'end'

        return {
            'before': before_text,
            'after': after_text,
            'full_text': text,
            'current_sentence': current_sentence,
            'position_in_sentence': position_in_sentence,
            'placeholder_position': pos
        }

    def _calculate_translation_score(self, chinese_term: str, translation_info: Dict,
                                     context: Dict, buddhist_context: str) -> float:
        """
        计算翻译在给定上下文中的适合度分数

        Args:
            chinese_term: 中文翻译
            translation_info: 翻译信息
            context: 上下文信息
            buddhist_context: 佛教语境类型

        Returns:
            分数 (0.0-1.0)
        """
        score = 0.0

        # 1. 基础置信度分数 (0-0.3)
        base_confidence = translation_info.get('confidence', 0.5)
        score += base_confidence * 0.3

        # 2. 佛教语境匹配分数 (0-0.25)
        trans_context = translation_info.get('context', 'GENERAL')

        if trans_context == buddhist_context:
            score += 0.25  # 完全匹配
        elif trans_context == 'GENERAL':
            score += 0.1  # 通用术语
        elif buddhist_context == 'GENERAL':
            score += 0.15  # 文本是通用语境，任何专业术语都可接受
        else:
            # 部分匹配 (例如，MADHYAMIKA_PRASANGIKA 和 MADHYAMIKA)
            if trans_context.split('_')[0] == buddhist_context.split('_')[0]:
                score += 0.15

        # 3. 术语位置相关性 (0-0.15)
        position = context.get('position_in_sentence', 'middle')
        term_position_score = 0

        if position == 'beginning' and translation_info.get('good_for_start', False):
            term_position_score = 0.15
        elif position == 'end' and translation_info.get('good_for_end', False):
            term_position_score = 0.15
        elif translation_info.get('position_neutral', True):
            term_position_score = 0.1

        score += term_position_score

        # 4. 上下文词汇匹配 (0-0.2)
        context_text = context.get('before', '') + ' ' + context.get('after', '')
        related_terms = translation_info.get('related_terms', [])

        if related_terms:
            match_count = sum(1 for term in related_terms if term in context_text)
            if match_count > 0:
                score += min(0.2, match_count * 0.05)

        # 5. 使用频率分数 (0-0.1)
        usage_count = translation_info.get('usage_count', 0)
        if usage_count > 100:
            score += 0.1
        elif usage_count > 50:
            score += 0.08
        elif usage_count > 20:
            score += 0.05
        elif usage_count > 5:
            score += 0.03

        return min(1.0, score)

    def _adapt_translation_to_context(self, translation: str, translation_info: Dict,
                                      text: str, placeholder: str, term_info: Dict) -> str:
        """
        根据上下文调整翻译

        Args:
            translation: 基础翻译
            translation_info: 翻译信息
            text: 原文
            placeholder: 占位符
            term_info: 术语信息

        Returns:
            调整后的翻译
        """
        # 1. 检查数量
        if self._is_plural_context(text, placeholder):
            pluralized = self._pluralize_term(translation, translation_info)
            if pluralized != translation:
                return pluralized

        # 2. 检查敬语
        if self._is_honorific_context(text, placeholder):
            honorific = self._apply_honorific(translation, translation_info)
            if honorific != translation:
                return honorific

        # 3. 检查否定
        if self._is_negation_context(text, placeholder):
            negated = self._apply_negation(translation, translation_info)
            if negated != translation:
                return negated

        # 4. 检查是否需要缩写/全称
        if self._should_use_abbreviation(text, placeholder):
            abbreviated = translation_info.get('abbreviated', translation)
            if abbreviated and len(abbreviated) < len(translation):
                return abbreviated
        else:
            full_form = translation_info.get('full_form', translation)
            if full_form and len(full_form) > len(translation):
                return full_form

        return translation

    def _is_plural_context(self, text: str, placeholder: str) -> bool:
        """检查是否为复数上下文"""
        # 找到占位符位置
        pos = text.find(placeholder)
        if pos == -1:
            return False

        # 检查前面是否有数量词
        before_text = text[max(0, pos - 10):pos]

        # 检查数量词
        quantity_markers = ['多', '些', '几', '众', '诸', '双', '对']
        has_quantity = any(marker in before_text for marker in quantity_markers)

        # 检查数字
        has_number = any(num in before_text for num in '一二三四五六七八九十百千万亿')
        has_arabic = re.search(r'\d+', before_text) is not None

        return has_quantity or has_number or has_arabic

    def _pluralize_term(self, term: str, translation_info: Dict) -> str:
        """将术语变为复数形式"""
        # 如果术语信息中有复数形式，使用它
        if 'plural_form' in translation_info:
            return translation_info['plural_form']

        # 一些术语在复数时需要加"诸"或"众"前缀
        if translation_info.get('type') == 'being' or translation_info.get('type') == 'deity':
            if not term.startswith('诸') and not term.startswith('众'):
                return '诸' + term

        # 一些术语加"们"后缀
        if translation_info.get('type') == 'person' and not term.endswith('们'):
            # 除非已经有复数标记
            if not any(marker in term for marker in ['等', '诸', '众', '辈']):
                return term + '们'

        return term

    def _is_honorific_context(self, text: str, placeholder: str) -> bool:
        """检查是否为敬语上下文"""
        # 找到占位符位置
        pos = text.find(placeholder)
        if pos == -1:
            return False

        # 提取前后文本
        before_text = text[max(0, pos - 30):pos]
        after_text = text[pos + len(placeholder):min(len(text), pos + len(placeholder) + 30)]

        # 检查敬语标记
        honorific_markers = ['尊', '圣', '佛', '菩萨', '上师', '尊者', '世尊']

        return any(marker in before_text or marker in after_text for marker in honorific_markers)


    def _should_use_extended_form(self, text: str, placeholder: str, base_term: str) -> bool:
        """判断是否应该使用扩展形式"""
        # 获取占位符前后文
        pos = text.find(placeholder)
        if pos == -1:
            return False

        before_text = text[max(0, pos - 10):pos]
        after_text = text[pos + len(placeholder):pos + len(placeholder) + 10]

        # 如果前面有"听"、"闻"、"学"等动词，使用"佛法"
        extended_triggers = ['听', '闻', '学', '讲', '说', '传']
        for trigger in extended_triggers:
            if trigger in before_text:
                return True

        return False

    def _apply_honorific(self, term: str, translation_info: Dict) -> str:
        """应用敬语形式"""
        # 如果术语信息中有敬语形式，使用它
        if 'honorific_form' in translation_info:
            return translation_info['honorific_form']

        # 一些术语在敬语时需要加特定前缀
        if translation_info.get('type') == 'person':
            if not any(prefix in term for prefix in ['尊', '圣', '佛']):
                return '尊' + term

        return term

    def _is_negation_context(self, text: str, placeholder: str) -> bool:
        """检查是否为否定上下文"""
        # 找到占位符位置
        pos = text.find(placeholder)
        if pos == -1:
            return False

        # 提取前后文本
        before_text = text[max(0, pos - 10):pos]
        after_text = text[pos + len(placeholder):min(len(text), pos + len(placeholder) + 10)]

        # 检查否定标记
        negation_markers = ['不', '非', '无', '未', '莫', '勿', '没']

        return any(marker in before_text for marker in negation_markers)

    def _apply_negation(self, term: str, translation_info: Dict) -> str:
        """应用否定形式"""
        # 如果术语信息中有否定形式，使用它
        if 'negated_form' in translation_info:
            return translation_info['negated_form']

        # 一些术语有特定的否定形式
        negation_mappings = {
            '有': '无',
            '是': '非',
            '可': '不可',
            '能': '不能'
        }

        for positive, negative in negation_mappings.items():
            if term.startswith(positive):
                return term.replace(positive, negative, 1)

        return term

    def _should_use_abbreviation(self, text: str, placeholder: str) -> bool:
        """检查是否应使用缩略形式"""
        # 找到占位符位置
        pos = text.find(placeholder)
        if pos == -1:
            return False

        # 检查是否已经出现过完整形式
        full_text = text[:pos]

        # 如果是第二次或更多次出现，通常使用缩写
        term_count = full_text.count(placeholder)

        return term_count > 0

    def _has_quantity_before(self, text: str, placeholder: str) -> bool:
        """检查占位符前是否有数量词"""
        pos = text.find(placeholder)
        if pos == -1:
            return False

        before_text = text[max(0, pos - 5):pos]
        quantity_words = ['诸', '众', '一切', '所有', '多', '无量', '无数']

        return any(word in before_text for word in quantity_words)

    def _pluralize_if_needed(self, term: str, term_info: Dict) -> str:
        """根据需要复数化术语"""
        # 中文通常不需要复数形式，但某些情况下可以调整
        # 例如："菩萨"在"诸菩萨"中不需要变化
        # "众生"本身就是复数概念
        return term