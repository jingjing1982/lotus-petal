"""
上下文提取器 - 提取并组织翻译所需的上下文信息
"""
from typing import Dict, List, Optional
from dataclasses import dataclass, field


@dataclass
class TranslationContext:
    """翻译上下文数据结构"""
    # 原始文本
    original_text: str = ""

    # 分词信息
    tokens: List[Dict] = field(default_factory=list)

    # 句子信息
    sentences: List[Dict] = field(default_factory=list)

    # 术语映射
    term_mappings: Dict[str, Dict] = field(default_factory=dict)

    # 语法分析
    grammatical_analyses: List[Dict] = field(default_factory=list)

    # 敬语标记
    honorific_markers: List[List[str]] = field(default_factory=list)

    # 结构信息
    has_parallel_structure: bool = False
    parallel_patterns: List[Dict] = field(default_factory=list)
    has_enumeration: bool = False
    enumeration_items: List[Dict] = field(default_factory=list)
    is_verse: bool = False
    verse_structure: Optional[Dict] = None

    # 句子边界
    sentence_boundaries: List[int] = field(default_factory=list)

    # 质量评分
    confidence_scores: Dict[str, float] = field(default_factory=dict)

    # 格助词信息
    case_particles: List[Dict] = field(default_factory=list)

    # 时态信息
    tense_info: Dict[str, str] = field(default_factory=dict)


class ContextExtractor:
    def __init__(self):
        """初始化上下文提取器"""
        self.min_confidence = 0.7

    # 在 context_extractor.py 的 extract_context 方法中修改
    def extract_context(self, botok_analysis: Dict, protection_map: Dict) -> TranslationContext:
        """
        从Botok分析结果中提取翻译上下文
        """
        context = TranslationContext()

        # 提取基础信息
        context.original_text = botok_analysis.get('original_text', '')
        context.tokens = botok_analysis.get('tokens', [])
        context.sentences = botok_analysis.get('sentences', [])

        # 提取术语映射
        context.term_mappings = protection_map

        # 提取语法分析
        self._extract_grammatical_info(botok_analysis, context)

        # 提取敬语信息
        self._extract_honorific_info(botok_analysis, context)

        # 提取结构信息
        self._extract_structure_info(botok_analysis, context)

        # 提取句子边界
        self._extract_sentence_boundaries(botok_analysis, context)

        # 计算置信度
        self._calculate_confidence_scores(context)

        # 添加原始语法分析供后处理使用 (新增)
        context.original_grammar_analysis = botok_analysis.get('grammar', {})

        return context

    def _extract_grammatical_info(self, analysis: Dict, context: TranslationContext):
        """提取语法信息"""
        grammar = analysis.get('grammar', {})

        # 提取格助词信息
        context.case_particles = grammar.get('case_particles', [])

        # 为每个句子创建语法分析
        for sentence in context.sentences:
            sentence_grammar = {
                'type': sentence.get('type', 'declarative'),
                'case_particles': [],
                'syntactic_roles': [],
                'tense': None
            }

            # 找出属于这个句子的格助词
            sentence_start = sentence['start']
            sentence_end = sentence['end']

            for particle in context.case_particles:
                if sentence_start <= particle.get('position', 0) < sentence_end:
                    sentence_grammar['case_particles'].append(particle)

            # 找出句法角色
            for role in grammar.get('syntactic_roles', []):
                if sentence_start <= role.get('position', 0) < sentence_end:
                    sentence_grammar['syntactic_roles'].append(role)

            # 确定时态
            sentence_grammar['tense'] = self._determine_sentence_tense(sentence, grammar)

            context.grammatical_analyses.append(sentence_grammar)

    def _determine_sentence_tense(self, sentence: Dict, grammar: Dict) -> Optional[str]:
        """确定句子时态"""
        # 首先检查整体时态标记
        if grammar.get('tense'):
            return grammar['tense']

        # 检查句子中的动词和时态标记
        sentence_tokens = sentence.get('tokens', [])
        for token in sentence_tokens:
            if token.get('pos') == 'VERB':
                # 根据动词形态判断时态
                verb_text = token.get('text', '')
                if verb_text.endswith(('བྱུང་', 'སོང་')):
                    return 'past'
                elif verb_text.endswith(('གི་', 'གིན་')):
                    return 'present_progressive'
                elif verb_text.endswith('རྒྱུ་'):
                    return 'future'

        return 'present'  # 默认现在时

    def _extract_honorific_info(self, analysis: Dict, context: TranslationContext):
        """提取敬语信息"""
        honorifics = analysis.get('honorifics', [])

        # 为每个句子创建敬语标记列表
        for sentence in context.sentences:
            sentence_markers = []
            sentence_start = sentence['start']
            sentence_end = sentence['end']

            for honorific in honorifics:
                if sentence_start <= honorific.get('position', 0) < sentence_end:
                    marker_type = honorific.get('type', 'general_honorific')
                    if marker_type not in sentence_markers:
                        sentence_markers.append(marker_type)

            context.honorific_markers.append(sentence_markers)

    def _extract_structure_info(self, botok_analysis: Dict, context) -> None:
        """
        提取文本结构信息（诗偈、列举、平行结构等）

        Args:
            botok_analysis: Botok分析结果
            context: 翻译上下文对象
        """
        # 默认设置
        context.is_verse = False
        context.has_enumeration = False
        context.has_parallel_structure = False
        context.verse_structure = None
        context.enumeration_structure = None
        context.parallel_structure = None

        # 获取原始文本和句子
        text = botok_analysis.get('original_text', '')
        sentences = botok_analysis.get('sentences', [])

        # 1. 诗偈检测（韵律、规则行长、特殊标记）
        verse_detected = self._detect_verse_structure(text, sentences)
        if verse_detected:
            context.is_verse = True
            context.verse_structure = verse_detected

        # 2. 列举结构检测
        enumeration_detected = self._detect_enumeration_structure(text, sentences, botok_analysis)
        if enumeration_detected:
            context.has_enumeration = True
            context.enumeration_structure = enumeration_detected

        # 3. 平行结构检测
        parallel_detected = self._detect_parallel_structure(text, sentences, botok_analysis)
        if parallel_detected:
            context.has_parallel_structure = True
            context.parallel_structure = parallel_detected

    def _detect_verse_structure(self, text: str, sentences: List[Dict]) -> Optional[Dict]:
        """
        检测文本是否为诗偈结构

        返回格式:
        {
            'type': 'verse',
            'line_count': 总行数,
            'stanza_count': 总节数,
            'lines': [行信息列表],
            'stanzas': [节信息列表],
            'average_length': 平均行长度,
            'pattern': 韵律模式
        }
        """
        # 如果没有足够的句子，不可能是诗偈
        if len(sentences) < 2:
            return None

        verse_signs = [
            'ཤླཽ', 'ཞེས་', 'གསུངས་', 'སོ།', 'ཅེས་'  # 藏文中诗偈的常见标记
        ]

        # 检查诗偈标记
        has_verse_marker = any(sign in text for sign in verse_signs)

        # 提取可能的行
        lines = []
        line_texts = []
        current_stanza = []
        stanzas = []
        line_lengths = []

        # 基于句子分析潜在的诗行
        for sentence in sentences:
            sentence_text = sentence.get('text', '')

            # 跳过太长的句子，可能不是诗行
            if len(sentence_text) > 40:
                continue

            # 检测句子是否有诗行特征
            if self._has_verse_line_features(sentence_text):
                line_info = {
                    'text': sentence_text,
                    'start': sentence.get('start', 0),
                    'end': sentence.get('end', 0),
                    'length': len(sentence_text),
                    'has_end_marker': any(sign in sentence_text for sign in verse_signs)
                }

                lines.append(line_info)
                line_texts.append(sentence_text)
                line_lengths.append(len(sentence_text))
                current_stanza.append(line_info)

                # 检测是否为诗节结束
                if line_info['has_end_marker'] and current_stanza:
                    stanzas.append(current_stanza.copy())
                    current_stanza = []

        # 如果最后一个诗节未结束，添加它
        if current_stanza:
            stanzas.append(current_stanza)

        # 判断是否为诗偈结构
        if not lines or len(lines) < 2:
            return None

        # 计算行长度方差，诗偈通常有相似的行长度
        if line_lengths:
            avg_length = sum(line_lengths) / len(line_lengths)
            variance = sum((l - avg_length) ** 2 for l in line_lengths) / len(line_lengths)

            # 方差太大，行长度不一致，可能不是诗偈
            if variance > 100 and not has_verse_marker:
                return None

        # 分析韵律模式
        rhythm_pattern = self._analyze_rhythm_pattern(line_texts)

        return {
            'type': 'verse',
            'line_count': len(lines),
            'stanza_count': len(stanzas),
            'lines': lines,
            'stanzas': stanzas,
            'average_length': avg_length if line_lengths else 0,
            'pattern': rhythm_pattern
        }

    def _has_verse_line_features(self, text: str) -> bool:
        """检查文本是否具有诗行特征"""
        # 常见诗行末尾标志
        end_markers = ['།', '་', 'ཅེ', 'སོ']

        # 检查长度 - 诗行通常较短
        if len(text) > 40:
            return False

        # 检查末尾标志
        has_end_marker = any(text.endswith(marker) for marker in end_markers)

        # 检查结构特征 - 诗行通常有固定的语法结构
        has_structure_feature = '་' in text and text.count('་') < 10

        return has_end_marker or has_structure_feature

    def _analyze_rhythm_pattern(self, lines: List[str]) -> str:
        """分析诗行的韵律模式"""
        if not lines or len(lines) < 2:
            return 'unknown'

        # 分析藏文诗歌常见的7音节或9音节模式
        syllable_counts = []
        for line in lines:
            # 粗略估计音节数（藏文中通常用'་'分隔音节）
            syllables = line.split('་')
            syllable_counts.append(len(syllables))

        # 检查是否有一致的模式
        if all(count == 7 for count in syllable_counts):
            return '7-syllable'
        elif all(count == 9 for count in syllable_counts):
            return '9-syllable'
        elif len(syllable_counts) >= 4 and syllable_counts[0::2] == syllable_counts[1::2]:
            return 'alternating'
        else:
            return 'irregular'

    def _detect_enumeration_structure(self, text: str, sentences: List[Dict], botok_analysis: Dict) -> Optional[Dict]:
        """
        检测文本是否包含列举结构

        返回格式:
        {
            'type': 'enumeration',
            'items': [列举项列表],
            'marker_type': 标记类型（'numeric', 'sequence', 'bullet'),
            'total_items': 总项数
        }
        """
        # 藏文中的序数词和列举标记
        sequence_markers = ['དང་པོ', 'གཉིས་པ', 'གསུམ་པ', 'བཞི་པ', 'ལྔ་པ']  # 第一、第二...
        bullet_markers = ['༡', '༢', '༣', '༤', '༥']  # 藏文数字1,2,3,4,5

        # 查找数字标记
        numeric_pattern = r'(\d+[\.\)༽]|[༡-༩][\.\)༽])'
        numeric_matches = re.finditer(numeric_pattern, text)
        numeric_positions = [match.start() for match in numeric_matches]

        # 查找序数词标记
        sequence_positions = []
        for marker in sequence_markers:
            start = 0
            while True:
                pos = text.find(marker, start)
                if pos == -1:
                    break
                sequence_positions.append(pos)
                start = pos + len(marker)

        # 查找项目符号标记
        bullet_positions = []
        for marker in bullet_markers:
            start = 0
            while True:
                pos = text.find(marker, start)
                if pos == -1:
                    break
                bullet_positions.append(pos)
                start = pos + len(marker)

        # 分析标记类型
        marker_positions = []
        marker_type = None

        if len(numeric_positions) >= 2:
            marker_positions = numeric_positions
            marker_type = 'numeric'
        elif len(sequence_positions) >= 2:
            marker_positions = sequence_positions
            marker_type = 'sequence'
        elif len(bullet_positions) >= 2:
            marker_positions = bullet_positions
            marker_type = 'bullet'

        if not marker_type:
            # 尝试识别"一方面...另一方面"类型的列举
            alternative_markers = ['གཅིག་', 'གཞན་']
            alt_positions = []

            for marker in alternative_markers:
                pos = text.find(marker)
                if pos != -1:
                    alt_positions.append(pos)

            if len(alt_positions) >= 2:
                marker_positions = alt_positions
                marker_type = 'alternative'

        # 如果没有找到足够的标记，不是列举结构
        if not marker_type or len(marker_positions) < 2:
            return None

        # 提取列举项
        items = []
        marker_positions.sort()

        for i in range(len(marker_positions)):
            start = marker_positions[i]
            end = marker_positions[i + 1] if i + 1 < len(marker_positions) else len(text)

            # 寻找列举项的边界
            item_end = text.find('།', start, end)
            if item_end == -1:
                item_end = end

            item_text = text[start:item_end].strip()

            items.append({
                'position': i + 1,
                'text': item_text,
                'start': start,
                'end': item_end
            })

        return {
            'type': 'enumeration',
            'items': items,
            'marker_type': marker_type,
            'total_items': len(items)
        }

    def _detect_parallel_structure(self, text: str, sentences: List[Dict], botok_analysis: Dict) -> Optional[Dict]:
        """
        检测文本是否包含平行结构

        返回格式:
        {
            'type': 'parallel',
            'patterns': [平行模式列表],
            'total_patterns': 总模式数
        }
        """
        # 常见的平行结构标记
        parallel_markers = [
            'ཡང་', 'ནི་', 'དང་', 'མ་', 'མི་'  # 藏文中常见的平行结构标记
        ]

        # 分析词语重复模式
        tokens = botok_analysis.get('tokens', [])
        token_texts = [t.get('text', '') for t in tokens]

        # 查找重复模式
        patterns = []

        # 1. 检测否定平行结构（如：不X...不Y...不Z...）
        negation_pattern = self._detect_negation_parallel(token_texts, text)
        if negation_pattern:
            patterns.append(negation_pattern)

        # 2. 检测连词平行结构（如：X和Y和Z）
        conjunction_pattern = self._detect_conjunction_parallel(token_texts, text)
        if conjunction_pattern:
            patterns.append(conjunction_pattern)

        # 3. 检测重复词平行结构
        repetition_pattern = self._detect_repetition_parallel(token_texts, text)
        if repetition_pattern:
            patterns.append(repetition_pattern)

        # 如果没有找到平行结构，返回None
        if not patterns:
            return None

        return {
            'type': 'parallel',
            'patterns': patterns,
            'total_patterns': len(patterns)
        }

    def _detect_negation_parallel(self, tokens: List[str], text: str) -> Optional[Dict]:
        """检测否定平行结构"""
        negation_markers = ['མ་', 'མི་', 'མེད་']  # 藏文否定词

        positions = []
        for i, token in enumerate(tokens):
            if any(token.startswith(marker) for marker in negation_markers):
                # 找到token在原文中的位置
                start = text.find(token)
                if start != -1:
                    positions.append({
                        'token': token,
                        'position': start,
                        'index': i
                    })

        # 需要至少3个否定词才构成平行结构
        if len(positions) >= 3:
            return {
                'type': 'negation',
                'positions': positions,
                'count': len(positions)
            }

        return None

    def _detect_conjunction_parallel(self, tokens: List[str], text: str) -> Optional[Dict]:
        """检测连词平行结构"""
        conjunction_markers = ['དང་', 'ཡང་']  # 藏文连词

        positions = []
        for i, token in enumerate(tokens):
            if token in conjunction_markers:
                # 需要检查连词前后的词
                if i > 0 and i < len(tokens) - 1:
                    # 找到在原文中的位置
                    start = text.find(token)
                    if start != -1:
                        positions.append({
                            'token': token,
                            'position': start,
                            'index': i,
                            'before': tokens[i - 1] if i > 0 else '',
                            'after': tokens[i + 1] if i < len(tokens) - 1 else ''
                        })

        # 需要至少2个连词才构成平行结构
        if len(positions) >= 2:
            return {
                'type': 'conjunction',
                'positions': positions,
                'count': len(positions)
            }

        return None

    def _detect_repetition_parallel(self, tokens: List[str], text: str) -> Optional[Dict]:
        """检测重复词平行结构"""
        # 构建词频字典
        token_freq = {}
        for i, token in enumerate(tokens):
            if len(token) >= 2:  # 忽略太短的词
                if token not in token_freq:
                    token_freq[token] = []
                token_freq[token].append(i)

        # 寻找多次重复的词
        repeated_tokens = {t: pos for t, pos in token_freq.items() if len(pos) >= 3}

        if not repeated_tokens:
            return None

        # 找出重复最多的词
        most_repeated = max(repeated_tokens.items(), key=lambda x: len(x[1]))

        # 构建重复模式
        token, positions = most_repeated

        # 检查位置是否形成模式（大致等距）
        if len(positions) >= 3:
            distances = [positions[i + 1] - positions[i] for i in range(len(positions) - 1)]
            avg_distance = sum(distances) / len(distances)

            # 检查距离是否接近一致
            is_pattern = all(abs(d - avg_distance) <= 2 for d in distances)

            if is_pattern:
                return {
                    'type': 'repetition',
                    'token': token,
                    'positions': [p for p in positions],
                    'count': len(positions)
                }

        return None

    def _extract_sentence_boundaries(self, analysis: Dict, context: TranslationContext):
        """提取句子边界"""
        sentences = analysis.get('sentences', [])
        context.sentence_boundaries = [sent['end'] for sent in sentences]

    def _calculate_confidence_scores(self, context: TranslationContext):
        """计算各项分析的置信度"""
        # 句子分割置信度
        if context.sentences:
            # 基于句子结束标记的存在来评估
            valid_endings = sum(1 for sent in context.sentences
                                if sent.get('text', '').rstrip().endswith(('།', '༎', '༏')))
            context.confidence_scores['sentence_segmentation'] = valid_endings / len(context.sentences)

        # 术语识别置信度
        if context.term_mappings:
            # 基于术语在数据库中的存在
            context.confidence_scores['term_identification'] = 0.9  # 高置信度，因为使用了术语库

        # 语法分析置信度
        if context.grammatical_analyses:
            # 基于识别出的语法元素数量
            total_elements = sum(
                len(g.get('case_particles', [])) + len(g.get('syntactic_roles', []))
                for g in context.grammatical_analyses
            )
            context.confidence_scores['grammar_analysis'] = min(total_elements / 10, 1.0) * 0.8

        # 敬语识别置信度
        if any(context.honorific_markers):
            context.confidence_scores['honorific_detection'] = 0.85

        # 结构识别置信度
        if context.has_parallel_structure or context.has_enumeration or context.is_verse:
            context.confidence_scores['structure_detection'] = 0.75

    def merge_contexts(self, contexts: List[TranslationContext]) -> TranslationContext:
        """合并多个上下文（用于处理长文本）"""
        merged = TranslationContext()

        # 合并基础信息
        merged.original_text = ''.join(ctx.original_text for ctx in contexts)

        # 合并列表类型的信息
        for ctx in contexts:
            merged.tokens.extend(ctx.tokens)
            merged.sentences.extend(ctx.sentences)
            merged.term_mappings.update(ctx.term_mappings)
            merged.grammatical_analyses.extend(ctx.grammatical_analyses)
            merged.honorific_markers.extend(ctx.honorific_markers)
            merged.case_particles.extend(ctx.case_particles)
            merged.sentence_boundaries.extend(ctx.sentence_boundaries)

        # 合并结构信息（取OR）
        merged.has_parallel_structure = any(ctx.has_parallel_structure for ctx in contexts)
        merged.has_enumeration = any(ctx.has_enumeration for ctx in contexts)
        merged.is_verse = any(ctx.is_verse for ctx in contexts)

        # 重新计算置信度
        self._calculate_confidence_scores(merged)

        return merged