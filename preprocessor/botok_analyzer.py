"""
Botok分析器 - 负责藏文文本的语言学分析
"""
import re
import logging
from typing import List, Dict, Tuple, Optional, Any
from botok import Config, WordTokenizer
from pathlib import Path
from utils.term_database import TermDatabase

logger = logging.getLogger(__name__)


class BotokAnalyzer:
    def __init__(self, config_path: Optional[Path] = None, term_database=None):
        """初始化Botok分析器"""
        if config_path:
            self.config = Config(base_path=config_path)
        else:
            self.config = Config(dialect_name="custom")

        self.tokenizer = WordTokenizer(config=self.config)

        # 格助词映射
        self.case_particles = {
            'གི་': 'genitive',  # 属格
            'གིས་': 'ergative',  # 作格/施格
            'ལ་': 'dative',  # 与格/向格
            'ནས་': 'ablative',  # 从格/离格
            'ན་': 'locative',  # 位格
            'དང་': 'comitative',  # 共格
            'ལས་': 'comparative',  # 比较格
            'ཀྱི་': 'genitive',  # 属格变体
            'ཀྱིས་': 'ergative',  # 作格变体
            'སུ་': 'terminative',  # 终格
            'ཏུ་': 'terminative',  # 终格变体
            'དུ་': 'terminative',  # 终格变体
            'ར་': 'terminative',  # 终格变体
        }

        # 时态标记
        self.tense_markers = {
            'བྱུང་': 'past',  # 过去
            'གི་': 'present',  # 现在进行
            'རྒྱུ་': 'future',  # 将来
            'ཡིན་': 'present',  # 现在
            'རེད་': 'present',  # 现在
            'སོང་': 'past',  # 过去完成
        }

        # 敬语标记
        self.honorific_markers = {
            'གནང་': True,  # 敬语动词标记
            'མཛད་': True,  # 敬语动词标记
            'ཞུ་': True,  # 谦语标记
            'ལགས་': True,  # 敬语系词
        }

        # 句子结束标记
        self.sentence_endings = ['།', '༎', '༏', '༐', '༑']

        self.term_database = term_database  # 存储实例而非类
        self.known_terms_cache = set()  # 缓存常用术语
        if self.term_database:
            self._load_terms_from_database()

    def _load_terms_from_database(self):
        """从术语数据库加载术语"""
        if not self.term_database:
            return

        try:
            # 加载藏文术语
            self.term_database.cursor.execute("""
                                              SELECT tibetan
                                              FROM terms LIMIT 1000
                                              """)
            terms = [row[0] for row in self.term_database.cursor.fetchall() if row[0]]
            self.known_terms_cache.update(terms)
        except Exception as e:
            import logging
            logging.warning(f"从数据库加载术语失败: {e}")

    def _is_known_term(self, phrase: str) -> bool:
        """检查是否为已知术语"""
        # 首先检查缓存
        if phrase in self.known_terms_cache:
            return True

        # 尝试查询数据库
        if self.term_database:
            try:
                self.term_database.cursor.execute("""
                                                  SELECT id
                                                  FROM terms
                                                  WHERE tibetan = ?
                                                  """, (phrase,))
                result = self.term_database.cursor.fetchone()
                if result:
                    # 添加到缓存
                    self.known_terms_cache.add(phrase)
                    return True
            except Exception:
                pass

        # 回退到硬编码术语列表
        known_terms = [
            'བདེན་པ་བཞི་', 'བྱང་ཆུབ་སེམས་དཔའ་', 'ཕར་ཕྱིན་དྲུག་',
            'སྟོང་པ་ཉིད་', 'རྟེན་འབྲེལ་', 'ལས་དང་འབྲས་བུ་'
        ]
        return phrase in known_terms

    def analyze(self, text: str) -> Dict:
        """
        全面分析藏文文本
        返回包含词法、句法、语义信息的字典
        """
        try:
            # 分词
            tokens = self.tokenizer.tokenize(text)

            # 提取各类信息
            token_info = self._extract_token_info(tokens)
            sentences = self._detect_sentences(text, tokens)
            terms = self._identify_terms(tokens)
            grammar = self._analyze_grammar(tokens)
            honorifics = self._detect_honorifics(tokens)
            structure = self._analyze_structure(text, tokens)

            analysis = {
                'tokens': token_info,
                'sentences': sentences,
                'terms': terms,
                'grammar': grammar,
                'honorifics': honorifics,
                'structure': structure
            }

            return analysis
        except Exception as e:
            logger.error(f"藏文分析失败: {e}")
            import traceback
            logger.error(f"完整错误追踪:\n{traceback.format_exc()}")
            # 返回基本分析结果
            return {
                'tokens': [],
                'sentences': [{'text': text, 'start': 0, 'end': len(text), 'type': 'unknown'}],
                'error': str(e)
            }

    def _identify_terms(self, tokens) -> List[Dict]:
        """
        从分词结果中识别术语并转换为字典格式
        """
        terms = []
        i = 0

        while i < len(tokens):
            # 检查多词术语
            term_info = self._check_multiword_term(tokens, i)
            if term_info:
                terms.append(term_info)
                i += term_info['length']
            else:
                # 检查单词术语
                if self._is_term(tokens[i]):
                    # 将Token对象转换为字典
                    term_dict = {
                        'text': getattr(tokens[i], 'text', ''),
                        'start': getattr(tokens[i], 'start', 0),
                        'end': getattr(tokens[i], 'start', 0) + len(getattr(tokens[i], 'text', '')),
                        'length': len(getattr(tokens[i], 'text', '')),
                        'type': getattr(tokens[i], 'pos', 'general'),
                        'pos': getattr(tokens[i], 'pos', None)
                    }
                    terms.append(term_dict)
                i += 1

        return terms

    def _is_term(self, token) -> bool:
        """
        判断一个词元是否为术语
        """
        if not token:
            return False

        text = getattr(token, 'text', '')
        pos = getattr(token, 'pos', '')

        # 词性为名词、专有名词等的通常可能是术语
        is_noun = pos in ['NOUN', 'PROPN', 'N', 'n.']

        # 长度超过1的词更可能是术语
        is_long_enough = len(text) > 1

        # 不是标点符号
        is_not_punct = pos != 'PUNCT' and not (text in '།།། ་')

        # 检查是否为佛教术语
        is_buddhist_term = self._is_buddhist_term(token)

        return (is_noun and is_long_enough and is_not_punct) or is_buddhist_term

    def _extract_token_info(self, tokens) -> List[Dict]:
        """提取词元信息"""
        token_info = []
        for token in tokens:
            info = {
                'text': token.text,
                'pos': token.pos if hasattr(token, 'pos') else None,
                'lemma': token.lemma if hasattr(token, 'lemma') else token.text,
                'is_word': token.type == 'word' if hasattr(token, 'type') else False,
                'start': token.start if hasattr(token, 'start') else None,
                'end': token.end if hasattr(token, 'end') else None,
            }
            token_info.append(info)
        return token_info

    def _detect_sentences(self, text: str, tokens) -> List[Dict]:
        """检测句子边界"""
        sentences = []
        current_sentence = []
        sentence_start = 0

        for i, token in enumerate(tokens):
            current_sentence.append(token)

            # 检查是否为句子结束
            boundary_confidence = 0.0
            is_boundary = False

            # 判断句子边界
            if token.text in self.sentence_endings:
                # 明确的句子结束标记
                is_boundary = True
                boundary_confidence = 1.0
            elif i < len(tokens) - 1:
                # 使用改进的边界检测方法
                is_boundary, boundary_confidence = self._is_sentence_boundary(
                    token, tokens[i + 1], tokens, i
                )

            if is_boundary:
                # 找到句子边界，保存当前句子
                sentence_end = token.end if hasattr(token, 'end') else len(text)
                sentence_text = text[sentence_start:sentence_end]

                sentences.append({
                    'text': sentence_text,
                    'tokens': current_sentence,
                    'start': sentence_start,
                    'end': sentence_end,
                    'type': self._classify_sentence(current_sentence),
                    'boundary_confidence': boundary_confidence  # 存储边界置信度
                })

                # 重置为新句子
                current_sentence = []
                sentence_start = sentence_end

        # 处理最后一个句子
        if current_sentence:
            sentence_text = text[sentence_start:]
            sentences.append({
                'text': sentence_text,
                'tokens': current_sentence,
                'start': sentence_start,
                'end': len(text),
                'type': self._classify_sentence(current_sentence),
                'boundary_confidence': 1.0  # 文本结尾是确定的边界
            })

        return sentences

    def _is_sentence_boundary(self, current_token, next_token, tokens=None, position=None) -> Tuple[bool, float]:
        """
        判断是否为句子边界，同时返回置信度
        返回: (is_boundary, confidence)
        """
        # 高置信度标志 (0.9-1.0)
        if current_token.text in self.sentence_endings:
            return True, 1.0

        confidence = 0.0
        is_boundary = False

        # 收集证据
        evidence = []

        # 检查上下文 (如果可用)
        context_before = []
        context_after = []
        if tokens and position is not None:
            # 获取前后文 (最多3个词元)
            start_idx = max(0, position - 3)
            end_idx = min(len(tokens), position + 4)  # +4因为next_token已经是position+1
            context_before = tokens[start_idx:position]
            context_after = tokens[position + 2:end_idx] if position + 2 < end_idx else []

        # === 证据1: 句末助词 ===
        if next_token.text in ['འོ་', 'འོ།', 'སོ་', 'སོ།', 'ངོ་', 'ངོ།', 'ཏོ་', 'ཏོ།']:
            evidence.append(("sentence_final_particle", 0.95))

        # === 证据2: 动词结束 ===
        if hasattr(current_token, 'pos') and current_token.pos == 'VERB':
            # 高可靠性动词结束标记
            strong_verb_endings = ['བཞིན་ཡོད་', 'གི་ཡོད་', 'བྱུང་', 'འདུག་']
            if any(current_token.text.endswith(marker) for marker in strong_verb_endings):
                # 检查是否有后续连词
                conjunctions = ['དང་', 'ཡང་', 'འམ་', 'ནི་', 'ཀྱང་', 'ཡིན་ན་ཡང་', 'སྟེ་', 'ཞིང་']
                if next_token.text not in conjunctions:
                    # 检查后文是否开始新句子的迹象
                    if not context_after or (
                            context_after and not any(t.text in conjunctions for t in context_after[:2])):
                        evidence.append(("verb_ending_pattern", 0.8))

        # === 证据3: 问句标记 ===
        if current_token.text in ['འམ་', 'གམ་', 'ཏམ་', 'དམ་']:
            # 检查是否为复合词的一部分
            if next_token.text not in ['ཡང་', 'ཅི་', 'ཇི་', 'སུ་']:
                # 检查前文是否有问句特征
                if context_before and any(t.text in ['ཅི་', 'ཇི་', 'སུ་', 'གང་'] for t in context_before):
                    evidence.append(("question_marker", 0.85))
                else:
                    evidence.append(("question_marker", 0.7))  # 较低置信度

        # === 证据4: 句子结构完整性 ===
        # 检查是否形成了完整句法结构 (主谓结构)
        has_subject = False
        has_predicate = False

        if tokens and position is not None:
            # 简单检查前文是否可能包含主语
            for t in tokens[:position]:
                if hasattr(t, 'pos') and t.pos in ['NOUN', 'PROPN', 'PRON']:
                    has_subject = True
                    break

            # 检查当前词是否可能是谓语
            if hasattr(current_token, 'pos') and current_token.pos == 'VERB':
                has_predicate = True

        if has_subject and has_predicate:
            evidence.append(("complete_structure", 0.6))

        # === 证据5: 新句子开始标志 ===
        sentence_starters = ['དེ་ནས་', 'འོན་ཀྱང་', 'གལ་ཏེ་', 'དེས་ན་', 'དེ་ལྟར་', 'འདི་ལྟར་']
        if next_token.text in sentence_starters:
            # 只有当前文符合句子结尾特征时才考虑
            if (hasattr(current_token, 'pos') and current_token.pos in ['VERB', 'PART']):
                # 检查是否在引号或括号内
                in_quote = False  # 这需要更复杂的追踪来实现
                if not in_quote:
                    evidence.append(("new_sentence_starter", 0.75))

        # === 证据6: 特殊语法结构 ===
        if tokens and position is not None and position < len(tokens) - 2:
            # 检查"ཡིན་པ་རེད་"类型的结构
            if (current_token.text in ['ཡིན་', 'རེད་', 'ཡོད་'] and
                    next_token.text in ['པ་', 'བ་'] and
                    tokens[position + 2].text in ['རེད་', 'ཡིན་']):
                evidence.append(("special_grammar_pattern", 0.85))

        # === 计算总置信度 ===
        if evidence:
            # 取最高置信度的证据
            max_confidence = max([conf for _, conf in evidence])

            # 如果有多个证据，增加置信度
            if len(evidence) > 1:
                # 组合置信度，但不超过0.95
                confidence = min(0.95, max_confidence + 0.05 * (len(evidence) - 1))
            else:
                confidence = max_confidence

            # 是否判定为边界
            is_boundary = confidence >= 0.7  # 阈值可调整

        return is_boundary, confidence

    def _classify_sentence(self, tokens) -> str:
        """分类句子类型"""
        token_texts = [t.text for t in tokens]

        # 判断句 (... ཡིན། / ... རེད།)
        if any(text in ['ཡིན་', 'རེད་'] for text in token_texts[-3:]):
            return 'declarative'

        # 疑问句 (... གས། / ... ངས། / ... དམ།)
        if any(text in ['གས་', 'ངས་', 'དམ་'] for text in token_texts[-3:]):
            return 'interrogative'

        # 祈使句 (包含命令动词)
        if any(hasattr(t, 'pos') and t.pos == 'VERB' and
               any(imp in t.text for imp in ['ཤོག་', 'རོགས་']) for t in tokens):
            return 'imperative'

        # 感叹句
        if any(text in ['ཨ་', 'ཀྱེ་'] for text in token_texts):
            return 'exclamatory'

        return 'declarative'  # 默认为陈述句

    def _check_multiword_term(self, tokens, start_idx) -> Optional[Dict]:
        """检查多词术语"""
        # 最多检查4个词的组合
        max_length = min(4, len(tokens) - start_idx)

        for length in range(max_length, 0, -1):
            phrase = ''.join(tokens[start_idx + j].text for j in range(length))

            # 检查是否为已知术语
            if self._is_known_term(phrase):
                return {
                    'text': phrase,
                    'start': tokens[start_idx].start if hasattr(tokens[start_idx], 'start') else start_idx,
                    'end': tokens[start_idx + length - 1].end if hasattr(tokens[start_idx + length - 1], 'end') else (
                                start_idx + length),
                    'length': length,
                    'type': 'buddhist',
                    'confidence': 0.95
                }

        return None

    def _is_buddhist_term(self, token) -> bool:
        """判断是否为佛教术语"""
        # 简单的规则判断，实际应用中应该使用术语库
        buddhist_keywords = [
            'སངས་རྒྱས་', 'ཆོས་', 'དགེ་འདུན་', 'བྱང་ཆུབ་',
            'སེམས་དཔའ་', 'བདེན་པ་', 'ལམ་', 'འཕགས་པ་'
        ]

        return token.text in buddhist_keywords or \
            (hasattr(token, 'pos') and token.pos == 'PROPN')

    def _analyze_grammar(self, tokens) -> Dict:
        """分析语法信息"""
        grammar = {
            'case_particles': [],
            'tense': None,
            'tense_markers': [],  # 记录所有时态标记
            'syntactic_roles': [],
            'clause_structure': []
        }

        for i, token in enumerate(tokens):
            # 识别格助词
            if token.text in self.case_particles:
                particle_info = {
                    'particle': token.text,
                    'type': self.case_particles[token.text],
                    'position': i,
                    'attached_to': tokens[i - 1].text if i > 0 else None
                }
                grammar['case_particles'].append(particle_info)

            # 识别时态 - 改进版：记录所有时态标记
            if token.text in self.tense_markers:
                tense_info = {
                    'marker': token.text,
                    'tense': self.tense_markers[token.text],
                    'position': i,
                    'context': self._get_context(tokens, i, 2)  # 获取上下文
                }
                grammar['tense_markers'].append(tense_info)

            # 识别句法角色
            role = self._identify_syntactic_role(token, i, tokens)
            if role:
                grammar['syntactic_roles'].append(role)

        # 确定主要时态
        if grammar['tense_markers']:
            grammar['tense'] = self._determine_primary_tense(grammar['tense_markers'])

        return grammar

    def _get_context(self, tokens, position, window_size=2):
        """获取指定位置附近的上下文词元"""
        start = max(0, position - window_size)
        end = min(len(tokens), position + window_size + 1)
        return [tokens[i].text for i in range(start, end)]

    def _determine_primary_tense(self, tense_markers):
        """根据多个时态标记确定主要时态"""
        # 时态优先级：1.完成时 2.过去时 3.现在时 4.将来时
        priority_order = {
            'past': 2,
            'present': 3,
            'future': 4
        }

        # 计算各时态的频率
        tense_counts = {}
        for marker in tense_markers:
            tense = marker['tense']
            tense_counts[tense] = tense_counts.get(tense, 0) + 1

        # 首先尝试基于频率判断
        max_count = 0
        most_frequent_tenses = []
        for tense, count in tense_counts.items():
            if count > max_count:
                max_count = count
                most_frequent_tenses = [tense]
            elif count == max_count:
                most_frequent_tenses.append(tense)

        # 如果有明显的高频时态，使用它
        if len(most_frequent_tenses) == 1:
            return most_frequent_tenses[0]

        # 如果多个时态频率相同，使用优先级高的
        if most_frequent_tenses:
            return min(most_frequent_tenses, key=lambda x: priority_order.get(x, 5))

        # 默认返回现在时
        return 'present'

    def _identify_syntactic_role(self, token, position, tokens) -> Optional[Dict]:
        """识别句法角色"""
        # 施事者（通常后跟作格助词）
        if position < len(tokens) - 1 and tokens[position + 1].text in ['གིས་', 'ཀྱིས་', 'གྱིས་', 'ཡིས་']:
            return {
                'text': token.text,
                'role': 'agent',
                'position': position,
                'confidence': 0.9
            }

        # 受事者（通常在动词前，无格助词或有与格助词）
        if position < len(tokens) - 1:
            # 检查后续是否有动词
            for j in range(position + 1, min(position + 4, len(tokens))):
                if hasattr(tokens[j], 'pos') and tokens[j].pos == 'VERB':
                    # 检查是否有其他格助词
                    has_case_particle = any(
                        tokens[k].text in self.case_particles
                        for k in range(position + 1, j)
                    )
                    if not has_case_particle:
                        return {
                            'text': token.text,
                            'role': 'patient',
                            'position': position,
                            'confidence': 0.7
                        }

        return None

    def _detect_honorifics(self, tokens) -> List[Dict]:
        """检测敬语使用"""
        honorifics = []

        for i, token in enumerate(tokens):
            if token.text in self.honorific_markers:
                context_start = max(0, i - 3)
                context_end = min(len(tokens), i + 3)
                context = [tokens[j].text for j in range(context_start, context_end)]

                honorifics.append({
                    'marker': token.text,
                    'position': i,
                    'type': self._classify_honorific(token.text),
                    'context': context,
                    'applies_to': self._find_honorific_target(tokens, i)
                })

        return honorifics

    def _classify_honorific(self, marker: str) -> str:
        """分类敬语类型"""
        if marker in ['གནང་', 'མཛད་']:
            return 'verb_honorific'
        elif marker == 'ཞུ་':
            return 'humble_form'
        elif marker == 'ལགས་':
            return 'copula_honorific'
        return 'general_honorific'

    def _find_honorific_target(self, tokens, marker_position) -> Optional[str]:
        """找出敬语所指对象"""
        # 通常敬语动词的主语是敬语对象
        start = marker_position - 1
        end = max(0, marker_position - 5) - 1  # 减1确保正确的范围
        for i in range(start, end, -1):
            if hasattr(tokens[i], 'pos') and tokens[i].pos in ['NOUN', 'PROPN']:
                # 检查是否为人名或称谓
                if self._is_person_reference(tokens[i]):
                    return tokens[i].text
        return None

    def _is_person_reference(self, token) -> bool:
        """判断是否为人称指代"""
        person_markers = ['པ་', 'མ་', 'པོ་', 'མོ་']
        titles = ['རིན་པོ་ཆེ་', 'བླ་མ་', 'སློབ་དཔོན་']

        return (token.text.endswith(tuple(person_markers)) or
                token.text in titles or
                (hasattr(token, 'pos') and token.pos == 'PROPN'))

    def _analyze_structure(self, text: str, tokens) -> Dict:
        """分析文本结构"""
        structure = {
            'has_parallel': False,
            'parallel_patterns': [],
            'has_enumeration': False,
            'enumeration_items': [],
            'is_verse': False,
            'verse_info': None
        }

        # 检测平行结构
        parallel_patterns = self._detect_parallel_structure(tokens)
        if parallel_patterns:
            structure['has_parallel'] = True
            structure['parallel_patterns'] = parallel_patterns

        # 检测列举结构
        enum_items = self._detect_enumeration(tokens)
        if enum_items:
            structure['has_enumeration'] = True
            structure['enumeration_items'] = enum_items

        # 检测诗偈格式
        if self._is_verse_format(text):
            structure['is_verse'] = True
            structure['verse_info'] = self._analyze_verse_structure(text)

        return structure

    def _detect_parallel_structure(self, tokens) -> List[Dict]:
        """检测平行结构"""
        patterns = []

        # 检测重复的否定词
        neg_positions = []
        for i, token in enumerate(tokens):
            if token.text in ['མི་', 'མ་', 'མེད་']:
                neg_positions.append(i)

        # 如果有3个或以上的否定词，可能是平行结构
        if len(neg_positions) >= 3:
            # 检查间距是否相似
            distances = [neg_positions[i + 1] - neg_positions[i]
                         for i in range(len(neg_positions) - 1)]
            if all(abs(d - distances[0]) <= 2 for d in distances):
                patterns.append({
                    'type': 'negation',
                    'positions': neg_positions,
                    'pattern': 'repeated_negation'
                })

        return patterns

    def _detect_enumeration(self, tokens) -> List[Dict]:
        """检测列举结构 - 改进版"""
        items = []

        # 1. 基于连词'དང་'的列举
        conjunction_items = self._detect_conjunction_enumeration(tokens)
        if conjunction_items:
            items.extend(conjunction_items)

        # 2. 基于数字的列举
        number_items = self._detect_numbered_enumeration(tokens)
        if number_items:
            items.extend(number_items)

        # 3. 基于顺序词的列举
        sequence_items = self._detect_sequence_enumeration(tokens)
        if sequence_items:
            items.extend(sequence_items)

        # 按开始位置排序
        items.sort(key=lambda x: x.get('start', 0))

        return items if len(items) >= 2 else []

    def _detect_conjunction_enumeration(self, tokens) -> List[Dict]:
        """基于连词'དང་'检测列举"""
        items = []
        current_item = []
        start_idx = None

        for i, token in enumerate(tokens):
            if token.text == 'དང་':
                if current_item:
                    items.append({
                        'text': ''.join(t.text for t in current_item),
                        'start': start_idx,
                        'end': i,
                        'type': 'conjunction'
                    })
                    current_item = []
                    start_idx = None
            else:
                if not current_item:
                    start_idx = i
                current_item.append(token)

        # 处理最后一项（即使没有'དང་'连接符）
        if current_item and items:  # 确保至少有一项已添加，说明这是一个列举
            items.append({
                'text': ''.join(t.text for t in current_item),
                'start': start_idx,
                'end': len(tokens),
                'type': 'conjunction'
            })

        return items

    def _detect_numbered_enumeration(self, tokens) -> List[Dict]:
        """检测基于数字的列举"""
        items = []
        tibetan_digits = ['༠', '༡', '༢', '༣', '༤', '༥', '༦', '༧', '༨', '༩']

        i = 0
        while i < len(tokens):
            # 检查是否为数字开头
            if (tokens[i].text in tibetan_digits or
                    tokens[i].text.isdigit() or
                    tokens[i].text in ['དང་པོ་', 'གཉིས་པ་', 'གསུམ་པ་']):
                # 找到项目的结束位置
                start_idx = i
                item_tokens = []

                # 跳过数字/序号
                i += 1

                # 收集该项目的内容直到下一个数字或结束
                while i < len(tokens) and not (tokens[i].text in tibetan_digits or
                                               tokens[i].text.isdigit() or
                                               tokens[i].text in ['དང་པོ་', 'གཉིས་པ་', 'གསུམ་པ་']):
                    item_tokens.append(tokens[i])
                    i += 1

                if item_tokens:
                    items.append({
                        'text': ''.join(t.text for t in item_tokens),
                        'start': start_idx,
                        'end': i,
                        'type': 'numbered'
                    })
            else:
                i += 1

        return items

    def _detect_sequence_enumeration(self, tokens) -> List[Dict]:
        """检测基于顺序词的列举"""
        items = []
        sequence_markers = [
            'ཐོག་མར་', 'དང་པོ་', 'གཉིས་པ་', 'གསུམ་པ་', 'བཞི་པ་', 'ལྔ་པ་',
            'དེ་ནས་', 'དེ་རྗེས་', 'མཐའ་མར་', 'མཇུག་ཏུ་'
        ]

        i = 0
        while i < len(tokens):
            # 检查是否为顺序标记
            if any(tokens[i].text.startswith(marker) for marker in sequence_markers):
                # 找到项目的结束位置
                start_idx = i
                item_tokens = [tokens[i]]
                i += 1

                # 收集该项目的内容直到下一个顺序标记或结束
                while i < len(tokens) and not any(tokens[i].text.startswith(marker) for marker in sequence_markers):
                    item_tokens.append(tokens[i])
                    i += 1

                items.append({
                    'text': ''.join(t.text for t in item_tokens),
                    'start': start_idx,
                    'end': i,
                    'type': 'sequence'
                })
            else:
                i += 1

        return items

    def _is_verse_format(self, text: str) -> bool:
        """判断是否为诗偈格式 - 改进版"""
        # 分行处理，移除空行
        lines = [line.strip() for line in text.split('།') if line.strip()]
        if len(lines) < 4:  # 至少需要4行才可能是诗偈
            return False

        # 1. 检查行长度模式
        line_lengths = [len(line) for line in lines]
        avg_length = sum(line_lengths) / len(line_lengths)
        variance = sum((l - avg_length) ** 2 for l in line_lengths) / len(line_lengths)

        # 2. 检查音节模式
        syllable_counts = [self._count_syllables(line) for line in lines]
        avg_syllables = sum(syllable_counts) / len(syllable_counts)
        syllable_variance = sum((s - avg_syllables) ** 2 for s in syllable_counts) / len(syllable_counts)

        # 3. 检查韵律模式 - 寻找重复的韵脚
        line_endings = [self._get_line_ending(line) for line in lines]
        has_rhyme_pattern = self._has_rhyme_pattern(line_endings)

        # 4. 检查行结构的规律性
        has_regular_structure = self._check_regular_structure(lines)

        # 基于多个因素判断
        # 条件1：行长度接近（低方差）
        length_regularity = variance < avg_length * 0.3

        # 条件2：音节数接近或呈现规律模式
        syllable_regularity = syllable_variance < avg_syllables * 0.2

        # 决策逻辑：满足主要条件之一，同时考虑辅助条件
        is_verse = (length_regularity and syllable_regularity) or \
                   (has_rhyme_pattern and (length_regularity or syllable_regularity)) or \
                   (has_regular_structure and (length_regularity or has_rhyme_pattern))

        return is_verse

    def _count_syllables(self, line: str) -> int:
        """计算藏文行的音节数"""
        try:
            # 使用Botok的分词功能，它会分割音节
            tokens = self.tokenizer.tokenize(line)
            # 计算所有词元的音节数
            syllable_count = 0
            for token in tokens:
                if hasattr(token, 'syls') and token.syls:
                    syllable_count += len(token.syls)
                elif token.text.strip():  # 如果没有音节信息但有文本
                    # 回退到简单计数方法
                    syllable_count += self._simple_syllable_count(token.text)
            return syllable_count
        except Exception:
            # 如果Botok分析失败，使用简单方法
            return self._simple_syllable_count(line)

    def _simple_syllable_count(self, text: str) -> int:
        """简单的音节计数方法（作为备选）"""
        # 藏文中，音节通常由辅音加元音标记组成
        # 使用"་"（音节分隔符）来估计音节数
        return len(text.split('་')) - 1 if '་' in text else 1

    def _get_line_ending(self, line: str) -> str:
        """获取行末的韵脚（通常是最后一个音节）"""
        # 简化处理：取最后一个词或最后几个字符
        words = line.split()
        if not words:
            return ""

        last_word = words[-1]
        # 取最后一个词的最后2-3个字符作为韵脚特征
        return last_word[-3:] if len(last_word) >= 3 else last_word

    def _has_rhyme_pattern(self, line_endings: List[str]) -> bool:
        """检查是否存在韵律模式"""
        if len(line_endings) < 4:
            return False

        # 检查常见的韵律模式：AABB, ABAB, ABCB等
        patterns = []

        # 尝试AABB模式 (每两行同韵)
        aabb_match = True
        for i in range(0, len(line_endings) - 2, 2):
            if line_endings[i] != line_endings[i + 1]:
                aabb_match = False
                break
        patterns.append(aabb_match)

        # 尝试ABAB模式 (交替韵脚)
        abab_match = True
        if len(line_endings) >= 4:
            for i in range(0, len(line_endings) - 2, 2):
                if (line_endings[i] != line_endings[i + 2]) or (
                        i + 1 < len(line_endings) and i + 3 < len(line_endings) and line_endings[i + 1] != line_endings[
                    i + 3]):
                    abab_match = False
                    break
        patterns.append(abab_match)

        # 尝试ABCB模式 (2,4行韵脚相同)
        abcb_match = True
        if len(line_endings) >= 4:
            for i in range(1, len(line_endings) - 2, 2):
                if line_endings[i] != line_endings[i + 2]:
                    abcb_match = False
                    break
        patterns.append(abcb_match)

        return any(patterns)

    def _check_regular_structure(self, lines: List[str]) -> bool:
        """检查行结构的规律性"""
        # 检查行首是否有重复模式
        if len(lines) >= 4:
            # 检查行首重复词
            first_words = [line.split()[0] if line.split() else "" for line in lines]
            has_repeating_first_word = len(set(first_words)) < len(lines) * 0.7

            # 检查行末重复模式
            last_words = [line.split()[-1] if line.split() else "" for line in lines]
            has_repeating_last_word = len(set(last_words)) < len(lines) * 0.7

            return has_repeating_first_word or has_repeating_last_word

        return False

    def _analyze_verse_structure(self, text: str) -> Dict:
        """分析诗偈结构"""
        lines = [line.strip() for line in text.split('།') if line.strip()]

        return {
            'line_count': len(lines),
            'lines': lines,
            'average_length': sum(len(line) for line in lines) / len(lines) if lines else 0,
            'pattern': self._detect_verse_pattern(lines)
        }

    def _detect_verse_pattern(self, lines: List[str]) -> str:
        """检测诗偈韵律模式"""
        if len(lines) == 4:
            return 'quatrain'  # 四句偈
        elif len(lines) == 8:
            return 'octave'  # 八句偈
        else:
            return 'free_verse'  # 自由诗

    def get_grammatical_info_for_translation(self, text: str) -> Dict:
        """
        提取用于翻译的语法信息

        Args:
            text: 原始藏文文本

        Returns:
            适合传递给翻译器和后处理器的格式化信息
        """
        # 获取完整分析
        analysis = self.analyze(text)

        # 创建专为翻译设计的结构化信息
        translation_info = {
            'original_text': text,
            'sentences': [],
            'terms': analysis.get('terms', []),
            'structure_type': 'prose',  # 默认为散文
            'global_grammar': {
                'tense_tendency': self._detect_global_tense(analysis),
                'voice_tendency': self._detect_global_voice(analysis),
                'formality_level': self._detect_formality(analysis),
                'negation_pattern': self._detect_negation_pattern(analysis)
            }
        }

        # 确定文本结构类型
        structure = analysis.get('structure', {})
        if structure.get('is_verse', False):
            translation_info['structure_type'] = 'verse'
            translation_info['verse_info'] = structure.get('verse_structure', {})
        elif structure.get('has_enumeration', False):
            translation_info['structure_type'] = 'enumeration'
            translation_info['enumeration_info'] = structure.get('enumeration_structure', {})
        elif structure.get('has_parallel_structure', False):
            translation_info['structure_type'] = 'parallel'
            translation_info['parallel_info'] = structure.get('parallel_structure', {})

        # 处理每个句子的信息
        for sentence in analysis.get('sentences', []):
            # 基本句子信息
            sentence_info = {
                'text': sentence.get('text', ''),
                'start': sentence.get('start', 0),
                'end': sentence.get('end', 0),
                'type': sentence.get('type', 'declarative'),
                'grammatical_analysis': {
                    'case_particles': [],
                    'tense': None,
                    'voice': None,
                    'syntactic_roles': []
                }
            }

            # 提取句子语法信息
            self._extract_sentence_grammar(sentence, analysis, sentence_info)

            # 添加到结果中
            translation_info['sentences'].append(sentence_info)

        return translation_info

    def _detect_global_tense(self, analysis: Dict) -> str:
        """检测文本的整体时态倾向"""
        tense_markers = analysis.get('grammar', {}).get('tense_markers', [])

        if not tense_markers:
            return 'present'  # 默认为现在时

        # 统计各时态标记
        tense_count = {'past': 0, 'present': 0, 'future': 0, 'present_progressive': 0}

        for marker in tense_markers:
            tense = marker.get('tense')
            if tense in tense_count:
                tense_count[tense] += 1

        # 返回出现最多的时态
        return max(tense_count.items(), key=lambda x: x[1])[0]

    def _detect_global_voice(self, analysis: Dict) -> str:
        """检测文本的整体语态倾向"""
        # 分析主动/被动语态标记
        voice_markers = {
            'active': 0,
            'passive': 0
        }

        grammar = analysis.get('grammar', {})

        # 检查格助词中的被动标记
        for particle in grammar.get('case_particles', []):
            if particle.get('type') == 'instrumental' and particle.get('usage') == 'agent':
                voice_markers['passive'] += 1
            elif particle.get('type') == 'ergative':
                voice_markers['active'] += 1

        # 返回主要语态
        return 'passive' if voice_markers['passive'] > voice_markers['active'] else 'active'

    def _detect_formality(self, analysis: Dict) -> str:
        """检测文本的正式程度"""
        # 检查敬语标记
        honorific_count = 0

        for token in analysis.get('tokens', []):
            if token.get('is_honorific', False):
                honorific_count += 1

        total_tokens = len(analysis.get('tokens', []))

        if total_tokens == 0:
            return 'neutral'

        # 计算敬语密度
        honorific_ratio = honorific_count / total_tokens

        if honorific_ratio > 0.2:
            return 'formal'
        elif honorific_ratio > 0.05:
            return 'semi-formal'
        else:
            return 'informal'

    def _detect_negation_pattern(self, analysis: Dict) -> str:
        """检测文本的否定模式"""
        # 计算否定词频率
        negation_count = 0

        for token in analysis.get('tokens', []):
            if token.get('is_negation', False):
                negation_count += 1

        total_tokens = len(analysis.get('tokens', []))

        if total_tokens == 0:
            return 'standard'

        # 计算否定密度
        negation_ratio = negation_count / total_tokens

        if negation_ratio > 0.15:
            return 'heavy_negation'
        elif negation_ratio > 0.05:
            return 'moderate_negation'
        else:
            return 'standard'

    def _extract_sentence_grammar(self, sentence, full_analysis, sentence_info):
        """提取单个句子的语法信息"""
        # 获取句子范围 - 添加安全访问
        start_pos = self._safe_get(sentence, 'start', 0)
        end_pos = self._safe_get(sentence, 'end', 0)

        # 从全局语法分析中提取该句的信息 - 添加安全访问
        grammar = self._safe_get(full_analysis, 'grammar', {})

        # 1. 提取格助词信息
        for particle in self._safe_get(grammar, 'case_particles', []):
            try:
                token_pos = self._safe_get(particle, 'position', 0)
                # 计算实际字符位置
                char_pos = self._get_token_position(token_pos, self._safe_get(full_analysis, 'tokens', []))

                # 检查是否在当前句子范围内
                if start_pos <= char_pos < end_pos:
                    # 复制并添加相对位置信息 - 添加安全复制
                    particle_info = self._safe_copy(particle)
                    particle_info['relative_position'] = char_pos - start_pos
                    sentence_info['grammatical_analysis']['case_particles'].append(particle_info)
            except Exception as e:
                logger.warning(f"处理格助词时出错: {e}")

        # 2. 提取时态信息
        for marker in self._safe_get(grammar, 'tense_markers', []):
            try:
                token_pos = self._safe_get(marker, 'position', 0)
                char_pos = self._get_token_position(token_pos, self._safe_get(full_analysis, 'tokens', []))

                if start_pos <= char_pos < end_pos:
                    # 设置句子主要时态
                    sentence_info['grammatical_analysis']['tense'] = self._safe_get(marker, 'tense')
                    break
            except Exception as e:
                logger.warning(f"处理时态标记时出错: {e}")

        # 3. 提取语态信息
        try:
            sentence_info['grammatical_analysis']['voice'] = self._detect_sentence_voice(
                sentence, full_analysis, start_pos, end_pos
            )
        except Exception as e:
            logger.warning(f"检测语态时出错: {e}")
            sentence_info['grammatical_analysis']['voice'] = 'active'  # 默认值

        # 4. 提取句法角色
        for role in self._safe_get(grammar, 'syntactic_roles', []):
            try:
                token_pos = self._safe_get(role, 'position', 0)
                char_pos = self._get_token_position(token_pos, self._safe_get(full_analysis, 'tokens', []))

                if start_pos <= char_pos < end_pos:
                    # 复制并添加相对位置
                    role_info = self._safe_copy(role)
                    role_info['relative_position'] = char_pos - start_pos
                    sentence_info['grammatical_analysis']['syntactic_roles'].append(role_info)
            except Exception as e:
                logger.warning(f"处理句法角色时出错: {e}")

    def _detect_sentence_voice(self, sentence, full_analysis, start_pos, end_pos):
        """检测句子的语态"""
        grammar = self._safe_get(full_analysis, 'grammar', {})

        # 检查被动标记
        for particle in self._safe_get(grammar, 'case_particles', []):
            token_pos = self._safe_get(particle, 'position', 0)
            char_pos = self._get_token_position(token_pos, self._safe_get(full_analysis, 'tokens', []))

            if start_pos <= char_pos < end_pos:
                if self._safe_get(particle, 'type') == 'instrumental' and self._safe_get(particle, 'usage') == 'agent':
                    return 'passive'

        # 默认为主动语态
        return 'active'

    def _get_token_position(self, token_index, tokens):
        """获取词元在原文中的实际字符位置"""
        if token_index < len(tokens):
            return self._safe_get(tokens[token_index], 'start', 0)
        return 0

    def _safe_get(self, obj: Any, key: str, default=None) -> Any:
        """安全获取属性或键值，支持对象和字典"""
        if obj is None:
            return default

        # 尝试字典访问
        if hasattr(obj, 'get'):
            return obj.get(key, default)

        # 尝试属性访问
        if hasattr(obj, key):
            return getattr(obj, key)

        # 尝试索引访问
        try:
            return obj[key]
        except (TypeError, KeyError, IndexError):
            return default

    def _safe_copy(self, obj):
        """安全复制对象，支持字典和对象"""
        if hasattr(obj, 'copy'):
            return obj.copy()
        elif hasattr(obj, '__dict__'):
            # 如果是对象，复制其__dict__
            return obj.__dict__.copy()
        else:
            # 简单的字典转换
            try:
                return dict(obj)
            except (TypeError, ValueError):
                # 最后的退路：创建新字典
                return {}