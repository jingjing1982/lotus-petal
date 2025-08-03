"""
概念关系提取器 - 从对译文本中提取佛教概念关系
"""
import re
import logging
from typing import List, Dict, Tuple, Set, Optional
from collections import defaultdict, Counter
import jieba
import jieba.posseg as pseg
from utils.term_database import TermDatabase

logger = logging.getLogger(__name__)


class ConceptRelationExtractor:
    def __init__(self, term_database: TermDatabase):
        self.db = term_database
        self.relation_patterns = self._initialize_relation_patterns()

        # 从数据库加载术语
        self._load_terms()

        # 词频统计和共现矩阵
        self.concept_counter = Counter()
        self.co_occurrence = defaultdict(Counter)

    def _initialize_relation_patterns(self):
        """初始化关系提取模式"""
        return {
            'opposite': [
                r'([\u4e00-\u9fff]{1,6})与([\u4e00-\u9fff]{1,6})相对',
                r'([\u4e00-\u9fff]{1,6})和([\u4e00-\u9fff]{1,6})相反',
                r'([\u4e00-\u9fff]{1,6})对立于([\u4e00-\u9fff]{1,6})',
                r'非([\u4e00-\u9fff]{1,6})即([\u4e00-\u9fff]{1,6})',
            ],
            'includes': [
                r'([\u4e00-\u9fff]{1,6})包括([\u4e00-\u9fff]{1,6})',
                r'([\u4e00-\u9fff]{1,6})包含([\u4e00-\u9fff]{1,6})',
                r'([\u4e00-\u9fff]{1,6})中的([\u4e00-\u9fff]{1,6})',
            ],
            'related': [
                r'([\u4e00-\u9fff]{1,6})与([\u4e00-\u9fff]{1,6})相关',
                r'([\u4e00-\u9fff]{1,6})和([\u4e00-\u9fff]{1,6})有关',
            ],
            'stage': [
                r'([\u4e00-\u9fff]{1,6})之后是([\u4e00-\u9fff]{1,6})',
                r'从([\u4e00-\u9fff]{1,6})到([\u4e00-\u9fff]{1,6})',
                r'([\u4e00-\u9fff]{1,6})进入([\u4e00-\u9fff]{1,6})',
            ],
            'method_goal': [
                r'通过([\u4e00-\u9fff]{1,6})实现([\u4e00-\u9fff]{1,6})',
                r'([\u4e00-\u9fff]{1,6})导向([\u4e00-\u9fff]{1,6})',
                r'([\u4e00-\u9fff]{1,6})的目标是([\u4e00-\u9fff]{1,6})',
            ]
        }

    def _load_terms(self):
        """从数据库加载术语列表"""
        self.terms = set()

        try:
            # 获取所有中文术语
            self.db.cursor.execute("SELECT chinese FROM translations")
            terms = self.db.cursor.fetchall()
            for term in terms:
                if term[0] and len(term[0]) > 1:
                    self.terms.add(term[0])
                    jieba.add_word(term[0])  # 添加到jieba词典

            logger.info(f"已加载 {len(self.terms)} 个术语")
        except Exception as e:
            logger.error(f"加载术语失败: {e}")

    def process_parallel_texts(self, tibetan_texts: List[str], chinese_texts: List[str]):
        """处理藏汉平行语料库"""
        if len(tibetan_texts) != len(chinese_texts):
            logger.error("藏文和中文文本数量不匹配")
            return

        for i, (tibetan, chinese) in enumerate(zip(tibetan_texts, chinese_texts)):
            if i % 100 == 0:
                logger.info(f"正在处理第 {i} 对文本")

            # 提取关系
            relations = self.extract_relations_from_text(chinese, tibetan)

            # 保存关系
            for rel in relations:
                self.db.add_concept_relation(
                    relation_type=rel['type'],
                    source=rel['source'],
                    target=rel['target'],
                    bidirectional=rel.get('bidirectional', False),
                    confidence=rel.get('confidence', 0.8),
                    source_type='extracted',
                    reference_text=f"藏文: {tibetan[:50]}... 中文: {chinese[:50]}..."
                )

            # 更新共现统计
            self._update_co_occurrence(chinese)

    def extract_relations_from_text(self, chinese_text: str, tibetan_text: str = None) -> List[Dict]:
        """从文本中提取概念关系"""
        results = []

        # 1. 使用模式匹配提取显式关系
        for relation_type, patterns in self.relation_patterns.items():
            for pattern in patterns:
                matches = re.finditer(pattern, chinese_text)
                for match in matches:
                    source = match.group(1)
                    target = match.group(2)

                    # 验证是否为已知术语
                    if source in self.terms and target in self.terms:
                        results.append({
                            'type': relation_type,
                            'source': source,
                            'target': target,
                            'bidirectional': relation_type in ['opposite', 'related'],
                            'confidence': 0.9,
                            'pattern': pattern
                        })

        # 2. 根据特定词汇和上下文推断隐含关系
        segments = list(pseg.cut(chinese_text))
        for i, (word, pos) in enumerate(segments):
            if word in self.terms:
                # 寻找附近的其他术语
                nearby_terms = self._find_nearby_terms(segments, i, window=5)
                for nearby_term, distance, context in nearby_terms:
                    if nearby_term == word:
                        continue

                    # 根据上下文分析推断关系类型
                    relation = self._infer_relation_from_context(word, nearby_term, context)
                    if relation:
                        results.append(relation)

        return results

    def _find_nearby_terms(self, segments, position, window=5):
        """查找附近的术语"""
        results = []
        start = max(0, position - window)
        end = min(len(segments), position + window + 1)

        context_words = []
        for i in range(start, end):
            word, pos = segments[i]
            context_words.append(word)

            if i != position and word in self.terms:
                context = ''.join(context_words)
                distance = abs(i - position)
                results.append((word, distance, context))

        return results

    def _infer_relation_from_context(self, term1: str, term2: str, context: str) -> Optional[Dict]:
        """
        从上下文推断两个术语之间的关系

        Args:
            term1: 第一个术语
            term2: 第二个术语
            context: 包含两个术语的上下文文本

        Returns:
            推断出的关系或None
        """
        # 1. 应用规则匹配策略
        relation = self._apply_relation_rules(term1, term2, context)
        if relation:
            return relation

        # 2. 应用语义分析策略
        relation = self._apply_semantic_analysis(term1, term2, context)
        if relation:
            return relation

        # 3. 应用知识库查询策略
        relation = self._query_knowledge_base(term1, term2)
        if relation:
            return relation

        # 4. 如果以上方法都未找到关系，返回None
        return None

    def _apply_relation_rules(self, term1: str, term2: str, context: str) -> Optional[Dict]:
        """应用基于规则的关系推断"""
        # 定义关系指示词和对应的关系类型
        relation_indicators = {
            'opposite': ['对立', '相反', '相对', '非', '不是', '并非', '有别于'],
            'includes': ['包含', '包括', '涵盖', '属于', '归类于', '是...的一种'],
            'stage': ['阶段', '次第', '等级', '步骤', '进程', '发展为'],
            'method_goal': ['方法', '达到', '获得', '证得', '成就', '修习', '目标'],
            'cause_effect': ['导致', '引起', '产生', '造成', '使得', '令', '致使'],
            'property': ['特性', '性质', '特点', '特征', '表现为', '以...为特征'],
            'related': ['相关', '关联', '连接', '联系', '涉及']
        }

        # 检查各类关系指示词
        for relation_type, indicators in relation_indicators.items():
            for indicator in indicators:
                if indicator in context:
                    # 分析指示词前后的文本，确定关系方向
                    indicator_pos = context.find(indicator)
                    before_text = context[:indicator_pos]
                    after_text = context[indicator_pos + len(indicator):]

                    # 确定关系方向
                    bidirectional = relation_type in ['opposite', 'related']

                    # 确定源和目标
                    if term1 in before_text and term2 in after_text:
                        source, target = term1, term2
                    elif term2 in before_text and term1 in after_text:
                        source, target = term2, term1
                    else:
                        # 如果无法确定方向，使用默认顺序
                        source, target = term1, term2

                    # 构建关系
                    return {
                        'type': relation_type,
                        'source': source,
                        'target': target,
                        'bidirectional': bidirectional,
                        'confidence': 0.7,
                        'context': context,
                        'indicator': indicator
                    }

        # 分析句法结构中的特殊模式
        if '是' in context:
            # "A是B"模式：可能表示包含关系
            is_pos = context.find('是')
            before_text = context[:is_pos]
            after_text = context[is_pos + 1:]

            if term1 in before_text and term2 in after_text:
                # A是B: B包含A
                return {
                    'type': 'includes',
                    'source': term2,
                    'target': term1,
                    'bidirectional': False,
                    'confidence': 0.6,
                    'context': context,
                    'indicator': '是'
                }
            elif term2 in before_text and term1 in after_text:
                # B是A: A包含B
                return {
                    'type': 'includes',
                    'source': term1,
                    'target': term2,
                    'bidirectional': False,
                    'confidence': 0.6,
                    'context': context,
                    'indicator': '是'
                }

        return None

    def _apply_semantic_analysis(self, term1: str, term2: str, context: str) -> Optional[Dict]:
        """应用语义分析策略"""
        # 这里可以集成更复杂的NLP模型
        # 简化实现：分析术语的语义相似度和位置关系

        # 检查术语是否为反义词
        antonym_pairs = [
            ('善', '恶'), ('有', '无'), ('常', '无常'),
            ('生', '灭'), ('净', '染'), ('真', '妄'),
            ('明', '暗'), ('乐', '苦'), ('动', '静')
        ]

        for a, b in antonym_pairs:
            if (a in term1 and b in term2) or (b in term1 and a in term2):
                return {
                    'type': 'opposite',
                    'source': term1,
                    'target': term2,
                    'bidirectional': True,
                    'confidence': 0.65,
                    'context': context,
                    'indicator': 'semantic_opposite'
                }

        # 检查术语间的部分-整体关系
        if len(term1) > len(term2) and term2 in term1:
            # term1包含term2，可能是整体-部分关系
            return {
                'type': 'includes',
                'source': term1,
                'target': term2,
                'bidirectional': False,
                'confidence': 0.5,
                'context': context,
                'indicator': 'substring_relation'
            }
        elif len(term2) > len(term1) and term1 in term2:
            # term2包含term1，可能是整体-部分关系
            return {
                'type': 'includes',
                'source': term2,
                'target': term1,
                'bidirectional': False,
                'confidence': 0.5,
                'context': context,
                'indicator': 'substring_relation'
            }

        # 检查佛教修行阶段术语
        practice_stages = ['闻', '思', '修', '证']
        stage1_idx = -1
        stage2_idx = -1

        for idx, stage in enumerate(practice_stages):
            if stage in term1:
                stage1_idx = idx
            if stage in term2:
                stage2_idx = idx

        if stage1_idx != -1 and stage2_idx != -1 and stage1_idx != stage2_idx:
            # 如果两个术语包含不同的修行阶段，表示阶段关系
            if stage1_idx < stage2_idx:
                return {
                    'type': 'stage',
                    'source': term1,
                    'target': term2,
                    'bidirectional': False,
                    'confidence': 0.6,
                    'context': context,
                    'indicator': 'practice_stage'
                }
            else:
                return {
                    'type': 'stage',
                    'source': term2,
                    'target': term1,
                    'bidirectional': False,
                    'confidence': 0.6,
                    'context': context,
                    'indicator': 'practice_stage'
                }

        return None

    def _query_knowledge_base(self, term1: str, term2: str) -> Optional[Dict]:
        """查询知识库获取术语关系"""
        try:
            # 尝试从术语数据库获取已知关系
            relation = self.term_database.get_relation(term1, term2)
            if relation:
                return relation

            # 如果没有直接关系，尝试推断关系
            # 例如，如果A包含B，B包含C，则A可能包含C
            indirect_relations = self.term_database.get_indirect_relations(term1, term2)
            if indirect_relations:
                # 选择置信度最高的间接关系
                return max(indirect_relations, key=lambda r: r.get('confidence', 0))

        except Exception as e:
            logger.error(f"查询知识库失败: {e}")

        return None

    def _update_co_occurrence(self, text):
        """更新术语共现统计"""
        found_terms = []

        # 分词并提取术语
        segments = jieba.cut(text)
        for word in segments:
            if word in self.terms:
                self.concept_counter[word] += 1
                found_terms.append(word)

        # 更新共现矩阵
        for i, term1 in enumerate(found_terms):
            for term2 in found_terms[i + 1:]:
                self.co_occurrence[term1][term2] += 1
                self.co_occurrence[term2][term1] += 1

    def analyze_co_occurrence(self, min_count=5, threshold=0.5):
        """分析共现数据，提取可能的关系"""
        results = []

        # 计算每对术语的共现强度
        for term1, counters in self.co_occurrence.items():
            for term2, count in counters.items():
                if count < min_count:
                    continue

                # 计算PMI (点互信息)
                p_x = self.concept_counter[term1] / sum(self.concept_counter.values())
                p_y = self.concept_counter[term2] / sum(self.concept_counter.values())
                p_xy = count / sum([sum(counter.values()) for counter in self.co_occurrence.values()])

                if p_x * p_y > 0:
                    pmi = p_xy / (p_x * p_y)

                    if pmi > threshold:
                        results.append({
                            'type': 'related',  # 默认为相关关系
                            'source': term1,
                            'target': term2,
                            'bidirectional': True,
                            'confidence': min(pmi / 10, 0.9),  # 归一化
                            'count': count,
                            'pmi': pmi
                        })

        # 按PMI排序
        results.sort(key=lambda x: x['pmi'], reverse=True)

        # 保存到数据库
        for rel in results[:1000]:  # 限制数量
            self.db.add_concept_relation(
                relation_type=rel['type'],
                source=rel['source'],
                target=rel['target'],
                bidirectional=rel['bidirectional'],
                confidence=rel['confidence'],
                source_type='learned',
                reference_text=f"共现分析: 出现次数={rel['count']}, PMI={rel['pmi']:.2f}"
            )

        return results