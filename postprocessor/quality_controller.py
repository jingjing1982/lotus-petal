"""
质量控制器 - 完整实现版本
"""
import re
from typing import Dict, List, Tuple, Optional, Set
import logging
from collections import defaultdict
import jieba
import jieba.posseg as pseg
from utils.term_database import TermDatabase, FlexibleContextDetector
from .adapter import ProcessingAdapter

logger = logging.getLogger(__name__)


class ConceptGraph:
    """概念关系图 - 用于管理佛教概念之间的语义关系"""

    def __init__(self, term_database=None):
        # 概念之间的关系类型
        self.relations = {
            'opposite': {},  # 对立关系
            'includes': {},  # 包含关系
            'related': {},  # 相关关系
            'stage': {},  # 阶段关系
            'method_goal': {},  # 方法-目标关系
        }

        # 概念的属性
        self.concept_properties = {}

        # 数据库连接
        self.term_database = term_database

        # 从数据库或文件加载概念关系
        self._load_concept_relations()

    def _load_concept_relations(self):
        """从知识库加载概念关系"""
        # 先尝试从数据库加载
        if self.term_database:
            try:
                # 检查表是否存在
                self.term_database.cursor.execute('''
                                                  SELECT name
                                                  FROM sqlite_master
                                                  WHERE type = 'table'
                                                    AND name = 'concept_relations'
                                                  ''')

                table_exists = self.term_database.cursor.fetchone() is not None

                if table_exists:
                    # 从数据库加载关系
                    loaded = self._load_relations_from_database()

                    if loaded:
                        logger.info("从数据库成功加载概念关系")
                        return
                    else:
                        logger.warning("数据库中没有找到概念关系，使用默认关系")
                else:
                    logger.warning("concept_relations表不存在，使用默认关系")
            except Exception as e:
                logger.warning(f"从数据库加载概念关系失败: {e}")

        # 如果没有数据库或加载失败，使用硬编码关系
        self._load_default_relations()

    def _load_relations_from_database(self):
        """从数据库加载关系到内存"""
        try:
            # 获取所有关系
            self.term_database.cursor.execute('''
                                              SELECT relation_type, source_concept, target_concept, bidirectional
                                              FROM concept_relations
                                              ''')

            relations = self.term_database.cursor.fetchall()
            relation_count = 0

            # 清空现有关系
            for rel_type in self.relations:
                self.relations[rel_type] = {}

            # 加载到内存
            for relation_type, source, target, bidirectional in relations:
                if relation_type in self.relations:
                    self.add_relation(relation_type, source, target, bidirectional == 1)
                    relation_count += 1

            return relation_count > 0
        except Exception as e:
            logger.error(f"从数据库加载关系失败: {e}")
            return False

    def _load_default_relations(self):
        """加载默认的硬编码关系"""
        logger.info("加载默认概念关系")

        # 对立关系
        self.add_relation('opposite', '轮回', '涅槃', bidirectional=True)
        self.add_relation('opposite', '有为', '无为', bidirectional=True)
        self.add_relation('opposite', '生死', '不生不灭', bidirectional=True)
        self.add_relation('opposite', '染污', '清净', bidirectional=True)
        self.add_relation('opposite', '迷', '悟', bidirectional=True)

        # 包含关系
        self.add_relation('includes', '六度', ['布施', '持戒', '忍辱', '精进', '禅定', '般若'])
        self.add_relation('includes', '三宝', ['佛', '法', '僧'])
        self.add_relation('includes', '四圣谛', ['苦', '集', '灭', '道'])

        # 阶段关系
        self.add_relation('stage', '资粮道', '加行道')
        self.add_relation('stage', '加行道', '见道')
        self.add_relation('stage', '见道', '修道')
        self.add_relation('stage', '修道', '无学道')

        # 方法-目标关系
        self.add_relation('method_goal', '修行', '解脱')
        self.add_relation('method_goal', '止观', '证悟')
        self.add_relation('method_goal', '念佛', '往生')

    def add_relation(self, relation_type: str, concept1: str, concept2, bidirectional: bool = False):
        """
        添加概念关系到内存缓存

        Parameters:
            relation_type: 关系类型（opposite, includes, related, stage, method_goal）
            concept1: 源概念
            concept2: 目标概念或概念列表
            bidirectional: 是否为双向关系
        """
        if relation_type not in self.relations:
            return

        if concept1 not in self.relations[relation_type]:
            self.relations[relation_type][concept1] = []

        if isinstance(concept2, list):
            self.relations[relation_type][concept1].extend(concept2)
        else:
            self.relations[relation_type][concept1].append(concept2)

        if bidirectional and not isinstance(concept2, list):
            if concept2 not in self.relations[relation_type]:
                self.relations[relation_type][concept2] = []
            self.relations[relation_type][concept2].append(concept1)

    def get_relation(self, concept: str, relation_type: str) -> List[str]:
        """
        获取概念的特定关系

        Parameters:
            concept: 源概念
            relation_type: 关系类型

        Returns:
            目标概念列表
        """
        return self.relations.get(relation_type, {}).get(concept, [])

    def check_compatibility(self, concept1: str, concept2: str, context: str) -> Tuple[bool, str]:
        """
        检查两个概念在特定上下文中的兼容性

        Parameters:
            concept1: 第一个概念
            concept2: 第二个概念
            context: 上下文文本

        Returns:
            (是否兼容, 关系类型)
        """
        # 检查对立关系
        if concept2 in self.get_relation(concept1, 'opposite'):
            # 但如果是对比语境，则兼容
            if self._is_contrastive_context(context, concept1, concept2):
                return True, "contrastive"
            return False, "opposite"

        # 检查包含关系
        if concept2 in self.get_relation(concept1, 'includes'):
            return True, "includes"

        # 检查阶段关系
        if concept2 in self.get_relation(concept1, 'stage'):
            return True, "progression"

        # 检查方法-目标关系
        if concept2 in self.get_relation(concept1, 'method_goal'):
            return True, "method_goal"

        # 默认认为兼容
        return True, "default"

    def _is_contrastive_context(self, context: str, concept1: str, concept2: str) -> bool:
        """判断是否为对比语境"""
        # 更复杂的对比语境识别
        contrast_patterns = [
            r'不是.*而是',
            r'非.*乃',
            r'与其.*不如',
            r'.*但.*',
            r'.*然而.*',
            r'.*却.*',
            r'既.*又.*',
            r'一方面.*另一方面',
        ]

        for pattern in contrast_patterns:
            if re.search(pattern, context):
                return True

        # 检查概念之间的文本
        pos1 = context.find(concept1)
        pos2 = context.find(concept2)
        if pos1 != -1 and pos2 != -1:
            between_text = context[min(pos1, pos2):max(pos1, pos2)]
            if len(between_text) < 20:  # 概念距离很近
                return '而' in between_text or '非' in between_text

        return False

    def sync_to_database(self, term_database=None):
        """将内存中的关系同步到数据库"""
        db = term_database or self.term_database
        if not db:
            logger.error("无法同步到数据库：数据库连接不可用")
            return False

        try:
            # 检查概念关系表是否存在
            db.cursor.execute('''
                              SELECT name
                              FROM sqlite_master
                              WHERE type = 'table'
                                AND name = 'concept_relations'
                              ''')

            if not db.cursor.fetchone():
                logger.error("concept_relations表不存在，无法同步")
                return False

            # 计数器
            added = 0

            # 遍历内存中的所有关系
            for relation_type, concepts in self.relations.items():
                for source, targets in concepts.items():
                    for target in targets:
                        # 检查数据库中是否已有此关系
                        db.cursor.execute('''
                                          SELECT id
                                          FROM concept_relations
                                          WHERE relation_type = ?
                                            AND source_concept = ?
                                            AND target_concept = ?
                                          ''', (relation_type, source, target))

                        if not db.cursor.fetchone():
                            # 确定双向性
                            bidirectional = target in self.relations.get(relation_type, {}).get(source, [])

                            # 添加到数据库
                            db.cursor.execute('''
                                              INSERT INTO concept_relations
                                              (relation_type, source_concept, target_concept, bidirectional,
                                               source_type, confidence)
                                              VALUES (?, ?, ?, ?, ?, ?)
                                              ''', (relation_type, source, target, bidirectional,
                                                    'synced_from_memory', 0.9))
                            added += 1

            db.conn.commit()
            logger.info(f"成功同步 {added} 个关系到数据库")
            return True
        except Exception as e:
            logger.error(f"同步关系到数据库失败: {e}")
            db.conn.rollback()
            return False


class SemanticAnalyzer:
    """语义分析器 - 进行深度语义分析"""

    def __init__(self, term_database):
        self.term_database = term_database
        self.concept_graph = ConceptGraph()

        # 初始化分词器，添加佛教词汇
        self._init_jieba()

        # 语义角色标注
        self.semantic_roles = {
            'agent': [],  # 施事
            'patient': [],  # 受事
            'instrument': [],  # 工具
            'location': [],  # 地点
            'time': [],  # 时间
        }

    def _init_jieba(self):
        """初始化jieba分词器，添加佛教词汇"""
        # 这里应该从术语数据库加载所有术语
        buddhist_terms = [
            '菩萨', '佛陀', '涅槃', '轮回', '般若', '空性',
            '中观', '唯识', '如来藏', '缘起', '业力', '因果'
        ]
        for term in buddhist_terms:
            jieba.add_word(term)

    def analyze_sentence_semantics(self, sentence: str) -> Dict:
        """深度分析句子语义"""
        # 分词和词性标注
        words = pseg.cut(sentence)
        word_list = [(word, flag) for word, flag in words]

        # 提取语义成分
        semantics = {
            'subjects': [],
            'predicates': [],
            'objects': [],
            'modifiers': [],
            'concepts': [],
            'relations': []
        }

        # 识别语义角色
        for i, (word, pos) in enumerate(word_list):
            if pos.startswith('n'):  # 名词
                # 判断是主语还是宾语
                if self._is_subject(word_list, i):
                    semantics['subjects'].append(word)
                elif self._is_object(word_list, i):
                    semantics['objects'].append(word)

                # 检查是否为佛教概念
                if self._is_buddhist_concept(word):
                    semantics['concepts'].append(word)

            elif pos.startswith('v'):  # 动词
                semantics['predicates'].append(word)

            elif pos in ['a', 'ad', 'an']:  # 形容词、副词
                semantics['modifiers'].append(word)

        # 分析概念之间的关系
        semantics['relations'] = self._analyze_concept_relations(semantics['concepts'], sentence)

        # 检查语义完整性
        semantics['completeness'] = self._check_semantic_completeness(semantics)

        return semantics

    def _is_subject(self, word_list: List[Tuple[str, str]], index: int) -> bool:
        """判断是否为主语"""
        word, pos = word_list[index]

        # 主语通常在动词前
        for i in range(index + 1, min(index + 5, len(word_list))):
            if word_list[i][1].startswith('v'):
                # 检查中间是否有其他名词（可能是宾语）
                has_other_noun = any(
                    word_list[j][1].startswith('n')
                    for j in range(index + 1, i)
                )
                return not has_other_noun

        # 句首的名词很可能是主语
        if index < 3:
            return True

        return False

    def _is_object(self, word_list: List[Tuple[str, str]], index: int) -> bool:
        """判断是否为宾语"""
        # 宾语通常在动词后
        for i in range(max(0, index - 5), index):
            if word_list[i][1].startswith('v'):
                return True
        return False

    def _is_buddhist_concept(self, word: str) -> bool:
        """判断是否为佛教概念"""
        # 从术语数据库查询
        # 这里简化为关键词匹配
        buddhist_concepts = {
            '佛', '法', '僧', '菩萨', '涅槃', '轮回', '解脱',
            '般若', '空性', '慈悲', '智慧', '修行', '证悟'
        }
        return word in buddhist_concepts or len(word) > 2 and any(
            char in word for char in ['佛', '法', '禅', '觉', '悟', '慧']
        )

    def _analyze_concept_relations(self, concepts: List[str], sentence: str) -> List[Dict]:
        """分析概念之间的关系"""
        relations = []

        for i, concept1 in enumerate(concepts):
            for concept2 in concepts[i + 1:]:
                # 检查概念兼容性
                compatible, relation_type = self.concept_graph.check_compatibility(
                    concept1, concept2, sentence
                )

                if not compatible:
                    relations.append({
                        'concept1': concept1,
                        'concept2': concept2,
                        'type': 'conflict',
                        'reason': relation_type
                    })
                else:
                    relations.append({
                        'concept1': concept1,
                        'concept2': concept2,
                        'type': relation_type,
                        'compatible': True
                    })

        return relations

    def _check_semantic_completeness(self, semantics: Dict) -> Dict:
        """检查语义完整性"""
        completeness = {
            'has_subject': len(semantics['subjects']) > 0,
            'has_predicate': len(semantics['predicates']) > 0,
            'has_object': len(semantics['objects']) > 0,
            'is_complete': True
        }

        # 基本完整性：至少有主语和谓语
        if not completeness['has_subject'] or not completeness['has_predicate']:
            completeness['is_complete'] = False

        # 及物动词需要宾语
        transitive_verbs = ['修', '证', '念', '观', '度', '教', '说']
        if any(verb in semantics['predicates'] for verb in transitive_verbs):
            if not completeness['has_object']:
                completeness['is_complete'] = False
                completeness['missing'] = 'object_for_transitive_verb'

        return completeness


class QualityController:
    def __init__(self, term_database):
        """初始化质量控制器"""
        self.term_database = term_database
        self.semantic_analyzer = SemanticAnalyzer(term_database)
        self.concept_graph = ConceptGraph()

        # 动态加载搭配规则
        self.collocation_rules = self._load_collocation_rules()

        # 统计信息
        self.correction_stats = defaultdict(int)

    def _load_collocation_rules(self) -> Dict:
        """从知识库加载搭配规则"""
        # 实际应用中应该从数据库加载
        # 这里展示结构
        return {
            'subject_verb': defaultdict(lambda: {'valid': set(), 'invalid': set()}),
            'verb_object': defaultdict(lambda: {'valid': set(), 'invalid': set()}),
            'modifier_noun': defaultdict(lambda: {'valid': set(), 'invalid': set()})
        }

    def refine(self, text: str, context) -> str:
        """执行质量优化"""
        # 分句处理
        sentences = self._split_sentences(text)
        refined_sentences = []

        # 获取上下文信息
        buddhist_context = ProcessingAdapter.get_buddhist_context(context)

        for sentence in sentences:
            if not sentence.strip():
                continue

            # 1. 语义分析
            semantics = self.semantic_analyzer.analyze_sentence_semantics(sentence)

            # 2. 检查并修正语义问题
            sentence = self._fix_semantic_issues(sentence, semantics, context)

            # 3. 确保术语一致性
            sentence = self._ensure_term_consistency(sentence, context)

            # 4. 优化表达
            sentence = self._optimize_expression(sentence, semantics)

            refined_sentences.append(sentence)

        # 重组文本
        result = self._reconstruct_text(refined_sentences)

        # 全局优化
        result = self._global_optimization(result, context)

        return result

    def _split_sentences(self, text: str) -> List[str]:
        """智能分句"""
        # 不只是按标点分割，还要考虑语义完整性
        sentences = []
        current = ""

        # 使用正则表达式进行初步分割
        parts = re.split(r'([。！？；])', text)

        for i in range(0, len(parts), 2):
            if i + 1 < len(parts):
                sentence = parts[i] + parts[i + 1]
            else:
                sentence = parts[i]

            current += sentence

            # 检查语义完整性
            if self._is_semantically_complete(current):
                sentences.append(current.strip())
                current = ""

        if current.strip():
            sentences.append(current.strip())

        return sentences

    def _is_semantically_complete(self, text: str) -> bool:
        """检查语义是否完整"""
        # 简单检查：是否有句末标点
        if text.endswith(('。', '！', '？', '；')):
            # 进一步检查：引号是否配对
            if text.count('"') % 2 == 0 and text.count("'") % 2 == 0:
                return True
        return False

    def _fix_semantic_issues(self, sentence: str, semantics: Dict, context) -> str:
        """修正语义问题"""
        # 1. 检查概念冲突
        for relation in semantics['relations']:
            if relation['type'] == 'conflict':
                sentence = self._resolve_concept_conflict(
                    sentence, relation['concept1'], relation['concept2'], context
                )

        # 2. 检查搭配问题
        sentence = self._fix_collocations(sentence, semantics)

        # 3. 补充缺失成分
        if not semantics['completeness']['is_complete']:
            sentence = self._complete_sentence(sentence, semantics)

        return sentence

    def _resolve_concept_conflict(self, sentence: str, concept1: str, concept2: str, context) -> str:
        """
        解决概念冲突

        Args:
            sentence: 需要处理的句子
            concept1: 第一个概念
            concept2: 第二个概念
            context: 上下文信息

        Returns:
            修正后的句子
        """
        # 1. 分析冲突类型
        conflict_type = self._analyze_conflict_type(sentence, concept1, concept2, context)

        # 2. 根据不同冲突类型应用不同策略
        if conflict_type == 'logical_contradiction':
            return self._resolve_logical_contradiction(sentence, concept1, concept2)
        elif conflict_type == 'hierarchical_mismatch':
            return self._resolve_hierarchical_mismatch(sentence, concept1, concept2)
        elif conflict_type == 'semantic_overlap':
            return self._resolve_semantic_overlap(sentence, concept1, concept2)
        elif conflict_type == 'invalid_combination':
            return self._resolve_invalid_combination(sentence, concept1, concept2)
        else:
            # 未识别的冲突类型或无严重冲突，尝试通用解决方法
            return self._apply_general_conflict_resolution(sentence, concept1, concept2)

    def _analyze_conflict_type(self, sentence: str, concept1: str, concept2: str, context) -> str:
        """分析概念冲突的类型"""
        from .adapter import ProcessingAdapter

        # 获取佛教语境
        buddhist_context = ProcessingAdapter.get_buddhist_context(context)

        # 尝试从知识库获取概念信息
        try:
            concept1_info = self.concept_database.get_concept_info(concept1)
            concept2_info = self.concept_database.get_concept_info(concept2)
        except:
            concept1_info = {}
            concept2_info = {}

        # 1. 检查逻辑矛盾（如"空"与"不空"同时为真）
        if self._are_concepts_contradictory(concept1, concept2, concept1_info, concept2_info):
            return 'logical_contradiction'

        # 2. 检查层级错误（如下位概念描述上位概念）
        if self._is_hierarchical_mismatch(concept1, concept2, concept1_info, concept2_info, sentence):
            return 'hierarchical_mismatch'

        # 3. 检查语义重叠（冗余表达，如"涅槃寂静"）
        if self._is_semantic_overlap(concept1, concept2, concept1_info, concept2_info):
            return 'semantic_overlap'

        # 4. 检查非法组合（如"轮回解脱"作为一个状态）
        if self._is_invalid_combination(concept1, concept2, sentence, buddhist_context):
            return 'invalid_combination'

        # 5. 默认为轻微冲突
        return 'minor_conflict'

    def _analyze_conflict(self, sentence: str, concept1: str, concept2: str) -> Dict:
        """分析概念冲突的严重程度和解决方案"""
        analysis = {
            'severity': 'low',
            'reason': '',
            'suggested_fix': None
        }

        # 检查是否为根本性对立
        if self.concept_graph.get_relation(concept1, 'opposite'):
            if concept2 in self.concept_graph.get_relation(concept1, 'opposite'):
                analysis['severity'] = 'high'
                analysis['reason'] = 'fundamental_opposition'

                # 检查语境，提供修正建议
                if '同时' in sentence or '既' in sentence:
                    # 不能同时具有对立属性
                    analysis['suggested_fix'] = self._suggest_alternative_expression(
                        sentence, concept1, concept2
                    )

        return analysis

    def _are_concepts_contradictory(self, concept1: str, concept2: str,
                                    concept1_info: Dict, concept2_info: Dict) -> bool:
        """检查两个概念是否在逻辑上矛盾"""
        # 常见对立概念对
        contradictory_pairs = [
            ('空', '不空'), ('有', '无'), ('常', '无常'),
            ('清净', '污染'), ('善', '恶'), ('生', '灭')
        ]

        # 直接检查是否是已知的矛盾对
        for c1, c2 in contradictory_pairs:
            if (c1 in concept1 and c2 in concept2) or (c2 in concept1 and c1 in concept2):
                return True

        # 从知识库检查
        if concept1_info and concept2_info:
            relations = concept1_info.get('relations', [])
            for relation in relations:
                if (relation.get('target') == concept2 and
                        relation.get('type') in ['opposite', 'contradicts']):
                    return True

        # 语言学分析（简化版）
        negation_markers = ['不', '非', '无', '没']
        if concept1 == concept2:
            # 检查一个是否为另一个的否定形式
            for marker in negation_markers:
                if (marker + concept1 == concept2) or (concept1 == marker + concept2):
                    return True

        return False

    def _is_hierarchical_mismatch(self, concept1: str, concept2: str,
                                  concept1_info: Dict, concept2_info: Dict, sentence: str) -> bool:
        """检查是否存在层级错误"""
        # 知识库检查
        if concept1_info and concept2_info:
            # 检查是否有直接的层级关系
            relations = concept1_info.get('relations', [])
            for relation in relations:
                if relation.get('target') == concept2:
                    if relation.get('type') == 'includes':
                        # concept1 包含 concept2，检查句中是否有误用
                        return self._is_hierarchy_used_incorrectly(concept1, concept2, sentence, True)
                    elif relation.get('type') == 'included_in':
                        # concept1 被 concept2 包含，检查句中是否有误用
                        return self._is_hierarchy_used_incorrectly(concept1, concept2, sentence, False)

        # 语言分析
        # 检查一个概念是否明显是另一个的子类/超类
        if concept2 in concept1 and len(concept1) > len(concept2):
            # concept1 可能是 concept2 的子类，检查句中用法
            return self._is_hierarchy_used_incorrectly(concept1, concept2, sentence, False)
        elif concept1 in concept2 and len(concept2) > len(concept1):
            # concept2 可能是 concept1 的子类，检查句中用法
            return self._is_hierarchy_used_incorrectly(concept1, concept2, sentence, True)

        return False

    def _is_hierarchy_used_incorrectly(self, concept1: str, concept2: str,
                                       sentence: str, concept1_includes_concept2: bool) -> bool:
        """检查层级关系在句子中的使用是否不当"""
        # 查找两个概念在句子中的位置
        pos1 = sentence.find(concept1)
        pos2 = sentence.find(concept2)

        if pos1 == -1 or pos2 == -1:
            return False

        # 检查是否有不当的修饰关系
        if concept1_includes_concept2:  # concept1 包含 concept2
            # 错误情况：下位概念修饰上位概念，如"大乘是小乘的一种"
            if pos2 < pos1 and self._is_modifying(sentence, pos2, pos1):
                return True
        else:  # concept2 包含 concept1
            # 错误情况：上位概念被描述为下位概念的一部分，如"小乘包含大乘"
            if pos1 < pos2 and self._is_modifying(sentence, pos1, pos2):
                return True

        # 检查动词暗示的关系是否与实际层级关系矛盾
        inclusion_verbs = ['包含', '包括', '囊括', '涵盖']
        part_of_verbs = ['属于', '是...的一部分', '归属']

        for verb in inclusion_verbs:
            if verb in sentence[min(pos1, pos2):max(pos1, pos2)]:
                # 如果句子表达"A包含B"，但实际上B包含A或A、B无关
                if pos1 < pos2 and not concept1_includes_concept2:
                    return True
                elif pos2 < pos1 and concept1_includes_concept2:
                    return True

        return False

    def _is_modifying(self, sentence: str, start_pos: int, end_pos: int) -> bool:
        """检查前一个概念是否在修饰后一个概念"""
        # 简化实现：检查常见的修饰标记
        modifiers = ['的', '之', '性', '式', '型']
        segment = sentence[start_pos:end_pos]

        return any(modifier in segment for modifier in modifiers)

    def _is_semantic_overlap(self, concept1: str, concept2: str,
                             concept1_info: Dict, concept2_info: Dict) -> bool:
        """检查两个概念是否存在语义重叠"""
        # 常见的同义重复表达
        synonymous_pairs = [
            ('涅槃', '寂静'), ('轮回', '生死'), ('空性', '无自性'),
            ('法身', '真如'), ('解脱', '自在'), ('清净', '无垢')
        ]

        # 检查是否是已知的同义对
        for c1, c2 in synonymous_pairs:
            if (c1 in concept1 and c2 in concept2) or (c2 in concept1 and c1 in concept2):
                return True

        # 从知识库检查
        if concept1_info and concept2_info:
            relations = concept1_info.get('relations', [])
            for relation in relations:
                if (relation.get('target') == concept2 and
                        relation.get('type') in ['synonym', 'equivalent']):
                    return True

        return False

    def _is_invalid_combination(self, concept1: str, concept2: str, sentence: str, buddhist_context: str) -> bool:
        """检查是否是无效的概念组合"""
        # 常见的不兼容概念组合
        incompatible_pairs = [
            ('轮回', '解脱'), ('无明', '智慧'), ('烦恼', '菩提'),
            ('我执', '无我'), ('有相', '无相'), ('生死', '涅槃')
        ]

        # 检查是否是不兼容对
        for c1, c2 in incompatible_pairs:
            if ((c1 in concept1 and c2 in concept2) or (c2 in concept1 and c1 in concept2)):
                # 检查是否作为一个组合概念使用（无效）
                if abs(sentence.find(concept1) - sentence.find(concept2)) <= 3:
                    return True

        # 根据佛教语境检查特定的无效组合
        if buddhist_context == 'MADHYAMIKA':  # 中观特有的不兼容组合
            madhyamika_incompatible = [
                ('实有', '空性'), ('自性', '缘起'), ('边见', '中道')
            ]
            for c1, c2 in madhyamika_incompatible:
                if ((c1 in concept1 and c2 in concept2) or (c2 in concept1 and c1 in concept2)):
                    if abs(sentence.find(concept1) - sentence.find(concept2)) <= 3:
                        return True

        return False

    def _resolve_logical_contradiction(self, sentence: str, concept1: str, concept2: str) -> str:
        """解决逻辑矛盾"""
        # 查找两个概念在句子中的位置
        pos1 = sentence.find(concept1)
        pos2 = sentence.find(concept2)

        if pos1 == -1 or pos2 == -1:
            return sentence

        # 策略1：如果概念之间没有明确的关系词，添加一个转折词
        midpoint = (pos1 + len(concept1) + pos2) // 2 if pos1 < pos2 else (pos2 + len(concept2) + pos1) // 2

        # 检查两个概念之间是否已有转折词
        segment_between = sentence[min(pos1 + len(concept1), pos2 + len(concept2)):
                                   max(pos1, pos2)]

        transition_words = ['但', '然而', '而', '却', '不过']
        has_transition = any(word in segment_between for word in transition_words)

        if not has_transition:
            # 寻找合适的插入点（通常是逗号后）
            insert_pos = sentence.find('，', min(pos1, pos2), max(pos1, pos2))
            if insert_pos != -1:
                return sentence[:insert_pos + 1] + '但' + sentence[insert_pos + 1:]

        # 策略2：如果是并列的矛盾陈述，改为条件性陈述
        if '和' in segment_between or '与' in segment_between or '及' in segment_between:
            if pos1 < pos2:
                return sentence.replace(f"{concept1}和{concept2}", f"{concept1}而非{concept2}")
            else:
                return sentence.replace(f"{concept2}和{concept1}", f"{concept2}而非{concept1}")

        # 策略3：添加解释性内容
        explanation = '（在不同层面上）'
        return sentence[:max(pos1, pos2) + len(concept1 if pos1 > pos2 else concept2)] + explanation + sentence[
                                                                                                       max(pos1,
                                                                                                           pos2) + len(
                                                                                                           concept1 if pos1 > pos2 else concept2):]

    def _resolve_hierarchical_mismatch(self, sentence: str, concept1: str, concept2: str) -> str:
        """解决层级不匹配问题"""
        # 查找两个概念在句子中的位置
        pos1 = sentence.find(concept1)
        pos2 = sentence.find(concept2)

        if pos1 == -1 or pos2 == -1:
            return sentence

        # 检查是否有不当的包含关系表述
        inclusion_verbs = ['包含', '包括', '囊括', '涵盖']
        part_of_verbs = ['属于', '是...的一部分', '归属']

        # 寻找可能的错误关系表达
        segment = sentence[min(pos1, pos2):max(pos1, pos2) + max(len(concept1), len(concept2))]

        for verb in inclusion_verbs:
            if verb in segment:
                # 尝试反转关系
                if pos1 < pos2:  # concept1 verb concept2
                    return sentence.replace(f"{concept1}{verb}{concept2}", f"{concept2}{verb}{concept1}")
                else:  # concept2 verb concept1
                    return sentence.replace(f"{concept2}{verb}{concept1}", f"{concept1}{verb}{concept2}")

        # 如果没有明确的动词，但有层级关系的误用
        if pos1 < pos2:
            if self._is_modifying(sentence, pos1, pos2):
                return sentence.replace(f"{concept1}的{concept2}", f"{concept2}的{concept1}")
        else:
            if self._is_modifying(sentence, pos2, pos1):
                return sentence.replace(f"{concept2}的{concept1}", f"{concept1}的{concept2}")

        return sentence

    def _resolve_semantic_overlap(self, sentence: str, concept1: str, concept2: str) -> str:
        """解决语义重叠问题"""
        # 查找两个概念在句子中的位置
        pos1 = sentence.find(concept1)
        pos2 = sentence.find(concept2)

        if pos1 == -1 or pos2 == -1:
            return sentence

        # 策略：移除冗余表达，保留更准确/完整的概念
        if len(concept1) >= len(concept2):
            keep, remove = concept1, concept2
        else:
            keep, remove = concept2, concept1

        # 如果两个概念紧挨着，直接移除冗余部分
        if abs(pos1 - pos2) <= len(concept1) + len(concept2):
            adjacent = min(pos1, pos2) + len(concept1 if pos1 < pos2 else concept2) >= max(pos1, pos2)
            if adjacent:
                return sentence.replace(f"{concept1}{concept2}", keep).replace(f"{concept2}{concept1}", keep)

        # 检查是否有表示同义的词
        equivalence_markers = ['即', '也就是', '亦称', '或称为', '也称为']
        for marker in equivalence_markers:
            pattern1 = f"{concept1}{marker}{concept2}"
            pattern2 = f"{concept2}{marker}{concept1}"

            if pattern1 in sentence:
                return sentence.replace(pattern1, keep)
            elif pattern2 in sentence:
                return sentence.replace(pattern2, keep)

        return sentence

    def _resolve_invalid_combination(self, sentence: str, concept1: str, concept2: str) -> str:
        """解决无效组合问题"""
        # 查找两个概念在句子中的位置
        pos1 = sentence.find(concept1)
        pos2 = sentence.find(concept2)

        if pos1 == -1 or pos2 == -1:
            return sentence

        # 如果两个概念紧挨着形成了错误组合
        if abs(pos1 - pos2) <= 3:
            # 策略1：加入关系词澄清
            if pos1 < pos2:
                return sentence.replace(f"{concept1}{concept2}", f"{concept1}与{concept2}")
            else:
                return sentence.replace(f"{concept2}{concept1}", f"{concept2}与{concept1}")

        # 策略2：添加上下文限定词
        context_qualifiers = {
            '轮回': '在轮回状态中',
            '解脱': '在解脱境界中',
            '无明': '处于无明状态的',
            '智慧': '具有智慧的',
            '烦恼': '烦恼时的',
            '菩提': '证得菩提后的'
        }

        for concept, qualifier in context_qualifiers.items():
            if concept in concept1:
                return sentence.replace(concept1, qualifier)
            elif concept in concept2:
                return sentence.replace(concept2, qualifier)

        return sentence

    def _apply_general_conflict_resolution(self, sentence: str, concept1: str, concept2: str) -> str:
        """应用通用的冲突解决策略"""
        # 查找两个概念在句子中的位置
        pos1 = sentence.find(concept1)
        pos2 = sentence.find(concept2)

        if pos1 == -1 or pos2 == -1:
            return sentence

        # 1. 添加转折词使逻辑通顺
        if pos1 < pos2:
            # 在两个概念之间添加转折
            insert_pos = sentence.find('，', pos1, pos2)
            if insert_pos != -1:
                return sentence[:insert_pos] + '，但' + sentence[insert_pos + 1:]

        # 2. 添加限定词使概念更明确
        clarification = '从某种意义上说，'
        return clarification + sentence

    def _suggest_alternative_expression(self, sentence: str, concept1: str,
                                      concept2: str) -> str:
        """建议替代表达"""
        # 这里应该使用更复杂的NLP技术
        # 现在提供简单的规则

        if concept1 == '轮回' and concept2 == '涅槃':
            if '同时' in sentence:
                return sentence.replace('同时', '从...到')
            elif '既' in sentence:
                return sentence.replace('既', '超越')

        return sentence

    def _fix_collocations(self, sentence: str, semantics: Dict) -> str:
        """修正搭配问题"""
        # 检查主谓搭配
        for subject in semantics['subjects']:
            for predicate in semantics['predicates']:
                if not self._is_valid_collocation('subject_verb', subject, predicate):
                    # 查找更好的搭配
                    better_predicate = self._find_better_collocation(
                        'subject_verb', subject, predicate, sentence
                    )
                    if better_predicate:
                        sentence = sentence.replace(predicate, better_predicate)

        # 检查动宾搭配
        for verb in semantics['predicates']:
            for obj in semantics['objects']:
                if not self._is_valid_collocation('verb_object', verb, obj):
                    better_object = self._find_better_collocation(
                        'verb_object', verb, obj, sentence
                    )
                    if better_object:
                        sentence = sentence.replace(obj, better_object)

        return sentence

    def _is_valid_collocation(self, collocation_type: str, word1: str, word2: str) -> bool:
        """检查搭配是否有效"""
        # 从数据库查询搭配规则
        # 这里使用简化逻辑

        if collocation_type == 'subject_verb':
            # 使用语义相似度和共现频率判断
            return self._check_semantic_compatibility(word1, word2, 'subject_verb')
        elif collocation_type == 'verb_object':
            return self._check_semantic_compatibility(word1, word2, 'verb_object')

        return True

    def _check_semantic_compatibility(self, word1: str, word2: str, 
                                    relation_type: str) -> bool:
        """检查语义兼容性"""
        # 这里应该使用词向量或知识图谱
        # 现在使用规则匹配

        incompatible_pairs = {
            'subject_verb': [
                ('佛陀', ['吃饭', '睡觉', '生气', '害怕']),
                ('菩萨', ['贪婪', '嗔恨', '愚痴']),
            ],
            'verb_object': [
                ('修', ['电脑', '汽车', '房子']),
                ('证', ['文凭', '执照', '合同']),
            ]
        }

        if relation_type in incompatible_pairs:
            for subject, invalid_verbs in incompatible_pairs[relation_type]:
                if word1 == subject and word2 in invalid_verbs:
                    return False

        return True

    def _find_better_collocation(self, collocation_type: str, word1: str, word2: str, context: str) -> Optional[str]:
        """
        查找更好的搭配

        Args:
            collocation_type: 搭配类型
            word1: 第一个词
            word2: 第二个词
            context: 上下文字符串

        Returns:
            更好的搭配表达，如果没有则返回None
        """
        # 1. 尝试从数据库获取更好的搭配
        try:
            better_collocation = self.collocation_database.find_better_collocation(
                collocation_type, word1, word2, context
            )
            if better_collocation:
                return better_collocation
        except:
            # 如果数据库访问失败，使用内置规则
            pass

        # 2. 使用内置规则作为后备方案
        # 针对佛教人物的活动描述
        if collocation_type == 'person_action':
            return self._find_better_person_action(word1, word2)

        # 针对佛教概念的描述方式
        elif collocation_type == 'concept_description':
            return self._find_better_concept_description(word1, word2)

        # 针对佛教修行活动
        elif collocation_type == 'practice_description':
            return self._find_better_practice_description(word1, word2)

        # 针对一般动词搭配
        elif collocation_type == 'verb_object':
            return self._find_better_verb_object(word1, word2)

        return None

    def _find_better_person_action(self, person: str, action: str) -> Optional[str]:
        """为佛教人物找到更合适的行动描述"""
        # 对应不同类型的人物和活动
        buddha_mappings = {
            '吃饭': '受食',
            '吃': '受食',
            '睡觉': '安住',
            '睡': '安住',
            '走': '行',
            '走路': '行',
            '说话': '说法',
            '说': '宣说',
            '讲': '开示',
            '死': '入灭',
            '死亡': '涅槃',
            '生气': '示现忿怒',
            '开心': '示现欢喜',
            '高兴': '示现喜悦'
        }

        bodhisattva_mappings = {
            '帮助': '度化',
            '帮': '救度',
            '教': '教导',
            '教学': '教导',
            '学习': '修学',
            '学': '修学',
            '思考': '思惟',
            '想': '思惟',
            '给': '布施',
            '送': '布施',
            '保护': '守护'
        }

        monk_mappings = {
            '念经': '诵经',
            '诵': '持诵',
            '拜': '礼敬',
            '拜佛': '礼佛',
            '打坐': '禅修',
            '坐禅': '禅修',
            '修行': '修持',
            '受戒': '持戒'
        }

        # 判断人物类型
        buddha_terms = ['佛', '佛陀', '如来', '世尊', '善逝', '调御丈夫']
        bodhisattva_terms = ['菩萨', '观音', '文殊', '普贤', '弥勒', '地藏', '大士']
        monk_terms = ['比丘', '沙门', '阿阇黎', '上师', '法师', '大师', '和尚', '喇嘛', '仁波切']

        # 选择合适的映射表
        if any(term in person for term in buddha_terms):
            mapping = buddha_mappings
        elif any(term in person for term in bodhisattva_terms):
            mapping = bodhisattva_mappings
        elif any(term in person for term in monk_terms):
            mapping = monk_mappings
        else:
            return None

        # 查找替换
        return mapping.get(action)

    def _find_better_concept_description(self, concept: str, description: str) -> Optional[str]:
        """为佛教概念找到更合适的描述方式"""
        # 不同类型概念的描述映射
        dharma_mappings = {
            '很深': '甚深',
            '深': '深奥',
            '难懂': '难解',
            '难理解': '难解',
            '好': '善妙',
            '美好': '妙善',
            '大': '广大',
            '细致': '微妙',
            '详细': '详尽',
            '重要': '殊胜'
        }

        state_mappings = {
            '快乐': '安乐',
            '痛苦': '苦',
            '生气': '嗔',
            '开心': '喜',
            '悲伤': '忧',
            '害怕': '怖',
            '害怕的': '畏惧',
            '平静': '寂静',
            '宁静': '寂静',
            '稳定': '安住'
        }

        practice_mappings = {
            '好的': '善巧',
            '有效的': '善巧',
            '高级的': '增上',
            '进阶的': '增上',
            '正确的': '正',
            '对的': '正',
            '错的': '邪',
            '错误的': '邪'
        }

        # 判断概念类型
        dharma_terms = ['法', '教法', '经', '论', '律', '藏', '乘']
        state_terms = ['心', '境', '识', '受', '想', '行', '果', '道', '定', '果位']
        practice_terms = ['修', '行', '持', '观', '止', '禅', '瑜伽', '戒', '忍']

        # 选择合适的映射表
        if any(term in concept for term in dharma_terms):
            mapping = dharma_mappings
        elif any(term in concept for term in state_terms):
            mapping = state_mappings
        elif any(term in concept for term in practice_terms):
            mapping = practice_mappings
        else:
            return None

        # 查找替换
        return mapping.get(description)

    def _find_better_practice_description(self, practice: str, description: str) -> Optional[str]:
        """为修行活动找到更合适的描述"""
        # 不同修行活动的描述映射
        meditation_mappings = {
            '做': '修',
            '做好': '善修',
            '很好': '善修',
            '专注': '专一',
            '集中': '专注',
            '安静': '寂静',
            '稳定': '稳固',
            '长期': '长时',
            '长时间': '长时'
        }

        recitation_mappings = {
            '念': '持诵',
            '读': '诵读',
            '读诵': '诵持',
            '背': '忆持',
            '记': '忆持',
            '大声': '高声',
            '响亮': '高声',
            '默默': '默然',
            '不出声': '默然'
        }

        ritual_mappings = {
            '做': '行',
            '举行': '修行',
            '进行': '修行',
            '完成': '圆满',
            '成功': '成就',
            '规范': '如法',
            '正确': '如法',
            '虔诚': '至诚',
            '认真': '专一'
        }

        # 判断修行类型
        meditation_terms = ['禅', '修', '观', '止', '三摩地', '三昧', '定']
        recitation_terms = ['诵', '念', '持', '读', '颂']
        ritual_terms = ['法会', '仪轨', '灌顶', '加持', '供养', '会供', '火供']

        # 选择合适的映射表
        if any(term in practice for term in meditation_terms):
            mapping = meditation_mappings
        elif any(term in practice for term in recitation_terms):
            mapping = recitation_mappings
        elif any(term in practice for term in ritual_terms):
            mapping = ritual_mappings
        else:
            return None

        # 查找替换
        return mapping.get(description)

    def _find_better_verb_object(self, verb: str, object: str) -> Optional[str]:
        """为一般动词找到更合适的搭配对象"""
        # 特定动词的搭配映射
        mappings = {
            '修电脑': '修行',
            '看书': '阅读经论',
            '看电视': '观修',
            '打坐': '禅修',
            '睡觉': '休息',
            '打人': '调伏',
            '骂人': '呵责',
            '吵架': '论辩',
            '争论': '法义辨析',
            '问问题': '请法',
            '解决问题': '解惑',
            '解释': '开示',
            '回答': '解答'
        }

        # 尝试直接匹配完整搭配
        complete_collocation = verb + object
        if complete_collocation in mappings:
            return mappings[complete_collocation]

        # 尝试部分匹配
        for k, v in mappings.items():
            if verb in k and object in k:
                return v

        return None

    def _complete_sentence(self, sentence: str, semantics: Dict) -> str:
        """补充句子缺失成分"""
        completeness = semantics['completeness']

        if not completeness['has_subject']:
            # 从上下文推断主语
            implied_subject = self._infer_subject(sentence, semantics)
            if implied_subject:
                sentence = implied_subject + sentence

        if completeness.get('missing') == 'object_for_transitive_verb':
            # 为及物动词补充宾语
            for verb in semantics['predicates']:
                if self._is_transitive_verb(verb) and verb in sentence:
                    default_object = self._get_default_object(verb)
                    if default_object:
                        verb_pos = sentence.find(verb)
                        sentence = sentence[:verb_pos + len(verb)] + default_object + \
                                 sentence[verb_pos + len(verb):]

        return sentence

    def _infer_subject(self, sentence: str, semantics: Dict) -> Optional[str]:
        """推断隐含主语"""
        # 基于动词推断可能的主语
        verb_subject_mapping = {
            '说法': '佛陀',
            '修行': '行者',
            '证悟': '修行者',
            '度众': '菩萨',
        }

        for verb in semantics['predicates']:
            if verb in verb_subject_mapping:
                return verb_subject_mapping[verb]

        return None

    def _is_transitive_verb(self, verb: str) -> bool:
        """判断是否为及物动词"""
        transitive_verbs = {
            '修', '证', '念', '观', '度', '教', '说', '学',
            '持', '诵', '礼', '供', '护', '摄', '化', '利'
        }
        return verb in transitive_verbs

    def _get_default_object(self, verb: str) -> Optional[str]:
        """获取动词的默认宾语"""
        default_objects = {
            '修': '法',
            '证': '果',
            '念': '佛',
            '观': '心',
            '度': '众生',
            '教': '法',
        }
        return default_objects.get(verb)

    def _ensure_term_consistency(self, sentence: str, context) -> str:
        """确保术语一致性"""
        # 使用适配器获取术语映射
        term_mappings = ProcessingAdapter.get_term_mappings(context)

        # 提取所有已使用的术语变体
        term_usage_map = {}

        for placeholder, term_info in term_mappings.items():
            tibetan = term_info.get('tibetan', '')
            chinese = term_info.get('chinese', '')

            if tibetan and chinese:
                if tibetan not in term_usage_map:
                    term_usage_map[tibetan] = set()
                term_usage_map[tibetan].add(chinese)

        # 确保同一术语在整个文档中翻译一致
        for tibetan, chinese_variants in term_usage_map.items():
            if len(chinese_variants) > 1:
                # 选择最佳翻译
                primary_translation = self._select_primary_translation(
                    tibetan, chinese_variants, context
                )

                # 替换所有变体
                for variant in chinese_variants:
                    if variant != primary_translation and variant in sentence:
                        sentence = sentence.replace(variant, primary_translation)

        return sentence

    def _select_primary_translation(self, tibetan: str, variants: Set[str], context) -> str:
        """选择主要翻译"""
        # 构建合适的上下文信息
        term_context = {
            'detected_context': getattr(context, 'detected_context', None),
            'detected_function': getattr(context, 'detected_function', None),
            'text': getattr(context, 'original_text', ''),
            'position': 0  # 默认位置
        }

        # 使用更新后的term_database获取翻译
        best_translation, _ = self.term_database.get_translation(tibetan, term_context)

        if best_translation and best_translation in variants:
            return best_translation

        # 返回最常用的变体
        return max(variants, key=lambda v: context.term_usage_count.get(v, 0))

    def _optimize_expression(self, sentence: str, semantics: Dict) -> str:
        """优化表达方式，使译文更流畅自然"""
        # 1. 优化句式结构 - 仅修正明显问题
        sentence = self._optimize_sentence_structure(sentence, semantics)

        # 2. 改进词语搭配 - 仅修正不自然搭配
        sentence = self._improve_word_collocations(sentence)

        # 3. 仅在极少数明确需要的情况下添加固定表达
        if self._needs_idiomatic_expression(sentence, semantics):
            sentence = self._add_minimal_idiomatic_expressions(sentence, semantics)

        return sentence

    def _needs_idiomatic_expression(self, sentence: str, semantics: Dict) -> bool:
        """判断是否需要添加固定表达"""
        # 仅在以下情况考虑添加：
        # 1. 句子过于生硬，难以理解
        # 2. 存在明显的佛教专业术语需要标准化表达

        # 检查是否有生硬表达
        awkward_expressions = ['做修行', '进行观察', '给予帮助', '作出决定']
        has_awkward = any(expr in sentence for expr in awkward_expressions)

        # 检查是否有需要标准化的佛教术语
        buddhist_terms = ['佛法', '众生', '智慧', '修行']
        has_term = any(term in sentence for term in buddhist_terms)

        # 只有同时满足两个条件才添加
        return has_awkward and has_term

    def _add_minimal_idiomatic_expressions(self, sentence: str, semantics: Dict) -> str:
        """添加最小必要的固定表达"""
        # 仅替换极少数明确不自然的表达
        replacements = {
            '做修行': '修行',
            '进行观察': '观察',
            '给予帮助': '帮助',
            '作出决定': '决定',
            '很大智慧': '深广智慧',
            '努力修行': '精进修行'
        }

        for awkward, natural in replacements.items():
            if awkward in sentence:
                sentence = sentence.replace(awkward, natural)
                # 每句只替换一次，避免过度修饰
                break

        return sentence

    def _optimize_sentence_structure(self, sentence: str, semantics: Dict) -> str:
        """优化句式结构"""
        # 检测并修正生硬的句式结构

        # 1. 修正过长的句子
        if len(sentence) > 40 and ('，' in sentence or '；' in sentence):
            return self._break_long_sentence(sentence)

        # 2. 修正过于简单的句式
        sentence_type = semantics.get('type', 'simple')
        if sentence_type == 'simple' and len(sentence) > 15:
            return self._enrich_simple_sentence(sentence)

        # 3. 修正不当的被动句
        if '被' in sentence:
            return self._optimize_passive_structure(sentence)

        # 4. 处理"是...的"结构
        if '是' in sentence and '的' in sentence:
            return self._optimize_shi_de_structure(sentence)

        return sentence

    def _break_long_sentence(self, sentence: str) -> str:
        """将过长的句子分解为更简短的句子"""
        # 查找合适的断句点
        if '；' in sentence:
            # 优先在分号处断句
            parts = sentence.split('；')
            for i in range(len(parts) - 1):
                if not parts[i].endswith('。'):
                    parts[i] += '。'
            return '；'.join(parts)

        elif '，' in sentence and sentence.count('，') >= 2:
            # 在逗号处断句，但要确保语义完整
            parts = sentence.split('，')
            result = ''

            if len(parts) >= 4:
                # 尝试每两个分句组合
                for i in range(0, len(parts), 2):
                    if i + 1 < len(parts):
                        result += parts[i] + '，' + parts[i + 1]
                        if i + 2 < len(parts):
                            result += '。'
                    else:
                        result += parts[i]

                return result

        return sentence

    def _enrich_simple_sentence(self, sentence: str) -> str:
        """丰富简单句的表达"""
        # 识别句子主题
        first_comma = sentence.find('，')

        if first_comma == -1:
            topic = sentence[:min(5, len(sentence))]
        else:
            topic = sentence[:first_comma]

        # 添加适当的修饰语
        topic_enrichers = {
            '佛': ['慈悲的', '智慧的', '圆满的', '伟大的'],
            '菩萨': ['大悲的', '利他的', '精进的', '慈悲的'],
            '修行': ['精进的', '持续的', '如理的', '如法的'],
            '智慧': ['甚深的', '超越的', '圆满的', '无碍的'],
            '众生': ['轮回中的', '迷惑的', '无明的', '有情的'],
            '法': ['微妙的', '究竟的', '了义的', '甚深的']
        }

        for key, enrichers in topic_enrichers.items():
            if key in topic:
                for enricher in enrichers:
                    # 避免重复添加
                    if enricher not in sentence:
                        # 在主题前添加修饰语
                        insert_pos = sentence.find(key)
                        if insert_pos != -1:
                            return sentence[:insert_pos] + enricher + sentence[insert_pos:]

        return sentence

    def _optimize_passive_structure(self, sentence: str) -> str:
        """优化被动句结构"""
        # 检查是否为不必要的被动句
        patterns = [
            (r'被(.{1,3})所(.{1,3})', r'被\1\2'),  # 移除多余的"所"
            (r'被(.{1,3})给(.{1,3})', r'被\1\2'),  # 移除多余的"给"
            (r'被([^动作]+)(了|的)', r'被动\1\2')  # 修复缺少动词的被动句
        ]

        for pattern, replacement in patterns:
            sentence = re.sub(pattern, replacement, sentence)

        return sentence

    def _optimize_shi_de_structure(self, sentence: str) -> str:
        """优化"是...的"结构"""
        # 检查并优化不当的"是...的"结构
        patterns = [
            (r'是(.{0,10})的(.{0,2})$', r'\1\2'),  # 移除句尾不必要的"是...的"
            (r'是(.{0,5})的时候', r'当\1时'),  # 优化时间表达
            (r'是因为(.{0,10})的', r'因为\1')  # 优化原因表达
        ]

        for pattern, replacement in patterns:
            sentence = re.sub(pattern, replacement, sentence)

        return sentence

    def _improve_word_collocations(self, sentence: str) -> str:
        """改进词语搭配"""
        # 检测并修正不自然的词语搭配
        awkward_collocations = {
            '做修行': '修行',
            '进行观察': '观察',
            '进行禅修': '禅修',
            '做出理解': '理解',
            '给予帮助': '帮助',
            '进行说法': '说法',
            '实行布施': '布施',
            '作出决定': '决定',
            '进行思考': '思考',
            '做好准备': '准备'
        }

        for awkward, natural in awkward_collocations.items():
            if awkward in sentence:
                sentence = sentence.replace(awkward, natural)

        # 改进动宾搭配
        verb_object_pairs = {
            '修成佛': '修成佛果',
            '学习法': '学习佛法',
            '证得果': '证得果位',
            '持诵经': '持诵经文',
            '礼拜佛': '礼敬佛陀'
        }

        for awkward, natural in verb_object_pairs.items():
            if awkward in sentence:
                sentence = sentence.replace(awkward, natural)

        return sentence

    def _add_idiomatic_expressions(self, sentence: str, semantics: Dict) -> str:
        """添加恰当的成语或固定表达"""
        # 根据语义场景添加适当的成语或固定表达
        context = semantics.get('context', '')

        # 根据上下文选择合适的成语
        if '修行' in sentence and '努力' in sentence:
            return sentence.replace('努力修行', '精进修行')

        if '智慧' in sentence and '很大' in sentence:
            return sentence.replace('很大智慧', '甚深智慧')

        if '佛法' in sentence and '学习' in sentence:
            return sentence.replace('学习佛法', '闻思佛法')

        if '众生' in sentence and '帮助' in sentence:
            return sentence.replace('帮助众生', '普度众生')

        # 根据语义类型添加固定表达
        sentence_type = semantics.get('type', '')

        if sentence_type == 'contrast':
            # 转折类句子
            if '但是' in sentence:
                return sentence.replace('但是', '然而')
            elif '可是' in sentence:
                return sentence.replace('可是', '然而')

        elif sentence_type == 'cause':
            # 因果类句子
            if '因为' in sentence and '所以' in sentence:
                return sentence.replace('因为', '由于').replace('所以', '因此')

        elif sentence_type == 'condition':
            # 条件类句子
            if '如果' in sentence and '就' in sentence:
                return sentence.replace('如果', '若').replace('就', '则')

        return sentence

    def _improve_rhetoric(self, sentence: str, semantics: Dict) -> str:
        """优化修辞手法"""
        # 根据语义和句子类型应用适当的修辞
        sentence_type = semantics.get('type', '')

        # 对于描述性句子，考虑添加比喻
        if sentence_type == 'descriptive':
            if '智慧' in sentence and '如' not in sentence:
                return sentence.replace('智慧', '如日般的智慧')

            if '慈悲' in sentence and '如' not in sentence:
                return sentence.replace('慈悲', '如海般的慈悲')

        # 对于强调性句子，考虑使用排比
        elif sentence_type == 'emphatic':
            if sentence.count('，') == 2:
                parts = sentence.split('，')
                if len(parts) >= 3 and len(parts[0]) > 2 and len(parts[1]) > 2 and len(parts[2]) > 2:
                    # 检测是否可能形成排比
                    if parts[0][-1] == parts[1][-1] or parts[1][-1] == parts[2][-1]:
                        # 调整语气助词，使三部分结构一致
                        for i in range(3):
                            if not parts[i].endswith('也') and not parts[i].endswith('矣') and not parts[i].endswith(
                                    '焉'):
                                if i < 2:
                                    parts[i] += '，'
                                else:
                                    parts[i] += '。'

                        return ''.join(parts)

        # 对于教义性句子，考虑使用对偶
        elif sentence_type == 'doctrinal':
            if '修' in sentence and '证' in sentence:
                return sentence.replace('修', '修行').replace('证', '证悟')

            if '福' in sentence and '慧' in sentence:
                return sentence.replace('福', '福德').replace('慧', '智慧')

        return sentence

    def _remove_redundancy(self, sentence: str) -> str:
        """删除冗余表达"""
        # 检测并删除重复的语义成分
        # 使用更智能的方法而不是简单的模式匹配

        # 分词
        words = list(jieba.cut(sentence))

        # 检测语义重复
        cleaned_words = []
        seen_concepts = set()

        for i, word in enumerate(words):
            # 获取词的语义类别
            semantic_category = self._get_semantic_category(word)

            if semantic_category:
                # 检查是否已有相同语义的词
                if semantic_category not in seen_concepts:
                    cleaned_words.append(word)
                    seen_concepts.add(semantic_category)
                else:
                    # 检查是否为必要的重复（如强调）
                    if not self._is_necessary_repetition(words, i, word):
                        continue
                    else:
                        cleaned_words.append(word)
            else:
                cleaned_words.append(word)

        return ''.join(cleaned_words)

    def _get_semantic_category(self, word: str) -> Optional[str]:
        """获取词的语义类别"""
        # 这里应该使用词义消歧技术
        # 简化实现

        semantic_categories = {
            '非常': 'intensifier',
            '很': 'intensifier',
            '十分': 'intensifier',
            '已经': 'aspect',
            '曾经': 'aspect',
        }

        return semantic_categories.get(word)

    def _is_necessary_repetition(self, words: List[str], position: int, word: str) -> bool:
        """判断是否为必要的重复"""
        # 某些重复是修辞需要，如"慢慢地"、"好好地"
        if position > 0 and words[position-1] == word:
            if word in ['慢', '好', '细', '轻']:
                return True

        # 佛教文献中的特殊重复，如"如是如是"
        if word == '如是' and position > 0 and words[position-1] == '如是':
            return True

        return False

    def _optimize_word_order(self, sentence: str, semantics: Dict) -> str:
        """优化语序"""
        # 中文的标准语序：主语+状语+谓语+补语+宾语
        # 但佛教文献可能有特殊语序

        # 这里需要句法分析
        # 简化处理：确保主语在谓语前

        if semantics['subjects'] and semantics['predicates']:
            subject = semantics['subjects'][0]
            predicate = semantics['predicates'][0]

            subj_pos = sentence.find(subject)
            pred_pos = sentence.find(predicate)

            if subj_pos > pred_pos and subj_pos != -1 and pred_pos != -1:
                # 需要调整语序
                # 这里需要更复杂的处理
                pass

        return sentence

    def _polish_expression(self, sentence: str) -> str:
        """润色表达"""
        # 使用更自然的表达方式
        polish_rules = [
            (r'进行(\S{2})', r'\1'),  # "进行修行" -> "修行"
            (r'(\S{2})活动', r'\1'),  # "修行活动" -> "修行"
        ]

        for pattern, replacement in polish_rules:
            sentence = re.sub(pattern, replacement, sentence)

        return sentence

    def _reconstruct_text(self, sentences: List[str]) -> str:
        """重组文本"""
        # 不是简单地连接句子，而是确保段落连贯
        if not sentences:
            return ""

        result = sentences[0]

        for i in range(1, len(sentences)):
            # 检查是否需要段落分隔
            if self._should_start_new_paragraph(sentences[i-1], sentences[i]):
                result += '\n\n' + sentences[i]
            else:
                # 检查是否需要连接词
                connector = self._get_sentence_connector(sentences[i-1], sentences[i])
                if connector:
                    result += connector + sentences[i]
                else:
                    result += sentences[i]

        return result

    def _should_start_new_paragraph(self, prev_sentence: str, curr_sentence: str) -> bool:
        """判断是否应该开始新段落"""
        # 基于语义转换判断

        # 如果主题变化很大，开始新段落
        prev_concepts = self._extract_main_concepts(prev_sentence)
        curr_concepts = self._extract_main_concepts(curr_sentence)

        overlap = len(prev_concepts.intersection(curr_concepts))
        if overlap == 0 and len(prev_concepts) > 0 and len(curr_concepts) > 0:
            return True

        # 如果有明显的转折标志
        if curr_sentence.startswith(('其次', '再次', '最后', '总之')):
            return True

        return False

    def _extract_main_concepts(self, sentence: str) -> Set[str]:
        """提取句子的主要概念"""
        concepts = set()
        words = jieba.cut(sentence)

        for word in words:
            if self._is_buddhist_concept(word) or len(word) > 2:
                concepts.add(word)

        return concepts

    def _is_buddhist_concept(self, word: str) -> bool:
        """判断是否为佛教概念"""
        # 应该查询术语数据库
        key_chars = ['佛', '法', '僧', '禅', '觉', '悟', '修', '证']
        return any(char in word for char in key_chars)

    def _get_sentence_connector(self, prev_sentence: str, curr_sentence: str) -> str:
        """获取句子连接词"""
        # 基于句子关系选择连接词

        # 如果当前句是对前句的解释
        if self._is_explanation(prev_sentence, curr_sentence):
            return '也就是说，'

        # 如果是递进关系
        if self._is_progression(prev_sentence, curr_sentence):
            return '进而'

        return ''

    def _is_explanation(self, prev: str, curr: str) -> bool:
        """判断是否为解释关系"""
        explanation_markers = ['即', '也就是', '换言之', '所谓']
        return any(marker in curr[:10] for marker in explanation_markers)

    def _is_progression(self, prev: str, curr: str) -> bool:
        """判断是否为递进关系"""
        progression_markers = ['更', '还', '甚至', '何况']
        return any(marker in curr[:10] for marker in progression_markers)

    def _global_optimization(self, text: str, context) -> str:
        """
        全局优化文本

        Args:
            text: 整体文本
            context: 上下文信息

        Returns:
            优化后的文本
        """
        from .adapter import ProcessingAdapter

        # 获取文本类型信息
        is_verse = ProcessingAdapter.is_verse(context)
        has_enumeration = ProcessingAdapter.has_enumeration(context)
        buddhist_context = ProcessingAdapter.get_buddhist_context(context)

        # 跳过特殊格式的文本
        if is_verse or has_enumeration:
            return text

        # 1. 全局一致性优化
        text = self._ensure_global_consistency(text, buddhist_context)

        # 2. 风格统一
        text = self._unify_style(text, buddhist_context)

        # 3. 添加必要的上下文
        text = self._add_necessary_context(text, buddhist_context)

        # 4. 优化段落结构
        text = self._optimize_paragraph_structure(text)

        return text

    def _ensure_global_consistency(self, text: str, buddhist_context: str) -> str:
        """确保全局一致性"""
        # 1. 术语一致性
        text = self._ensure_term_consistency(text, buddhist_context)

        # 2. 称呼一致性
        text = self._ensure_name_consistency(text)

        # 3. 标点符号一致性
        text = self._ensure_punctuation_consistency(text)

        # 4. 时态一致性
        text = self._ensure_tense_consistency(text)

        return text

    def _ensure_term_consistency(self, text: str, buddhist_context: str) -> str:
        """确保术语使用一致"""
        # 术语变体映射
        term_variants = {
            'MADHYAMIKA': {  # 中观派术语变体
                '空性': ['空', '性空', '自性空'],
                '中观': ['中道观', '中论'],
                '二谛': ['二谛理', '二种谛']
            },
            'YOGACARA': {  # 唯识派术语变体
                '阿赖耶识': ['阿赖耶', '藏识', '第八识'],
                '唯识': ['唯心', '惟识', '唯识无境'],
                '三性': ['三自性', '遍计所执性']
            },
            'VAJRAYANA': {  # 密宗术语变体
                '金刚乘': ['密乘', '金刚道', '秘密乘'],
                '本尊': ['尊', '本尊佛', '本尊法'],
                '灌顶': ['灌顶法', '灌顶仪轨']
            },
            'GENERAL': {  # 通用术语变体
                '佛陀': ['佛', '世尊', '如来'],
                '涅槃': ['般涅槃', '寂灭'],
                '菩提': ['觉', '菩提心', '觉悟']
            }
        }

        # 选择当前语境的术语变体
        context_variants = term_variants.get(buddhist_context, term_variants['GENERAL'])

        # 对每个主要术语，统一使用其首选形式
        for primary_term, variants in context_variants.items():
            # 检查文本中是否使用了多种变体
            used_variants = []

            for variant in variants:
                if variant in text:
                    used_variants.append(variant)

            # 如果同时使用了变体和主要术语，统一为主要术语
            if used_variants:
                for variant in used_variants:
                    text = text.replace(variant, primary_term)

        return text

    def _ensure_name_consistency(self, text: str) -> str:
        """确保人名称呼一致"""
        # 检测文本中的人名称呼
        name_patterns = [
            (r'佛[陀祖]?', '佛陀'),
            (r'世尊', '世尊'),
            (r'如来', '如来'),
            (r'阿弥陀[佛]?', '阿弥陀佛'),
            (r'释迦[牟尼]?[佛]?', '释迦牟尼佛'),
            (r'观[世]?[音]?[菩萨]?', '观世音菩萨'),
            (r'文殊[菩萨]?', '文殊菩萨')
        ]

        # 检测各称呼在文本中的出现次数
        name_counts = {}

        for pattern, full_name in name_patterns:
            matches = re.finditer(pattern, text)
            name_counts[full_name] = len(list(matches))

        # 对出现次数最多的称呼，统一使用其完整形式
        for pattern, full_name in name_patterns:
            if name_counts.get(full_name, 0) > 1:
                text = re.sub(pattern, full_name, text)

        return text

    def _ensure_punctuation_consistency(self, text: str) -> str:
        """确保标点符号使用一致"""
        # 统一使用中文标点
        punctuation_mapping = {
            '.': '。',
            ',': '，',
            ':': '：',
            ';': '；',
            '?': '？',
            '!': '！',
            '"': '「',
            '"': '」',
            '(': '（',
            ')': '）'
        }

        for western, chinese in punctuation_mapping.items():
            text = text.replace(western, chinese)

        # 修正不一致的引号使用
        open_quotes = text.count('「')
        close_quotes = text.count('」')

        if open_quotes > close_quotes:
            text += '」' * (open_quotes - close_quotes)
        elif close_quotes > open_quotes:
            text = '「' * (close_quotes - open_quotes) + text

        return text

    def _ensure_tense_consistency(self, text: str) -> str:
        """确保时态一致"""
        # 检测主要时态
        past_markers = ['了', '过', '已经']
        present_markers = ['正在', '现在', '当下']
        future_markers = ['将', '会', '将要']

        past_count = sum(text.count(marker) for marker in past_markers)
        present_count = sum(text.count(marker) for marker in present_markers)
        future_count = sum(text.count(marker) for marker in future_markers)

        # 确定主要时态
        main_tense = 'present'  # 默认为现在时
        if past_count > present_count and past_count > future_count:
            main_tense = 'past'
        elif future_count > past_count and future_count > present_count:
            main_tense = 'future'

        # 分析句子并调整不一致的时态
        sentences = re.split(r'([。！？])', text)
        result = []

        for i in range(0, len(sentences), 2):
            sentence = sentences[i]
            if i + 1 < len(sentences):
                sentence += sentences[i + 1]

            # 检测句子时态
            sentence_past = any(marker in sentence for marker in past_markers)
            sentence_present = any(marker in sentence for marker in present_markers)
            sentence_future = any(marker in sentence for marker in future_markers)

            # 如果句子有明确时态标记且与主要时态不同，尝试调整
            if main_tense == 'past' and (sentence_present or sentence_future) and not sentence_past:
                # 调整为过去时
                if '正在' in sentence:
                    sentence = sentence.replace('正在', '正')
                if '将要' in sentence:
                    sentence = sentence.replace('将要', '即将')
                if '会' in sentence and not '会' + sentence[-1] == '会。':
                    sentence = sentence.replace('会', '将会')

                # 添加过去时标记
                if '。' in sentence:
                    sentence = sentence.replace('。', '了。')
                elif '！' in sentence:
                    sentence = sentence.replace('！', '了！')
                elif '？' in sentence:
                    # 疑问句一般不加"了"
                    pass

            elif main_tense == 'present' and sentence_past and not sentence_present:
                # 调整为现在时
                if '了' in sentence:
                    sentence = sentence.replace('了', '')

            elif main_tense == 'future' and sentence_past and not sentence_future:
                # 调整为将来时
                if '了' in sentence:
                    sentence = sentence.replace('了', '将')

            result.append(sentence)

        return ''.join(result)

    def _unify_style(self, text: str, buddhist_context: str) -> str:
        """统一文风格调"""
        # 检测当前文风
        formal_markers = ['之', '乃', '焉', '矣', '哉', '夫', '尔']
        informal_markers = ['啊', '呢', '吧', '啦', '嘛', '呀']

        formal_count = sum(text.count(marker) for marker in formal_markers)
        informal_count = sum(text.count(marker) for marker in informal_markers)

        # 确定文风
        is_formal = formal_count > informal_count

        # 根据佛教语境推荐的文风
        recommended_style = 'formal'  # 默认推荐正式文风

        if buddhist_context == 'MADHYAMIKA' or buddhist_context == 'YOGACARA':
            # 中观和唯识文献通常更正式
            recommended_style = 'formal'
        elif buddhist_context == 'VAJRAYANA':
            # 密宗文献可以有些灵活
            if informal_count > 0:
                recommended_style = 'mixed'

        # 调整文风，如果当前文风与推荐文风不一致
        if recommended_style == 'formal' and not is_formal:
            # 调整为更正式的文风
            for marker in informal_markers:
                if marker in text:
                    # 移除过于口语化的语气词
                    text = text.replace(marker, '')

            # 替换为更正式的表达
            text = text.replace('的', '之').replace('和', '与')

        elif recommended_style == 'mixed' and (formal_count > 2 * informal_count):
            # 调整为混合文风，保留一些正式表达但增加可读性
            text = text.replace('之', '的').replace('乃', '就是')

        return text

    def _add_necessary_context(self, text: str, buddhist_context: str) -> str:
        """添加必要的上下文"""
        # 仅在以下条件下添加上下文：
        # 1. 文本较短且缺乏上下文
        # 2. 内容难以理解且需要额外说明

        # 检查是否需要添加上下文信息
        needs_context = False

        # 短文本可能缺乏上下文
        if len(text) < 50:
            needs_context = True

        # 文本中包含专业术语但缺乏解释
        key_terms = ['二谛', '三性', '四灌', '八识']
        if any(term in text for term in key_terms) and '（' not in text:
            needs_context = True

        # 只有确实需要时才添加上下文
        if needs_context:
            # 最多添加一个简短前缀，不添加术语注解
            if buddhist_context == 'MADHYAMIKA' and '空性' in text and not '中观' in text:
                text = '依中观义，' + text

            elif buddhist_context == 'YOGACARA' and '唯识' in text and not '唯识宗' in text:
                text = '依唯识义，' + text

            elif buddhist_context == 'VAJRAYANA' and '密' in text and not '密宗' in text:
                text = '依密乘义，' + text

        return text

    def _optimize_paragraph_structure(self, text: str) -> str:
        """优化段落结构"""
        # 分析当前段落结构
        paragraphs = text.split('\n\n')

        # 如果只有一个长段落，尝试拆分
        if len(paragraphs) == 1 and len(paragraphs[0]) > 200:
            sentences = re.split(r'([。！？])', paragraphs[0])
            new_paragraphs = []
            current = ''

            for i in range(0, len(sentences), 2):
                if i + 1 < len(sentences):
                    current += sentences[i] + sentences[i + 1]
                else:
                    current += sentences[i]

                # 每3-4个句子或达到一定长度时形成新段落
                if (i // 2) % 3 == 2 or len(current) > 100:
                    new_paragraphs.append(current)
                    current = ''

            if current:
                new_paragraphs.append(current)

            text = '\n\n'.join(new_paragraphs)

        # 检查段落之间的逻辑连接
        paragraphs = text.split('\n\n')
        if len(paragraphs) >= 2:
            for i in range(1, len(paragraphs)):
                # 检查是否需要添加过渡词
                prev_end = paragraphs[i - 1][-1] if paragraphs[i - 1] else ''
                current_start = paragraphs[i][:10] if paragraphs[i] else ''

                # 如果前一段末尾没有明确的结束感，当前段开头也没有过渡词
                if prev_end in '。！？' and not any(
                        marker in current_start for marker in ['因此', '所以', '然而', '但是', '此外', '接着']):
                    # 根据上下文添加适当的过渡词
                    if '反' in paragraphs[i - 1] or '但' in paragraphs[i]:
                        paragraphs[i] = '然而，' + paragraphs[i]
                    elif '因' in paragraphs[i - 1] or '所以' in paragraphs[i]:
                        paragraphs[i] = '因此，' + paragraphs[i]
                    elif i == len(paragraphs) - 1:
                        paragraphs[i] = '总之，' + paragraphs[i]
                    else:
                        paragraphs[i] = '此外，' + paragraphs[i]

        return '\n\n'.join(paragraphs)

    def _ensure_global_term_consistency(self, text: str, context) -> str:
        """确保全局术语一致性"""
        # 获取上下文中的术语信息
        term_mappings = getattr(context, 'term_mappings', {})
        if not term_mappings:
            return text

        # 确定每个藏文术语的首选翻译
        preferred_translations = {}
        for term_id, term_info in term_mappings.items():
            tibetan = term_info.get('tibetan', '')
            chinese = term_info.get('chinese', '')
            if tibetan and chinese:
                if tibetan not in preferred_translations:
                    preferred_translations[tibetan] = chinese

        # 对文本进行统一替换
        for tibetan, preferred in preferred_translations.items():
            # 查找该术语的所有变体翻译
            variants = self._find_term_variants(tibetan, context)
            for variant in variants:
                if variant != preferred:
                    text = text.replace(variant, preferred)

        return text

    def _find_term_variants(self, tibetan: str, context) -> Set[str]:
        """查找术语的所有翻译变体"""
        variants = set()
        for term_info in getattr(context, 'term_mappings', {}).values():
            if term_info.get('tibetan') == tibetan:
                variants.add(term_info.get('chinese', ''))
        return variants

    def _optimize_paragraph_structure(self, text: str) -> str:
        """优化段落结构"""
        paragraphs = text.split('\n\n')
        optimized = []

        for para in paragraphs:
            if not para.strip():
                continue

            # 检查段落长度
            if len(para) > 500:
                # 尝试分割长段落
                split_paras = self._split_long_paragraph(para)
                optimized.extend(split_paras)
            elif len(para) < 50 and optimized:
                # 短段落可能需要合并
                optimized[-1] += para
            else:
                optimized.append(para)

        return '\n\n'.join(optimized)

    def _split_long_paragraph(self, paragraph: str) -> List[str]:
        """分割长段落"""
        # 寻找合适的分割点
        sentences = re.split(r'[。！？]', paragraph)

        paragraphs = []
        current = ""

        for sent in sentences:
            if not sent.strip():
                continue

            current += sent + '。'

            # 在语义转折处分割
            if len(current) > 200 and self._is_semantic_boundary(sent):
                paragraphs.append(current.strip())
                current = ""

        if current.strip():
            paragraphs.append(current.strip())

        return paragraphs

    def _is_semantic_boundary(self, sentence: str) -> bool:
        """判断是否为语义边界"""
        boundary_markers = ['其次', '再者', '最后', '总之', '因此', '所以']
        return any(marker in sentence for marker in boundary_markers)

    def _fix_punctuation(self, text: str) -> str:
        """修正标点符号"""
        # 智能标点修正，不只是模式替换

        # 修正引号
        text = self._fix_quotes(text)

        # 修正句号
        text = self._fix_periods(text)

        # 修正逗号
        text = self._fix_commas(text)

        return text

    def _fix_quotes(self, text: str) -> str:
        """修正引号使用"""
        # 确保引号配对
        quote_stack = []
        fixed_text = []

        i = 0
        while i < len(text):
            char = text[i]

            if char == '"':
                if not quote_stack:
                    quote_stack.append('"')
                    fixed_text.append('"')
                else:
                    quote_stack.pop()
                    fixed_text.append('"')
            elif char == '"' and quote_stack:
                quote_stack.pop()
                fixed_text.append(char)
            elif char == '"' and not quote_stack:
                quote_stack.append('"')
                fixed_text.append('"')
            else:
                fixed_text.append(char)

            i += 1

        # 如果还有未配对的引号
        if quote_stack:
            fixed_text.append('"')

        return ''.join(fixed_text)

    def _fix_periods(self, text: str) -> str:
        """修正句号使用"""
        # 删除重复句号
        text = re.sub(r'。{2,}', '。', text)

        # 确保段落末尾有句号
        lines = text.split('\n')
        fixed_lines = []

        for line in lines:
            line = line.strip()
            if line and not line.endswith(('。', '！', '？', '：', '；', '"', "'")):
                line += '。'
            fixed_lines.append(line)

        return '\n'.join(fixed_lines)

    def _fix_commas(self, text: str) -> str:
        """修正逗号使用"""
        # 删除句首逗号
        text = re.sub(r'^，', '', text, flags=re.MULTILINE)

        # 删除重复逗号
        text = re.sub(r'，{2,}', '，', text)

        # 修正逗号和句号连用
        text = re.sub(r'，。', '。', text)

        return text