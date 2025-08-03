"""
标签管理器 - 管理和统一术语数据库中的上下文和功能标签
"""
import logging
from typing import List, Dict, Set, Tuple
from utils.term_database import TermDatabase

logger = logging.getLogger(__name__)


class LabelManager:
    """管理术语数据库中的标签系统"""

    def __init__(self, term_database: TermDatabase):
        self.db = term_database

        # 标准化的标签映射
        self.context_label_map = {}
        self.function_label_map = {}

        # 加载现有标签
        self._load_existing_labels()

    def _load_existing_labels(self):
        """加载数据库中现有的标签"""
        try:
            # 加载上下文标签
            self.db.cursor.execute('''
                                   SELECT DISTINCT context_label
                                   FROM translations
                                   WHERE context_label IS NOT NULL
                                     AND context_label != ''
                                   ''')
            context_labels = set(row[0] for row in self.db.cursor.fetchall())

            self.db.cursor.execute('''
                                   SELECT DISTINCT context_label
                                   FROM term_combinations
                                   WHERE context_label IS NOT NULL
                                     AND context_label != ''
                                   ''')
            context_labels.update(row[0] for row in self.db.cursor.fetchall())

            # 加载功能标签
            self.db.cursor.execute('''
                                   SELECT DISTINCT function_label
                                   FROM translations
                                   WHERE function_label IS NOT NULL
                                     AND function_label != ''
                                   ''')
            function_labels = set(row[0] for row in self.db.cursor.fetchall())

            self.db.cursor.execute('''
                                   SELECT DISTINCT function_label
                                   FROM term_combinations
                                   WHERE function_label IS NOT NULL
                                     AND function_label != ''
                                   ''')
            function_labels.update(row[0] for row in self.db.cursor.fetchall())

            # 初始化映射（默认映射到自身）
            for label in context_labels:
                self.context_label_map[label] = label

            for label in function_labels:
                self.function_label_map[label] = label

            logger.info(f"已加载 {len(context_labels)} 个上下文标签和 {len(function_labels)} 个功能标签")
        except Exception as e:
            logger.error(f"加载标签失败: {e}")

    def define_context_label_mapping(self, mapping: Dict[str, str]):
        """定义上下文标签的标准化映射"""
        self.context_label_map.update(mapping)

    def define_function_label_mapping(self, mapping: Dict[str, str]):
        """定义功能标签的标准化映射"""
        self.function_label_map.update(mapping)

    def get_standardized_context_label(self, label: str) -> str:
        """获取标准化的上下文标签"""
        if not label:
            return ''
        return self.context_label_map.get(label, label)

    def get_standardized_function_label(self, label: str) -> str:
        """获取标准化的功能标签"""
        if not label:
            return ''
        return self.function_label_map.get(label, label)

    def apply_standardization(self) -> Dict[str, int]:
        """应用标签标准化到数据库"""
        updates = {'translations': 0, 'term_combinations': 0}

        try:
            # 更新翻译表的上下文标签
            for original, standardized in self.context_label_map.items():
                if original != standardized:
                    self.db.cursor.execute('''
                                           UPDATE translations
                                           SET context_label = ?
                                           WHERE context_label = ?
                                           ''', (standardized, original))
                    updates['translations'] += self.db.cursor.rowcount

            # 更新翻译表的功能标签
            for original, standardized in self.function_label_map.items():
                if original != standardized:
                    self.db.cursor.execute('''
                                           UPDATE translations
                                           SET function_label = ?
                                           WHERE function_label = ?
                                           ''', (standardized, original))
                    updates['translations'] += self.db.cursor.rowcount

            # 更新术语组合表的上下文标签
            for original, standardized in self.context_label_map.items():
                if original != standardized:
                    self.db.cursor.execute('''
                                           UPDATE term_combinations
                                           SET context_label = ?
                                           WHERE context_label = ?
                                           ''', (standardized, original))
                    updates['term_combinations'] += self.db.cursor.rowcount

            # 更新术语组合表的功能标签
            for original, standardized in self.function_label_map.items():
                if original != standardized:
                    self.db.cursor.execute('''
                                           UPDATE term_combinations
                                           SET function_label = ?
                                           WHERE function_label = ?
                                           ''', (standardized, original))
                    updates['term_combinations'] += self.db.cursor.rowcount

            self.db.conn.commit()
            logger.info(f"已更新 {updates['translations']} 个翻译标签和 {updates['term_combinations']} 个组合标签")
            return updates
        except Exception as e:
            logger.error(f"应用标签标准化失败: {e}")
            self.db.conn.rollback()
            return updates

    def check_label_consistency(self) -> List[Dict]:
        """检查标签一致性"""
        issues = []

        try:
            # 检查相同藏文术语在不同表中使用不同标签的情况
            self.db.cursor.execute('''
                                   SELECT t.tibetan, tr.context_label, tc.context_label
                                   FROM terms t
                                            JOIN translations tr ON t.id = tr.term_id
                                            JOIN term_combinations tc ON t.tibetan = tc.combined_tibetan
                                   WHERE tr.context_label != tc.context_label
                  AND tr.context_label IS NOT NULL 
                  AND tc.context_label IS NOT NULL
                                   ''')

            for tibetan, trans_context, comb_context in self.db.cursor.fetchall():
                issues.append({
                    'type': 'context_label_mismatch',
                    'term': tibetan,
                    'translations_label': trans_context,
                    'combinations_label': comb_context
                })

            # 检查相同藏文术语在不同表中使用不同功能标签的情况
            self.db.cursor.execute('''
                                   SELECT t.tibetan, tr.function_label, tc.function_label
                                   FROM terms t
                                            JOIN translations tr ON t.id = tr.term_id
                                            JOIN term_combinations tc ON t.tibetan = tc.combined_tibetan
                                   WHERE tr.function_label != tc.function_label
                  AND tr.function_label IS NOT NULL 
                  AND tc.function_label IS NOT NULL
                                   ''')

            for tibetan, trans_function, comb_function in self.db.cursor.fetchall():
                issues.append({
                    'type': 'function_label_mismatch',
                    'term': tibetan,
                    'translations_label': trans_function,
                    'combinations_label': comb_function
                })

            return issues
        except Exception as e:
            logger.error(f"检查标签一致性失败: {e}")
            return []

    def suggest_label_mappings(self) -> Dict[str, Dict[str, str]]:
        """基于现有数据建议标签标准化映射"""
        suggestions = {
            'context': {},
            'function': {}
        }

        # 获取所有上下文标签及其使用频率
        context_counts = self._get_label_frequencies('context_label')
        function_counts = self._get_label_frequencies('function_label')

        # 基于相似性分组标签
        context_groups = self._group_similar_labels(context_counts.keys())
        function_groups = self._group_similar_labels(function_counts.keys())

        # 为每组选择标准标签（使用频率最高的）
        for group in context_groups:
            if not group:
                continue
            standard = max(group, key=lambda x: context_counts.get(x, 0))
            for label in group:
                if label != standard:
                    suggestions['context'][label] = standard

        for group in function_groups:
            if not group:
                continue
            standard = max(group, key=lambda x: function_counts.get(x, 0))
            for label in group:
                if label != standard:
                    suggestions['function'][label] = standard

        return suggestions

    def _get_label_frequencies(self, label_type: str) -> Dict[str, int]:
        """获取标签使用频率"""
        counts = {}

        try:
            # 查询翻译表
            self.db.cursor.execute(f'''
                SELECT {label_type}, COUNT(*) as count
                FROM translations
                WHERE {label_type} IS NOT NULL AND {label_type} != ''
                GROUP BY {label_type}
            ''')

            for label, count in self.db.cursor.fetchall():
                counts[label] = count

            # 查询组合表
            self.db.cursor.execute(f'''
                SELECT {label_type}, COUNT(*) as count
                FROM term_combinations
                WHERE {label_type} IS NOT NULL AND {label_type} != ''
                GROUP BY {label_type}
            ''')

            for label, count in self.db.cursor.fetchall():
                counts[label] = counts.get(label, 0) + count

            return counts
        except Exception as e:
            logger.error(f"获取标签频率失败: {e}")
            return {}

    def _group_similar_labels(self, labels: Set[str]) -> List[Set[str]]:
        """将相似的标签分组"""
        groups = []
        remaining = set(labels)

        while remaining:
            label = next(iter(remaining))
            group = {label}
            remaining.remove(label)

            # 找出相似标签
            similar = {other for other in remaining
                       if self._is_similar_label(label, other)}

            group.update(similar)
            remaining -= similar
            groups.append(group)

        return groups

    def _is_similar_label(self, label1: str, label2: str) -> bool:
        """判断两个标签是否相似"""
        # 简单相似性：忽略大小写和空格
        norm1 = label1.lower().replace(' ', '')
        norm2 = label2.lower().replace(' ', '')

        # 完全匹配
        if norm1 == norm2:
            return True

        # 包含关系
        if norm1 in norm2 or norm2 in norm1:
            return True

        # 编辑距离（可选：使用Levenshtein距离）
        # 简化：如果两个标签有80%以上的字符相同，认为相似
        common_chars = set(norm1) & set(norm2)
        if common_chars and len(common_chars) / max(len(norm1), len(norm2)) > 0.8:
            return True

        return False