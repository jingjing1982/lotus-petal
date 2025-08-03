"""
概念关系管理器 - 管理佛教概念知识图谱
"""
import logging
from typing import List, Dict, Tuple, Set, Optional
from collections import defaultdict
from utils.term_database import TermDatabase

logger = logging.getLogger(__name__)


class ConceptRelationManager:
    def __init__(self, term_database: TermDatabase):
        self.db = term_database

        # 关系缓存
        self.relations = defaultdict(lambda: defaultdict(list))
        self.metadata = {}

        # 加载关系
        self._load_relations()

    def _load_relations(self):
        """从数据库加载关系"""
        try:
            relations = self.db.get_concept_relations()

            for relation in relations:
                rel_id, rel_type, source, target, bidirectional, confidence, source_type, reference, created_at = relation

                # 添加到缓存
                self.relations[rel_type][source].append({
                    'id': rel_id,
                    'target': target,
                    'bidirectional': bidirectional,
                    'confidence': confidence,
                    'source_type': source_type
                })

                # 如果是双向关系，也添加反向映射
                if bidirectional:
                    self.relations[rel_type][target].append({
                        'id': rel_id,
                        'target': source,
                        'bidirectional': bidirectional,
                        'confidence': confidence,
                        'source_type': source_type
                    })

            logger.info(f"已加载 {len(relations)} 个概念关系")
        except Exception as e:
            logger.error(f"加载概念关系失败: {e}")

    def get_relations(self, concept: str, relation_type: Optional[str] = None) -> List[Dict]:
        """获取概念的关系"""
        if relation_type:
            return self.relations[relation_type].get(concept, [])

        # 返回所有类型的关系
        results = []
        for rel_type, rel_map in self.relations.items():
            results.extend(rel_map.get(concept, []))

        return results

    def add_relation(self, relation_type: str, source: str, target: str,
                     bidirectional: bool = False, confidence: float = 1.0,
                     source_type: str = 'manual', reference_text: str = None):
        """添加新关系"""
        # 检查是否已存在相同关系
        existing = self.check_relation_exists(relation_type, source, target)

        if existing:
            logger.info(f"关系已存在: {relation_type} - {source} -> {target}")
            return existing

        # 添加到数据库
        success = self.db.add_concept_relation(
            relation_type, source, target, bidirectional,
            confidence, source_type, reference_text
        )

        if success:
            # 刷新缓存
            self._load_relations()

            # 返回新添加的关系ID
            relations = self.db.get_concept_relations(source, relation_type)
            for rel in relations:
                if rel[2] == source and rel[3] == target and rel[1] == relation_type:
                    return rel[0]

        return None

    def check_relation_exists(self, relation_type: str, source: str, target: str) -> Optional[int]:
        """检查关系是否已存在"""
        for rel in self.relations[relation_type].get(source, []):
            if rel['target'] == target:
                return rel['id']
        return None

    def delete_relation(self, relation_id: int):
        """删除关系"""
        success = self.db.delete_concept_relation(relation_id)
        if success:
            # 刷新缓存
            self._load_relations()
            return True
        return False

    def get_relation_path(self, concept1: str, concept2: str, max_depth: int = 3) -> List[List[Dict]]:
        """查找两个概念之间的关系路径"""
        paths = []
        visited = set()

        def dfs(current, target, path, depth):
            if depth > max_depth:
                return

            if current == target:
                paths.append(path.copy())
                return

            visited.add(current)

            for rel_type, rel_map in self.relations.items():
                for rel in rel_map.get(current, []):
                    next_concept = rel['target']
                    if next_concept not in visited:
                        path.append({
                            'type': rel_type,
                            'source': current,
                            'target': next_concept,
                            'confidence': rel['confidence']
                        })
                        dfs(next_concept, target, path, depth + 1)
                        path.pop()

            visited.remove(current)

        dfs(concept1, concept2, [], 0)
        return paths

    def check_consistency(self) -> List[Dict]:
        """检查知识图谱的一致性问题"""
        issues = []

        # 检查冲突关系
        for source, targets in self.relations['opposite'].items():
            for rel in targets:
                target = rel['target']

                # 检查是否同时存在包含关系
                if any(r['target'] == target for r in self.relations['includes'].get(source, [])):
                    issues.append({
                        'type': 'conflict',
                        'description': f"概念 '{source}' 和 '{target}' 同时有对立和包含关系",
                        'concepts': [source, target],
                        'relations': ['opposite', 'includes']
                    })

        # 检查循环依赖
        for source, targets in self.relations['stage'].items():
            for rel in targets:
                target = rel['target']

                # 检查是否有反向依赖
                if any(r['target'] == source for r in self.relations['stage'].get(target, [])):
                    issues.append({
                        'type': 'cycle',
                        'description': f"概念 '{source}' 和 '{target}' 之间存在循环阶段依赖",
                        'concepts': [source, target],
                        'relations': ['stage']
                    })

        return issues