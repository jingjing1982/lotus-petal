"""
数据库同步工具 - 同步和维护术语数据库的一致性
"""
import argparse
import logging
from utils.term_database import TermDatabase
from utils.label_manager import LabelManager
from postprocessor.quality_controller import ConceptGraph

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description='术语数据库同步和一致性维护')
    parser.add_argument('--db', default='terms.db', help='术语数据库路径')
    parser.add_argument('--sync-combinations', action='store_true',
                        help='同步术语组合到概念关系')
    parser.add_argument('--sync-relations', action='store_true',
                        help='同步概念关系到术语组合')
    parser.add_argument('--unify-labels', action='store_true',
                        help='统一标签系统')
    parser.add_argument('--check-consistency', action='store_true',
                        help='检查数据一致性')
    parser.add_argument('--full-sync', action='store_true',
                        help='执行完整同步（所有操作）')

    args = parser.parse_args()

    # 初始化数据库
    db = TermDatabase(db_path=args.db)

    # 根据参数执行操作
    if args.full_sync:
        perform_full_sync(db)
    else:
        if args.sync_combinations:
            logger.info("同步术语组合到概念关系...")
            relations_added = db.sync_term_combinations_to_relations()
            logger.info(f"添加了 {relations_added} 个概念关系")

        if args.sync_relations:
            logger.info("同步概念关系到术语组合...")
            combinations_added = db.sync_relations_to_term_combinations()
            logger.info(f"添加了 {combinations_added} 个术语组合")

        if args.unify_labels:
            logger.info("统一标签系统...")
            label_manager = LabelManager(db)

            # 获取标签标准化建议
            suggestions = label_manager.suggest_label_mappings()
            logger.info(f"上下文标签建议: {suggestions['context']}")
            logger.info(f"功能标签建议: {suggestions['function']}")

            # 应用建议的映射
            label_manager.define_context_label_mapping(suggestions['context'])
            label_manager.define_function_label_mapping(suggestions['function'])

            # 执行标准化
            updates = label_manager.apply_standardization()
            logger.info(f"已更新 {sum(updates.values())} 个标签")

        if args.check_consistency:
            check_database_consistency(db)


def perform_full_sync(db):
    """执行完整的数据库同步和一致性维护"""
    logger.info("开始执行完整同步...")

    # 1. 同步概念图内存到数据库
    logger.info("同步内存中的概念关系到数据库...")
    concept_graph = ConceptGraph(db)
    sync_result = concept_graph.sync_to_database()
    if sync_result:
        logger.info("概念关系同步成功")

    # 2. 双向同步术语组合和概念关系
    logger.info("同步术语组合和概念关系...")
    consistency_result = db.ensure_consistency()
    logger.info(f"同步结果: {consistency_result}")

    # 3. 统一标签系统
    logger.info("统一标签系统...")
    label_manager = LabelManager(db)
    suggestions = label_manager.suggest_label_mappings()

    # 应用建议的映射
    label_manager.define_context_label_mapping(suggestions['context'])
    label_manager.define_function_label_mapping(suggestions['function'])
    updates = label_manager.apply_standardization()
    logger.info(f"已更新 {sum(updates.values())} 个标签")

    # 4. 检查数据一致性
    check_database_consistency(db)

    logger.info("完整同步完成")


def check_database_consistency(db):
    """检查数据库一致性"""
    logger.info("检查数据库一致性...")

    # 1. 检查标签一致性
    label_manager = LabelManager(db)
    label_issues = label_manager.check_label_consistency()
    if label_issues:
        logger.warning(f"发现 {len(label_issues)} 个标签一致性问题:")
        for issue in label_issues[:10]:  # 只显示前10个
            logger.warning(f"  - {issue}")
    else:
        logger.info("标签一致性检查通过")

    # 2. 检查概念关系一致性
    try:
        db.cursor.execute('''
                          SELECT cr.source_concept, cr.target_concept
                          FROM concept_relations cr
                          WHERE cr.relation_type = 'includes'
                            AND EXISTS (SELECT 1
                                        FROM concept_relations cr2
                                        WHERE cr2.relation_type = 'opposite'
                                          AND cr2.source_concept = cr.source_concept
                                          AND cr2.target_concept = cr.target_concept)
                          ''')

        relation_issues = db.cursor.fetchall()
        if relation_issues:
            logger.warning(f"发现 {len(relation_issues)} 个概念关系冲突:")
            for source, target in relation_issues[:10]:
                logger.warning(f"  - 概念 '{source}' 和 '{target}' 同时存在对立和包含关系")
        else:
            logger.info("概念关系一致性检查通过")
    except Exception as e:
        logger.error(f"检查概念关系一致性失败: {e}")


if __name__ == "__main__":
    main()