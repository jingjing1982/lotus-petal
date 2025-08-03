import pandas as pd
import sqlite3
import json
from pathlib import Path
import re
from datetime import datetime

# 尝试导入botok用于藏文分词
try:
    import botok

    HAS_BOTOK = True
except ImportError:
    print("警告: 未找到botok库，将使用简单分词规则。如需高级分词请安装: pip install botok")
    HAS_BOTOK = False


def import_terms_from_excel(excel_path, db_path):
    """从Excel导入术语到数据库"""
    print(f"开始导入术语数据：从 {excel_path} 到 {db_path}")

    # 连接数据库
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # 读取Excel，跳过前两行标题
    df = pd.read_excel(excel_path, skiprows=2)

    # 重命名列以便处理
    df.columns = ["序号", "藏文术语", "汉文术语", "书名"] + list(df.columns)[4:]

    # 初始化botok分词器（如果可用）
    wt = None
    if HAS_BOTOK:
        wt = botok.WordTokenizer()

    # 统计信息
    stats = {
        "terms_added": 0,
        "terms_existing": 0,
        "translations_added": 0,
        "combinations_detected": 0,
        "relations_added": 0,
        "context_labels_added": 0
    }

    # 处理每一行数据
    for _, row in df.iterrows():
        if pd.isna(row["藏文术语"]) or pd.isna(row["汉文术语"]):
            continue

        tibetan = row["藏文术语"].strip()
        chinese_terms = process_chinese_terms(row["汉文术语"])
        book_source = row["书名"] if not pd.isna(row["书名"]) else "未知来源"

        # 根据书名识别语境标签
        context_label = detect_context_from_book(book_source)

        # 1. 添加到terms表
        term_id = add_or_get_term(cursor, tibetan)
        if term_id == "new":
            stats["terms_added"] += 1
        else:
            stats["terms_existing"] += 1

        # 2. 添加翻译到translations表（带语境标签）
        for i, chinese in enumerate(chinese_terms):
            if add_translation(cursor, term_id, chinese, book_source, context_label, i == 0):
                stats["translations_added"] += 1
                if context_label:
                    stats["context_labels_added"] += 1

        # 3. 检测并处理术语组合（使用botok或启发式方法）
        if is_combination_term(tibetan):
            components = detect_components(tibetan, cursor, wt)
            if components and len(components) > 1:
                add_term_combination(cursor, tibetan, chinese_terms[0], components)
                stats["combinations_detected"] += 1

                # 4. 添加相应的概念关系
                for component in components:
                    add_concept_relation(cursor, tibetan, component)
                    stats["relations_added"] += 1

    # 提交所有更改
    conn.commit()

    # 输出统计信息
    print(f"\n导入完成！统计信息：")
    print(f"- 添加新术语：{stats['terms_added']}个")
    print(f"- 已存在术语：{stats['terms_existing']}个")
    print(f"- 添加翻译：{stats['translations_added']}个")
    print(f"- 添加语境标签：{stats['context_labels_added']}个")
    print(f"- 检测到组合术语：{stats['combinations_detected']}个")
    print(f"- 添加概念关系：{stats['relations_added']}个")

    conn.close()
    return stats


def detect_context_from_book(book_name):
    """根据书名判断术语的语境"""
    if not book_name or not isinstance(book_name, str):
        return None

    # 大圆满语境关键词 (优先级最高)
    dzogchen_keywords = [
        '大圆满', '心性休息', '上师心滴', '虚幻休息', '大圆胜慧',
        '文殊大圆满', '佛一子续', '大鹏展翅', '禅定休息', '教言合集'
    ]

    # 戒律语境关键词
    vinaya_keywords = [
        '戒论', '三戒', '律藏', '戒学', '别解脱', '比丘尼', '比丘戒',
        '沙弥', '律仪', '毗尼'
    ]

    # 密乘语境关键词
    tantra_keywords = [
        '密宗', '密乘', '莲师', '成就者', '藏密', '金刚', '灌顶',
        '敦珠', '色拉康卓', '大成就', '密教', '坛城'
    ]

    # 按优先级检查 (大圆满 > 密乘 > 戒律)
    for keyword in dzogchen_keywords:
        if keyword in book_name:
            return "DZOGCHEN"  # 大圆满语境

    for keyword in tantra_keywords:
        if keyword in book_name:
            return "TANTRA"  # 密乘语境

    for keyword in vinaya_keywords:
        if keyword in book_name:
            return "VINAYA"  # 戒律语境

    return None  # 无法确定语境


def process_chinese_terms(chinese_str):
    """处理可能包含多个翻译的汉文术语"""
    if not isinstance(chinese_str, str):
        return [""]

    # 分隔符可能包括：顿号、分号、逗号、斜杠等
    separators = ['、', '；', ';', '，', ',', '/', '|']

    for sep in separators:
        if sep in chinese_str:
            terms = [term.strip() for term in chinese_str.split(sep) if term.strip()]
            if terms:
                return terms

    return [chinese_str.strip()]


def add_or_get_term(cursor, tibetan):
    """添加藏文术语到terms表，如果已存在则返回ID"""
    # 检查术语是否已存在
    cursor.execute("SELECT id FROM terms WHERE tibetan = ?", (tibetan,))
    result = cursor.fetchone()

    if result:
        return result[0]  # 返回已存在术语的ID

    # 添加新术语
    cursor.execute("""
                   INSERT INTO terms (tibetan, ambiguity_level, frequency)
                   VALUES (?, 1, 1)
                   """, (tibetan,))

    return cursor.lastrowid  # 返回新插入的ID


def add_translation(cursor, term_id, chinese, reference_source, context_label=None, is_primary=False):
    """添加翻译到translations表（支持语境标签）"""
    try:
        # 检查是否已存在相同翻译和语境
        cursor.execute("""
                       SELECT id
                       FROM translations
                       WHERE term_id = ?
                         AND chinese = ?
                         AND (context_label = ? OR (context_label IS NULL AND ? IS NULL))
                       """, (term_id, chinese, context_label, context_label))

        result = cursor.fetchone()
        if result:
            # 更新使用次数和参考源
            cursor.execute("""
                           UPDATE translations
                           SET usage_count      = usage_count + 1,
                               reference_source = CASE
                                                      WHEN reference_source IS NULL THEN ?
                                                      WHEN reference_source NOT LIKE '%' || ? || '%'
                                                          THEN reference_source || '; ' || ?
                                                      ELSE reference_source
                                   END
                           WHERE id = ?
                           """, (reference_source, reference_source, reference_source, result[0]))
            return False

        # 添加新翻译（带语境标签）
        cursor.execute("""
                       INSERT INTO translations
                       (term_id, chinese, context_label, priority, is_primary, reference_source, usage_count,
                        is_direct_translation)
                       VALUES (?, ?, ?, ?, ?, ?, 1, 1)
                       """, (term_id, chinese, context_label, 5, is_primary, reference_source))
        return True

    except Exception as e:
        print(f"添加翻译失败：{chinese} - {e}")
        return False


def is_combination_term(tibetan):
    """判断是否为组合术语（简单启发式方法）"""
    # 判断依据：长度超过一定字符、包含特定分隔符等
    if len(tibetan) > 10:  # 假设超过10个字符可能是组合术语
        return True

    # 检查是否包含空格或藏文分隔符
    if ' ' in tibetan or '་' in tibetan:
        return True

    return False


def detect_components(tibetan, cursor, wt=None):
    """检测组合术语的组成部分（使用botok或启发式方法）"""
    components = []

    # 方法1: 使用botok分词（如果可用）
    if wt and HAS_BOTOK:
        try:
            tokens = wt.tokenize(tibetan)
            botok_tokens = [token.text for token in tokens if token.pos != 'PART' and len(token.text) > 1]
            if len(botok_tokens) > 1:
                return botok_tokens
        except Exception as e:
            print(f"Botok分词失败: {e}")

    # 方法2: 查询数据库中是否有这个术语的部分
    cursor.execute("SELECT tibetan FROM terms WHERE ? LIKE '%' || tibetan || '%' AND tibetan != ?", (tibetan, tibetan))
    potential_components = cursor.fetchall()

    for comp in potential_components:
        components.append(comp[0])

    # 方法3: 简单的分割尝试（基于藏文语法特点）
    if not components:
        # 按藏文分隔符和空格分割
        parts = re.split(r'་|\s+', tibetan)
        valid_parts = [part for part in parts if len(part) > 1]  # 过滤掉太短的部分
        if len(valid_parts) > 1:
            components = valid_parts

    return components


def add_term_combination(cursor, combined_tibetan, combined_chinese, components):
    """添加术语组合"""
    try:
        # 检查是否已存在
        cursor.execute("SELECT id FROM term_combinations WHERE combined_tibetan = ?", (combined_tibetan,))
        if cursor.fetchone():
            return False

        # 组件列表转JSON
        components_json = json.dumps(components, ensure_ascii=False)

        # 添加组合术语
        cursor.execute("""
                       INSERT INTO term_combinations
                           (combined_tibetan, combined_chinese, component_terms, formation_type)
                       VALUES (?, ?, ?, 'auto_detected')
                       """, (combined_tibetan, combined_chinese, components_json))

        return True

    except Exception as e:
        print(f"添加术语组合失败：{combined_tibetan} - {e}")
        return False


def add_concept_relation(cursor, source, target):
    """添加概念包含关系"""
    try:
        # 检查关系是否已存在
        cursor.execute("""
                       SELECT id
                       FROM concept_relations
                       WHERE relation_type = 'includes'
                         AND source_concept = ?
                         AND target_concept = ?
                       """, (source, target))

        if cursor.fetchone():
            return False

        # 添加包含关系
        cursor.execute("""
                       INSERT INTO concept_relations
                       (relation_type, source_concept, target_concept, bidirectional, confidence, source_type)
                       VALUES ('includes', ?, ?, 0, 0.7, 'auto_imported')
                       """, (source, target))

        return True

    except Exception as e:
        print(f"添加概念关系失败：{source} 包含 {target} - {e}")
        return False


if __name__ == "__main__":
    # 使用示例
    excel_path = "术语表.xlsx"
    db_path = "terms.db"

    import_terms_from_excel(excel_path, db_path)