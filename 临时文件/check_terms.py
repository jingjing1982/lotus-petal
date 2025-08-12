#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""检查术语数据库中是否存在特定术语"""

import sqlite3
import sys
from pathlib import Path


def check_term(db_path, tibetan_term):
    """检查术语是否存在于数据库中"""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # 检查原始术语
    cursor.execute('SELECT id, tibetan FROM terms WHERE tibetan = ?', (tibetan_term,))
    result = cursor.fetchone()

    # 如果没找到，检查去掉尾部的点（་）的版本
    if not result and tibetan_term.endswith('་'):
        clean_term = tibetan_term.rstrip('་')
        cursor.execute('SELECT id, tibetan FROM terms WHERE tibetan = ?', (clean_term,))
        result = cursor.fetchone()
        if result:
            print(f"找到术语(去掉尾部点): ID={result[0]}, 藏文={result[1]}")

            # 获取翻译
            cursor.execute('''
                           SELECT chinese, context_label, function_label, priority, is_primary
                           FROM translations
                           WHERE term_id = ?
                           ORDER BY priority DESC
                           ''', (result[0],))

            translations = cursor.fetchall()
            if translations:
                print("\n可用翻译:")
                for i, trans in enumerate(translations, 1):
                    primary = "主要" if trans[4] else "备选"
                    print(f"  {i}. {trans[0]} (上下文: {trans[1]}, 功能: {trans[2]}, 优先级: {trans[3]}, {primary})")
            else:
                print("\n没有找到翻译")
            return True

    # 如果找到了原始术语
    if result:
        print(f"找到术语: ID={result[0]}, 藏文={result[1]}")

        # 获取翻译
        cursor.execute('''
                       SELECT chinese, context_label, function_label, priority, is_primary
                       FROM translations
                       WHERE term_id = ?
                       ORDER BY priority DESC
                       ''', (result[0],))

        translations = cursor.fetchall()
        if translations:
            print("\n可用翻译:")
            for i, trans in enumerate(translations, 1):
                primary = "主要" if trans[4] else "备选"
                print(f"  {i}. {trans[0]} (上下文: {trans[1]}, 功能: {trans[2]}, 优先级: {trans[3]}, {primary})")
        else:
            print("\n没有找到翻译")
        return True

    print(f"数据库中不存在术语: '{tibetan_term}'")
    return False


def main():
    # 默认数据库路径
    db_path = "../terms.db"

    if len(sys.argv) > 1:
        # 如果提供了术语作为命令行参数
        term = sys.argv[1]
    else:
        # 否则检查常见问题术语
        terms_to_check = ["སངས་རྒྱས", "ཆོས", "ཆོས་", "གསུངས", "ཀྱིས"]
        print(f"检查数据库: {db_path}\n")

        found_terms = []
        missing_terms = []

        for term in terms_to_check:
            print(f"检查术语: '{term}'")
            print("-" * 40)
            if check_term(db_path, term):
                found_terms.append(term)
            else:
                missing_terms.append(term)
            print("\n")

        print("=" * 50)
        print(f"结果总结: 共检查 {len(terms_to_check)} 个术语")
        print(f"找到: {len(found_terms)} 个 - {', '.join(found_terms)}")
        print(f"缺失: {len(missing_terms)} 个 - {', '.join(missing_terms)}")
        return

    # 检查单个术语
    check_term(db_path, term)


if __name__ == "__main__":
    main()