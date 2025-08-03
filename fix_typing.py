import os
import re


def add_typing_imports(file_path):
    """添加缺少的typing导入"""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # 检查文件是否已经导入typing模块
    has_typing_import = re.search(r'from\s+typing\s+import', content) is not None

    # 检查文件中使用的类型注解
    type_annotations = set(re.findall(r':\s*([A-Z][a-zA-Z]*)', content))
    type_annotations.update(re.findall(r'def\s+\w+\(\s*.*\)\s*->\s*([A-Z][a-zA-Z]*)', content))

    # 常用类型注解
    common_types = {
        'List', 'Dict', 'Tuple', 'Set', 'Optional', 'Union', 'Any',
        'Callable', 'Iterable', 'Iterator', 'Sequence', 'Mapping'
    }

    # 找出需要导入的类型
    types_to_import = type_annotations.intersection(common_types)

    if types_to_import and not has_typing_import:
        # 构建导入语句
        import_statement = f"from typing import {', '.join(sorted(types_to_import))}\n"

        # 在导入部分之后添加导入语句
        import_section_end = 0
        for match in re.finditer(r'^import\s+|^from\s+\w+\s+import', content, re.MULTILINE):
            import_section_end = max(import_section_end, match.end())

        if import_section_end > 0:
            # 在最后一个导入语句之后添加
            position = content.find('\n', import_section_end) + 1
            if position > 0:
                new_content = content[:position] + import_statement + content[position:]
            else:
                new_content = import_statement + content
        else:
            # 在文件开头添加
            new_content = import_statement + content

        # 写回文件
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(new_content)

        print(f"Added typing imports to {file_path}: {', '.join(sorted(types_to_import))}")


def process_directory(directory):
    """处理目录中的所有Python文件"""
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.py'):
                file_path = os.path.join(root, file)
                add_typing_imports(file_path)


if __name__ == "__main__":
    # 修复项目中的所有Python文件
    project_root = os.path.dirname(os.path.abspath(__file__))
    process_directory(project_root)