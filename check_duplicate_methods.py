import os
import re
from collections import defaultdict


def check_duplicates(project_dir):
    """检查项目中的重复方法定义，排除第三方库文件"""
    # 要排除的目录
    exclude_dirs = [
        '.venv', 'venv', 'env',  # 虚拟环境
        'lib', 'site-packages', 'dist',  # 库和发布目录
        '__pycache__', '.git', '.idea',  # 缓存和配置目录
        'node_modules'  # JS库
    ]

    methods_by_file = defaultdict(list)

    # 遍历所有Python文件
    for root, dirs, files in os.walk(project_dir):
        # 排除不需要的目录
        dirs[:] = [d for d in dirs if d not in exclude_dirs]

        for file in files:
            if file.endswith('.py'):
                file_path = os.path.join(root, file)

                # 跳过明显的库文件
                if any(segment in file_path for segment in exclude_dirs):
                    continue

                try:
                    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                        content = f.read()

                    # 查找所有方法定义
                    method_matches = re.finditer(r'^\s*def\s+(\w+)', content, re.MULTILINE)

                    for match in method_matches:
                        method_name = match.group(1)
                        line_number = content[:match.start()].count('\n') + 1
                        methods_by_file[file_path].append((method_name, line_number))

                except Exception as e:
                    print(f"无法读取文件 {file_path}: {e}")

    # 检查每个文件中的重复方法
    print("=== 文件内重复方法 ===")
    found_duplicates = False

    for file_path, methods in methods_by_file.items():
        method_names = [m[0] for m in methods]
        duplicates = set()

        for method_name in method_names:
            if method_names.count(method_name) > 1 and method_name not in duplicates:
                duplicates.add(method_name)

                print(f"\n文件: {file_path}")
                print(f"重复方法: {method_name}")
                print("出现位置:")
                for m, line in methods:
                    if m == method_name:
                        print(f"  第 {line} 行")

                found_duplicates = True

    if not found_duplicates:
        print("未发现文件内重复方法定义")


if __name__ == "__main__":
    project_dir = "."  # 当前目录
    check_duplicates(project_dir)