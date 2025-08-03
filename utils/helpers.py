"""
辅助工具函数
"""
import re
import unicodedata
from typing import List, Tuple, Optional, Dict
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


# 文本处理工具
class TextUtils:
    @staticmethod
    def normalize_tibetan(text: str) -> str:
        """规范化藏文文本"""
        # 规范化Unicode
        text = unicodedata.normalize('NFC', text)

        # 替换常见的变体字符
        replacements = {
            '༌': '་',  # 统一使用标准的音节分隔符
            '༎': '།',  # 统一句号
            '༏': '།།',  # 双句号
        }

        for old, new in replacements.items():
            text = text.replace(old, new)

        # 删除多余的空格
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'་\s+་', '་', text)  # 修复分隔符间的空格

        return text.strip()

    @staticmethod
    def is_tibetan(text: str) -> bool:
        """检查是否为藏文文本"""
        # 藏文Unicode范围：U+0F00-U+0FFF
        tibetan_pattern = re.compile(r'[\u0F00-\u0FFF]')
        tibetan_chars = len(tibetan_pattern.findall(text))
        total_chars = len(text.replace(' ', ''))

        # 如果超过50%是藏文字符，认为是藏文
        return tibetan_chars / total_chars > 0.5 if total_chars > 0 else False

    @staticmethod
    def split_mixed_text(text: str) -> List[Tuple[str, str]]:
        """
        分割混合文本（藏文和其他语言混合）
        返回：[(text, language), ...]
        """
        segments = []
        current_segment = ""
        current_lang = None

        for char in text:
            if '\u0F00' <= char <= '\u0FFF':
                # 藏文字符
                if current_lang != 'tibetan':
                    if current_segment:
                        segments.append((current_segment, current_lang))
                    current_segment = char
                    current_lang = 'tibetan'
                else:
                    current_segment += char
            elif '\u4E00' <= char <= '\u9FFF':
                # 中文字符
                if current_lang != 'chinese':
                    if current_segment:
                        segments.append((current_segment, current_lang))
                    current_segment = char
                    current_lang = 'chinese'
                else:
                    current_segment += char
            else:
                # 其他字符（包括标点、英文等）
                if current_lang != 'other':
                    if current_segment:
                        segments.append((current_segment, current_lang))
                    current_segment = char
                    current_lang = 'other'
                else:
                    current_segment += char

        # 添加最后一个段
        if current_segment:
            segments.append((current_segment, current_lang))

        return segments

    @staticmethod
    def clean_translation(text: str) -> str:
        """清理翻译文本"""
        # 删除可能的控制字符
        text = ''.join(char for char in text if not unicodedata.category(char).startswith('C'))

        # 修复标点符号周围的空格
        text = re.sub(r'\s+([。！？，、；：])', r'\1', text)
        text = re.sub(r'([。！？])\s*', r'\1 ', text)

        # 删除重复的标点
        text = re.sub(r'([。！？，])\1+', r'\1', text)

        # 规范化引号
        text = text.replace('"', '"').replace('"', '"')
        text = text.replace(''', ''').replace(''', ''')

        return text.strip()

    @staticmethod
    def estimate_similarity(text1: str, text2: str) -> float:
        """估算两个文本的相似度（简单版本）"""
        # 使用Jaccard相似度
        set1 = set(text1)
        set2 = set(text2)

        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))

        return intersection / union if union > 0 else 0.0


# 文件处理工具
class FileUtils:
    @staticmethod
    def read_text_file(filepath: Path, encoding: str = 'utf-8') -> str:
        """读取文本文件"""
        try:
            with open(filepath, 'r', encoding=encoding) as f:
                return f.read()
        except UnicodeDecodeError:
            # 尝试其他编码
            for enc in ['utf-16', 'gb18030', 'big5']:
                try:
                    with open(filepath, 'r', encoding=enc) as f:
                        return f.read()
                except:
                    continue
            raise ValueError(f"Cannot decode file {filepath}")

    @staticmethod
    def write_text_file(filepath: Path, content: str, encoding: str = 'utf-8'):
        """写入文本文件"""
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'w', encoding=encoding) as f:
            f.write(content)

    @staticmethod
    def process_directory(input_dir: Path, output_dir: Path,
                          processor_func, file_pattern: str = '*.txt'):
        """批量处理目录中的文件"""
        input_files = list(input_dir.glob(file_pattern))

        for input_file in input_files:
            try:
                # 读取输入文件
                content = FileUtils.read_text_file(input_file)

                # 处理内容
                processed = processor_func(content)

                # 构建输出路径
                relative_path = input_file.relative_to(input_dir)
                output_file = output_dir / relative_path

                # 写入输出文件
                FileUtils.write_text_file(output_file, processed)

                logger.info(f"Processed: {input_file} -> {output_file}")

            except Exception as e:
                logger.error(f"Failed to process {input_file}: {e}")


# 验证工具
class ValidationUtils:
    @staticmethod
    def validate_translation_pair(source: str, translation: str) -> Dict[str, bool]:
        """验证翻译对的基本质量"""
        checks = {
            'source_not_empty': bool(source.strip()),
            'translation_not_empty': bool(translation.strip()),
            'source_is_tibetan': TextUtils.is_tibetan(source),
            'translation_is_chinese': ValidationUtils._is_chinese(translation),
            'reasonable_length_ratio': ValidationUtils._check_length_ratio(source, translation),
            'no_untranslated_terms': ValidationUtils._check_untranslated(translation),
        }

        return checks

    @staticmethod
    def _is_chinese(text: str) -> bool:
        """检查是否主要是中文"""
        chinese_pattern = re.compile(r'[\u4E00-\u9FFF]')
        chinese_chars = len(chinese_pattern.findall(text))
        total_chars = len(text.replace(' ', ''))

        return chinese_chars / total_chars > 0.5 if total_chars > 0 else False

    @staticmethod
    def _check_length_ratio(source: str, translation: str) -> bool:
        """检查长度比例是否合理"""
        source_len = len(source.strip())
        trans_len = len(translation.strip())

        if source_len == 0:
            return False

        ratio = trans_len / source_len
        # 藏文到中文的合理比例范围
        return 0.3 <= ratio <= 2.0

    @staticmethod
    def _check_untranslated(translation: str) -> bool:
        """检查是否有未翻译的内容"""
        # 检查常见的未翻译标记
        untranslated_patterns = [
            r'<extra_id_\d+>',  # MT5的特殊标记
            r'TERM\d{3}',  # 未还原的术语占位符
            r'\[.*?\]',  # 可能的标记
        ]

        for pattern in untranslated_patterns:
            if re.search(pattern, translation):
                return False

        return True


# 统计工具
class StatisticsUtils:
    @staticmethod
    def calculate_translation_stats(translations: List[Dict]) -> Dict:
        """计算翻译统计信息"""
        total = len(translations)
        if total == 0:
            return {}

        stats = {
            'total_count': total,
            'average_source_length': sum(len(t['source']) for t in translations) / total,
            'average_translation_length': sum(len(t['translation']) for t in translations) / total,
            'average_quality_score': sum(t.get('quality_score', 0) for t in translations) / total,
            'success_rate': sum(1 for t in translations if t.get('translation')) / total,
        }

        # 质量分布
        quality_distribution = {
            'excellent': 0,  # >= 0.9
            'good': 0,  # >= 0.7
            'fair': 0,  # >= 0.5
            'poor': 0  # < 0.5
        }

        for t in translations:
            score = t.get('quality_score', 0)
            if score >= 0.9:
                quality_distribution['excellent'] += 1
            elif score >= 0.7:
                quality_distribution['good'] += 1
            elif score >= 0.5:
                quality_distribution['fair'] += 1
            else:
                quality_distribution['poor'] += 1

        stats['quality_distribution'] = quality_distribution

        return stats


# 导出工具类实例
text_utils = TextUtils()
file_utils = FileUtils()
validation_utils = ValidationUtils()
statistics_utils = StatisticsUtils()