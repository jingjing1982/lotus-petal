"""
文本终结器 - 最终的文本处理和格式化
"""
import re
from typing import Dict, List, Optional
import logging
from .adapter import ProcessingAdapter

logger = logging.getLogger(__name__)


class TextFinalizer:
    def __init__(self):
        """初始化文本终结器"""
        # 标点符号规范化规则
        self.punctuation_rules = [
            # 中文标点规范化
            (r'([。！？])([^\s。！？\n])', r'\1 \2'),  # 句号后加空格
            (r'\s+([。！？，、；：])', r'\1'),  # 标点前不要空格
            (r'([，、；：])\s*', r'\1'),  # 标点后不要空格
            (r'。\s*。', '。'),  # 删除重复句号
            (r'，\s*，', '，'),  # 删除重复逗号
            (r'([。！？])\s*\n\s*', r'\1\n'),  # 段落结尾规范化
        ]

        # 引号规范化
        self.quote_rules = [
            (r'"([^"]+)"', r'"\1"'),  # 英文引号转中文
            (r"'([^']+)'", r"'\1'"),  # 英文单引号转中文
            (r'「([^」]+)」', r'"\1"'),  # 日式引号转中文
            (r'『([^』]+)』', r"'\1'"),  # 日式单引号转中文
        ]

    def finalize(self, text: str, context) -> str:
        """执行最终处理"""
        # 1. 标点符号规范化
        text = self._normalize_punctuation(text)

        # 2. 根据文本类型格式化
        is_verse = ProcessingAdapter.is_verse(context)
        has_enumeration = ProcessingAdapter.has_enumeration(context)

        if is_verse:
            text = self._format_verse(text, context)
        elif has_enumeration:
            text = self._format_enumeration(text, context)
        else:
            text = self._beautify_format(text, context)

        # 3. 特殊格式处理
        text = self._handle_special_formats(text, context)

        # 4. 最终检查
        text = self._final_check(text)

        return text.strip()

    def _normalize_punctuation(self, text: str) -> str:
        """规范化标点符号"""
        # 应用标点规则
        for pattern, replacement in self.punctuation_rules:
            text = re.sub(pattern, replacement, text)

        # 应用引号规则
        for pattern, replacement in self.quote_rules:
            text = re.sub(pattern, replacement, text)

        # 处理特殊标点
        text = self._handle_special_punctuation(text)

        return text

    def _handle_special_punctuation(self, text: str) -> str:
        """处理特殊标点符号"""
        # 处理省略号
        text = re.sub(r'\.{3,}', '……', text)
        text = re.sub(r'。{3,}', '……', text)

        # 处理破折号
        text = re.sub(r'--+', '——', text)
        text = re.sub(r'—', '——', text)

        # 处理书名号
        text = re.sub(r'《\s+', '《', text)
        text = re.sub(r'\s+》', '》', text)

        # 处理括号
        text = re.sub(r'\(\s+', '（', text)
        text = re.sub(r'\s+\)', '）', text)
        text = re.sub(r'\(', '（', text)
        text = re.sub(r'\)', '）', text)

        return text

    def _beautify_format(self, text: str, context) -> str:
        """
        美化文本格式，处理普通散文文本

        Args:
            text: 原始文本
            context: 上下文信息

        Returns:
            格式化后的文本
        """
        from .adapter import ProcessingAdapter

        # 检查文本类型
        is_verse = ProcessingAdapter.is_verse(context)
        has_enumeration = ProcessingAdapter.has_enumeration(context)
        has_parallel = ProcessingAdapter.has_parallel_structure(context)

        # 如果是特殊类型，跳过普通美化
        if is_verse or has_enumeration:
            return text

        # 获取佛教语境
        buddhist_context = ProcessingAdapter.get_buddhist_context(context)

        # 1. 标准化段落
        text = self._normalize_paragraphs(text)

        # 2. 处理引用
        text = self._format_quotations(text, buddhist_context)

        # 3. 处理平行结构
        if has_parallel:
            text = self._format_parallel_structure(text, context)

        # 4. 处理特殊佛教文本格式
        text = self._apply_buddhist_text_format(text, buddhist_context)

        return text

    def _normalize_paragraphs(self, text: str) -> str:
        """标准化段落格式"""
        # 移除多余空行
        text = re.sub(r'\n{3,}', '\n\n', text)

        # 确保段落间有空行
        paragraphs = [p.strip() for p in text.split('\n\n')]
        paragraphs = [p for p in paragraphs if p]

        # 检查是否有太长的段落（超过100字）
        formatted_paragraphs = []
        for paragraph in paragraphs:
            if len(paragraph) > 100 and '。' in paragraph:
                # 尝试在自然的句子边界分割长段落
                sentences = []
                current = ''

                for char in paragraph:
                    current += char
                    if char in '。！？':
                        sentences.append(current)
                        current = ''

                if current:
                    sentences.append(current)

                # 重组为适当长度的段落
                new_paragraphs = []
                temp = ''

                for sentence in sentences:
                    if len(temp) + len(sentence) > 100:
                        new_paragraphs.append(temp)
                        temp = sentence
                    else:
                        temp += sentence

                if temp:
                    new_paragraphs.append(temp)

                formatted_paragraphs.extend(new_paragraphs)
            else:
                formatted_paragraphs.append(paragraph)

        return '\n\n'.join(formatted_paragraphs)

    def _format_quotations(self, text: str, buddhist_context: str) -> str:
        """格式化文本中的引用"""
        # 识别并格式化经文引用
        quotation_patterns = [
            (r'「([^」]+)」', r'「\1」'),  # 保留原有引号格式
            (r'"([^"]+)"', r'「\1」'),  # 将西式引号转为中式引号
            (r'\'([^\']+)\'', r'「\1」'),  # 将单引号转为中式引号
            (r'经云[：:"]([^"」]+)[」"]', r'经云：「\1」'),  # 标准化经文引用
            (r'如是说[：:"]([^"」]+)[」"]', r'如是说：「\1」'),  # 标准化佛说引用
            (r'佛言[：:"]([^"」]+)[」"]', r'佛言：「\1」')  # 标准化佛言引用
        ]

        for pattern, replacement in quotation_patterns:
            text = re.sub(pattern, replacement, text)

        # 处理偈颂引用
        verse_markers = ['偈言', '偈曰', '颂曰', '偈']
        for marker in verse_markers:
            pattern = rf"{marker}[：:]([^\"」]+)[」\"]"
            if re.search(pattern, text):
                text = re.sub(pattern, f"{marker}：\n「\\1」", text)

        return text

    def _format_parallel_structure(self, text: str, context) -> str:
        """格式化平行结构"""
        from .adapter import ProcessingAdapter

        # 获取平行结构信息
        parallel_structure = ProcessingAdapter.extract_info_from_context(
            context, 'parallel_structure', {}
        )

        if not parallel_structure:
            return text

        patterns = parallel_structure.get('patterns', [])

        for pattern in patterns:
            pattern_type = pattern.get('type')

            if pattern_type == 'negation':
                # 处理否定并列结构：统一格式
                positions = pattern.get('positions', [])
                if len(positions) >= 3:
                    # 尝试识别文本中的位置
                    tokens = [pos.get('token', '') for pos in positions]
                    for i in range(len(tokens) - 2):
                        pattern_text = f"{tokens[i]}.*?{tokens[i + 1]}.*?{tokens[i + 2]}"
                        match = re.search(pattern_text, text)
                        if match:
                            matched_text = match.group(0)
                            formatted_text = matched_text.replace('，', '；').replace('。', '；')
                            formatted_text = formatted_text.rstrip('；') + '。'
                            text = text.replace(matched_text, formatted_text)

            elif pattern_type == 'conjunction':
                # 处理连词并列结构
                positions = pattern.get('positions', [])
                if len(positions) >= 2:
                    # 尝试识别文本中的位置
                    conjunctions = [pos.get('token', '') for pos in positions]
                    for i in range(len(conjunctions) - 1):
                        pattern_text = f"{conjunctions[i]}.*?{conjunctions[i + 1]}"
                        match = re.search(pattern_text, text)
                        if match:
                            matched_text = match.group(0)
                            # 使用顿号替换逗号
                            formatted_text = matched_text.replace('，', '、')
                            text = text.replace(matched_text, formatted_text)

        return text

    def _apply_buddhist_text_format(self, text: str, buddhist_context: str) -> str:
        """应用佛教文本特有的格式"""
        # 处理经文开头 - 这是标准格式，不是过度修饰
        if text.startswith('如是我闻'):
            # 确保"如是我闻"后有合适的标点
            if not any(text[4:].startswith(p) for p in ['：', ':', '，', '。']):
                text = '如是我闻：' + text[4:]

        # 谨慎添加梵文，只在特定条件下添加一次
        if buddhist_context in ['MADHYAMIKA', 'YOGACARA', 'VAJRAYANA']:
            # 检查文本长度，只在较长文本中添加注解
            if len(text) > 100:
                # 检查是否已有梵文注解
                has_sanskrit = re.search(r'（.*?[a-zA-Z].*?）', text) is not None

                # 如果没有任何梵文注解，可以添加一个核心术语的注解
                if not has_sanskrit:
                    key_terms = {
                        'MADHYAMIKA': [('缘起性空', 'pratītyasamutpāda-śūnyatā')],
                        'YOGACARA': [('三性', 'trisvabhāva')],
                        'VAJRAYANA': [('金刚乘', 'vajrayāna')]
                    }

                    for term, sanskrit in key_terms.get(buddhist_context, []):
                        if term in text and f"（{sanskrit}）" not in text:
                            # 只添加一次，并且只添加核心术语
                            text = text.replace(term, f"{term}（{sanskrit}）", 1)
                            break

        return text

    def _handle_special_formats(self, text: str, context) -> str:
        """
        处理各种特殊格式

        Args:
            text: 要处理的文本
            context: 上下文信息

        Returns:
            处理后的文本
        """
        from .adapter import ProcessingAdapter

        # 获取上下文信息
        buddhist_context = ProcessingAdapter.get_buddhist_context(context)
        function_type = ProcessingAdapter.extract_info_from_context(context, 'function_type', '')

        # 1. 处理佛经标题
        if function_type == 'title':
            text = self._format_title(text, buddhist_context)

        # 2. 处理佛经结尾
        if text.endswith('作礼而去') or text.endswith('信受奉行'):
            text = text.replace('作礼而去', '作礼而去。')
            text = text.replace('信受奉行', '信受奉行。')

        # 3. 处理梵文转写
        text = self._format_sanskrit(text, buddhist_context)

        # 4. 处理数字格式
        text = self._format_numbers(text)

        # 5. 处理引用格式 (来自第二个方法)
        text = self._format_quotations(text, buddhist_context)

        # 6. 处理列表格式
        text = self._format_lists(text)

        # 7. 处理标题格式
        text = self._format_titles(text)

        # 8. 处理佛经特殊格式
        text = self._format_sutra_conventions(text, context)

        return text

    def _format_title(self, text: str, buddhist_context: str) -> str:
        """格式化佛经标题"""
        # 经名通常包含以下部分：[传统][语言]佛说[名称][类型]
        if '经' in text and not text.endswith('经'):
            text = text.replace('经', '经》')
            if not text.startswith('《'):
                text = '《' + text
        elif text.endswith('经'):
            if not text.startswith('《'):
                text = '《' + text + '》'
            else:
                text = text + '》'

        # 添加汉译者信息（如果有）
        translator_patterns = [
            r'(三藏法师|法师|大师|尊者|和尚|阿阇黎)([\u4e00-\u9fff]{1,6})译',
            r'([\u4e00-\u9fff]{1,6})(译师|法师)译'
        ]

        for pattern in translator_patterns:
            if re.search(pattern, text):
                if not text.endswith('译'):
                    text = text.replace('译', '译》')
                    if not text.startswith('《'):
                        text = '《' + text
                break

        return text

    def _format_sanskrit(self, text: str, buddhist_context: str) -> str:
        """格式化梵文转写"""
        # 处理常见的梵文转写
        sanskrit_patterns = [
            (r'([a-zA-Z]+\-[a-zA-Z]+\-[a-zA-Z]+)', r'（\1）'),  # 多连字符梵文
            (r'([a-zA-Z]{7,})', r'（\1）')  # 较长的单词可能是梵文转写
        ]

        for pattern, replacement in sanskrit_patterns:
            matches = re.finditer(pattern, text)
            offset = 0

            for match in matches:
                # 检查匹配内容是否已在括号内
                start, end = match.span()
                start += offset
                end += offset

                # 获取前后字符
                pre_char = text[start - 1] if start > 0 else ''
                post_char = text[end] if end < len(text) else ''

                # 只有当不在括号内时才添加括号
                if pre_char != '（' and post_char != '）':
                    matched_text = text[start:end]
                    replaced_text = f"（{matched_text}）"
                    text = text[:start] + replaced_text + text[end:]
                    offset += len(replaced_text) - len(matched_text)

        return text

    def _format_numbers(self, text: str) -> str:
        """格式化数字"""

        # 将阿拉伯数字替换为中文数字（在特定上下文中）
        def replace_arabic_to_chinese(match):
            num = int(match.group(0))
            if 1 <= num <= 10:
                chinese_nums = ['一', '二', '三', '四', '五', '六', '七', '八', '九', '十']
                return chinese_nums[num - 1]
            return match.group(0)

        # 在特定词语后使用中文数字
        for term in ['第', '约', '计', '数', '共', '计']:
            pattern = f"{term}(\\d{{1,2}})"

            # 重构 lambda 函数，避免在 f-string 中嵌套复杂表达式
            def replace_term_number(m, current_term=term):
                num_str = re.search(r'\d+', m.group(0)).group(0)
                num_chinese = replace_arabic_to_chinese(num_str)
                return f"{current_term}{num_chinese}"

            text = re.sub(pattern, replace_term_number, text)

        # 将带单位的数字标准化
        for unit in ['人', '次', '种', '遍', '部', '卷']:
            pattern = f"(\\d+){unit}"
            # 简化 lambda 表达式
            text = re.sub(pattern, lambda m: m.group(1) + unit, text)

        return text

    def _final_check(self, text: str) -> str:
        """最终检查与修正"""
        # 1. 修复标点符号
        text = self._fix_punctuation(text)

        # 2. 确保段落格式正确
        text = self._ensure_paragraph_format(text)

        # 3. 修复可能的错误格式
        text = self._fix_common_format_issues(text)

        return text

    def _fix_punctuation(self, text: str) -> str:
        """修复标点符号问题"""
        # 修复连续的标点符号
        text = re.sub(r'[，,]{2,}', '，', text)
        text = re.sub(r'[。.]{2,}', '。', text)
        text = re.sub(r'[！!]{2,}', '！', text)
        text = re.sub(r'[？?]{2,}', '？', text)

        # 修复不当的标点组合
        text = re.sub(r'，[。？！]', lambda m: m.group(0)[-1], text)
        text = re.sub(r'。，', '。', text)
        text = re.sub(r'[，。]"', '"', text)

        # 确保引号成对出现
        open_quotes = text.count('「')
        close_quotes = text.count('」')

        if open_quotes > close_quotes:
            text += '」' * (open_quotes - close_quotes)
        elif close_quotes > open_quotes:
            text = '「' * (close_quotes - open_quotes) + text

        # 修复中西文标点混用
        text = text.replace('"', '「').replace('"', '」')
        text = text.replace('\'', '「').replace('\'', '」')

        return text

    def _ensure_paragraph_format(self, text: str) -> str:
        """确保段落格式正确"""
        # 移除段落开头的空白
        text = re.sub(r'\n\s+', '\n', text)

        # 确保段落结束有标点
        lines = text.split('\n')
        for i in range(len(lines)):
            line = lines[i].strip()
            if line and i < len(lines) - 1 and lines[i + 1].strip():
                if not line[-1] in '。！？：；,，.!?:;':
                    lines[i] = line + '。'

        return '\n'.join(lines)

    def _fix_common_format_issues(self, text: str) -> str:
        """修复常见格式问题"""
        # 修复错误的空格使用
        text = re.sub(r'([\u4e00-\u9fff])\s+([\u4e00-\u9fff])', r'\1\2', text)

        # 修复英文和中文之间的空格
        text = re.sub(r'([\u4e00-\u9fff])([a-zA-Z])', r'\1 \2', text)
        text = re.sub(r'([a-zA-Z])([\u4e00-\u9fff])', r'\1 \2', text)

        # 修复括号问题
        text = re.sub(r'([（\(])([^）\)]*?)([，。！？])([^）\)]*?)([）\)])', r'\1\2\4\5\3', text)

        # 修复缺失的空行
        if len(text) > 100 and '\n\n' not in text and '。' in text:
            sentences = re.split(r'([。！？])', text)
            formatted = ''
            current_paragraph = ''

            for i in range(0, len(sentences), 2):
                if i + 1 < len(sentences):
                    current_sentence = sentences[i] + sentences[i + 1]
                else:
                    current_sentence = sentences[i]

                current_paragraph += current_sentence

                # 每3-5个句子或超过100字就分段
                if (i // 2) % 4 == 3 or len(current_paragraph) > 100:
                    formatted += current_paragraph + '\n\n'
                    current_paragraph = ''

            if current_paragraph:
                formatted += current_paragraph

            text = formatted.strip()

        return text

    # 移除重复的_format_parallel_structure方法，上面已经定义过了

    def _format_enumeration(self, text: str, context) -> str:
        """
        格式化列举结构文本

        Args:
            text: 原始文本
            context: 上下文信息

        Returns:
            格式化后的列举结构
        """
        from .adapter import ProcessingAdapter

        # 检查是否有列举结构
        has_enumeration = ProcessingAdapter.has_enumeration(context)
        if not has_enumeration:
            return text

        # 提取列举结构详情
        enumeration_structure = ProcessingAdapter.extract_info_from_context(
            context, 'enumeration_structure', {}
        )

        if not enumeration_structure:
            # 如果没有详细结构信息，使用启发式方法
            return self._format_enumeration_heuristic(text)

        marker_type = enumeration_structure.get('marker_type', '')
        items = enumeration_structure.get('items', [])
        total_items = enumeration_structure.get('total_items', 0)

        # 如果没有足够的项目信息，回退到启发式方法
        if not items or total_items < 2:
            return self._format_enumeration_heuristic(text)

        # 对文本进行预处理
        text = self._preprocess_enumeration_text(text)

        # 识别文本中的列举项
        identified_items = self._identify_enumeration_items(text, total_items, marker_type)

        # 如果无法识别足够的项目，回退到启发式方法
        if len(identified_items) < total_items:
            return self._format_enumeration_heuristic(text)

        # 根据识别出的项目格式化文本
        return self._apply_enumeration_format(text, identified_items, marker_type)

    def _preprocess_enumeration_text(self, text: str) -> str:
        """预处理列举文本"""
        # 去除多余空格
        text = re.sub(r'\s+', ' ', text).strip()

        # 处理可能的列举引导词
        for marker in ['有如下', '包括', '分别是', '如下', '有以下']:
            if marker in text:
                parts = text.split(marker, 1)
                if len(parts) > 1 and not parts[1].startswith('：'):
                    text = parts[0] + marker + '：' + parts[1]

        return text

    def _identify_enumeration_items(self, text: str, expected_count: int, marker_type: str) -> List[Dict]:
        """识别文本中的列举项"""
        items = []

        # 根据标记类型选择识别方法
        if marker_type == 'numeric':
            # 数字标记，如"1. 项目一"
            pattern = r'(\d+[\.、]|\(\d+\)|\d+\))'
            matches = list(re.finditer(pattern, text))

            for i, match in enumerate(matches):
                start = match.start()
                end = matches[i + 1].start() if i + 1 < len(matches) else len(text)

                items.append({
                    'marker': match.group(),
                    'start': start,
                    'end': end,
                    'text': text[start:end]
                })

        elif marker_type == 'sequence':
            # 序数词标记，如"第一，项目一"
            sequence_markers = ['第一', '第二', '第三', '第四', '第五', '第六', '第七', '第八', '第九', '第十']

            for i, marker in enumerate(sequence_markers[:expected_count]):
                start = text.find(marker)
                if start != -1:
                    next_marker = sequence_markers[i + 1] if i + 1 < len(sequence_markers) else None
                    end = text.find(next_marker) if next_marker else len(text)

                    items.append({
                        'marker': marker,
                        'start': start,
                        'end': end,
                        'text': text[start:end]
                    })

        elif marker_type == 'bullet':
            # 项目符号标记，如"• 项目一"
            pattern = r'([•·※◎○●▪︎▫︎☆★]|\*|\-)'
            matches = list(re.finditer(pattern, text))

            for i, match in enumerate(matches):
                start = match.start()
                end = matches[i + 1].start() if i + 1 < len(matches) else len(text)

                items.append({
                    'marker': match.group(),
                    'start': start,
                    'end': end,
                    'text': text[start:end]
                })

        elif marker_type == 'alternative':
            # "一方面...另一方面"类型
            markers = ['一方面', '另一方面', '一则', '再则', '一者', '二者']

            for i, marker in enumerate(markers[:expected_count]):
                start = text.find(marker)
                if start != -1:
                    next_marker = markers[i + 1] if i + 1 < len(markers) else None
                    end = text.find(next_marker) if next_marker else len(text)

                    items.append({
                        'marker': marker,
                        'start': start,
                        'end': end,
                        'text': text[start:end]
                    })

        # 如果使用标准方法无法识别足够的项目，尝试通用方法
        if len(items) < expected_count:
            # 按句号分割，尝试识别
            sentences = re.split(r'([。！？])', text)
            if len(sentences) >= 2 * expected_count:  # 每个句子及其标点
                for i in range(0, len(sentences), 2):
                    if i < len(sentences) and sentences[i].strip():
                        start = text.find(sentences[i])
                        if start != -1:
                            items.append({
                                'marker': f"{(i // 2) + 1}.",  # 生成数字标记
                                'start': start,
                                'end': start + len(sentences[i]) + (1 if i + 1 < len(sentences) else 0),
                                'text': sentences[i] + (sentences[i + 1] if i + 1 < len(sentences) else "")
                            })

        # 按位置排序
        items.sort(key=lambda x: x['start'])

        return items

    def _format_enumeration_heuristic(self, text: str) -> str:
        """使用启发式方法格式化列举结构"""
        # 预处理
        text = self._preprocess_enumeration_text(text)

        # 尝试识别常见的列举标记
        numeric_pattern = r'(\d+[\.、]|\(\d+\)|\d+\))'
        has_numeric = re.search(numeric_pattern, text) is not None

        sequence_markers = ['第一', '第二', '第三', '第四', '第五']
        has_sequence = any(marker in text for marker in sequence_markers)

        bullet_pattern = r'([•·※◎○●▪︎▫︎☆★]|\*|\-)'
        has_bullet = re.search(bullet_pattern, text) is not None

        # 如果没有发现任何标记，尝试分析句子结构
        if not (has_numeric or has_sequence or has_bullet):
            return self._format_by_sentence_structure(text)

        # 根据标记类型处理
        if has_numeric:
            matches = list(re.finditer(numeric_pattern, text))
            return self._format_with_matches(text, matches, 'numeric')
        elif has_sequence:
            # 找出所有序数词位置
            positions = []
            for marker in sequence_markers:
                start = 0
                while True:
                    pos = text.find(marker, start)
                    if pos == -1:
                        break
                    positions.append((pos, marker))
                    start = pos + len(marker)

            # 按位置排序
            positions.sort()

            # 创建匹配对象列表
            class MockMatch:
                def __init__(self, start, group_text):
                    self.start_pos = start
                    self.group_text = group_text

                def start(self):
                    return self.start_pos

                def group(self):
                    return self.group_text

            matches = [MockMatch(pos, marker) for pos, marker in positions]
            return self._format_with_matches(text, matches, 'sequence')
        elif has_bullet:
            matches = list(re.finditer(bullet_pattern, text))
            return self._format_with_matches(text, matches, 'bullet')

        return text

    def _format_with_matches(self, text: str, matches, marker_type: str) -> str:
        """根据匹配结果格式化文本"""
        if not matches:
            return text

        result = text[:matches[0].start()]

        for i, match in enumerate(matches):
            start = match.start()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
            item_text = text[start:end].strip()

            # 根据标记类型生成格式化的项目
            if marker_type == 'numeric':
                marker = match.group()
                # 标准化数字标记格式
                if not marker.endswith('.'):
                    marker = marker.rstrip(')、') + '.'
                result += f"\n{marker} {item_text[len(match.group()):].strip()}"
            elif marker_type == 'sequence':
                marker = match.group()
                result += f"\n{marker}，{item_text[len(match.group()):].strip()}"
            elif marker_type == 'bullet':
                marker = match.group()
                result += f"\n{marker} {item_text[len(match.group()):].strip()}"

        return result

    def _format_by_sentence_structure(self, text: str) -> str:
        """根据句子结构格式化列举"""
        # 分割成句子
        sentences = []
        raw_sentences = re.split(r'([。！？])', text)

        # 重组句子（包含标点）
        i = 0
        while i < len(raw_sentences):
            if i + 1 < len(raw_sentences) and raw_sentences[i + 1] in '。！？':
                sentences.append(raw_sentences[i] + raw_sentences[i + 1])
                i += 2
            else:
                if raw_sentences[i].strip():
                    sentences.append(raw_sentences[i])
                i += 1

        # 如果句子太少，可能不是列举结构
        if len(sentences) < 3:
            return text

        # 检查句子的相似性（长度、结构）以判断是否为列举
        lengths = [len(s) for s in sentences]
        avg_length = sum(lengths) / len(lengths)
        is_similar_length = all(abs(l - avg_length) < avg_length * 0.3 for l in lengths)

        # 检查句子是否有相似的开头或结构
        first_chars = [s[:2] for s in sentences if len(s) > 2]
        has_similar_start = len(set(first_chars)) < len(first_chars) * 0.7

        if is_similar_length or has_similar_start:
            # 可能是列举结构，添加数字标记
            result = []
            for i, sentence in enumerate(sentences):
                result.append(f"{i + 1}. {sentence}")

            return "\n".join(result)

        return text

    def _apply_enumeration_format(self, text: str, items: List[Dict], marker_type: str) -> str:
        """
        应用列举格式

        Args:
            text: 原始文本
            items: 识别出的列举项
            marker_type: 标记类型

        Returns:
            格式化后的文本
        """
        if not items:
            return text

        # 处理列举的前缀部分（如"有如下三点："）
        prefix = text[:items[0]['start']].strip()
        if prefix:
            # 确保前缀有合适的结尾
            if not any(prefix.endswith(end) for end in ['：', ':', '，', '。']):
                prefix += '：'

        # 构建格式化后的文本
        result = prefix + '\n' if prefix else ''

        # 根据标记类型应用不同的格式
        if marker_type == 'numeric':
            # 数字标记：标准化为"1. 内容"的格式
            for i, item in enumerate(items):
                # 提取项目内容，去除原始标记
                content = item['text'][len(item['marker']):].strip()
                # 使用标准化的数字标记
                result += f"{i + 1}. {content}\n"

        elif marker_type == 'sequence':
            # 序数词标记：保留原始序数词，统一格式
            for item in items:
                content = item['text'][len(item['marker']):].strip()
                # 如果内容以标点开始，去除它
                if content and content[0] in '，、：:;；':
                    content = content[1:].strip()
                result += f"{item['marker']}，{content}\n"

        elif marker_type == 'bullet':
            # 项目符号：统一使用"•"
            for item in items:
                content = item['text'][len(item['marker']):].strip()
                result += f"• {content}\n"

        elif marker_type == 'alternative':
            # "一方面...另一方面"类型：保留原始标记
            for item in items:
                content = item['text'].strip()
                # 确保内容以原始标记开始
                if not content.startswith(item['marker']):
                    content = item['marker'] + content
                result += f"{content}\n"

        else:
            # 默认格式：使用数字标记
            for i, item in enumerate(items):
                content = item['text'].strip()
                result += f"{i + 1}. {content}\n"

        # 移除最后的换行符
        result = result.rstrip('\n')

        # 添加总结段落（如果原文有）
        last_item_end = items[-1]['end'] if items else 0
        if last_item_end < len(text):
            suffix = text[last_item_end:].strip()
            if suffix:
                result += '\n\n' + suffix

        return result

    def _format_verse(self, text: str, context) -> str:
        """
        格式化诗偈文本

        Args:
            text: 原始文本
            context: 上下文信息

        Returns:
            格式化后的诗偈
        """
        from .adapter import ProcessingAdapter

        # 获取诗偈结构信息
        is_verse = ProcessingAdapter.is_verse(context)
        if not is_verse:
            return text

        # 提取诗偈结构详情
        verse_structure = ProcessingAdapter.extract_info_from_context(context, 'verse_structure', {})
        if not verse_structure:
            # 如果没有详细结构信息，使用启发式方法
            return self._format_verse_heuristic(text)

        line_count = verse_structure.get('line_count', 0)
        stanza_count = verse_structure.get('stanza_count', 0)
        pattern = verse_structure.get('pattern', 'unknown')
        lines = verse_structure.get('lines', [])

        # 对文本进行预处理
        text = self._preprocess_verse_text(text)

        # 分割成句子
        sentences = self._split_into_verse_lines(text, line_count)

        # 如果无法获得足够的句子，回退到启发式方法
        if len(sentences) < line_count:
            return self._format_verse_heuristic(text)

        # 根据原始结构格式化
        if stanza_count > 1:
            # 多节诗偈
            return self._format_multi_stanza_verse(sentences, verse_structure)
        else:
            # 单节诗偈
            return self._format_single_stanza_verse(sentences, pattern)

    def _preprocess_verse_text(self, text: str) -> str:
        """预处理诗偈文本"""
        # 去除多余空格
        text = re.sub(r'\s+', ' ', text).strip()

        # 将偈颂引导词后的冒号转换为换行
        for marker in ['偈言：', '偈曰：', '颂曰：', '所谓：']:
            if marker in text:
                parts = text.split(marker)
                if len(parts) > 1:
                    text = parts[0] + marker + '\n' + parts[1]

        return text

    def _split_into_verse_lines(self, text: str, expected_lines: int) -> List[str]:
        """将文本分割为诗行"""
        # 如果已有明确的换行，直接使用
        if '\n' in text:
            lines = [line.strip() for line in text.split('\n') if line.strip()]
            if len(lines) >= expected_lines:
                return lines

        # 尝试按句号分割
        sentences = []
        raw_sentences = re.split(r'([。！？])', text)

        # 重组句子（包含标点）
        i = 0
        while i < len(raw_sentences):
            if i + 1 < len(raw_sentences) and raw_sentences[i + 1] in '。！？':
                sentences.append(raw_sentences[i] + raw_sentences[i + 1])
                i += 2
            else:
                if raw_sentences[i].strip():
                    sentences.append(raw_sentences[i])
                i += 1

        # 如果句子数量少于预期行数，尝试按逗号分割
        if len(sentences) < expected_lines:
            comma_sentences = []
            for sentence in sentences:
                parts = re.split(r'([，；：])', sentence)

                # 重组（包含标点）
                i = 0
                while i < len(parts):
                    if i + 1 < len(parts) and parts[i + 1] in '，；：':
                        comma_sentences.append(parts[i] + parts[i + 1])
                        i += 2
                    else:
                        if parts[i].strip():
                            comma_sentences.append(parts[i])
                        i += 1

            sentences = comma_sentences

        # 返回分割结果
        return sentences

    def _format_verse_heuristic(self, text: str) -> str:
        """使用启发式方法格式化诗偈"""
        # 预处理
        text = self._preprocess_verse_text(text)

        # 检查是否有明确的换行
        if '\n' in text:
            return text  # 保留现有格式

        # 尝试识别四句偈、八句偈等常见结构
        sentences = self._split_into_verse_lines(text, 0)

        # 四句偈：通常为两联，每联两句
        if len(sentences) == 4:
            return f"{sentences[0]}，{sentences[1]}；\n{sentences[2]}，{sentences[3]}。"

        # 八句偈：通常为四联，每联两句
        elif len(sentences) == 8:
            return (f"{sentences[0]}，{sentences[1]}；\n"
                    f"{sentences[2]}，{sentences[3]}；\n"
                    f"{sentences[4]}，{sentences[5]}；\n"
                    f"{sentences[6]}，{sentences[7]}。")

        # 六句偈：通常为三联，每联两句
        elif len(sentences) == 6:
            return (f"{sentences[0]}，{sentences[1]}；\n"
                    f"{sentences[2]}，{sentences[3]}；\n"
                    f"{sentences[4]}，{sentences[5]}。")

        # 其他情况：根据句子长度尝试分组
        else:
            result = []
            line_buffer = []

            for sentence in sentences:
                line_buffer.append(sentence)

                # 累计约30-40字成一行
                if sum(len(s) for s in line_buffer) >= 30:
                    result.append("，".join(line_buffer))
                    line_buffer = []

            # 处理剩余部分
            if line_buffer:
                result.append("，".join(line_buffer))

            return "\n".join(result)

    def _format_single_stanza_verse(self, sentences: List[str], pattern: str) -> str:
        """格式化单节诗偈"""
        # 根据不同的韵律模式格式化
        if pattern == '7-syllable' or pattern == '5-syllable':
            # 整齐的七言或五言偈
            # 每四句一组，中间用逗号和顿号分隔
            result = []
            for i in range(0, len(sentences), 4):
                group = sentences[i:i + 4]
                if len(group) == 4:
                    result.append(f"{group[0]}，{group[1]}；\n{group[2]}，{group[3]}。")
                else:
                    result.append("，".join(group) + "。")

            return "\n".join(result)

        elif pattern == '9-syllable':
            # 九言偈：每两句一组
            result = []
            for i in range(0, len(sentences), 2):
                if i + 1 < len(sentences):
                    result.append(f"{sentences[i]}，{sentences[i + 1]}。")
                else:
                    result.append(f"{sentences[i]}。")

            return "\n".join(result)

        elif pattern == 'alternating':
            # 交替格式：通常为长短句交替
            result = []
            for i in range(0, len(sentences), 2):
                if i + 1 < len(sentences):
                    result.append(f"{sentences[i]}，{sentences[i + 1]}；")
                else:
                    result.append(f"{sentences[i]}。")

            return "\n".join(result)

        else:
            # 默认格式：四句一组
            result = []
            for i in range(0, len(sentences), 4):
                group = sentences[i:min(i + 4, len(sentences))]
                if len(group) == 4:
                    result.append(f"{group[0]}，{group[1]}；\n{group[2]}，{group[3]}。")
                elif len(group) == 3:
                    result.append(f"{group[0]}，{group[1]}；\n{group[2]}。")
                elif len(group) == 2:
                    result.append(f"{group[0]}，{group[1]}。")
                else:
                    result.append(f"{group[0]}。")

            return "\n".join(result)

    def _format_multi_stanza_verse(self, sentences: List[str], verse_structure: Dict) -> str:
        """格式化多节诗偈"""
        stanzas = verse_structure.get('stanzas', [])
        pattern = verse_structure.get('pattern', 'unknown')

        if not stanzas:
            # 尝试均匀分配句子到节
            stanza_count = verse_structure.get('stanza_count', 2)
            lines_per_stanza = len(sentences) // stanza_count

            result = []
            for i in range(0, len(sentences), lines_per_stanza):
                stanza_sentences = sentences[i:i + lines_per_stanza]
                stanza_text = self._format_single_stanza_verse(stanza_sentences, pattern)
                result.append(stanza_text)

            return "\n\n".join(result)

        # 根据原始节结构格式化
        stanza_line_counts = [len(stanza) for stanza in stanzas]

        result = []
        start_idx = 0

        for count in stanza_line_counts:
            if start_idx >= len(sentences):
                break

            stanza_sentences = sentences[start_idx:start_idx + count]
            stanza_text = self._format_single_stanza_verse(stanza_sentences, pattern)
            result.append(stanza_text)

            start_idx += count

        return "\n\n".join(result)

    def _format_by_length(self, text: str, target_length: int) -> str:
        """按长度格式化文本"""
        # 分句
        sentences = re.split(r'([。！？])', text)

        formatted_lines = []
        current_line = ""

        for i in range(0, len(sentences), 2):
            if i + 1 < len(sentences):
                sentence = sentences[i] + sentences[i + 1]
            else:
                sentence = sentences[i] if sentences[i].strip() else ""

            if not sentence:
                continue

            # 如果当前行加上新句子不超过目标长度，添加到当前行
            if len(current_line) + len(sentence) <= target_length * 1.5:
                current_line += sentence
            else:
                # 否则换行
                if current_line:
                    formatted_lines.append(current_line)
                current_line = sentence

        # 添加最后一行
        if current_line:
            formatted_lines.append(current_line)

        return '\n'.join(formatted_lines)

    def _optimize_paragraphs(self, text: str) -> str:
        """优化段落结构"""
        # 分割段落
        paragraphs = text.split('\n')
        optimized_paragraphs = []

        for para in paragraphs:
            para = para.strip()
            if not para:
                continue

            # 检查段落是否过长
            if len(para) > 300:
                # 尝试在合适的位置分段
                split_para = self._split_long_paragraph(para)
                optimized_paragraphs.extend(split_para)
            else:
                optimized_paragraphs.append(para)

        # 合并过短的段落
        final_paragraphs = []
        i = 0
        while i < len(optimized_paragraphs):
            current = optimized_paragraphs[i]

            # 如果当前段落太短且不是最后一段
            if len(current) < 50 and i < len(optimized_paragraphs) - 1:
                next_para = optimized_paragraphs[i + 1]
                # 如果下一段也不长，合并
                if len(next_para) < 100:
                    final_paragraphs.append(current + next_para)
                    i += 2
                    continue

            final_paragraphs.append(current)
            i += 1

        return '\n\n'.join(final_paragraphs)

    def _split_long_paragraph(self, para: str) -> List[str]:
        """分割长段落"""
        # 查找合适的分割点
        sentences = re.split(r'([。！？])', para)

        paragraphs = []
        current_para = ""
        sentence_count = 0

        for i in range(0, len(sentences), 2):
            if i + 1 < len(sentences):
                sentence = sentences[i] + sentences[i + 1]
            else:
                sentence = sentences[i] if sentences[i].strip() else ""

            if not sentence:
                continue

            current_para += sentence
            sentence_count += 1

            # 每3-5个句子分一段，或者达到150字符
            if (sentence_count >= 3 and len(current_para) > 150) or len(current_para) > 250:
                paragraphs.append(current_para.strip())
                current_para = ""
                sentence_count = 0

        # 添加剩余部分
        if current_para.strip():
            # 如果剩余部分太短，与上一段合并
            if len(current_para) < 50 and paragraphs:
                paragraphs[-1] += current_para
            else:
                paragraphs.append(current_para.strip())

        return paragraphs

    def _balance_quotes(self, text: str) -> str:
        """平衡引号"""
        # 统计引号
        double_open = text.count('"')
        double_close = text.count('"')

        # 如果不平衡，尝试修复
        if double_open > double_close:
            # 在文末添加闭引号
            text = text.rstrip() + '"'
        elif double_close > double_open:
            # 移除多余的闭引号
            text = re.sub(r'"([^"]*?)$', r'\1', text)

        return text

    def _format_lists(self, text: str) -> str:
        """格式化列表"""
        # 识别数字列表
        list_patterns = [
            # 一、二、三...
            (r'([一二三四五六七八九十]+)、(\S+)', r'\1、\2'),
            # 1. 2. 3. ...
            (r'(\d+)\.\s*(\S+)', r'\1. \2'),
            # (1) (2) (3) ...
            (r'（(\d+)）\s*(\S+)', r'（\1）\2'),
        ]

        for pattern, replacement in list_patterns:
            text = re.sub(pattern, replacement, text)

        # 处理项目符号列表
        text = re.sub(r'[·•‧]\s*(\S+)', r'• \1', text)

        return text

    def _format_titles(self, text: str) -> str:
        """格式化标题"""
        # 识别可能的标题（短句子，可能全是名词）
        lines = text.split('\n')
        formatted_lines = []

        for line in lines:
            line = line.strip()
            if not line:
                formatted_lines.append('')
                continue

            # 判断是否为标题
            if self._is_likely_title(line):
                # 标题后加空行
                formatted_lines.append(line)
                if len(formatted_lines) < len(lines):  # 不是最后一行
                    formatted_lines.append('')
            else:
                formatted_lines.append(line)

        return '\n'.join(formatted_lines)

    def _is_likely_title(self, line: str) -> bool:
        """判断是否可能是标题"""
        # 标题特征：
        # 1. 较短（少于20字）
        # 2. 没有句号
        # 3. 可能包含"品"、"章"、"节"等

        if len(line) > 20:
            return False

        if '。' in line:
            return False

        title_keywords = ['品', '章', '节', '分', '第', '之', '论', '说', '经']
        return any(keyword in line for keyword in title_keywords)

    def _format_sutra_conventions(self, text: str, context) -> str:
        """格式化佛经特殊惯例"""
        # 处理"如是我闻"开头
        if text.startswith('如是我闻'):
            text = '如是我闻：\n\n' + text[4:]

        # 处理"尔时"段落
        text = re.sub(r'(尔时[^。]+。)', r'\n\1', text)

        # 处理重复三次的赞叹（如：善哉善哉善哉）
        text = re.sub(r'(善哉){3,}', '善哉！善哉！善哉！', text)

        # 处理咒语格式（如果有）
        text = self._format_mantras(text)

        return text

    def _format_mantras(self, text: str) -> str:
        """格式化咒语"""
        # 识别可能的咒语（通常有特定标记）
        mantra_markers = ['嗡', '吽', '咒曰', '陀罗尼曰']

        for marker in mantra_markers:
            if marker in text:
                # 查找咒语部分
                pattern = f'{marker}[^。！？]+'
                matches = re.finditer(pattern, text)

                for match in matches:
                    mantra = match.group()
                    # 咒语通常需要特殊格式
                    formatted_mantra = f'\n{mantra}\n'
                    text = text.replace(mantra, formatted_mantra)

        return text

    def _apply_final_touches(self, text: str) -> str:
        """应用最后的润色"""
        # 确保开头没有空白
        text = text.lstrip()

        # 确保结尾合适
        text = text.rstrip()
        if text and text[-1] not in '。！？；】》）"\'':
            text += '。'

        # 处理特殊情况
        # 如果整篇只有一段，不需要段落缩进
        if '\n' not in text:
            return text

        # 如果是对话体，保持原样
        if text.count('"') > 4 or text.count('：') > 2:
            return text

        return text