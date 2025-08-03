# 藏译汉工具 (Tibetan to Chinese Translation Tool)

专注于佛法文档翻译的藏语到中文翻译工具。

## 特性

- 基于NLLB-200模型的高质量翻译
- 整合modern-botok进行语言分析
- 智能术语识别和保护
- 语法感知的后处理
- 支持批量翻译

## 安装

```bash
# 克隆仓库
git clone https://github.com/yourusername/tibetan-translator.git
cd tibetan-translator

# 安装依赖
pip install poetry
poetry install

# 安装botok
pip install git+https://github.com/OpenPecha/Botok

# 配置modern-botok（参见文档）
```

## 使用方法

### 命令行使用

```bash
# 翻译文本
tibetan-translator translate "དེ་ལྟར་ན་བྱང་ཆུབ་སེམས་དཔའ་"

# 翻译文件
tibetan-translator translate-file input.txt -o output.txt

# 批量翻译
tibetan-translator translate-dir input_folder output_folder

# 验证翻译
tibetan-translator validate source.txt translation.txt
```

### Python API使用

```python
from tibetan_translator import TranslationManager

# 初始化翻译器
translator = TranslationManager()

# 翻译文本
result = translator.translate("སངས་རྒྱས་བ