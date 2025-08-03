"""
藏译汉工具主程序
"""
import argparse
import logging
from pathlib import Path
import json
import sys
from typing import Optional

from translator import TranslationManager
from utils import text_utils, file_utils, validation_utils
from config import Config

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('translation.log', encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)


class TibetanTranslatorCLI:
    def __init__(self):
        self.translator = None
        self.config = Config()

    def initialize_translator(self, args):
        """初始化翻译器"""
        config = {
            'sentence_split_threshold': args.split_threshold,
            'use_alternatives': args.alternatives,
            'batch_size': args.batch_size
        }

        logger.info("Initializing translator...")
        self.translator = TranslationManager(config)
        logger.info("Translator initialized successfully")

    def translate_text(self, args):
        """翻译文本"""
        self.initialize_translator(args)

        # 读取输入
        if args.input_file:
            text = file_utils.read_text_file(Path(args.input_file))
        else:
            text = args.text

        if not text:
            logger.error("No input text provided")
            return

        # 验证输入
        if not text_utils.is_tibetan(text):
            logger.warning("Input text doesn't appear to be Tibetan")

        # 执行翻译
        logger.info(f"Translating text (length: {len(text)})")
        result = self.translator.translate(text)

        # 输出结果
        if result['translation']:
            print("\n=== Translation ===")
            print(result['translation'])

            print("\n=== Quality Score ===")
            print(f"Overall: {result['quality_score']:.2f}")

            if args.verbose:
                print("\n=== Metadata ===")
                print(json.dumps(result['metadata'], indent=2, ensure_ascii=False))

            # 保存到文件
            if args.output_file:
                file_utils.write_text_file(
                    Path(args.output_file),
                    result['translation']
                )
                logger.info(f"Translation saved to {args.output_file}")
        else:
            logger.error("Translation failed")
            if 'error' in result:
                logger.error(f"Error: {result['error']}")

    def translate_file(self, args):
        """翻译文件"""
        self.initialize_translator(args)

        input_path = Path(args.input_file)
        if not input_path.exists():
            logger.error(f"Input file not found: {input_path}")
            return

        # 读取文件
        text = file_utils.read_text_file(input_path)

        # 执行翻译
        logger.info(f"Translating file: {input_path}")
        result = self.translator.translate_document(
            text,
            progress_callback=self._show_progress if args.show_progress else None
        )

        # 保存结果
        if args.output_file:
            output_path = Path(args.output_file)
        else:
            output_path = input_path.with_suffix('.translated.txt')

        file_utils.write_text_file(output_path, result['translation'])
        logger.info(f"Translation saved to {output_path}")

        # 保存元数据
        if args.save_metadata:
            metadata_path = output_path.with_suffix('.json')
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(result['metadata'], f, ensure_ascii=False, indent=2)
            logger.info(f"Metadata saved to {metadata_path}")

    def translate_directory(self, args):
        """批量翻译目录"""
        self.initialize_translator(args)

        input_dir = Path(args.input_dir)
        output_dir = Path(args.output_dir)

        if not input_dir.exists():
            logger.error(f"Input directory not found: {input_dir}")
            return

        # 创建输出目录
        output_dir.mkdir(parents=True, exist_ok=True)

        # 处理文件
        def translate_file_func(content):
            result = self.translator.translate(content)
            return result['translation']

        file_utils.process_directory(
            input_dir,
            output_dir,
            translate_file_func,
            file_pattern=args.file_pattern
        )

        logger.info(f"Batch translation completed. Results in {output_dir}")

    def _show_progress(self, current: int, total: int):
        """显示进度"""
        percentage = (current / total) * 100
        bar_length = 50
        filled_length = int(bar_length * current / total)
        bar = '█' * filled_length + '-' * (bar_length - filled_length)

        print(f'\rProgress: |{bar}| {percentage:.1f}% ({current}/{total})', end='')
        if current == total:
            print()  # 换行

    def validate_translation(self, args):
        """验证翻译质量"""
        # 读取源文本和译文
        source = file_utils.read_text_file(Path(args.source_file))
        translation = file_utils.read_text_file(Path(args.translation_file))

        # 执行验证
        checks = validation_utils.validate_translation_pair(source, translation)

        print("=== Translation Validation ===")
        for check, passed in checks.items():
            status = "✓" if passed else "✗"
            print(f"{status} {check}")

        # 详细分析（如果需要）
        if args.detailed:
            print("\n=== Detailed Analysis ===")
            # 可以添加更详细的分析


def main():
    parser = argparse.ArgumentParser(
        description='Tibetan to Chinese Translation Tool',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    subparsers = parser.add_subparsers(dest='command', help='Available commands')

    # 翻译文本命令
    text_parser = subparsers.add_parser('translate', help='Translate text')
    text_parser.add_argument('text', nargs='?', help='Text to translate')
    text_parser.add_argument('-i', '--input-file', help='Input file path')
    text_parser.add_argument('-o', '--output-file', help='Output file path')
    text_parser.add_argument('-v', '--verbose', action='store_true', help='Verbose output')
    text_parser.add_argument('--split-threshold', type=int, default=100,
                             help='Sentence split threshold')
    text_parser.add_argument('--alternatives', action='store_true',
                             help='Generate alternative translations')
    text_parser.add_argument('--batch-size', type=int, default=8,
                             help='Batch size for translation')

    # 翻译文件命令
    file_parser = subparsers.add_parser('translate-file', help='Translate a file')
    file_parser.add_argument('input_file', help='Input file path')
    file_parser.add_argument('-o', '--output-file', help='Output file path')
    file_parser.add_argument('--show-progress', action='store_true',
                             help='Show translation progress')
    file_parser.add_argument('--save-metadata', action='store_true',
                             help='Save translation metadata')
    file_parser.add_argument('--split-threshold', type=int, default=100)
    file_parser.add_argument('--batch-size', type=int, default=8)

    # 批量翻译命令
    batch_parser = subparsers.add_parser('translate-dir', help='Translate directory')
    batch_parser.add_argument('input_dir', help='Input directory')
    batch_parser.add_argument('output_dir', help='Output directory')
    batch_parser.add_argument('--file-pattern', default='*.txt',
                              help='File pattern to match')
    batch_parser.add_argument('--split-threshold', type=int, default=100)
    batch_parser.add_argument('--batch-size', type=int, default=8)

    # 验证命令
    validate_parser = subparsers.add_parser('validate', help='Validate translation')
    validate_parser.add_argument('source_file', help='Source file (Tibetan)')
    validate_parser.add_argument('translation_file', help='Translation file (Chinese)')
    validate_parser.add_argument('--detailed', action='store_true',
                                 help='Show detailed analysis')

    args = parser.parse_args()

    # 创建CLI实例
    cli = TibetanTranslatorCLI()

    # 执行命令
    if args.command == 'translate':
        cli.translate_text(args)
    elif args.command == 'translate-file':
        cli.translate_file(args)
    elif args.command == 'translate-dir':
        cli.translate_directory(args)
    elif args.command == 'validate':
        cli.validate_translation(args)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()