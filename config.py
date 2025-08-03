"""
配置文件 - 添加术语系统相关配置
"""
import os
from pathlib import Path


class Config:
    # 基础配置
    BASE_DIR = Path(__file__).parent
    DATA_DIR = BASE_DIR / "data"

    # Botok配置
    BOTOK_DIALECT = "custom"
    BOTOK_BASE_PATH = Path.home() / "Documents" / "pybo"

    # NLLB配置
    NLLB_MODEL = "facebook/nllb-200-distilled-600M"
    SOURCE_LANG = "bod_Tibt"
    TARGET_LANG = "zho_Hans"
    MAX_LENGTH = 512

    # 翻译配置
    BATCH_SIZE = 8
    USE_GPU = True

    # 术语数据库配置
    TERM_DATABASE_PATH = DATA_DIR / "buddhist_terms.db"
    TERM_DATABASE_TYPE = "sqlite"  # 可选: "sqlite", "json"

    # 术语系统配置
    TERM_CACHE_SIZE = 1000
    TERM_CONFIDENCE_THRESHOLD = 0.7

    # 语境检测配置
    CONTEXT_DETECTION_WINDOW = 100  # 检测语境时的文本窗口大小
    MIXED_TEXT_THRESHOLD = 0.3  # 混合文本的阈值

    # 质量控制阈值
    QUALITY_CONFIDENCE_THRESHOLD = 0.7
    MIN_TERM_LENGTH = 2

    # 日志配置
    LOG_LEVEL = "INFO"
    LOG_FILE = BASE_DIR / "logs" / "translation.log"