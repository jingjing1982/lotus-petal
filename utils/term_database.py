"""
术语数据库管理 - 支持语境和功能标签的术语系统
"""
import json
import sqlite3
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Set, Any
import logging
from datetime import datetime
from enum import Enum
import re
from collections import defaultdict

logger = logging.getLogger(__name__)


class ContextLabel(Enum):
    """术语语境标签"""
    PRAMANA = "因明语境"  # 因明学
    MADHYAMIKA = "中观语境"  # 中观学
    ABHISAMAYA = "现观语境"  # 现观庄严论
    ABHIDHARMA = "俱舍语境"  # 俱舍论
    SUTRA = "显宗语境"  # 显宗文本
    TANTRA = "密乘语境"  # 密乘文本
    DZOGCHEN = "大圆满语境"  # 大圆满文本
    VINAYA = "戒律语境"  # 戒律文本
    GENERAL = "通用语境"  # 通用佛教文本


class FunctionLabel(Enum):
    """术语功能标签"""
    PRACTICE_METHOD = "修行方法描述"
    PHILOSOPHY = "哲学概念表达"
    REALIZATION = "境界描述"
    RITUAL = "仪式指导"
    PROPER_NOUN = "专有名词"
    GENERAL_TERM = "一般术语"


class FlexibleContextDetector:
    """灵活语境判断器 - 用于判断佛教文本所属的语境类别"""

    def __init__(self):
        # 初始化各种特征数据
        self.context_keywords = self._initialize_keywords()
        self.unique_markers = self._initialize_unique_markers()
        self.ambiguous_terms = self._initialize_ambiguous_terms()
        self.co_occurrence_patterns = self._initialize_co_occurrence_patterns()
        self.function_keywords = self._initialize_function_keywords()

    def _initialize_keywords(self) -> Dict[str, List[str]]:
        """初始化语境关键词"""
        return {
            'PRAMANA': [
                'གཏན་ཚིགས་', 'རྟགས་', 'ཕྱོགས་ཆོས་', 'ཁྱབ་པ་', 'དཔེ་', 'སྒྲུབ་བྱེད་',
                'མཐུན་དཔེ་', 'མི་མཐུན་དཔེ་', 'ཆོས་ཅན་', 'བསྒྲུབ་བྱ་', 'ཚད་མ་', 'ཕྱོགས་',
                """ 定量、标志、宗法性、遍计、比量、成就、同喻、异喻、法相、所立、因 """
                'གཞན་སེལ་', 'མཚོན་བྱ་', 'རང་མཚན་', """遮遣、所表、自相"""
                                                   'རྗེས་དཔག་', 'མངོན་སུམ་', 'སྒྲོ་འདོགས་', """ 比量、现量、增益"""
                                                                                            'སྐྱོན་', 'ལྡོག་ཁྱབ་',
                'རྗེས་ཁྱབ་', """过失、返遍、随遍"""
                             'ཧེ་ཏུ་', 'ཕྱོགས་', 'དྲུག་སྒྲ་', """因、宗、六声"""
                                                              'མ་གྲུབ་པ་', 'མ་ངེས་པ་', 'འགལ་བ་', """不成因、不定因、相违因"""
                                                                                                 'ཐལ་འགྱུར་',
                'སུན་འབྱིན་', 'ལྟག་ཆོད་'     """应成、能破、似破"""
            ],
            'MADHYAMIKA': [
                'སྟོང་པ་ཉིད་', 'རང་བཞིན་མེད་', 'རྟེན་འབྲེལ་', 'བདེན་གཉིས་', 'མཐའ་བྲལ་',
                'དབུ་མ་', 'རྟག་ཆད་', 'གཉིས་མེད་', 'བདེན་པར་གྲུབ་པ་མེད་', """空性、自性不存在、依他起、二谛、无边际、中道、永恒不变、无二"""
                                                                         'མུ་བཞི་', 'སྐྱེ་བ་བཀག་', 'བདེན་སྟོང་',
                """ 四句生、遮止生起、真实空 """
                'ངོ་བོ་ཉིད་མེད་', 'རང་རྒྱུད་', 'ཐལ་འགྱུར་བ་', """无实性、自续派、应成派 """
                                                              'དབུ་མ་ཐལ་འགྱུར་', 'དབུ་མ་རང་རྒྱུད་', """中观应成、中观自续 """
                                                                                                    'སྤྲོས་བྲལ་',
                'སྟོང་ཉིད་', 'ནམ་མཁའི་མེ་ཏོག་', """ 离戏、空性、虚空花 """
                                                'བདག་མེད་', 'གང་ཟག་གི་བདག་མེད་', """无我、人无我 """
                                                                                 'ཆོས་ཀྱི་བདག་མེད་', 'རྨི་ལམ་',
                'སྒྱུ་མ་'      """法无我、梦、幻 """
            ],
            'ABHISAMAYA': [
                'མངོན་རྟོགས་', 'རྣམ་མཁྱེན་', 'ལམ་ཤེས་', 'གཞི་ཤེས་', 'རྣམ་རྫོགས་',
                'སྦྱོར་བ་', 'སྐུ་གསུམ་', 'རྒྱུད་བླ་མ་', 'ཡེ་ཤེས་ཆེན་པོ་', 'འདུལ་བ་', 'བདེ་གཤེགས་སྙིང་པོ་', 'རིགས་',
                """现观、智慧、行相、根本、无上瑜伽、清净心、大智慧、种姓 """
                'སྦས་དོན་', 'མངོན་རྟོགས་རྒྱན་', 'རྒྱས་འབྲིང་བསྡུས་གསུམ་', """隐义、现观庄严论、三般若 """
                                                                          'རྣམ་པ་ཉི་ཤུ་', 'སྦྱོར་བ་བཞི་', 'ཆོས་སྐུ་',
                """二十行相、四加行、法身 """
                'གཟུགས་སྐུ་', 'ལོངས་སྐུ་', 'སྤྲུལ་སྐུ་', """ 色身、受用身、化身 """
                                                         'སྐབས་བརྒྱད་', 'དོན་བདུན་ཅུ་', 'རྣམ་པ་བརྒྱ་དང་བཅུ་',
                """八事、七十义、一百一十行相 """
                'གཉེན་པོ་ཕྱོགས་', 'མི་མཐུན་ཕྱོགས་', 'དགེ་འདུན་ཉི་ཤུ་', """对治分、障碍分、二十僧 """
                                                                       'སྤང་བྱ་', 'གཉེན་པོ་',
                'རྣམ་བྱང་སེམས་བསྐྱེད་'          """所断、对治、清净发心 """
            ],
            'ABHIDHARMA': [
                'ཁམས་', 'ཕུང་པོ་', 'སྐྱེ་མཆེད་', 'རྣམ་ཤེས་', 'སེམས་བྱུང་',
                'རྒྱུ་འབྲས་', 'སྐད་ཅིག་མ་', 'མཚན་ཉིད་', 'དབྱེ་བ་', 'མངོན་པ་མཛོད་', """色界、色、心所、识、心、因果、无常、名相、俱舍"""
                                                                                   'མཛོད་', 'ཟག་བཅས་', 'ཟག་མེད་',
                """俱舍、有漏、无漏"""
                'སོ་སོར་བརྟགས་འགོག་', 'བརྟགས་མིན་གྱི་འགོག་པ་', """择灭、非择灭"""
                                                               'འདུས་བྱས་', 'འདུས་མ་བྱས་', 'མཚུངས་ལྡན་',
                """有为法、无为法、相应"""
                'རྫས་ཡོད་', 'བཏགས་ཡོད་', 'ལྡན་མིན་འདུ་བྱེད་', """实有、假有、心不相应行"""
                                                              'སྐབས་བརྒྱད་', 'ལས་', 'ཉོན་མོངས་པ་', """八品、业、烦恼"""
                                                                                                   'གཟུགས་ཁམས་',
                'འདོད་ཁམས་', 'གཟུགས་མེད་ཁམས་', """色界、欲界、无色界"""
                                               'འཕགས་ལམ་ཡན་ལག་བརྒྱད་', 'བདེན་པ་བཞི་'         """八正道、四谛"""
            ],
            'TANTRA': [
                'དབང་', 'བསྐྱེད་རིམ་', 'རྫོགས་རིམ་', 'དཀྱིལ་འཁོར་', 'སྔགས་',
                'རྡོ་རྗེ་', 'ལྷ་', 'བདུད་རྩི་', 'ཕྱག་རྒྱ་', 'ཡབ་ཡུམ་', """"灌顶、生起次第、圆满次第、坛城、咒语、金刚、护法、手印、父母"""
                                                                       'གསང་སྔགས་', 'སྨིན་གྲོལ་', 'རིག་པའི་དབང་ལྔ་',
                """密咒、成熟解脱、五智慧灌顶"""
                'རྩ་རླུང་ཐིག་ལེ་', 'གཏུམ་མོ་', 'འཕོ་བ་', """脉气明点、拙火、迁识"""
                                                         'རིམ་གཉིས་', 'གསང་བ་', 'བྱིན་རླབས་', """二次第、秘密、加持"""
                                                                                              'རྡོ་རྗེ་', 'དཀྱིལ་འཁོར་',
                'ཡི་དམ་', """金刚、坛城、本尊"""
                          'སྒྲུབ་ཐབས་', 'གཡོན་སྐོར་', 'དབང་བཞི་', """ 修法仪轨、逆行、四灌顶"""
                                                                  'རྩ་', 'རླུང་', 'ཐིག་ལེ་', """脉、气、明点"""
                                                                                             'སྐུ་གསུང་ཐུགས་',
                'ཕྱག་རྒྱ་ཆེན་པོ་'              """身语意、大手印"""
            ],
            'DZOGCHEN': [
                'རིག་པ་', 'ཀ་དག་', 'ལྷུན་གྲུབ་', 'རང་ཤར་', 'གཞི་སྣང་',
                'ཐོད་རྒལ་', 'ཁྲེགས་ཆོད་', 'འཁོར་འདས་རུ་ཤན་', 'ཡེ་གདངས་', """智慧、本净、任运、自然显现、本初基、中阴、彻断、光明、虹身"""
                                                                         'བྱང་ཆུབ་སེམས་', 'རང་གྲོལ་', 'རང་བྱུང་ཡེ་ཤེས་',
                """菩提心、自解脱、自生智慧"""
                'མ་བཅོས་པ་', 'ཟང་ཐལ་', 'གདོད་མའི་གཞི་', """无改造、透彻、本初基"""
                                                        'བཅོས་མིན་', 'རང་ངོ་', 'རང་སྣང་', """无造作、自面、自现"""
                                                                                          'སྐྱེ་མེད་', 'ཐོག་རྩལ་',
                'ཁྲེགས་ཆོད་', """无生、顿超、彻断"""
                              'གཞི་སྣང་', 'སྣང་བཞི་', 'བར་དོ་', """基现、四现、中阴"""
                                                                'འོད་གསལ་', 'འཇའ་ལུས་', 'རིག་པའི་རྩལ་', """光明、虹身、明觉之力"""
                                                                                                        'གསང་བ་སྙིང་པོ་',
                'རྫོགས་པ་ཆེན་པོ་'             """秘密精髓、大圆满"""
            ],
            'VINAYA': [
                'སོ་ཐར་', 'ལྟུང་བ་', 'སྤང་བ་', 'བསླབ་པ་', 'ཉེས་བྱས་',
                'གསོ་སྦྱོང་', 'དགེ་སློང་', 'དགེ་ཚུལ་', 'དགེ་སློང་མ་', """别解脱、布萨、灭摈、学处、具足戒、比丘、比丘尼"""
                                                                      'ལྟུང་བྱེད་', 'ཕམ་པ་', 'ལྷག་མ་', """波逸提、波罗夷、僧残"""
                                                                                                       'སྡོམ་པ་',
                'གནས་ནས་དབྱུང་བ་', 'གསོ་སྦྱོང་', """律仪、灭摈、布萨"""
                                                 'དབྱར་གནས་', 'དགག་དབྱེ་', 'དགེ་སློང་གི་སྡོམ་པ་', """安居、自恣、比丘戒"""
                                                                                                  'བསྙེན་རྫོགས་',
                'ཉམས་པ་', 'གཅོད་པ་', """具足戒、毁犯、断除"""
                                     'ཚུལ་ཁྲིམས་', 'བསླབ་པ་བཅས་པ་', 'རྩ་ལྟུང་', """律仪、学处、根本罪"""
                                                                                'ལས་', 'འདུལ་བ་ལུང་', 'རྣམ་འབྱེད་',
                """羯磨、律藏、分别"""
                'ཕྲན་ཚེགས་', 'གཞི་', 'མདོ་རྩ་'             """杂事、事、根本律经"""
            ],
            'SUTRA': [
                'མདོ་', 'བཅོམ་ལྡན་འདས་', 'དགེ་སློང་', 'བྱང་ཆུབ་སེམས་དཔའ་',
                'ཉན་ཐོས་', 'ཆོས་སྟོན་པ་', 'མདོ་སྡེ་', 'ཤཱ་རིའི་བུ་', 'ལུང་བསྟན་པ་', """经、佛陀、比丘、菩萨、听闻、教法、经藏、释迦牟尼、传承"""
                                                                                    'ཤེར་ཕྱིན་', 'ཤེས་རབ་སྙིང་པོ་',
                'རྒྱས་པ་', """般若波罗蜜多、心经、广本 """
                           'མདོ་སྡེ་དཀོན་མཆོག་', 'སྤྱོད་འཇུག་', 'བསླབ་བཏུས་', """经藏宝、入行论、学处摄颂 """
                                                                              'ལང་ཀར་གཤེགས་པ་', 'སངས་རྒྱས་ཕལ་པོ་ཆེ་',
                """入楞伽经、华严经 """
                'མདོ་སྡུད་པ་', 'བྱམས་ཆོས་', 'དེ་བཞིན་གཤེགས་པ་', """摄颂经、弥勒五论、如来 """
                                                                'ཐོས་པ་', 'ངེས་འབྱུང་', 'ལུང་', """闻法、出离、教法 """
                                                                                                'བསྟན་བཅོས་',
                'གདམས་ངག་', 'བཀའ་', """论著、教授、佛语 """
                                    'རྣམ་གྲོལ་', 'ཞི་བ་', 'སྐྱབས་འགྲོ་', """解脱、寂静、皈依 """
            ]
        }

    def _initialize_unique_markers(self) -> Dict[str, List[str]]:
        """初始化高度特异性标记词"""
        return {
            'PRAMANA': [
                'ཕྱོགས་ཆོས་', 'མཐུན་དཔེ་', 'མི་མཐུན་དཔེ་', 'ལྡོག་ཁྱབ་', 'རྗེས་ཁྱབ་',
                'གཞན་སེལ་', 'མ་གྲུབ་པའི་གཏན་ཚིགས་', 'འགལ་བའི་གཏན་ཚིགས་',
                '宗法性', '能立', '所立'  # 保留原来的中文词
            ],
            'MADHYAMIKA': [
                'རང་བཞིན་མེད་', 'བདེན་པར་གྲུབ་པ་མེད་', 'དབུ་མ་', 'དབུ་མ་ཐལ་འགྱུར་',
                'དབུ་མ་རང་རྒྱུད་', 'སྤྲོས་བྲལ་', 'མུ་བཞི་སྐྱེ་བ་བཀག་',
                '中道', '离四边'  # 保留原来的中文词
            ],
            'ABHISAMAYA': [
                'མངོན་རྟོགས་', 'རྣམ་མཁྱེན་', 'ལམ་ཤེས་', 'མངོན་རྟོགས་རྒྱན་',
                'སྐབས་བརྒྱད་', 'དོན་བདུན་ཅུ་', 'རྣམ་པ་ཉི་ཤུ་',
                '现观庄严', '八事七十义'  # 保留原来的中文词
            ],
            'ABHIDHARMA': [
                'སྐད་ཅིག་མ་', 'སེམས་བྱུང་', 'མཛོད་', 'ཁམས་བཅོ་བརྒྱད་',
                'སྐྱེ་མཆེད་བཅུ་གཉིས་', 'ཕུང་པོ་ལྔ་', 'ལྡན་མིན་འདུ་བྱེད་',
                '俱舍', '十八界', '七十五法'  # 保留原来的中文词
            ],
            'TANTRA': [
                'བསྐྱེད་རིམ་', 'རྫོགས་རིམ་', 'དཀྱིལ་འཁོར་', 'རྩ་རླུང་ཐིག་ལེ་',
                'དབང་བཞི་', 'སྨིན་གྲོལ་', 'གཏུམ་མོ་',
                '金刚乘', '无上瑜伽'  # 保留原来的中文词
            ],
            'DZOGCHEN': [
                'ཀ་དག་', 'ལྷུན་གྲུབ་', 'ཐོད་རྒལ་', 'ཁྲེགས་ཆོད་', 'རིག་པའི་རྩལ་',
                'གཞི་སྣང་', 'འཇའ་ལུས་', 'རྫོགས་པ་ཆེན་པོ་',
                '大圆满', '三身'  # 保留原来的中文词
            ],
            'VINAYA': [
                'སོ་ཐར་', 'ལྟུང་བ་', 'གསོ་སྦྱོང་', 'ཕམ་པ་', 'ལྷག་མ་',
                'བསྙེན་རྫོགས་', 'དབྱར་གནས་', 'དགག་དབྱེ་',
                '毗尼', '律藏'  # 保留原来的中文词
            ]
        }

    def _initialize_ambiguous_terms(self) -> List[str]:
        """初始化多义术语列表"""
        return [
            'ཆོས་', 'སེམས་', 'རྟགས་', 'སྒྲུབ་', 'ལམ་', 'འབྲས་བུ་',
            'རང་བཞིན་', 'དོན་', 'ཡིན་', 'ཡོད་', 'མེད་', 'ལྟ་བ་',
            'ཡེ་ཤེས་', 'སྟོང་པ་', 'ངོ་བོ་', 'བདག་', 'བདེན་པ་',
            'རྒྱུད་', 'གཟུགས་', 'དེ་ཁོ་ན་ཉིད་', 'རྟོགས་པ་', 'རིག་པ་',
            '法', '识', '果', '性', '有', '无', '见'  # 保留原来的中文词
        ]

    def _initialize_co_occurrence_patterns(self) -> Dict[str, List[List[str]]]:
        """初始化共现模式"""
        return {
            'PRAMANA': [
                ['རྟགས་', 'ཕྱོགས་ཆོས་', 'ཁྱབ་པ་'],  # 因+宗法+周遍
                ['གཏན་ཚིགས་', 'བསྒྲུབ་བྱ་', 'སྒྲུབ་བྱེད་'],  # 因+所立+能立
                ['ཡིན་ཏེ་', 'ཕྱིར་', 'ཡིན་པའི་']  # 论式标准结构
            ],
            'MADHYAMIKA': [
                ['སྟོང་པ་', 'རྟེན་འབྲེལ་'],  # 空+缘起
                ['དབུ་མ་', 'མཐའ་གཉིས་'],  # 中观+二边
                ['རང་བཞིན་', 'མེད་པ་', 'སྟོང་པ་ཉིད་']  # 自性+无+空性
            ],
            'ABHISAMAYA': [
                ['མངོན་རྟོགས་', 'སྦྱོར་བ་'],  # 现观+加行
                ['རྣམ་མཁྱེན་', 'ལམ་ཤེས་', 'གཞི་ཤེས་'],  # 一切相智+道智+基智
                ['སྐུ་གསུམ་', 'ཆོས་སྐུ་', 'ལོངས་སྐུ་', 'སྤྲུལ་སྐུ་']  # 三身+法身+报身+化身
            ],
            'ABHIDHARMA': [
                ['ཁམས་', 'སྐྱེ་མཆེད་', 'ཕུང་པོ་'],  # 界+处+蕴
                ['ཟག་བཅས་', 'ཟག་མེད་', 'འདུས་བྱས་'],  # 有漏+无漏+有为
                ['དགེ་བ་', 'མི་དགེ་བ་', 'ལུང་མ་བསྟན་']  # 善+不善+无记
            ],
            'TANTRA': [
                ['དབང་', 'དཀྱིལ་འཁོར་'],  # 灌顶+坛城
                ['བསྐྱེད་རིམ་', 'རྫོགས་རིམ་'],  # 生起次第+圆满次第
                ['རྩ་', 'རླུང་', 'ཐིག་ལེ་']  # 脉+气+明点
            ],
            'DZOGCHEN': [
                ['ཀ་དག་', 'ལྷུན་གྲུབ་'],  # 本净+任运
                ['རིག་པ་', 'གཞི་སྣང་'],  # 明觉+基现
                ['ཁྲེགས་ཆོད་', 'ཐོད་རྒལ་', 'སྐུ་གསུམ་']  # 立断+顿超+三身
            ]
        }

    def _initialize_function_keywords(self) -> Dict[str, List[str]]:
        """初始化功能关键词"""
        return {
            'PRACTICE_METHOD': [
                'སྒོམ་', 'ཉམས་ལེན་', 'བསྒྲུབ་', 'སྒྲུབ་ཐབས་', 'སྤྱད་པ་',
                'བསླབ་', 'ཉམས་སུ་བླངས་', 'འབད་', 'ཞུགས་', 'དམིགས་',
                'བསྒོམས་པས་', 'སྤྱོད་པ་', 'ཐབས་', 'སྒྲུབ་པ་པོ་',
                '修持', '修法', '实修', '行持', '修习', '观想'  # 保留原来的中文词
            ],
            'PHILOSOPHY': [
                'རྣམ་གཞག་', 'མཚན་ཉིད་', 'དབྱེ་བ་', 'རང་བཞིན་', 'ངོ་བོ་',
                'རིག་པ་', 'དགག་སྒྲུབ་', 'གྲུབ་མཐའ་', 'འདོད་པ་', 'སྨྲ་བ་',
                'གྲུབ་དོན་', 'ཡིན་ལུགས་', 'གནས་ལུགས་',
                '安立', '定义', '分类', '本质', '观点', '宗义'  # 保留原来的中文词
            ],
            'REALIZATION': [
                'རྟོགས་པ་', 'ཉམས་', 'མྱོང་བ་', 'མངོན་དུ་གྱུར་', 'གྲོལ་བ་',
                'རྟོགས་པ་སྐྱེས་', 'ཤར་', 'རྒྱུད་ལ་སྐྱེས་', 'ཡེ་ཤེས་', 'བདེ་བ་',
                '证悟', '觉受', '体验', '现前', '解脱', '智慧'  # 保留原来的中文词
            ],
            'RITUAL': [
                'ཆོ་ག་', 'མཆོད་པ་', 'བསྐང་བ་', 'བཟླས་པ་', 'གཏོར་མ་',
                'རབ་གནས་', 'སྦྱིན་སྲེག་', 'དབང་', 'སྒྲུབ་མཆོད་', 'མཆོད་རྟེན་',
                '仪轨', '供养', '酬补', '念诵', '施食', '灌顶'  # 保留原来的中文词
            ],
            'PROPER_NOUN': [
                'རྡོ་རྗེ་', 'པདྨ་', 'དཔལ་ལྡན་', 'བཅོམ་ལྡན་འདས་', 'ཤཱཀྱ་',
                'བྱང་ཆུབ་སེམས་དཔའ་', 'སངས་རྒྱས་', 'རྒྱལ་བ་', 'བླ་མ་',
                '金刚', '莲花', '世尊', '佛陀', '菩萨', '上师'  # 保留原来的中文词
            ]
        }

    def detect(self, text: str) -> Dict:
        """检测文本的语境和功能"""
        # 获取语境概率分布
        context_distribution = self.get_context_distribution(text)

        # 获取功能概率分布
        function_distribution = self.get_function_distribution(text)

        # 根据分布情况进行判断
        primary_context = max(context_distribution, key=context_distribution.get)
        primary_context_score = context_distribution[primary_context]

        primary_function = max(function_distribution, key=function_distribution.get)
        primary_function_score = function_distribution[primary_function]

        result = {
            'primary_context': primary_context,
            'context_confidence': primary_context_score,
            'context_distribution': context_distribution,
            'is_mixed_context': self._is_mixed_distribution(context_distribution),

            'primary_function': primary_function,
            'function_confidence': primary_function_score,
            'function_distribution': function_distribution,
        }

        # 如果是混合文本，识别次要语境
        if result['is_mixed_context']:
            result['secondary_contexts'] = self._get_secondary_contexts(context_distribution)

        return result

    def get_context_distribution(self, text: str) -> Dict[str, float]:
        """获取语境概率分布"""
        raw_scores = self._calculate_context_scores(text)

        # 归一化得分为概率分布
        total = sum(raw_scores.values())
        if total == 0:
            return {'GENERAL': 1.0}

        probabilities = {
            context: score / total for context, score in raw_scores.items()
        }

        return probabilities

    def get_function_distribution(self, text: str) -> Dict[str, float]:
        """获取功能概率分布"""
        function_scores = defaultdict(float)

        # 统计各功能关键词出现频率
        for function, keywords in self.function_keywords.items():
            for keyword in keywords:
                count = text.count(keyword)
                function_scores[function] += count

        # 特殊模式识别 - 使用扩展的高质量模式
        # 修行方法模式
        practice_pattern = r'བསྒོམ་པར་བྱ|ཉམས་སུ་ལེན|སྒྲུབ་པར་བྱེད|ཉམས་ལེན་གྱི་གནད|འདི་ལྟར་ཉམས་སུ་བླང|དམིགས་པ་གཏད|སྤྱོད་པ་ལ་བསླབ|གོམས་པར་བྱ|ཐབས་ལ་མཁས་པ|རྣལ་འབྱོར་པས་བྱ|གོ་རིམ་བཞིན་དུ་བསླབ|ཤེས་པར་བྱས་ནས་ཉམས་སུ་བླང|བྱ་བའི་ཚུལ་ནི|ལག་ལེན་དུ་འགྱེར|རྒྱུད་ལ་སྦྱང'
        if re.search(practice_pattern, text):
            function_scores['PRACTICE_METHOD'] += 2.0

        # 哲学概念模式
        philosophy_pattern = r'ཞེས་བྱ་བ.*?དོན་ནི|རྣམ་པར་བཞག|མཚན་ཉིད་ནི|དེ་ཁོ་ན་ཉིད་ནི|རྣམ་དབྱེ་ནི|དགོངས་པ་ནི|ཡིན་པའི་ཕྱིར|འདི་ལྟར་གསུངས་ཏེ|དབྱེ་ན་རྣམ་པ|གྲུབ་མཐའི་ལུགས|རིགས་པས་དཔྱད་ན|འདི་ལ་.*?ལན་ནི|རྣམ་གཞག་ནི|སྒྲ་བཤད་ནི|གྲུབ་དོན་ནི'
        if re.search(philosophy_pattern, text):
            function_scores['PHILOSOPHY'] += 2.0

        # 境界描述模式
        realization_pattern = r'མྱོང་བ་སྐྱེས|རྟོགས་པ་འཆར|ཉམས་སུ་མྱོང|མངོན་སུམ་དུ་གྱུར|ཞི་གནས་ཀྱི་ཉམས|ལྷག་མཐོང་གི་ཉམས|ཉམས་རྟོགས་སྐྱེས|མཐོང་བ་རྙེད|ཤར་བར་འགྱུར|གསལ་སྣང་འཆར|འོད་གསལ་ཤར|རྟོགས་པ་བརྙེས|ཡེ་ཤེས་སྐྱེས|ཚོར་བ་བྱུང|གྲོལ་བར་གྱུར'
        if re.search(realization_pattern, text):
            function_scores['REALIZATION'] += 2.0

        # 仪式指导模式
        ritual_pattern = r'ཆོ་ག་བྱེད|བཟླས་པ་བྱ|མཆོད་པ་འབུལ|ལས་རིམ་ནི|དཀྱིལ་འཁོར་བྲི|ཕྱག་རྒྱ་བཅིང|སྔགས་འདི་བཟླས|རབ་གནས་བྱ|གཏོར་མ་གཏོང|དབང་བསྐུར་བྱ|ཚོགས་འཁོར་བསྐོར|རྡུལ་ཚོན་གྱིས|བསང་གསོལ་བྱ|མཆོད་རྫས་བཤམ|རོལ་མོ་དཀྲོལ'
        if re.search(ritual_pattern, text):
            function_scores['RITUAL'] += 2.0

        # 归一化
        total = sum(function_scores.values())
        if total == 0:
            return {'GENERAL_TERM': 1.0}

        return {
            function: score / total for function, score in function_scores.items()
        }

    def _calculate_context_scores(self, text: str) -> Dict[str, float]:
        """计算各语境的得分"""
        scores = defaultdict(float)

        # 1. 特征短语匹配（高权重）
        scores = self._check_definitive_patterns(text, scores)

        # 2. 关键词评分
        for context, keywords in self.context_keywords.items():
            for keyword in keywords:
                # 计算加权分数
                weight = 1.0
                if context in self.unique_markers and keyword in self.unique_markers[context]:
                    weight = 3.0  # 高度特异性术语
                elif keyword in self.ambiguous_terms:
                    weight = 0.3  # 歧义术语

                # 统计词频并计分
                count = text.count(keyword)
                scores[context] += count * weight

        # 3. 术语共现分析
        for context, patterns in self.co_occurrence_patterns.items():
            for words in patterns:
                # 检查是否所有词都出现
                if all(word in text for word in words):
                    scores[context] += 3.0

                    # 检查词的接近程度
                    positions = [text.find(word) for word in words]
                    positions = [p for p in positions if p >= 0]
                    if positions:
                        span = max(positions) - min(positions)
                        # 如果词汇出现位置集中，增加权重
                        if span < 100:  # 在100字符范围内
                            scores[context] += 2.0

        # 4. 添加通用语境的基本分
        scores['GENERAL'] = 0.1

        return dict(scores)

    def _check_definitive_patterns(self, text: str, scores: Dict[str, float]) -> Dict[str, float]:
        """检查决定性的语境模式"""
        # 1. 因明学 (PRAMANA) 决定性模式
        pramana_patterns = [
            r'ཡིན་ཏེ།.*?ཡིན་པའི་ཕྱིར|ཡིན་ཏེ།.*?ཕྱིར་རོ',  # "是...因为是..." (典型论式结构)
            r'རྟགས་གྲུབ་སྟེ|ཁྱབ་པ་གྲུབ་སྟེ',  # "因已成立" "遍已成立"
            r'ཕྱོགས་ཆོས་གྲུབ|འབྲེལ་པ་ངེས',  # "宗法性成立" "关系确定"
            r'མ་གྲུབ་ན།.*?འགྲུབ་སྟེ',  # "若不成立...则成立"(辩论格式)
            r'རྟགས་མ་གྲུབ|ཁྱབ་པ་མ་གྲུབ|མ་ངེས|འགལ་',  # "因不成立" "遍不成立" "不定" "相违"
            r'ཁས་བླངས་.*?འགལ་|ཧ་ཅང་ཐལ་|ཐལ་བར་འགྱུར་',  # "与所许相违" "太过" "应成"
            r'རང་རིག་མངོན་སུམ་|དོན་སྤྱི་སྣང་བའི་རྟོག་པ',  # "自证现量" "显现义共相的分别"
        ]

        for pattern in pramana_patterns:
            if re.search(pattern, text):
                scores['PRAMANA'] += 3.0

        # 2. 中观学派 (MADHYAMIKA) 决定性模式
        madhyamika_patterns = [
            r'རང་བཞིན་གྱིས་མ་གྲུབ|རང་བཞིན་མེད་པ་ཡིན|སྟོང་པ་ཉིད་ཀྱི་དོན',  # "自性不成立" "是无自性" "空性的意义"
            r'སྤྲོས་པའི་མཐའ་དང་བྲལ|མཐའ་བཞི་དང་བྲལ',  # "离戏论边" "离四边"
            r'རང་བཞིན་གྱིས་སྟོང་|བདེན་པར་གྲུབ་པ་མེད',  # "自性空" "无真实成立"
            r'སྐྱེ་བ་བཀག|འགག་པ་བཀག|གཉིས་སྣང་.*?འཁྲུལ་པ',  # "遮破生起" "遮破灭亡" "二现...迷乱"
            r'མུ་བཞི་སྐྱེ་བ་བཀག|ཐལ་འགྱུར་བའི་ལུགས',  # "破四边生" "应成派的观点"
            r'བདེན་གཉིས་.*?དབྱེར་མེད|ཀུན་རྫོབ་དང་དོན་དམ',  # "二谛...不可分" "世俗谛与胜义谛"
            r'གཟུང་འཛིན་གཉིས་མེད|སྣང་སྟོང་དབྱེར་མེད',  # "所取能取无二" "显空不可分"
        ]

        for pattern in madhyamika_patterns:
            if re.search(pattern, text):
                scores['MADHYAMIKA'] += 3.0

        # 3. 唯识学派 (YOGACARA) 决定性模式
        yogacara_patterns = [
            r'རྣམ་པར་ཤེས་པ་ཙམ|སེམས་ཙམ་|ཀུན་གཞིའི་རྣམ་ཤེས',  # "唯识" "唯心" "阿赖耶识"
            r'ལངས་ཀར་གཤེགས་པའི་མདོ|དབྱིག་གཉེན|ཐོགས་མེད',  # "楞伽经" "世亲" "无著"
            r'བག་ཆགས་.*?ས་བོན|ཉོན་ཡིད|རྣམ་ཤེས་ཚོགས་བརྒྱད',  # "习气...种子" "末那识" "八识"
            r'གཞན་དབང་|ཀུན་བཏགས་|ཡོངས་གྲུབ',  # "依他起" "遍计所执" "圆成实"
            r'རང་རིག་རང་གསལ|སེམས་ཀྱི་ངོ་བོ་འོད་གསལ',  # "自明自证" "心之本性明光"
            r'རྣམ་རིག་ཙམ|གཉིས་མེད་ཀྱི་ཡེ་ཤེས|མངོན་པར་བྱང་ཆུབ',  # "唯识" "无二智" "现等觉"
        ]

        for pattern in yogacara_patterns:
            if re.search(pattern, text):
                scores['YOGACARA'] += 3.0

        # 4. 俱舍学派 (ABHIDHARMA) 决定性模式
        abhidharma_patterns = [
            r'ཆོས་མངོན་པ་མཛོད|ཕུང་པོ་ལྔ|སྐྱེ་མཆེད་བཅུ་གཉིས',  # "阿毗达磨俱舍" "五蕴" "十二处"
            r'ཁམས་བཅོ་བརྒྱད|རྟེན་འབྲེལ་ཡན་ལག་བཅུ་གཉིས',  # "十八界" "十二缘起支"
            r'དགེ་བ་བཅུ|མི་དགེ་བ་བཅུ|རྫས་.*?ཡོད་པ',  # "十善" "十不善" "实...存在"
            r'རྡུལ་ཕྲ་རབ|དུས་གསུམ་པོ|ཀུན་འགྲོའི་རྒྱུ',  # "极微" "三世" "遍行因"
            r'སྐད་ཅིག་མ|མཚུངས་ལྡན་ཀྱི་རྒྱུ|སོ་སོར་བརྟགས་འགོག',  # "刹那" "相应因" "择灭"
        ]

        for pattern in abhidharma_patterns:
            if re.search(pattern, text):
                scores['ABHIDHARMA'] += 3.0

        # 5. 密续 (TANTRA) 决定性模式
        tantra_patterns = [
            r'བསྐྱེད་རིམ་|རྫོགས་རིམ་|དབང་བསྐུར',  # "生起次第" "圆满次第" "灌顶"
            r'རྡོ་རྗེ་འཆང་|རིག་འཛིན་|ཕཊ་|ཧཱུྂ་',  # "持金刚" "持明" 和特殊种子字
            r'རྩ་|རླུང་|ཐིག་ལེ|གཏུམ་མོ',  # "脉" "气" "明点" "拙火"
            r'ཡབ་ཡུམ|དཀྱིལ་འཁོར་|གསང་སྔགས',  # "父母" "坛城" "密咒"
            r'སྙིང་པོའི་སྔགས|ཕྱག་རྒྱ་ཆེན་པོ|བདེ་སྟོང་',  # "心咒" "大手印" "乐空"
            r'ལྷན་ཅིག་སྐྱེས་པའི་ཡེ་ཤེས|གསང་བ་.*?རྒྱུད|དབང་བཞི',  # "俱生智" "密续" "四灌"
        ]

        for pattern in tantra_patterns:
            if re.search(pattern, text):
                scores['TANTRA'] += 3.0

        # 6. 大手印 (MAHAMUDRA) 决定性模式
        mahamudra_patterns = [
            r'ཕྱག་རྒྱ་ཆེན་པོ་|ཕྱག་ཆེན་|ལྷན་ཅིག་སྐྱེས་སྦྱོར',  # "大手印" "大手印" "俱生瑜伽"
            r'གནས་ལུགས་.*?སེམས་ཀྱི་ངོ་བོ|རིག་པ་.*?གཉུག་མ',  # "实相...心性" "觉性...本来"
            r'མཉམ་བཞག|རྗེས་ཐོབ|ཉམས་ལེན་བཞི',  # "入定" "出定" "四加行"
            r'རྣམ་རྟོག་མི་འགོག|ཤེས་པ་རང་སོར་བཞག',  # "不遮念头" "知性任运安住"
            r'གཞི་ལམ་འབྲས་གསུམ|རང་གྲོལ|རྟོག་མེད',  # "基道果三" "自解脱" "无分别"
            r'དྭངས་གསལ་མི་རྟོག་པ|ངོ་སྤྲོད་ཀྱི་གདམས་ངག',  # "澄明无分别" "直指要诀"
        ]

        for pattern in mahamudra_patterns:
            if re.search(pattern, text):
                scores['MAHAMUDRA'] += 3.0

        # 7. 大圆满 (DZOGCHEN) 决定性模式
        dzogchen_patterns = [
            r'རྫོགས་པ་ཆེན་པོ|འོད་གསལ་རྡོ་རྗེ་སྙིང་པོ',  # "大圆满" "光明金刚心"
            r'ཀ་དག|ལྷུན་གྲུབ|སྤྲོས་བྲལ',  # "本净" "任运" "离戏"
            r'ཐོད་རྒལ|ཁྲེགས་ཆོད|རིག་པ་.*?ཆེན་པོ',  # "顿超" "彻断" "大觉性"
            r'བྱང་ཆུབ་ཀྱི་སེམས|སེམས་ཉིད|ཀུན་ཏུ་བཟང་པོ',  # "菩提心" "心性" "普贤王"
            r'རང་བྱུང་ཡེ་ཤེས|རིག་པའི་རྩལ|རང་གྲོལ',  # "自生智慧" "觉性的游舞" "自解脱"
            r'གདོད་མའི་གཞི|གཞི་སྟོང་པ་ཆེན་པོ|བཞག་ཐབས་མེད',  # "本初基" "大空基" "无需刻意安住"
            r'འཁོར་འདས་རུ་ཤན་|ཆོས་ཉིད་ཟད་ས',  # "分离轮涅" "法性竭处"
        ]

        for pattern in dzogchen_patterns:
            if re.search(pattern, text):
                scores['DZOGCHEN'] += 3.0

        # 8. 净土派 (PURE LAND) 决定性模式
        pureland_patterns = [
            r'འོད་དཔག་མེད|བདེ་བ་ཅན|སྨོན་ལམ་.*?བདེ་བ་ཅན',  # "阿弥陀佛" "极乐" "往生极乐愿"
            r'ཟང་ཟིང་གི་མཆོད་པ|ཞིང་ཁམས་རྣམ་དག|བདེ་སྐྱིད',  # "财物供养" "清净刹土" "安乐"
            r'མགོན་པོ་འོད་དཔག་མེད|སྣང་བ་མཐའ་ཡས|ཚེ་དཔག་མེད',  # "怙主无量光" "无量光" "无量寿"
            r'དད་པས་གསོལ་བ་འདེབས|མོས་གུས་ཀྱི་སྒོ་ནས|སྐྱབས་གནས',  # "虔诚祈请" "恭敬之门" "皈依境"
        ]

        for pattern in pureland_patterns:
            if re.search(pattern, text):
                scores['PURE_LAND'] += 3.0

        return scores

    def _is_mixed_distribution(self, distribution: Dict[str, float]) -> bool:
        """判断是否为混合语境"""
        # 按得分降序排列
        scores = sorted(distribution.values(), reverse=True)

        # 如果第二高分超过最高分的60%，认为是混合文本
        if len(scores) > 1 and scores[1] > scores[0] * 0.6:
            return True

        # 或者如果有多个语境得分较高
        significant = [s for s in distribution.values() if s > 0.2]
        return len(significant) > 1

    def _get_secondary_contexts(self, distribution: Dict[str, float]) -> List[str]:
        """获取次要语境"""
        # 按得分降序排序
        sorted_contexts = sorted(
            distribution.items(),
            key=lambda x: x[1],
            reverse=True
        )[1:]  # 排除第一个（主语境）

        # 保留得分超过主语境30%的次要语境
        primary_score = max(distribution.values())
        threshold = primary_score * 0.3

        return [
            context for context, score in sorted_contexts
            if score > threshold
        ]

        def analyze_sample(self, text: str) -> None:
            """分析文本样本并打印结果（用于演示）"""
            result = self.detect(text)

            print("\n===== 文本语境分析 =====")
            print(f"主要语境: {result['primary_context']} (置信度: {result['context_confidence']:.2f})")

            if result['is_mixed_context']:
                print(f"次要语境: {', '.join(result['secondary_contexts'])}")

            print("\n语境分布:")
            for context, prob in sorted(result['context_distribution'].items(), key=lambda x: x[1], reverse=True):
                if prob > 0.05:  # 只显示概率超过5%的语境
                    print(f"  - {context}: {prob:.2f}")

            print(f"\n主要功能: {result['primary_function']} (置信度: {result['function_confidence']:.2f})")

            print("\n功能分布:")
            for function, prob in sorted(result['function_distribution'].items(), key=lambda x: x[1], reverse=True):
                if prob > 0.05:  # 只显示概率超过5%的功能
                    print(f"  - {function}: {prob:.2f}")

class TermDatabase:
    def __init__(self, db_path: Optional[Path] = None, context_detector=None):
        """初始化术语数据库"""

        self.db_path = db_path or Path("terms.db")
        self.conn = None
        self.cursor = None


        # 缓存常用术语
        self.cache = {}
        self.cache_size = 1000

        # 文本语境识别器
        self.context_detector = context_detector or FlexibleContextDetector()

        # 初始化数据库 - 只调用一次
        # self._init_database()

    def _init_database(self):
        """初始化数据库结构"""
        self.conn = sqlite3.connect(str(self.db_path))
        self.cursor = self.conn.cursor()

        # 主术语表
        self.cursor.execute('''
                            CREATE TABLE IF NOT EXISTS terms
                            (
                                id
                                INTEGER
                                PRIMARY
                                KEY
                                AUTOINCREMENT,
                                tibetan
                                TEXT
                                UNIQUE
                                NOT
                                NULL,
                                tibetan_wylie
                                TEXT,
                                pos
                                TEXT,
                                domain
                                TEXT,
                                etymology
                                TEXT,
                                literal_meaning
                                TEXT,
                                ambiguity_level
                                INTEGER
                                DEFAULT
                                1,
                                frequency
                                INTEGER
                                DEFAULT
                                0,
                                semantic_field
                                TEXT,
                                created_at
                                TIMESTAMP
                                DEFAULT
                                CURRENT_TIMESTAMP,
                                updated_at
                                TIMESTAMP
                                DEFAULT
                                CURRENT_TIMESTAMP
                            )
                            ''')

        # 术语翻译变体表
        self.cursor.execute('''
                            CREATE TABLE IF NOT EXISTS translations
                            (
                                id
                                INTEGER
                                PRIMARY
                                KEY
                                AUTOINCREMENT,
                                term_id
                                INTEGER
                                NOT
                                NULL,
                                chinese
                                TEXT
                                NOT
                                NULL,
                                context_label
                                TEXT,
                                function_label
                                TEXT,
                                priority
                                INTEGER
                                DEFAULT
                                5,
                                is_primary
                                BOOLEAN
                                DEFAULT
                                FALSE,
                                usage_notes
                                TEXT,
                                example_sentence
                                TEXT,
                                example_translation
                                TEXT,
                                reference_source
                                TEXT,
                                usage_count
                                INTEGER
                                DEFAULT
                                0,
                                interpretation_level
                                INTEGER
                                DEFAULT
                                3,
                                is_direct_translation
                                BOOLEAN
                                DEFAULT
                                FALSE,
                                FOREIGN
                                KEY
                            (
                                term_id
                            ) REFERENCES terms
                            (
                                id
                            ),
                                UNIQUE
                            (
                                term_id,
                                chinese,
                                context_label,
                                function_label
                            )
                                )
                            ''')

        # 术语组合表
        self.cursor.execute('''
                            CREATE TABLE IF NOT EXISTS term_combinations
                            (
                                id
                                INTEGER
                                PRIMARY
                                KEY
                                AUTOINCREMENT,
                                combined_tibetan
                                TEXT
                                UNIQUE
                                NOT
                                NULL,
                                combined_chinese
                                TEXT
                                NOT
                                NULL,
                                context_label
                                TEXT,
                                function_label
                                TEXT,
                                component_terms
                                TEXT,
                                formation_type
                                TEXT,
                                literal_meaning
                                TEXT,
                                semantic_meaning
                                TEXT,
                                priority
                                INTEGER
                                DEFAULT
                                5,
                                example_usage
                                TEXT
                            )
                            ''')

        # 概念关系表
        self.cursor.execute('''
                                CREATE TABLE concept_relations (
                                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                                    relation_type TEXT NOT NULL,  -- opposite, includes, stage, method_goal, related
                                    source_concept TEXT NOT NULL,
                                    target_concept TEXT NOT NULL,
                                    bidirectional BOOLEAN DEFAULT FALSE,
                                    confidence FLOAT DEFAULT 1.0,  -- 关系可信度
                                    source_type TEXT DEFAULT 'manual',  -- manual, extracted, learned
                                    reference_text TEXT,  -- 来源文本或引用
                                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                                )
                            ''')

        # 创建索引
        self.cursor.execute('CREATE INDEX IF NOT EXISTS idx_tibetan ON terms(tibetan)')
        self.cursor.execute(
            'CREATE INDEX IF NOT EXISTS idx_translations ON translations(term_id, context_label, function_label)')
        self.cursor.execute('CREATE INDEX IF NOT EXISTS idx_combinations ON term_combinations(combined_tibetan)')
        self.cursor.execute('CREATE INDEX idx_relation_source ON concept_relations(relation_type, source_concept)')
        self.cursor.execute('CREATE INDEX idx_relation_target ON concept_relations(relation_type, target_concept)')
        self.conn.commit()

    def add_term(self, tibetan: str, **kwargs) -> int:
        """
        添加或更新术语基本信息

        Args:
            tibetan: 藏文术语
            **kwargs: 可选字段，包括：
                - tibetan_wylie: 威利转写
                - pos: 词性
                - domain: 领域
                - etymology: 词源
                - literal_meaning: 字面意义
                - ambiguity_level: 多义性级别
                - frequency: 使用频率
                - semantic_field: 语义场

        Returns:
            术语ID
        """
        try:
            # 准备字段和值
            fields = ['tibetan']
            values = [tibetan]

            for key, value in kwargs.items():
                if key in ['tibetan_wylie', 'pos', 'domain', 'etymology',
                           'literal_meaning', 'ambiguity_level', 'frequency',
                           'semantic_field']:
                    fields.append(key)
                    values.append(value)

            # 构建SQL语句
            placeholder = '?' + ', ?' * (len(fields) - 1)
            fields_str = ', '.join(fields)

            # 插入或更新术语
            self.cursor.execute(f'''
                INSERT OR IGNORE INTO terms ({fields_str}) 
                VALUES ({placeholder})
            ''', values)

            # 获取术语ID
            self.cursor.execute('SELECT id FROM terms WHERE tibetan = ?', (tibetan,))
            term_id = self.cursor.fetchone()[0]

            # 如果已存在且有新数据，则更新
            if self.cursor.rowcount == 0 and len(fields) > 1:
                updates = ', '.join([f"{field} = ?" for field in fields[1:]])
                update_values = values[1:] + [tibetan]

                self.cursor.execute(f'''
                    UPDATE terms SET {updates}, updated_at = CURRENT_TIMESTAMP
                    WHERE tibetan = ?
                ''', update_values)

            self.conn.commit()
            return term_id

        except Exception as e:
            logger.error(f"Failed to add term: {e}")
            self.conn.rollback()
            return None

    def add_translation(self, term_id: int, chinese: str, **kwargs) -> int:
        """
        添加术语翻译

        Args:
            term_id: 术语ID
            chinese: 中文翻译
            **kwargs: 可选字段，包括：
                - context_label: 语境标签
                - function_label: 功能标签
                - priority: 优先级
                - is_primary: 是否主要翻译
                - usage_notes: 使用注释
                - example_sentence: 示例句子
                - example_translation: 示例翻译
                - reference_source: 参考来源
                - interpretation_level: 解释程度（1-5）
                - is_direct_translation: 是否直译
        Returns:
            翻译ID
        """
        try:
            # 准备字段和值
            fields = ['term_id', 'chinese']
            values = [term_id, chinese]

            for key, value in kwargs.items():
                if key in ['context_label', 'function_label', 'priority', 'is_primary',
                           'usage_notes', 'example_sentence', 'example_translation',
                           'reference_source', 'interpretation_level', 'is_direct_translation']:
                    fields.append(key)
                    values.append(value)

            # 构建SQL语句
            placeholder = '?' + ', ?' * (len(fields) - 1)
            fields_str = ', '.join(fields)

            # 插入或替换翻译
            self.cursor.execute(f'''
                INSERT OR REPLACE INTO translations ({fields_str})
                VALUES ({placeholder})
            ''', values)

            translation_id = self.cursor.lastrowid
            self.conn.commit()
            return translation_id

        except Exception as e:
            logger.error(f"Failed to add translation: {e}")
            self.conn.rollback()
            return None

    def set_global_translation_preferences(self, preferences: Dict) -> None:
        """设置全局翻译偏好"""
        self.global_preferences = preferences

        # 清空缓存以确保新设置生效
        self.cache = {}

    def add_term_with_translations(self, tibetan: str, translations: List[Dict]) -> bool:
        """
        添加术语及其所有翻译变体（保持兼容性）

        Args:
            tibetan: 藏文术语
            translations: 翻译列表，每个包含:
                - chinese: 中文翻译
                - context_label: 语境标签
                - function_label: 功能标签
                - priority: 优先级
                - is_primary: 是否为主要翻译
                - 其他可选字段：usage_notes, example_sentence等
        """
        try:
            # 添加术语
            term_id = self.add_term(tibetan)
            if not term_id:
                return False

            # 添加翻译变体
            for trans in translations:
                if 'chinese' not in trans:
                    continue

                # 提取翻译数据
                chinese = trans.pop('chinese')

                # 添加翻译
                self.add_translation(term_id, chinese, **trans)

            self.conn.commit()
            return True

        except Exception as e:
            logger.error(f"Failed to add term with translations: {e}")
            self.conn.rollback()
            return False

    def add_term_combination(self, combined_tibetan: str, combined_chinese: str, **kwargs) -> int:
        """
        添加术语组合并自动同步到概念关系

        Args:
            combined_tibetan: 组合藏文
            combined_chinese: 组合中文
            **kwargs: 可选字段，包括：
                - context_label: 语境标签
                - function_label: 功能标签
                - component_terms: 组成术语 (可以是列表或JSON字符串)
                - formation_type: 组合类型
                - literal_meaning: 字面意义
                - semantic_meaning: 实际语义
                - priority: 优先级
                - example_usage: 使用实例

        Returns:
            组合ID
        """
        try:
            # 准备字段和值
            fields = ['combined_tibetan', 'combined_chinese']
            values = [combined_tibetan, combined_chinese]

            # 保存component_terms供后续处理
            component_terms = None

            for key, value in kwargs.items():
                if key in ['context_label', 'function_label', 'component_terms',
                           'formation_type', 'literal_meaning', 'semantic_meaning',
                           'priority', 'example_usage']:
                    # 如果是组件列表，转为JSON
                    if key == 'component_terms':
                        component_terms = value  # 保存原始值
                        if isinstance(value, list):
                            value = json.dumps(value, ensure_ascii=False)
                    fields.append(key)
                    values.append(value)

            # 构建SQL语句
            placeholder = '?' + ', ?' * (len(fields) - 1)
            fields_str = ', '.join(fields)

            # 插入或替换组合
            self.cursor.execute(f'''
                INSERT OR REPLACE INTO term_combinations ({fields_str})
                VALUES ({placeholder})
            ''', values)

            combination_id = self.cursor.lastrowid

            # 处理概念关系同步 - 如果有组件术语，添加"包含"关系
            if component_terms:
                # 解析组件术语
                components = []
                if isinstance(component_terms, str):
                    try:
                        # 尝试解析JSON
                        components = json.loads(component_terms)
                    except json.JSONDecodeError:
                        # 如果不是JSON，按逗号分割
                        components = [term.strip() for term in component_terms.split(',')]
                elif isinstance(component_terms, list):
                    components = component_terms

                # 检查是否有concept_relations表
                self.cursor.execute('''
                                    SELECT name
                                    FROM sqlite_master
                                    WHERE type = 'table'
                                      AND name = 'concept_relations'
                                    ''')

                if self.cursor.fetchone():  # 表存在
                    # 为每个组件添加包含关系
                    for component in components:
                        # 如果组件是ID，需要查询对应的术语
                        if isinstance(component, int) or (isinstance(component, str) and component.isdigit()):
                            self.cursor.execute('SELECT tibetan FROM terms WHERE id = ?', (int(component),))
                            result = self.cursor.fetchone()
                            if result:
                                component = result[0]

                        # 检查关系是否已存在
                        self.cursor.execute('''
                                            SELECT id
                                            FROM concept_relations
                                            WHERE relation_type = 'includes'
                                              AND source_concept = ?
                                              AND target_concept = ?
                                            ''', (combined_tibetan, component))

                        if not self.cursor.fetchone():
                            # 添加包含关系
                            self.cursor.execute('''
                                                INSERT INTO concept_relations
                                                (relation_type, source_concept, target_concept,
                                                 bidirectional, source_type, confidence)
                                                VALUES (?, ?, ?, ?, ?, ?)
                                                ''', ('includes', combined_tibetan, component,
                                                      False, 'auto_synced_from_combination', 0.9))

            self.conn.commit()
            return combination_id

        except Exception as e:
            logger.error(f"添加术语组合失败: {e}")
            self.conn.rollback()
            return None

    def get_translation(self, tibetan: str, context: Dict, preferences: Dict = None) -> Tuple[str, float]:
        """
        根据上下文获取最佳翻译（保持兼容性）

        Args:
            tibetan: 藏文术语
            context: 上下文信息，包含:
                - text: 完整文本
                - position: 术语位置
                - detected_context: 检测到的语境标签
                - detected_function: 检测到的功能标签
                - surrounding_terms: 周围的术语
            preferences: 翻译偏好，包含:
                - prefer_direct: 偏好直译
                - max_interpretation_level: 最大解释程度
                - fallback_to_direct: 无合适译法时退回直译

        Returns:
            (翻译, 置信度)
        """
        # 设置默认偏好
        if preferences is None:
            preferences = {
                'prefer_direct': True,  # 偏好直译
                'max_interpretation_level': 3,  # 最大解释程度（1-5，1最直接，5最解释性）
                'fallback_to_direct': True  # 无合适译法时退回直译
            }

        # 检查缓存 (需要加入偏好参数)
        cache_key = (
            f"{tibetan}:{context.get('detected_context', '')}:"
            f"{context.get('detected_function', '')}:{str(preferences)}"
        )
        if cache_key in self.cache:
            return self.cache[cache_key]

        # 1. 首先检查是否为组合术语
        self.cursor.execute(
            'SELECT combined_chinese, priority FROM term_combinations WHERE combined_tibetan = ?',
            (tibetan,)
        )
        combination = self.cursor.fetchone()
        if combination:
            chinese, priority = combination
            confidence = min(priority / 10.0 + 0.3, 1.0)  # 组合术语有较高基础置信度

            # 更新缓存
            if len(self.cache) >= self.cache_size:
                self.cache.pop(next(iter(self.cache)))
            self.cache[cache_key] = (chinese, confidence)

            return chinese, confidence

            # 2. 获取所有可能的翻译
        self.cursor.execute('''
                            SELECT t.id,
                                   t.chinese,
                                   t.context_label,
                                   t.function_label,
                                   t.priority,
                                   t.is_primary,
                                   t.usage_count,
                                   t.example_sentence,
                                   t.is_direct_translation,
                                   t.interpretation_level
                            FROM translations t
                                     JOIN terms tm ON t.term_id = tm.id
                            WHERE tm.tibetan = ?
                            ORDER BY t.priority DESC, t.usage_count DESC
                            ''', (tibetan,))

        translations = self.cursor.fetchall()
        if not translations:
            return None, 0.0

        # 3. 应用过滤器
        filtered_translations = []
        direct_translation = None

        for trans in translations:
            trans_id, chinese, ctx_label, func_label, priority, is_primary, usage, example, is_direct, interp_level = trans

            # 保存直译选项，以备需要时使用
            if is_direct:
                direct_translation = trans

            # 应用解释程度过滤
            if interp_level > preferences.get('max_interpretation_level', 5):
                continue

            filtered_translations.append(trans)

        # 如果过滤后没有合适翻译且设置了退回直译
        if not filtered_translations and preferences.get('fallback_to_direct', True) and direct_translation:
            filtered_translations = [direct_translation]

        # 如果仍然没有可用翻译，返回原始结果
        if not filtered_translations:
            filtered_translations = translations

        # 4. 计算每个翻译的得分
        best_translation = None
        best_score = 0.0

        for trans in filtered_translations:
            trans_id, chinese, ctx_label, func_label, priority, is_primary, usage, example, is_direct, interp_level = trans

            # 基础得分
            score = priority / 10.0

            # 应用直译偏好
            if preferences.get('prefer_direct', False) and is_direct:
                score += 2.0  # 直译加分

            # 根据解释程度调整分数 (如果偏好直译，则解释程度越低分数越高)
            if preferences.get('prefer_direct', False):
                # 解释程度1-5，转换为0.8-0的降序分数
                score += (6 - interp_level) * 0.2

            # 语境匹配加分 (保持原有逻辑)
            if ctx_label == context.get('detected_context'):
                score += 3.0
            elif ctx_label == ContextLabel.GENERAL.value:
                score += 0.5

            # 功能匹配加分 (保持原有逻辑)
            if func_label == context.get('detected_function'):
                score += 2.0

            # 其他得分计算 (保持原有逻辑)
            if is_primary:
                score += 1.0
            score += min(usage / 1000.0, 1.0)

            if example and context.get('text'):
                similarity_score = self._calculate_context_similarity(example, context.get('text', ''))
                score += similarity_score

            if score > best_score:
                best_score = score
                best_translation = chinese

        # 计算置信度 (保持原有逻辑)
        confidence = min(best_score / 10.0, 1.0)

        # 更新缓存 (保持原有逻辑)
        if len(self.cache) >= self.cache_size:
            self.cache.pop(next(iter(self.cache)))
        self.cache[cache_key] = (best_translation, confidence)

        # 记录使用统计 (保持原有逻辑)
        self._update_usage_count(tibetan, best_translation)

        return best_translation, confidence

    def _calculate_context_similarity(self, example: str, context: str) -> float:
        """计算示例句子与当前上下文的相似度"""
        # 简单实现：检查关键词重叠
        example_words = set(example.split())
        context_words = set(context.split())

        overlap = example_words.intersection(context_words)
        if not example_words:
            return 0.0

        similarity = len(overlap) / len(example_words)
        return similarity * 0.5  # 最高加0.5分

    def _update_usage_count(self, tibetan: str, translation: str):
        """更新翻译使用计数"""
        try:
            self.cursor.execute('''
                                UPDATE translations
                                SET usage_count = usage_count + 1
                                WHERE chinese = ?
                                  AND term_id IN (SELECT id
                                                  FROM terms
                                                  WHERE tibetan = ?)
                                ''', (translation, tibetan))

            self.conn.commit()
        except Exception as e:
            logger.error(f"Failed to update usage count: {e}")

    def detect_text_context(self, text: str) -> Tuple[str, str]:
        """检测文本的主导语境和功能"""
        context_analysis = self.context_detector.detect(text)
        return context_analysis['primary_context'], context_analysis['primary_function']

    def get_mixed_context_strategy(self, text: str) -> Dict:
        """获取混合文本的处理策略"""
        return self.context_detector.detect(text)

    # 在 TermDatabase 类中添加的新方法

    def add_concept_relation(self, relation_type: str, source: str, target: str,
                             bidirectional: bool = False, confidence: float = 1.0,
                             source_type: str = 'manual', reference_text: str = None):
        """添加概念关系到数据库"""
        try:
            self.cursor.execute('''
                                INSERT INTO concept_relations
                                (relation_type, source_concept, target_concept, bidirectional,
                                 confidence, source_type, reference_text)
                                VALUES (?, ?, ?, ?, ?, ?, ?)
                                ''', (relation_type, source, target, bidirectional,
                                      confidence, source_type, reference_text))

            self.conn.commit()

            # 如果是"包含"关系，检查是否需要创建术语组合
            if relation_type == 'includes' and source_type != 'auto_synced_from_combination':
                # 查询该源概念的所有包含关系
                self.cursor.execute('''
                                    SELECT target_concept
                                    FROM concept_relations
                                    WHERE relation_type = 'includes'
                                      AND source_concept = ?
                                    ''', (source,))

                targets = [row[0] for row in self.cursor.fetchall()]

                # 如果有多个组件，考虑创建术语组合
                if len(targets) > 1:
                    # 检查是否已存在该组合
                    self.cursor.execute('''
                                        SELECT id
                                        FROM term_combinations
                                        WHERE combined_tibetan = ?
                                        ''', (source,))

                    if not self.cursor.fetchone():
                        # 查找源概念的中文翻译
                        chinese = self._get_term_translation(source)

                        # 创建术语组合
                        self.add_term_combination(
                            combined_tibetan=source,
                            combined_chinese=chinese or source,
                            component_terms=json.dumps(targets),
                            formation_type='auto_from_relations'
                        )

            self.conn.commit()
            return True
        except Exception as e:
            logger.error(f"添加概念关系失败: {e}")
            self.conn.rollback()
            return False

    def _get_term_translation(self, tibetan):
        """获取术语的中文翻译"""
        try:
            # 首先查找主要翻译
            self.cursor.execute('''
                                SELECT t.chinese
                                FROM terms
                                         JOIN translations t ON terms.id = t.term_id
                                WHERE terms.tibetan = ?
                                  AND t.is_primary = 1
                                ''', (tibetan,))

            result = self.cursor.fetchone()
            if result:
                return result[0]

            # 如果没有主要翻译，查找任何翻译
            self.cursor.execute('''
                                SELECT t.chinese
                                FROM terms
                                         JOIN translations t ON terms.id = t.term_id
                                WHERE terms.tibetan = ? LIMIT 1
                                ''', (tibetan,))

            result = self.cursor.fetchone()
            if result:
                return result[0]

            return None
        except Exception as e:
            logger.warning(f"获取术语翻译失败: {e}")
            return None

    def get_concept_relations(self, concept: str = None, relation_type: str = None):
        """获取概念关系"""
        try:
            query = "SELECT * FROM concept_relations WHERE 1=1"
            params = []

            if concept:
                query += " AND (source_concept = ? OR target_concept = ?)"
                params.extend([concept, concept])

            if relation_type:
                query += " AND relation_type = ?"
                params.append(relation_type)

            self.cursor.execute(query, params)
            return self.cursor.fetchall()
        except Exception as e:
            logger.error(f"获取概念关系失败: {e}")
            return []

    def delete_concept_relation(self, relation_id: int):
        """删除概念关系"""
        try:
            self.cursor.execute("DELETE FROM concept_relations WHERE id = ?", (relation_id,))
            self.conn.commit()
            return True
        except Exception as e:
            logger.error(f"删除概念关系失败: {e}")
            self.conn.rollback()
            return False

    # 添加到 TermDatabase 类

    def sync_term_combinations_to_relations(self):
        """同步术语组合到概念关系"""
        self.cursor.execute('''
                            SELECT combined_tibetan, combined_chinese, component_terms
                            FROM term_combinations
                            ''')

        # 获取所有术语组合
        combinations = self.cursor.fetchall()
        relations_added = 0

        for combined_tibetan, combined_chinese, component_terms in combinations:
            # 解析组件术语（假设是JSON格式存储的术语ID或名称列表）
            try:
                if component_terms:
                    components = json.loads(component_terms)

                    # 如果组件是ID，需要查询对应的术语
                    if isinstance(components[0], int):
                        component_names = []
                        for comp_id in components:
                            self.cursor.execute('SELECT tibetan FROM terms WHERE id = ?', (comp_id,))
                            result = self.cursor.fetchone()
                            if result:
                                component_names.append(result[0])
                    else:
                        component_names = components

                    # 创建包含关系 - 组合术语包含其组件
                    for component in component_names:
                        # 检查关系是否已存在
                        self.cursor.execute('''
                                            SELECT id
                                            FROM concept_relations
                                            WHERE relation_type = 'includes'
                                              AND source_concept = ?
                                              AND target_concept = ?
                                            ''', (combined_tibetan, component))

                        if not self.cursor.fetchone():
                            self.cursor.execute('''
                                                INSERT INTO concept_relations
                                                (relation_type, source_concept, target_concept, bidirectional,
                                                 source_type, confidence)
                                                VALUES (?, ?, ?, ?, ?, ?)
                                                ''', ('includes', combined_tibetan, component, False,
                                                      'auto_synced', 0.9))
                            relations_added += 1
            except (json.JSONDecodeError, TypeError, IndexError) as e:
                logger.warning(f"解析组件术语失败: {component_terms}, 错误: {e}")
                continue

        self.conn.commit()
        return relations_added

    def sync_relations_to_term_combinations(self):
        """同步概念关系到术语组合"""
        # 获取所有"包含"关系
        self.cursor.execute('''
                            SELECT source_concept, target_concept
                            FROM concept_relations
                            WHERE relation_type = 'includes'
                            ''')

        relations = self.cursor.fetchall()
        combinations_added = 0

        # 按源概念分组
        relation_groups = {}
        for source, target in relations:
            if source not in relation_groups:
                relation_groups[source] = []
            relation_groups[source].append(target)

        # 处理每个源概念及其包含的所有目标
        for source, targets in relation_groups.items():
            if len(targets) > 1:  # 只处理包含多个组件的概念
                # 检查是否已存在术语组合
                self.cursor.execute('''
                                    SELECT id
                                    FROM term_combinations
                                    WHERE combined_tibetan = ?
                                    ''', (source,))

                if not self.cursor.fetchone():
                    # 查找源概念的中文翻译
                    self.cursor.execute('''
                                        SELECT t.chinese
                                        FROM terms
                                                 JOIN translations t ON terms.id = t.term_id
                                        WHERE terms.tibetan = ?
                                          AND t.is_primary = 1
                                        ''', (source,))

                    chinese = self.cursor.fetchone()
                    if chinese:
                        chinese = chinese[0]
                    else:
                        chinese = source  # 如果没有找到翻译，使用原文

                    # 将组件列表存储为JSON
                    component_terms_json = json.dumps(targets)

                    # 添加到术语组合表
                    self.cursor.execute('''
                                        INSERT INTO term_combinations
                                            (combined_tibetan, combined_chinese, component_terms, formation_type)
                                        VALUES (?, ?, ?, ?)
                                        ''', (source, chinese, component_terms_json, 'auto_from_relations'))

                    combinations_added += 1

        self.conn.commit()
        return combinations_added

    def ensure_consistency(self):
        """确保各表之间的一致性"""
        # 1. 同步术语组合到概念关系
        relations_added = self.sync_term_combinations_to_relations()

        # 2. 同步概念关系到术语组合
        combinations_added = self.sync_relations_to_term_combinations()

        # 3. 统一上下文标签
        contexts_unified = self.unify_context_labels()

        return {
            'relations_added': relations_added,
            'combinations_added': combinations_added,
            'contexts_unified': contexts_unified
        }

    def unify_context_labels(self):
        """统一各表中的上下文标签"""
        # 获取所有使用的上下文标签
        context_labels = set()

        # 从翻译表获取
        self.cursor.execute('SELECT DISTINCT context_label FROM translations WHERE context_label IS NOT NULL')
        for row in self.cursor.fetchall():
            if row[0]:
                context_labels.add(row[0])

        # 从术语组合表获取
        self.cursor.execute('SELECT DISTINCT context_label FROM term_combinations WHERE context_label IS NOT NULL')
        for row in self.cursor.fetchall():
            if row[0]:
                context_labels.add(row[0])

        # 创建标准化映射（可选：手动维护一个标准化映射表）
        normalized_labels = {}
        for label in context_labels:
            # 简单的标准化例子：小写并移除多余空格
            normalized = label.lower().strip()
            normalized_labels[label] = normalized

        # 应用标准化 - 更新翻译表
        updates = 0
        for original, normalized in normalized_labels.items():
            if original != normalized:
                self.cursor.execute(
                    'UPDATE translations SET context_label = ? WHERE context_label = ?',
                    (normalized, original)
                )
                updates += self.cursor.rowcount

                self.cursor.execute(
                    'UPDATE term_combinations SET context_label = ? WHERE context_label = ?',
                    (normalized, original)
                )
                updates += self.cursor.rowcount

        self.conn.commit()
        return updates

    def get_related_terms(self, term: str, relation_type: Optional[str] = None) -> List[Dict]:
        """
        获取与给定术语相关的其他术语

        Args:
            term: 查询术语
            relation_type: 关系类型（可选）

        Returns:
            相关术语列表
        """
        try:
            if relation_type:
                # 查询特定类型的关系
                self.cursor.execute("""
                                    SELECT r.relation_type,
                                           r.source,
                                           r.target,
                                           r.bidirectional,
                                           r.confidence,
                                           t1.chinese AS source_chinese,
                                           t2.chinese AS target_chinese
                                    FROM concept_relations r
                                             LEFT JOIN translations t1 ON r.source = t1.tibetan
                                             LEFT JOIN translations t2 ON r.target = t2.tibetan
                                    WHERE (r.source = ? OR (r.target = ? AND r.bidirectional = 1))
                                      AND r.relation_type = ?
                                    ORDER BY r.confidence DESC
                                    """, (term, term, relation_type))
            else:
                # 查询所有类型的关系
                self.cursor.execute("""
                                    SELECT r.relation_type,
                                           r.source,
                                           r.target,
                                           r.bidirectional,
                                           r.confidence,
                                           t1.chinese AS source_chinese,
                                           t2.chinese AS target_chinese
                                    FROM concept_relations r
                                             LEFT JOIN translations t1 ON r.source = t1.tibetan
                                             LEFT JOIN translations t2 ON r.target = t2.tibetan
                                    WHERE r.source = ?
                                       OR (r.target = ? AND r.bidirectional = 1)
                                    ORDER BY r.confidence DESC
                                    """, (term, term))

            rows = self.cursor.fetchall()

            # 转换为字典列表
            result = []
            for row in rows:
                relation_type, source, target, bidirectional, confidence, source_chinese, target_chinese = row

                # 确定相关术语
                related_term = target if source == term else source
                related_chinese = target_chinese if source == term else source_chinese

                result.append({
                    'relation_type': relation_type,
                    'term': related_term,
                    'chinese': related_chinese,
                    'bidirectional': bool(bidirectional),
                    'confidence': confidence,
                    'is_source': source == term
                })

            return result

        except Exception as e:
            logger.error(f"获取相关术语失败: {e}")
            return []