"""
NLLB模型封装器
"""
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from typing import List, Dict, Optional
import logging

logger = logging.getLogger(__name__)


class NLLBTranslator:
    def __init__(self, model_name: str = "facebook/nllb-200-distilled-600M",
                 device: Optional[str] = None):
        """初始化NLLB翻译器"""
        # 自动选择设备
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        logger.info(f"Loading NLLB model on {self.device}...")

        # 加载模型和分词器
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()

        # 语言代码
        self.source_lang = "bod_Tibt"  # 藏语
        self.target_lang = "zho_Hans"  # 简体中文

        # 翻译参数
        self.max_length = 512
        self.num_beams = 5
        self.temperature = 0.9

        logger.info("NLLB model loaded successfully")

    def translate(self, text: str, **kwargs) -> str:
        """
        翻译单个文本
        """
        # 更新参数
        max_length = kwargs.get('max_length', self.max_length)
        num_beams = kwargs.get('num_beams', self.num_beams)
        temperature = kwargs.get('temperature', self.temperature)

        try:
            # 设置源语言
            self.tokenizer.src_lang = self.source_lang

            # 编码输入
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_length
            ).to(self.device)

            # 生成翻译
            with torch.no_grad():
                generated_tokens = self.model.generate(
                    **inputs,
                    forced_bos_token_id=self.tokenizer.lang_code_to_id[self.target_lang],
                    max_length=max_length,
                    num_beams=num_beams,
                    temperature=temperature,
                    do_sample=False,  # 使用束搜索
                    early_stopping=True
                )

            # 解码输出
            translation = self.tokenizer.batch_decode(
                generated_tokens,
                skip_special_tokens=True
            )[0]

            return translation

        except Exception as e:
            logger.error(f"Translation error: {str(e)}")
            return ""

    def translate_batch(self, texts: List[str], **kwargs) -> List[str]:
        """
        批量翻译
        """
        # 更新参数
        max_length = kwargs.get('max_length', self.max_length)
        num_beams = kwargs.get('num_beams', self.num_beams)
        temperature = kwargs.get('temperature', self.temperature)
        batch_size = kwargs.get('batch_size', 8)

        translations = []

        # 分批处理
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]

            try:
                # 设置源语言
                self.tokenizer.src_lang = self.source_lang

                # 编码输入
                inputs = self.tokenizer(
                    batch_texts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=max_length
                ).to(self.device)

                # 生成翻译
                with torch.no_grad():
                    generated_tokens = self.model.generate(
                        **inputs,
                        forced_bos_token_id=self.tokenizer.lang_code_to_id[self.target_lang],
                        max_length=max_length,
                        num_beams=num_beams,
                        temperature=temperature,
                        do_sample=False,
                        early_stopping=True
                    )

                # 解码输出
                batch_translations = self.tokenizer.batch_decode(
                    generated_tokens,
                    skip_special_tokens=True
                )

                translations.extend(batch_translations)

            except Exception as e:
                logger.error(f"Batch translation error: {str(e)}")
                # 对失败的批次返回空字符串
                translations.extend([""] * len(batch_texts))

        return translations

    def translate_with_alternatives(self, text: str, num_alternatives: int = 3) -> List[str]:
        """
        生成多个翻译候选
        """
        try:
            # 设置源语言
            self.tokenizer.src_lang = self.source_lang

            # 编码输入
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.max_length
            ).to(self.device)

            # 生成多个翻译
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    forced_bos_token_id=self.tokenizer.lang_code_to_id[self.target_lang],
                    max_length=self.max_length,
                    num_beams=num_alternatives * 2,
                    num_return_sequences=num_alternatives,
                    temperature=self.temperature,
                    do_sample=True,
                    early_stopping=True
                )

            # 解码所有输出
            translations = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)

            # 去重
            unique_translations = []
            seen = set()
            for trans in translations:
                if trans not in seen:
                    seen.add(trans)
                    unique_translations.append(trans)

            return unique_translations[:num_alternatives]

        except Exception as e:
            logger.error(f"Alternative translation error: {str(e)}")
            return [self.translate(text)]  # 退回到单个翻译

    def get_attention_scores(self, text: str) -> Dict:
        """
        获取注意力分数（用于分析）
        """
        try:
            # 设置源语言
            self.tokenizer.src_lang = self.source_lang

            # 编码输入
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.max_length
            ).to(self.device)

            # 前向传播并获取注意力
            with torch.no_grad():
                outputs = self.model(
                    **inputs,
                    decoder_input_ids=inputs['input_ids'],
                    output_attentions=True
                )

            # 提取注意力分数
            encoder_attentions = outputs.encoder_attentions
            decoder_attentions = outputs.decoder_attentions
            cross_attentions = outputs.cross_attentions

            return {
                'encoder_attentions': encoder_attentions,
                'decoder_attentions': decoder_attentions,
                'cross_attentions': cross_attentions
            }

        except Exception as e:
            logger.error(f"Attention extraction error: {str(e)}")
            return {}