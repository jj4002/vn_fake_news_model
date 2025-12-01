# services/text_pipeline.py
import logging
from typing import Optional

from services.text_normalizer import VietnameseTextNormalizer
from services.ngram_normalizer import NGramLanguageModel, VNSpellCorrector


logger = logging.getLogger(__name__)


class VNTextPipeline:
    def __init__(
        self,
        bi_path: str = "data/bi_gram.txt",
        tri_path: str = "data/tri_gram.txt",
        four_path: str = "data/four_gram.txt",
    ):
        # 1) VnCoreNLP để word segmentation
        # self.seg = VietnameseTextNormalizer()   # ✅ dùng class thật

        # 2) N-gram LM + spell corrector
        # self.lm = NGramLanguageModel(
        #     bi_path=bi_path,
        #     tri_path=tri_path,
        #     four_path=four_path,
        # )
        # self.spell = VNSpellCorrector(self.lm)
        self.seg = None
        self.lm = None
        self.spell = None
        

    def normalize_text(self, text: Optional[str]) -> str:
        if not text:
            return ""
        raw = text.strip()
        if not raw:
            return ""

        # Tạm thời: chỉ trả về raw cho nhanh (bạn đang làm đúng)
        return raw
        # Sau này muốn bật full pipeline thì thay bằng:
        # seg = self.seg.word_segment(raw)
        # seg_plain = seg.replace("_", " ")
        # fixed = self.spell.correct_sentence(seg_plain)
        # return fixed


# Singleton cho toàn app
_vn_text_pipeline: Optional[VNTextPipeline] = None

def get_text_pipeline() -> VNTextPipeline:
    global _vn_text_pipeline
    if _vn_text_pipeline is None:
        _vn_text_pipeline = VNTextPipeline()
    return _vn_text_pipeline
