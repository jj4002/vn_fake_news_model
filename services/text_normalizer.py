# services/text_normalizer.py
import re
import logging
from pathlib import Path
from typing import Optional, List
import os

logger = logging.getLogger(__name__)

try:
    from vncorenlp import VnCoreNLP
    VNCORENLP_AVAILABLE = True
except ImportError:
    VNCORENLP_AVAILABLE = False
    logger.warning("⚠️ vncorenlp not available")


class VietnameseTextNormalizer:
    """Vietnamese text normalizer using VnCoreNLP"""
    
    def __init__(self, vncorenlp_path: Optional[str] = None):
        self.annotator = None
        
        if VNCORENLP_AVAILABLE:
            try:
                # Find VnCoreNLP JAR
                jar_path = vncorenlp_path or self._find_vncorenlp_jar()
                
                if jar_path and Path(jar_path).exists():
                    # Start VnCoreNLP server
                    self.annotator = VnCoreNLP(
                        jar_path, 
                        annotators="wseg,pos",  # Word segmentation + POS tagging
                        max_heap_size='-Xmx2g'
                    )
                    logger.info(f"✅ VnCoreNLP initialized: {jar_path}")
                else:
                    logger.error(f"❌ VnCoreNLP JAR not found at: {jar_path}")
                    logger.info("Run: python scripts/setup_vncorenlp.py")
                    
            except Exception as e:
                logger.error(f"❌ VnCoreNLP init failed: {e}")
                self.annotator = None
    
    def _find_vncorenlp_jar(self) -> Optional[str]:
        """Find VnCoreNLP JAR file in ./vncorenlp"""
        base = Path(__file__).parent.parent / "vncorenlp"
        candidates = list(base.glob("VnCoreNLP-*.jar"))
        if candidates:
            return str(candidates[0])
        return None

    
    def word_segment(self, text: str) -> str:
        """Word segmentation using VnCoreNLP"""
        if not self.annotator or not text:
            return text
        
        try:
            # VnCoreNLP word segmentation
            sentences = self.annotator.tokenize(text)
            
            # Join words with underscore
            segmented = ' '.join([' '.join(sentence) for sentence in sentences])
            return segmented
            
        except Exception as e:
            logger.error(f"❌ Word segmentation error: {e}")
            return text
    
    def pos_tag(self, text: str) -> List[tuple]:
        """POS tagging using VnCoreNLP"""
        if not self.annotator or not text:
            return []
        
        try:
            annotated = self.annotator.annotate(text)
            
            # Extract (word, pos) pairs
            pos_tags = []
            for sentence in annotated['sentences']:
                for i, word in enumerate(sentence):
                    pos = sentence[i].get('posTag', 'X')
                    pos_tags.append((word['form'], pos))
            
            return pos_tags
            
        except Exception as e:
            logger.error(f"❌ POS tagging error: {e}")
            return []
    
    def normalize(self, text: str, use_segmentation: bool = True) -> str:
        """
        Normalize Vietnamese text
        
        Args:
            text: Input text
            use_segmentation: Use VnCoreNLP word segmentation (recommended)
        
        Returns:
            Normalized text with underscores for compound words
        """
        if not text or len(text.strip()) == 0:
            return text
        
        # Preprocessing
        text = text.lower().strip()
        
        # Remove extra spaces
        text = re.sub(r'\s+', ' ', text)
        
        # Fix spacing around numbers
        text = re.sub(r'(\d+)([a-zà-ỹ])', r'\1 \2', text)
        text = re.sub(r'([a-zà-ỹ])(\d+)', r'\1 \2', text)
        
        # Word segmentation (if available)
        if use_segmentation and self.annotator:
            text = self.word_segment(text)
        
        # Remove extra spaces again
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def normalize_for_model(self, text: str) -> str:
        """
        Normalize and convert to space-separated format for PhoBERT
        
        PhoBERT expects: "video quay lại vụ cháy" (no underscores)
        """
        # Get segmented text with underscores
        segmented = self.normalize(text, use_segmentation=True)
        
        # Replace underscores with spaces for PhoBERT
        model_input = segmented.replace('_', ' ')
        
        return model_input
    
    def normalize_batch(self, texts: list, use_segmentation: bool = True) -> list:
        """Normalize multiple texts"""
        return [self.normalize(t, use_segmentation) for t in texts]
    
    def __del__(self):
        """Cleanup VnCoreNLP server"""
        if self.annotator:
            try:
                self.annotator.close()
                logger.info("🔌 VnCoreNLP closed")
            except:
                pass


# ✅ Singleton instance
_normalizer_instance = None

def get_normalizer() -> VietnameseTextNormalizer:
    """Get singleton normalizer instance"""
    global _normalizer_instance
    if _normalizer_instance is None:
        _normalizer_instance = VietnameseTextNormalizer()
    return _normalizer_instance
