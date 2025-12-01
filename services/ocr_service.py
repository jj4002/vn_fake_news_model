# services/ocr_service.py
import logging
from typing import List, Optional
import numpy as np

logger = logging.getLogger(__name__)

try:
    import easyocr
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False
    logger.warning("EasyOCR not available")

class OCRService:
    def __init__(self):
        self.available = OCR_AVAILABLE
        
        if self.available:
            try:
                self.reader = easyocr.Reader(['vi', 'en'], gpu=False)
                logger.info("✅ EasyOCR loaded (Vietnamese + English)")
            except Exception as e:
                logger.error(f"Failed to load EasyOCR: {e}")
                self.available = False
        else:
            self.reader = None
    
    def extract_text_from_frames(self, frames: List[np.ndarray]) -> str:
        """Extract text from multiple frames"""
        
        if not self.available or not frames:
            return ""
        
        try:
            all_text = []
            
            for i, frame in enumerate(frames):
                results = self.reader.readtext(frame)
                
                # Extract text only
                frame_text = ' '.join([text for (bbox, text, prob) in results if prob > 0.3])
                
                if frame_text:
                    all_text.append(frame_text)
                    logger.debug(f"Frame {i}: {frame_text[:50]}...")
            
            # Combine and deduplicate
            combined = ' '.join(all_text)
            return combined.strip()
            
        except Exception as e:
            logger.error(f"OCR error: {e}")
            return ""
