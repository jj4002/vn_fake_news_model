# services/ocr_service.py
import logging
from typing import List
import numpy as np
from PIL import Image
import cv2

logger = logging.getLogger(__name__)

try:
    from vietocr.tool.predictor import Predictor
    from vietocr.tool.config import Cfg
    HAS_VIETOCR = True
except ImportError:
    HAS_VIETOCR = False
    logger.warning("VietOCR not available")


class OCRService:
    def __init__(self):
        self.available = HAS_VIETOCR
        self.predictor = None
        
        if self.available:
            try:
                # Load VietOCR config
                config = Cfg.load_config_from_name('vgg_transformer')
                config['device'] = 'cpu'
                config['predictor']['beamsearch'] = False  # Tắt beamsearch cho nhanh hơn
                
                self.predictor = Predictor(config)
                logger.info("✅ VietOCR loaded (Vietnamese optimized)")
            except Exception as e:
                logger.error(f"Failed to load VietOCR: {e}")
                self.available = False
        else:
            logger.warning("VietOCR not available - OCR will be disabled")
    
    def extract_text_from_frames(self, frames: List[np.ndarray]) -> str:
        """Extract text from multiple frames using VietOCR"""
        
        if not self.available or not frames:
            logger.warning("VietOCR not available or no frames")
            return ""
        
        try:
            all_text = []
            seen_text = set()  # Deduplicate
            
            for i, frame in enumerate(frames):
                try:
                    # Convert BGR (OpenCV) to RGB (PIL)
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    
                    # Convert to PIL Image
                    pil_image = Image.fromarray(frame_rgb)
                    
                    # VietOCR predict
                    text = self.predictor.predict(pil_image)
                    
                    # Clean and deduplicate
                    text = text.strip()
                    if text and text not in seen_text:
                        all_text.append(text)
                        seen_text.add(text)
                        logger.debug(f"Frame {i}: {text[:50]}...")
                
                except Exception as e:
                    logger.error(f"Error processing frame {i}: {e}")
                    continue
            
            # Combine all text
            combined = ' '.join(all_text)
            logger.info(f"✅ VietOCR extracted {len(combined)} chars from {len(frames)} frames")
            return combined
            
        except Exception as e:
            logger.error(f"VietOCR error: {e}")
            return ""
