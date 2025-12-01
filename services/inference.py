# services/inference.py
import onnxruntime as ort
import numpy as np
from transformers import AutoTokenizer
from typing import Dict
import os
import logging

logger = logging.getLogger(__name__)

class ONNXInference:
    """ONNX model inference service"""
    
    def __init__(self):
        model_path = os.getenv("MODEL_PATH", "./models/phobert_fakenews_int8.onnx")
        tokenizer_path = os.getenv("TOKENIZER_PATH", "./models/tokenizer")
        
        logger.info(f"Loading model: {model_path}")
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        # Load ONNX model
        self.session = ort.InferenceSession(
            model_path,
            providers=['CPUExecutionProvider']
        )
        
        # ✅ FIX: Load tokenizer từ local hoặc HuggingFace
        logger.info(f"Loading tokenizer: {tokenizer_path}")
        
        if os.path.exists(tokenizer_path):
            # Load từ local folder
            logger.info("Loading tokenizer from local path...")
            self.tokenizer = AutoTokenizer.from_pretrained(
                tokenizer_path,
                local_files_only=True  # ← Chỉ dùng file local
            )
        else:
            # Fallback: Load từ HuggingFace (PhoBERT official)
            logger.warning(f"Local tokenizer not found at {tokenizer_path}")
            logger.info("Downloading PhoBERT tokenizer from HuggingFace...")
            self.tokenizer = AutoTokenizer.from_pretrained(
                "vinai/phobert-base-v2"  # ← PhoBERT tokenizer chính thức
            )
        
        # Label token IDs
        self.label_token_ids = {
            0: self.tokenizer.convert_tokens_to_ids("thật"),
            1: self.tokenizer.convert_tokens_to_ids("giả")
        }
        
        logger.info("✅ Model loaded successfully")
        logger.info(f"   Label tokens: {self.label_token_ids}")
    
    def _truncate_text(self, text: str, max_chars: int = 400) -> str:
        """Smart truncation"""
        if len(text) <= max_chars:
            return text
        
        first_part = int(max_chars * 0.6)
        last_part = int(max_chars * 0.4)
        
        return text[:first_part] + " [...] " + text[-last_part:]
    
    def predict(self, title: str, content: str, threshold: float = 0.5) -> Dict:
        """Base model prediction"""
        
        # Truncate content
        content_trunc = self._truncate_text(content, max_chars=500)
        
        # Build prompt (mask at beginning)
        mask_token = self.tokenizer.mask_token
        prompt = f"Bài viết này là {mask_token} . Tiêu_đề : {title} . Nội_dung : {content_trunc}"
        
        # Tokenize
        inputs = self.tokenizer(
            prompt,
            max_length=256,
            padding='max_length',
            truncation=True,
            return_tensors='np'
        )
        
        # Check mask token exists
        mask_token_id = self.tokenizer.mask_token_id
        if mask_token_id not in inputs['input_ids'][0]:
            logger.error("Mask token lost during tokenization!")
            raise ValueError("Input too long, mask token was truncated")
        
        # Run inference
        outputs = self.session.run(
            None,
            {
                'input_ids': inputs['input_ids'].astype(np.int64),
                'attention_mask': inputs['attention_mask'].astype(np.int64)
            }
        )
        
        logits = outputs[0]
        
        # Find mask position
        mask_pos = np.where(inputs['input_ids'][0] == mask_token_id)[0][0]
        
        # Extract logits at mask
        mask_logits = logits[0, mask_pos, :]
        label_logits = mask_logits[[self.label_token_ids[0], self.label_token_ids[1]]]
        
        # Softmax
        exp_logits = np.exp(label_logits - np.max(label_logits))
        probs = exp_logits / np.sum(exp_logits)
        
        prediction = int(np.argmax(probs))
        confidence = float(probs[prediction])
        
        return {
            'prediction': 'FAKE' if prediction == 1 else 'REAL',
            'confidence': confidence,
            'probabilities': {
                'REAL': float(probs[0]),
                'FAKE': float(probs[1])
            }
        }
    
    def predict_with_context(self, title: str, content: str, context: str) -> Dict:
        """RAG-enhanced prediction"""
        
        title_trunc = self._truncate_text(title, max_chars=100)
        content_trunc = self._truncate_text(content, max_chars=300)
        context_trunc = self._truncate_text(context, max_chars=200)
        
        mask_token = self.tokenizer.mask_token
        prompt = (
            f"Bài viết này là {mask_token} . "
            f"Tiêu_đề : {title_trunc} . "
            f"Nội_dung : {content_trunc} . "
            f"Thông_tin_từ_báo : {context_trunc}"
        )
        
        inputs = self.tokenizer(
            prompt,
            max_length=256,
            padding='max_length',
            truncation=True,
            return_tensors='np'
        )
        
        mask_token_id = self.tokenizer.mask_token_id
        if mask_token_id not in inputs['input_ids'][0]:
            logger.warning("RAG: Mask lost, falling back to base prediction")
            return self.predict(title, content)
        
        outputs = self.session.run(
            None,
            {
                'input_ids': inputs['input_ids'].astype(np.int64),
                'attention_mask': inputs['attention_mask'].astype(np.int64)
            }
        )
        
        logits = outputs[0]
        mask_pos = np.where(inputs['input_ids'][0] == mask_token_id)[0][0]
        mask_logits = logits[0, mask_pos, :]
        label_logits = mask_logits[[self.label_token_ids[0], self.label_token_ids[1]]]
        
        exp_logits = np.exp(label_logits - np.max(label_logits))
        probs = exp_logits / np.sum(exp_logits)
        
        prediction = int(np.argmax(probs))
        confidence = float(probs[prediction])
        
        return {
            'prediction': 'FAKE' if prediction == 1 else 'REAL',
            'confidence': confidence,
            'probabilities': {
                'REAL': float(probs[0]),
                'FAKE': float(probs[1])
            }
        }
