# services/inference.py

import onnxruntime as ort
import numpy as np
from transformers import AutoTokenizer
from sentence_transformers import SentenceTransformer, util
from typing import Dict, List
import logging
import os
import re
import unicodedata
import torch

logger = logging.getLogger(__name__)


# ===========================
# TEXT PREPROCESSING
# ===========================

class VietnameseTextNormalizer:
    """Text normalizer - GIỐNG TRAINING"""
    
    def __init__(self):
        try:
            from underthesea import word_tokenize
            self.use_word_segment = True
            logger.info("✅ Underthesea available - word segmentation enabled")
        except:
            self.use_word_segment = False
            logger.warning("⚠️ Underthesea not available - word segmentation disabled")
    
    def normalize_unicode(self, text: str) -> str:
        return unicodedata.normalize("NFC", text)
    
    def clean_special_chars(self, text: str) -> str:
        text = re.sub(r'https?://\S+', '', text)
        text = re.sub(
            r'[^a-zA-ZàáảãạăằắẳẵặâầấẩẫậèéẻẽẹêềếểễệìíỉĩịòóỏõọôồốổỗộơờớởỡợùúủũụưừứửữựỳýỷỹỵđĐ0-9\s.,!?;:]',
            ' ', text
        )
        return text
    
    def word_segment(self, text: str) -> str:
        if not self.use_word_segment:
            return text
        try:
            from underthesea import word_tokenize
            text = word_tokenize(text, format="text")
        except:
            pass
        return text
    
    def normalize(self, text: str) -> str:
        if not text or not isinstance(text, str):
            return ""
        text = self.normalize_unicode(text)
        text = text.strip()
        text = self.clean_special_chars(text)
        text = re.sub(r'\s+', ' ', text)
        text = self.word_segment(text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text


class SemanticChunkRetriever:
    """Chunk retriever - GIỐNG TRAINING"""
    
    def __init__(self, chunk_size=400):
        self.chunk_size = chunk_size
    
    def chunk_document(self, text: str) -> List[str]:
        if not text or len(text.strip()) == 0:
            return []
        
        sentences = re.split(r'[.!?\-]\s+', text)  # ← GIỐNG TRAINING
        chunks = []
        current_chunk = []
        current_len = 0
        
        for sent in sentences:
            sent = sent.strip()
            if not sent:
                continue
            
            sent_len = len(sent)
            
            # ✅ Thêm logic handle oversized chunks (từ training)
            if current_len + sent_len > self.chunk_size:
                if current_chunk:
                    chunks.append('. '.join(current_chunk))
                
                # ✅ Handle single long sentence
                if sent_len > self.chunk_size * 1.5:
                    words = sent.split()
                    temp_chunk = []
                    temp_len = 0
                    for word in words:
                        if temp_len + len(word) > self.chunk_size:
                            if temp_chunk:
                                chunks.append(' '.join(temp_chunk))
                            temp_chunk = [word]
                            temp_len = len(word)
                        else:
                            temp_chunk.append(word)
                            temp_len += len(word) + 1
                    if temp_chunk:
                        current_chunk = temp_chunk
                        current_len = temp_len
                else:
                    current_chunk = [sent]
                    current_len = sent_len
            else:
                current_chunk.append(sent)
                current_len += sent_len
        
        if current_chunk:
            chunks.append('. '.join(current_chunk))
        
        return chunks



# ===========================
# HAN ONNX INFERENCE
# ===========================

class HANONNXInference:
    """
    HAN Model Inference với ONNX Runtime
    
    CHẾ ĐỘ DUY NHẤT: Base prediction
    RAG chỉ dùng để adjust confidence bên ngoài (trong router)
    """
    
    def __init__(
        self,
        model_path: str = None,
        tokenizer_path: str = None,
        retriever_model: str = "keepitreal/vietnamese-sbert",
        top_k: int = 5,
        chunk_size: int = 400,
        max_length: int = 256
    ):
        """
        Initialize HAN ONNX Inference
        
        Args:
            model_path: Path to ONNX model (han_model.onnx)
            tokenizer_path: Local tokenizer dir or HF model name
            retriever_model: SentenceTransformer for RAG chunk selection
            top_k: Number of chunks to select (default: 5, giống training)
            chunk_size: Max chars per chunk (default: 400)
            max_length: Max sequence length (default: 256)
        """
        # Get paths from env or use defaults
        model_path = model_path or os.getenv("MODEL_PATH", "./models/han_model.onnx")
        tokenizer_path = tokenizer_path or os.getenv("TOKENIZER_PATH", "vinai/phobert-base-v2")
        
        logger.info("=" * 70)
        logger.info("🔧 Initializing HAN ONNX Inference Service")
        logger.info("=" * 70)
        logger.info(f"   Model path: {model_path}")
        logger.info(f"   Tokenizer: {tokenizer_path}")
        logger.info(f"   Retriever: {retriever_model}")
        logger.info(f"   Config: top_k={top_k}, chunk_size={chunk_size}, max_length={max_length}")
        
        # Check model exists
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"❌ Model not found: {model_path}")
        
        # ===========================
        # 1. LOAD ONNX MODEL
        # ===========================
        logger.info("📦 Loading ONNX model...")
        try:
            self.session = ort.InferenceSession(
                model_path,
                providers=['CPUExecutionProvider']
            )
            logger.info("✅ ONNX model loaded")
        except Exception as e:
            logger.error(f"❌ Failed to load ONNX model: {e}")
            raise
        
        # ===========================
        # 2. LOAD TOKENIZER
        # ===========================
        logger.info("📦 Loading tokenizer...")
        try:
            if os.path.exists(tokenizer_path) and os.path.isdir(tokenizer_path):
                # Load from local directory
                logger.info("   Loading from local path...")
                self.tokenizer = AutoTokenizer.from_pretrained(
                    tokenizer_path,
                    local_files_only=True
                )
            else:
                # Load from HuggingFace
                logger.info("   Downloading from HuggingFace...")
                self.tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base-v2")
            
            logger.info("✅ Tokenizer loaded")
        except Exception as e:
            logger.error(f"❌ Failed to load tokenizer: {e}")
            raise
        
        # ===========================
        # 3. LOAD TEXT NORMALIZER
        # ===========================
        logger.info("📦 Initializing text normalizer...")
        self.normalizer = VietnameseTextNormalizer()
        
        # ===========================
        # 4. LOAD SENTENCE RETRIEVER (for RAG chunk selection)
        # ===========================
        logger.info("📦 Loading sentence retriever...")
        try:
            self.retriever = SentenceTransformer(retriever_model)
            logger.info("✅ Sentence retriever loaded")
        except Exception as e:
            logger.error(f"❌ Failed to load retriever: {e}")
            raise
        
        # ===========================
        # 5. INITIALIZE CHUNKER
        # ===========================
        self.chunker = SemanticChunkRetriever(chunk_size=chunk_size)
        
        # ===========================
        # 6. SET CONFIG
        # ===========================
        self.top_k = top_k
        self.max_length = max_length
        
        logger.info("=" * 70)
        logger.info("✅ HAN ONNX Inference Service initialized successfully!")
        logger.info("=" * 70)
    
    def _select_chunks_with_rag(self, title: str, content: str) -> List[str]:
        """
        Internal RAG: Chọn top-k chunks từ CHÍNH BÀI VIẾT
        
        Dùng title làm query để tìm chunks quan trọng nhất
        GIỐNG TRAINING!
        
        Args:
            title: Video title/caption (normalized)
            content: Video content (normalized)
        
        Returns:
            List of selected chunks (max: self.top_k)
        """
        # 1. Chunk content
        raw_chunks = self.chunker.chunk_document(content)
        
        if not raw_chunks:
            logger.warning("   No chunks generated, returning empty")
            return [""]
        
        logger.info(f"   Generated {len(raw_chunks)} chunks from content")
        
        # 2. Nếu ít chunks hơn top_k, lấy hết
        if len(raw_chunks) <= self.top_k:
            logger.info(f"   Using all {len(raw_chunks)} chunks (≤ top_k)")
            return raw_chunks
        
        # 3. RAG: Dùng title làm query
        query = title if len(title) > 5 else raw_chunks[0]
        
        try:
            # Encode query và chunks
            query_emb = self.retriever.encode(query, convert_to_tensor=True)
            chunk_embs = self.retriever.encode(raw_chunks, convert_to_tensor=True)
            
            # Cosine similarity
            scores = util.cos_sim(query_emb, chunk_embs)[0]
            
            # Top-k indices
            topk_indices = torch.topk(
                scores, 
                k=min(self.top_k, len(raw_chunks))
            ).indices.tolist()
            
            # Sort để giữ thứ tự xuất hiện trong content
            topk_indices.sort()
            
            selected_chunks = [raw_chunks[i] for i in topk_indices]
            
            logger.info(f"   ✅ RAG selected {len(selected_chunks)}/{len(raw_chunks)} chunks")
            
            return selected_chunks
        
        except Exception as e:
            logger.warning(f"   ⚠️ RAG failed: {e}, using fallback")
            
            # Fallback: Lấy chunks đầu + cuối
            mid = self.top_k // 2
            selected_chunks = raw_chunks[:mid] + raw_chunks[-self.top_k + mid:]
            
            logger.info(f"   Using fallback: first {mid} + last {self.top_k - mid} chunks")
            
            return selected_chunks[:self.top_k]
    
    def predict(self, title: str, content: str, return_top_chunk: bool = False) -> Dict:
        """
        Base model prediction
        
        Args:
            title: Video title/caption
            content: Video content
            return_top_chunk: If True, return top chunk for RAG query
        
        Returns:
            {
                'prediction': 'FAKE' | 'REAL',
                'confidence': float,
                'probabilities': {...},
                'top_chunk': str (optional)  # ← THÊM
            }
        """
        try:
            logger.info("🤖 Running HAN model prediction...")
            
            # 1. Normalize
            title_norm = self.normalizer.normalize(title)
            content_norm = self.normalizer.normalize(content)
            
            # 2. RAG chunk selection
            selected_chunks = self._select_chunks_with_rag(title_norm, content_norm)
            
            # LƯU TOP CHUNK (chunk quan trọng nhất)
            top_chunk = selected_chunks[0] if selected_chunks else ""
            
            # DUPLICATE thay vì PAD EMPTY
            while len(selected_chunks) < self.top_k:
                selected_chunks.append(
                    selected_chunks[0] if selected_chunks else ""
                )
            
            # Truncate nếu quá
            selected_chunks = selected_chunks[:self.top_k]
            
            # 4-7. Tokenize + Inference (giữ nguyên)
            encoded = self.tokenizer(
                selected_chunks,
                max_length=self.max_length,
                padding='max_length',
                truncation=True,
                return_tensors='np'
            )
            
            chunk_input_ids = np.expand_dims(encoded['input_ids'], axis=0).astype(np.int64)
            chunk_attention_masks = np.expand_dims(encoded['attention_mask'], axis=0).astype(np.int64)
            
            onnx_inputs = {
                'chunk_input_ids': chunk_input_ids,
                'chunk_attention_masks': chunk_attention_masks
            }
            
            onnx_outputs = self.session.run(None, onnx_inputs)
            logits = onnx_outputs[0][0]
            
            exp_logits = np.exp(logits - np.max(logits))
            probs = exp_logits / np.sum(exp_logits)
            
            prediction_idx = int(np.argmax(probs))
            confidence = float(probs[prediction_idx])
            prediction = 'FAKE' if prediction_idx == 1 else 'REAL'
            
            logger.info(f"✅ Prediction: {prediction} ({confidence:.4f})")
            
            result = {
                'prediction': prediction,
                'confidence': confidence,
                'probabilities': {
                    'REAL': float(probs[0]),
                    'FAKE': float(probs[1])
                }
            }
            
            # ✅ THÊM TOP CHUNK NẾU CẦN
            if return_top_chunk:
                result['top_chunk'] = top_chunk
                logger.info(f"   Top chunk: {top_chunk[:100]}...")
            
            return result
        
        except Exception as e:
            logger.error(f"❌ Inference error: {e}", exc_info=True)
            raise
