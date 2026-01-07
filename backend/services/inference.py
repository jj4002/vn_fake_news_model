# services/inference.py
# Version: 4.3 (synced with RAG_HAN_v4_3.ipynb)
# Updated: 2025-12-26
# Key changes: Added chunk_overlap=50 to match training

import onnxruntime as ort
import numpy as np
from transformers import AutoTokenizer
from sentence_transformers import SentenceTransformer
import torch
import torch.nn.functional as F
from typing import Dict, List
import logging
import os
import re
import unicodedata

logger = logging.getLogger(__name__)


# ===========================
# TEXT PREPROCESSING
# ===========================

class VietnameseTextNormalizer:
    """Text normalizer - EXACT MATCH với training code"""
    
    def __init__(self):
        try:
            from underthesea import word_tokenize
            self.use_word_segment = True
            logger.info("✅ Underthesea available")
        except ImportError:
            self.use_word_segment = False
            logger.warning("⚠️ Underthesea not available")
    
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
            return word_tokenize(text, format="text")
        except:
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
    """Semantic Chunker with OVERLAP - EXACT MATCH training"""
    
    def __init__(self, chunk_size: int = 400, chunk_overlap: int = 50):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        logger.info(f"✅ SemanticChunkRetriever: size={chunk_size}, overlap={chunk_overlap}")
    
    def chunk_document(self, text: str) -> List[str]:
        if not text or len(text.strip()) == 0:
            return []
        
        sentences = re.split(r'[.!?\-]\s+', text)
        chunks = []
        current_chunk = []
        current_len = 0
        
        for sent in sentences:
            sent = sent.strip()
            if not sent:
                continue
            
            sent_len = len(sent)
            
            # Handle oversized sentence
            if sent_len > self.chunk_size * 1.5:
                if current_chunk:
                    chunks.append('. '.join(current_chunk))
                    current_chunk = []
                    current_len = 0
                
                words = sent.split()
                for i in range(0, len(words), 50):
                    chunks.append(' '.join(words[i:i+50]))
                continue
            
            # Create chunk with overlap
            if current_len + sent_len > self.chunk_size:
                if current_chunk:
                    chunks.append('. '.join(current_chunk))
                    
                    # Keep last N sentences for overlap
                    overlap_sents = []
                    overlap_len = 0
                    for s in reversed(current_chunk):
                        if overlap_len + len(s) <= self.chunk_overlap:
                            overlap_sents.insert(0, s)
                            overlap_len += len(s) + 1
                        else:
                            break
                    
                    current_chunk = overlap_sents
                    current_len = overlap_len
            
            current_chunk.append(sent)
            current_len += sent_len + 1
        
        if current_chunk:
            chunks.append('. '.join(current_chunk))
        
        return chunks


# ===========================
# HAN ONNX INFERENCE
# ===========================

class HANONNXInference:
    """HAN Model Inference - SYNCED WITH v4.3 TRAINING"""
    
    def __init__(
        self,
        model_path: str = None,
        tokenizer_path: str = None,
        retriever_model: str = "keepitreal/vietnamese-sbert",
        top_k: int = 5,
        chunk_size: int = 400,
        chunk_overlap: int = 50,  # ← KEY CHANGE
        max_length: int = 256,
        min_chunks: int = 3,
        min_similarity: float = 0.15
    ):
        model_path = model_path or os.getenv("MODEL_PATH", "./models/han_rag_model.onnx")
        tokenizer_path = tokenizer_path or os.getenv("TOKENIZER_PATH", "vinai/phobert-base-v2")
        
        logger.info("="*70)
        logger.info("🚀 HAN ONNX Inference v4.3")
        logger.info(f"  chunk_overlap={chunk_overlap} ← NEW")
        
        # Load ONNX
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if torch.cuda.is_available() else ['CPUExecutionProvider']
        self.session = ort.InferenceSession(model_path, providers=providers)
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        
        # Load normalizer
        self.normalizer = VietnameseTextNormalizer()
        
        # Load retriever
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.retriever = SentenceTransformer(retriever_model, device=device)
        
        # Chunker with overlap
        self.chunker = SemanticChunkRetriever(chunk_size, chunk_overlap)
        
        self.top_k = top_k
        self.max_length = max_length
        self.min_chunks = min_chunks
        self.min_similarity = min_similarity
        
        logger.info("✅ Initialized!")
    
    def _select_chunks_with_rag(self, title: str, content: str) -> List[str]:
        raw_chunks = self.chunker.chunk_document(content)
        
        if not raw_chunks:
            return [title] * self.top_k
        
        if len(raw_chunks) < self.min_chunks:
            while len(raw_chunks) < self.min_chunks:
                raw_chunks.extend(raw_chunks[:self.min_chunks - len(raw_chunks)])
        
        if len(raw_chunks) <= self.top_k:
            selected = raw_chunks[:]
        else:
            query = title if len(title) > 5 else raw_chunks[0]
            
            try:
                query_emb = self.retriever.encode(query, convert_to_tensor=True)
                chunk_embs = self.retriever.encode(raw_chunks, convert_to_tensor=True)
                
                similarities = F.cosine_similarity(query_emb.unsqueeze(0), chunk_embs, dim=1)
                valid_indices = (similarities >= self.min_similarity).nonzero(as_tuple=True)[0]
                
                if len(valid_indices) < self.top_k:
                    top_indices = similarities.argsort(descending=True)[:self.top_k]
                else:
                    valid_sims = similarities[valid_indices]
                    sorted_valid = valid_indices[valid_sims.argsort(descending=True)]
                    top_indices = sorted_valid[:self.top_k]
                
                selected = [raw_chunks[i] for i in top_indices.tolist()]
            except Exception as e:
                logger.warning(f"RAG failed: {e}")
                mid = self.top_k // 2
                selected = raw_chunks[:mid] + raw_chunks[-self.top_k+mid:]
        
        while len(selected) < self.top_k:
            selected.append(selected[0] if selected else title)
        
        return selected[:self.top_k]
    
    def predict(self, title: str, content: str) -> Dict:
        # Normalize
        title_norm = self.normalizer.normalize(title)
        content_norm = self.normalizer.normalize(content)
        
        # Select chunks
        chunks = self._select_chunks_with_rag(title_norm, content_norm)
        
        # Tokenize
        encodings = self.tokenizer(
            chunks,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='np'
        )
        
        # ONNX inference
        inputs = {
            'chunk_input_ids': np.expand_dims(encodings['input_ids'], 0).astype(np.int64),
            'chunk_attention_masks': np.expand_dims(encodings['attention_mask'], 0).astype(np.int64)
        }
        outputs = self.session.run(None, inputs)
        
        # Post-process
        logits = outputs[0][0]
        probs = np.exp(logits - np.max(logits))
        probs = probs / np.sum(probs)
        
        pred_idx = int(np.argmax(probs))
        prediction = 'FAKE' if pred_idx == 1 else 'REAL'
        
        return {
            'prediction': prediction,
            'confidence': float(probs[pred_idx]),
            'probabilities': {'REAL': float(probs[0]), 'FAKE': float(probs[1])}
        }
