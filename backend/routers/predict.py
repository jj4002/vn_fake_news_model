# routers/predict.py

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional
import time
import logging
import re

from services.supabase_client import SupabaseService
from services.rag_service import RAGService
from services.inference import HANONNXInference

logger = logging.getLogger(__name__)
router = APIRouter()

# Initialize services
db = SupabaseService()
rag_service = RAGService()
model = HANONNXInference()


class PredictRequest(BaseModel):
    video_id: str
    video_url: str
    caption: str
    ocr_text: Optional[str] = ""
    stt_text: Optional[str] = ""
    author_id: Optional[str] = None


class PredictResponse(BaseModel):
    video_id: str
    prediction: str
    confidence: float
    method: str  # 'cached' | 'base_model' | 'rag_enhanced'
    rag_used: bool
    probabilities: dict
    processing_time_ms: float


@router.post("/predict", response_model=PredictResponse)
async def predict(request: PredictRequest):
    start = time.time()
    
    try:
        logger.info("=" * 70)
        logger.info(f"📥 NEW REQUEST: {request.video_id}")
        
        # =========================
        # VALIDATE INPUT
        # =========================
        if not request.caption or len(request.caption.strip()) == 0:
            logger.error("❌ Invalid input: Empty caption")
            raise HTTPException(status_code=400, detail="Caption is required")
        
        # =========================
        # CHECK CACHE
        # =========================
        logger.info(f"🔍 Checking cache for video: {request.video_id}")
        cached = db.get_video(request.video_id)
        
        if cached:
            logger.info(f"✅ Cache hit: {request.video_id}")
            logger.info(f"   Cached prediction: {cached['prediction']} ({cached['confidence']:.4f})")
            logger.info(f"   Cached method: {cached.get('method', 'unknown')}")
            
            # Build probabilities
            if cached["prediction"] == "FAKE":
                probs = {
                    "FAKE": cached["confidence"],
                    "REAL": 1 - cached["confidence"],
                }
            else:
                probs = {
                    "REAL": cached["confidence"],
                    "FAKE": 1 - cached["confidence"],
                }
            
            processing_time = (time.time() - start) * 1000
            logger.info(f"⚡ Completed in {processing_time:.0f}ms (cached)")
            logger.info("=" * 70)
            
            return PredictResponse(
                video_id=request.video_id,
                prediction=cached["prediction"],
                confidence=cached["confidence"],
                method="cached",
                rag_used=False,
                probabilities=probs,
                processing_time_ms=processing_time,
            )
        
        logger.info("✅ No cache found, running model...")
        
        # =========================
        # PREPARE INPUT
        # =========================
        title_raw = request.caption
        title = title_raw.strip()
        
        content_parts = []
        
        if request.ocr_text:
            ocr_fixed = request.ocr_text.strip()
            content_parts.append(ocr_fixed)
            logger.info(f"   OCR: {len(request.ocr_text)} chars")
        
        if request.stt_text:
            stt_fixed = request.stt_text.strip()
            content_parts.append(stt_fixed)
            logger.info(f"   STT: {len(request.stt_text)} chars")
        
        if not content_parts:
            content = title
        else:
            content = " ".join(content_parts)
        
        # Log input
        logger.info("📝 Input:")
        logger.info(f"   Title: {title[:100]}...")
        logger.info(f"   Content length: {len(content)} chars")
        
        # =========================
        # BASE PREDICTION
        # =========================
        logger.info("🤖 Running base model...")
        base_result = model.predict(
            title=title,
            content=content,
            return_top_chunk=True
        )
        
        logger.info(f"   Base result: {base_result['prediction']} ({base_result['confidence']:.4f})")
        logger.info(f"   Probabilities: REAL={base_result['probabilities']['REAL']:.4f}, FAKE={base_result['probabilities']['FAKE']:.4f}")
        
        # =========================
        # HEURISTIC ADJUSTMENTS
        # =========================
        title_lower = title.lower()
        content_lower = content.lower()
        combined_text = title_lower + ' ' + content_lower
        
        logger.info("🔍 Running heuristic checks...")
        
        adjustment_factor = 1.0
        adjustment_reasons = []
        
        # Pattern 1: Debunk signals
        debunk_keywords = [
            'thật hư', 'sự thật hay dối trá', 'có phải sự thật',
            'tin đồn', 'giả mạo', 'lừa đảo', 'thông tin sai lệch'
        ]
        
        if any(kw in combined_text for kw in debunk_keywords):
            if base_result["prediction"] == "REAL":
                logger.warning("⚠️ HEURISTIC: Debunk signal with REAL prediction")
                adjustment_factor *= 0.55
                adjustment_reasons.append("Debunk signal detected")
        
        # Pattern 2: Financial claims without official source
        if re.search(r'(phát|tặng|nhận).{0,20}\d+\s*(triệu|trăm nghìn)', combined_text):
            logger.warning("⚠️ HEURISTIC: Financial claim detected")
            
            official_sources = [
                'theo vnexpress', 'báo vtv', 'vov đưa tin', 'theo chính phủ',
                'bộ tài chính thông báo', 'thủ tướng ký', 'nghị định', 'quyết định số'
            ]
            
            has_source = any(src in combined_text for src in official_sources)
            
            if not has_source:
                adjustment_factor *= 0.70
                adjustment_reasons.append("Financial claim without official source")
                logger.warning("   No official source found")
        
        # Pattern 3: Clickbait + Financial
        clickbait_keywords = [
            'gây sốc', 'cực sốc', 'tin sốc', 'nóng hổi',
            'khẩn cấp', 'gấp', 'lập tức'
        ]
        clickbait_count = sum(1 for kw in clickbait_keywords if kw in combined_text)
        
        if clickbait_count > 0:
            has_financial = any(kw in combined_text for kw in 
                               ['phát tiền', 'tặng tiền', 'triệu', 'hỗ trợ tiền'])
            if has_financial:
                logger.warning(f"⚠️ HEURISTIC: Clickbait ({clickbait_count}) + financial")
                adjustment_factor *= 0.85
                adjustment_reasons.append(f"Clickbait + financial")
        
        # Pattern 4: Phone numbers in OCR (spam indicator)
        if request.ocr_text and re.search(r'\d{9,11}', request.ocr_text):
            logger.warning("⚠️ HEURISTIC: Phone number in OCR")
            adjustment_factor *= 0.90
            adjustment_reasons.append("Phone number in OCR")
        
        # Apply adjustments
        if adjustment_factor < 1.0:
            original_conf = base_result["confidence"]
            adjusted_conf = max(0.50, original_conf * adjustment_factor)
            
            logger.warning("=" * 70)
            logger.warning("🔧 HEURISTIC ADJUSTMENTS APPLIED")
            logger.warning(f"   Original: {original_conf:.4f}")
            logger.warning(f"   Factor: {adjustment_factor:.2f}")
            logger.warning(f"   Adjusted: {adjusted_conf:.4f}")
            for reason in adjustment_reasons:
                logger.warning(f"     - {reason}")
            logger.warning("=" * 70)
            
            base_result = {
                "prediction": base_result["prediction"],
                "confidence": adjusted_conf,
                "probabilities": {
                    "REAL": adjusted_conf if base_result["prediction"] == "REAL" else 1 - adjusted_conf,
                    "FAKE": adjusted_conf if base_result["prediction"] == "FAKE" else 1 - adjusted_conf,
                },
            }
        
        # Flip logic: Low confidence REAL → FAKE
        if base_result["prediction"] == "REAL" and base_result["confidence"] < 0.60:
            logger.warning("=" * 70)
            logger.warning("🔄 FLIPPING PREDICTION")
            logger.warning(f"   Reason: REAL confidence too low ({base_result['confidence']:.2f} < 0.60)")
            logger.warning("=" * 70)
            
            base_result = {
                "prediction": "FAKE",
                "confidence": 1 - base_result["confidence"],
                "probabilities": {
                    "FAKE": 1 - base_result["confidence"],
                    "REAL": base_result["confidence"],
                },
            }
        
        # Initialize final result
        rag_used = False
        method = "base_model"  # ← GIỮ NGUYÊN METHOD CŨ
        final_result = base_result
        
        # =========================
        # RAG VERIFICATION
        # =========================
        if rag_service.should_use_rag(
            title, 
            content, 
            base_result["confidence"], 
            request.author_id or ""
        ):
            logger.info("🔍 Running RAG verification...")
            
            # Extract top chunk từ base result
            top_chunk = base_result.get('top_chunk', '')
            
            verification = rag_service.verify_with_sources(
                title=title,
                content=content,
                top_chunk=top_chunk
            )
            
            logger.info(f"   RAG result: {verification['recommendation']}")
            logger.info(f"   Similarity: {verification['similarity_score']:.2f}")
            
            if verification["matching_articles"]:
                rag_used = True
                
                # Log top match
                top_article = verification['matching_articles'][0]
                logger.info(f"   Top match: {top_article['source']}")
                logger.info(f"   Title: {top_article['title'][:80]}...")
                
                similarity = verification["similarity_score"]
                recommendation = verification["recommendation"]
                
                # ============================================
                # RAG DECISION LOGIC
                # ============================================
                
                if base_result["prediction"] == "FAKE":
                    logger.info("🔍 Base prediction is FAKE, checking RAG evidence...")
                    
                    if recommendation == "VERIFIED_REAL" and similarity >= 0.85:
                        # Flip to REAL với high similarity
                        logger.warning("⚠️ VERIFIED_REAL (similarity ≥ 0.85) → Switching to REAL")
                        method = "rag_enhanced"  # ← CHỈ ĐỔI KHI RAG CAN THIỆP MẠNH
                        new_conf = max(0.7, min(0.95, 0.7 + (similarity - 0.85) * 3))
                        final_result = {
                            "prediction": "REAL",
                            "confidence": new_conf,
                            "probabilities": {
                                "REAL": new_conf,
                                "FAKE": 1 - new_conf,
                            },
                        }
                        logger.info(f"   Overridden to: REAL ({new_conf:.4f})")
                    
                    elif recommendation == "NEEDS_REVIEW" and similarity >= 0.75:
                        # Giảm confidence nhẹ
                        logger.warning("⚠️ NEEDS_REVIEW → Slightly reducing FAKE confidence")
                        adjusted_conf = max(0.55, base_result["confidence"] * 0.95)
                        final_result = {
                            "prediction": "FAKE",
                            "confidence": adjusted_conf,
                            "probabilities": {
                                "FAKE": adjusted_conf,
                                "REAL": 1 - adjusted_conf,
                            },
                        }
                        logger.info(f"   Reduced confidence: {base_result['confidence']:.4f} → {adjusted_conf:.4f}")
                        # method vẫn là "base_model"
                    
                    else:
                        logger.info("   No strong evidence → Keep base prediction")
                        final_result = base_result
                
                elif base_result["prediction"] == "REAL":
                    logger.info("✅ Base prediction is REAL, checking RAG confirmation...")
                    
                    if recommendation == "VERIFIED_REAL" and similarity >= 0.85:
                        # Boost confidence
                        logger.info("✅ Base REAL confirmed by RAG (strong)")
                        method = "rag_enhanced"  # ← CHỈ ĐỔI KHI RAG CAN THIỆP MẠNH
                        boosted_conf = min(0.98, base_result["confidence"] * 1.15)
                        final_result = {
                            "prediction": "REAL",
                            "confidence": boosted_conf,
                            "probabilities": {
                                "REAL": boosted_conf,
                                "FAKE": 1 - boosted_conf,
                            },
                        }
                        logger.info(f"   Boosted confidence: {base_result['confidence']:.4f} → {boosted_conf:.4f}")
                    
                    else:
                        logger.info("   RAG did not strongly confirm → Keep base")
                        final_result = base_result
            
            else:
                logger.info("   No matching articles found, using base result")
                final_result = base_result
        
        # =========================
        # EXTRACT FINAL RESULT
        # =========================
        prediction = final_result["prediction"]
        confidence = final_result["confidence"]
        probabilities = final_result["probabilities"]
        
        logger.info(f"📊 Final result:")
        logger.info(f"   Prediction: {prediction}")
        logger.info(f"   Confidence: {confidence:.4f}")
        logger.info(f"   Method: {method}")
        logger.info(f"   RAG used: {rag_used}")
        
        # =========================
        # SAVE TO CACHE
        # =========================
        try:
            logger.info("💾 Saving to database...")
            save_data = {
                "video_id": request.video_id,
                "video_url": request.video_url,
                "caption": request.caption,
                "ocr_text": request.ocr_text,
                "stt_text": request.stt_text,
                "author_id": request.author_id,
                "prediction": prediction,
                "confidence": confidence,
                "method": method,
            }
            
            db.save_video(save_data)
            logger.info("✅ Saved to cache")
        
        except Exception as save_error:
            logger.error(f"❌ Cache save error: {save_error}")
            logger.warning("⚠️ Continuing without cache...")
        
        # =========================
        # RETURN RESPONSE
        # =========================
        processing_time = (time.time() - start) * 1000
        logger.info(f"✅ Completed in {processing_time:.0f}ms")
        logger.info("=" * 70)
        
        return PredictResponse(
            video_id=request.video_id,
            prediction=prediction,
            confidence=confidence,
            method=method,
            rag_used=rag_used,
            probabilities=probabilities,
            processing_time_ms=processing_time,
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Prediction error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
