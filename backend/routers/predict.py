# routers/predict.py - FINAL VERSION (Corrected RAG Logic)

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional
import time
import logging

from services.supabase_client import SupabaseService
from services.rag_service import RAGService
from services.inference import HANONNXInference

logger = logging.getLogger(__name__)
router = APIRouter()

# Initialize services
db = SupabaseService()
rag_service = RAGService()
model = HANONNXInference()

# Method mapping for database constraint
METHOD_MAP = {
    'rag_enhanced': 'rag_enhanced',
    'rag_weak_support': 'rag_enhanced',
    'rag_override': 'rag_enhanced',
    'rag_doubt': 'base_model',
    'no_evidence': 'base_model',
    'base_model': 'base_model',
}


class PredictRequest(BaseModel):
    video_id: str
    video_url: str
    caption: str
    ocr_text: Optional[str] = ""
    stt_text: Optional[str] = ""
    author_id: Optional[str] = None


class TextPredictRequest(BaseModel):
    text: str
    author_id: Optional[str] = None


class PredictResponse(BaseModel):
    video_id: str
    prediction: str
    confidence: float
    method: str
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

            if cached["prediction"] == "FAKE":
                probs = {"FAKE": cached["confidence"], "REAL": 1 - cached["confidence"]}
            else:
                probs = {"REAL": cached["confidence"], "FAKE": 1 - cached["confidence"]}

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
        title = request.caption.strip()

        content_parts = []
        if request.ocr_text and len(request.ocr_text.strip()) > 10:
            content_parts.append(request.ocr_text.strip())
            logger.info(f"   OCR: {len(request.ocr_text)} chars")

        if request.stt_text and len(request.stt_text.strip()) > 50:
            content_parts.append(request.stt_text.strip())
            logger.info(f"   STT: {len(request.stt_text)} chars")

        content = " ".join(content_parts)[:2000] if content_parts else title

        logger.info("📝 Input:")
        logger.info(f"   Title: {title[:100]}...")
        logger.info(f"   Content: {len(content)} chars")

        # =========================
        # BASE PREDICTION
        # =========================
        logger.info("🤖 Running base model...")
        base_result = model.predict(
            title=title,
            content=content
        )

        logger.info(f"   Base result: {base_result['prediction']} ({base_result['confidence']:.4f})")
        logger.info(f"   Probabilities: REAL={base_result['probabilities']['REAL']:.4f}, FAKE={base_result['probabilities']['FAKE']:.4f}")

        # =========================
        # RAG VERIFICATION (FINAL LOGIC)
        # =========================

        rag_used = False
        method = "base_model"
        final_result = base_result

        if rag_service.should_use_rag(
            title,
            content,
            base_result["confidence"],
            request.author_id or "",
        ):
            logger.info("🔍 Running RAG verification...")

            verification = rag_service.verify(
                title=title,
                content=content,
                top_k=5,
            )

            similarity = verification["similarity_score"]
            recommendation = verification["recommendation"]
            rag_used = True

            logger.info(f"   RAG result: {recommendation}")
            logger.info(f"   Similarity: {similarity:.4f}")

            base_pred = base_result["prediction"]
            base_conf = base_result["confidence"]

            # Handle case when matching_articles is empty
            if not verification["matching_articles"]:
                recommendation = "NO_RELIABLE_SOURCE_FOUND"
                logger.info("   No matching articles found")

            if verification["matching_articles"]:
                top_article = verification["matching_articles"][0]
                logger.info(f"   Top match: {top_article.get('source', 'unknown')}")
                logger.info(f"   Title: {top_article.get('title', '')[:80]}...")

            # ==================== DECISION LOGIC (FINAL) ====================

            if base_pred == "REAL":
                # ========== BASE REAL ==========

                if recommendation == "VERIFIED":
                    # Strong evidence confirms REAL
                    logger.info("✅ VERIFIED: Strong evidence confirms REAL")
                    method = "rag_enhanced"
                    final_conf = min(0.98, base_conf * 1.15)
                    prediction = "REAL"
                    logger.info(f"   Boosted: {base_conf:.4f} → {final_conf:.4f}")

                elif recommendation == "NEEDS_REVIEW":
                    # Weak evidence supports REAL
                    logger.info("⚠️ NEEDS_REVIEW: Weak evidence supports REAL")
                    method = "rag_weak_support"
                    final_conf = min(0.95, base_conf * 1.05)
                    prediction = "REAL"
                    logger.info(f"   Boosted slightly: {base_conf:.4f} → {final_conf:.4f}")

                else:  # NO_RELIABLE_SOURCE_FOUND
                    # No evidence found (DB might not be updated)
                    logger.warning("🚨 NO_SOURCE: REAL claim but no evidence found")
                    logger.warning("   → Could be new news, local news, or suspicious")
                    method = "no_evidence"
                    final_conf = base_conf * 0.75  # 25% penalty
                    prediction = "REAL"  # Keep REAL, don't flip!
                    logger.info(f"   Penalty: {base_conf:.4f} → {final_conf:.4f}")

            else:
                # ========== BASE FAKE ==========

                if recommendation == "VERIFIED":
                    # Found real news - check similarity
                    if similarity >= 0.83:
                        # Strong evidence → Override to REAL
                        logger.warning("⚠️ OVERRIDE: Found strong evidence of REAL news!")
                        method = "rag_override"
                        final_conf = similarity
                        prediction = "REAL"
                        logger.info(f"   Overridden FAKE → REAL: {final_conf:.4f}")
                    else:
                        # Weak evidence → Keep FAKE but reduce confidence
                        logger.info("⚠️ VERIFIED but weak (<0.83) → Reduce FAKE confidence")
                        method = "rag_doubt"
                        final_conf = base_conf * 0.85
                        prediction = "FAKE"
                        logger.info(f"   Reduced: {base_conf:.4f} → {final_conf:.4f}")

                elif recommendation == "NEEDS_REVIEW":
                    # Similar real news exists → Doubt FAKE
                    logger.info("⚠️ NEEDS_REVIEW: Found similar real news")
                    method = "rag_doubt"
                    final_conf = base_conf * 0.85
                    prediction = "FAKE"
                    logger.info(f"   Reduced: {base_conf:.4f} → {final_conf:.4f}")

                else:  # NO_RELIABLE_SOURCE_FOUND
                    # Expected behavior - fake news not in real news DB
                    logger.info("✅ NO_SOURCE: FAKE + no real news = Expected behavior")
                    method = "base_model"
                    final_conf = base_conf  # No change!
                    prediction = "FAKE"
                    logger.info(f"   Keeping original: {base_conf:.4f}")

            # Build final result
            final_result = {
                "prediction": prediction,
                "confidence": final_conf,
                "probabilities": {
                    prediction: final_conf,
                    "REAL" if prediction == "FAKE" else "FAKE": 1 - final_conf
                }
            }

        prediction = final_result["prediction"]
        confidence = final_result["confidence"]
        probabilities = final_result["probabilities"]

        logger.info("📊 Final result:")
        logger.info(f"   Prediction: {prediction}")
        logger.info(f"   Confidence: {confidence:.4f}")
        logger.info(f"   Method: {method}")
        logger.info(f"   RAG used: {rag_used}")

        # =========================
        # SAVE TO CACHE
        # =========================
        try:
            logger.info("💾 Saving to database...")

            # Map method to DB constraint value
            db_method = METHOD_MAP.get(method, 'base_model')

            save_data = {
                "video_id": request.video_id,
                "video_url": request.video_url,
                "caption": request.caption,
                "ocr_text": request.ocr_text,
                "stt_text": request.stt_text,
                "author_id": request.author_id,
                "prediction": prediction,
                "confidence": confidence,
                "method": db_method,  # Use mapped method
            }

            db.save_video(save_data)
            logger.info(f"✅ Saved to cache (method: {method} → {db_method})")

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
            method=method,  # Return internal method name
            rag_used=rag_used,
            probabilities=probabilities,
            processing_time_ms=processing_time,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Prediction error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/predict-text", response_model=PredictResponse)
async def predict_text(request: TextPredictRequest):
    """
    Phân tích trực tiếp một đoạn text (không lưu database, vẫn dùng RAG nếu cần)
    """
    start = time.time()

    try:
        logger.info("=" * 70)
        logger.info("📥 NEW TEXT REQUEST")

        text = (request.text or "").strip()
        if not text:
            logger.error("❌ Invalid input: Empty text")
            raise HTTPException(status_code=400, detail="Text is required")

        # Coi toàn bộ text là title + content
        title = text[:200]
        content = text

        logger.info("📝 Text Input:")
        logger.info(f"   Preview: {text[:120]}...")
        logger.info(f"   Length: {len(text)} chars")

        # BASE PREDICTION
        logger.info("🤖 Running base model (text)...")
        base_result = model.predict(
            title=title,
            content=content
        )

        logger.info(f"   Base result: {base_result['prediction']} ({base_result['confidence']:.4f})")
        logger.info(
            f"   Probabilities: REAL={base_result['probabilities']['REAL']:.4f}, "
            f"FAKE={base_result['probabilities']['FAKE']:.4f}"
        )

        # RAG VERIFICATION (SAME LOGIC AS /predict)
        rag_used = False
        method = "base_model"
        final_result = base_result

        if rag_service.should_use_rag(
            title,
            content,
            base_result["confidence"],
            request.author_id or "",
        ):
            logger.info("🔍 Running RAG verification for text...")

            verification = rag_service.verify(
                title=title,
                content=content,
                top_k=5,
            )

            similarity = verification["similarity_score"]
            recommendation = verification["recommendation"]
            rag_used = True

            logger.info(f"   RAG result: {recommendation}")
            logger.info(f"   Similarity: {similarity:.4f}")

            base_pred = base_result["prediction"]
            base_conf = base_result["confidence"]

            # Handle empty matching_articles
            if not verification["matching_articles"]:
                recommendation = "NO_RELIABLE_SOURCE_FOUND"
                logger.info("   No matching articles found")

            if verification["matching_articles"]:
                top_article = verification["matching_articles"][0]
                logger.info(f"   Top match: {top_article.get('source', 'unknown')}")
                logger.info(f"   Title: {top_article.get('title', '')[:80]}...")

            # ==================== DECISION LOGIC (SAME AS /predict) ====================

            if base_pred == "REAL":
                # ========== BASE REAL ==========

                if recommendation == "VERIFIED":
                    logger.info("✅ VERIFIED: Strong evidence confirms REAL")
                    method = "rag_enhanced"
                    final_conf = min(0.98, base_conf * 1.15)
                    prediction = "REAL"
                    logger.info(f"   Boosted: {base_conf:.4f} → {final_conf:.4f}")

                elif recommendation == "NEEDS_REVIEW":
                    logger.info("⚠️ NEEDS_REVIEW: Weak evidence supports REAL")
                    method = "rag_weak_support"
                    final_conf = min(0.95, base_conf * 1.05)
                    prediction = "REAL"
                    logger.info(f"   Boosted slightly: {base_conf:.4f} → {final_conf:.4f}")

                else:  # NO_RELIABLE_SOURCE_FOUND
                    logger.warning("🚨 NO_SOURCE: REAL claim but no evidence found")
                    method = "no_evidence"
                    final_conf = base_conf * 0.75
                    prediction = "REAL"
                    logger.info(f"   Penalty: {base_conf:.4f} → {final_conf:.4f}")

            else:
                # ========== BASE FAKE ==========

                if recommendation == "VERIFIED":
                    if similarity >= 0.83:
                        logger.warning("⚠️ OVERRIDE: Found strong evidence of REAL news!")
                        method = "rag_override"
                        final_conf = similarity
                        prediction = "REAL"
                        logger.info(f"   Overridden FAKE → REAL: {final_conf:.4f}")
                    else:
                        logger.info("⚠️ VERIFIED but weak (<0.83) → Reduce FAKE confidence")
                        method = "rag_doubt"
                        final_conf = base_conf * 0.85
                        prediction = "FAKE"
                        logger.info(f"   Reduced: {base_conf:.4f} → {final_conf:.4f}")

                elif recommendation == "NEEDS_REVIEW":
                    logger.info("⚠️ NEEDS_REVIEW: Found similar real news")
                    method = "rag_doubt"
                    final_conf = base_conf * 0.85
                    prediction = "FAKE"
                    logger.info(f"   Reduced: {base_conf:.4f} → {final_conf:.4f}")

                else:  # NO_RELIABLE_SOURCE_FOUND
                    logger.info("✅ NO_SOURCE: FAKE + no real news = Expected")
                    method = "base_model"
                    final_conf = base_conf
                    prediction = "FAKE"
                    logger.info(f"   Keeping original: {base_conf:.4f}")

            final_result = {
                "prediction": prediction,
                "confidence": final_conf,
                "probabilities": {
                    prediction: final_conf,
                    "REAL" if prediction == "FAKE" else "FAKE": 1 - final_conf
                }
            }

        prediction = final_result["prediction"]
        confidence = final_result["confidence"]
        probabilities = final_result["probabilities"]

        logger.info("📊 Final text result:")
        logger.info(f"   Prediction: {prediction}")
        logger.info(f"   Confidence: {confidence:.4f}")
        logger.info(f"   Method: {method}")
        logger.info(f"   RAG used: {rag_used}")

        processing_time = (time.time() - start) * 1000
        logger.info(f"✅ Completed text analysis in {processing_time:.0f}ms")
        logger.info("=" * 70)

        return PredictResponse(
            video_id="TEXT_INPUT",
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
        logger.error(f"❌ Text prediction error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))