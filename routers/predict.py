# routers/predict.py
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional
import time
import logging

from services.supabase_client import SupabaseService
from services.rag_service import RAGService
from services.inference import ONNXInference

logger = logging.getLogger(__name__)
router = APIRouter()

# Initialize core services
db = SupabaseService()
rag_service = RAGService()
model = ONNXInference()


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

        # Validate input
        if not request.caption or len(request.caption.strip()) == 0:
            logger.error("❌ Invalid input: Empty caption")
            raise HTTPException(status_code=400, detail="Caption is required")

        # Check cache
        logger.info(f"🔍 Checking cache for video: {request.video_id}")
        cached = db.get_video(request.video_id)

        if cached:
            logger.info(f"✅ Cache hit: {request.video_id}")
            logger.info(
                f"   Cached prediction: {cached['prediction']} ({cached['confidence']:.4f})"
            )
            logger.info(f"   Cached method: {cached.get('method', 'unknown')}")

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
        # Prepare input (simple, no VNTextPipeline)
        # =========================
        title_raw = request.caption
        title = title_raw.strip()

        content_parts = []

        if request.ocr_text and len(request.ocr_text.strip()) > 10:
            ocr_fixed = request.ocr_text.strip()
            content_parts.append(ocr_fixed)
            logger.info(f"   OCR: {len(request.ocr_text)} chars")

        if request.stt_text and len(request.stt_text.strip()) > 50:  # STT ưu tiên
            stt_fixed = request.stt_text.strip()
            content_parts.append(stt_fixed)
            logger.info(f"   STT: {len(request.stt_text)} chars")

        # ✅ FIX: Dùng content thực thay vì caption metadata
        if content_parts:
            content = " ".join(content_parts)[:2000]  # Giới hạn độ dài
            title = content[:200]  # Title preview cho log
        else:
            content = request.caption
            title = request.caption[:200]

        logger.info(f"📝 Input:")
        logger.info(f"   Title preview: {title}...")
        logger.info(f"   Content length: {len(content)} chars")
        logger.info(f"   Content preview: {content[:150]}...")

        # ================
        # Base prediction
        # ================
        logger.info("🤖 Running base model...")
        base_result = model.predict(title, content)

        logger.info(
            f"   Base result: {base_result['prediction']} ({base_result['confidence']:.4f})"
        )

        # Initialize final result
        rag_used = False
        method = "base_model"
        final_result = base_result

        # ===================
        # RAG verification
        # ===================
        if rag_service.should_use_rag(title, content, base_result["confidence"], request.author_id or ""):
            logger.info("🔍 Running RAG verification...")

            verification = rag_service.verify_with_sources(title, content)

            logger.info(f"   RAG result: {verification['recommendation']}")
            logger.info(f"   Similarity: {verification['similarity_score']:.2f}")

            if verification["matching_articles"]:
                rag_used = True
                method = "rag_enhanced"

                similarity = verification["similarity_score"]
                recommendation = verification["recommendation"]

                if base_result["prediction"] == "FAKE":
                    logger.info(
                        "🔍 Base prediction is FAKE, checking RAG evidence..."
                    )

                    if recommendation == "VERIFIED_REAL":
                        # Chỉ khi rất giống (>= 0.8) mới flip sang REAL
                        logger.warning("⚠️ VERIFIED_REAL → Switching to REAL")
                        new_conf = max(0.7, min(0.95, 0.7 + (similarity - 0.8)))
                        final_result = {
                            "prediction": "REAL",
                            "confidence": new_conf,
                            "probabilities": {
                                "REAL": new_conf,
                                "FAKE": 1 - new_conf,
                            },
                        }
                        logger.info(
                            f"   Overridden to: REAL ({final_result['confidence']:.4f})"
                        )

                    elif recommendation == "NEEDS_REVIEW":
                        # Giảm rất nhẹ, chỉ để báo “nên xem lại”
                        logger.warning(
                            "⚠️ NEEDS_REVIEW → Slightly reducing FAKE confidence"
                        )
                        adjusted_conf = max(
                            0.5, base_result["confidence"] * 0.9
                        )  # ví dụ 1.0 → 0.9
                        final_result = {
                            "prediction": "FAKE",
                            "confidence": adjusted_conf,
                            "probabilities": {
                                "FAKE": adjusted_conf,
                                "REAL": 1 - adjusted_conf,
                            },
                        }
                        logger.info(
                            f"   Reduced confidence: {base_result['confidence']:.4f} → "
                            f"{adjusted_conf:.4f}"
                        )
                    else:
                        logger.info("   No strong evidence → Keep base prediction")
                        final_result = base_result

                elif base_result["prediction"] == "REAL":
                    if recommendation == "VERIFIED_REAL":
                        logger.info("✅ Base REAL confirmed by RAG (strong)")
                        boosted_conf = min(0.98, base_result["confidence"] * 1.1)
                        final_result = {
                            "prediction": "REAL",
                            "confidence": boosted_conf,
                            "probabilities": {
                                "REAL": boosted_conf,
                                "FAKE": 1 - boosted_conf,
                            },
                        }
                        logger.info(
                            f"   Boosted confidence: {base_result['confidence']:.4f} → "
                            f"{boosted_conf:.4f}"
                        )
                    else:
                        logger.info("   RAG did not strongly confirm → Keep base")
                        final_result = base_result
            else:
                logger.info("   No matching articles found, using base result")
                final_result = base_result


        # Extract final prediction
        prediction = final_result["prediction"]
        confidence = final_result["confidence"]
        probabilities = final_result["probabilities"]

        # Save to cache
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
