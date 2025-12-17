# routers/media.py
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional
import logging

from services.media_processor import MediaProcessor
from services.ocr_service import OCRService
from services.stt_service import STTService
from services.supabase_client import SupabaseService  # ← ADD THIS

logger = logging.getLogger(__name__)
router = APIRouter()

# Initialize services
media_processor = MediaProcessor()
ocr_service = OCRService()
stt_service = STTService()
db = SupabaseService()  # ← ADD THIS

class MediaRequest(BaseModel):
    video_id: str
    video_url: str

class MediaResponse(BaseModel):
    video_id: str
    ocr_text: str
    stt_text: str
    processing_time_ms: float

@router.post("/process-media", response_model=MediaResponse)
async def process_media(request: MediaRequest):
    import time
    start = time.time()

    file_path = None
    audio_path = None

    try:
        logger.info("="*70)
        logger.info(f"📹 Processing media: {request.video_id}")

        # ✅ CHECK CACHE FIRST
        logger.info("🔍 Checking cache before processing...")
        cached = db.get_video(request.video_id)

        if cached:
            logger.info(f"✅ Cache hit: {request.video_id}")
            logger.info(f"   Using cached OCR/STT data")

            processing_time = (time.time() - start) * 1000
            logger.info(f"✅ Returned from cache in {processing_time:.0f}ms")
            logger.info("="*70)

            return MediaResponse(
                video_id=request.video_id,
                ocr_text=cached.get("ocr_text", ""),
                stt_text=cached.get("stt_text", ""),
                processing_time_ms=processing_time,
            )

        logger.info("✅ No cache found, processing media...")

        # Download
        logger.info("⬇️ Downloading...")
        file_path, media_type = media_processor.download_media(
            request.video_url, request.video_id
        )

        logger.info(f"   Downloaded: {file_path}")
        logger.info(f"   Type: {media_type}")

        ocr_text = ""
        stt_text = ""

        if media_type == "video":
            logger.info("🎬 Processing VIDEO (OCR + STT)")

            # Extract frames for OCR
            logger.info("📸 Extracting frames for OCR...")
            frames = media_processor.extract_frames(file_path, max_frames=5)

            # Run OCR
            logger.info(f"🖼️ Running OCR on {len(frames)} frames...")
            ocr_text = ocr_service.extract_text_from_frames(frames)
            logger.info(f"   ✅ OCR: {len(ocr_text)} chars")
            logger.info(f"   OCR preview: {ocr_text[:100]}...")

            # Extract audio for STT
            logger.info("🔊 Extracting audio for STT...")
            audio_path = media_processor.extract_audio(file_path)
            logger.info(f"   Audio saved: {audio_path}")

            # Run STT
            logger.info("🎤 Running Speech-to-Text...")
            stt_text = stt_service.transcribe_audio(audio_path, language="vi") or ""
            logger.info(f"   ✅ STT: {len(stt_text)} chars")
            logger.info(f"   STT preview: {stt_text[:100]}...")

        elif media_type == "image":
            logger.info("🖼️ Processing IMAGE (OCR only)")
            logger.info("📸 Running OCR on image...")
            # Bạn đã có hàm extract_text_from_image hoặc dùng frames = [cv2.imread(...)]
            ocr_text = ocr_service.extract_text_from_image(file_path)
            logger.info(f"   ✅ OCR: {len(ocr_text)} chars")
            logger.info(f"   OCR preview: {ocr_text[:100]}...")

        elif media_type == "audio":
            # Case TikTok photo mode mà yt-dlp chỉ trả nhạc nền
            logger.info("🎧 Audio-only media (photo mode) → bỏ qua STT, không có hình để OCR")
            # Nếu sau này audio có lời, có thể bật STT ở đây:
            # stt_text = stt_service.transcribe_audio(file_path, language="vi") or ""

        else:
            logger.warning(f"⚠️ Unsupported media_type: {media_type} → skip OCR/STT")

        processing_time = (time.time() - start) * 1000

        logger.info("="*70)
        logger.info("✅ Media processing complete:")
        logger.info(f"   OCR: {len(ocr_text)} chars")
        logger.info(f"   STT: {len(stt_text)} chars")
        logger.info(f"   Time: {processing_time:.0f}ms")
        logger.info("="*70)

        return MediaResponse(
            video_id=request.video_id,
            ocr_text=ocr_text,
            stt_text=stt_text,
            processing_time_ms=processing_time,
        )

    except Exception as e:
        logger.error(f"❌ Media processing error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        # 🧹 Luôn dọn file tạm (video + audio) sau khi xử lý xong
        try:
            media_processor.cleanup(file_path, audio_path)
        except Exception:
            # tránh làm vỡ response nếu cleanup lỗi
            pass

