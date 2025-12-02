# services/stt_service.py
import logging
import os
from typing import Optional
import numpy as np

logger = logging.getLogger(__name__)    

try:
    import whisper
    STT_AVAILABLE = True
except ImportError:
    STT_AVAILABLE = False

try:
    import librosa
    LIBROSA_AVAILABLE = True
    logger.info("✅ librosa available for audio loading")
except ImportError:
    LIBROSA_AVAILABLE = False
    logger.warning("⚠️ librosa not available")

class STTService:
    def __init__(self, model_name: str = "large-v3", device: str = "cpu"):
        self.available = STT_AVAILABLE
        self.device = device
        self.model = None

        if self.available:
            try:
                logger.info(f"⏳ Loading Whisper model: {model_name} on {device}...")
                self.model = whisper.load_model(model_name, device=device)
                logger.info(f"✅ Whisper loaded ({model_name}, device={device})")
            except Exception as e:
                logger.error(f"❌ Failed to load Whisper: {e}")
                self.available = False
        else:
            logger.warning("⚠️ whisper package not available")

    def transcribe_audio(self, audio_path: str, language: str = "vi") -> Optional[str]:
        if not self.available or not audio_path:
            return None

        audio_path = os.path.abspath(audio_path)
        if not os.path.exists(audio_path):
            logger.error(f"❌ Audio file not found: {audio_path}")
            return None

        try:
            logger.info(f"🎤 Transcribing: {audio_path}")
            logger.info(f"   File size: {os.path.getsize(audio_path) / 1024:.1f} KB")

            if LIBROSA_AVAILABLE:
                logger.info("   Loading audio with librosa...")
                audio, sr = librosa.load(audio_path, sr=16000, mono=True)
                audio = audio.astype(np.float32)
                logger.info(f"   Audio loaded: {len(audio) / sr:.1f}s")
            else:
                logger.warning("   Using Whisper audio loader (requires FFmpeg)")
                audio = audio_path

            result = self.model.transcribe(
                audio,
                language=language,
                fp16=(self.device != "cpu"),
                verbose=False,
                # initial_prompt="Đây là nội dung tiếng Việt",
            )

            text = result.get("text", "").strip()
            if text:
                logger.info(f"✅ STT: {len(text)} chars")
                logger.info(f"   Preview: {text[:100]}...")
                return text
            else:
                logger.warning("⚠️ STT empty")
                return None

        except Exception as e:
            logger.error(f"❌ STT error: {e}", exc_info=True)
            return None
