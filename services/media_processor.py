# services/media_processor.py
import os
from pathlib import Path
import cv2
import logging
from typing import List
import numpy as np

logger = logging.getLogger(__name__)

# Check yt-dlp
try:
    import yt_dlp
    YT_DLP_AVAILABLE = True
except ImportError:
    YT_DLP_AVAILABLE = False
    logger.warning("⚠️ yt-dlp not available")


class MediaProcessor:
    def __init__(self, temp_dir: str = "./temp"):
        self.temp_dir = Path(temp_dir)
        self.temp_dir.mkdir(exist_ok=True, parents=True)
        logger.info(f"🗂 Temp dir: {self.temp_dir.resolve()}")

    def download_media(self, url: str, video_id: str = None):
        """Download TikTok media (video or photo) using yt-dlp"""
        if not YT_DLP_AVAILABLE:
            raise ImportError("yt-dlp is not installed")

        try:
            logger.info(f"📥 Downloading: {url}")

            # ✅ FIX: TikTok photo mode → convert /photo/ to /video/
            if "/photo/" in url:
                logger.warning("⚠️ Photo URL detected, converting to /video/ endpoint for yt-dlp")
                url = url.replace("/photo/", "/video/")
                logger.info(f"   Converted URL: {url}")

            out_tmpl = str(self.temp_dir / '%(id)s.%(ext)s')
            ydl_opts = {
                # Video: best video+audio
                # Photo mode: yt-dlp sẽ tải từng ảnh
                'format': 'best',
                'outtmpl': out_tmpl,
                'quiet': True,
                'no_warnings': True,
                'no_check_certificates': True,
            }

            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=True)
                logger.debug(f"yt-dlp info keys: {list(info.keys())}")

                # Photo mode có thể trả về "entries" (nhiều ảnh)
                if 'entries' in info:
                    logger.info(f"📷 TikTok photo mode detected: {len(info['entries'])} images")
                    entry = info['entries'][0]
                    file_path = ydl.prepare_filename(entry)
                else:
                    file_path = ydl.prepare_filename(info)

            if not os.path.exists(file_path):
                raise FileNotFoundError(f"Downloaded file not found: {file_path}")

            ext = Path(file_path).suffix.lower()
            if ext in ['.mp4', '.webm', '.mov', '.avi']:
                media_type = 'video'
            elif ext in ['.jpg', '.jpeg', '.png', '.webp']:
                media_type = 'image'
            elif ext in ['.m4a', '.mp3', '.aac', '.wav']:
                media_type = 'audio'
            else:
                media_type = 'unknown'


            size_mb = os.path.getsize(file_path) / (1024 * 1024)
            logger.info(f"✅ Downloaded: {file_path} ({media_type}, {size_mb:.2f} MB)")
            return str(file_path), media_type

        except Exception as e:
            logger.error(f"❌ Download error: {e}")
            raise

    def extract_frames(self, video_path: str, max_frames: int = 5) -> List[np.ndarray]:
        """Extract frames from video for OCR"""
        frames: List[np.ndarray] = []

        try:
            if not video_path:
                logger.error("Video path is empty")
                return []

            video_path = str(video_path)

            if not os.path.exists(video_path):
                logger.error(f"Video file not found: {video_path}")
                return []

            cap = cv2.VideoCapture(video_path)

            if not cap.isOpened():
                logger.error(f"Cannot open video: {video_path}")
                return []

            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            duration = total_frames / fps if fps and fps > 0 else 0.0

            logger.info(
                f"🎞 Video info → frames={total_frames}, fps={fps:.2f}, "
                f"duration={duration:.2f}s"
            )

            if total_frames == 0:
                logger.warning("Video has 0 frames")
                cap.release()
                return []

            # Video ngắn / slideshow → lấy frame dày hơn
            if duration <= 15:
                step = max(int(fps // 2) if fps and fps > 0 else 1, 1)  # 0.5s/frame
                indices = list(range(0, total_frames, step))
                logger.info(
                    f"📸 Short video/slideshow detected → step={step}, "
                    f"candidate frames={len(indices)}"
                )
            else:
                interval = max(1, total_frames // max_frames)
                indices = list(range(0, total_frames, interval))
                logger.info(
                    f"📸 Long video → interval={interval}, "
                    f"candidate frames={len(indices)}"
                )

            extracted = 0
            for i in indices:
                if extracted >= max_frames:
                    break

                cap.set(cv2.CAP_PROP_POS_FRAMES, i)
                ret, frame = cap.read()

                if not ret or frame is None:
                    logger.debug(f"⚠️ Failed to read frame at index {i}")
                    continue

                h, w = frame.shape[:2]
                if w > 1280:
                    scale = 1280 / w
                    new_w = 1280
                    new_h = int(h * scale)
                    frame = cv2.resize(frame, (new_w, new_h))
                    logger.debug(f"↳ Resized frame {i} from {w}x{h} to {new_w}x{new_h}")

                frames.append(frame)
                extracted += 1
                logger.debug(f"📸 Captured frame index={i} (#{extracted})")

            cap.release()
            logger.info(f"✅ Extracted {len(frames)} frames for OCR")
            return frames

        except Exception as e:
            logger.error(f"Frame extraction error: {e}")
            return []

    def extract_audio(self, video_path: str):
        """Extract audio using moviepy"""
        try:
            audio_path = str(self.temp_dir / f"{Path(video_path).stem}.wav")

            logger.info(f"🔊 Extracting audio with moviepy... ({video_path})")

            try:
                from moviepy.editor import VideoFileClip

                video = VideoFileClip(video_path)

                if video.audio is None:
                    logger.warning("⚠️ Video has no audio track")
                    video.close()
                    return None

                video.audio.write_audiofile(
                    audio_path,
                    fps=16000,
                    nbytes=2,
                    codec='pcm_s16le',
                    verbose=False,
                    logger=None,
                )
                video.close()

                if os.path.exists(audio_path):
                    file_size = os.path.getsize(audio_path) / 1024
                    logger.info(f"✅ Audio extracted: {file_size:.1f} KB → {audio_path}")
                    return audio_path
                else:
                    logger.error("❌ Audio file not created")
                    return None

            except ImportError:
                logger.error("❌ moviepy not installed")
                return None

        except Exception as e:
            logger.error(f"❌ Audio extraction error: {e}")
            return None

    def cleanup(self, *paths):
        """Cleanup temp files"""
        for path in paths:
            try:
                if path and os.path.exists(path):
                    os.remove(path)
                    logger.debug(f"🗑️ Cleaned: {path}")
            except Exception as e:
                logger.error(f"Cleanup error: {e}")
