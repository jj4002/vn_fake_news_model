# services/media_processor.py
import os
from pathlib import Path
import cv2
import logging


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
    
    def download_media(self, url: str, video_id: str = None):
        """Download TikTok media using yt-dlp"""
        if not YT_DLP_AVAILABLE:
            raise ImportError("yt-dlp is not installed")
        
        try:
            logger.info(f"📥 Downloading: {url}")
            
            ydl_opts = {
                'format': 'best',
                'outtmpl': str(self.temp_dir / '%(id)s.%(ext)s'),
                'quiet': True,
                'no_warnings': True,
                'no_check_certificates': True
            }
            
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=True)
                file_path = ydl.prepare_filename(info)
                
                if not os.path.exists(file_path):
                    raise FileNotFoundError(f"Downloaded file not found: {file_path}")
                
                # Determine type
                ext = Path(file_path).suffix.lower()
                if ext in ['.mp4', '.webm', '.mov', '.avi']:
                    media_type = 'video'
                elif ext in ['.jpg', '.jpeg', '.png', '.webp']:
                    media_type = 'image'
                else:
                    media_type = 'unknown'
                
                logger.info(f"✅ Downloaded: {file_path} ({media_type})")
                return file_path, media_type
                
        except Exception as e:
            logger.error(f"❌ Download error: {e}")
            raise
    
    def extract_frames(self, video_path: str, num_frames: int = 3):
        """Extract frames from video using OpenCV"""
        try:
            logger.info(f"📸 Extracting {num_frames} frames...")
            
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                logger.error("❌ Failed to open video")
                return []
            
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if total_frames == 0:
                cap.release()
                return []
            
            # Sample frames evenly
            frame_indices = [int(total_frames * i / (num_frames + 1)) 
                           for i in range(1, num_frames + 1)]
            frames = []
            
            for idx in frame_indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret, frame = cap.read()
                if ret:
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frames.append(frame_rgb)
            
            cap.release()
            logger.info(f"✅ Extracted {len(frames)} frames")
            return frames
            
        except Exception as e:
            logger.error(f"❌ Frame extraction error: {e}")
            return []
    
    def extract_audio(self, video_path: str):
        """Extract audio using moviepy"""
        try:
            audio_path = str(self.temp_dir / f"{Path(video_path).stem}.wav")
            
            logger.info(f"🔊 Extracting audio with moviepy...")
            
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
                    logger=None
                )
                video.close()
                
                if os.path.exists(audio_path):
                    file_size = os.path.getsize(audio_path) / 1024
                    logger.info(f"✅ Audio extracted: {file_size:.1f} KB")
                    return audio_path
                else:
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
