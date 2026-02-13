"""
Download models from Hugging Face on first run
"""
import os
from huggingface_hub import snapshot_download

MODEL_DIR = "./models"
HUGGINGFACE_REPO = "jamus0702/vn_fake_news_v4.1"

def download_models():
    """Download models if not present"""
    if not os.path.exists(MODEL_DIR) or not os.listdir(MODEL_DIR):
        print("Downloading models from Hugging Face...")
        os.makedirs(MODEL_DIR, exist_ok=True)
        
        try:
            snapshot_download(
                repo_id=HUGGINGFACE_REPO,
                local_dir=MODEL_DIR,
                local_dir_use_symlinks=False
            )
            print("Models downloaded successfully!")
        except Exception as e:
            print(f"Failed to download models: {e}")
            print("Please ensure you have internet connection and the model exists.")
            return False
    else:
        print("Models already exist, skipping download")
    return True

if __name__ == "__main__":
    download_models()