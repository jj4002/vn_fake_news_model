# scripts/setup_vncorenlp.py
from py_vncorenlp import download_model

if __name__ == "__main__":
    # Tải VnCoreNLP-1.x.jar + models vào thư mục ./vncorenlp
    download_model(save_dir='./vncorenlp')
    print("✅ Downloaded VnCoreNLP jar + models to ./vncorenlp")
