# 🔍 TikTok Fake News Detector

Hệ thống phát hiện tin giả trên TikTok sử dụng AI, tích hợp Chrome Extension và Backend API với các công nghệ Machine Learning tiên tiến.

## 📋 Tổng quan

Dự án này là một hệ thống hoàn chỉnh để phát hiện tin giả trên nền tảng TikTok, bao gồm:

- **Chrome Extension**: Extension trình duyệt để phân tích video TikTok trực tiếp trên trang web
- **Backend API**: API server Python sử dụng FastAPI để xử lý phân tích và dự đoán
- **Machine Learning Model**: Mô hình HAN (Hierarchical Attention Network) được tối ưu hóa với ONNX Runtime
- **RAG System**: Hệ thống Retrieval-Augmented Generation để xác minh thông tin với nguồn tin đáng tin cậy
- **Media Processing**: Xử lý video/ảnh với OCR (Optical Character Recognition) và STT (Speech-to-Text)

## 🏗️ Kiến trúc hệ thống

```
┌─────────────────┐
│ Chrome Extension│
│  (extension/)   │
└────────┬────────┘
         │ HTTP API
         ▼
┌─────────────────┐
│  FastAPI Server │
│   (backend/)    │
└────────┬────────┘
         │
    ┌────┴────┐
    │        │
    ▼        ▼
┌────────┐ ┌──────────┐
│  HAN   │ │   RAG    │
│ Model  │ │  Service │
└────────┘ └────┬─────┘
                │
                ▼
         ┌──────────────┐
         │  Supabase DB │
         │  (PostgreSQL)│
         └──────────────┘
```

## 📁 Cấu trúc thư mục

```
detect-fake-news/
├── backend/              # Python Backend API
│   ├── routers/         # API endpoints
│   ├── services/        # Business logic
│   ├── scripts/         # Utility scripts
│   └── main.py          # FastAPI app entry
│
├── extension/            # Chrome Extension
│   ├── background/       # Service worker
│   ├── content/          # Content scripts
│   ├── popup/            # Extension popup UI
│   └── manifest.json     # Extension manifest
│
├── crawl/               # Data crawling scripts
│   ├── crawl_video.py   # TikTok video crawler
│   └── *.ipynb          # Data processing notebooks
│
└── train/               # Model training & experiments
    ├── train-baseline-phobert.py    # Experiment 1: Baseline PhoBERT
    ├── train-author-embedding.py    # Experiment 2: PhoBERT + Author Embedding
    ├── train-MLM_Prompt.py          # Experiment 3: Prompt-based MLM
    └── train-rag-han.ipynb          # Experiment 4: HAN + RAG (Production)
```

## 🚀 Cài đặt và Chạy

### Yêu cầu hệ thống

- Python 3.8+
- Node.js 16+
- Chrome/Edge browser
- PostgreSQL với pgvector extension (hoặc Supabase)
- FFmpeg (cho xử lý media)

### 1. Cài đặt Backend API

```bash
cd backend
pip install -r requirement.txt
```

Tạo file `.env`:
```env
SUPABASE_URL=your_supabase_url
SUPABASE_KEY=your_supabase_key
MODEL_PATH=./models/han_rag_model.onnx
TOKENIZER_PATH=vinai/phobert-base-v2
EMBEDDING_MODEL=keepitreal/vietnamese-sbert
PORT=8000
HOST=0.0.0.0
```

Chạy server:
```bash
python main.py
```

### 2. Cài đặt Chrome Extension

```bash
cd extension
npm install
```

Load extension vào Chrome:
1. Mở `chrome://extensions/`
2. Bật "Developer mode"
3. Click "Load unpacked"
4. Chọn thư mục `extension/`

### 3. Setup Database

Chạy SQL schema từ `extension/database/supabase_schema.sql` trên Supabase hoặc PostgreSQL.

## 🎯 Tính năng chính

### 1. Phân tích Video TikTok
- Tự động trích xuất caption, OCR text, và STT từ video
- Dự đoán tin giả/thật với độ tin cậy
- Cache kết quả để tối ưu hiệu suất

### 2. RAG Verification
- Tìm kiếm bài viết tương tự từ nguồn tin đáng tin cậy
- Xác minh thông tin với similarity search
- Điều chỉnh confidence dựa trên bằng chứng

### 3. Heuristic Rules
- Phát hiện clickbait patterns
- Nhận diện tuyên bố tài chính không có nguồn chính thức
- Xử lý các pattern đặc biệt của tiếng Việt

### 4. User Reporting
- Người dùng có thể báo cáo kết quả sai
- Hệ thống tracking để cải thiện model

## 🔧 Công nghệ sử dụng

### Backend
- **FastAPI**: Web framework
- **ONNX Runtime**: Model inference tối ưu
- **Supabase**: Database và vector search
- **Sentence Transformers**: Embedding generation
- **VietOCR**: OCR tiếng Việt
- **Whisper**: Speech-to-Text
- **yt-dlp**: Video download

### Frontend
- **Chrome Extension API**: Extension development
- **Vanilla JavaScript**: UI logic
- **ONNX Runtime Web**: Client-side inference (optional)

### ML/AI
- **HAN Model**: Hierarchical Attention Network
- **PhoBERT**: Vietnamese BERT tokenizer
- **Vietnamese SBERT**: Sentence embeddings
- **RAG**: Retrieval-Augmented Generation

## 📊 Model Architecture

### HAN Model
- **Input**: Title (caption) + Content (OCR + STT)
- **Tokenizer**: PhoBERT-base-v2
- **Architecture**: Hierarchical Attention với chunk selection
- **Output**: Binary classification (REAL/FAKE) với confidence score
- **Model trên HuggingFace**: [vn_fake_news_v2](https://huggingface.co/jamus0702/vn_fake_news_v2/tree/main)

### RAG Pipeline
1. Chunk selection từ content dựa trên title similarity
2. Vector search trong news corpus
3. Similarity threshold: 0.75
4. Confidence adjustment dựa trên matching articles

## 📝 API Endpoints

### `/api/v1/predict`
Dự đoán tin giả/thật từ video TikTok

**Request:**
```json
{
  "video_id": "1234567890",
  "video_url": "https://tiktok.com/@user/video/123",
  "caption": "Video caption...",
  "ocr_text": "Text from OCR...",
  "stt_text": "Text from STT...",
  "author_id": "username"
}
```

**Response:**
```json
{
  "video_id": "1234567890",
  "prediction": "FAKE",
  "confidence": 0.85,
  "method": "rag_enhanced",
  "rag_used": true,
  "probabilities": {
    "REAL": 0.15,
    "FAKE": 0.85
  },
  "processing_time_ms": 1234.5
}
```

### `/api/v1/process-media`
Xử lý media (OCR + STT)

### `/api/v1/report`
Báo cáo kết quả sai

## 🧪 Testing

```bash
# Test API
curl http://localhost:8000/health

# Test prediction
curl -X POST http://localhost:8000/api/v1/predict \
  -H "Content-Type: application/json" \
  -d @test_request.json
```

## 📈 Performance

- **Prediction time**: ~1-3 giây (không cache)
- **Cache hit**: <100ms
- **Media processing**: ~5-10 giây (OCR + STT)
- **RAG search**: ~500ms-1s

## 🔒 Bảo mật

- Row Level Security (RLS) trên Supabase
- Service role authentication
- Input validation và sanitization
- CORS middleware

## 📚 Tài liệu thêm

- [Backend API README](backend/README.md)
- [Chrome Extension README](extension/README.md)
- [Crawling Scripts README](crawl/README.md)
- [Training & Experiments Guide](train/README.md)

## 📄 License

Dự án này được phát hành dưới giấy phép MIT.

## 👥 Tác giả

- *[Đặng Thị Bích Trâm](https://github.com/jj4002)*
- *[Đỗ Minh Bảo Huy](https://github.com/ddooxhuy09)*
- *[Trần Anh Tuấn](https://github.com/tuanhqv123)*

## 🙏 Acknowledgments

- PhoBERT team cho Vietnamese BERT model
- VietOCR team cho OCR tiếng Việt
- OpenAI Whisper cho STT
- Supabase cho infrastructure
- Model được đăng tải trên [HuggingFace](https://huggingface.co/jamus0702/vn_fake_news_v2/tree/main)

## 📊 Datasets

Dự án sử dụng các datasets sau cho training và evaluation:

- **[Vietnamese Fake News Detection](https://github.com/hiepnguyenduc2005/Vietnamese-Fake-News-Detection)**: Dataset từ ReINTEL với gần 10,000 examples được gán nhãn, sử dụng cho training baseline models
- **[VFND Vietnamese Fake News Datasets](https://github.com/WhySchools/VFND-vietnamese-fake-news-datasets)**: Tập hợp các bài báo tiếng Việt và Facebook posts được phân loại (228-254 bài), bao gồm cả Article Contents và Social Contents