# Backend API Server

FastAPI server cung cấp API để phát hiện tin giả trên TikTok với các tính năng ML/AI tiên tiến.

## 📋 Tổng quan

Backend này cung cấp:
- **Prediction API**: Dự đoán tin giả/thật từ video TikTok
- **Media Processing**: OCR và Speech-to-Text từ video
- **RAG Verification**: Xác minh với nguồn tin đáng tin cậy
- **Caching**: Lưu kết quả để tối ưu performance
- **Reporting**: Hệ thống báo cáo để cải thiện model

## 🏗️ Kiến trúc

```
┌──────────────┐
│   FastAPI    │
│   (main.py)  │
└──────┬───────┘
       │
   ┌───┴───┐
   │       │
   ▼       ▼
┌──────┐ ┌──────────┐
│Router│ │ Services │
└──┬───┘ └────┬─────┘
   │          │
   │    ┌─────┴─────┐
   │    │           │
   ▼    ▼           ▼
┌────┐ ┌────┐ ┌──────────┐
│Pred│ │Med │ │   RAG    │
│ict │ │ia  │ │ Service  │
└──┬─┘ └──┬─┘ └─────┬─────┘
   │      │         │
   │      │         │
   ▼      ▼         ▼
┌──────┐ ┌──────┐ ┌──────────┐
│HAN   │ │OCR/  │ │ Supabase │
│Model │ │STT   │ │   DB     │
└──────┘ └──────┘ └──────────┘
```

## 📁 Cấu trúc thư mục

```
backend/
├── main.py                 # FastAPI app entry point
├── requirement.txt          # Python dependencies
│
├── routers/                # API endpoints
│   ├── predict.py          # Prediction endpoint
│   ├── media.py            # Media processing endpoint
│   └── reports.py          # Reporting endpoint
│
├── services/               # Business logic
│   ├── inference.py        # HAN model inference
│   ├── rag_service.py      # RAG verification
│   ├── media_processor.py  # Video/image processing
│   ├── ocr_service.py     # OCR service
│   ├── stt_service.py     # Speech-to-Text service
│   └── supabase_client.py # Database client
│
└── scripts/                # Utility scripts
    ├── generate_embeddings.py
    └── regenerate_embeddings.py
```

## 🚀 Cài đặt

### 1. Cài đặt dependencies

```bash
pip install -r requirement.txt
```

**Key dependencies:**
- `fastapi`: Web framework
- `uvicorn`: ASGI server
- `onnxruntime`: Model inference
- `sentence-transformers`: Embeddings
- `supabase`: Database client
- `vietocr`: Vietnamese OCR
- `openai-whisper`: Speech-to-Text
- `yt-dlp`: Video download
- `opencv-python`: Image processing
- `moviepy`: Audio extraction

### 2. Cấu hình Environment Variables

Tạo file `.env`:

```env
# Supabase
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_KEY=your-service-role-key

# Model paths
MODEL_PATH=./models/han_rag_model.onnx
TOKENIZER_PATH=vinai/phobert-base-v2
EMBEDDING_MODEL=keepitreal/vietnamese-sbert

# Server
PORT=8000
HOST=0.0.0.0
```

### 3. Setup Database

Chạy SQL schema từ `extension/database/supabase_schema.sql` trên Supabase.

### 4. Chạy server

```bash
python main.py
```

Hoặc với uvicorn:
```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

Server sẽ chạy tại: `http://localhost:8000`

API docs: `http://localhost:8000/docs`

## 📝 API Endpoints

### 1. Health Check

```http
GET /health
```

**Response:**
```json
{
  "status": "healthy",
  "model": "loaded",
  "database": "connected"
}
```

### 2. Predict (`/api/v1/predict`)

Dự đoán tin giả/thật từ video TikTok.

**Request:**
```json
{
  "video_id": "1234567890",
  "video_url": "https://tiktok.com/@user/video/123",
  "caption": "Video caption text...",
  "ocr_text": "Text extracted from video frames...",
  "stt_text": "Transcribed audio text...",
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

**Prediction Methods:**
- `cached`: Kết quả từ cache
- `base_model`: Chỉ dùng HAN model
- `rag_enhanced`: Có sử dụng RAG verification

### 3. Process Media (`/api/v1/process-media`)

Xử lý media để extract OCR và STT.

**Request:**
```json
{
  "video_id": "1234567890",
  "video_url": "https://tiktok.com/@user/video/123"
}
```

**Response:**
```json
{
  "video_id": "1234567890",
  "ocr_text": "Text from OCR...",
  "stt_text": "Text from STT...",
  "processing_time_ms": 5678.9
}
```

### 4. Report (`/api/v1/report`)

Báo cáo kết quả prediction sai.

**Request:**
```json
{
  "video_id": "1234567890",
  "reported_prediction": "FAKE",
  "reason": "Optional reason text..."
}
```

**Response:**
```json
{
  "status": "success",
  "message": "Report saved successfully"
}
```

### 5. Get Pending Reports (`/api/v1/reports/pending`)

Lấy danh sách reports đang chờ review (admin).

**Query params:**
- `limit`: Số lượng reports (default: 50)

## 🔧 Services Chi tiết

### Inference Service (`services/inference.py`)

**HANONNXInference Class:**
- Load ONNX model
- Text normalization (Vietnamese)
- Chunk selection với RAG
- Model prediction

**Methods:**
- `predict(title, content)`: Dự đoán với HAN model
- `_select_chunks_with_rag()`: Chọn chunks quan trọng

### RAG Service (`services/rag_service.py`)

**RAGService Class:**
- Vector similarity search
- Verification với news corpus
- Confidence adjustment

**Methods:**
- `should_use_rag()`: Quyết định có dùng RAG không
- `verify_with_sources()`: Tìm kiếm và verify

**RAG Triggers:**
- High confidence (>0.95)
- Clickbait patterns
- Sensitive topics
- Breaking news keywords
- Unknown source với high confidence

### Media Processor (`services/media_processor.py`)

**MediaProcessor Class:**
- Download video/image từ TikTok
- Extract frames cho OCR
- Extract audio cho STT

**Methods:**
- `download_media()`: Download với yt-dlp
- `extract_frames()`: Extract frames từ video
- `extract_audio()`: Extract audio track

### OCR Service (`services/ocr_service.py`)

**OCRService Class:**
- Sử dụng VietOCR (Vietnamese optimized)
- Extract text từ frames/images

**Methods:**
- `extract_text_from_frames()`: OCR từ video frames
- `extract_text_from_image()`: OCR từ image

### STT Service (`services/stt_service.py`)

**STTService Class:**
- Sử dụng OpenAI Whisper (large-v3)
- Transcribe audio sang text

**Methods:**
- `transcribe_audio()`: Speech-to-Text

### Supabase Client (`services/supabase_client.py`)

**SupabaseService Class:**
- Database operations
- Vector search
- Caching

**Methods:**
- `get_video()`: Lấy cached prediction
- `save_video()`: Lưu prediction
- `search_similar_news()`: Vector similarity search
- `save_report()`: Lưu user report

## 🧪 Testing

### Test với curl

```bash
# Health check
curl http://localhost:8000/health

# Predict
curl -X POST http://localhost:8000/api/v1/predict \
  -H "Content-Type: application/json" \
  -d '{
    "video_id": "test123",
    "video_url": "https://tiktok.com/@test/video/123",
    "caption": "Test caption"
  }'
```

### Test với Python

```python
import requests

response = requests.post(
    "http://localhost:8000/api/v1/predict",
    json={
        "video_id": "test123",
        "video_url": "https://tiktok.com/@test/video/123",
        "caption": "Test caption"
    }
)
print(response.json())
```

## 📊 Performance

### Benchmarks

- **Prediction (no cache)**: ~1-3 giây
- **Prediction (cached)**: <100ms
- **Media processing**: ~5-10 giây
- **RAG search**: ~500ms-1s

### Optimization

1. **Caching**: Kết quả được cache trong database
2. **Batch processing**: Có thể batch process media
3. **Async operations**: FastAPI async support
4. **Model optimization**: ONNX Runtime cho inference nhanh

## 🐛 Troubleshooting

### Model không load

**Vấn đề:** `FileNotFoundError: Model not found`
- **Giải pháp:** Kiểm tra `MODEL_PATH` trong `.env`

### Database connection failed

**Vấn đề:** `Supabase connection failed`
- **Giải pháp:** Kiểm tra `SUPABASE_URL` và `SUPABASE_KEY`

### OCR/STT không hoạt động

**Vấn đề:** `VietOCR/Whisper not available`
- **Giải pháp:** 
  - Cài đặt dependencies: `pip install vietocr openai-whisper`
  - Kiểm tra FFmpeg đã cài đặt

### Memory issues

**Vấn đề:** Out of memory khi process media
- **Giải pháp:**
  - Giảm số frames cho OCR
  - Sử dụng GPU nếu có
  - Tăng swap space

## 🔒 Security

- **CORS**: Configured cho extension origin
- **Input validation**: Pydantic models
- **SQL injection**: Supabase client tự động escape
- **RLS**: Row Level Security trên database

## 📈 Monitoring

### Logging

Server sử dụng Python logging:
- Level: INFO
- Format: Timestamp, level, message
- Output: Console

### Metrics (có thể thêm)

- Request count
- Response time
- Error rate
- Cache hit rate

## 🔮 Future Improvements

- [ ] WebSocket support cho real-time updates
- [ ] Batch prediction API
- [ ] Model versioning
- [ ] A/B testing framework
- [ ] Prometheus metrics
- [ ] Distributed caching (Redis)
- [ ] GPU support cho inference

## 📄 License

MIT License

