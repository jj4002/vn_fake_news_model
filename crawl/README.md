# Data Crawling Scripts

Các script để crawl và xử lý dữ liệu từ TikTok và các nguồn khác để xây dựng dataset cho training model.

## 📋 Tổng quan

Thư mục này chứa các script và notebook để:
- Crawl video TikTok theo channel/keyword
- Extract text từ video (STT)
- Clean và merge datasets
- Chuẩn bị data cho training

## 📁 Cấu trúc thư mục

```
crawl/
├── crawl_video.py              # Crawl video TikTok cơ bản
├── crawl_video_by_channel.py   # Crawl theo channel
├── clean_data_tiktok.ipynb      # Clean data TikTok
├── merge_dataset_fb,pp.ipynb    # Merge datasets từ Facebook, etc.
├── stt_fake_2.ipynb            # STT cho fake videos
├── stt_real.ipynb              # STT cho real videos
├── keyword_fake.txt            # Keywords để tìm fake news
├── list_channel_real.txt        # List channels đáng tin cậy
├── fake_all.csv                # Dataset fake news
└── tiktok_videos_all_keywords_real.csv  # Dataset real news
```

## 🚀 Sử dụng

### 1. Crawl Video TikTok

#### Crawl cơ bản (`crawl_video.py`)

```bash
python crawl_video.py
```

**Chức năng:**
- Đọc URLs từ CSV file
- Download video từ TikTok
- Transcribe audio với Whisper
- Lưu kết quả vào CSV

**Input:** `fake2_1.csv` (chứa URLs)
**Output:** `output2_1.csv` (chứa URLs + transcribed text)

#### Crawl theo Channel (`crawl_video_by_channel.py`)

```bash
python crawl_video_by_channel.py
```

**Chức năng:**
- Crawl videos từ TikTok channels
- Filter theo keywords
- Extract metadata và transcript

### 2. Data Processing Notebooks

#### Clean Data (`clean_data_tiktok.ipynb`)

**Chức năng:**
- Remove duplicates
- Clean text (remove special chars, normalize)
- Filter invalid entries
- Export cleaned dataset

#### Merge Datasets (`merge_dataset_fb,pp.ipynb`)

**Chức năng:**
- Merge datasets từ nhiều nguồn (Facebook, TikTok, etc.)
- Standardize format
- Balance classes (fake/real)

#### STT Processing (`stt_fake_2.ipynb`, `stt_real.ipynb`)

**Chức năng:**
- Batch process videos để extract STT
- Handle errors và retries
- Save progress để resume

## 📝 Chi tiết Scripts

### crawl_video.py

**Dependencies:**
- `yt-dlp`: Download video
- `whisper`: Speech-to-Text
- `torch`: PyTorch cho Whisper

**Functions:**
- `download_and_transcribe()`: Download và transcribe video
- `read_urls_from_csv()`: Đọc URLs từ CSV
- `save_result_to_csv()`: Lưu kết quả
- `process_videos_from_csv()`: Main processing function

**Usage:**
```python
# Sửa input/output files trong main()
input_csv = "fake2_1.csv"
output_csv = "output2_1.csv"

python crawl_video.py
```

### crawl_video_by_channel.py

**Chức năng:**
- Crawl videos từ TikTok channels
- Filter theo keywords từ `keyword_fake.txt`
- Extract metadata (caption, author, views, etc.)
- Save to CSV

**Usage:**
```bash
# Cấu hình channels và keywords trong script
python crawl_video_by_channel.py
```

## 📊 Data Format

### Input CSV Format

```csv
url,text
https://tiktok.com/@user/video/123,
https://tiktok.com/@user/video/456,
```

### Output CSV Format

```csv
url,text
https://tiktok.com/@user/video/123,Transcribed text from video...
https://tiktok.com/@user/video/456,Another transcribed text...
```

### Dataset Format (cho training)

```csv
title,content,label
Video caption,OCR text + STT text,FAKE
Another caption,More text content,REAL
```

## 🔧 Configuration

### Keywords (`keyword_fake.txt`)

Danh sách keywords để tìm fake news:
```
tặng tiền
phát tiền
nhận tiền ngay
virus mới
bệnh lạ
...
```

### Real Channels (`list_channel_real.txt`)

Danh sách channels đáng tin cậy:
```
@vnexpress
@vtv24
@vovtv
@60giay
...
```

## 🧪 Testing

### Test crawl single video

```python
from crawl_video import download_and_transcribe

video_url = "https://tiktok.com/@user/video/123"
text = download_and_transcribe(video_url, "test123")
print(text)
```

### Test với sample data

1. Tạo file `test_urls.csv` với vài URLs
2. Chạy script
3. Kiểm tra output

## 🐛 Troubleshooting

### Download failed

**Vấn đề:** `yt-dlp` không download được
- **Giải pháp:** 
  - Update yt-dlp: `pip install --upgrade yt-dlp`
  - Check TikTok URL format
  - Có thể cần VPN nếu bị block

### STT failed

**Vấn đề:** Whisper không transcribe được
- **Giải pháp:**
  - Check audio file tồn tại
  - Check FFmpeg đã cài
  - Thử model nhỏ hơn (base, small)

### Memory issues

**Vấn đề:** Out of memory khi process nhiều videos
- **Giải pháp:**
  - Process từng video một
  - Cleanup files sau mỗi video
  - Sử dụng batch processing với limit

### Rate limiting

**Vấn đề:** TikTok block requests
- **Giải pháp:**
  - Thêm delays giữa requests
  - Sử dụng proxies
  - Rotate user agents

## 📈 Best Practices

1. **Incremental processing**: Lưu progress để resume
2. **Error handling**: Catch và log errors
3. **Rate limiting**: Không spam requests
4. **Data validation**: Validate data trước khi save
5. **Backup**: Backup datasets thường xuyên

## 🔒 Legal & Ethics

⚠️ **Lưu ý quan trọng:**

- Tuân thủ TikTok Terms of Service
- Không crawl quá nhiều để tránh rate limit
- Respect privacy và copyright
- Chỉ sử dụng data cho research/training
- Không redistribute crawled data

## 📚 Related Files

- Training notebooks: `../train/`
- Dataset files: `*.csv` trong thư mục này
- Keywords/channels: `*.txt` files

## 🔮 Future Improvements

- [ ] Async crawling với aiohttp
- [ ] Database storage thay vì CSV
- [ ] Automatic retry với exponential backoff
- [ ] Progress tracking với tqdm
- [ ] Parallel processing
- [ ] Data validation pipeline

## 📄 License

MIT License - Chỉ sử dụng cho research/training purposes

