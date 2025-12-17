import csv
import os
import whisper
import torch
from typing import List
import yt_dlp


def download_and_transcribe(video_url: str, video_id: str) -> str:
    if not video_url:
        return ""
    
    out_path = None
    try:
        os.makedirs('videos2', exist_ok=True)

        print(f"Đang tải video từ: {video_url}")

        ydl_opts = {
            'outtmpl': f'videos2/video_{video_id}.%(ext)s',
            'format': 'bestvideo[height<=720][vcodec~="h264"]/best[height<=720]/best',
            'merge_output_format': 'mp4',
            'noplaylist': True,
        }

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([video_url])

        mp4_path = f'videos2/video_{video_id}.mp4'
        if os.path.exists(mp4_path):
            out_path = mp4_path
            print(f"✓ Tìm thấy file MP4 H.264: {out_path}")
        else:
            print("❌ Không tìm thấy file MP4 H.264")
            return ""

        print(f"Đang chuyển đổi video thành text: {out_path}")

        if not os.path.exists(out_path):
            print(f"❌ File không tồn tại: {out_path}")
            return ""

        file_size = os.path.getsize(out_path)
        print(f"📁 File size: {file_size / 1024 / 1024:.1f} MB")

        device = "cpu"
        print(f"🔧 Debug: Using device: {device}")

        print("🔧 Debug: Đang tải model Whisper large-v3...")
        model = whisper.load_model("large-v3", device=device)
        print(f"🔧 Debug: Model large-v3 đã tải trên CPU")

        result = model.transcribe(
            out_path,
            language="vi",
            task="transcribe"
        )

        text = result.get("text", "").strip()
        print(f"✅ Text length: {len(text)} chars")
        return text

    except Exception as e:
        print(f"Lỗi khi download và transcribe: {e}")
        import traceback
        traceback.print_exc()
        return ""
    finally:
        if out_path and os.path.exists(out_path):
            try:
                os.remove(out_path)
                print(f"✅ Đã xóa file video: {out_path}")
            except:
                pass




def crawl_tiktok_video(video_url: str) -> str:
    """Simplified function that just returns the video URL for download"""
    print(f"Đang crawl video: {video_url}")
    return video_url

def read_urls_from_csv(csv_file_path: str) -> List[str]:
    """Đọc URLs từ file CSV"""
    urls = []
    try:
        with open(csv_file_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if 'url' in row and row['url']:
                    urls.append(row['url'])
        print(f"Đọc được {len(urls)} URLs từ file {csv_file_path}")
    except Exception as e:
        print(f"Lỗi khi đọc file CSV: {e}")
    return urls

def save_result_to_csv(url: str, text: str, output_file: str, is_first_write: bool = False):
    """Lưu một kết quả vào file CSV (append mode)"""
    try:
        # Kiểm tra file có tồn tại không
        file_exists = os.path.exists(output_file)
        
        # Chọn mode: 'w' nếu file chưa tồn tại, 'a' nếu đã tồn tại
        mode = 'w' if not file_exists or is_first_write else 'a'
        
        with open(output_file, mode, newline='', encoding='utf-8') as f:
            fieldnames = ['url', 'text']
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            
            # Chỉ ghi header khi tạo file mới
            if not file_exists or is_first_write:
                writer.writeheader()
            
            # Ghi row mới
            writer.writerow({
                'url': url,
                'text': text
            })
        
        print(f"✅ Đã lưu kết quả vào file {output_file}")
    except Exception as e:
        print(f"❌ Lỗi khi lưu file CSV: {e}")

def process_videos_from_csv(input_csv: str, output_csv: str):
    """Xử lý videos từ file CSV và lưu kết quả ngay lập tức"""
    # Đọc URLs từ file CSV
    urls = read_urls_from_csv(input_csv)
    
    if not urls:
        print("Không có URLs để xử lý")
        return
    
    # Kiểm tra xem file output đã tồn tại chưa
    is_first_write = not os.path.exists(output_csv)
    
    for i, url in enumerate(urls, 1):
        print(f"\n[{i}/{len(urls)}] Đang xử lý: {url}")
        
        try:
            # Lấy video URL (simplified)
            video_url = crawl_tiktok_video(url)
            
            if video_url:
                print(f"Video URL: {video_url}")
                
                # Tạo video_id từ URL
                video_id = f"video_{i}_{hash(url) % 10000}"
                
                # Download và transcribe
                print("Đang download và chuyển đổi video...")
                text = download_and_transcribe(video_url, video_id)
                
                if text:
                    print(f"✅ Thành công! Text: {text[:100]}...")
                    # Lưu ngay lập tức vào CSV
                    save_result_to_csv(url, text, output_csv, is_first_write)
                    is_first_write = False  # Sau lần đầu thì dùng append mode
                else:
                    print("❌ Không thể chuyển đổi video thành text")
                    # Vẫn lưu với text rỗng
                    save_result_to_csv(url, '', output_csv, is_first_write)
                    is_first_write = False
            else:
                print("❌ Không lấy được video URL")
                # Lưu với text rỗng
                save_result_to_csv(url, '', output_csv, is_first_write)
                is_first_write = False
                
        except Exception as e:
            print(f"❌ Lỗi khi xử lý video: {e}")
            # Lưu với text rỗng
            save_result_to_csv(url, '', output_csv, is_first_write)
            is_first_write = False

def main():
    """Hàm main để xử lý videos từ CSV"""
    input_csv = "fake2_1.csv"
    output_csv = "output2_1.csv"
    
    print(f"Đang xử lý videos từ file: {input_csv}")
    print(f"Kết quả sẽ được lưu vào file: {output_csv}")
    
    process_videos_from_csv(input_csv, output_csv)

if __name__ == "__main__":
    main()
