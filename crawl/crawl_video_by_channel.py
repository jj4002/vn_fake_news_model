import asyncio
from playwright.async_api import async_playwright
import json
import requests
import urllib.parse
import csv
from datetime import datetime
import os



def parse_tiktok_data_to_csv(data, csv_file_path: str, is_first_write: bool = False, keyword: str = ""):
    try:
        # Tìm dữ liệu video trong response - cấu trúc itemList
        videos = []
        if 'itemList' in data:
            videos = data['itemList']
        elif 'data' in data:
            for item in data['data']:
                if item.get('type') == 1 and 'item' in item:
                    videos.append(item['item'])
        
        if not videos:
            print("⚠️  Không tìm thấy videos trong response")
            return 0  # Return số videos = 0
        
        print(f"✓ Tìm thấy {len(videos)} videos")
        
        # Chuẩn bị dữ liệu cho CSV
        csv_data = []
        for video in videos:
            try:
                author_id = video.get('author', {}).get('id', '')
                video_id = video.get('id', '')
                video_url = f"https://www.tiktok.com/@{author_id}/video/{video_id}"
                
                create_time = video.get('createTime', 0)
                create_datetime = datetime.fromtimestamp(create_time).strftime('%Y-%m-%d %H:%M:%S') if create_time else ''
                
                stats = video.get('stats', {})
                
                csv_data.append({
                    'keyword': keyword,
                    'createTime': create_datetime,
                    'url': video_url,
                    'author_id': author_id,
                    'video_id': video_id,
                    'author_nickname': video.get('author', {}).get('nickname', ''),
                    'author_unique_id': video.get('author', {}).get('uniqueId', ''),
                    'desc': video.get('desc', ''),
                    'shareCount': stats.get('shareCount', 0),
                    'commentCount': stats.get('commentCount', 0),
                    'playCount': stats.get('playCount', 0),
                    'diggCount': stats.get('diggCount', 0),
                    'collectCount': stats.get('collectCount', 0),
                    'thumnail_url': video.get('video', {}).get('cover', '')
                })
                
            except Exception as e:
                print(f"❌ Lỗi parse video: {e}")
                continue
        
        # Lưu CSV
        if csv_data:
            fieldnames = [
                'keyword', 'createTime', 'url', 'author_id', 'video_id', 'author_nickname', 
                'author_unique_id', 'desc', 'shareCount', 'commentCount', 
                'playCount', 'diggCount', 'collectCount', 'thumnail_url'
            ]
            
            mode = 'w' if is_first_write else 'a'
            with open(csv_file_path, mode, newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                if is_first_write:
                    writer.writeheader()
                writer.writerows(csv_data)
            
            print(f"💾 Đã lưu {len(csv_data)} videos vào CSV")
            return len(csv_data)
        
        return 0
            
    except Exception as e:
        print(f"❌ Lỗi parse data: {e}")
        return 0



async def open_tiktok_profile(profile_url: str):
    """Crawl TikTok profile và bắt API"""
    async with async_playwright() as p:
        # Thêm args để bypass detection
        browser = await p.chromium.launch(
            headless=False,  # ← QUAN TRỌNG: Dùng headless=False
            args=['--disable-blink-features=AutomationControlled']
        )
        
        context = await browser.new_context(
            user_agent='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            viewport={'width': 1920, 'height': 1080}
        )
        
        page = await context.new_page()
        
        # Xóa webdriver flag
        await page.add_init_script("""
            Object.defineProperty(navigator, 'webdriver', {
                get: () => undefined
            })
        """)
        
        total_videos = 0
        csv_file = 'tiktok_videos_all_keywords_real.csv'
        is_first = not os.path.exists(csv_file)
        
        # Bắt API response
        async def handle_response(response):
            nonlocal total_videos
            
            # Bắt API item_list
            if '/api/post/item_list/' in response.url:
                try:
                    print(f"🎯 Bắt được API: {response.url[:80]}...")
                    
                    data = await response.json()
                    videos_count = parse_tiktok_data_to_csv(
                        data, 
                        csv_file, 
                        is_first_write=(is_first and total_videos == 0),
                        keyword=profile_url
                    )
                    total_videos += videos_count
                    
                    print(f"📊 Tổng: {total_videos} videos")
                    
                except Exception as e:
                    print(f"⚠️  Lỗi xử lý response: {e}")
        
        page.on('response', handle_response)
        
        # Navigate
        print(f"🌐 Đang mở: {profile_url}")
        await page.goto(profile_url, wait_until='networkidle', timeout=60000)
        
        # Đợi page load
        await page.wait_for_timeout(5000)
        
        # Cuộn để trigger API
        print("📜 Bắt đầu cuộn...")
        for i in range(10):  # Cuộn nhiều hơn
            if total_videos >= 100:
                print(f"✅ Đã đủ {total_videos} videos")
                break
            
            await page.evaluate('window.scrollTo(0, document.body.scrollHeight)')
            await page.wait_for_timeout(3000)
            print(f"   Cuộn lần {i+1} - Đã có {total_videos} videos")
        
        print(f"\n=== KẾT QUẢ ===")
        print(f"✓ Tổng videos: {total_videos}")
        print(f"💾 File: {csv_file}")
        
        await browser.close()
        return total_videos



async def process_all_profiles(file_path: str):
    """Xử lý tất cả profile URLs"""
    with open(file_path, 'r', encoding='utf-8') as f:
        urls = [line.strip() for line in f if line.strip()]
    
    print(f"📋 Đã đọc {len(urls)} profiles\n")
    
    try:
        for i, url in enumerate(urls, 1):
            print(f"\n{'='*60}")
            print(f"🔄 [{i}/{len(urls)}] {url}")
            print(f"{'='*60}")
            
            try:
                await open_tiktok_profile(url)
            except Exception as e:
                print(f"❌ Lỗi: {e}")
            
            if i < len(urls):
                print(f"\n⏸️  Chờ 30s...")
                await asyncio.sleep(30)
    
    except KeyboardInterrupt:
        print(f"\n\n⚠️  Dừng bởi Ctrl+C")
        print(f"✓ Đã xử lý {i}/{len(urls)} profiles")
    
    print(f"\n{'='*60}")
    print(f"✅ HOÀN THÀNH!")
    print(f"📊 Đã xử lý {len(urls)} profiles")
    print(f"{'='*60}")



if __name__ == "__main__":
    asyncio.run(process_all_profiles("list_channel_real.txt"))
