"""
HARD FAKE GENERATOR - GROQ + SUPABASE
3-TIER FAKE DIFFICULTY → TỐI ƯU CHO PHOBERT TRAINING
"""

from groq import Groq
from supabase import create_client, Client
import pandas as pd
import time
from tqdm import tqdm
import random
import os
import json
from dotenv import load_dotenv

# ===========================
# CONFIG
# ===========================
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

if not GROQ_API_KEY:
    raise ValueError("❌ GROQ_API_KEY not found! Get key: https://console.groq.com/keys")
if not SUPABASE_KEY:
    raise ValueError("❌ SUPABASE_KEY not found in environment!")

groq_client = Groq(api_key=GROQ_API_KEY)
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

TRUSTED_SOURCES = ["vnexpress", "tuoitre", "thanhnien", "laodong", "dantri", "thuvienphapluat"]

# ===========================
# 3-TIER FAKE TEMPLATES - TỐI ƯU PHOBERT
# ===========================

# TIER 1: EASY FAKE (30%) - Model học dễ, build confidence
FAKE_EASY = [
    """Tạo tin GIẢ DỄ PHÁT HIỆN với DẤU HIỆU RÕ:

YÊU CẦU:
- PHÓNG ĐẠI CỰC ĐỘ 10-50 lần (30 tỷ → 500 tỷ)
- Dùng từ CLICKBAIT: "NÓNG", "SỐC", "BẤT NGỜ", "CHƯA TỪNG CÓ"
- THIẾU NGUỒN TIN: "theo nguồn tin", "được biết", "tin đồn", "một số nguồn"
- Quote quá dài/chi tiết không thực tế
- Deadline không hợp lý ("áp dụng từ ngày mai", "có hiệu lực ngay")

TIN THẬT:
{article}

TIN GIẢ CLICKBAIT:""",

    """Tạo tin GIẢ với SAI LOGIC RÕ RÀNG:

YÊU CẦU:
- Phóng đại 20x+ số liệu
- Mâu thuẫn nội bộ (tiêu đề nói 100B, nội dung nói 200B)
- Thời gian phi lý (chính sách 2030 áp dụng từ 2024)
- Không có trích dẫn cụ thể từ quan chức

TIN THẬT:
{article}

TIN GIẢ SAI LOGIC:""",
]

# TIER 2: MEDIUM FAKE (50%) - Thách thức vừa phải
FAKE_MEDIUM = [
    """Tạo tin GIẢ VỪA PHẢI với DẤU HIỆU TINH VI:

YÊU CẦU:
- Phóng đại 4-7 lần
- Style chuyên nghiệp NHƯNG có 2-3 hint nhỏ:
  + Thiếu nguồn tin chính thống (không nêu tên Bộ/Sở cụ thể)
  + Dùng "một số chuyên gia", "theo đánh giá" (mơ hồ)
  + Quote hơi quá chi tiết/hoàn hảo
- Thêm xếp hạng quốc tế KHÔNG XÁC MINH: "Top 10 châu Á theo Forbes"
- Timeline hơi vội (đề xuất 2026 → áp dụng 2025)

TIN THẬT:
{article}

TIN GIẢ MEDIUM:""",

    """Tạo tin GIẢ BÓP MÉO chính sách VỪA PHẢI:

YÊU CẦU:
- Đổi GẦN HẾT ý chính sách (không chỉ 1 từ):
  + "Thí điểm 3 tỉnh" → "Áp dụng toàn quốc bắt buộc"
  + "Đề xuất giảm 10%" → "Chính thức giảm 30%"
- Giữ style nghiêm túc nhưng thêm:
  + Ngày tháng cụ thể quá (không có trong tin gốc)
  + Phạt/Quyền lợi không được nhắc trong tin thật

TIN THẬT:
{article}

TIN GIẢ POLICY TWIST:""",

    """Tạo tin GIẢ "NGUỒN QUỐC TẾ" với DẤU HIỆU:

YÊU CẦU:
- Mở đầu: "Theo Forbes/Bloomberg/Reuters..."
- Phóng đại 5-8x về tầm quan trọng
- Thêm bảng xếp hạng KHÔNG CÓ THẬT
- Style quốc tế NHƯNG:
  + Không có link nguồn cụ thể
  + Không nêu tên tác giả/ngày đăng
  + Dịch thuật hơi stiff (dấu hiệu dịch máy)

TIN THẬT:
{article}

TIN GIẢ INTERNATIONAL:""",
]

# TIER 3: HARD FAKE (20%) - Tinh vi, thử thách model
FAKE_HARD = [
    """Tạo tin GIẢ CỰC KỲ TINH VI, gần như KHÔNG THỂ PHÂN BIỆT:

YÊU CẦU:
- Phóng đại CHỈ 2-3 lần (đủ sai nhưng nghe hợp lý)
- Style 100% GIỐNG báo lớn: VnExpress, Tuổi Trẻ
- Giữ cấu trúc: Lead → Body → Quote → Kết
- Sai CHÍNH XÁC 1 ĐIỂM quan trọng:
  + SỐ LIỆU (30 tỷ → 90 tỷ)
  + THỜI GIAN (2026 → 2025)
  + PHẠM VI (3 tỉnh → toàn quốc)
- KHÔNG được clickbait
- KHÔNG được thiếu nguồn tin (phải có tên Bộ/cơ quan)

Mục tiêu: CHỈ FACT-CHECK KỸ MỚI PHÁT HIỆN được!

TIN THẬT:
{article}

TIN GIẢ SIÊU TINH VI:""",

    """Tạo tin GIẢ "DƯƠNG ĐÔNG KÍCH TÂY" cực tinh vi:

YÊU CẦU:
- Giữ 80% nội dung ĐÚNG
- THAY ĐỔI TRỌNG TÂM một cách TINH TẾ:
  + Ý kiến chuyên gia → Quyết định chính thức
  + Kiến nghị → Chính sách đã thông qua
  + "Đang nghiên cứu" → "Sẽ áp dụng"
- Tiêu đề lái hiểu nhầm nhưng CÓ THỂ GIẢI THÍCH được
- Style 100% báo lớn, KHÔNG CÓ dấu hiệu rõ ràng

TIN THẬT:
{article}

TIN GIẢ MISLEADING:""",
]

# ===========================
# WEIGHTED SAMPLING
# ===========================
def get_random_template():
    """Random template theo distribution tối ưu PhoBERT"""
    rand = random.random()
    
    if rand < 0.3:  # 30% easy
        return random.choice(FAKE_EASY)
    elif rand < 0.8:  # 50% medium
        return random.choice(FAKE_MEDIUM)
    else:  # 20% hard
        return random.choice(FAKE_HARD)

# ===========================
# FUNCTIONS
# ===========================
def get_real_articles(limit=500):
    """Lấy real news từ Supabase"""
    print("📡 Fetching real articles from Supabase...")
    
    response = supabase.table("news_corpus").select(
        "title, content, source"
    ).in_(
        "source", TRUSTED_SOURCES
    ).gte(
        "published_date", "2024-01-01"
    ).not_.is_(
        "embedding", "null"
    ).limit(limit).execute()
    
    articles = []
    for row in response.data:
        title = (row.get('title') or "").strip()
        content = (row.get('content') or "").strip()
        
        if len(title + content) > 200:
            articles.append({
                "title": title,
                "content": content[:2000],
                "source": row.get('source')
            })
    
    print(f"✅ Found {len(articles)} real articles")
    return articles

def generate_fake_groq(article: dict, template: str) -> dict:
    """Tạo 1 fake bằng GROQ"""
    try:
        prompt = template.format(
            article=f"Tiêu đề: {article['title']}\n\nNội dung: {article['content'][:1500]}"
        )
        
        response = groq_client.chat.completions.create(
            # model="llama-3.3-70b-versatile",  # 500 cái đầu dùng cái này
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": "Bạn là chuyên gia tạo tin giả. Tạo CHÍNH XÁC theo yêu cầu, đừng thêm giải thích."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=800,
            temperature=0.8  # ✅ Tăng lên 0.8 để đa dạng hơn
        )
        
        fake_text = response.choices[0].message.content.strip()
        
        # Parse title/content
        if "\n\n" in fake_text:
            parts = fake_text.split("\n\n", 1)
            title = parts[0].replace("Tiêu đề:", "").replace("**", "").strip()[:150]
            content = parts[1].replace("Nội dung:", "").strip()[:1800]
        else:
            lines = fake_text.split('\n')
            title = lines[0].strip()[:150]
            content = '\n'.join(lines[1:]).strip()[:1800]
        
        return {
            "title": title,
            "content": content,
            "label": "FAKE"
        }
        
    except Exception as e:
        print(f"❌ GROQ error: {e}")
        return None

def generate_dataset(num_fakes=5000):
    """Main - Generate dataset with 3-tier difficulty"""
    print("🔥 HARD FAKE GENERATOR - 3-TIER DIFFICULTY!\n")
    print("📊 Distribution: 30% Easy, 50% Medium, 20% Hard")
    print("⚡ FAST MODE: 2s/fake (30 req/min)")
    print(f"⏱️  Estimated time: {num_fakes * 2 / 3600:.1f} hours\n")
    
    checkpoint_file = "checkpoint.json"
    
    if os.path.exists(checkpoint_file):
        print("📂 Found checkpoint, resuming...")
        with open(checkpoint_file, 'r', encoding='utf-8') as f:
            dataset = json.load(f)
        print(f"   Loaded {len(dataset)} articles from checkpoint\n")
    else:
        real_articles = get_real_articles(limit=500)
        if len(real_articles) == 0:
            print("❌ No real articles found!")
            return
        
        dataset = []
        print("\n📋 Adding REAL articles...")
        for article in real_articles:
            dataset.append({
                "title": article["title"],
                "content": article["content"],
                "label": "REAL"
            })
        
        with open("real_articles.json", 'w', encoding='utf-8') as f:
            json.dump(real_articles, f, ensure_ascii=False)
    
    if not os.path.exists("real_articles.json"):
        print("❌ real_articles.json not found!")
        return
        
    with open("real_articles.json", 'r', encoding='utf-8') as f:
        real_articles = json.load(f)
    
    fake_count = sum(1 for d in dataset if d['label'] == 'FAKE')
    
    print(f"\n🎯 Generating {num_fakes - fake_count} more hard fakes...")
    print(f"   Current: {fake_count}/{num_fakes} fakes\n")
    
    pbar = tqdm(initial=fake_count, total=num_fakes, desc="Generating")
    
    # ✅ Track distribution
    easy_count = medium_count = hard_count = 0
    
    while fake_count < num_fakes:
        article = random.choice(real_articles)
        template = get_random_template()  # ✅ 3-tier sampling
        
        # Track difficulty
        if template in FAKE_EASY:
            easy_count += 1
        elif template in FAKE_MEDIUM:
            medium_count += 1
        else:
            hard_count += 1
        
        fake = generate_fake_groq(article, template)
        
        if fake and len(fake["title"]) > 20 and len(fake["content"]) > 100:
            dataset.append(fake)
            fake_count += 1
            pbar.update(1)
            
            if fake_count % 100 == 0:
                with open(checkpoint_file, 'w', encoding='utf-8') as f:
                    json.dump(dataset, f, ensure_ascii=False)
                print(f"\n💾 Checkpoint: {fake_count} | Easy:{easy_count} Med:{medium_count} Hard:{hard_count}")
        
        time.sleep(2.0)
    
    pbar.close()
    
    # Stats
    real_count = sum(1 for d in dataset if d['label'] == 'REAL')
    fake_count = len(dataset) - real_count
    
    print(f"\n📊 DATASET STATS:")
    print(f"   Total:  {len(dataset):,}")
    print(f"   REAL:   {real_count:,} ({real_count/len(dataset)*100:.1f}%)")
    print(f"   FAKE:   {fake_count:,} ({fake_count/len(dataset)*100:.1f}%)")
    print(f"\n📊 FAKE DISTRIBUTION:")
    print(f"   Easy:   {easy_count} (~{easy_count/fake_count*100:.0f}%)")
    print(f"   Medium: {medium_count} (~{medium_count/fake_count*100:.0f}%)")
    print(f"   Hard:   {hard_count} (~{hard_count/fake_count*100:.0f}%)")
    
    random.shuffle(dataset)
    
    print("\n💾 Saving final CSV...")
    df = pd.DataFrame(dataset)
    df.to_csv("fake_news_dataset.csv", index=False, encoding='utf-8-sig')
    print(f"   ✅ fake_news_dataset.csv ({len(df):,} rows)")
    
    if os.path.exists(checkpoint_file):
        os.remove(checkpoint_file)
    if os.path.exists("real_articles.json"):
        os.remove("real_articles.json")
    
    print("\n🎉 DONE! 3-TIER DATASET FOR PHOBERT!")
    print("\n✅ PhoBERT sẽ học:")
    print("   - Easy cases: Build confidence, learn obvious patterns")
    print("   - Medium cases: Learn subtle hints & context")
    print("   - Hard cases: Deep fact-checking, số liệu chính xác")

if __name__ == "__main__":
    generate_dataset(5000)
