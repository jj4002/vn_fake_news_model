# worker/news_crawler.py
import schedule
import time
import feedparser
from sentence_transformers import SentenceTransformer
from services.supabase_client import SupabaseService
from datetime import datetime, timedelta
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class NewsCrawlerWorker:
    def __init__(self):
        self.supabase = SupabaseService()
        self.embedding_model = SentenceTransformer('keepitreal/vietnamese-sbert')
        
        # ✅ EXPANDED RSS FEEDS
        self.rss_feeds = {
            # Báo chính thống
            'VnExpress': 'https://vnexpress.net/rss/tin-moi-nhat.rss',
            'Tuổi Trẻ': 'https://tuoitre.vn/rss/tin-moi-nhat.rss',
            'Thanh Niên': 'https://thanhnien.vn/rss/home.rss',
            'Dân Trí': 'https://dantri.com.vn/rss/trang-chu.rss',
            'Việt Nam Net': 'https://vietnamnet.vn/rss/home.rss',
            
            # Chính phủ & Cơ quan nhà nước
            'Báo Chính phủ': 'https://baochinhphu.vn/rss/home.rss',
            'Cổng TTĐT Chính phủ': 'https://chinhphu.vn/rss/home.rss',
            'Công báo': 'https://congbao.chinhphu.vn/cac_van_ban_moi_ban_hanh.rss',
            'Báo Công an': 'https://congan.com.vn/rss/home.rss',
            
            # Y tế & Sức khỏe
            'Sức khỏe Đời sống': 'https://suckhoedoisong.vn/rss/home.rss',
            
            # Pháp luật
            'Pháp luật.vn': 'https://phaply.net.vn/rss/home.rss',
            
            # Kinh tế
            'Đầu tư': 'https://baodautu.vn/rss/home.rss'
        }
    
    def crawl_and_index(self):
        """
        Cào tin TỪ 24H QUA và lưu vào Supabase
        """
        logger.info("="*70)
        logger.info(f"🕐 [{datetime.now()}] Starting news crawl...")
        
        cutoff_date = datetime.now() - timedelta(hours=24)
        new_articles = []
        total_fetched = 0
        
        for source, feed_url in self.rss_feeds.items():
            try:
                logger.info(f"📡 Fetching {source}...")
                feed = feedparser.parse(feed_url)
                
                source_count = 0
                
                for entry in feed.entries:
                    try:
                        # Parse date
                        if hasattr(entry, 'published_parsed') and entry.published_parsed:
                            pub_date = datetime(*entry.published_parsed[:6])
                        else:
                            pub_date = datetime.now()
                    except:
                        pub_date = datetime.now()
                    
                    # Only articles from last 24h
                    if pub_date < cutoff_date:
                        continue
                    
                    # Get content
                    title = entry.get('title', '')
                    summary = entry.get('summary', entry.get('description', ''))
                    link = entry.get('link', '')
                    
                    if not title or not link:
                        continue
                    
                    # Generate embedding with normalization
                    text = f"{title} {summary}"
                    embedding = self.embedding_model.encode(
                        text, 
                        normalize_embeddings=True
                    )
                    
                    article = {
                        'title': title[:500],  # Limit length
                        'content': summary[:1000],
                        'source': source,
                        'url': link,
                        'published_date': pub_date.isoformat(),
                        'embedding': embedding.tolist()
                    }
                    
                    new_articles.append(article)
                    source_count += 1
                
                total_fetched += source_count
                logger.info(f"   ✅ {source}: {source_count} articles")
                
            except Exception as e:
                logger.error(f"   ❌ Error crawling {source}: {e}")
        
        # Batch insert to Supabase
        if new_articles:
            try:
                logger.info(f"💾 Inserting {len(new_articles)} articles to database...")
                
                # Insert in batches of 50
                batch_size = 50
                success_count = 0
                
                for i in range(0, len(new_articles), batch_size):
                    batch = new_articles[i:i+batch_size]
                    
                    response = self.supabase.client.table('news_corpus')\
                        .upsert(batch, on_conflict='url')\
                        .execute()
                    
                    success_count += len(batch)
                    logger.info(f"   Inserted batch {i//batch_size + 1}: {len(batch)} articles")
                
                logger.info(f"✅ Successfully indexed {success_count}/{len(new_articles)} articles")
                
            except Exception as e:
                logger.error(f"❌ Database insert error: {e}")
        else:
            logger.info("ℹ️ No new articles found in last 24 hours")
        
        logger.info("="*70)
    
    def start(self):
        """
        Start 24/7 worker
        """
        logger.info("🚀 News crawler worker started")
        logger.info(f"📰 Monitoring {len(self.rss_feeds)} news sources")
        logger.info("⏰ Schedule: Every 4 hours")
        logger.info("="*70)
        
        # Run immediately on start
        self.crawl_and_index()
        
        # Schedule: Every 4 hours
        schedule.every(4).hours.do(self.crawl_and_index)
        
        while True:
            schedule.run_pending()
            time.sleep(60)  # Check every minute

if __name__ == "__main__":
    worker = NewsCrawlerWorker()
    worker.start()
