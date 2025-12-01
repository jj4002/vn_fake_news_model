# backend/scripts/generate_embeddings.py
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# ✅ LOAD .ENV BEFORE IMPORTING SERVICES
from dotenv import load_dotenv
load_dotenv()  # ← ADD THIS LINE

from sentence_transformers import SentenceTransformer
from services.supabase_client import SupabaseService
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def generate_embeddings():
    """Generate embeddings for news_corpus table"""
    
    logger.info("🚀 Starting embedding generation...")
    
    # Load model
    logger.info("Loading SentenceTransformer model...")
    model = SentenceTransformer('keepitreal/vietnamese-sbert')
    logger.info("✅ Model loaded")
    
    # Connect to Supabase
    supabase = SupabaseService()
    
    # Get all articles
    logger.info("Fetching articles from news_corpus...")
    response = supabase.client.table('news_corpus').select('*').execute()
    
    articles = response.data
    logger.info(f"Found {len(articles)} articles")
    
    if not articles:
        logger.warning("No articles found!")
        return
    
    # Process each article
    success_count = 0
    for i, article in enumerate(articles, 1):
        try:
            # Generate embedding
            text = f"{article['title']} {article['content']}"
            embedding = model.encode(text)
            
            # Convert to list
            embedding_list = embedding.tolist()
            
            # Update database
            supabase.client.table('news_corpus').update({
                'embedding': embedding_list
            }).eq('id', article['id']).execute()
            
            success_count += 1
            logger.info(f"✅ [{i}/{len(articles)}] {article['title'][:50]}...")
            
        except Exception as e:
            logger.error(f"❌ [{i}/{len(articles)}] Error: {e}")
    
    logger.info(f"🎉 Done! Successfully generated {success_count}/{len(articles)} embeddings")

if __name__ == "__main__":
    generate_embeddings()
