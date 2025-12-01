# scripts/regenerate_embeddings.py
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

from sentence_transformers import SentenceTransformer
from services.supabase_client import SupabaseService
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def regenerate_embeddings():
    """Regenerate ALL embeddings with same model"""
    
    logger.info("🚀 Regenerating ALL embeddings...")
    
    # ✅ LOAD MODEL ONCE
    logger.info("Loading model...")
    model = SentenceTransformer('keepitreal/vietnamese-sbert')
    logger.info(f"✅ Model loaded: {model.max_seq_length} max_seq_length")
    
    # Connect to Supabase
    supabase = SupabaseService()
    
    # Get all articles
    logger.info("Fetching all articles...")
    response = supabase.client.table('news_corpus').select('*').execute()
    
    articles = response.data
    logger.info(f"Found {len(articles)} articles")
    
    # ✅ REGENERATE ALL (even if already have embeddings)
    success = 0
    for i, article in enumerate(articles, 1):
        try:
            text = f"{article['title']} {article['content']}"
            
            # Generate with SAME model
            embedding = model.encode(text, normalize_embeddings=True)  # ← NORMALIZE!
            
            # Update
            supabase.client.table('news_corpus').update({
                'embedding': embedding.tolist()
            }).eq('id', article['id']).execute()
            
            success += 1
            logger.info(f"✅ [{i}/{len(articles)}] {article['title'][:50]}...")
            
        except Exception as e:
            logger.error(f"❌ [{i}/{len(articles)}] Error: {e}")
    
    logger.info(f"🎉 Done! {success}/{len(articles)} embeddings regenerated")

if __name__ == "__main__":
    regenerate_embeddings()
