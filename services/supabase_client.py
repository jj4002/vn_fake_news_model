# services/supabase_client.py
from supabase import create_client, Client
import os
import numpy as np
from typing import Optional, List, Dict
import logging

logger = logging.getLogger(__name__)

class SupabaseService:  # ← TÊN CLASS ĐÚNG
    """Supabase database client"""
    
    def __init__(self):
        try:
            url = os.getenv("SUPABASE_URL")
            key = os.getenv("SUPABASE_KEY")
            
            if not url or not key:
                raise ValueError("SUPABASE_URL or SUPABASE_KEY not set in .env")
            
            self.client: Client = create_client(url, key)
            logger.info("✅ Supabase connected")
            
        except Exception as e:
            logger.error(f"Supabase connection failed: {e}")
            raise
    
    # ===== VIDEO CACHE =====
    
    def get_video(self, video_id: str):
        """Get cached video prediction"""
        try:
            response = self.client.table('videos')\
                .select('*')\
                .eq('video_id', video_id)\
                .execute()
            
            if response.data and len(response.data) > 0:
                logger.info(f"Cache found for video: {video_id}")
                return response.data[0]
            else:
                logger.info(f"No cache found for video: {video_id}")
                return None
                
        except Exception as e:
            logger.error(f"Error getting video cache: {e}")
            return None
    
    def save_video(self, data: Dict) -> bool:
        """Save video prediction to cache"""
        try:
            self.client.table('videos').upsert({
                'video_id': data['video_id'],
                'video_url': data['video_url'],
                'caption': data.get('caption'),
                'ocr_text': data.get('ocr_text'),
                'stt_text': data.get('stt_text'),
                'author_id': data.get('author_id'),
                'prediction': data['prediction'],
                'confidence': data['confidence'],
                'method': data['method']
            }, on_conflict='video_id').execute()
            
            logger.info(f"Saved video: {data['video_id']}")
            return True
            
        except Exception as e:
            logger.error(f"Save video error: {e}")
            return False
    
    # ===== RAG NEWS =====
    
    def search_similar_news(self, query_embedding, top_k: int = 5, threshold: float = 0.5):
        """Search for similar news"""
        try:
            import numpy as np
            import json  # ← ADD THIS
            
            # Convert query embedding
            if hasattr(query_embedding, 'tolist'):
                query_embedding = query_embedding.tolist()
            
            query_vec = np.array(query_embedding)
            
            logger.info(f"🔍 Query embedding shape: {query_vec.shape}")
            logger.info(f"🔍 Searching with threshold: {threshold}, top_k: {top_k}")
            
            # Get all records
            response = self.client.table('news_corpus')\
                .select('id, title, content, source, url, published_date, embedding')\
                .not_.is_('embedding', 'null')\
                .execute()
            
            if not response.data:
                logger.warning("❌ No records found")
                return []
            
            logger.info(f"📊 Comparing with {len(response.data)} articles...")
            
            results = []
            
            for record in response.data:
                try:
                    # ✅ HANDLE STRING/LIST/ARRAY
                    db_emb_raw = record['embedding']
                    
                    if isinstance(db_emb_raw, str):
                        # Parse JSON string
                        db_vec = np.array(json.loads(db_emb_raw), dtype=np.float32)
                    elif isinstance(db_emb_raw, list):
                        db_vec = np.array(db_emb_raw, dtype=np.float32)
                    else:
                        db_vec = np.array(db_emb_raw, dtype=np.float32)
                    
                    # Validate shape
                    if db_vec.shape != query_vec.shape:
                        logger.error(f"Shape mismatch for record {record['id']}: {db_vec.shape} vs {query_vec.shape}")
                        continue
                    
                    # Cosine similarity
                    dot_product = np.dot(query_vec, db_vec)
                    norm_query = np.linalg.norm(query_vec)
                    norm_db = np.linalg.norm(db_vec)
                    
                    if norm_query == 0 or norm_db == 0:
                        continue
                    
                    similarity = float(dot_product / (norm_query * norm_db))
                    
                    if similarity > threshold:
                        results.append({
                            'id': record['id'],
                            'title': record['title'],
                            'content': record['content'],
                            'source': record['source'],
                            'url': record.get('url', ''),
                            'published_date': record.get('published_date'),
                            'similarity': similarity
                        })
                        
                except json.JSONDecodeError as e:
                    logger.error(f"JSON decode error for record {record.get('id')}: {e}")
                    continue
                except Exception as e:
                    logger.error(f"Error processing record {record.get('id')}: {e}")
                    continue
            
            # Sort by similarity
            results.sort(key=lambda x: x['similarity'], reverse=True)
            results = results[:top_k]
            
            logger.info(f"✅ Found {len(results)} matches above threshold {threshold}")
            
            if results:
                for i, r in enumerate(results[:3], 1):
                    logger.info(f"   [{i}] {r['source']}: {r['title'][:50]}... (sim: {r['similarity']:.4f})")
            
            return results
            
        except Exception as e:
            logger.error(f"❌ Search error: {e}", exc_info=True)
            return []

    
    def add_news_articles(self, articles: List[Dict]) -> bool:
        """Batch insert news articles"""
        try:
            self.client.table('news_corpus').insert(articles).execute()
            logger.info(f"Added {len(articles)} news articles")
            return True
            
        except Exception as e:
            logger.error(f"Add news error: {e}")
            return False
    
    # ===== REPORTS =====
    
    def save_report(self, video_id: str, reported_prediction: str, 
                    reason: str = None) -> bool:
        """Save user report"""
        try:
            self.client.table('reports').insert({
                'video_id': video_id,
                'reported_prediction': reported_prediction,
                'reason': reason
            }).execute()
            
            logger.info(f"Report saved for video: {video_id}")
            return True
            
        except Exception as e:
            logger.error(f"Save report error: {e}")
            return False
    
    def get_disputed_videos(self, limit: int = 50) -> List[Dict]:
        """Get videos with reports for retraining"""
        try:
            result = self.client.rpc('get_videos_for_review', {
                'limit_count': limit
            }).execute()
            
            return result.data or []
            
        except Exception as e:
            logger.error(f"Get reports error: {e}")
            return []
