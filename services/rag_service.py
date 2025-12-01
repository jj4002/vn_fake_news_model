# services/rag_service.py
from typing import Optional, Dict, List
import logging
import os

logger = logging.getLogger(__name__)

try:
    from sentence_transformers import SentenceTransformer
    EMBEDDINGS_AVAILABLE = True
except ImportError:
    EMBEDDINGS_AVAILABLE = False
    logger.warning("sentence-transformers not available, RAG disabled")

from services.supabase_client import SupabaseService  # ← CHỈ CẦN CÁI NÀY

class RAGService:
    """Retrieval-Augmented Generation service"""
    
    def __init__(self):
        self.supabase = SupabaseService()
        self.available = EMBEDDINGS_AVAILABLE
        
        if self.available:
            try:
                model_name = os.getenv("EMBEDDING_MODEL", "keepitreal/vietnamese-sbert")
                self.embedding_model = SentenceTransformer(model_name)
                logger.info(f"✅ RAG enabled with {model_name}")
            except Exception as e:
                logger.error(f"Failed to load embedding model: {e}")
                self.available = False
        else:
            self.embedding_model = None
            logger.warning("RAG service disabled")
    
    def should_use_rag(self, title: str, content: str, base_confidence: float, author_id: str = "") -> bool:
        """Decide whether to use RAG"""
        
        if not self.available:
            return False
        
        text_lower = (title + ' ' + content).lower()
        
        # Rule 1: High confidence predictions (verify)
        if base_confidence > 0.95:
            logger.info("RAG triggered: Very high confidence - needs verification")
            return True
        
        # Rule 2: Clickbait patterns
        clickbait_keywords = [
            'ngay lập tức', 'khẩn', 'nóng', 'sốc', 'chấn động',
            'trước', 'bị phạt', 'tiền triệu', 'hãy vào', 
            'kiểm tra ngay', 'cảnh báo', 'bí mật'
        ]
        clickbait_count = sum(1 for kw in clickbait_keywords if kw in text_lower)
        
        if clickbait_count >= 3:
            logger.info(f"RAG triggered: Clickbait detected ({clickbait_count} keywords)")
            return True
        
        # Rule 3: Sensitive topics prone to misinformation
        sensitive_keywords = [
            # Chính sách tài chính - "Nhà nước tặng tiền"
            'tặng tiền', 'phát tiền', 'hỗ trợ tiền mặt', 'nhận tiền',
            'bộ tài chính phát', 'chính phủ tặng', 'mỗi người dân',
            'trợ cấp', 'tiền hỗ trợ', 'tiền thưởng',
            
            # Y tế - Bệnh mới, thuốc thần kỳ, vaccine
            'virus mới', 'bệnh lạ', 'dịch bệnh', 'bùng phát',
            'vaccine nguy hiểm', 'vô sinh', 'chết sau tiêm',
            'thuốc chữa khỏi', 'bí quyết chữa', 'phương pháp thần kỳ',
            'tay chân miệng', 'covid', 'cúm', 'ung thư',
            
            # Chính trị - Phát ngôn, đảo ngược sự kiện
            'bộ trưởng', 'tổng bí thư', 'thủ tướng', 'chủ tịch',
            'từ chức', 'bê bối', 'tham nhũng', 'bí mật',
            'thu hồi đất', 'cưỡng chế', 'biểu tình',
            
            # Thiên tai - Thổi phồng
            'động đất', 'lũ lụt', 'ngập lụt', 'sóng thần',
            'bão lớn', 'siêu bão', 'thảm họa', 'sạt lở',
            'hàng trăm người chết', 'hàng ngàn nạn nhân',
            
            # Kinh tế - Ngân hàng, phá sản
            'ngân hàng sụp đổ', 'phá sản', 'vỡ nợ', 'rút tiền',
            'bitcoin tăng', 'đầu tư ngay', 'làm giàu nhanh',
            'cổ phiếu sụp', 'khủng hoảng tài chính',
            
            # Giáo dục - Thi cử, chính sách
            'bỏ thi', 'miễn học phí', 'thay đổi quy định',
            'tốt nghiệp thpt', 'điểm chuẩn', 'xét tuyển',
            'bộ giáo dục', 'cấm học sinh',
            
            # Xã hội - Tội phạm, bắt cóc
            'bắt cóc trẻ em', 'xâm hại', 'cướp giật',
            'người lạ', 'nghi phạm', 'tội phạm',
            'mất tích', 'nạn nhân', 'đánh người',
            
            # Chính phủ & Cơ quan (từ Rule 3 cũ)
            'vneid', 'chính phủ', 'bộ', 'công an',
            'thông báo chính thức', 'quy định', 'luật',
            'bhyt', 'bhxh', 'cccd'
        ]
        if any(kw in text_lower for kw in sensitive_keywords):
            logger.info("RAG triggered: Official topic - needs verification")
            return True
        
        # Rule 4: Breaking news
        breaking_keywords = [
            # Urgency - Khẩn cấp
            'khẩn cấp', 'vừa xong', 'breaking', 'mới nhất', 'tin nóng',
            
            # Time sensitivity - Thời gian
            'ngay lúc này', 'ngay bây giờ', 'hiện tại', 'đang diễn ra',
            'vừa mới', 'phút trước', 'giờ trước', 'hôm nay',
            
            # Emergency - Khẩn
            'cần ngay', 'lập tức', 'gấp', 'hỏa tốc', 'khẩn',
            
            # Alerts - Cảnh báo
            'cảnh báo khẩn', 'thông báo khẩn', 'báo động', 
            'tai nạn', 'thảm họa', 'nguy hiểm',
            
            # Updates - Cập nhật
            'cập nhật mới nhất', 'tin mới', 'vừa phát hiện',
            'thông tin mới', 'diễn biến mới'
        ]
        if any(kw in text_lower for kw in breaking_keywords):
            logger.info("RAG triggered: Breaking news")
            return True
        
        # Rule 5: Unknown source with high confidence
        trusted_sources = ['vnexpress', 'vtv', 'vov', '60giay', 'thuvienphapluat']
        is_trusted = any(source in author_id.lower() for source in trusted_sources)
        
        if not is_trusted and base_confidence > 0.85:
            logger.info("RAG triggered: Unknown source with high confidence")
            return True
        
        return False
    
    def verify_with_sources(self, title: str, content: str, top_k: int = 5) -> Dict:
        """Verify content với real news sources"""
        
        if not self.available:
            return {
                "has_reliable_source": False,
                "similarity_score": 0.0,
                "matching_articles": [],
                "recommendation": "NO_RELIABLE_SOURCE_FOUND",
            }
        
        try:
            # Generate query embedding
            query_text = f"{title} {content[:500]}"
            logger.info(f"🔍 RAG Query: {query_text[:100]}...")

            # Normalize embedding
            query_embedding = self.embedding_model.encode(
                query_text,
                normalize_embeddings=True,
            )

            # ✅ LOWER THRESHOLD TO 0.35
            results = self.supabase.search_similar_news(
                query_embedding,
                top_k=top_k,
                threshold=0.35,  # ← LOWER threshold
            )

            if not results or len(results) == 0:
                logger.warning("❌ No matching real news found (threshold: 0.35)")
                return {
                    "has_reliable_source": False,
                    "similarity_score": 0.0,
                    "matching_articles": [],
                    "recommendation": "NO_RELIABLE_SOURCE_FOUND",
                }

            # Analyze results
            best_match = results[0]
            similarity = best_match.get("similarity", 0.0)

            logger.info(f"✅ Found {len(results)} matching articles")
            logger.info(
                f"   Best match: {best_match['source']} (similarity: {similarity:.2f})"
            )
            logger.info(f"   Title: {best_match['title'][:80]}...")

            # ✅ ADJUST THRESHOLDS
            if similarity > 0.65:  # was 0.75
                recommendation = "VERIFIED_REAL"
                has_source = True
            elif similarity > 0.45:  # was 0.60
                recommendation = "NEEDS_REVIEW"
                has_source = True
            else:
                recommendation = "LOW_SIMILARITY"
                has_source = False

            return {
                "has_reliable_source": has_source,
                "similarity_score": similarity,
                "matching_articles": results[:3],
                "recommendation": recommendation,
            }

        except Exception as e:
            logger.error(f"❌ RAG verification error: {e}", exc_info=True)
            return {
                "has_reliable_source": False,
                "similarity_score": 0.0,
                "matching_articles": [],
                "recommendation": "NO_RELIABLE_SOURCE_FOUND",
            }




    
    def retrieve_context(self, title: str, content: str, top_k: int = 3) -> Optional[str]:
        """Retrieve similar news articles (legacy method)"""
        
        verification = self.verify_with_sources(title, content, top_k)
        
        if not verification['matching_articles']:
            return None
        
        # Build context string
        context_parts = []
        for article in verification['matching_articles']:
            snippet = f"[{article['source']}] {article['title']}"
            context_parts.append(snippet)
        
        return " | ".join(context_parts)
