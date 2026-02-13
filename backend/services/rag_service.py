# services/rag_service.py

from typing import Optional, Dict, List, Tuple
import logging
import os
import re

logger = logging.getLogger(__name__)

try:
    from sentence_transformers import SentenceTransformer
    EMBEDDINGS_AVAILABLE = True
except ImportError:
    EMBEDDINGS_AVAILABLE = False
    logger.warning("sentence-transformers not available, RAG disabled")

from services.supabase_client import SupabaseService


# ============================================================
# HELPER CLASS 1: ContentChunker
# ============================================================

class ContentChunker:
    """Smart content chunker with sentence awareness."""

    @staticmethod
    def clean_hashtags(text: str) -> str:
        """Remove hashtags and mentions from text."""
        text = re.sub(r'#[^\s]+', '', text)  # Remove hashtags
        text = re.sub(r'@[^\s]+', '', text)  # Remove mentions
        return re.sub(r'\s+', ' ', text).strip()

    @staticmethod
    def extract_key_sentences(text: str, top_k: int = 3) -> str:
        """Extract key sentences (first + longest + last)."""
        text = ContentChunker.clean_hashtags(text)
        sentences = re.split(r"[.!?]+\s+", text.strip())
        sentences = [s.strip() for s in sentences if len(s.strip()) > 20]

        if not sentences:
            return text[:300]

        if len(sentences) <= top_k:
            return " ".join(sentences)

        key_sentences: List[str] = [sentences[0]]

        # Longest middle sentence
        if len(sentences) > 2:
            longest = max(sentences[1:-1], key=len)
        else:
            longest = sentences[-1]
        if longest not in key_sentences:
            key_sentences.append(longest)

        # Last sentence
        if sentences[-1] not in key_sentences:
            key_sentences.append(sentences[-1])

        return " ".join(key_sentences[:top_k])


# ============================================================
# HELPER CLASS 2: QueryGenerator
# ============================================================

class QueryGenerator:
    """Generate multiple search queries for better coverage."""

    def __init__(self, chunker: ContentChunker):
        self.chunker = chunker

    def generate_queries(
        self,
        title: str,
        content: str,
    ) -> List[Tuple[str, float]]:

        queries: List[Tuple[str, float]] = []

        # Clean hashtags from title
        clean_title = self.chunker.clean_hashtags(title) if title else ""

        # Clean hashtags from content
        clean_content = self.chunker.clean_hashtags(content) if content else ""

        # Query 1: Title only (weight 1.0)
        if clean_title and len(clean_title.strip()) > 10:
            queries.append((clean_title.strip(), 1.0))

        # Query 2: Title + key content (weight 1.3 - most important)
        if clean_content:
            key_sentences = self.chunker.extract_key_sentences(clean_content, top_k=2)
            combined = f"{clean_title} {key_sentences}"[:500]
            if len(combined.strip()) > 10:
                queries.append((combined, 1.3))

        # Query 3: Key content only (fallback for short titles with hashtags)
        if clean_content:
            key_sentences = self.chunker.extract_key_sentences(clean_content, top_k=3)
            key_sentences = key_sentences[:800]  # Limit to 800 chars
            if key_sentences and len(key_sentences.strip()) > 20:
                queries.append((key_sentences, 1.0))

        return queries if queries else [(clean_title or clean_content, 1.0)]


# ============================================================
# HELPER CLASS 3: AdaptiveThreshold
# ============================================================

class AdaptiveThreshold:
    """Calculate adaptive thresholds based on text length."""

    @staticmethod
    def calculate_threshold(title_length: int, content_length: int) -> Tuple[float, float]:
        """
        Calculate adaptive thresholds.

        Logic:
        - Short text (<250): Lower threshold (easier to find)
        - Long text (>1000): Higher threshold (more strict)
        - Normal text: Base threshold

        Returns:
            (search_threshold, verify_threshold)
        """
        base_search = 0.65      # Base search threshold (decreased from 0.72)
        base_verify = 0.80      # Base verify threshold (decreased from 0.87)

        total_len = title_length + content_length

        # Adjust based on length
        if total_len < 250:
            len_adj = -0.02     # Shorter text → easier to match
        elif total_len > 1000:
            len_adj = 0.02      # Longer text → stricter
        else:
            len_adj = 0.0       # Normal

        # Calculate final thresholds with bounds
        search_th = max(0.60, min(0.70, base_search + len_adj))
        verify_th = max(0.75, min(0.85, base_verify + len_adj))

        return search_th, verify_th


# ============================================================
# HELPER CLASS 4: ResultReranker
# ============================================================

class ResultReranker:
    """Re-rank results by similarity × source credibility."""

    SOURCE_SCORES = {
        "vnexpress": 1.2,
        "vtv": 1.2,
        "vov": 1.15,
        "60giay": 1.15,
        "thuvienphapluat": 1.15,
        "tuoitre": 1.1,
        "thanhnien": 1.1,
        "dantri": 1.1,
        "vietnamplus": 1.1,
        "zingnews": 1.05,
    }

    @classmethod
    def rerank(cls, results: List[Dict]) -> List[Dict]:
        """
        Re-rank by similarity × source credibility.

        Note: Uses ORIGINAL similarity, not weighted_score.
        """
        if not results:
            return []

        # Dedup by id/title, keep highest similarity
        seen: Dict[str, Dict] = {}
        for r in results:
            article_id = r.get("id") or r.get("title")
            sim = r.get("similarity", 0.0)

            if article_id in seen:
                if sim > seen[article_id]["similarity"]:
                    seen[article_id] = r
            else:
                seen[article_id] = r

        unique = list(seen.values())

        # Calculate final score = similarity × source multiplier
        for r in unique:
            sim = r.get("similarity", 0.0)
            source = (r.get("source") or "").lower()

            # Find source multiplier
            mult = 1.0
            for source_key, score in cls.SOURCE_SCORES.items():
                if source_key in source:
                    mult = score
                    break

            r["source_multiplier"] = mult
            r["final_score"] = sim * mult

        # Sort by final score
        unique.sort(key=lambda x: x.get("final_score", 0.0), reverse=True)
        return unique


# ============================================================
# MAIN CLASS: RAGService
# ============================================================

class RAGService:
    """Retrieval-Augmented Generation service for fact verification."""

    def __init__(self):
        self.supabase = SupabaseService()
        self.available = EMBEDDINGS_AVAILABLE

        if self.available:
            try:
                model_name = os.getenv(
                    "EMBEDDING_MODEL", "keepitreal/vietnamese-sbert"
                )
                self.embedding_model = SentenceTransformer(model_name)
                logger.info(f"✅ RAG enabled with {model_name}")
            except Exception as e:
                logger.error(f"Failed to load embedding model: {e}")
                self.available = False
        else:
            self.embedding_model = None
            logger.warning("RAG service disabled")

        # Helper components
        self.chunker = ContentChunker()
        self.query_generator = QueryGenerator(self.chunker)
        self.threshold_calculator = AdaptiveThreshold()
        self.reranker = ResultReranker()

    # ==================== SHOULD_USE_RAG ====================
    def should_use_rag(
        self,
        title: str,
        content: str,
        base_confidence: float,
        author_id: str = "",
    ) -> bool:
        """
        Decide whether to use RAG based on CONTENT SENSITIVITY.

        Changes:
        - ❌ REMOVED: High confidence (>0.95) trigger
        - ✅ ADDED: Arrest & legal keywords
        - ❌ REMOVED: Unknown source rule
        """
        if not self.available:
            return False

        text_lower = (title + " " + content).lower()

        # Rule 1: Clickbait patterns
        clickbait_keywords = [
            "ngay lập tức", "khẩn", "nóng", "sốc", "chấn động",
            "trước", "bị phạt", "tiền triệu", "hãy vào",
            "kiểm tra ngay", "cảnh báo", "bí mật",
        ]
        clickbait_count = sum(1 for kw in clickbait_keywords if kw in text_lower)

        if clickbait_count >= 3:
            logger.info("RAG triggered: Clickbait detected (%d keywords)", clickbait_count)
            return True

        # Rule 2: Sensitive topics (EXPANDED with arrest keywords)
        sensitive_keywords = [
            # ⭐ Arrest & Legal
            "bị bắt", "bắt giữ", "điều tra", "công an", "khởi tố",
            "tạm giam", "khám xét", "truy nã", "giam giữ", "bắt tạm giam",
            "cơ quan điều tra", "bộ công an", "công an tp", "bị tạm giữ",
            "ra tòa", "khai báo", "làm việc với công an",

            # Tài chính
            "tặng tiền", "phát tiền", "hỗ trợ tiền mặt", "nhận tiền",
            "bộ tài chính phát", "chính phủ tặng", "mỗi người dân",
            "trợ cấp", "tiền hỗ trợ", "tiền thưởng",

            # Y tế
            "virus mới", "bệnh lạ", "dịch bệnh", "bùng phát",
            "vaccine nguy hiểm", "vô sinh", "chết sau tiêm",
            "thuốc chữa khỏi", "bí quyết chữa", "phương pháp thần kỳ",
            "tay chân miệng", "covid", "cúm", "ung thư",

            # Chính trị
            "bộ trưởng", "tổng bí thư", "thủ tướng", "chủ tịch",
            "từ chức", "bê bối", "tham nhũng", "bí mật",
            "thu hồi đất", "cưỡng chế", "biểu tình",

            # Thiên tai
            "động đất", "lũ lụt", "ngập lụt", "sóng thần",
            "bão lớn", "siêu bão", "thảm họa", "sạt lở",
            "hàng trăm người chết", "hàng ngàn nạn nhân",

            # Kinh tế
            "ngân hàng sụp đổ", "phá sản", "vỡ nợ", "rút tiền",
            "bitcoin tăng", "đầu tư ngay", "làm giàu nhanh",
            "cổ phiếu sụp", "khủng hoảng tài chính",

            # Giáo dục
            "bỏ thi", "miễn học phí", "thay đổi quy định",
            "tốt nghiệp thpt", "điểm chuẩn", "xét tuyển",
            "bộ giáo dục", "cấm học sinh",

            # Xã hội
            "bắt cóc trẻ em", "xâm hại", "cướp giật", "người lạ",
            "nghi phạm", "tội phạm", "mất tích", "nạn nhân", "đánh người",

            # Chính phủ & cơ quan
            "vneid", "chính phủ", "bộ", "thông báo chính thức",
            "quy định", "luật", "bhyt", "bhxh", "cccd",
        ]

        if any(kw in text_lower for kw in sensitive_keywords):
            logger.info("RAG triggered: Sensitive topic detected")
            return True

        # Rule 3: Breaking news
        breaking_keywords = [
            "khẩn cấp", "vừa xong", "breaking", "mới nhất", "tin nóng",
            "ngay lúc này", "ngay bây giờ", "hiện tại", "đang diễn ra",
            "vừa mới", "phút trước", "giờ trước", "hôm nay",
            "cần ngay", "lập tức", "gấp", "hỏa tốc", "khẩn",
            "cảnh báo khẩn", "thông báo khẩn", "báo động",
            "tai nạn", "thảm họa", "nguy hiểm", "cập nhật mới nhất",
            "tin mới", "vừa phát hiện", "thông tin mới", "diễn biến mới",
        ]

        if any(kw in text_lower for kw in breaking_keywords):
            logger.info("RAG triggered: Breaking news")
            return True

        # Rule 4: Fallback - Always use RAG for medium confidence (DEBUG)
        # This helps ensure RAG runs even if no keywords match
        if 0.5 <= base_confidence <= 0.95:
            logger.info("RAG triggered: Medium confidence fallback (%.2f)", base_confidence)
            return True

        logger.info("RAG not triggered: Normal content")
        return False

    # ==================== VERIFY (MAIN METHOD) ====================
    def verify(
        self,
        title: str,
        content: str,
        top_k: int = 5,
    ) -> Dict:
        """
        Verify content against news corpus.

        Simplified logic:
        1. Generate multiple queries
        2. Calculate adaptive thresholds
        3. Search all queries (keep original similarity)
        4. Re-rank by similarity × source credibility
        5. Use best match for decision

        Returns:
            {
                "recommendation": "VERIFIED" | "NEEDS_REVIEW" | "NO_RELIABLE_SOURCE_FOUND",
                "similarity_score": float,
                "matching_articles": List[Dict]
            }
        """
        if not self.available:
            return self._empty_result()

        try:
            # 1. Generate queries
            queries = self.query_generator.generate_queries(title, content)
            logger.info("🔍 Generated %d RAG queries", len(queries))

            # 2. Calculate adaptive thresholds
            search_th, verify_th = self.threshold_calculator.calculate_threshold(
                len(title), len(content)
            )
            logger.info("📊 Thresholds: search=%.2f, verify=%.2f", search_th, verify_th)

            # 3. Search all queries
            all_results: List[Dict] = []

            for query_text, weight in queries:
                logger.info("   Q(w=%.1f): %s...", weight, query_text[:60])

                # Encode query
                query_embedding = self.embedding_model.encode(
                    query_text,
                    normalize_embeddings=True,
                )

                # Search in database
                results = self.supabase.search_similar_news(
                    query_embedding,
                    top_k=top_k * 2,  # Get more for dedup
                    threshold=search_th,
                )

                # Add query weight but keep original similarity
                for r in results:
                    r["query_weight"] = weight
                    r["weighted_score"] = r.get("similarity", 0.0) * weight

                all_results.extend(results)

            if not all_results:
                logger.info("❌ No matching articles found")
                return self._empty_result()

            # 4. Re-rank by similarity × source credibility
            ranked = self.reranker.rerank(all_results)
            logger.info("✅ %d articles after rerank", len(ranked))

            # 5. Use best match
            best = ranked[0]
            similarity = best.get("similarity", 0.0)  # Original similarity

            logger.info(
                "   Best: %s (sim=%.2f, mult=%.2f)",
                best.get("source"),
                similarity,
                best.get("source_multiplier", 1.0),
            )

            # 6. Make recommendation
            if similarity >= verify_th:
                recommendation = "VERIFIED"
                logger.info("   ✅ VERIFIED (≥%.2f)", verify_th)
            elif similarity >= search_th:
                recommendation = "NEEDS_REVIEW"
                logger.info("   ⚠️ NEEDS_REVIEW (%.2f ≤ sim < %.2f)", search_th, verify_th)
            else:
                recommendation = "NO_RELIABLE_SOURCE_FOUND"
                logger.info("   ❌ NO_RELIABLE_SOURCE (<%.2f)", search_th)

            return {
                "recommendation": recommendation,
                "similarity_score": similarity,
                "matching_articles": ranked[:3],
            }

        except Exception as e:
            logger.error("❌ RAG verification error: %s", e, exc_info=True)
            return self._empty_result()

    # ==================== HELPER METHODS ====================
    def _empty_result(self) -> Dict:
        """Return empty result when no articles found."""
        return {
            "recommendation": "NO_RELIABLE_SOURCE_FOUND",
            "similarity_score": 0.0,
            "matching_articles": [],
        }

    def retrieve_context(
        self,
        title: str,
        content: str,
        top_k: int = 3,
    ) -> Optional[str]:
        """
        Retrieve similar news articles as context string.

        Returns:
            Context string or None if no articles found.
        """
        verification = self.verify(title, content, top_k)

        if not verification["matching_articles"]:
            return None

        context_parts: List[str] = []
        for article in verification["matching_articles"]:
            snippet = f"[{article['source']}] {article['title']}"
            context_parts.append(snippet)

        return " | ".join(context_parts)