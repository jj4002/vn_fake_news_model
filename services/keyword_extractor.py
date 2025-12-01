# services/keyword_extractor.py
import re
from typing import List, Dict
import logging

logger = logging.getLogger(__name__)

class KeywordExtractor:
    """Simple keyword extraction (can be enhanced with NER later)"""
    
    def __init__(self):
        self.stopwords = set([
            'là', 'của', 'và', 'có', 'trong', 'được', 'các', 'cho',
            'đã', 'một', 'này', 'để', 'với', 'không', 'trên', 'người'
        ])
    
    def extract_keywords(self, text: str, top_n: int = 5) -> List[str]:
        """Extract top N keywords"""
        
        # Remove special characters
        text = re.sub(r'[^\w\s]', ' ', text.lower())
        
        # Split into words
        words = text.split()
        
        # Filter stopwords and short words
        keywords = [w for w in words if w not in self.stopwords and len(w) > 2]
        
        # Take unique top N
        seen = set()
        result = []
        for word in keywords:
            if word not in seen:
                seen.add(word)
                result.append(word)
                if len(result) >= top_n:
                    break
        
        return result
    
    def build_search_query(self, title: str, content: str) -> Dict:
        """Build search query info"""
        
        keywords = self.extract_keywords(title + ' ' + content, top_n=5)
        query = ' '.join(keywords[:3])  # Top 3 keywords
        
        # Check breaking news signals
        breaking_signals = ['khẩn cấp', 'nóng', 'vừa xong', 'breaking', 'cực nóng']
        text_lower = (title + ' ' + content).lower()
        is_breaking = any(signal in text_lower for signal in breaking_signals)
        
        return {
            'query': query,
            'keywords': keywords,
            'is_breaking_news': is_breaking
        }
# services/keyword_extractor.py
import re
from typing import List, Dict
import logging

logger = logging.getLogger(__name__)

class KeywordExtractor:
    """Simple keyword extraction (can be enhanced with NER later)"""
    
    def __init__(self):
        self.stopwords = set([
            'là', 'của', 'và', 'có', 'trong', 'được', 'các', 'cho',
            'đã', 'một', 'này', 'để', 'với', 'không', 'trên', 'người'
        ])
    
    def extract_keywords(self, text: str, top_n: int = 5) -> List[str]:
        """Extract top N keywords"""
        
        # Remove special characters
        text = re.sub(r'[^\w\s]', ' ', text.lower())
        
        # Split into words
        words = text.split()
        
        # Filter stopwords and short words
        keywords = [w for w in words if w not in self.stopwords and len(w) > 2]
        
        # Take unique top N
        seen = set()
        result = []
        for word in keywords:
            if word not in seen:
                seen.add(word)
                result.append(word)
                if len(result) >= top_n:
                    break
        
        return result
    
    def build_search_query(self, title: str, content: str) -> Dict:
        """Build search query info"""
        
        keywords = self.extract_keywords(title + ' ' + content, top_n=5)
        query = ' '.join(keywords[:3])  # Top 3 keywords
        
        # Check breaking news signals
        breaking_signals = ['khẩn cấp', 'nóng', 'vừa xong', 'breaking', 'cực nóng']
        text_lower = (title + ' ' + content).lower()
        is_breaking = any(signal in text_lower for signal in breaking_signals)
        
        return {
            'query': query,
            'keywords': keywords,
            'is_breaking_news': is_breaking
        }
