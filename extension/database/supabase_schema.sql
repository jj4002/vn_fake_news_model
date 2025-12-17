-- ============================================
-- SETUP: Enable pgvector extension
-- ============================================
CREATE EXTENSION IF NOT EXISTS vector;

-- ============================================
-- TABLE 1: videos
-- Lưu thông tin video + prediction (video_id là PK)
-- ============================================
CREATE TABLE videos (
    video_id VARCHAR(100) PRIMARY KEY,  -- TikTok video ID làm khóa chính
    video_url TEXT NOT NULL UNIQUE,
    
    -- Input data
    caption TEXT,
    ocr_text TEXT,
    stt_text TEXT,
    author_id VARCHAR(100),
    
    -- Prediction result
    prediction VARCHAR(10) NOT NULL CHECK (prediction IN ('REAL', 'FAKE')),
    confidence FLOAT NOT NULL CHECK (confidence >= 0 AND confidence <= 1),
    method VARCHAR(20) NOT NULL CHECK (method IN ('base_model', 'rag_enhanced')),
    
    -- Timestamp
    created_at TIMESTAMP DEFAULT NOW()
);

-- Indexes
CREATE INDEX idx_videos_url ON videos(video_url);
CREATE INDEX idx_videos_prediction ON videos(prediction);
CREATE INDEX idx_videos_created ON videos(created_at DESC);

-- ============================================
-- TABLE 2: news_corpus
-- RAG vector database
-- ============================================
CREATE TABLE news_corpus (
    id BIGSERIAL PRIMARY KEY,
    
    -- Content
    title TEXT NOT NULL,
    content TEXT NOT NULL,
    url TEXT UNIQUE NOT NULL,
    
    -- Metadata
    source VARCHAR(50) NOT NULL,
    published_date TIMESTAMP NOT NULL,
    
    -- Vector embedding
    embedding vector(768) NOT NULL,
    
    created_at TIMESTAMP DEFAULT NOW()
);

-- Indexes
CREATE INDEX idx_news_embedding 
ON news_corpus 
USING hnsw (embedding vector_cosine_ops)
WITH (m = 16, ef_construction = 64);

CREATE INDEX idx_news_source ON news_corpus(source);
CREATE INDEX idx_news_published ON news_corpus(published_date DESC);

-- ============================================
-- TABLE 3: reports
-- User báo cáo prediction sai (Foreign key to videos)
-- ============================================
CREATE TABLE reports (
    id BIGSERIAL PRIMARY KEY,
    
    -- Foreign key to videos (CASCADE delete)
    video_id VARCHAR(100) NOT NULL REFERENCES videos(video_id) ON DELETE CASCADE,
    
    -- Report data
    reported_prediction VARCHAR(10) NOT NULL,  -- Model đoán gì
    reason TEXT,  -- User giải thích tại sao sai
    
    -- Status
    status VARCHAR(20) DEFAULT 'pending' CHECK (status IN ('pending', 'reviewed', 'resolved', 'rejected')),
    
    created_at TIMESTAMP DEFAULT NOW()
);

-- Indexes
CREATE INDEX idx_reports_video_id ON reports(video_id);
CREATE INDEX idx_reports_status ON reports(status);
CREATE INDEX idx_reports_created ON reports(created_at DESC);

-- Constraint: Chỉ cho phép 1 report/video/user để tránh spam
-- (Vì không có user_id, chỉ giới hạn 1 report pending per video)
CREATE UNIQUE INDEX idx_one_pending_report_per_video 
ON reports(video_id) 
WHERE status = 'pending';

-- ============================================
-- FUNCTION: Vector similarity search
-- ============================================
CREATE OR REPLACE FUNCTION search_similar_news(
    query_embedding vector(768),
    similarity_threshold float DEFAULT 0.5,
    max_results int DEFAULT 3
)
RETURNS TABLE (
    id bigint,
    title text,
    content text,
    source varchar,
    url text,
    published_date timestamp,
    similarity float
)
LANGUAGE plpgsql
AS $$
BEGIN
    RETURN QUERY
    SELECT 
        nc.id,
        nc.title,
        nc.content,
        nc.source,
        nc.url,
        nc.published_date,
        1 - (nc.embedding <=> query_embedding) as similarity
    FROM news_corpus nc
    WHERE 1 - (nc.embedding <=> query_embedding) > similarity_threshold
    ORDER BY nc.embedding <=> query_embedding
    LIMIT max_results;
END;
$$;

-- ============================================
-- VIEWS: Analytics
-- ============================================

-- Daily stats
CREATE VIEW daily_stats AS
SELECT 
    DATE(created_at) as date,
    COUNT(*) as total_predictions,
    COUNT(*) FILTER (WHERE prediction = 'FAKE') as fake_count,
    COUNT(*) FILTER (WHERE prediction = 'REAL') as real_count,
    COUNT(*) FILTER (WHERE method = 'rag_enhanced') as rag_used_count,
    AVG(confidence) as avg_confidence
FROM videos
GROUP BY DATE(created_at)
ORDER BY date DESC;

-- Most reported videos (for retraining)
CREATE VIEW disputed_videos AS
SELECT 
    v.video_id,
    v.video_url,
    v.prediction as model_prediction,
    v.confidence,
    COUNT(r.id) as report_count,
    MAX(r.created_at) as last_report_time,
    ARRAY_AGG(r.reason) as reasons
FROM videos v
INNER JOIN reports r ON v.video_id = r.video_id
WHERE r.status = 'pending'
GROUP BY v.video_id, v.video_url, v.prediction, v.confidence
ORDER BY report_count DESC;

-- ============================================
-- POLICIES: Row Level Security
-- ============================================

ALTER TABLE videos ENABLE ROW LEVEL SECURITY;
ALTER TABLE news_corpus ENABLE ROW LEVEL SECURITY;
ALTER TABLE reports ENABLE ROW LEVEL SECURITY;

-- Videos: Public read, service role write
CREATE POLICY "Public read videos" 
ON videos FOR SELECT USING (true);

CREATE POLICY "Service write videos" 
ON videos FOR ALL 
USING (auth.role() = 'service_role');

-- News: Public read, service role write
CREATE POLICY "Public read news" 
ON news_corpus FOR SELECT USING (true);

CREATE POLICY "Service write news" 
ON news_corpus FOR ALL 
USING (auth.role() = 'service_role');

-- Reports: Anyone can insert (nếu video tồn tại), service role manage
CREATE POLICY "Anyone can report" 
ON reports FOR INSERT 
WITH CHECK (
    EXISTS (SELECT 1 FROM videos WHERE videos.video_id = reports.video_id)
);

CREATE POLICY "Service manage reports" 
ON reports FOR ALL 
USING (auth.role() = 'service_role');

CREATE POLICY "Public read pending reports count"
ON reports FOR SELECT
USING (status = 'pending');

-- ============================================
-- FUNCTIONS: Helper utilities
-- ============================================

-- Check if video exists and get prediction
CREATE OR REPLACE FUNCTION get_video_prediction(vid VARCHAR(100))
RETURNS TABLE (
    video_id varchar,
    video_url text,
    prediction varchar,
    confidence float,
    method varchar,
    caption text,
    ocr_text text,
    stt_text text,
    created_at timestamp
)
LANGUAGE plpgsql
AS $$
BEGIN
    RETURN QUERY
    SELECT 
        v.video_id,
        v.video_url,
        v.prediction,
        v.confidence,
        v.method,
        v.caption,
        v.ocr_text,
        v.stt_text,
        v.created_at
    FROM videos v
    WHERE v.video_id = vid;
END;
$$;

-- Get videos pending review (có reports nhưng chưa reviewed)
CREATE OR REPLACE FUNCTION get_videos_for_review(limit_count int DEFAULT 50)
RETURNS TABLE (
    video_id varchar,
    video_url text,
    prediction varchar,
    confidence float,
    report_count bigint,
    sample_reason text
)
LANGUAGE plpgsql
AS $$
BEGIN
    RETURN QUERY
    SELECT 
        v.video_id,
        v.video_url,
        v.prediction,
        v.confidence,
        COUNT(r.id) as report_count,
        MAX(r.reason) as sample_reason
    FROM videos v
    INNER JOIN reports r ON v.video_id = r.video_id
    WHERE r.status = 'pending'
    GROUP BY v.video_id, v.video_url, v.prediction, v.confidence
    ORDER BY report_count DESC, MAX(r.created_at) DESC
    LIMIT limit_count;
END;
$$;

-- ============================================
-- TRIGGERS: Auto-calculate disputed label
-- ============================================

-- Thêm cột để track số lượng reports
ALTER TABLE videos ADD COLUMN report_count INT DEFAULT 0;

-- Trigger: Update report_count khi có report mới
CREATE OR REPLACE FUNCTION update_report_count()
RETURNS TRIGGER AS $$
BEGIN
    IF TG_OP = 'INSERT' THEN
        UPDATE videos 
        SET report_count = report_count + 1 
        WHERE video_id = NEW.video_id;
    ELSIF TG_OP = 'DELETE' THEN
        UPDATE videos 
        SET report_count = report_count - 1 
        WHERE video_id = OLD.video_id;
    END IF;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trigger_update_report_count
AFTER INSERT OR DELETE ON reports
FOR EACH ROW
EXECUTE FUNCTION update_report_count();

-- ============================================
-- MAINTENANCE: Cleanup functions
-- ============================================

-- Clean old videos (> 180 days, no reports)
CREATE OR REPLACE FUNCTION cleanup_old_videos()
RETURNS int
LANGUAGE plpgsql
AS $$
DECLARE
    deleted_count int;
BEGIN
    DELETE FROM videos
    WHERE 
        created_at < NOW() - INTERVAL '180 days'
        AND report_count = 0;
    
    GET DIAGNOSTICS deleted_count = ROW_COUNT;
    RETURN deleted_count;
END;
$$;

-- Clean old news (> 180 days)
CREATE OR REPLACE FUNCTION cleanup_old_news()
RETURNS int
LANGUAGE plpgsql
AS $$
DECLARE
    deleted_count int;
BEGIN
    DELETE FROM news_corpus
    WHERE published_date < NOW() - INTERVAL '180 days';
    
    GET DIAGNOSTICS deleted_count = ROW_COUNT;
    RETURN deleted_count;
END;
$$;

-- ============================================
-- SAMPLE QUERIES (for documentation)
-- ============================================

-- Query 1: Check if video exists
-- SELECT * FROM get_video_prediction('video_123');

-- Query 2: Search similar news
-- SELECT * FROM search_similar_news(
--     ARRAY[...]::vector(768),
--     0.5,
--     3
-- );

-- Query 3: Get videos needing review
-- SELECT * FROM get_videos_for_review(20);

-- Query 4: Get daily stats
-- SELECT * FROM daily_stats LIMIT 30;

-- Query 5: Get most disputed videos
-- SELECT * FROM disputed_videos LIMIT 10;

-- Query 6: Manual cleanup
-- SELECT cleanup_old_videos();
-- SELECT cleanup_old_news();

-- ============================================
-- INITIAL DATA (for testing)
-- ============================================

-- Test video
INSERT INTO videos (
    video_id, video_url, caption, prediction, confidence, method
) VALUES (
    'test_123',
    'https://tiktok.com/@user/video/test_123',
    'Test caption',
    'REAL',
    0.85,
    'base_model'
) ON CONFLICT (video_id) DO NOTHING;
