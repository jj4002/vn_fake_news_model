// content/content.js
console.log('✅ TikTok Fake News Detector - Content Script Loaded');

let currentVideoData = null;
let lastUrl = location.href;

// Scrape function
function scrapeTikTokData() {
    try {
        console.log('🔍 Scraping TikTok data...');
        const url = window.location.href;

        // Extract video ID
        let video_id = null;
        let m = url.match(/\/video\/(\d+)/);
        if (m) {
            video_id = m[1];
        } else {
            m = url.match(/\/photo\/(\d+)/);
            if (m) video_id = m[1];
        }

        if (!video_id) {
            console.warn('❌ No video/photo ID in URL');
            return null;
        }

        console.log('✅ Video ID:', video_id);

        let caption = '';
        let author_id = '';

        // ✅ METHOD 1 (Priority): SIGI_STATE with video_id matching
        try {
            const sigiScript = document.querySelector('script#SIGI_STATE');
            if (sigiScript) {
                const sigiData = JSON.parse(sigiScript.textContent);
                const itemModule = sigiData.ItemModule;

                // ✅ Lấy theo video_id chính xác
                if (itemModule && itemModule[video_id]) {
                    const videoData = itemModule[video_id];
                    console.log('✅ Found in SIGI_STATE (exact video_id match)');
                    return {
                        video_id,
                        video_url: url,
                        caption: videoData.desc || 'TikTok video',
                        author_id: videoData.author || 'unknown',
                    };
                }
            }
        } catch (e) {
            console.warn('SIGI_STATE failed:', e);
        }

        // ✅ METHOD 2: __UNIVERSAL_DATA_FOR_REHYDRATION__ with video_id matching
        try {
            const scripts = document.querySelectorAll('script');
            for (const script of scripts) {
                if (script.textContent.includes('__UNIVERSAL_DATA_FOR_REHYDRATION__')) {
                    const jsonMatch = script.textContent.match(
                        /window\['__UNIVERSAL_DATA_FOR_REHYDRATION__'\]\s*=\s*({.+?});/
                    );
                    if (jsonMatch) {
                        const data = JSON.parse(jsonMatch[1]);
                        const videoData = data.__DEFAULT_SCOPE__?.['webapp.video-detail']?.itemInfo?.itemStruct;

                        // ✅ Kiểm tra video_id có khớp không
                        if (videoData && videoData.id === video_id) {
                            console.log('✅ Found in __UNIVERSAL_DATA_FOR_REHYDRATION__ (exact video_id match)');
                            return {
                                video_id,
                                video_url: url,
                                caption: videoData.desc || 'TikTok video',
                                author_id: videoData.author?.uniqueId || 'unknown',
                            };
                        }
                    }
                }
            }
        } catch (e) {
            console.warn('__UNIVERSAL_DATA_FOR_REHYDRATION__ failed:', e);
        }

        // ✅ METHOD 3: DOM scraping (fallback, KHÔNG DÙNG META TAG)
        console.log('⚠️ Using DOM scraping fallback (less reliable)...');

        const captionSelectors = [
            '[data-e2e="browse-video-desc"]',
            '[data-e2e="video-desc"]',
            'h1[data-e2e="browse-video-title"]',
            // ❌ BỎ META TAG (chúng không update real-time)
            // 'meta[name="description"]',
            // 'meta[property="og:description"]',
        ];

        for (const selector of captionSelectors) {
            const el = document.querySelector(selector);
            if (el) {
                caption = el.textContent.trim();
                if (caption) {
                    console.log(`✅ Caption from DOM: ${selector}`);
                    break;
                }
            }
        }

        const authorSelectors = ['[data-e2e="browse-username"]', 'a[href*="/@"]'];
        for (const selector of authorSelectors) {
            const el = document.querySelector(selector);
            if (el) {
                author_id = el.textContent.trim().replace('@', '');
                if (author_id) break;
            }
        }
        if (!author_id) {
            const match = url.match(/@([^/]+)/);
            author_id = match ? match[1] : 'unknown';
        }

        if (!caption) caption = 'TikTok video';

        console.warn('⚠️ Fallback scrape result may be inaccurate');
        return {
            video_id,
            video_url: url,
            caption,
            author_id,
        };
    } catch (error) {
        console.error('❌ Scraping error:', error);
        return null;
    }
}


// Update cached data
function updateVideoData() {
    console.log('🔄 Updating video data...');
    currentVideoData = scrapeTikTokData();
    console.log('✅ Cached:', currentVideoData);
}

// Listen URL changes (TikTok SPA)
new MutationObserver(() => {
    if (location.href !== lastUrl) {
        lastUrl = location.href;
        console.log('🔗 URL changed:', lastUrl);
        setTimeout(updateVideoData, 800); // ← Đợi DOM render
    }
}).observe(document, { subtree: true, childList: true });

// Init
updateVideoData();

// Listen message from popup
chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
    // ✅ Ping check
    if (request.action === 'ping') {
        sendResponse({ success: true });
        return true;
    }

    if (request.action === 'getTikTokData') {
        console.log('📤 Popup requested data');

        // ✅ LUÔN SCRAPE MỚI khi popup request
        console.log('🔄 Force re-scraping for popup request...');
        console.log('   Current URL:', location.href);

        currentVideoData = scrapeTikTokData();

        console.log('✅ Fresh scraped data:');
        console.log('   video_id:', currentVideoData?.video_id);
        console.log('   caption:', currentVideoData?.caption?.substring(0, 80));
        console.log('   url:', currentVideoData?.video_url);

        sendResponse({ success: true, data: currentVideoData });
    }
    return true;
});



