// popup/popup.js
const API_BASE_URL = 'http://localhost:8000/api/v1';

console.log('🔗 API URL:', API_BASE_URL);

let currentVideoData = null;
let isAnalyzing = false; // ← THÊM FLAG ĐỂ PREVENT DOUBLE CLICK

document.addEventListener('DOMContentLoaded', function () {
    console.log('✅ Popup loaded');

    const analyzeBtn = document.getElementById('analyzeBtn');
    const resultDiv = document.getElementById('result');
    const reportBtn = document.getElementById('reportBtn');
    const reportSection = document.getElementById('reportSection');

    // Analyze button
    analyzeBtn.addEventListener('click', async () => {
        // ✅ PREVENT DOUBLE CLICK
        if (isAnalyzing) {
            console.warn('⚠️ Already analyzing, ignoring click');
            return;
        }

        // ✅ RESET STATE MỖI LẦN BẤM
        currentVideoData = null;
        resultDiv.className = 'result';
        resultDiv.innerHTML = '';
        reportSection.style.display = 'none';

        console.log('🔍 Analyze clicked');
        isAnalyzing = true;


        // Check if on TikTok
        const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });

        if (!tab.url.includes('tiktok.com')) {
            showError('❌ Vui lòng mở trang TikTok!');
            isAnalyzing = false; // ← RESET FLAG
            return;
        }

        // Show loading
        showLoading();
        analyzeBtn.disabled = true;
        analyzeBtn.textContent = '⏳ Đang xử lý...'; // ← CHANGE TEXT

        let tiktokData = null;

        try {
            console.log('📝 Getting TikTok data from content script...');

            // ✅ PING content script trước, nếu có response thì không inject
            let needsInjection = false;
            try {
                await chrome.tabs.sendMessage(tab.id, { action: 'ping' });
                console.log('✅ Content script already loaded');
            } catch (e) {
                console.log('⚠️ Content script not loaded, injecting...');
                needsInjection = true;
            }

            if (needsInjection) {
                await chrome.scripting.executeScript({
                    target: { tabId: tab.id },
                    files: ['content/content.js']
                });
                console.log('✅ Content script injected');
                await new Promise(resolve => setTimeout(resolve, 200));
            }

            const response = await chrome.tabs.sendMessage(tab.id, {
                action: 'getTikTokData'
            });

            if (!response || !response.success) {
                throw new Error('Không thể lấy dữ liệu. Vui lòng reload trang TikTok.');
            }

            tiktokData = response.data;
            console.log('✅ Scraped:', tiktokData);
            console.log('   video_id:', tiktokData.video_id);
            console.log('   caption:', tiktokData.caption);


            if (!tiktokData || !tiktokData.video_id) {
                throw new Error('Không thể lấy dữ liệu video. Vui lòng thử lại.');
            }

            currentVideoData = tiktokData;

            // Step 2: Call backend API
            console.log('📤 Calling API...');
            console.log('   video_id:', tiktokData.video_id);
            console.log('   caption:', tiktokData.caption.substring(0, 50) + '...');

            const prediction = await analyzeTikTokVideo(tiktokData);

            console.log('📥 Result:', prediction);
            console.log('   prediction:', prediction.prediction);
            console.log('   confidence:', prediction.confidence);
            console.log('   method:', prediction.method);
            console.log('   rag_used:', prediction.rag_used);

            // Step 3: Display result
            displayResult(prediction);

            // Show report button
            reportSection.style.display = 'block';

        } catch (error) {
            console.error('❌ Error:', error);

            // ✅ XỬ LÝ LỖI KẾT NỐI
            if (error.message.includes('Could not establish connection')) {
                showError('❌ Extension chưa sẵn sàng.\n\n🔄 Vui lòng reload trang TikTok (F5) rồi thử lại.');
            } else {
                showError('❌ Lỗi: ' + error.message);
            }
        } finally {
            analyzeBtn.disabled = false;
            analyzeBtn.textContent = 'Phân tích video';
            isAnalyzing = false;
        }
    });

    // Report button
    reportBtn.addEventListener('click', async () => {
        if (!currentVideoData) return;

        const reason = prompt('Tại sao bạn nghĩ kết quả này sai?\n(Tùy chọn - có thể để trống)');

        // User cancelled
        if (reason === null) return;

        try {
            reportBtn.disabled = true;
            reportBtn.textContent = '⏳ Đang gửi...';

            const response = await fetch(`${API_BASE_URL}/report`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    video_id: currentVideoData.video_id,
                    reported_prediction: result.prediction,
                    reason: reason || null
                })
            });

            if (response.ok) {
                alert('✅ Cảm ơn phản hồi của bạn!\nChúng tôi sẽ xem xét và cải thiện model.');
                reportSection.style.display = 'none';
            } else {
                const errorData = await response.json();
                console.error('Report failed:', errorData);
                alert('❌ Không thể gửi báo cáo. Vui lòng thử lại sau.');
            }

        } catch (error) {
            console.error('Report error:', error);
            alert('❌ Lỗi kết nối. Vui lòng thử lại.');
        } finally {
            reportBtn.disabled = false;
            reportBtn.textContent = '⚠️ Báo cáo kết quả sai';
        }
    });
});


// ===== API CALL =====
async function analyzeTikTokVideo(data) {
    try {
        // ✅ VALIDATE INPUT
        if (!data.caption || data.caption.trim().length === 0) {
            throw new Error('Không thể lấy caption video');
        }

        let ocr_text = '';
        let stt_text = '';

        // ✅ ENABLE MEDIA PROCESSING
        console.log('🎬 Processing media...');
        try {
            const mediaResponse = await fetch(`${API_BASE_URL}/process-media`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    video_url: data.video_url,
                    video_id: data.video_id
                })
            });

            if (mediaResponse.ok) {
                const mediaData = await mediaResponse.json();
                ocr_text = mediaData.ocr_text || '';
                stt_text = mediaData.stt_text || '';
                console.log('✅ Media processed:', {
                    ocr: ocr_text.length,
                    stt: stt_text.length
                });
            } else {
                console.warn('⚠️ Media processing failed, continuing without OCR/STT');
            }
        } catch (mediaError) {
            console.warn('⚠️ Media processing error:', mediaError);
            // Continue without media data
        }

        // Step 2: Predict
        console.log('🤖 Getting prediction...');
        const predictResponse = await fetch(`${API_BASE_URL}/predict`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                video_id: data.video_id,
                video_url: data.video_url,
                caption: data.caption,
                ocr_text: ocr_text,
                stt_text: stt_text,
                author_id: data.author_id
            })
        });

        if (!predictResponse.ok) {
            const errorData = await predictResponse.json().catch(() => ({}));
            throw new Error(errorData.detail || `API error: ${predictResponse.status}`);
        }

        const result = await predictResponse.json();
        console.log('✅ Prediction:', result);

        return result;

    } catch (error) {
        console.error('❌ Error:', error);
        throw error;
    }
}


// ===== UI FUNCTIONS =====

function showLoading() {
    const resultDiv = document.getElementById('result');
    resultDiv.className = 'result show loading';
    resultDiv.innerHTML = `
        <div class="loading-spinner"></div>
        <p>Đang phân tích video...</p>
        <p style="font-size: 11px; color: #999; margin-top: 5px;">
            Có thể mất 5-10 giây
        </p>
    `;
}

function showError(message) {
    const resultDiv = document.getElementById('result');
    resultDiv.className = 'result show error';
    resultDiv.textContent = message;
}

function displayResult(result) {
    const resultDiv = document.getElementById('result');
    resultDiv.className = 'result show';

    const prediction = result.prediction || 'UNCERTAIN';
    const confidence = result.confidence || 0;
    const method = result.method || 'base_model';
    const ragUsed = result.rag_used || false;
    const probabilities = result.probabilities || {};

    // Determine label class and emoji
    let labelClass = 'label-uncertain';
    let emoji = '❓';
    let labelText = 'KHÔNG RÕ';

    if (prediction === 'REAL') {
        labelClass = 'label-real';
        emoji = '✅';
        labelText = 'TIN THẬT';
    } else if (prediction === 'FAKE') {
        labelClass = 'label-fake';
        emoji = '⚠️';
        labelText = 'TIN GIẢ';
    }

    // Build HTML
    let html = `
        <div class="result-content">
            <div class="label ${labelClass}">
                ${emoji} ${labelText}
            </div>
            
            <div class="confidence">
                Độ tin cậy: ${Math.round(confidence * 100)}%
            </div>
            
            <div class="confidence-bar">
                <div class="confidence-fill ${prediction === 'FAKE' ? 'fake' : ''}" 
                     style="width: ${confidence * 100}%"></div>
            </div>
            
            <div class="method-badge ${ragUsed ? 'rag' : ''}" title="Phương thức: ${method}">
                ${ragUsed ? '🔍 RAG Enhanced' : '🤖 Base Model'}
            </div>
    `;

    // Add probabilities
    if (Object.keys(probabilities).length > 0 && probabilities.REAL !== 0 && probabilities.FAKE !== 0) {
        html += `<div class="probabilities">`;
        html += `<div style="font-weight: 600; margin-bottom: 5px;">Chi tiết xác suất:</div>`;

        for (const [label, prob] of Object.entries(probabilities)) {
            const displayLabel = label === 'REAL' ? 'Tin thật' : 'Tin giả';
            html += `
                <div class="prob-item">
                    <span>${displayLabel}</span>
                    <span>${Math.round(prob * 100)}%</span>
                </div>
            `;
        }
        html += `</div>`;
    }

    // Add processing time
    if (result.processing_time_ms) {
        html += `
            <div class="video-info">
                ⏱️ Thời gian xử lý: ${Math.round(result.processing_time_ms)}ms
            </div>
        `;
    }

    // Add method info (cached warning)
    if (method === 'cached') {
        html += `
            <div class="video-info" style="color: #ff9800; margin-top: 8px;">
                📦 Kết quả từ cache (đã phân tích trước đó)
            </div>
        `;
    }

    html += `</div>`;
    resultDiv.innerHTML = html;

    // Store current prediction
    currentVideoData = {
        ...(currentVideoData || {}),
        prediction: prediction,
        confidence: confidence
    };
}
