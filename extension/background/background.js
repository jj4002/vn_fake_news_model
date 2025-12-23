// background.js
console.log('Background loaded');

// Tạo context menu cho text được bôi đen
chrome.runtime.onInstalled.addListener(() => {
    try {
        chrome.contextMenus.create({
            id: 'detect-fake-news-selected-text',
            title: 'Kiểm tin giả đoạn văn bản này (PTIT)',
            contexts: ['selection']
        });
        console.log('✅ Context menu created');
    } catch (e) {
        console.error('❌ Failed to create context menu:', e);
    }
});

// Xử lý click vào context menu
chrome.contextMenus.onClicked.addListener((info, tab) => {
    if (info.menuItemId === 'detect-fake-news-selected-text' && info.selectionText) {
        const text = info.selectionText.trim();
        if (!text) return;

        console.log('📝 Selected text from context menu:', text.slice(0, 120));

        // Lưu text vào storage để popup lấy lại
        chrome.storage.local.set({ selectedTextForCheck: text }, () => {
            console.log('✅ Saved selected text for popup');

            // Thử mở popup (hỗ trợ trên MV3 với user gesture)
            if (chrome.action && chrome.action.openPopup) {
                try {
                    chrome.action.openPopup();
                } catch (e) {
                    console.warn('⚠️ Cannot open popup programmatically:', e);
                }
            }
        });
    }
});

// Logic cũ (nếu vẫn cần dùng cho các message khác)
chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
    console.log('Background received:', request.action);

    if (request.action === 'analyzeVideo') {
        chrome.tabs.query({ active: true, currentWindow: true }, function (tabs) {
            if (tabs[0]) {
                chrome.tabs.sendMessage(
                    tabs[0].id,
                    { action: 'analyzeText', text: request.post_message },
                    function (response) {
                        if (chrome.runtime.lastError) {
                            sendResponse({
                                success: false,
                                error: 'Vui lòng refresh trang TikTok'
                            });
                        } else {
                            sendResponse(response);
                        }
                    }
                );
            } else {
                sendResponse({ success: false, error: 'No tab found' });
            }
        });
        return true;
    }
});
