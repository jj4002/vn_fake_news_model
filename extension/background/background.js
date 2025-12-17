// background.js
console.log('Background loaded');

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
