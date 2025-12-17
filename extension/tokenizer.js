// tokenizer.js

// Hàm tiền xử lý văn bản
export function preprocessText(text) {
    console.log('Preprocessing text:', text);
    return text
        .replace(/http\S+|www\S+|https\S+/gi, '')
        .replace(/\s+/g, ' ')
        .trim();
}

// Hàm load tokenizer từ các file JSON/Text
export async function loadTokenizer() {
    console.log('Loading tokenizer...');
    const vocabPath = chrome.runtime.getURL('phobert_fake_news_onnx/vocab.txt');
    const mergesPath = chrome.runtime.getURL('phobert_fake_news_onnx/bpe.codes'); // hoặc merges.txt
    const configPath = chrome.runtime.getURL('phobert_fake_news_onnx/tokenizer_config.json');
    const specialTokensPath = chrome.runtime.getURL('phobert_fake_news_onnx/special_tokens_map.json');

    console.log('Fetching tokenizer files...');
    const [vocabResponse, mergesResponse, configResponse, specialTokensResponse] = await Promise.all([
        fetch(vocabPath),
        fetch(mergesPath),
        fetch(configPath),
        fetch(specialTokensPath)
    ]);

    console.log('Processing tokenizer data...');
    const vocab = await vocabResponse.text();
    const merges = await mergesResponse.text();
    const config = await configResponse.json();
    const specialTokens = await specialTokensResponse.json();

    // Tạo vocab map: token -> id
    const vocabMap = {};
    const lines = vocab.split('\n');
    lines.forEach((line, index) => {
        if (line.trim()) {
            vocabMap[line.trim()] = index;
        }
    });

    // Parse merges
    const mergeRules = new Map();
    const mergeLines = merges.split('\n');
    for (const line of mergeLines) {
        if (line.trim() && !line.startsWith('#version')) {
            const [pair, _] = line.trim().split(/\s+/);
            const [left, right] = pair.split(/(?<=.)@(?=.)/); // Giả sử có @@
            mergeRules.set(`${left} ${right}`, pair);
        }
    }

    // Hàm tokenize đơn giản với BPE giả lập
    function tokenize(text) {
        // Bước 1: Tách từ (giả sử đã có word segmentation)
        let tokens = text.split(/\s+/).filter(t => t.length > 0).map(t => t + "_"); // PhoBERT thêm underscore

        // Bước 2: Áp dụng BPE
        tokens = tokens.flatMap(applyBPE);

        // Bước 3: Thêm special tokens
        tokens = [config.bos_token, ...tokens, config.eos_token];

        // Bước 4: Padding / Truncate
        const maxLength = config.model_max_length || 256;
        if (tokens.length > maxLength) {
            tokens = tokens.slice(0, maxLength);
        } else {
            while (tokens.length < maxLength) {
                tokens.push(config.pad_token);
            }
        }

        // Bước 5: Chuyển thành IDs và mask
        const inputIds = tokens.map(token => vocabMap[token] || vocabMap[config.unk_token]);
        const attentionMask = tokens.map(token => (token === config.pad_token ? 0 : 1));

        return { inputIds, attentionMask };
    }

    function applyBPE(word) {
        let pieces = word.split('');
        if (pieces.length === 1) return pieces;

        // Lặp lại quá trình ghép subword theo mergeRules
        for (let i = 0; i < 100; i++) { // Giới hạn số lần lặp
            let best = null;
            let bestScore = Infinity;

            for (let j = 0; j < pieces.length - 1; j++) {
                const pair = `${pieces[j]} ${pieces[j + 1]}`;
                if (mergeRules.has(pair)) {
                    if (pair.length < bestScore) {
                        best = { pos: j, pair };
                        bestScore = pair.length;
                    }
                }
            }

            if (best === null) break;

            const { pos, pair } = best;
            pieces = pieces.slice(0, pos).concat(mergeRules.get(pair)).concat(pieces.slice(pos + 2));
        }

        return pieces;
    }

    console.log('Tokenizer loaded successfully');
    return { tokenize };
}