// model-loader.js
import { loadONNXRuntime } from './ort-loader.js';
import { loadTokenizer, preprocessText } from './tokenizer.js';

let session = null;
let tokenizer = null;
let ort = null;

export async function initializeModel() {
    if (session && tokenizer) {
        console.log('Model already initialized');
        return { session, tokenizer };
    }

    try {
        // Load ONNX Runtime
        ort = await loadONNXRuntime();

        // Load model
        const modelPath = chrome.runtime.getURL('phobert_fake_news_onnx/phobert_fake_news.onnx');
        console.log('Loading model from:', modelPath);

        session = await ort.InferenceSession.create(modelPath, {
            executionProviders: ['wasm']
        });

        console.log('Model loaded successfully');

        // Load tokenizer
        tokenizer = await loadTokenizer();
        console.log('Tokenizer loaded successfully');

        return { session, tokenizer };
    } catch (error) {
        console.error('Failed to initialize model:', error);
        throw error;
    }
}

export async function classifyText(text) {
    if (!session || !tokenizer || !ort) {
        throw new Error('Model not initialized');
    }

    const cleanText = preprocessText(text);
    const tokens = tokenizer.tokenize(cleanText);

    // Convert to BigInt64Array
    const inputIdsTensor = new ort.Tensor(
        'int64',
        BigInt64Array.from(tokens.inputIds.map(id => BigInt(id))),
        [1, tokens.inputIds.length]
    );

    const attentionMaskTensor = new ort.Tensor(
        'int64',
        BigInt64Array.from(tokens.attentionMask.map(m => BigInt(m))),
        [1, tokens.attentionMask.length]
    );

    // Run inference
    const outputs = await session.run({
        input_ids: inputIdsTensor,
        attention_mask: attentionMaskTensor
    });

    const logits = outputs.logits.data;
    const exps = Array.from(logits).map(x => Math.exp(x));
    const sumExps = exps.reduce((a, b) => a + b, 0);
    const probs = exps.map(x => x / sumExps);

    const label = probs[0] > 0.85 ? "thật" : (probs[1] > 0.85 ? "giả" : "chưa_xác_nhận");
    const confidence = Math.max(...probs);

    return {
        label,
        confidence,
        probabilities: { thật: probs[0], giả: probs[1] }
    };
}
