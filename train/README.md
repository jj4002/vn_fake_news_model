# Model Training & Experiments

Thư mục chứa các script và notebook để train và experiment với nhiều kiến trúc model khác nhau cho fake news detection trên TikTok.

## 📋 Tổng quan

Dự án này bao gồm **4 experiments chính** với các approaches khác nhau:

1. **Baseline PhoBERT** (`train-baseline-phobert.py`) - Sequence Classification đơn giản
2. **PhoBERT + Author Embedding** (`train-author-embedding.py`) - Multi-modal với author information
3. **Prompt-based MLM** (`train-MLM_Prompt.py`) - Masked Language Modeling với prompts
4. **HAN + RAG** (`train-rag-han.ipynb`) - Hierarchical Attention Network với RAG (Production)

## 📁 Files

```
train/
├── train-baseline-phobert.py    # Experiment 1: Baseline PhoBERT
├── train-author-embedding.py    # Experiment 2: PhoBERT + Author Embedding
├── train-MLM_Prompt.py          # Experiment 3: Prompt-based MLM
└── train-rag-han.ipynb          # Experiment 4: HAN + RAG (Production)
```

## 🔬 Experiments Overview

### Experiment 1: Baseline PhoBERT (`train-baseline-phobert.py`)

**Mục đích:** Baseline đơn giản với PhoBERT sequence classification

**Kiến trúc:**
- **Model**: `RobertaForSequenceClassification`
- **Input**: Text only (title + content)
- **Output**: Binary classification (REAL/FAKE)

**Hyperparameters:**
- Learning rate: 2e-5
- Batch size: 16
- Epochs: 5
- Max length: 256 tokens
- Optimizer: AdamW
- Loss: CrossEntropyLoss

**Kết quả:** Baseline performance để so sánh với các models khác

---

### Experiment 2: PhoBERT + Author Embedding (`train-author-embedding.py`)

**Mục đích:** Tận dụng thông tin author để cải thiện accuracy

**Kiến trúc:**
- **Backbone**: PhoBERT-base-v2
- **Author Embedding**: Embedding layer cho từng author
- **Adaptive Gating**: Tự động học khi nào tin author, khi nào chỉ dùng text
- **Dual Branch**: 
  - Text-only branch (cho unknown authors)
  - Combined branch (text + author embedding)

**Features:**
- Author encoding với LabelEncoder
- Gating mechanism để điều chỉnh importance của author
- Weighted Focal Loss với label smoothing
- Mixed precision training (FP16)

**Hyperparameters:**
- Learning rate: 2e-5 (different rates cho từng component)
- Batch size: 16
- Epochs: 8
- Author embedding dim: 64
- Dropout: 0.3
- Focal loss: alpha=0.7, gamma=2

**Kết quả:** Cải thiện đáng kể khi có author information

---

### Experiment 3: Prompt-based MLM (`train-MLM_Prompt.py`)

**Mục đích:** Fine-tune PhoBERT với Masked Language Modeling và prompt engineering

**Kiến trúc:**
- **Model**: `AutoModelForMaskedLM` (PhoBERT MLM)
- **Prompt Format**: `"Bài viết này là <mask> . Tiêu_đề : {title} . Nội_dung : {content}"`
- **Verbalizer**: 
  - Label 0 (REAL) → token "thật"
  - Label 1 (FAKE) → token "giả"
- **Training**: Predict token tại vị trí `<mask>`

**Features:**
- Vietnamese text normalizer (không cần vinorm)
- Teencode handling
- Word segmentation với underthesea
- Class-weighted loss
- Gradient accumulation

**Hyperparameters:**
- Learning rate: 2e-5
- Batch size: 16
- Gradient accumulation: 2 steps
- Epochs: 4
- Max length: 256 tokens
- Warmup: 10% of total steps

**Kết quả:** Tận dụng pre-trained knowledge tốt hơn với prompt

---

### Experiment 4: HAN + RAG (`train-rag-han.ipynb`) ⭐ **PRODUCTION**

**Mục đích:** Hierarchical Attention Network với RAG verification (model được sử dụng trong production)

**Kiến trúc:**
- **HAN Model**: 
  - Chunk content thành segments
  - RAG-based chunk selection (top-k chunks dựa trên title similarity)
  - Hierarchical attention (chunk-level → document-level)
- **RAG Integration**:
  - Vector search trong news corpus
  - Similarity threshold: 0.75
  - Confidence adjustment dựa trên matching articles

**Features:**
- Text normalization giống training
- Semantic chunk retriever với SentenceTransformer
- ONNX export cho production
- Cache mechanism

**Hyperparameters:**
- Learning rate: 2e-5
- Batch size: 16
- Epochs: 5-10
- Max length: 256 tokens
- Chunk size: 400 chars
- Top-k chunks: 5

**Kết quả:** Best performance với RAG verification, được deploy trong production

---

## 📊 So sánh Experiments

| Experiment | Model | Input Features | Complexity | Performance | Use Case |
|------------|-------|----------------|------------|-------------|----------|
| 1. Baseline | PhoBERT SC | Text only | Low | Baseline | Quick test |
| 2. Author Embed | PhoBERT + Author | Text + Author | Medium | Good | When author info available |
| 3. Prompt MLM | PhoBERT MLM | Text + Prompt | Medium | Good | Leverage pre-trained knowledge |
| 4. HAN + RAG | HAN + RAG | Text + Chunks | High | **Best** | **Production** |

## 🚀 Training Pipeline (Chung cho tất cả experiments)

### 1. Data Preparation

**Input:**
- Dataset từ `crawl/` folder
- Format: CSV với columns `title`, `content` (hoặc `text`), `label`
- Optional: `author_id` (cho Experiment 2)

**Preprocessing:**
- Text normalization (Vietnamese)
- Word segmentation với underthesea
- Chunking content thành segments (cho HAN)
- Train/val/test split (stratified)

### 2. Training Process

**Common steps:**
1. Load và preprocess data
2. Initialize model và tokenizer
3. Create DataLoaders
4. Setup optimizer và scheduler
5. Train với validation
6. Evaluate trên test set
7. Export model (ONNX cho production)

### 3. Evaluation Metrics

- **Accuracy**: Overall correctness
- **Precision/Recall**: Per-class metrics
- **F1-score**: Weighted F1
- **Confusion Matrix**: Visual representation
- **ROC-AUC**: (Optional) Area under curve

## 📝 Usage

### Setup Environment

```bash
# Core dependencies
pip install torch transformers sentence-transformers
pip install underthesea  # Vietnamese NLP
pip install onnx onnxruntime
pip install pandas numpy scikit-learn

# Additional for specific experiments
pip install ydata_profiling  # For data profiling (train-baseline-phobert.py)
pip install optimum[onnxruntime]  # For ONNX export (train-MLM_Prompt.py)
```

### Run Experiments

#### Experiment 1: Baseline PhoBERT

```bash
python train-baseline-phobert.py
```

**Input files:**
- `combined_train.csv` - Combined training data
- `val_clean.csv` - Validation set
- `test_clean.csv` - Test set

**Output:**
- `best_phobert_fake_news.pt` - Best model weights
- `phobert_fake_news_model/` - Saved model directory

#### Experiment 2: PhoBERT + Author Embedding

```bash
python train-author-embedding.py
```

**Input files:**
- `final_train_stratified.csv` - Training với author_id
- `final_val_stratified.csv` - Validation với author_id
- `final_test_stratified.csv` - Test với author_id

**Output:**
- `phobert_for_onnx/best_model_weights.pt` - Model weights
- `phobert_for_onnx/model_config.json` - Config
- `phobert_for_onnx/author_classes.json` - Author mappings
- `phobert_fake_news.onnx` - ONNX model

#### Experiment 3: Prompt-based MLM

```bash
python train-MLM_Prompt.py
```

**Input:**
- Merged dataset với `title`, `text`, `label` columns

**Output:**
- Trained MLM model
- Evaluation metrics

#### Experiment 4: HAN + RAG (Production)

1. Mở notebook: `train-rag-han.ipynb`
2. Cấu hình paths:
   - Dataset path
   - Model save path
   - Output path
3. Chạy cells theo thứ tự

**Export to ONNX:**

```python
# Export HAN model to ONNX
torch.onnx.export(
    model,
    dummy_input,
    "han_rag_model.onnx",
    input_names=['chunk_input_ids', 'chunk_attention_masks'],
    output_names=['logits'],
    dynamic_axes={
        'chunk_input_ids': {0: 'batch_size'},
        'chunk_attention_masks': {0: 'batch_size'}
    }
)
```

## 🔧 Configuration

### Data Paths (Tùy theo experiment)

**Experiment 1:**
```python
TRAIN_CSV = "combined_train.csv"
VAL_CSV = "val_clean.csv"
TEST_CSV = "test_clean.csv"
```

**Experiment 2:**
```python
TRAIN_CSV = "final_train_stratified.csv"
VAL_CSV = "final_val_stratified.csv"
TEST_CSV = "final_test_stratified.csv"
```

**Experiment 4 (HAN):**
```python
TRAIN_CSV = "../crawl/fake_all.csv"
VAL_CSV = "../crawl/val_data.csv"
TEST_CSV = "../crawl/test_data.csv"
```

### Model Config (Chung)

```python
MODEL_NAME = "vinai/phobert-base-v2"
MAX_LENGTH = 256
NUM_LABELS = 2
```

**HAN-specific:**
```python
CHUNK_SIZE = 400
TOP_K_CHUNKS = 5
RETRIEVER_MODEL = "keepitreal/vietnamese-sbert"
```

**Author Embedding (Exp 2):**
```python
AUTHOR_EMBED_DIM = 64
DROPOUT_RATE = 0.3
```

### Training Config (Chung)

```python
BATCH_SIZE = 16
LEARNING_RATE = 2e-5
NUM_EPOCHS = 5-8  # Tùy experiment
WARMUP_RATIO = 0.1
WEIGHT_DECAY = 0.01-0.02
```

**Experiment-specific:**
- **Exp 2**: Different learning rates cho từng component
- **Exp 3**: Gradient accumulation = 2
- **Exp 4**: Chunk-based processing

## 📊 Dataset Requirements

### Format

CSV với columns:
- `title`: Video caption/title
- `content`: OCR + STT text (hoặc chỉ caption nếu không có)
- `label`: `FAKE` hoặc `REAL`

### Size Recommendations

- **Minimum**: 1000 samples mỗi class
- **Recommended**: 5000+ samples mỗi class
- **Ideal**: 10000+ samples mỗi class

### Data Balance

- Cân bằng giữa FAKE và REAL
- Nếu không cân bằng, sử dụng class weights

## 🧪 Evaluation

### Metrics

```python
# Calculate metrics
accuracy = (tp + tn) / (tp + tn + fp + fn)
precision = tp / (tp + fp)
recall = tp / (tp + fn)
f1 = 2 * (precision * recall) / (precision + recall)
```

### Validation

- Validation trên held-out set
- Early stopping nếu validation loss không giảm
- Save best model dựa trên F1-score

## 🐛 Troubleshooting

### Out of Memory

**Vấn đề:** CUDA out of memory
- **Giải pháp:**
  - Giảm batch size
  - Giảm max_length
  - Sử dụng gradient accumulation

### Training không converge

**Vấn đề:** Loss không giảm
- **Giải pháp:**
  - Check learning rate
  - Check data quality
  - Try different optimizers
  - Add warmup steps

### Overfitting

**Vấn đề:** Train accuracy cao nhưng val thấp
- **Giải pháp:**
  - Add dropout
  - Increase weight decay
  - Add more data
  - Early stopping

## 📈 Best Practices

1. **Data Quality**: Clean và validate data kỹ
2. **Cross-validation**: Sử dụng k-fold nếu dataset nhỏ
3. **Hyperparameter tuning**: Grid search hoặc random search
4. **Model checkpointing**: Save model mỗi epoch
5. **Logging**: Log metrics và losses
6. **Reproducibility**: Set random seeds

## 🔒 Model Security

- **Model validation**: Test model trên edge cases
- **Bias checking**: Check bias trên different groups
- **Adversarial testing**: Test với adversarial examples

## 🔮 Future Improvements

- [ ] Multi-task learning
- [ ] Transfer learning từ models khác
- [ ] Ensemble methods
- [ ] Hyperparameter optimization với Optuna
- [ ] Model distillation
- [ ] Quantization cho mobile deployment

## 📚 References

### Papers & Models

- **HAN**: [Hierarchical Attention Networks for Document Classification](https://www.cs.cmu.edu/~diyiy/docs/naacl16.pdf)
- **PhoBERT**: [PhoBERT: Pre-trained language models for Vietnamese](https://arxiv.org/abs/2003.00744)
- **Prompt Learning**: [GPT-3 Paper](https://arxiv.org/abs/2005.14165) (inspiration)
- **Focal Loss**: [Focal Loss for Dense Object Detection](https://arxiv.org/abs/1708.02002)

### Technical Docs

- **ONNX Export**: [PyTorch to ONNX](https://pytorch.org/tutorials/advanced/super_resolution_with_onnxruntime.html)
- **Transformers**: [HuggingFace Transformers](https://huggingface.co/docs/transformers)
- **Sentence Transformers**: [Sentence-BERT](https://www.sbert.net/)

### Datasets

- **[Vietnamese Fake News Detection](https://github.com/hiepnguyenduc2005/Vietnamese-Fake-News-Detection)**: Dataset từ ReINTEL với gần 10,000 examples được gán nhãn. Dataset này được sử dụng chính cho training baseline models và các experiments.
- **[VFND Vietnamese Fake News Datasets](https://github.com/WhySchools/VFND-vietnamese-fake-news-datasets)**: Tập hợp các bài báo tiếng Việt và Facebook posts được phân loại (228-254 bài), bao gồm cả Article Contents và Social Contents. Dataset này được sử dụng để bổ sung và đa dạng hóa training data.

## 📄 License

MIT License

