# Model Card: Vietnamese Paraphrase Identification

## Model Details

- **Model name:** vnpi_model_checkpoint_3135
- **Architecture:** RoBERTa-based (PhoBERT-base-v2 + classification head)
- **Parameters:** ~135M
- **Language:** Vietnamese
- **Task:** Binary sentence-pair classification (paraphrase / not_paraphrase)
- **Checkpoint:** [vmhdaica/vnpi_model_checkpoint_3135](https://huggingface.co/vmhdaica/vnpi_model_checkpoint_3135)

## Intended Use

Determining whether two Vietnamese sentences express the same meaning.

**Use cases:**
- Duplicate question detection in Vietnamese Q&A systems
- News deduplication
- Plagiarism detection for Vietnamese text
- Semantic search / retrieval augmentation

**Out of scope:**
- Non-Vietnamese text
- Semantic similarity scoring (this model outputs binary labels, not continuous scores)
- Long document comparison (max 256 tokens)

## Training Data

| Dataset | Size | Description |
|---------|------|-------------|
| VNPC | ~10K pairs | Vietnamese news paraphrase corpus |
| vnPara | ~10K pairs | General Vietnamese paraphrases |
| ViSP | 50K sampled | Sentence-level Vietnamese paraphrases |
| ViQP | All pairs | Vietnamese question paraphrases |
| **Hard-neg mined** | ~10K pairs | TF-IDF nearest-neighbor non-paraphrases |

**Total training pairs:** ~40K after dedup and conflict removal

## Training Procedure

### Preprocessing
1. Unicode NFC normalization
2. VnCoreNLP word segmentation (required by PhoBERT)
3. Canonical pair ordering + deduplication
4. Union-Find grouping → StratifiedGroupKFold splitting

### Hyperparameters

| Parameter | Value |
|-----------|-------|
| Learning rate | 2e-5 |
| Effective batch size | 64 |
| Max sequence length | 256 |
| Epochs | 10 (early stop patience=3) |
| Dropout | 0.2 |
| Label smoothing | 0.05 |
| Weight decay | 0.01 |
| Class weights | Inverse frequency balanced |

### Hardware
- 2× NVIDIA Tesla T4 (Kaggle)
- Mixed precision (FP16)

## Evaluation Results

Evaluated on a held-out test set of 6,702 samples using StratifiedGroupKFold:

| Metric | Score |
|--------|-------|
| Accuracy | 97.02% |
| Macro F1 | 0.876 |
| F1 (not_paraphrase) | 0.768 |
| F1 (paraphrase) | 0.984 |
| PR-AUC (positive) | 0.9995 |

### Confusion Matrix

|  | Pred: 0 | Pred: 1 |
|---|:---:|:---:|
| **Actual: 0** | 331 | 28 |
| **Actual: 1** | 172 | 6,171 |

## Limitations

- **Vietnamese only** — not trained on or tested with other languages
- **Max 256 tokens** — longer sentence pairs are truncated
- **Imbalanced test set** — test data is ~95% positive (paraphrase), so F1 on the negative class (0.77) is notably lower
- **Domain bias** — training data is primarily news and questions; performance may degrade on informal text (social media, chat)
- **Word segmentation dependency** — requires VnCoreNLP for inference

## Ethical Considerations

- The model should not be used as the sole basis for automated content moderation decisions
- Training data sourced from publicly available datasets — see individual dataset licenses
- The model may reflect biases present in the training data
