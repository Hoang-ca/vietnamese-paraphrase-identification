# 🇻🇳 Vietnamese Paraphrase Identification

[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org)
[![HuggingFace](https://img.shields.io/badge/🤗-HuggingFace-yellow.svg)](https://huggingface.co/vmhdaica/vnpi_model_checkpoint_3135)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/vmhdaica/vietnamese-paraphrase-identification/blob/main/notebooks/inference_demo.ipynb)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

**Binary classification of Vietnamese sentence pairs** — determines whether two Vietnamese sentences convey the same meaning, using a fine-tuned [PhoBERT-base-v2](https://github.com/VinAIResearch/PhoBERT) backbone with multi-dataset training, hard-negative mining, and leakage-aware group splitting.

---

## ✨ Key Features

- **PhoBERT-base-v2 backbone** with VnCoreNLP word segmentation — purpose-built for Vietnamese
- **Four public datasets** merged: VNPC, vnPara, ViSP, ViQP (40K+ training pairs)
- **Hard-negative mining** via TF-IDF + Nearest Neighbors to improve discrimination on similar non-paraphrase pairs
- **Leakage-free evaluation**: Union-Find grouping + StratifiedGroupKFold prevents the same sentence from appearing in both train and test
- **Class-balanced weighted loss** with per-sample weights and label smoothing
- **Early stopping** on macro-F1 to prevent overfitting

---

## 📊 Results (Held-Out Test Set — 6,702 samples)

| Metric | Score |
|--------|-------|
| **Accuracy** | **97.02%** |
| **Macro F1** | **0.876** |
| F1 (not_paraphrase) | 0.768 |
| F1 (paraphrase) | 0.984 |
| PR-AUC (positive) | 0.9995 |
| Test Loss | 0.129 |

### Confusion Matrix

|  | Pred: Not Para | Pred: Para |
|---|:---:|:---:|
| **Actual: Not Para** | 331 | 28 |
| **Actual: Para** | 172 | 6,171 |

### Classification Report

```
              precision    recall  f1-score   support
  not_para       0.66      0.92      0.77       359
      para       1.00      0.97      0.98      6343
  accuracy                           0.97      6702
```

> **Note:** All metrics are from the held-out test set using `StratifiedGroupKFold` — no data leakage. The test set reflects the natural class distribution of the merged datasets (~94.6% paraphrase, ~5.4% non-paraphrase), so accuracy is inflated by the majority class. **Macro F1 (0.876)** is the primary evaluation metric as it equally weights both classes. Training mitigates this imbalance via hard-negative mining, inverse-frequency class weights, and label smoothing.

---

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Preprocess Data (download + split)

```bash
python -m src.preprocess
```

This automatically downloads all four datasets, applies word segmentation, deduplicates, performs hard-negative mining, and saves train/val/test splits as Parquet files.

### 3. Train

```bash
python -m src.train
```

On 2× Tesla T4 (Kaggle), training takes ~30 minutes for 10 epochs with early stopping.

### 4. Evaluate

```bash
# Evaluate local model
python -m src.evaluate

# Or evaluate from HuggingFace Hub directly
python -m src.evaluate --hf_model vmhdaica/vnpi_model_checkpoint_3135
```

### 5. Quick Inference

```python
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch

model = AutoModelForSequenceClassification.from_pretrained(
    "vmhdaica/vnpi_model_checkpoint_3135"
).eval()
tokenizer = AutoTokenizer.from_pretrained(
    "vmhdaica/vnpi_model_checkpoint_3135"
)

s1 = "Hôm nay trời mưa rất to."
s2 = "Thời tiết hôm nay mưa lớn."

inputs = tokenizer(s1, s2, return_tensors="pt", truncation=True, max_length=256)
inputs.pop("token_type_ids", None)

with torch.no_grad():
    probs = torch.softmax(model(**inputs).logits, dim=-1)[0]

print(f"Paraphrase probability: {probs[1]:.4f}")
# → Paraphrase probability: 0.9902
```

### 6. Gradio Demo (interactive UI)

```bash
pip install gradio
python app.py
```

Or try it instantly on Colab: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/vmhdaica/vietnamese-paraphrase-identification/blob/main/notebooks/inference_demo.ipynb)

---

## 📁 Project Structure

```
vietnamese-paraphrase-identification/
├── README.md                          ← You are here
├── MODEL_CARD.md                      ← Model description & limitations
├── LICENSE                            ← MIT
├── app.py                             ← Gradio demo (interactive UI)
├── requirements.txt                   ← Pinned dependencies
├── .gitignore
├── src/
│   ├── config.py                      ← All hyperparameters
│   ├── preprocess.py                  ← Data pipeline (load → clean → split → mine)
│   ├── train.py                       ← Training with WeightedTrainer
│   └── evaluate.py                    ← Eval + classification report + demo
├── notebooks/
│   ├── paraphrase_identification.ipynb← Training notebook (Kaggle)
│   └── inference_demo.ipynb           ← ⚡ Click-to-run Colab demo
├── artifacts/
│   └── metrics.json                   ← Test metrics snapshot
└── scripts/
    └── download_data.md               ← Dataset sources & instructions
```

---

## 📝 Training Details

| Parameter | Value |
|-----------|-------|
| Backbone | `vinai/phobert-base-v2` |
| Max sequence length | 256 |
| Effective batch size | 64 (16 × 2 accum × 2 GPUs) |
| Learning rate | 2e-5 |
| Epochs | 10 (early stop @ patience=3) |
| Dropout | 0.2 (hidden + attention + classifier) |
| Label smoothing | 0.05 |
| Optimizer | AdamW (weight decay 0.01) |
| Hardware | 2× Tesla T4 (Kaggle) |

---

## 📚 Datasets

| Dataset | Type | Size | License |
|---------|------|------|---------|
| [VNPC](https://huggingface.co/datasets/chienpham/vnpc-train) | News paraphrase corpus | ~10K pairs | Public |
| [vnPara](https://huggingface.co/datasets/chienpham/vnpara) | General paraphrases | ~10K pairs | Public |
| [ViSP](https://github.com/ngwgsang/ViSP) | Sentence paraphrases | 50K sampled | Public |
| [ViQP](https://huggingface.co/datasets/SCM-LAB/ViQP) | Question paraphrases | All | Public |

All datasets are publicly available. The preprocessing script downloads them automatically.

---

## 🏗 Anti-Leakage Strategy

Paraphrase datasets often contain the **same sentence in multiple pairs**, creating transitive leakage between train and test. This project addresses it with:

1. **Canonical pair ordering** — `(A, B)` and `(B, A)` are treated as the same pair
2. **Conflict removal** — pairs with contradicting labels are dropped
3. **Union-Find grouping** — sentences connected by any paraphrase relationship are assigned the same group ID
4. **StratifiedGroupKFold** — all pairs sharing a group stay in the same split

---

## 🤗 Model Checkpoint

The fine-tuned model is hosted on HuggingFace Hub:

**[vmhdaica/vnpi_model_checkpoint_3135](https://huggingface.co/vmhdaica/vnpi_model_checkpoint_3135)**

---

## 📖 References

- [PhoBERT: Pre-trained language models for Vietnamese](https://github.com/VinAIResearch/PhoBERT) — VinAI Research
- [VnCoreNLP](https://github.com/vncorenlp/VnCoreNLP) — Vietnamese NLP toolkit
- Datasets: VNPC, vnPara, ViSP, ViQP (see [scripts/download_data.md](scripts/download_data.md))
