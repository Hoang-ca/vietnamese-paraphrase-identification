"""
Evaluation script for Vietnamese Paraphrase Identification.

Loads the trained model and runs it on the held-out test split, printing
a full classification report, confusion matrix, and saving metrics to JSON.

Usage:
    python -m src.evaluate                                  # default model dir
    python -m src.evaluate --model_dir ./my_checkpoint      # custom path
    python -m src.evaluate --hf_model vmhdaica/vnpi_model_checkpoint_3135
"""

import argparse, json, os, warnings
import numpy as np
import pandas as pd
import torch

from datasets import Dataset
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification, AutoConfig,
    Trainer, DataCollatorWithPadding,
)
from sklearn.metrics import (
    f1_score, accuracy_score, average_precision_score,
    confusion_matrix, classification_report,
)

from src.config import (
    SEED, MODEL_NAME, USE_WORD_SEG, MAX_LENGTH, EVAL_BS, PREP_DIR, OUT_DIR,
)

warnings.filterwarnings("ignore", category=FutureWarning)


# ─── Tokenization (mirrors train.py) ────────────────────────────
def tokenize_dataset(dfx: pd.DataFrame, tokenizer) -> Dataset:
    ds = Dataset.from_pandas(dfx, preserve_index=False)

    def tok(batch):
        enc = tokenizer(
            batch["sentence1"], batch["sentence2"],
            truncation=True, max_length=MAX_LENGTH, padding=False,
        )
        if "token_type_ids" in enc:
            del enc["token_type_ids"]
        enc["labels"] = [int(x) for x in batch["label"]]
        return enc

    ds = ds.map(tok, batched=True,
                remove_columns=["sentence1", "sentence2", "label",
                                "source", "sample_weight"])
    ds.set_format("torch")
    return ds


def compute_metrics(ep):
    logits, labels = ep
    probs = torch.softmax(
        torch.tensor(logits, dtype=torch.float32), dim=-1
    ).numpy()
    preds = probs.argmax(axis=-1)

    mf1 = f1_score(labels, preds, average="macro")
    f0 = f1_score(labels, preds, pos_label=0, average="binary")
    f1 = f1_score(labels, preds, pos_label=1, average="binary")
    acc = accuracy_score(labels, preds)
    try:
        prauc = average_precision_score(labels, probs[:, 1])
    except Exception:
        prauc = 0.0

    return {
        "macro_f1": round(mf1, 4),
        "f1_0": round(f0, 4),
        "f1_1": round(f1, 4),
        "accuracy": round(acc, 4),
        "pr_auc_pos": round(prauc, 4),
    }


# ─── Demo Predictions ───────────────────────────────────────────
def demo(model, tokenizer, device="cuda"):
    """Run a few demo predictions to sanity-check the model."""
    import unicodedata, re

    def norm(s):
        s = unicodedata.normalize("NFC", str(s))
        return re.sub(r"\s+", " ", s).strip()

    pairs = [
        ("Hôm nay trời mưa rất to.", "Thời tiết hôm nay mưa lớn."),
        ("Giá vàng tăng mạnh.", "Trận đấu tối qua rất hấp dẫn."),
        ("Thủ tướng đã họp với các bộ trưởng.",
         "Cuộc họp của Thủ tướng với nội các đã diễn ra."),
        ("Hà Nội là thủ đô của Việt Nam.",
         "TP.HCM là thành phố lớn nhất Việt Nam."),
    ]

    print("\n── Demo Predictions ─────────────────────────────")
    model.eval()
    for s1, s2 in pairs:
        enc = tokenizer(norm(s1), norm(s2), truncation=True,
                        max_length=MAX_LENGTH, return_tensors="pt").to(device)
        enc.pop("token_type_ids", None)
        with torch.no_grad():
            probs = torch.softmax(model(**enc).logits, dim=-1)[0].cpu().numpy()
        label = "paraphrase" if probs[1] > probs[0] else "not_paraphrase"
        print(f"\n  S1: {s1}\n  S2: {s2}")
        print(f"  → {label} ({probs[1]:.4f})")


# ─── Main ───────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Evaluate VNPI model")
    parser.add_argument("--model_dir", type=str, default=OUT_DIR,
                        help="Path to local model directory")
    parser.add_argument("--hf_model", type=str, default=None,
                        help="HuggingFace model ID (overrides --model_dir)")
    parser.add_argument("--output", type=str, default="artifacts/metrics.json",
                        help="Path to save metrics JSON")
    args = parser.parse_args()

    model_path = args.hf_model or args.model_dir
    print(f"Loading model from: {model_path}")

    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    # Load test data
    df_test = pd.read_parquet(os.path.join(PREP_DIR, "test.parquet"))
    test_ds = tokenize_dataset(df_test, tokenizer)
    print(f"Test samples: {len(test_ds)}")

    # Collator
    class SafeCollator(DataCollatorWithPadding):
        def __call__(self, features):
            for f in features:
                f.pop("token_type_ids", None)
            return super().__call__(features)

    # Predict
    trainer = Trainer(
        model=model,
        processing_class=tokenizer,
        data_collator=SafeCollator(tokenizer=tokenizer),
        compute_metrics=compute_metrics,
    )
    test_out = trainer.predict(test_ds)
    preds = test_out.predictions.argmax(axis=-1)
    lbls = test_out.label_ids

    # Results
    cm = confusion_matrix(lbls, preds)
    print("\n=== Test Metrics ===")
    for k, v in test_out.metrics.items():
        print(f"  {k}: {v}")
    print(f"\nConfusion Matrix:\n  Pred →   0     1")
    print(f"  Act 0: {cm[0][0]:>5} {cm[0][1]:>5}")
    print(f"  Act 1: {cm[1][0]:>5} {cm[1][1]:>5}")
    print(classification_report(lbls, preds,
                                target_names=["not_para", "para"]))

    # Save metrics
    metrics = dict(test_out.metrics)
    metrics["confusion_matrix"] = cm.tolist()
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics saved to {args.output}")

    # Demo
    demo(model, tokenizer, device)

    print("\n✅ Done!")


if __name__ == "__main__":
    main()
