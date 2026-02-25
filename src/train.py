"""
Training script for Vietnamese Paraphrase Identification.

Fine-tunes PhoBERT-base-v2 (or XLM-R) on the preprocessed data produced
by `preprocess.py`.  Uses a custom WeightedTrainer that supports:
  • Per-sample weights  (for hard-negative mining)
  • Class-balanced loss
  • Label smoothing
  • Early stopping on macro-F1

Usage:
    python -m src.train          # from repo root
    python src/train.py          # or directly
"""

import os, json, warnings, gc
import numpy as np
import pandas as pd
import torch

from datasets import Dataset
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification, AutoConfig,
    TrainingArguments, Trainer, EarlyStoppingCallback, DataCollatorWithPadding,
)
from sklearn.metrics import (
    f1_score, accuracy_score, average_precision_score,
)

from src.config import (
    SEED, MODEL_NAME, USE_WORD_SEG, MAX_LENGTH,
    EPOCHS, LR, TRAIN_BS, EVAL_BS, GRAD_ACCUM,
    WARMUP_RATIO, WEIGHT_DECAY, PATIENCE, DROPOUT_RATE, LABEL_SMOOTH,
    PREP_DIR, OUT_DIR,
)

warnings.filterwarnings("ignore", category=FutureWarning)

import random
random.seed(SEED); np.random.seed(SEED)
torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)


# ─── Tokenization ───────────────────────────────────────────────
def tokenize_dataset(dfx: pd.DataFrame, tokenizer) -> Dataset:
    ds = Dataset.from_pandas(dfx, preserve_index=False)

    def tok(batch):
        enc = tokenizer(
            batch["sentence1"], batch["sentence2"],
            truncation=True, max_length=MAX_LENGTH, padding=False,
        )
        # PhoBERT / RoBERTa has type_vocab_size=1 → remove token_type_ids
        if "token_type_ids" in enc:
            del enc["token_type_ids"]
        enc["labels"] = [int(x) for x in batch["label"]]
        enc["sample_weight"] = [float(x) for x in batch["sample_weight"]]
        return enc

    ds = ds.map(tok, batched=True,
                remove_columns=["sentence1", "sentence2", "label",
                                "source", "sample_weight"])
    ds.set_format("torch")
    return ds


# ─── Collator ────────────────────────────────────────────────────
class WeightedPaddingCollator(DataCollatorWithPadding):
    """Pads inputs and carries through sample weights."""

    def __call__(self, features):
        sw = (
            [f.pop("sample_weight") for f in features]
            if "sample_weight" in features[0] else None
        )
        for f in features:
            f.pop("token_type_ids", None)
        batch = super().__call__(features)
        if sw is not None:
            batch["sample_weight"] = torch.tensor(sw, dtype=torch.float32)
        return batch


# ─── Trainer ─────────────────────────────────────────────────────
class WeightedTrainer(Trainer):
    """Trainer with class-balanced weighted cross-entropy loss."""

    def __init__(self, *args, class_weights=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        sw = inputs.pop("sample_weight", None)
        inputs.pop("token_type_ids", None)  # safety

        out = model(**inputs)
        logits = out.logits

        cw = (self.class_weights.to(logits.device)
              if self.class_weights is not None else None)
        loss = torch.nn.CrossEntropyLoss(weight=cw, reduction="none")(
            logits, labels)

        if sw is not None:
            loss = loss * sw.to(loss.device)
        loss = loss.mean()

        return (loss, out) if return_outputs else loss


# ─── Metrics ─────────────────────────────────────────────────────
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


# ─── Main ───────────────────────────────────────────────────────
def main():
    # Load preprocessed splits
    df_train = pd.read_parquet(os.path.join(PREP_DIR, "train.parquet"))
    df_val   = pd.read_parquet(os.path.join(PREP_DIR, "val.parquet"))

    # Class weights (inverse frequency)
    n0 = (df_train["label"] == 0).sum()
    n1 = (df_train["label"] == 1).sum()
    w0 = len(df_train) / (2.0 * max(1, n0))
    w1 = len(df_train) / (2.0 * max(1, n1))
    class_weights = torch.tensor([w0, w1], dtype=torch.float32)
    print(f"Class weights: 0={w0:.3f}, 1={w1:.3f}")

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)

    train_ds = tokenize_dataset(df_train, tokenizer)
    val_ds   = tokenize_dataset(df_val, tokenizer)
    print(f"Tokenized: train={len(train_ds)}, val={len(val_ds)}")

    # Model
    config = AutoConfig.from_pretrained(MODEL_NAME, num_labels=2)
    config.hidden_dropout_prob = DROPOUT_RATE
    config.attention_probs_dropout_prob = DROPOUT_RATE
    config.classifier_dropout = DROPOUT_RATE
    config.id2label = {0: "not_paraphrase", 1: "paraphrase"}
    config.label2id = {"not_paraphrase": 0, "paraphrase": 1}

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, config=config)
    model.resize_token_embeddings(len(tokenizer))
    print(f"Vocab size: tokenizer={len(tokenizer)}, model={model.config.vocab_size}")

    # Training arguments
    training_args = TrainingArguments(
        output_dir=OUT_DIR,
        learning_rate=LR,
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=TRAIN_BS,
        per_device_eval_batch_size=EVAL_BS,
        gradient_accumulation_steps=GRAD_ACCUM,
        warmup_ratio=WARMUP_RATIO,
        weight_decay=WEIGHT_DECAY,
        fp16=True,
        logging_steps=100,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        save_total_limit=2,
        report_to="none",
        dataloader_num_workers=2,
        dataloader_pin_memory=True,
        seed=SEED,
        label_smoothing_factor=LABEL_SMOOTH,
    )

    trainer = WeightedTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        processing_class=tokenizer,
        data_collator=WeightedPaddingCollator(tokenizer=tokenizer),
        compute_metrics=compute_metrics,
        class_weights=class_weights,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=PATIENCE)],
    )

    n_gpu = torch.cuda.device_count()
    print(f"\nTraining: {MODEL_NAME}")
    print(f"  Epochs={EPOCHS}, EffBS={TRAIN_BS * GRAD_ACCUM * max(1, n_gpu)}, "
          f"LR={LR}, Dropout={DROPOUT_RATE}")

    trainer.train()

    # Save best model
    trainer.save_model(OUT_DIR)
    tokenizer.save_pretrained(OUT_DIR)
    print(f"\nModel saved to {OUT_DIR} ✓")


if __name__ == "__main__":
    main()
