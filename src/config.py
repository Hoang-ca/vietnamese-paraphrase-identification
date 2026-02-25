"""
Centralized configuration for Vietnamese Paraphrase Identification.

All hyperparameters and paths are defined here so that every script
(`preprocess.py`, `train.py`, `evaluate.py`) shares the same settings.
"""

import os

# ── Reproducibility ──────────────────────────────────────────────
SEED = 42

# ── Backbone ─────────────────────────────────────────────────────
MODEL_NAME   = "vinai/phobert-base-v2"  # requires word segmentation
USE_WORD_SEG = MODEL_NAME.startswith("vinai/phobert")

# ── Tokenizer / Sequence ─────────────────────────────────────────
MAX_LENGTH = 256

# ── Training ─────────────────────────────────────────────────────
EPOCHS       = 10
LR           = 2e-5
TRAIN_BS     = 16        # per-GPU batch size
EVAL_BS      = 64
GRAD_ACCUM   = 2         # effective BS = TRAIN_BS × GRAD_ACCUM × n_gpu
WARMUP_RATIO = 0.1
WEIGHT_DECAY = 0.01
PATIENCE     = 3         # early-stopping patience (epochs)
DROPOUT_RATE = 0.2
LABEL_SMOOTH = 0.05

# ── Dataset flags ────────────────────────────────────────────────
VISP_SAMPLE_SIZE = 50_000
VIQP_ENABLED     = True

# ── Hard-negative mining ─────────────────────────────────────────
DO_HARD_NEG_MINING = True
HARD_NEG_PER_SENT  = 2
HARD_NEG_WEIGHT    = 0.5

# ── Paths ────────────────────────────────────────────────────────
PREP_DIR = os.getenv("VNPI_PREP_DIR", "./data/prepared")
OUT_DIR  = os.getenv("VNPI_OUT_DIR",  "./outputs/vnpi_model")

os.makedirs(PREP_DIR, exist_ok=True)
os.makedirs(OUT_DIR, exist_ok=True)
