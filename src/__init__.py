"""
Vietnamese Paraphrase Identification (VNPI)

Fine-tunes PhoBERT-base-v2 for binary sentence-pair classification,
determining whether two Vietnamese sentences convey the same meaning.

Modules:
    config      — Centralized hyperparameters and paths
    preprocess  — Data loading, cleaning, splitting, hard-neg mining
    train       — Training loop with WeightedTrainer
    evaluate    — Test evaluation, metrics export, and demo predictions
"""

__version__ = "1.0.0"
