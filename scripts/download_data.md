# Downloading the Datasets

This project uses **four public Vietnamese paraphrase datasets**.
No manual download is needed — the scripts pull them automatically.

## Automatic (recommended)

Run the preprocessing script and everything is fetched for you:

```bash
python -m src.preprocess
```

## Manual Sources

| Dataset | Source | How it's loaded |
|---------|--------|-----------------|
| **VNPC** | [chienpham/vnpc-train](https://huggingface.co/datasets/chienpham/vnpc-train) | `datasets.load_dataset(...)` |
| **vnPara** | [chienpham/vnpara](https://huggingface.co/datasets/chienpham/vnpara) | `datasets.load_dataset(...)` |
| **ViSP** | [github.com/ngwgsang/ViSP](https://github.com/ngwgsang/ViSP) | `git clone` |
| **ViQP** | [SCM-LAB/ViQP](https://huggingface.co/datasets/SCM-LAB/ViQP) | `datasets.load_dataset(...)` |

## Pretrained Model Checkpoint

The fine-tuned model is available on HuggingFace Hub:

```
vmhdaica/vnpi_model_checkpoint_3135
```

You can load it directly:

```python
from transformers import AutoModelForSequenceClassification, AutoTokenizer

model = AutoModelForSequenceClassification.from_pretrained(
    "vmhdaica/vnpi_model_checkpoint_3135"
)
tokenizer = AutoTokenizer.from_pretrained(
    "vmhdaica/vnpi_model_checkpoint_3135"
)
```

Or evaluate via CLI:

```bash
python -m src.evaluate --hf_model vmhdaica/vnpi_model_checkpoint_3135
```
