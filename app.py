"""
Gradio Demo — Vietnamese Paraphrase Identification

Launch locally:
    pip install gradio
    python app.py

Or deploy to HuggingFace Spaces.
"""

import torch
import gradio as gr
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# ── Load model ───────────────────────────────────────────────────
MODEL_ID = "vmhdaica/vnpi_model_checkpoint_3135"
MAX_LENGTH = 256

print(f"Loading model: {MODEL_ID}")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=True)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_ID)
device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device).eval()
print(f"Model loaded on {device} ✓")


# ── Prediction function ─────────────────────────────────────────
def predict(sentence1: str, sentence2: str) -> dict:
    """
    Compare two Vietnamese sentences and return paraphrase probability.
    """
    if not sentence1.strip() or not sentence2.strip():
        return {"paraphrase": 0.0, "not_paraphrase": 1.0}

    inputs = tokenizer(
        sentence1, sentence2,
        truncation=True, max_length=MAX_LENGTH,
        return_tensors="pt",
    ).to(device)
    inputs.pop("token_type_ids", None)

    with torch.no_grad():
        logits = model(**inputs).logits
        probs = torch.softmax(logits, dim=-1)[0].cpu().numpy()

    return {
        "paraphrase": float(probs[1]),
        "not_paraphrase": float(probs[0]),
    }


# ── Examples ─────────────────────────────────────────────────────
examples = [
    ["Hôm nay trời mưa rất to.", "Thời tiết hôm nay mưa lớn."],
    ["Giá vàng tăng mạnh.", "Trận đấu tối qua rất hấp dẫn."],
    ["Thủ tướng đã họp với các bộ trưởng.",
     "Cuộc họp của Thủ tướng với nội các đã diễn ra."],
    ["Hà Nội là thủ đô của Việt Nam.",
     "TP.HCM là thành phố lớn nhất Việt Nam."],
    ["Cô ấy rất giỏi tiếng Anh.",
     "Khả năng tiếng Anh của cô ấy rất tốt."],
    ["Tôi đi ăn phở sáng nay.",
     "Sáng nay tôi đã thưởng thức một tô phở."],
]


# ── Gradio Interface ─────────────────────────────────────────────
with gr.Blocks(
    title="🇻🇳 Vietnamese Paraphrase Identification",
    theme=gr.themes.Soft(primary_hue="blue"),
) as demo:

    gr.Markdown(
        """
        # 🇻🇳 Vietnamese Paraphrase Identification

        Determine whether two Vietnamese sentences convey the **same meaning**.

        **Model:** [PhoBERT-base-v2](https://github.com/VinAIResearch/PhoBERT)
        fine-tuned on 40K+ sentence pairs from 4 public datasets
        · **97.02% accuracy** · **0.876 macro-F1**
        · [Model checkpoint](https://huggingface.co/vmhdaica/vnpi_model_checkpoint_3135)
        """
    )

    with gr.Row():
        with gr.Column():
            txt1 = gr.Textbox(
                label="Câu 1 (Sentence 1)",
                placeholder="Nhập câu tiếng Việt thứ nhất...",
                lines=3,
            )
            txt2 = gr.Textbox(
                label="Câu 2 (Sentence 2)",
                placeholder="Nhập câu tiếng Việt thứ hai...",
                lines=3,
            )
            btn = gr.Button("🔍 So sánh / Compare", variant="primary", size="lg")

        with gr.Column():
            output = gr.Label(label="Kết quả / Result", num_top_classes=2)

    gr.Examples(
        examples=examples,
        inputs=[txt1, txt2],
        outputs=output,
        fn=predict,
        cache_examples=False,
    )

    btn.click(fn=predict, inputs=[txt1, txt2], outputs=output)
    txt2.submit(fn=predict, inputs=[txt1, txt2], outputs=output)

    gr.Markdown(
        """
        ---
        **How it works:**
        The model tokenizes both sentences, feeds them through PhoBERT-base-v2,
        and outputs a probability for each class.
        Trained with hard-negative mining and class-balanced loss
        on VNPC + vnPara + ViSP + ViQP datasets.

        [GitHub](https://github.com/Hoang-ca/vietnamese-paraphrase-identification)
        · [Model Card](https://huggingface.co/vmhdaica/vnpi_model_checkpoint_3135)
        """
    )

if __name__ == "__main__":
    demo.launch(share=False)
