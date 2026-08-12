"""
Cassava Leaf Disease Assistant.

Loads the checkpoint saved by the notebook, classifies a leaf photo, then sends the
prediction to Claude for a written recommendation.

Run:
    export ANTHROPIC_API_KEY=sk-ant-...
    python app.py
"""

import os

import gradio as gr
import torch
import torch.nn as nn
from anthropic import Anthropic
from torchvision import models, transforms

CKPT = os.environ.get("CASSAVA_CKPT", "cassava_resnet18.pt")
LLM_MODEL = os.environ.get("CLAUDE_MODEL", "claude-sonnet-4-5")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DISEASE_NAMES = {
    "cbb": "Cassava Bacterial Blight",
    "cbsd": "Cassava Brown Streak Disease",
    "cgm": "Cassava Green Mite damage",
    "cmd": "Cassava Mosaic Disease",
    "healthy": "Healthy",
}

# ---------------------------------------------------------------- model

ckpt = torch.load(CKPT, map_location=DEVICE, weights_only=False)
CLASSES = ckpt["classes"]
IMG_SIZE = ckpt.get("img_size", 224)

model = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, len(CLASSES))
model.load_state_dict(ckpt["state_dict"])
model.eval().to(DEVICE)

eval_tf = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

# ---------------------------------------------------------------- llm

client = Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY", ""))


def explain(top):
    # top is a list of (class_name, probability) tuples, highest probability first.
    lines = "\n".join(f"- {DISEASE_NAMES.get(c, c)}: {p:.1%}" for c, p in top)
    prompt = (
        "An image classifier looked at a cassava leaf photo and produced these probabilities:\n"
        f"{lines}\n\n"
        "You are an agricultural extension officer talking to a smallholder farmer in Uganda. "
        "In under 150 words: say what this most likely is, what to do in the next week, and "
        "whether the confidence is low enough that a human should check. Plain language, no jargon."
    )
    msg = client.messages.create(model=LLM_MODEL, max_tokens=400,
                                 messages=[{"role": "user", "content": prompt}])
    return msg.content[0].text


# ---------------------------------------------------------------- ui

def classify(image):
    if image is None:
        return {}, "Upload a leaf photo first."
    x = eval_tf(image.convert("RGB")).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        probs = torch.softmax(model(x).float(), 1)[0].cpu().numpy()

    scores = {DISEASE_NAMES.get(c, c): float(p) for c, p in zip(CLASSES, probs)}
    top = sorted(zip(CLASSES, probs), key=lambda t: -t[1])[:3]
    try:
        advice = explain([(c, float(p)) for c, p in top])
    except Exception as e:
        # Usually a missing or expired key, or a model string that no longer exists.
        advice = f"Classifier worked, but the Claude call failed: {e}"
    return scores, advice


demo = gr.Interface(
    fn=classify,
    inputs=gr.Image(type="pil", label="Cassava leaf photo"),
    outputs=[gr.Label(num_top_classes=5, label="Model output"),
             gr.Textbox(label="What to do about it", lines=8)],
    title="Cassava Leaf Disease Assistant",
    description=("ResNet18 fine-tuned on the iCassava 2019 dataset. The prediction is passed "
                 "to Claude, which writes the advice. Trained on field photos from Uganda, so "
                 "it will be less reliable on other crops or on lab-style white backgrounds."),
    flagging_mode="never",
)

if __name__ == "__main__":
    demo.launch()
