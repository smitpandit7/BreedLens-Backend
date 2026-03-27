from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import torch
import torch.nn as nn
from torchvision import models, transforms
from torchvision.models import EfficientNet_B3_Weights
from PIL import Image
import requests
from io import BytesIO
import json

import os
from flask_cors import CORS

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "https://breedlens-frontend.vercel.app"}}) # allows frontend to call this backend

# ─── Load model once at startup ───────────────────────────────────────────────

device = torch.device("cpu")

# Load class names saved during training
with open("class_names.json", "r") as f:
    class_names = json.load(f)

num_classes = len(class_names)

# Rebuild the same model architecture you trained
model = models.efficientnet_b3(weights=None)
in_features = model.classifier[1].in_features
model.classifier = nn.Sequential(
    nn.Dropout(0.4),
    nn.Linear(in_features, 512),
    nn.SiLU(),
    nn.Dropout(0.3),
    nn.Linear(512, num_classes)
)

# Load your saved weights
checkpoint = torch.load("dog_breed_model.pth", map_location=device)
model.load_state_dict(checkpoint["model_state_dict"])
model.to(device)
model.eval()

print(f"✅ Model loaded — {num_classes} breeds on {device}")

# ─── Image transform (same as val_transform in training) ─────────────────────

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# ─── Helper: predict from PIL image ──────────────────────────────────────────

def predict(image: Image.Image, top_k: int = 3):
    img_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(img_tensor)
        probs  = torch.softmax(output, dim=1)
        top_probs, top_indices = probs[0].topk(top_k)

    results = []
    for prob, idx in zip(top_probs, top_indices):
        raw   = class_names[idx.item()]                        # e.g. n02106662-German_shepherd
        breed = raw.split("-", 1)[-1].replace("_", " ").title()
        results.append({
            "breed": breed,
            "pct":   round(prob.item() * 100, 2)
        })
    return results
    del img_tensor, output, probs

# ─── Routes ──────────────────────────────────────────────────────────────────
# Serve frontend directly from Flask
@app.route("/")
def home():
    return send_from_directory(".", "dog_breed_ui.html")


@app.route("/health")
def health():
    return jsonify({"status": "BarkID backend is running 🐾"})


@app.route("/predict/url", methods=["POST"])
def predict_from_url():
    """
    Body: { "url": "https://..." }
    Returns: { "predictions": [ {breed, pct}, ... ] }
    """
    data = request.get_json()
    url  = data.get("url", "").strip()

    if not url:
        return jsonify({"error": "No URL provided"}), 400

    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        image = Image.open(BytesIO(response.content)).convert("RGB")
    except Exception as e:
        return jsonify({"error": f"Could not load image: {str(e)}"}), 400

    predictions = predict(image)
    return jsonify({"predictions": predictions})


@app.route("/predict/upload", methods=["POST"])
def predict_from_upload():
    """
    Form-data: file = <image file>
    Returns: { "predictions": [ {breed, pct}, ... ] }
    """
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]

    try:
        image = Image.open(file.stream).convert("RGB")
    except Exception as e:
        return jsonify({"error": f"Could not read image: {str(e)}"}), 400

    predictions = predict(image)
    return jsonify({"predictions": predictions})


# ─── Run ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)