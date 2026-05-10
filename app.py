from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from PIL import Image
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.models import resnet50
import os

app = Flask(__name__)
CORS(app)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
device = torch.device("cpu")

CATEGORIES = [
    'bottle', 'cable', 'capsule', 'carpet', 'grid',
    'hazelnut', 'leather', 'metal_nut', 'pill', 'screw',
    'tile', 'toothbrush', 'transistor', 'wood', 'zipper'
]
QUALITY_TYPES = ['good', 'bad']
NUM_CLASSES = len(CATEGORIES) * 2  # 30
CLASSES = [f"{cat}_{qual}" for cat in CATEGORIES for qual in QUALITY_TYPES]

# Build model — MUST match train.py exactly
model = resnet50(weights=None)
for param in model.parameters():
    param.requires_grad = False

model.fc = nn.Sequential(
    nn.Linear(model.fc.in_features, 512),
    nn.BatchNorm1d(512),
    nn.ReLU(),
    nn.Dropout(0.4),
    nn.Linear(512, 256),
    nn.ReLU(),
    nn.Dropout(0.2),
    nn.Linear(256, NUM_CLASSES)
)

MODEL_PATH = os.path.join(BASE_DIR, 'best_exact_class_model.pth')
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f"Model not found: {MODEL_PATH}\n"
        "Run 'python train.py' first to generate the model file."
    )

model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location=device,
        weights_only=False
    )
)
model.to(device)
model.eval()
print(f"Model loaded: {MODEL_PATH}")
print(f"Classes: {NUM_CLASSES} | Device: {device}")

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


@app.route('/')
def index():
    return send_from_directory(BASE_DIR, 'index.html')


@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400

    try:
        file = request.files['image']
        img = Image.open(file.stream).convert('RGB')
        tensor = transform(img).unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = model(tensor)
            probs = torch.nn.functional.softmax(outputs, dim=1)
            conf, index = torch.max(probs, 1)

        class_string = CLASSES[index.item()]
        product, status = class_string.rsplit('_', 1)

        return jsonify({
            'prediction': {
                'class': class_string,
                'product': product,
                'is_defective': status == 'bad',
                'status': status.upper(),
                'confidence': round(float(conf.item()), 4)
            }
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    print("\nFlask server starting at http://localhost:5000")
    print("Open http://localhost:5000 in browser to use the UI.")
    app.run(host='0.0.0.0', port=5000, debug=False)