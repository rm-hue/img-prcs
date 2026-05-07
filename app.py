from flask import Flask, request, jsonify, send_from_directory
from PIL import Image
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.models import resnet50
import os

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
device = torch.device("cpu") # <--- CHANGED: Removed cuda check since you are on CPU

# <--- CHANGED: Removed labels.json. We now define the 14 classes directly here 
# so it perfectly matches the train.py output (e.g., bottle_good, bottle_bad, etc.)
CATEGORIES = ['bottle', 'BSD', 'cable', 'capsule', 'carpet', 'grid', 'hazelnut']
QUALITY_TYPES = ['good', 'bad']
CLASSES_14 = [f"{cat}_{qual}" for cat in CATEGORIES for qual in QUALITY_TYPES]

# Load model
model = resnet50(weights=None)

# <--- CHANGED: The model architecture MUST match the new train.py (Sequential head with 512 neurons)
for param in model.parameters(): param.requires_grad = False
model.fc = nn.Sequential(
    nn.Linear(model.fc.in_features, 512),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(512, 14) # 14 classes instead of old num_classes
)

# <--- CHANGED: Load the new filename, and added weights_only=True for security
model.load_state_dict(
    torch.load(
        os.path.join(BASE_DIR, 'best_exact_class_model.pth'), 
        map_location=device, 
        weights_only=True
    )
)
model.to(device)
model.eval()

# <--- CHANGED: Transforms must match train.py exactly (Resize 256 -> CenterCrop 224)
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Route to serve your HTML file (No CORS needed anymore)
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
        
        # <--- CHANGED: Get string from our new CLASSES_14 list instead of labels.json
        class_string = CLASSES_14[index.item()]
        product, status = class_string.rsplit('_', 1)
        
        # This output perfectly matches your Javascript variables: p.product, p.status, p.confidence
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
    app.run(host='0.0.0.0', port=5000, debug=True)