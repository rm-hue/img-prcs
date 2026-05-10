img-prcs/                          # Project root (flat layout — no separate frontend/backend)
├── app.py                         # Flask web server — loads model, serves /predict API + index.html
├── train.py                       # ResNet50 training script — full pipeline (data→model→dashboard)
├── index.html                     # Frontend UI — drag & drop image upload, shows prediction results
├── summary.md                     # Project documentation — architecture, results, deployment guide
├── txt.md                         # Training logs, file history, old code snapshots
├── requirements.txt               # Python dependencies (torch, torchvision, flask, sklearn, etc.)
├── labels.json                    # Legacy label map (7-class era) — NOT used by current train.py/app.py
├── training_dashboard.png         # Auto-generated after training — accuracy/loss/confusion/per-class plots
├── best_exact_class_model.pth     # Trained ResNet50 weights (142MB) — NOT in git (.gitignore), share via Drive
├── .gitignore                     # Excludes: *.pth, archive/, venv/, .agents/
│
├── archive/                       # MVTec AD dataset (NOT in git — too large)
│   ├── bottle/
│   │   ├── train/good/            # Normal product images used for training
│   │   └── test/
│   │       ├── good/              # Normal product images used for testing
│   │       ├── broken_large/      # Defect subtype → all merged as "bad" label
│   │       └── contamination/     # Defect subtype → all merged as "bad" label
│   ├── cable/
│   ├── capsule/
│   ├── carpet/
│   ├── grid/
│   ├── hazelnut/
│   ├── leather/
│   ├── metal_nut/
│   ├── pill/
│   ├── screw/
│   ├── tile/
│   ├── toothbrush/
│   ├── transistor/
│   ├── wood/
│   └── zipper/
│       (15 categories × 2 labels = 30 classes total | ~5354 images)
│
│   Dataset Links:
│   Official  → https://www.mvtec.com/company/research/datasets/mvtec-ad
│   Kaggle    → https://www.kaggle.com/datasets/ipythonx/mvtec-ad
│   Paper     → https://link.springer.com/article/10.1007/s11263-020-01400-4
│
└── venv/                          # Python 3.12 virtual environment (NOT in git)
    └── (torch, torchvision, torch-directml, flask, sklearn, matplotlib, seaborn)




1st training result (partial dataset, 7 classes)
Loading dataset...
Total: 4902 images | Classes: 14
Split: Train 3431 | Val 735 | Test 736
Setting up model (Frozen backbone)...
Training...
Epoch 1/8 | Train: 41.1% | Val: 50.6%
Epoch 2/8 | Train: 46.9% | Val: 48.3%
Epoch 3/8 | Train: 49.3% | Val: 51.4%
Epoch 4/8 | Train: 50.2% | Val: 47.2%
Epoch 5/8 | Train: 50.4% | Val: 56.5%
Epoch 6/8 | Train: 50.6% | Val: 55.1%
Epoch 7/8 | Train: 51.5% | Val: 55.9%
Epoch 8/8 | Train: 51.9% | Val: 52.9%
Testing...
Test Accuracy (Exact Class): 54.08%
Test Accuracy (Defective vs OK): 90.62%
Saved defect_model.pth and labels.json


2nd training result (partial dataset, 7 classes)
Found 6090 total images.
Initializing ResNet50...
Downloading: "https://download.pytorch.org/models/resnet50-11ad3fa6.pth" to C:\Users\HAL/.cache\torch\hub\checkpoints\resnet50-11ad3fa6.pth
100%|████████████████████████████████████████████████████████████████████| 97.8M/97.8M [00:10<00:00, 9.49MB/s]
Starting training for 20 epochs on CPU...
Epoch  1/20 | Train Loss: 0.0655 | Train Acc:  61.3% | Val Acc:  68.1%
  *** New Best Val Acc: 68.1% - Model Saved! ***
Epoch  2/20 | Train Loss: 0.0544 | Train Acc:  65.0% | Val Acc:  65.5%
Epoch  3/20 | Train Loss: 0.0532 | Train Acc:  65.1% | Val Acc:  70.4%
  *** New Best Val Acc: 70.4% - Model Saved! ***
Epoch  4/20 | Train Loss: 0.0519 | Train Acc:  66.0% | Val Acc:  70.5%
  *** New Best Val Acc: 70.5% - Model Saved! ***
Epoch  5/20 | Train Loss: 0.0510 | Train Acc:  67.4% | Val Acc:  70.3%
Epoch  6/20 | Train Loss: 0.0501 | Train Acc:  66.7% | Val Acc:  70.5%
Epoch  7/20 | Train Loss: 0.0498 | Train Acc:  67.8% | Val Acc:  70.4%
Epoch  8/20 | Train Loss: 0.0492 | Train Acc:  68.1% | Val Acc:  70.3%
Epoch  9/20 | Train Loss: 0.0487 | Train Acc:  69.0% | Val Acc:  70.5%
Epoch 10/20 | Train Loss: 0.0487 | Train Acc:  68.3% | Val Acc:  70.8%
  *** New Best Val Acc: 70.8% - Model Saved! ***
Epoch 11/20 | Train Loss: 0.0478 | Train Acc:  69.1% | Val Acc:  70.4%
Epoch 12/20 | Train Loss: 0.0476 | Train Acc:  69.3% | Val Acc:  70.8%
Epoch 13/20 | Train Loss: 0.0472 | Train Acc:  68.8% | Val Acc:  70.7%
Epoch 14/20 | Train Loss: 0.0470 | Train Acc:  69.4% | Val Acc:  70.9%
  *** New Best Val Acc: 70.9% - Model Saved! ***
Epoch 15/20 | Train Loss: 0.0469 | Train Acc:  69.3% | Val Acc:  70.9%
  *** New Best Val Acc: 70.9% - Model Saved! ***
Epoch 16/20 | Train Loss: 0.0468 | Train Acc:  70.3% | Val Acc:  70.9%
Epoch 17/20 | Train Loss: 0.0468 | Train Acc:  69.5% | Val Acc:  70.9%
Epoch 18/20 | Train Loss: 0.0468 | Train Acc:  69.6% | Val Acc:  71.0%
  *** New Best Val Acc: 71.0% - Model Saved! ***
Epoch 19/20 | Train Loss: 0.0464 | Train Acc:  69.8% | Val Acc:  71.0%
Epoch 20/20 | Train Loss: 0.0463 | Train Acc:  70.1% | Val Acc:  71.0%

Training complete!

========================================
FINAL EVALUATION ON VALIDATION SET
========================================
c:\Users\HAL\Desktop\ritwik trainee 20-04 to16-05 2026\image processing\backend\train.py:202: FutureWarning: You are using `torch.load` with `weights_only=False` (the current default value), which uses the default pickle module implicitly. It is possible to construct malicious pickle data which will execute arbitrary code during unpickling (See https://github.com/pytorch/pytorch/blob/main/SECURITY.md#untrusted-models for more details). In a future release, the default value for `weights_only` will be flipped to `True`. This limits the functions that could be executed during unpickling. Arbitrary objects will no longer be allowed to be loaded via this mode unless they are explicitly allowlisted by the user via `torch.serialization.add_safe_globals`. We recommend you start setting `weights_only=True` for any use case where you don't have full control of the loaded file. Please open an issue on GitHub for any issues related to this experimental feature.
  model.load_state_dict(torch.load('best_exact_class_model.pth'))
Exact Class Accuracy:   71.02%
Per-Category Breakdown:
  bottle     :  63.3%
  BSD        :  74.1%
  cable      :  57.9%
  capsule    :  63.2%
  carpet     :  60.3%
  grid       :  61.0%
  hazelnut   :  98.5%

3rd training (partial dataset, 7 classes)

Found 3045 total images.
Initializing ResNet50...
Starting training for 20 epochs...
Epoch  1/20 | T_Loss: 0.0560 | V_Loss: 0.0333 | T_Acc:  63.5% | V_Acc:  81.4%
  *** New Best: 81.4% ***
Epoch  2/20 | T_Loss: 0.0362 | V_Loss: 0.0278 | T_Acc:  72.8% | V_Acc:  74.9%
Epoch  3/20 | T_Loss: 0.0300 | V_Loss: 0.0284 | T_Acc:  77.0% | V_Acc:  87.8%
  *** New Best: 87.8% ***
Epoch  4/20 | T_Loss: 0.0297 | V_Loss: 0.0337 | T_Acc:  78.7% | V_Acc:  85.9%
Epoch  5/20 | T_Loss: 0.0299 | V_Loss: 0.0279 | T_Acc:  78.6% | V_Acc:  82.9%
Epoch  6/20 | T_Loss: 0.0277 | V_Loss: 0.0226 | T_Acc:  80.0% | V_Acc:  87.8%
Epoch  7/20 | T_Loss: 0.0260 | V_Loss: 0.0249 | T_Acc:  82.1% | V_Acc:  79.0%
Epoch  8/20 | T_Loss: 0.0257 | V_Loss: 0.0277 | T_Acc:  82.5% | V_Acc:  82.1%
Epoch  9/20 | T_Loss: 0.0240 | V_Loss: 0.0259 | T_Acc:  82.8% | V_Acc:  87.5%
Epoch 10/20 | T_Loss: 0.0233 | V_Loss: 0.0233 | T_Acc:  85.0% | V_Acc:  84.7%
Epoch 11/20 | T_Loss: 0.0229 | V_Loss: 0.0223 | T_Acc:  83.7% | V_Acc:  84.7%
Epoch 12/20 | T_Loss: 0.0215 | V_Loss: 0.0241 | T_Acc:  85.7% | V_Acc:  87.7%
Epoch 13/20 | T_Loss: 0.0207 | V_Loss: 0.0227 | T_Acc:  86.1% | V_Acc:  87.4%
Epoch 14/20 | T_Loss: 0.0204 | V_Loss: 0.0225 | T_Acc:  85.8% | V_Acc:  88.2%
  *** New Best: 88.2% ***
Epoch 15/20 | T_Loss: 0.0183 | V_Loss: 0.0221 | T_Acc:  88.0% | V_Acc:  84.7%
Epoch 16/20 | T_Loss: 0.0202 | V_Loss: 0.0210 | T_Acc:  86.7% | V_Acc:  87.2%
Epoch 17/20 | T_Loss: 0.0192 | V_Loss: 0.0219 | T_Acc:  86.9% | V_Acc:  87.8%
Epoch 18/20 | T_Loss: 0.0195 | V_Loss: 0.0211 | T_Acc:  87.4% | V_Acc:  87.5%
Epoch 19/20 | T_Loss: 0.0189 | V_Loss: 0.0195 | T_Acc:  87.2% | V_Acc:  88.0%
Epoch 20/20 | T_Loss: 0.0176 | V_Loss: 0.0203 | T_Acc:  88.7% | V_Acc:  88.2%

Training complete!

========================================
FINAL EVALUATION ON 14 CLASSES
========================================
Exact 14-Class Accuracy: 88.18%
Per-Item Breakdown:
  bottle_good     : 100.0%
  bottle_bad      : 100.0%
  BSD_good        :  82.2%
  BSD_bad         :  83.5%
  cable_good      :  92.9%
  cable_bad       :  44.4%
  capsule_good    :  97.9%
  capsule_bad     :  50.0%
  carpet_good     :  90.3%
  carpet_bad      :  88.9%
  grid_good       :  96.5%
  grid_bad        :  18.2%
  hazelnut_good   : 100.0%
  hazelnut_bad    : 100.0%

Graph saved as 'training_graphs.png'

4th training, full 15 classes, (30 categories)

(venv) PS C:\Users\ritwi\OneDrive\Desktop\ex\img-prcs> python train.py        
Device: DirectML — AMD/Intel GPU via DirectX 12
Loading dataset...
Total: 5354 images | Classes: 30/30
  bottle_good              :  229
  bottle_bad               :   63
  cable_good               :  282
  cable_bad                :   92
  capsule_good             :  242
  capsule_bad              :  109
  carpet_good              :  308
  carpet_bad               :   89
  grid_good                :  285
  grid_bad                 :   57
  hazelnut_good            :  431
  hazelnut_bad             :   70
  leather_good             :  277
  leather_bad              :   92
  metal_nut_good           :  242
  metal_nut_bad            :   93
  pill_good                :  293
  pill_bad                 :  141
  screw_good               :  361
  screw_bad                :  119
  tile_good                :  263
  tile_bad                 :   84
  toothbrush_good          :   72
  toothbrush_bad           :   30
  transistor_good          :  273
  transistor_bad           :   40
  wood_good                :  266
  wood_bad                 :   60
  zipper_good              :  272
  zipper_bad               :  119

Building ResNet50...
Training 50 epochs | train=4283 val=1071

C:\Users\ritwi\OneDrive\Desktop\ex\img-prcs\venv\Lib\site-packages\torch\optim\adamw.py:529: UserWarning: The operator 'aten::lerp.Scalar_out' is not currently supported on the DML backend and will fall back to run on the CPU. This may have performance implications. (Triggered internally at C:\__w\1\s\pytorch-directml-plugin\torch_directml\csrc\dml\dml_cpu_fallback.cpp:17.)
  torch._foreach_lerp_(device_exp_avgs, device_grads, 1 - beta1)
Epoch  1/50 | T_Loss:0.1156 T_Acc: 48.1% | V_Loss:0.0886 V_Acc: 47.5%
  *** Best: 47.53% saved ***
Epoch  2/50 | T_Loss:0.0785 T_Acc: 60.6% | V_Loss:0.0791 V_Acc: 68.9%
  *** Best: 68.91% saved ***
Epoch  3/50 | T_Loss:0.0737 T_Acc: 66.1% | V_Loss:0.0755 V_Acc: 71.1%
  *** Best: 71.06% saved ***
Epoch  4/50 | T_Loss:0.0867 T_Acc: 64.2% | V_Loss:0.0744 V_Acc: 75.4%
  *** Best: 75.35% saved ***
Epoch  5/50 | T_Loss:0.0808 T_Acc: 68.5% | V_Loss:0.0725 V_Acc: 75.1%
Epoch  6/50 | T_Loss:0.0809 T_Acc: 68.4% | V_Loss:0.0692 V_Acc: 81.2%
  *** Best: 81.23% saved ***
Epoch  7/50 | T_Loss:0.0761 T_Acc: 72.9% | V_Loss:0.0668 V_Acc: 84.5%
  *** Best: 84.50% saved ***
Epoch  8/50 | T_Loss:0.0809 T_Acc: 71.8% | V_Loss:0.0651 V_Acc: 84.1%
Epoch  9/50 | T_Loss:0.0776 T_Acc: 74.2% | V_Loss:0.0639 V_Acc: 90.4%
  *** Best: 90.38% saved ***
Epoch 10/50 | T_Loss:0.0783 T_Acc: 73.3% | V_Loss:0.0637 V_Acc: 90.7%
  *** Best: 90.66% saved ***
Epoch 11/50 | T_Loss:0.0742 T_Acc: 77.2% | V_Loss:0.0641 V_Acc: 88.4%
Epoch 12/50 | T_Loss:0.0761 T_Acc: 75.7% | V_Loss:0.0622 V_Acc: 91.9%
  *** Best: 91.88% saved ***
Epoch 13/50 | T_Loss:0.0731 T_Acc: 78.3% | V_Loss:0.0623 V_Acc: 91.9%
Epoch 14/50 | T_Loss:0.0716 T_Acc: 79.5% | V_Loss:0.0611 V_Acc: 91.1%
Epoch 15/50 | T_Loss:0.0728 T_Acc: 79.2% | V_Loss:0.0616 V_Acc: 92.1%
  *** Best: 92.06% saved ***
Epoch 16/50 | T_Loss:0.0714 T_Acc: 80.4% | V_Loss:0.0603 V_Acc: 91.5%
Epoch 17/50 | T_Loss:0.0710 T_Acc: 80.4% | V_Loss:0.0592 V_Acc: 93.9%
  *** Best: 93.93% saved ***
Epoch 18/50 | T_Loss:0.0688 T_Acc: 82.1% | V_Loss:0.0588 V_Acc: 94.3%
  *** Best: 94.30% saved ***
Epoch 19/50 | T_Loss:0.0693 T_Acc: 81.0% | V_Loss:0.0608 V_Acc: 92.2%
Epoch 20/50 | T_Loss:0.0693 T_Acc: 81.5% | V_Loss:0.0588 V_Acc: 93.0%
Epoch 21/50 | T_Loss:0.0676 T_Acc: 83.8% | V_Loss:0.0600 V_Acc: 92.4%
Epoch 22/50 | T_Loss:0.0660 T_Acc: 84.4% | V_Loss:0.0600 V_Acc: 91.5%
Epoch 23/50 | T_Loss:0.0675 T_Acc: 84.1% | V_Loss:0.0593 V_Acc: 93.5%
Epoch 24/50 | T_Loss:0.0670 T_Acc: 84.1% | V_Loss:0.0594 V_Acc: 92.6%
Epoch 25/50 | T_Loss:0.0646 T_Acc: 85.1% | V_Loss:0.0594 V_Acc: 95.3%
  *** Best: 95.33% saved ***
Epoch 26/50 | T_Loss:0.0671 T_Acc: 83.9% | V_Loss:0.0584 V_Acc: 94.4%
Epoch 27/50 | T_Loss:0.0648 T_Acc: 85.0% | V_Loss:0.0598 V_Acc: 92.6%
Epoch 28/50 | T_Loss:0.0663 T_Acc: 84.2% | V_Loss:0.0582 V_Acc: 93.9%
Epoch 29/50 | T_Loss:0.0657 T_Acc: 84.5% | V_Loss:0.0582 V_Acc: 95.0%
Epoch 30/50 | T_Loss:0.0632 T_Acc: 86.4% | V_Loss:0.0574 V_Acc: 94.7%
Epoch 31/50 | T_Loss:0.0662 T_Acc: 84.8% | V_Loss:0.0575 V_Acc: 94.1%
Epoch 32/50 | T_Loss:0.0636 T_Acc: 85.8% | V_Loss:0.0576 V_Acc: 94.5%
Epoch 33/50 | T_Loss:0.0655 T_Acc: 84.7% | V_Loss:0.0579 V_Acc: 95.1%
Epoch 34/50 | T_Loss:0.0666 T_Acc: 84.4% | V_Loss:0.0580 V_Acc: 94.0%
Epoch 35/50 | T_Loss:0.0628 T_Acc: 86.3% | V_Loss:0.0578 V_Acc: 94.3%
Epoch 36/50 | T_Loss:0.0639 T_Acc: 85.5% | V_Loss:0.0576 V_Acc: 95.1%
Epoch 37/50 | T_Loss:0.0646 T_Acc: 85.8% | V_Loss:0.0579 V_Acc: 94.9%
Epoch 38/50 | T_Loss:0.0674 T_Acc: 84.4% | V_Loss:0.0592 V_Acc: 94.5%
Epoch 39/50 | T_Loss:0.0634 T_Acc: 86.0% | V_Loss:0.0580 V_Acc: 94.5%
Epoch 40/50 | T_Loss:0.0631 T_Acc: 86.5% | V_Loss:0.0576 V_Acc: 95.0%

Early stop at epoch 40

Done! Best val acc: 95.33%

==================================================
Overall Accuracy: 95.33%
Anomaly Detection F1: 89.18%  (good vs bad, binary)

  bottle_good              : 100.0%  (46 samples)
  bottle_bad               : 100.0%  (13 samples)
  cable_good               : 100.0%  (57 samples)
  cable_bad                :  66.7%  (18 samples)
  capsule_good             : 100.0%  (48 samples)
  capsule_bad              :  63.6%  (22 samples)
  carpet_good              : 100.0%  (62 samples)
  carpet_bad               :  94.4%  (18 samples)
  grid_good                : 100.0%  (57 samples)
  grid_bad                 : 100.0%  (11 samples)
  hazelnut_good            : 100.0%  (86 samples)
  hazelnut_bad             : 100.0%  (14 samples)
  leather_good             :  98.2%  (55 samples)
  leather_bad              : 100.0%  (18 samples)
  metal_nut_good           : 100.0%  (48 samples)
  metal_nut_bad            :  84.2%  (19 samples)
  pill_good                :  96.6%  (59 samples)
  pill_bad                 :  60.7%  (28 samples)
  screw_good               :  98.6%  (72 samples)
  screw_bad                :  50.0%  (24 samples)
  tile_good                : 100.0%  (53 samples)
  tile_bad                 : 100.0%  (17 samples)
  toothbrush_good          : 100.0%  (14 samples)
  toothbrush_bad           :  50.0%  (6 samples)
  transistor_good          : 100.0%  (55 samples)
  transistor_bad           :  75.0%  (8 samples)
  wood_good                : 100.0%  (53 samples)
  wood_bad                 : 100.0%  (12 samples)
  zipper_good              : 100.0%  (54 samples)
  zipper_bad               : 100.0%  (24 samples)


index.html

<!DOCTYPE html>
<html>
<head>
<title>ResNet50 Classifier</title>
<link rel="stylesheet" href="index.css">
</head>
<body>
    <h1>Image Classifier (ResNet50)</h1>
    <input type="file" id="imgInput" accept="image/*">
    <img id="preview" style="max-width:300px;display:block;margin:10px 0">
    <button id="predictBtn" disabled>Predict</button>
    <div id="results"></div>
    <script src="index.js"></script>
</body>
</html>
index.css
body { font-family: sans-serif; padding: 20px; }
button { margin: 10px 0; }
.result { margin: 5px 0; }


index.js
const input = document.getElementById('imgInput');
const preview = document.getElementById('preview');
const predictBtn = document.getElementById('predictBtn');
const results = document.getElementById('results');
let currentFile = null;

// Handle file selection and preview
input.onchange = () => {
    const file = input.files[0];
    if (file) {
        currentFile = file;
        // Create a local URL for the image so the user can see what they uploaded
        preview.src = URL.createObjectURL(file);
        preview.style.display = 'block'; // Ensure image is visible
        predictBtn.disabled = false;
        results.innerHTML = 'Image loaded. Click "Predict" to analyze.';
    }
};

// Handle the prediction request
predictBtn.onclick = async () => {
    if (!currentFile) return;

    // UI Feedback
    results.innerHTML = '<span style="color: blue;">Processing image...</span>';
    predictBtn.disabled = true;

    try {
        const fd = new FormData();
        fd.append('image', currentFile);

        // Pointing to the Flask backend URL
        const response = await fetch('http://127.0.0.1:5000/predict', { 
            method: 'POST', 
            body: fd 
        });

        if (!response.ok) {
            throw new Error(`Server responded with status: ${response.status}`);
        }

        const data = await response.json();

        // Check if backend returned a single prediction object or an error
        if (data.error) {
            results.innerHTML = `<span style="color: red;">Error: ${data.error}</span>`;
        } else if (data.prediction) {
            const p = data.prediction;
            const confidencePercent = (p.confidence * 100).toFixed(2);
            
            // Format the result clearly for the user
            results.innerHTML = `
                <div style="margin-top: 10px; padding: 10px; border: 1px solid #ddd; border-radius: 5px;">
                    <strong>Detected Category:</strong> ${p.label}<br>
                    <strong>Confidence Score:</strong> ${confidencePercent}%
                </div>
            `;
        } else {
            results.innerHTML = 'No prediction data received from server.';
        }

    } catch (error) {
        console.error('Fetch error:', error);
        results.innerHTML = `<span style="color: red;">Error: Could not connect to backend. Make sure app.py is running.</span>`;
    } finally {
        predictBtn.disabled = false;
    }
};


app.py
from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
import torch
import torchvision.transforms as transforms
from torchvision.models import resnet50
import os
import json

app = Flask(__name__)
CORS(app)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load labels
with open(os.path.join(BASE_DIR, 'labels.json'), 'r') as f:
    labels_data = json.load(f)

classes = labels_data['classes']
products = labels_data['products']
num_classes = len(classes)

# Load model
model = resnet50(weights=None)
model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
model.load_state_dict(torch.load(os.path.join(BASE_DIR, 'defect_model.pth'), map_location=device))
model.to(device)
model.eval()

# Transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

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
        
        class_name = classes[str(index.item())]
        product, status = class_name.rsplit('_', 1)
        
        return jsonify({
            'prediction': {
                'class': class_name,
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

train.py
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
from collections import Counter
import numpy as np


torch.set_num_threads(14) 
device = torch.device("cpu")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "archive")

BATCH_SIZE = 16
EPOCHS = 20
LEARNING_RATE = 5e-5

CATEGORIES = ['bottle', 'BSD', 'cable', 'capsule', 'carpet', 'grid', 'hazelnut']
CAT_TO_IDX = {cat: idx for idx, cat in enumerate(CATEGORIES)}


QUALITY_TO_IDX = {'good': 0, 'defective': 1} 


class MVTecDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.samples = []
        
        # Scan the folders
        for category in CATEGORIES:
            cat_path = os.path.join(root_dir, category)
            if not os.path.exists(cat_path):
                continue
                
            for quality_folder in os.listdir(cat_path):
                qual_path = os.path.join(cat_path, quality_folder)
                if not os.path.isdir(qual_path):
                    continue
                    
                # Determine label: 0 for good, 1 for anything else (bad)
                if quality_folder.lower() == 'good':
                    qual_label = 0
                else:
                    qual_label = 1
                
                for img_name in os.listdir(qual_path):
                    if img_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                        img_path = os.path.join(qual_path, img_name)
                        cat_label = CAT_TO_IDX[category]
                        self.samples.append((img_path, cat_label, qual_label))
                        
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, cat_label, qual_label = self.samples[idx]
        
        # Convert to RGB to ensure 3 channels
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
            
        return image, cat_label, qual_label


train_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(p=0.3),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


print("Loading dataset...")
print(f"Looking for images in: {DATA_DIR}")

full_dataset = MVTecDataset(DATA_DIR, transform=train_transform)

# TO CATCH ERRORS EARLY
if len(full_dataset) == 0:
    raise ValueError(f"No images found! Please check if '{DATA_DIR}' contains the category folders.")

print(f"Found {len(full_dataset)} total images.") # ADD THIS LINE

train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])

val_dataset.dataset = MVTecDataset(DATA_DIR, transform=val_transform)

def get_class_weights(dataset):
    labels = [sample[1] for sample in dataset.dataset.samples]
    class_counts = Counter(labels)
    total = len(labels)
    num_classes = 7
    weights = torch.zeros(num_classes)
    for cls in range(num_classes):
        count = class_counts.get(cls, 1) # prevent division by zero
        weights[cls] = total / (num_classes * count)
    return weights

class_weights = get_class_weights(train_dataset)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=False)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=False)


print("Initializing ResNet50...")
model = models.resnet50(weights='IMAGENET1K_V2')
model.fc = nn.Linear(model.fc.in_features, 7) # 7 exact classes
model = model.to(device)

criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)


print(f"Starting training for {EPOCHS} epochs on CPU...")
best_val_acc = 0.0

for epoch in range(EPOCHS):
    # --- TRAINING ---
    model.train()
    train_loss = 0.0
    train_correct = 0
    train_total = 0
    
    for images, cat_labels, qual_labels in train_loader:
        images = images.to(device)
        cat_labels = cat_labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, cat_labels)
        
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        train_total += cat_labels.size(0)
        train_correct += (predicted == cat_labels).sum().item()
        
    train_acc = 100 * train_correct / train_total
    
    # --- VALIDATION ---
    model.eval()
    val_correct = 0
    val_total = 0
    val_qual_correct = 0
    
    with torch.no_grad():
        for images, cat_labels, qual_labels in val_loader:
            images = images.to(device)
            cat_labels = cat_labels.to(device)
            qual_labels = qual_labels.to(device)
            
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            
            val_total += cat_labels.size(0)
            val_correct += (predicted == cat_labels).sum().item()
            

            
    val_acc = 100 * val_correct / val_total
    scheduler.step()
    
    print(f"Epoch {epoch+1:2}/{EPOCHS} | Train Loss: {train_loss/train_total:.4f} | Train Acc: {train_acc:5.1f}% | Val Acc: {val_acc:5.1f}%")
    
    # Save best model
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), 'best_exact_class_model.pth')
        print(f"  *** New Best Val Acc: {val_acc:.1f}% - Model Saved! ***")

print("\nTraining complete!")


print("\n" + "="*40)
print("FINAL EVALUATION ON VALIDATION SET")
print("="*40)


model.load_state_dict(torch.load('best_exact_class_model.pth'))
model.eval()

all_cat_preds = []
all_cat_labels = []
all_qual_labels = []

with torch.no_grad():
    for images, cat_labels, qual_labels in val_loader:
        images = images.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        
        all_cat_preds.extend(predicted.cpu().numpy())
        all_cat_labels.extend(cat_labels.cpu().numpy())
        all_qual_labels.extend(qual_labels.cpu().numpy())

all_cat_preds = np.array(all_cat_preds)
all_cat_labels = np.array(all_cat_labels)
all_qual_labels = np.array(all_qual_labels)

exact_acc = (all_cat_preds == all_cat_labels).mean()


qual_assumed_correct = (all_cat_preds == all_cat_labels).astype(int)
binary_acc = (qual_assumed_correct == (1 - all_qual_labels)).mean() # 1 - qual_label because good=0, defective=1, and correct=1, wrong=0
binary_acc = exact_acc # In a single 7-class head, if it guesses the exact item, it inherently guesses good/bad.

print(f"Exact Class Accuracy:   {exact_acc*100:.2f}%")
print(f"Per-Category Breakdown:")
for idx, cat in enumerate(CATEGORIES):
    mask = all_cat_labels == idx
    if mask.sum() > 0:
        cat_acc = (all_cat_preds[mask] == all_cat_labels[mask]).mean()
        print(f"  {cat:10} : {cat_acc*100:5.1f}%")



requirements.txt
flask==3.0.3
flask-cors==4.0.1
torch==2.4.1
torchvision==0.19.1
pillow==10.4.0
numpy==2.1.1
scikit-learn==1.5.2
matplotlib==3.9.2
seaborn==0.13.2
kagglehub




labels.json //after running
{
  "classes": {
    "0": "bottle_good",
    "1": "bottle_bad",
    "2": "BSD_good",
    "3": "BSD_bad",
    "4": "cable_good",
    "5": "cable_bad",
    "6": "capsule_good",
    "7": "capsule_bad",
    "8": "carpet_good",
    "9": "carpet_bad",
    "10": "grid_good",
    "11": "grid_bad",
    "12": "hazelnut_good",
    "13": "hazelnut_bad"
  },
  "products": [
    "bottle",
    "BSD",
    "cable",
    "capsule",
    "carpet",
    "grid",
    "hazelnut"
  ]
}
