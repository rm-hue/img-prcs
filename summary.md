# MVTec AD Image Classification using ResNet50

This repository demonstrates image classification using the ResNet50 model for industrial anomaly detection (identifying defects in 15 different product categories).

## 1. Hardware & Environment Setup
- **Device Support:** AMD Radeon 780M (Integrated GPU) via DirectML. Also supports CUDA (NVIDIA) and CPU.
- **Backend:** `torch-directml` (DirectX 12 hardware acceleration for Windows).
- **Python Version:** Python 3.12 (inside a virtual environment).
- **Optimization:** DataLoaders use `num_workers=4` and `pin_memory=True` to load images in parallel, preventing the GPU from waiting on the CPU.

## 2. Dataset Processing (MVTec AD)
- **Categories:** 15 distinct product types (bottle, cable, capsule, carpet, grid, hazelnut, leather, metal_nut, pill, screw, tile, toothbrush, transistor, wood, zipper).
- **Classes:** 30 total classes (15 categories × 2 states: `good` and `bad`).
- **Data Pipeline:** 
  - `train/good` + `test/good` folders are combined as `good` samples.
  - Defect subfolders inside `test/` are combined as `bad` samples.
  - Imbalanced class distributions are handled dynamically using PyTorch's `WeightedRandomSampler`, ensuring the model sees rare defects as often as normal items.
- **Augmentations:** Heavy augmentation to prevent overfitting on limited defect data:
  - Random Crop (224x224), Horizontal & Vertical Flips, Random Rotation (15°).
  - ColorJitter (brightness, contrast, saturation, hue).
  - RandomErasing (cuts out random blocks of pixels).
  - MixUp Data Augmentation (blends two random images and their labels together).

## 3. Model Architecture
- **Base Model (Backbone):** ResNet50 (Pretrained on `IMAGENET1K_V2`).
- **Layer Strategy:**
  - **All Layers Unfrozen:** Full ResNet50 (Layer 1, 2, 3, 4) fine-tuned. Allows deep feature adaptation to MVTec AD defect textures.
- **Custom Classification Head:**
  The default 1000-class ImageNet head is replaced with a custom Multi-Layer Perceptron (MLP):
  1. `Linear` (2048 in → 512 out)
  2. `BatchNorm1d(512)`
  3. `ReLU()`
  4. `Dropout(0.4)`
  5. `Linear` (512 in → 256 out)
  6. `ReLU()`
  7. `Dropout(0.2)`
  8. `Linear` (256 in → 30 out) (Final class scores)

## 4. Training Hyperparameters
- **Epochs:** 50 (maximum, early stop at 40 via patience).
- **Early Stopping Patience:** 15 epochs.
- **Batch Size:** 16 (reduced for DirectML iGPU VRAM stability).
- **Learning Rates:**
  - All backbone layers: `1e-5` (conservative to preserve ImageNet features).
  - Fully Connected Head: `5e-4`.
- **Loss Function:** `CrossEntropyLoss` with Label Smoothing (0.1) + class weights (bad classes penalized 2×).
- **Scheduler:** Warmup (3 epochs) → CosineAnnealingLR decay.
- **Optimizer:** AdamW (DirectML compatible).

## 5. How to Export / Share on GitHub
To share this trained model so someone else can run it:

1. **Upload Files:** Push `app.py`, `train.py`, `index.html`, and `requirements.txt` to GitHub.
2. **Handle Large Files:** The trained weights file (`best_exact_class_model.pth`) is ~100MB. You must use **Git LFS (Large File Storage)** to push this file to GitHub.
3. **Usage by Others:** 
   Anyone cloning the repo only needs to do:
   ```bash
   pip install -r requirements.txt
   python app.py
   ```
   *Note: They do not need the `archive/` dataset folder or `train.py` just to make predictions.*

## 6. Future Upgrades

### Option A: ResNet50 + VGG16 Ensemble

#### What is it?
Run each image through **two different CNN backbones simultaneously**, concatenate their extracted features, then classify using a shared MLP head.

```
                ┌─────────────────────────────┐
                │       Input Image           │
                └────────────┬────────────────┘
                             │
              ┌──────────────┴──────────────┐
              │                             │
     ResNet50 backbone            VGG16 backbone
     (skip connections)           (deep conv stacks)
     output: 2048-d vector        output: 4096-d vector
              │                             │
              └──────────────┬──────────────┘
                             │ concat
                     [2048 + 4096] = 6144-d
                             │
                    Shared MLP Head
                    6144 → 512 → 30 classes
```

#### Why it helps
- **ResNet50** is strong at global structure recognition (skip connections help it see large patterns).
- **VGG16** is strong at fine-grained texture (deep stacked convolutions capture subtle surface defects).
- `screw_manipulated_front` = subtle front-face manipulation → VGG16 catches this better than ResNet50 alone.
- Ensemble = each model votes with its strengths → reduces individual model blind spots.

#### Expected Results
| Metric | ResNet50 alone | ResNet50 + VGG16 |
|---|---|---|
| Overall Accuracy | 95.33% | **97–98%** |
| Anomaly F1 | 89.2% | **93–95%** |
| `screw_bad` | 50% | **65–75%** |
| `toothbrush_bad` | 50% | **60–70%** |
| Training time | ~6 hrs | **~12 hrs** |
| Inference time | ~30ms | **~60ms** |

#### Cost & Tradeoffs
- 2× forward passes per image → 2× GPU memory → may need `BATCH_SIZE=8` on iGPU.
- `best_exact_class_model.pth` becomes ~300MB (both backbones stored).
- `app.py` must load and run both models for every prediction.

### Option B: PatchCore (Industry Standard)
Switch from supervised classification to **unsupervised feature matching**.
- PatchCore builds a memory bank of normal (good) image patches using ResNet50 features.
- At inference: compares new image patches to memory bank → anomaly score = maximum distance to nearest normal patch.
- **Does not need bad samples during training at all.**
- Effect: Detects ANY unseen defect type including `manipulated_front` without having seen it.
- Achieves ~99% AUROC on MVTec AD (industry benchmark). This is why MVTec dataset was originally designed for PatchCore-style methods, not classification.
- **Downside:** Returns anomaly score, not category. Cannot tell you "which product" — only "is it defective?". Needs separate classifier for product category.

### Option C: Current Model is Good Enough
- 95.33% overall, ~99% on good class = production-grade quality gate.
- Use current model to flag potential defects, then human QC confirms.
- For student demo/mentor presentation: current results + dashboard are strong enough.

## 7. Development History & Fixes
- **Initial Flaws Fixed:** The original pipeline ignored defect images entirely (training only on `good` samples) and only supported 7 of 15 categories. `train.py` was rewritten to support all 15 categories and correctly inject `bad` samples.
- **Class Imbalance:** Added `WeightedRandomSampler` because `good` images heavily outnumbered `bad` images, causing the model to initially guess "good" for everything.
- **Hardware Acceleration Journey:**
  - Initial training ran on CPU (Python 3.13 constraint). 20 epochs took ~2 hours.
  - Upgraded to Python 3.12 local `venv` to enable `torch-directml`, unlocking the AMD Radeon 780M iGPU.
  - Fixed DirectML data starvation by setting `num_workers=4` and `pin_memory=True` in the DataLoader.
  - Resolved a PyTorch 2.6 security restriction (`weights_only=True`) triggered by DirectML's numpy fallback during tensor serialization.
- **Result:** Validation accuracy reached **95.33%** at Epoch 25 (full unfreeze config, 40 epochs total). Anomaly Detection F1 = **89.2%**.

## 8. Final Results & Deployment (Option A)

### Performance Summary
| Metric | Value |
|---|---|
| Overall Validation Accuracy | **95.33%** |
| Anomaly Detection F1 (Good vs Bad) | **89.2%** |
| Good class accuracy (avg) | **~99%** |
| Bad class accuracy (avg) | **~80%** |
| Weak classes | `toothbrush_bad` (6 val samples), `transistor_bad`, `screw_bad` |
| Training time | ~6 hours (40 epochs, AMD Radeon 780M iGPU) |

### Step-by-Step: Run Demo

**Prerequisites:** `best_exact_class_model.pth` must exist (generated by `train.py`).

**Step 1 — Activate the virtual environment:**
```bash
cd C:\Users\ritwi\OneDrive\Desktop\ex\img-prcs
.\venv\Scripts\activate
```

**Step 2 — Start the Flask server:**
```bash
python app.py
```
Expected output:
```
Model loaded: ...\best_exact_class_model.pth
Classes: 30 | Device: cpu
Flask server starting at http://localhost:5000
```

**Step 3 — Open the UI:**
Open browser → go to `http://localhost:5000`

**Step 4 — Test with an image:**
- Upload any MVTec product image (bottle, cable, etc.).
- The model returns: product category, `GOOD` or `BAD` status, and confidence score.
- Example output: `{ "product": "bottle", "status": "BAD", "confidence": 0.9732 }`

**Step 5 — Show mentor the dashboard:**
Open `training_dashboard.png` in the project folder. Shows:
- Accuracy and loss curves per epoch.
- Confusion matrix (normalised).
- Anomaly detection F1 confusion matrix.
- Per-class accuracy bar chart.
- Good vs Bad accuracy per product category.

### Sharing on GitHub
1. Push `app.py`, `train.py`, `index.html`, `requirements.txt`, `summary.md` to GitHub.
2. The weights file `best_exact_class_model.pth` (~100MB) requires **Git LFS**:
   ```bash
   git lfs install
   git lfs track "*.pth"
   git add .gitattributes
   git add best_exact_class_model.pth
   git commit -m "Add trained model weights"
   git push
   ```
3. Others clone and run:
   ```bash
   pip install -r requirements.txt
   python app.py
   ```
   No need to train again. Model weights load directly.
