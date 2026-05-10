import os
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset, WeightedRandomSampler
from torchvision import transforms, models
from PIL import Image
from collections import Counter
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, classification_report

torch.set_num_threads(os.cpu_count() or 4)

# ── Device selection ──────────────────────────────────────────────────────────
# Priority: DirectML (AMD/Intel GPU on Windows) → CUDA (NVIDIA) → CPU
#
# REQUIREMENT: torch-directml needs Python ≤ 3.12
#   Setup steps:
#     1. Install Python 3.12  (python.org)
#     2. python3.12 -m venv venv && venv\Scripts\activate
#     3. pip install torch torchvision torch-directml
#     4. Re-run train.py → will auto-use AMD Radeon 780M via DirectX 12
#
# ── CPU-ONLY MODE (uncomment block below, comment out rest) ──────────────────
# device = torch.device("cpu")
# torch.set_num_threads(os.cpu_count() or 4)
# print(f"Device: CPU ({os.cpu_count()} threads)")
# ─────────────────────────────────────────────────────────────────────────────

try:
    import torch_directml
    device = torch_directml.device()
    print(f"Device: DirectML — AMD/Intel GPU via DirectX 12")
except ImportError:
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Device: CUDA — {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print(f"Device: CPU ({os.cpu_count()} threads) — install Python 3.12 + torch-directml for GPU")
# ─────────────────────────────────────────────────────────────────────────────

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "archive")

# ── Hyperparameters ─────────────────────────────────────────────────────────
# BATCH_SIZE=16: Small batch → less GPU memory → no TDR crash on iGPU.
#   Effect: Noisier gradient updates, but often better generalization.
#   Larger batch (32-64) = faster per-epoch but crashes on Radeon 780M.
BATCH_SIZE = 16

# EPOCHS=50: Max training rounds. Early stopping (PATIENCE) prevents wasting time.
EPOCHS = 50

# LR_FC: Learning rate for the new classification head (randomly initialized).
#   High LR → head adapts fast to MVTec 30-class task.
#   Effect: Head learns correct classes quickly in first few epochs.
LR_FC = 5e-4

# LR_BACKBONE: Learning rate for ResNet50 backbone (pretrained on ImageNet).
#   Very low to avoid destroying ImageNet features (catastrophic forgetting).
#   Effect: Slowly fine-tunes texture/edge detectors for industrial defects.
LR_BACKBONE = 1e-5

# PATIENCE=15: If val accuracy doesn't improve for 15 epochs → stop training.
#   Effect: Prevents wasted compute and overfitting after plateau.
PATIENCE = 15

# WARMUP_EPOCHS=3: LR starts very low and ramps up over 3 epochs.
#   Effect: Prevents large gradient shocks in epoch 1 which can corrupt weights.
WARMUP_EPOCHS = 3

CATEGORIES = [
    'bottle', 'cable', 'capsule', 'carpet', 'grid',
    'hazelnut', 'leather', 'metal_nut', 'pill', 'screw',
    'tile', 'toothbrush', 'transistor', 'wood', 'zipper'
]
QUALITY_TYPES = ['good', 'bad']
NUM_CLASSES = len(CATEGORIES) * 2  # 30
CLASSES = [f"{cat}_{qual}" for cat in CATEGORIES for qual in QUALITY_TYPES]
CLASS_TO_IDX = {cls: idx for idx, cls in enumerate(CLASSES)}


# ── Dataset Class ────────────────────────────────────────────────────────────
# MVTecDataset reads the archive/ folder and builds (image_path, class_idx) pairs.
# MVTec AD folder layout:
#   archive/<category>/train/good/   → label = <category>_good
#   archive/<category>/test/good/    → label = <category>_good
#   archive/<category>/test/<defect>/ → label = <category>_bad
# ALL defect sub-types (scratch, bent, contamination...) collapse into one "bad" label.
# This is why screw_manipulated_front is hard — it's mixed with 4 other defect types.
class MVTecDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.transform = transform
        self.samples = []  # list of (image_path, class_index)

        for category in CATEGORIES:
            cat_path = os.path.join(root_dir, category)
            if not os.path.exists(cat_path):
                print(f"  [MISSING] {category}")
                continue

            # Collect GOOD images from both train/ and test/ splits.
            # Effect: Maximizes good sample count → better boundary between good/bad.
            for split in ['train', 'test']:
                good_path = os.path.join(cat_path, split, 'good')
                if os.path.exists(good_path):
                    for img in os.listdir(good_path):
                        if img.lower().endswith(('.png', '.jpg', '.jpeg')):
                            self.samples.append((
                                os.path.join(good_path, img),
                                CLASS_TO_IDX[f"{category}_good"]
                            ))

            # Collect BAD images — every subfolder in test/ except 'good'.
            # All defect subtypes merged → single bad label per category.
            test_path = os.path.join(cat_path, 'test')
            if os.path.exists(test_path):
                for defect in os.listdir(test_path):
                    if defect.lower() == 'good':
                        continue
                    defect_path = os.path.join(test_path, defect)
                    if not os.path.isdir(defect_path):
                        continue
                    for img in os.listdir(defect_path):
                        if img.lower().endswith(('.png', '.jpg', '.jpeg')):
                            self.samples.append((
                                os.path.join(defect_path, img),
                                CLASS_TO_IDX[f"{category}_bad"]
                            ))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, class_idx = self.samples[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, class_idx


# ── MixUp Augmentation ───────────────────────────────────────────────────────
# MixUp blends two random training images and their labels together:
#   mixed_image = lam * image_A + (1-lam) * image_B
#   loss = lam * loss(pred, label_A) + (1-lam) * loss(pred, label_B)
# Effect: Forces model to learn smooth decision boundaries instead of sharp ones.
#   Reduces overconfidence. Makes training harder → trains accuracy appears lower
#   (e.g. 84% train vs 95% val) but val accuracy is genuinely higher.
# Used from epoch 3 onward (after warmup stabilizes weights).
def mixup_data(x, y, alpha=0.3):
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1  # blend ratio from Beta distribution
    idx = torch.randperm(x.size(0)).to(device)              # random pair within batch
    return lam * x + (1 - lam) * x[idx], y, y[idx], lam


def mixup_loss(criterion, pred, ya, yb, lam):
    # Weighted combination of losses for both labels in the blended image
    return lam * criterion(pred, ya) + (1 - lam) * criterion(pred, yb)


# ── Data Augmentation ────────────────────────────────────────────────────────
# Applied ONLY to training set. Val/test use clean center-crop for fair evaluation.
# Each augmentation prevents a specific type of overfitting:
train_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomCrop(224),          # Random position crop → position invariance
    transforms.RandomHorizontalFlip(),   # Mirror → halves need for mirrored samples
    transforms.RandomVerticalFlip(p=0.3),# Vertical flip (useful for textures like carpet)
    transforms.RandomRotation(15),       # ±15° rotation → orientation robustness
    transforms.ColorJitter(              # Random color distortion → lighting invariance
        brightness=0.4, contrast=0.4, saturation=0.3, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize(                # ImageNet mean/std — MUST match pretrained weights
        [0.485, 0.456, 0.406],           # Effect: Centers pixel distribution → stable training
        [0.229, 0.224, 0.225]),
    transforms.RandomErasing(p=0.25,     # Randomly erase 2-20% of image
        scale=(0.02, 0.2))               # Effect: Forces model to not rely on single region
])

# Validation transform: deterministic, no randomness → reproducible metric
val_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),          # Center crop (not random) for consistent eval
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

print("Loading dataset...")
base_dataset = MVTecDataset(DATA_DIR, transform=None)
if len(base_dataset) == 0:
    raise ValueError(f"No images found in {DATA_DIR}")

targets = [s[1] for s in base_dataset.samples]
counts = Counter(targets)
print(f"Total: {len(base_dataset)} images | Classes: {len(counts)}/{NUM_CLASSES}")
for cls_name, cls_idx in CLASS_TO_IDX.items():
    c = counts.get(cls_idx, 0)
    print(f"  {cls_name:25}: {c:4d}")

train_idx, val_idx = train_test_split(
    np.arange(len(targets)), test_size=0.2, stratify=targets, random_state=42
)

train_dataset = Subset(MVTecDataset(DATA_DIR, transform=train_transform), train_idx)
val_dataset   = Subset(MVTecDataset(DATA_DIR, transform=val_transform),   val_idx)

# ── WeightedRandomSampler ─────────────────────────────────────────────────────
# Problem: toothbrush_bad has 30 images, hazelnut_good has 431.
# Without balancing → model sees hazelnut_good 14x more often → biased predictions.
# Fix: Assign each sample weight = 1 / class_count.
#   Rare classes (toothbrush_bad) get high weight → sampled more often.
#   Common classes (hazelnut_good) get low weight → sampled less often.
# Effect: Model trains on balanced classes → anomaly F1 improves significantly.
# CRITICAL LINE — removing this causes bad classes to be predicted as good.
train_labels = [targets[i] for i in train_idx]
train_counts = Counter(train_labels)
sample_weights = [1.0 / train_counts[targets[i]] for i in train_idx]
sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)

# num_workers=0: Windows multiprocessing spawn bug with DirectML — must stay 0.
# Effect: Data loads on main thread → slightly slower but crash-free.
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=sampler, num_workers=0)
val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False,   num_workers=0)

# ── Model Architecture ───────────────────────────────────────────────────────
# ResNet50 pretrained on ImageNet (1.28M images, 1000 classes).
# IMAGENET1K_V2 = best available weights (trained with modern augmentation).
# Effect: Model already knows edges, textures, shapes → needs only fine-tuning.
print("\nBuilding ResNet50...")
model = models.resnet50(weights='IMAGENET1K_V2')

# Unfreeze ALL layers → backbone + head all learn together.
# Trade-off: Slower convergence but higher accuracy ceiling on complex textures.
# requires_grad=True means gradients flow through entire network during backprop.
for param in model.parameters():
    param.requires_grad = True

# Replace the original 1000-class head with a custom 30-class MLP.
# Architecture: 2048 → 512 → 256 → 30
#   BatchNorm1d(512): Normalizes activations → faster convergence, stable training.
#   Dropout(0.4): Randomly zeros 40% neurons → prevents head from memorizing train set.
#   Dropout(0.2): Lighter dropout before final layer → small regularization.
# Effect: Without this, model would output 1000 ImageNet classes instead of 30.
model.fc = nn.Sequential(
    nn.Linear(model.fc.in_features, 512),  # 2048 ResNet features → 512
    nn.BatchNorm1d(512),                    # Stabilizes training, +1-2% accuracy
    nn.ReLU(),
    nn.Dropout(0.4),                        # Heavy dropout → reduces overfitting
    nn.Linear(512, 256),
    nn.ReLU(),
    nn.Dropout(0.2),
    nn.Linear(256, NUM_CLASSES)             # Final: 256 → 30 class scores (logits)
)
model = model.to(device)

# ── Optimizer ────────────────────────────────────────────────────────────────
# Split parameter groups: backbone uses slow LR, head uses fast LR.
# Reason: Head is random (needs fast learning). Backbone is pretrained (needs gentle nudge).
# AdamW = Adam + weight decay decoupled. Better generalization than plain Adam.
# weight_decay=1e-4: L2 regularization → shrinks large weights → prevents overfit.
optimizer = optim.AdamW([
    {'params': [p for n, p in model.named_parameters() if not n.startswith('fc.')], 'lr': LR_BACKBONE},
    {'params': model.fc.parameters(), 'lr': LR_FC}
], weight_decay=1e-4)

# ── Loss Function ─────────────────────────────────────────────────────────────
# anomaly_weights: Bad classes get 2x loss penalty.
# Reason: Missing a defect (false negative) is worse than false alarm.
#   e.g. toothbrush_bad misclassified as good → product ships defective → costly.
# label_smoothing=0.1: Softens hard labels [0,0,1,...] → [0.003,0.003,0.9,...].
#   Effect: Prevents model from being 100% confident → better calibration.
# Combined effect: Model is more aggressive about catching defects.
anomaly_weights = torch.ones(NUM_CLASSES)
for i, cls in enumerate(CLASSES):
    if cls.endswith('_bad'):
        anomaly_weights[i] = 2.0   # Bad class = 2x loss penalty
criterion = nn.CrossEntropyLoss(label_smoothing=0.1, weight=anomaly_weights.to(device))

# ── LR Scheduler: Warmup + Cosine Decay ──────────────────────────────────────
# Phase 1 (epochs 0-2): Warmup — LR ramps from 0 to full value.
#   Reason: Epoch 1 with full LR can corrupt pretrained weights (large gradient step).
# Phase 2 (epoch 3+): Cosine decay — LR smoothly decreases toward 0.
#   Effect: Model makes large updates early, fine adjustments late → better final accuracy.
# Combined: Prevents early instability + avoids oscillating around optimal weights.
def lr_lambda(epoch):
    if epoch < WARMUP_EPOCHS:  # Linear ramp-up
        return (epoch + 1) / WARMUP_EPOCHS
    progress = (epoch - WARMUP_EPOCHS) / max(1, EPOCHS - WARMUP_EPOCHS)
    return 0.5 * (1.0 + np.cos(np.pi * progress))  # Cosine curve: 1.0 → 0.0
scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

train_losses, val_losses, train_accs, val_accs = [], [], [], []
best_val_acc = 0.0
patience_counter = 0

print(f"Training {EPOCHS} epochs | train={len(train_dataset)} val={len(val_dataset)}\n")

for epoch in range(EPOCHS):
    model.train()
    t_loss, t_corr, t_tot = 0.0, 0, 0
    use_mixup = epoch >= 3

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()

        if use_mixup and random.random() > 0.5:
            images, ya, yb, lam = mixup_data(images, labels)
            out = model(images)
            loss = mixup_loss(criterion, out, ya, yb, lam)
            _, pred = torch.max(out, 1)
            t_corr += (lam*(pred==ya).float() + (1-lam)*(pred==yb).float()).sum().item()
        else:
            out = model(images)
            loss = criterion(out, labels)
            _, pred = torch.max(out, 1)
            t_corr += (pred == labels).sum().item()

        loss.backward()  # Compute gradients for all trainable parameters
        # Gradient clipping: cap gradient norm at 1.0
        # Effect: Prevents exploding gradients (sudden huge weight update that breaks training)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()  # Update weights using computed gradients
        t_loss += loss.item()
        t_tot  += labels.size(0)

    train_losses.append(t_loss / t_tot)
    train_accs.append(100 * t_corr / t_tot)

    model.eval()
    v_loss, v_corr, v_tot = 0.0, 0, 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            out  = model(images)
            loss = criterion(out, labels)
            _, pred = torch.max(out, 1)
            v_loss += loss.item()
            v_corr += (pred == labels).sum().item()
            v_tot  += labels.size(0)

    val_acc = 100 * v_corr / v_tot
    val_losses.append(v_loss / v_tot)
    val_accs.append(val_acc)
    scheduler.step()

    print(f"Epoch {epoch+1:2}/{EPOCHS} | T_Loss:{train_losses[-1]:.4f} T_Acc:{train_accs[-1]:5.1f}% | V_Loss:{val_losses[-1]:.4f} V_Acc:{val_acc:5.1f}%")

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        patience_counter = 0
        torch.save(model.state_dict(), os.path.join(BASE_DIR, 'best_exact_class_model.pth'))
        print(f"  *** Best: {val_acc:.2f}% saved ***")
    else:
        patience_counter += 1
        if patience_counter >= PATIENCE:
            print(f"\nEarly stop at epoch {epoch+1}")
            break

print(f"\nDone! Best val acc: {best_val_acc:.2f}%")

# Evaluation
print("\n" + "="*50)
model.load_state_dict(torch.load(
    os.path.join(BASE_DIR, 'best_exact_class_model.pth'),
    weights_only=False, map_location='cpu'
))
model = model.to(device)
model.eval()
all_preds, all_labels = [], []
with torch.no_grad():
    for images, labels in val_loader:
        out = model(images.to(device))
        _, pred = torch.max(out, 1)
        all_preds.extend(pred.cpu().numpy())
        all_labels.extend(labels.numpy())

all_preds  = np.array(all_preds)
all_labels = np.array(all_labels)
print(f"Overall Accuracy: {(all_preds == all_labels).mean()*100:.2f}%")

# Anomaly detection metric: good vs bad (binary)
binary_true = np.array([1 if CLASSES[l].endswith('_bad') else 0 for l in all_labels])
binary_pred = np.array([1 if CLASSES[p].endswith('_bad') else 0 for p in all_preds])
anomaly_f1 = f1_score(binary_true, binary_pred, zero_division=0)
print(f"Anomaly Detection F1: {anomaly_f1*100:.2f}%  (good vs bad, binary)\n")

seen_idx = set()
for idx, cls_name in enumerate(CLASSES):
    if idx in seen_idx:
        continue
    seen_idx.add(idx)
    mask = all_labels == idx
    if mask.sum() > 0:
        acc = (all_preds[mask] == all_labels[mask]).mean() * 100
        print(f"  {cls_name:25}: {acc:5.1f}%  ({mask.sum()} samples)")

# ── Rich Visualization Dashboard ─────────────────────────────────────────────
import seaborn as sns
from sklearn.metrics import confusion_matrix

epochs_range = range(1, len(train_accs) + 1)

fig = plt.figure(figsize=(22, 18))
fig.suptitle('MVTec AD ResNet50 — Training Dashboard', fontsize=16, fontweight='bold', y=1.01)

# ── 1. Accuracy Curve ─────────────────────────────────────────────────────────
ax1 = fig.add_subplot(3, 3, 1)
ax1.plot(epochs_range, train_accs, 'b-o', label='Train', markersize=4)
ax1.plot(epochs_range, val_accs,   'r-s', label='Val',   markersize=4)
ax1.axhline(best_val_acc, color='green', linestyle='--', linewidth=1, label=f'Best {best_val_acc:.1f}%')
ax1.set_title('Accuracy per Epoch'); ax1.set_xlabel('Epoch'); ax1.set_ylabel('Accuracy %')
ax1.legend(); ax1.grid(True, alpha=0.4)

# ── 2. Loss Curve ─────────────────────────────────────────────────────────────
ax2 = fig.add_subplot(3, 3, 2)
ax2.plot(epochs_range, train_losses, 'b-o', label='Train', markersize=4)
ax2.plot(epochs_range, val_losses,   'r-s', label='Val',   markersize=4)
ax2.set_title('Loss per Epoch'); ax2.set_xlabel('Epoch'); ax2.set_ylabel('Loss')
ax2.legend(); ax2.grid(True, alpha=0.4)

# ── 3. Accuracy Gap (Overfit detector) ───────────────────────────────────────
ax3 = fig.add_subplot(3, 3, 3)
gap = [t - v for t, v in zip(train_accs, val_accs)]
ax3.bar(epochs_range, gap, color=['red' if g > 5 else 'steelblue' for g in gap])
ax3.axhline(0, color='black', linewidth=0.8)
ax3.set_title('Train−Val Gap (Overfit Monitor)'); ax3.set_xlabel('Epoch'); ax3.set_ylabel('Gap %')
ax3.grid(True, alpha=0.4)

# ── 4. Confusion Matrix (30 classes, normalized) ─────────────────────────────
ax4 = fig.add_subplot(3, 3, 4)
cm = confusion_matrix(all_labels, all_preds, normalize='true')
sns.heatmap(cm, ax=ax4, cmap='Blues', xticklabels=False, yticklabels=False,
            linewidths=0, cbar=True)
ax4.set_title('Confusion Matrix (normalised)'); ax4.set_xlabel('Predicted'); ax4.set_ylabel('True')

# ── 5. Anomaly Binary Confusion Matrix ───────────────────────────────────────
ax5 = fig.add_subplot(3, 3, 5)
binary_cm = confusion_matrix(binary_true, binary_pred)
sns.heatmap(binary_cm, annot=True, fmt='d', cmap='Oranges', ax=ax5,
            xticklabels=['Pred Good', 'Pred Bad'],
            yticklabels=['True Good', 'True Bad'])
ax5.set_title(f'Anomaly Detection Confusion\nF1={anomaly_f1*100:.1f}%')

# ── 6. Per-Class Accuracy Bar ─────────────────────────────────────────────────
ax6 = fig.add_subplot(3, 3, 6)
cls_accs = []
for idx in range(NUM_CLASSES):
    mask = all_labels == idx
    acc = (all_preds[mask] == all_labels[mask]).mean() * 100 if mask.sum() > 0 else 0
    cls_accs.append(acc)
colors = ['tomato' if a < 90 else 'steelblue' for a in cls_accs]
ax6.barh(CLASSES, cls_accs, color=colors)
ax6.axvline(90, color='orange', linestyle='--', linewidth=1, label='90% line')
ax6.axvline(100, color='green', linestyle='--', linewidth=1, label='100% line')
ax6.set_title('Per-Class Accuracy'); ax6.set_xlabel('Accuracy %'); ax6.set_xlim(0, 105)
ax6.legend(fontsize=8); ax6.grid(True, alpha=0.4, axis='x')

# ── 7. Good vs Bad Accuracy per Category ─────────────────────────────────────
ax7 = fig.add_subplot(3, 1, 3)
cat_good_acc, cat_bad_acc = [], []
for cat in CATEGORIES:
    for qual, store in [('good', cat_good_acc), ('bad', cat_bad_acc)]:
        idx = CLASS_TO_IDX[f'{cat}_{qual}']
        mask = all_labels == idx
        acc = (all_preds[mask] == all_labels[mask]).mean() * 100 if mask.sum() > 0 else 0
        store.append(acc)
x = np.arange(len(CATEGORIES))
width = 0.35
ax7.bar(x - width/2, cat_good_acc, width, label='Good', color='steelblue', alpha=0.8)
ax7.bar(x + width/2, cat_bad_acc,  width, label='Bad (Anomaly)', color='tomato', alpha=0.8)
ax7.set_xticks(x); ax7.set_xticklabels(CATEGORIES, rotation=45, ha='right')
ax7.set_title('Good vs Bad Accuracy per Product Category')
ax7.set_ylabel('Accuracy %'); ax7.set_ylim(0, 110)
ax7.axhline(90, color='orange', linestyle='--', linewidth=1)
ax7.legend(); ax7.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
save_path = os.path.join(BASE_DIR, 'training_dashboard.png')
plt.savefig(save_path, dpi=150, bbox_inches='tight')
print(f"\nDashboard saved → {save_path}")
plt.show()