import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms, models
from PIL import Image
from collections import Counter
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

torch.set_num_threads(14) 
device = torch.device("cpu")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "archive")

BATCH_SIZE = 16
EPOCHS = 20
LEARNING_RATE = 1e-3

CATEGORIES = ['bottle', 'BSD', 'cable', 'capsule', 'carpet', 'grid', 'hazelnut']
QUALITY_TYPES = ['good', 'bad']
CLASSES_14 = [f"{cat}_{qual}" for cat in CATEGORIES for qual in QUALITY_TYPES]

class MVTecDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.samples = []
        CLASS_TO_IDX = {cls_name: idx for idx, cls_name in enumerate(CLASSES_14)}
        
        for category in CATEGORIES:
            cat_path = os.path.join(root_dir, category)
            if not os.path.exists(cat_path): continue
            
            for quality_folder in os.listdir(cat_path):
                qual_path = os.path.join(cat_path, quality_folder)
                if not os.path.isdir(qual_path): continue
                
                q_type = 'good' if quality_folder.lower() == 'good' else 'bad'
                class_idx = CLASS_TO_IDX[f"{category}_{q_type}"]
                
                for img_name in os.listdir(qual_path):
                    if img_name.lower().endswith(('.png')):
                        self.samples.append((os.path.join(qual_path, img_name), class_idx))
                        
    def __len__(self): return len(self.samples)
    def __getitem__(self, idx):
        img_path, class_idx = self.samples[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform: image = self.transform(image)
        return image, class_idx

train_transform = transforms.Compose([
    transforms.Resize(256), transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(), transforms.RandomVerticalFlip(p=0.3),
    transforms.ColorJitter(0.2, 0.2, 0.2), transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

print("Loading dataset...")
base_dataset = MVTecDataset(DATA_DIR, transform=None)
if len(base_dataset) == 0: raise ValueError(f"No images found in {DATA_DIR}")
print(f"Found {len(base_dataset)} total images.")

# STRATIFIED SPLIT (Ensures equal good/bad ratio in train and val)
targets = [s[1] for s in base_dataset.samples]
train_idx, val_idx = train_test_split(np.arange(len(targets)), test_size=0.2, stratify=targets, random_state=42)
train_dataset = Subset(MVTecDataset(DATA_DIR, transform=train_transform), train_idx)
val_dataset = Subset(MVTecDataset(DATA_DIR, transform=val_transform), val_idx)

# WEIGHTS FOR 14 CLASSES
def get_class_weights(indices):
    labels = [base_dataset.samples[i][1] for i in indices]
    counts = Counter(labels)
    weights = torch.zeros(14)
    for cls in range(14): weights[cls] = len(labels) / (14 * counts.get(cls, 1))
    return weights

class_weights = get_class_weights(train_idx)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

# MODEL WITH SMART HEAD (14 OUTPUTS)
print("Initializing ResNet50...")
model = models.resnet50(weights='IMAGENET1K_V2')
for param in model.parameters(): param.requires_grad = False

model.fc = nn.Sequential(
    nn.Linear(model.fc.in_features, 512),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(512, 14) # 14 CLASSES
)
model = model.to(device)

criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
optimizer = optim.AdamW(model.fc.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

train_losses, val_losses, train_accs, val_accs = [], [], [], []

print(f"Starting training for {EPOCHS} epochs...")
best_val_acc = 0.0

for epoch in range(EPOCHS):
    model.train()
    t_loss, t_corr, t_tot = 0.0, 0, 0
    # FIXED: Unpacking only 2 variables now
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        t_loss += loss.item()
        _, pred = torch.max(outputs, 1)
        t_tot += labels.size(0)
        t_corr += (pred == labels).sum().item()

    train_losses.append(t_loss / t_tot)
    train_accs.append(100 * t_corr / t_tot)

    model.eval()
    v_loss, v_corr, v_tot = 0.0, 0, 0
    with torch.no_grad():
        # FIXED: Unpacking only 2 variables now
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            v_loss += loss.item()
            _, pred = torch.max(outputs, 1)
            v_tot += labels.size(0)
            v_corr += (pred == labels).sum().item()

    val_losses.append(v_loss / v_tot)
    val_acc = 100 * v_corr / v_tot
    val_accs.append(val_acc)
    scheduler.step()
    
    print(f"Epoch {epoch+1:2}/{EPOCHS} | T_Loss: {train_losses[-1]:.4f} | V_Loss: {val_losses[-1]:.4f} | T_Acc: {train_accs[-1]:5.1f}% | V_Acc: {val_acc:5.1f}%")
    
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), 'best_exact_class_model.pth')
        print(f"  *** New Best: {val_acc:.1f}% ***")

print("\nTraining complete!")

# EVALUATION
print("\n" + "="*40)
print("FINAL EVALUATION ON 14 CLASSES")
print("="*40)

# FIXED: Added weights_only=True
model.load_state_dict(torch.load('best_exact_class_model.pth', weights_only=True, map_location=device))
model.eval()

all_preds, all_labels = [], []
with torch.no_grad():
    for images, labels in val_loader:
        outputs = model(images.to(device))
        _, pred = torch.max(outputs, 1)
        all_preds.extend(pred.cpu().numpy())
        all_labels.extend(labels.numpy())

all_preds, all_labels = np.array(all_preds), np.array(all_labels)
print(f"Exact 14-Class Accuracy: {(all_preds == all_labels).mean()*100:.2f}%")
print("Per-Item Breakdown:")
for idx, cls_name in enumerate(CLASSES_14):
    mask = all_labels == idx
    if mask.sum() > 0:
        print(f"  {cls_name:15} : {(all_preds[mask] == all_labels[mask]).mean()*100:5.1f}%")

# PLOTTING
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(range(1, EPOCHS+1), train_accs, label='Train', marker='o')
plt.plot(range(1, EPOCHS+1), val_accs, label='Val', marker='s')
plt.title('Accuracy'); plt.xlabel('Epoch'); plt.ylabel('%'); plt.legend(); plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(range(1, EPOCHS+1), train_losses, label='Train', marker='o')
plt.plot(range(1, EPOCHS+1), val_losses, label='Val', marker='s')
plt.title('Loss'); plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.legend(); plt.grid(True)

plt.tight_layout()
plt.savefig('training_graphs.png')
print("\nGraph saved as 'training_graphs.png'")
plt.show()