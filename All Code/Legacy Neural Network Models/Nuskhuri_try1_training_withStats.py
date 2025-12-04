import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torchvision.datasets.folder import default_loader
from torch.utils.data import DataLoader, Dataset, random_split

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, classification_report
import pandas as pd

# ----------------------------
# ✅ CONFIGURATION
# ----------------------------
lr = 0.001043
weight_decay = 0.0002131
dropout_rate = 0.18
f0 = 25
num_levels = 4
blocks_per_level = 2
batch_size = 16
num_epochs = 10

save_path = r"C:\Users\Sandro\repo\Nuskhuri Data\Neural Networks\Statistics Model"
data_path = r"C:\Users\Sandro\repo\Nuskhuri Data\Sorted"

os.makedirs(save_path, exist_ok=True)  # make sure directory exists

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ----------------------------
# 🧠 DYNAMIC MODEL
# ----------------------------
class DynamicCNN(nn.Module):
    def __init__(self, input_channels, num_classes, num_levels, blocks_per_level, filters, dropout_rate):
        super(DynamicCNN, self).__init__()
        layers = []
        in_channels = input_channels

        for level in range(num_levels):
            out_channels = filters[level]
            for _ in range(blocks_per_level):
                layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
                layers.append(nn.ReLU())
                layers.append(nn.BatchNorm2d(out_channels))
                in_channels = out_channels
            if level != num_levels - 1:
                layers.append(nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=2, padding=1))
                layers.append(nn.ReLU())
                layers.append(nn.BatchNorm2d(in_channels))

        self.features = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Flatten(),
            nn.Linear(in_channels, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        x = self.classifier(x)
        return x

# ----------------------------
# 🎨 TRANSFORMS
# ----------------------------
train_transform = transforms.Compose([
    transforms.Grayscale(1),
    transforms.Resize((64, 64)),
    transforms.RandomCrop(64, padding=4),
    transforms.RandomRotation((5, 10)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

test_transform = transforms.Compose([
    transforms.Grayscale(1),
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# ----------------------------
# 📦 DATASET
# ----------------------------
base_dataset = datasets.ImageFolder(data_path)
total_size = len(base_dataset)
train_size = int(0.8 * total_size)
test_size = total_size - train_size
train_indices, test_indices = torch.utils.data.random_split(range(total_size), [train_size, test_size])

class CustomSubset(Dataset):
    def __init__(self, dataset, indices, transform):
        self.dataset = dataset
        self.indices = indices
        self.transform = transform
        self.loader = default_loader

    def __getitem__(self, idx):
        path, label = self.dataset.samples[self.indices[idx]]
        image = self.loader(path)
        image = self.transform(image)
        return image, label

    def __len__(self):
        return len(self.indices)

train_set = CustomSubset(base_dataset, train_indices, train_transform)
test_set = CustomSubset(base_dataset, test_indices, test_transform)

train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)
num_classes = len(base_dataset.classes)

# ----------------------------
# 🚂 MODEL / OPTIMIZER
# ----------------------------
filters = [f0 * (2 ** i) for i in range(num_levels)]
model = DynamicCNN(1, num_classes, num_levels, blocks_per_level, filters, dropout_rate).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

# For logging
train_losses = []
val_losses = []
train_accuracies = []
val_accuracies = []

best_val_acc = 0.0
best_model_path = os.path.join(save_path, "best_model.pth")

# ----------------------------
# 🚂 TRAINING LOOP
# ----------------------------
for epoch in range(num_epochs):
    print(f"\n===== Epoch {epoch+1}/{num_epochs} =====")

    # ---- TRAIN ----
    model.train()
    running_loss = 0.0
    total, correct = 0, 0

    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()

        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()

        _, pred = torch.max(out, 1)
        correct += (pred == y).sum().item()
        total += y.size(0)
        running_loss += loss.item() * y.size(0)

    train_loss = running_loss / total
    train_acc = 100.0 * correct / total
    train_losses.append(train_loss)
    train_accuracies.append(train_acc)

    # ---- VALIDATION ----
    model.eval()
    val_running_loss = 0.0
    val_total, val_correct = 0, 0

    with torch.no_grad():
        for x, y in test_loader:
            # If you still want raw stats, uncomment:
            # raw_x = x * 0.5 + 0.5
            # print(f"[Stats] Min: {raw_x.min():.4f}, Max: {raw_x.max():.4f}, Mean: {raw_x.mean():.4f}")

            x, y = x.to(device), y.to(device)
            out = model(x)
            loss = criterion(out, y)

            _, pred = torch.max(out, 1)
            val_correct += (pred == y).sum().item()
            val_total += y.size(0)
            val_running_loss += loss.item() * y.size(0)

    val_loss = val_running_loss / val_total
    val_acc = 100.0 * val_correct / val_total
    val_losses.append(val_loss)
    val_accuracies.append(val_acc)

    scheduler.step()

    print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
    print(f" Val  Loss: {val_loss:.4f} |  Val  Acc: {val_acc:.2f}%")

    # Save best model by val accuracy
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), best_model_path)
        print(f"✅ Best model updated (Val Acc: {val_acc:.2f}%)")

print(f"\n🏁 Training complete. Best Val Acc: {best_val_acc:.2f}% (saved to {best_model_path})")

# ----------------------------
# 📈 PLOT TRAINING CURVES
# ----------------------------
epochs_range = range(1, num_epochs + 1)

# Loss curve
plt.figure(figsize=(6, 4))
plt.plot(epochs_range, train_losses, label="Train Loss")
plt.plot(epochs_range, val_losses, label="Val Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training and Validation Loss (Asomtavruli Try7)")
plt.legend()
plt.grid(True)
plt.tight_layout()
loss_fig_path = os.path.join(save_path, "training_validation_loss.png")
plt.savefig(loss_fig_path, dpi=300, bbox_inches="tight")
plt.close()

# Accuracy curve
plt.figure(figsize=(6, 4))
plt.plot(epochs_range, train_accuracies, label="Train Accuracy")
plt.plot(epochs_range, val_accuracies, label="Val Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy (%)")
plt.title("Training and Validation Accuracy (Asomtavruli Try7)")
plt.legend()
plt.grid(True)
plt.tight_layout()
acc_fig_path = os.path.join(save_path, "training_validation_accuracy.png")
plt.savefig(acc_fig_path, dpi=300, bbox_inches="tight")
plt.close()

print(f"📊 Saved training curves to:\n  {loss_fig_path}\n  {acc_fig_path}")

# ----------------------------
# 🔍 FINAL EVAL: CONFUSION MATRIX & METRICS
# ----------------------------

# Load best model for evaluation (optional but good practice)
model.load_state_dict(torch.load(best_model_path, map_location=device))
model.eval()

all_labels = []
all_preds = []

with torch.no_grad():
    for x, y in test_loader:
        x, y = x.to(device), y.to(device)
        out = model(x)
        _, pred = torch.max(out, 1)

        all_labels.extend(y.cpu().numpy())
        all_preds.extend(pred.cpu().numpy())

all_labels = np.array(all_labels)
all_preds = np.array(all_preds)

# Confusion matrix
cm = confusion_matrix(all_labels, all_preds, labels=list(range(num_classes)))

# Classification report (printed)
print("\n=== Classification Report (per class) ===")
print(classification_report(
    all_labels,
    all_preds,
    target_names=base_dataset.classes,
    digits=4
))

# Precision/Recall/F1 per class into CSV
precision, recall, f1, support = precision_recall_fscore_support(
    all_labels, all_preds,
    labels=list(range(num_classes)),
    zero_division=0
)

metrics_df = pd.DataFrame({
    "class_index": list(range(num_classes)),
    "class_name": base_dataset.classes,
    "precision": precision,
    "recall": recall,
    "f1_score": f1,
    "support": support
})

metrics_csv_path = os.path.join(save_path, "metrics_per_class.csv")
metrics_df.to_csv(metrics_csv_path, index=False)
print(f"📄 Saved per-class metrics table to: {metrics_csv_path}")

# Overall macro metrics (for table caption / text)
macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
    all_labels, all_preds,
    average="macro",
    zero_division=0
)
overall_acc = (all_labels == all_preds).mean()

print("\n=== Overall Metrics (macro-averaged) ===")
print(f"Accuracy:  {overall_acc:.4f}")
print(f"Precision: {macro_precision:.4f}")
print(f"Recall:    {macro_recall:.4f}")
print(f"F1-score:  {macro_f1:.4f}")

# Plot confusion matrix
fig, ax = plt.subplots(figsize=(8, 8))
im = ax.imshow(cm, interpolation="nearest")
fig.colorbar(im, ax=ax)

ax.set_xticks(np.arange(num_classes))
ax.set_yticks(np.arange(num_classes))
ax.set_xticklabels(base_dataset.classes, rotation=90)
ax.set_yticklabels(base_dataset.classes)

ax.set_xlabel("Predicted label")
ax.set_ylabel("True label")
ax.set_title("Confusion Matrix (Asomtavruli Try7)")

plt.tight_layout()
cm_fig_path = os.path.join(save_path, "confusion_matrix.png")
plt.savefig(cm_fig_path, dpi=300, bbox_inches="tight")
plt.close()

print(f"🧩 Saved confusion matrix figure to: {cm_fig_path}")
