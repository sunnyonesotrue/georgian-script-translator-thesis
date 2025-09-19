#NOTE: this is a try7 fork

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torchvision.datasets.folder import default_loader
from torch.utils.data import DataLoader, Dataset, random_split

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
num_epochs = 40
save_path = r"C:\Users\TOTORO\OneDrive\Documents\GitHub\Project-Lab---Asomtavruli-OCR\All Code\Neural Networks\Saves\best_dynamic_model_try10"
data_path = r"C:\Users\TOTORO\OneDrive\Documents\GitHub\Project-Lab---Asomtavruli-OCR\Asomtavruli data\Sorted"

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
# 🚂 TRAINING
# ----------------------------
filters = [f0 * (2 ** i) for i in range(num_levels)]
model = DynamicCNN(1, num_classes, num_levels, blocks_per_level, filters, dropout_rate).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

best_val_acc = 0.0

for epoch in range(num_epochs):
    model.train()
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
    scheduler.step()
    train_acc = 100 * correct / total

    # ------------------------
    # 🧪 VALIDATION + STATS
    # ------------------------
    model.eval()
    val_total, val_correct = 0, 0
    with torch.no_grad():
        for x, y in test_loader:
            # 🔍 Print unnormalized statistics
            raw_x = x * 0.5 + 0.5  # Reverse Normalize((0.5,), (0.5,))
            print(f"[Stats] Min: {raw_x.min():.4f}, Max: {raw_x.max():.4f}, Mean: {raw_x.mean():.4f}")

            x, y = x.to(device), y.to(device)
            out = model(x)
            _, pred = torch.max(out, 1)
            val_correct += (pred == y).sum().item()
            val_total += y.size(0)

    val_acc = 100 * val_correct / val_total
    print(f"Epoch {epoch+1}/{num_epochs} | Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}%")

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), save_path + f"_{val_acc:.2f}.pth")
        print(f"✅ Model saved (Val Acc: {val_acc:.2f}%)")

print(f"\n🏁 Training complete. Best Val Acc: {best_val_acc:.2f}%")
