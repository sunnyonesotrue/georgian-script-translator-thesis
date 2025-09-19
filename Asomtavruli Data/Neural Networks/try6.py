import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split

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
num_epochs = 20
save_path = "./best_dynamic_model.pth"

# ----------------------------
# 🧠 DYNAMIC MODEL
# ----------------------------
class DynamicCNN(nn.Module):
    def __init__(self, input_channels, num_classes,
                 num_levels, blocks_per_level,
                 filters, dropout_rate):
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
# 📦 DATASET
# ----------------------------
data_path = r"C:\Users\Sandro\Documents\GitHub\Project-Lab---Asomtavruli-OCR\Asomtavruli data\Sorted"
transform = transforms.Compose([
    transforms.Grayscale(1),
    transforms.Resize((64, 64)),
    transforms.RandomCrop(64, padding=4),
    transforms.RandomRotation((5, 10)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

dataset = datasets.ImageFolder(data_path, transform=transform)
num_classes = len(dataset.classes)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_set, test_set = random_split(dataset, [train_size, test_size])
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

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

    # Validation
    model.eval()
    val_total, val_correct = 0, 0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            _, pred = torch.max(out, 1)
            val_correct += (pred == y).sum().item()
            val_total += y.size(0)
    val_acc = 100 * val_correct / val_total

    print(f"Epoch {epoch+1}/{num_epochs} | Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}%")

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), save_path)
        print(f"✅ Model saved (Val Acc: {val_acc:.2f}%)")

print(f"\n🏁 Training complete. Best Val Acc: {best_val_acc:.2f}%")
