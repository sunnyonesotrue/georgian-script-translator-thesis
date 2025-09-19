import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from bayes_opt import BayesianOptimization

# Dataset and transforms
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((64, 64)),
    transforms.RandomCrop(64, padding=4),
    transforms.RandomRotation(degrees=(5, 10)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

data_path = '/Users/sunnysideup/Documents/data for ORCs/Asomtavruli data/Sorted'
dataset = datasets.ImageFolder(root=data_path, transform=transform)
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
num_classes = len(dataset.classes)

# CNN Model
class SimpleCNN(nn.Module):
    def __init__(self, num_classes, dropout_rate=0.3):
        super(SimpleCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(8),

            nn.Conv2d(8, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(16),

            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(16),

            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),

            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),

            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),

            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),

            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),

            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Flatten(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        x = self.classifier(x)
        return x

# Train + Eval Function for Bayesian Optimization
def train_with_params(lr, weight_decay, dropout_rate):
    lr = float(lr)
    weight_decay = float(weight_decay)
    dropout_rate = float(dropout_rate)

    model = SimpleCNN(num_classes=num_classes, dropout_rate=dropout_rate).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=3)

    # Train for 3 epochs
    model.train()
    for epoch in range(10):
        correct = 0
        total = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
        train_acc = 100 * correct / total
        scheduler.step()

    # Validation accuracy
    model.eval()
    val_correct, val_total = 0, 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            val_correct += (predicted == labels).sum().item()
            val_total += labels.size(0)

    val_acc = 100 * val_correct / val_total
    print(f"📊 Trial: Train Acc = {train_acc:.2f}%, Val Acc = {val_acc:.2f}% | lr = {lr:.5f}, wd = {weight_decay:.6f}, dropout = {dropout_rate:.2f}")
    return val_acc / 100  # return as float in [0, 1]

# Bayesian Optimization setup
pbounds = {
    'lr': (1e-4, 1e-2),
    'weight_decay': (1e-6, 1e-3),
    'dropout_rate': (0.1, 0.5)
}

optimizer = BayesianOptimization(
    f=train_with_params,
    pbounds=pbounds,
    random_state=42,
    verbose=2
)

# Run optimization
optimizer.maximize(init_points=3, n_iter=7)

# Print final best parameters
print("\n✅ Best Hyperparameters Found:")
print(optimizer.max)
