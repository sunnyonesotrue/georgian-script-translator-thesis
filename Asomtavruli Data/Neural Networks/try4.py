import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from bayes_opt import BayesianOptimization
import math

# ⚙️ Dynamic CNN Definition
class DynamicCNN(nn.Module):
    def __init__(self, input_channels, num_classes,
                 num_levels=3, blocks_per_level=2,
                 filters=[16, 32, 64], dropout_rate=0.3):
        super(DynamicCNN, self).__init__()

        assert num_levels <= len(filters), "Need at least as many filters as levels"
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

# 🔢 Dataset
data_path = '/Users/sunnysideup/Documents/data for ORCs/Asomtavruli data/Sorted'
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
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 🧪 Training function with dynamic batch size + architecture
def train_eval(lr, weight_decay, dropout_rate, f0, num_levels, blocks_per_level, log2_batch_size):
    try:
        # Convert hyperparameters
        batch_size = int(2 ** log2_batch_size)
        num_levels = int(num_levels)
        blocks_per_level = int(blocks_per_level)
        f0 = int(f0)
        filters = [f0 * (2 ** i) for i in range(num_levels)]

        # Dataloaders (reinitialized every trial with new batch size)
        train_size = int(0.8 * len(dataset))
        test_size = len(dataset) - train_size
        train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size)

        # Model
        model = DynamicCNN(1, num_classes, num_levels, blocks_per_level, filters, dropout_rate).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5)

        # Training (5 epochs)
        for epoch in range(5):
            model.train()
            correct = total = 0
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

        # Validation
        model.eval()
        correct = total = 0
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(device), y.to(device)
                out = model(x)
                _, pred = torch.max(out, 1)
                correct += (pred == y).sum().item()
                total += y.size(0)

        acc = 100 * correct / total
        print(f"✅ Trial | Acc: {acc:.2f}% | bs={batch_size} f0={f0} lvl={num_levels} blk={blocks_per_level} dr={dropout_rate:.2f} lr={lr:.5f} wd={weight_decay:.6f}")
        return acc / 100

    except Exception as e:
        print(f"❌ Failed trial: {e}")
        return 0.0

# 🔬 Search Space
pbounds = {
    'lr': (1e-4, 2e-3),
    'weight_decay': (1e-6, 1e-3),
    'dropout_rate': (0.0, 0.3),
    'f0': (8, 32),                    # base filter size
    'num_levels': (2, 4),             # depth of network
    'blocks_per_level': (1, 3),       # how many convs per level
    'log2_batch_size': (4, 6)         # batch sizes: 2^4 = 16 → 2^6 = 64
}

# 🔁 Run Bayesian Optimization
optimizer = BayesianOptimization(
    f=train_eval,
    pbounds=pbounds,
    random_state=42,
    verbose=2
)
optimizer.maximize(init_points=3, n_iter=7)

# 🏁 Best Result
print("\n🏆 Best Hyperparameters Found:")
print(optimizer.max)
