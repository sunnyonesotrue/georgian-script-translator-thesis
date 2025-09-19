import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from PIL import Image

# Step 1: Define the same transform you used during training
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Step 2: Define the model class (same as you trained)
class SimpleCNN(torch.nn.Module):
    def __init__(self, num_classes):
        super(SimpleCNN, self).__init__()
        self.features = torch.nn.Sequential(
            torch.nn.Conv2d(1, 32, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2, 2),
            torch.nn.Conv2d(32, 64, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2, 2),
        )
        self.classifier = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(64 * 16 * 16, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, num_classes)
        )
        
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

# Step 3: Load the test dataset
test_dataset = datasets.ImageFolder(root='/Users/sunnysideup/Documents/data for ORCs/Asomtavruli data/Reserves', transform=transform)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Step 4: Load the saved model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
num_classes = len(test_dataset.classes)  # number of classes in reserve set (should match training)
model = SimpleCNN(num_classes)
model.load_state_dict(torch.load('/Users/sunnysideup/Documents/data for ORCs/All Code/Neural Networks/Saves/Try1/model_weights.pth', map_location=device))
model.to(device)
model.eval()

# Step 5: Evaluate the model
correct = 0
total = 0

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = 100 * correct / total
print(f'Test Accuracy on Reserve Dataset: {accuracy:.2f}%')
