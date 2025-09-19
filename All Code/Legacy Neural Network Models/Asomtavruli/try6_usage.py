import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image, ImageDraw, ImageFont
import cv2
import numpy as np
import matplotlib.pyplot as plt
from torchvision import datasets
from torch.utils.data import DataLoader, random_split
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

input_path = "/Users/sunnysideup/Documents/data for ORCs/Asomtavruli data/test/input/hard5.jpg"  # TODO: replace this
output_path = "/Users/sunnysideup/Documents/data for ORCs/Asomtavruli data/test/output/translated_output.png"

import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image, ImageDraw, ImageFont
import cv2
import numpy as np
import os

# ===== Model Definition =====
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

# ===== Config =====
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_levels = 4
blocks_per_level = 2
dropout_rate = 0.18
f0 = 25
filters = [f0 * (2 ** i) for i in range(num_levels)]
num_classes = 38  # ⚠️ Change to your actual class count
save_path = "/Users/sunnysideup/Documents/data for ORCs/All Code/Neural Networks/Saves/best_dynamic_model_97.75%_try6.pth"

# idx_to_class should match your training labels
idx_to_class = {i: chr(0x10A0 + i) for i in range(num_classes)}  # Ⴀ to Ⴟ (Asomtavruli block)

# ===== Preprocessing Transform =====
transform = transforms.Compose([
    transforms.Grayscale(1),
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# ===== Load Model =====
model = DynamicCNN(1, num_classes, num_levels, blocks_per_level, filters, dropout_rate)
model.load_state_dict(torch.load(save_path, map_location=device))
model.to(device)
model.eval()

# ===== Segment Characters =====
def segment_letters(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"❌ Cannot read image at: {image_path}")
    _, binary = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = [cv2.boundingRect(c) for c in contours]
    boxes = sorted(boxes, key=lambda b: (b[1] // 10, b[0]))
    segments = [((x, y, w, h), img[y:y+h, x:x+w]) for (x, y, w, h) in boxes]
    return segments, img

# ===== Predict Each Letter =====
def classify_segment_topk(crop, model, transform, device, idx_to_class, k=3):
    pil_img = Image.fromarray(crop).convert("RGB")
    tensor = transform(pil_img).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(tensor)
        top_probs, top_idxs = torch.topk(output, k)
        top_labels = [idx_to_class.get(i.item(), "?") for i in top_idxs[0]]
    return top_labels


# ===== Draw Predictions on Image =====
def paint_predictions_on_image(original_img, segments, predictions, idx_to_class):
    pil_img = Image.fromarray(cv2.cvtColor(original_img, cv2.COLOR_GRAY2RGB))
    draw = ImageDraw.Draw(pil_img)

    try:
        font = ImageFont.truetype("/Users/sunnysideup/Documents/data for ORCs/All Code/Neural Networks/NotoSansGeorgian-VariableFont_wdth,wght.ttf", size=10)
    except:
        print("⚠️ Font fallback: Using default")
        font = ImageFont.load_default()

    for ((x, y, w, h), pred_list) in zip(segments, predictions):
        draw.rectangle([x, y, x+w, y+h], outline="red", width=1)
        label = " ".join(pred_list)  # space-separated top-k
        draw.text((x, y - 15), label, fill="green", font=font)

    return pil_img


# ===== Full Pipeline =====
def ocr_translate_image(image_path):
    print(f"📂 Processing: {image_path}")
    segments, orig_img = segment_letters(image_path)
    predictions = [classify_segment_topk(crop, model, transform, device, idx_to_class, k=3) for _, crop in segments]
    result_img = paint_predictions_on_image(orig_img, [seg for seg, _ in segments], predictions, idx_to_class)
    return result_img


# ===== Confusion Matrix =====
def plot_confusion_matrix(model, dataloader, idx_to_class, device, normalize=True):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            outputs = model(x)
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.cpu().numpy())

    # Generate confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    if normalize:
        cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

    # Map class indices to labels
    labels = [idx_to_class[i] for i in range(len(idx_to_class))]

    # Plot the confusion matrix
    plt.figure(figsize=(14, 12))
    sns.heatmap(cm, annot=True, fmt=".2f" if normalize else "d", 
                xticklabels=labels, yticklabels=labels, cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Normalized Confusion Matrix" if normalize else "Confusion Matrix")
    plt.tight_layout()
    plt.show()

    # Print classification report
    print("\n📊 Classification Report:\n")
    print(classification_report(all_labels, all_preds, target_names=labels))


# ===== Run =====
data_path = "/Users/sunnysideup/Documents/data for ORCs/Asomtavruli data/Sorted"  # same one used in training

# Recreate the dataset and split
dataset = datasets.ImageFolder(data_path, transform=transform)
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_set, test_set = random_split(dataset, [train_size, test_size])

batch_size = 16  # or whatever you used before
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

# Recreate idx_to_class just to be safe
idx_to_class = {v: k for k, v in dataset.class_to_idx.items()}


result = ocr_translate_image(input_path)
result.shxxow()
plot_confusion_matrix(model, test_loader, idx_to_class, device)
result.save(output_path)
print(f"✅ Output saved to: {output_path}")
