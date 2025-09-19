import torch
import torch.nn as nn
from torchvision import transforms, datasets
from torchvision.datasets.folder import default_loader
from PIL import Image, ImageDraw, ImageFont
import cv2
import numpy as np
import os
from ThresholdManagerCopy import ThresholdManager as TM

# ----------------------------
# ✅ CONFIGURATION
# ----------------------------
model_path = r"/Users/sunnysideup/Documents/data for ORCs/All Code/Neural Networks/Saves/best_dynamic_model_try10_97.60.pth"
data_path = r"/Users/sunnysideup/Documents/data for ORCs/Asomtavruli data/Sorted"
image_path = r"/Users/sunnysideup/Documents/data for ORCs/Asomtavruli data/test/input/easy5.tif"
output_path = r"/Users/sunnysideup/Documents/data for ORCs/Asomtavruli data/test/output/translated_output_try7.png"
fontLocation = r"/Users/sunnysideup/Documents/data for ORCs/All Code/Neural Networks/NotoSansGeorgian-VariableFont_wdth,wght.ttf"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ----------------------------
# 🧠 MODEL DEFINITION
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
# 🎨 TEST TRANSFORM
# ----------------------------
test_transform = transforms.Compose([
    transforms.Grayscale(1),         # Convert RGB → Grayscale
    transforms.Resize((64, 64)),     # Resize to model input
    transforms.ToTensor(),           # Convert to tensor
    transforms.Normalize((0.5,), (0.5,))
])

# ----------------------------
# 📦 CLASS LABELS
# ----------------------------
base_dataset = datasets.ImageFolder(data_path)
num_classes = len(base_dataset.classes)
idx_to_class = {v: k for k, v in base_dataset.class_to_idx.items()}

# ----------------------------
# 🔄 LOAD MODEL
# ----------------------------
f0 = 25
num_levels = 4
blocks_per_level = 2
dropout_rate = 0.18
filters = [f0 * (2 ** i) for i in range(num_levels)]

model = DynamicCNN(1, num_classes, num_levels, blocks_per_level, filters, dropout_rate)
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
model.eval()

# ----------------------------
# 🔍 OCR UTILITIES
# ----------------------------
def segment_letters(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"❌ Image not found: {image_path}")
    # variants = TM.threshold_variants_from_image(img)
    _, binary = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = [cv2.boundingRect(c) for c in contours]
    boxes = sorted(boxes, key=lambda b: (b[1] // 10, b[0]))
    return [(x, y, w, h) for (x, y, w, h) in boxes], img

def classify_segment_top1(crop, model, transform, device, idx_to_class):
    pil_img = Image.fromarray(crop).convert("RGB")  # Accept RGB input
    tensor = transform(pil_img).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(tensor)
        pred = torch.argmax(output, dim=1).item()
        return idx_to_class[pred]

def paint_predictions_on_image(original_img, boxes, predictions):
    pil_img = Image.fromarray(cv2.cvtColor(original_img, cv2.COLOR_GRAY2RGB))
    draw = ImageDraw.Draw(pil_img)
    try:
        font = ImageFont.truetype(fontLocation, size=20)
    except:
        font = ImageFont.load_default()

    for ((x, y, w, h), pred) in zip(boxes, predictions):
        draw.rectangle([x, y, x + w, y + h], outline="red", width=1)
        draw.text((x, y - 15), pred, fill="green", font=font)
    return pil_img

def run_ocr_on_image(image_path, output_path):
    print(f"📂 Processing: {image_path}")
    boxes, original_img = segment_letters(image_path)
    crops = [original_img[y:y+h, x:x+w] for (x, y, w, h) in boxes]
    predictions = [classify_segment_top1(crop, model, test_transform, device, idx_to_class) for crop in crops]
    result_img = paint_predictions_on_image(original_img, boxes, predictions)
    result_img.save(output_path)
    result_img.show()
    print(f"✅ Saved result: {output_path}")

# ----------------------------
# ▶️ RUN
# ----------------------------
run_ocr_on_image(image_path, output_path)
