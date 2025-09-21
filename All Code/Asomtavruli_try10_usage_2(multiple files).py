import torch
import torch.nn as nn
from torchvision import transforms, datasets
from PIL import Image, ImageDraw, ImageFont
import cv2
import numpy as np
import os
from ThresholdManager import ThresholdManager as TM

# ----------------------------
# ✅ CONFIGURATION
# ----------------------------
model_path = r"/Users/sunnysideup/Documents/data for ORCs/All Code/Neural Networks/Saves/Asomtavruli/best_dynamic_model_try10_97.60.pth"
data_path = r"/Users/sunnysideup/Documents/data for ORCs/Asomtavruli data/Sorted"
image_path = r"/Users/sunnysideup/Documents/data for ORCs/Asomtavruli data/test/input/1000034762.jpg"
output_folder = r"/Users/sunnysideup/Documents/data for ORCs/Asomtavruli data/test/output"
fontLocation = r"/Users/sunnysideup/Documents/data for ORCs/All Code/Neural Networks/Asomtavruli/NotoSansGeorgian-VariableFont_wdth,wght.ttf"
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
# 🧠 FIXED CLASS COUNT
# ----------------------------
fixed_num_classes = 39  # match model checkpoint
f0 = 25
num_levels = 4
blocks_per_level = 2
dropout_rate = 0.18
filters = [f0 * (2 ** i) for i in range(num_levels)]

model = DynamicCNN(1, fixed_num_classes, num_levels, blocks_per_level, filters, dropout_rate)
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
model.eval()

# ----------------------------
# 🎨 TRANSFORM
# ----------------------------
transform = transforms.Compose([
    transforms.Grayscale(1),
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# ----------------------------
# 🔡 CLASS LABELS (limited to 39)
# ----------------------------
base_dataset = datasets.ImageFolder(data_path)
class_to_idx = dict(list(base_dataset.class_to_idx.items())[:fixed_num_classes])
idx_to_class = {v: k for k, v in class_to_idx.items()}

# ----------------------------
# 🔍 OCR UTILITIES
# ----------------------------
def segment_letters(img):
    contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = sorted([cv2.boundingRect(c) for c in contours], key=lambda b: (b[1] // 10, b[0]))
    return boxes

def classify(crop):
    pil_img = Image.fromarray(crop).convert("RGB")
    tensor = transform(pil_img).unsqueeze(0).to(device)
    with torch.no_grad():
        pred = model(tensor).argmax(dim=1).item()
    return idx_to_class.get(pred, "<?>")

def paint_predictions(thresh_img, boxes, predictions, name_suffix):
    pil = Image.fromarray(cv2.cvtColor(thresh_img, cv2.COLOR_GRAY2RGB))
    draw = ImageDraw.Draw(pil)
    try:
        font = ImageFont.truetype(fontLocation, size=20)
    except:
        font = ImageFont.load_default()
    for (x, y, w, h), pred in zip(boxes, predictions):
        if pred == ",":
            continue  # Skip labeling comma
        draw.rectangle([x, y, x + w, y + h], outline="red", width=1)
        draw.text((x, y - 15), pred, fill="green", font=font)
    os.makedirs(output_folder, exist_ok=True)
    out_path = os.path.join(output_folder, f"ocr_result_{name_suffix}.png")
    pil.save(out_path)
    pil.show(title=name_suffix)
    print(f"✅ Saved: {out_path}")

# ----------------------------
# ▶️ MULTI-VARIANT OCR
# ----------------------------
def run_ocr_on_all_thresholds(image_path):
    print("🧪 Applying all thresholding methods and running OCR...")
    original = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    tm = TM()
    variants = tm.threshold_variants_from_image(original)
    names = ["mean_median", "mean_closing", "gaussian_median", "gaussian_closing", "otsu", "mean_bilateral","gaussian_bilateral","mean_NLMD","gaussian_NLMD", "all"]

    for variant, name in zip(variants, names):
        boxes = segment_letters(variant)
        crops = [variant[y:y+h, x:x+w] for (x, y, w, h) in boxes]
        predictions = [classify(crop) for crop in crops]
        paint_predictions(variant, boxes, predictions, name)

# ----------------------------
# ▶️ RUN
# ----------------------------
run_ocr_on_all_thresholds(image_path)
