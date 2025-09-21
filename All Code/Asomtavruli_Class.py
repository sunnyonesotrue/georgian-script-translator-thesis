"""
Asomtavruli OCR — Class-based utility
-------------------------------------

This module wraps the dynamic CNN classifier and the multi-threshold OCR pipeline into a
single, reusable class. You can:

- Initialize with your checkpoint and dataset root (to reconstruct label mapping)
- Run OCR on one image or across all thresholding variants from ThresholdManager
- Customize segmentation, drawing, and I/O

Usage (example):

    from asomtavruli_ocr import AsomtavruliOCR

    ocr = AsomtavruliOCR(
        model_path="/path/to/best_dynamic_model.pth",
        data_path="/path/to/Sorted",
        font_path="/path/to/NotoSansGeorgian-VariableFont_wdth,wght.ttf",
        output_dir="/path/to/output",
        fixed_num_classes=39,
        f0=25,
        num_levels=4,
        blocks_per_level=2,
        dropout_rate=0.18,
        image_size=64,
        device=None,  # auto-detect
    )

    # Single run across all threshold variants
    ocr.run_on_all_thresholds("/path/to/input.jpg", show=True)

    # Or run on a single pre-thresholded image (numpy uint8, 0..255)
    # boxes, preds = ocr.run_on_array(thresh_img)

Notes:
- Requires `ThresholdManagerCopy.ThresholdManager` to be importable as TM.
- Expects grayscale input for segmentation; class handles color → gray.
"""

from __future__ import annotations
import os
from typing import List, Tuple, Optional
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch
import torch.nn as nn
from torchvision import transforms, datasets

try:
    # Your local ThresholdManager; keep alias as TM
    from ThresholdManager import ThresholdManager as TM
except Exception as _e:
    TM = None  # defer error until actually used


# ----------------------------
# 🧠 Model Definition
# ----------------------------
class DynamicCNN(nn.Module):
    def __init__(
        self,
        input_channels: int,
        num_classes: int,
        num_levels: int,
        blocks_per_level: int,
        filters: List[int],
        dropout_rate: float,
    ) -> None:
        super().__init__()
        layers: List[nn.Module] = []
        in_channels = input_channels
        for level in range(num_levels):
            out_channels = filters[level]
            for _ in range(blocks_per_level):
                layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
                layers.append(nn.ReLU(inplace=True))
                layers.append(nn.BatchNorm2d(out_channels))
                in_channels = out_channels
            if level != num_levels - 1:
                layers.append(nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=2, padding=1))
                layers.append(nn.ReLU(inplace=True))
                layers.append(nn.BatchNorm2d(in_channels))
        self.features = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Flatten(),
            nn.Linear(in_channels, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.pool(x)
        x = self.classifier(x)
        return x


# ----------------------------
# 📦 OCR Wrapper
# ----------------------------
class AsomtavruliOCR:
    def __init__(
        self,
        model_path: str,
        data_path: str,
        font_path: Optional[str] = None,
        output_dir: Optional[str] = None,
        *,
        fixed_num_classes: int = 39,
        f0: int = 25,
        num_levels: int = 4,
        blocks_per_level: int = 2,
        dropout_rate: float = 0.18,
        image_size: int = 64,
        device: Optional[torch.device] = None,
    ) -> None:
        """Create an OCR runner.

        Args:
            model_path: Path to the trained checkpoint (.pth with state_dict).
            data_path: Root folder of the dataset used to learn class order (ImageFolder layout).
            font_path: Optional TTF path for label drawing.
            output_dir: Where to save rendered outputs; created if missing.
            fixed_num_classes: Number of classes expected by checkpoint.
            f0, num_levels, blocks_per_level, dropout_rate: CNN arch params.
            image_size: Resize side for classifier.
            device: torch.device to use; if None, auto-select CUDA if available.
        """
        self.model_path = model_path
        self.data_path = data_path
        self.font_path = font_path
        self.output_dir = output_dir or os.path.join(os.getcwd(), "ocr_outputs")
        os.makedirs(self.output_dir, exist_ok=True)

        self.fixed_num_classes = fixed_num_classes
        self.f0 = f0
        self.num_levels = num_levels
        self.blocks_per_level = blocks_per_level
        self.dropout_rate = dropout_rate
        self.image_size = image_size

        # device
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # model
        filters = [self.f0 * (2 ** i) for i in range(self.num_levels)]
        self.model = DynamicCNN(
            input_channels=1,
            num_classes=self.fixed_num_classes,
            num_levels=self.num_levels,
            blocks_per_level=self.blocks_per_level,
            filters=filters,
            dropout_rate=self.dropout_rate,
        )
        state = torch.load(self.model_path, map_location=self.device)
        self.model.load_state_dict(state)
        self.model.to(self.device)
        self.model.eval()

        # transform
        self.transform = transforms.Compose([
            transforms.Grayscale(1),
            transforms.Resize((self.image_size, self.image_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ])

        # class mapping (respect original class order; trim to fixed_num_classes)
        base_dataset = datasets.ImageFolder(self.data_path)
        class_to_idx = dict(list(base_dataset.class_to_idx.items())[: self.fixed_num_classes])
        self.idx_to_class = {v: k for k, v in class_to_idx.items()}

        # font
        self._font_cache: Optional[ImageFont.FreeTypeFont] = None

    # ------------------------
    # 🔍 Segmentation & Inference
    # ------------------------
    @staticmethod
    def segment_letters(binary_img: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """Find bounding boxes of connected components on a binary (or grayscale) image.
        Returns boxes sorted approximately top-to-bottom and left-to-right.
        """
        if binary_img.ndim == 3:
            gray = cv2.cvtColor(binary_img, cv2.COLOR_BGR2GRAY)
        else:
            gray = binary_img
        _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        boxes = [cv2.boundingRect(c) for c in contours]
        boxes = sorted(boxes, key=lambda b: (b[1] // 10, b[0]))
        return boxes

    def _to_tensor(self, crop: np.ndarray) -> torch.Tensor:
        if crop.ndim == 2:
            pil_img = Image.fromarray(crop).convert("RGB")
        else:
            pil_img = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
        tensor = self.transform(pil_img).unsqueeze(0).to(self.device)
        return tensor

    def classify_crop(self, crop: np.ndarray) -> str:
        with torch.no_grad():
            logits = self.model(self._to_tensor(crop))
            pred = int(logits.argmax(dim=1).item())
        return self.idx_to_class.get(pred, "<<?>>")

    # ------------------------
    # 🖼️ Rendering
    # ------------------------
    def _get_font(self, size: int = 20):
        if self._font_cache is not None:
            return self._font_cache
        try:
            if self.font_path and os.path.exists(self.font_path):
                self._font_cache = ImageFont.truetype(self.font_path, size=size)
            else:
                self._font_cache = ImageFont.load_default()
        except Exception:
            self._font_cache = ImageFont.load_default()
        return self._font_cache

    def draw_predictions(
        self,
        base_img: np.ndarray,
        boxes: List[Tuple[int, int, int, int]],
        predictions: List[str],
        name_suffix: str,
        skip_labels: Optional[set] = None,
        show: bool = False,
    ) -> str:
        """Draw boxes + labels and save to output_dir. Returns saved path."""
        skip_labels = skip_labels or set([","])  # default: skip comma label

        rgb = cv2.cvtColor(base_img, cv2.COLOR_GRAY2RGB) if base_img.ndim == 2 else base_img
        pil = Image.fromarray(rgb)
        draw = ImageDraw.Draw(pil)
        font = self._get_font(size=20)

        for (x, y, w, h), pred in zip(boxes, predictions):
            if pred in skip_labels:
                continue
            # rectangle
            draw.rectangle([x, y, x + w, y + h], outline="red", width=1)
            # label above box when possible
            text_y = y - 15 if y - 15 >= 0 else y + h + 2
            draw.text((x, text_y), pred, fill="green", font=font)

        out_path = os.path.join(self.output_dir, f"ocr_result_{name_suffix}.png")
        pil.save(out_path)
        if show:
            try:
                pil.show(title=name_suffix)
            except Exception:
                pass
        return out_path

    # ------------------------
    # ▶️ Pipeline Entrypoints
    # ------------------------
    def run_on_array(self, binary_or_gray_img: np.ndarray) -> Tuple[List[Tuple[int, int, int, int]], List[str]]:
        """Segment, classify, and return (boxes, predictions). Does not draw/save."""
        boxes = self.segment_letters(binary_or_gray_img)
        crops = [binary_or_gray_img[y : y + h, x : x + w] for (x, y, w, h) in boxes]
        preds = [self.classify_crop(c) for c in crops]
        return boxes, preds

    def run_on_image(self, image_path: str, *, name_suffix: str = "single", show: bool = False) -> str:
        """Load image (grayscale), segment + classify + draw + save. Returns output path."""
        gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if gray is None:
            raise FileNotFoundError(f"Could not read image: {image_path}")
        boxes, preds = self.run_on_array(gray)
        out_path = self.draw_predictions(gray, boxes, preds, name_suffix=name_suffix, show=show)
        return out_path

    def run_on_all_thresholds(self, image_path: str, *, show: bool = False) -> List[str]:
        """Apply all threshold variants (via ThresholdManager) and render each result.
        Returns list of saved output paths.
        """
        if TM is None:
            raise ImportError(
                "ThresholdManagerCopy could not be imported. Ensure it is on PYTHONPATH."
            )
        original = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if original is None:
            raise FileNotFoundError(f"Could not read image: {image_path}")

        tm = TM()
        variants: List[np.ndarray] = tm.threshold_variants_from_image(original)

        # Default names; if counts differ, extend with idx
        default_names = [
            "mean_median",
            "mean_closing",
            "gaussian_median",
            "gaussian_closing",
            "otsu",
            "mean_bilateral",
            "gaussian_bilateral",
            "mean_NLMD",
            "gaussian_NLMD",
            "all",
        ]
        if len(default_names) < len(variants):
            default_names += [f"var_{i}" for i in range(len(default_names), len(variants))]
        names = default_names[: len(variants)]

        saved_paths: List[str] = []
        for variant, name in zip(variants, names):
            boxes, preds = self.run_on_array(variant)
            out_path = self.draw_predictions(variant, boxes, preds, name_suffix=name, show=show)
            saved_paths.append(out_path)
        return saved_paths


__all__ = ["AsomtavruliOCR", "DynamicCNN"]


# ocr = AsomtavruliOCR(
#     model_path="/path/to/best_dynamic_model.pth",
#     data_path="/path/to/Sorted",
#     font_path="/path/to/NotoSansGeorgian-VariableFont_wdth,wght.ttf",
#     output_dir="/path/to/output",
#     fixed_num_classes=39,
#     f0=25, num_levels=4, blocks_per_level=2, dropout_rate=0.18, image_size=64
# )