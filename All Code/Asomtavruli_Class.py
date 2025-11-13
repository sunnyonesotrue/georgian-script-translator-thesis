"""
Asomtavruli OCR — Class-based utility with text generation
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
    from ThresholdManager import ThresholdManager as TM
except Exception as _e:
    TM = None


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
        """Create an OCR runner."""
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

        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

        self.transform = transforms.Compose([
            transforms.Grayscale(1),
            transforms.Resize((self.image_size, self.image_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ])

        base_dataset = datasets.ImageFolder(self.data_path)
        class_to_idx = dict(list(base_dataset.class_to_idx.items())[: self.fixed_num_classes])
        self.idx_to_class = {v: k for k, v in class_to_idx.items()}

        self._font_cache: Optional[ImageFont.FreeTypeFont] = None

    # ------------------------
    # 🔍 Segmentation & Inference
    # ------------------------
    @staticmethod
    def segment_letters(binary_img: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """Find bounding boxes of connected components on a binary (or grayscale) image."""
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
    # 📝 NEW: Text Generation
    # ------------------------
    def generate_text_from_boxes(self, boxes: List[Tuple[int, int, int, int]], 
                                 predictions: List[str],
                                 image_shape: Tuple[int, int]) -> str:
        """
        Generate spatially-aware text from OCR results.
        Groups letters by line and adds appropriate spacing.
        Filters out noise marked with ','.
        """
        if not boxes or not predictions:
            return ""
        
        h, w = image_shape
        
        # Filter out noise (comma markers)
        filtered = [(box, pred) for box, pred in zip(boxes, predictions) if pred != ","]
        
        if not filtered:
            return ""
        
        boxes, predictions = zip(*filtered)
        boxes = list(boxes)
        predictions = list(predictions)
        
        # Calculate median character height for line grouping
        heights = [bh for (_, _, _, bh) in boxes]
        median_height = int(np.median(heights)) if heights else 10
        line_threshold = max(median_height // 2, 5)
        
        # Group boxes by line
        lines = {}
        for (x, y, bw, bh), pred in zip(boxes, predictions):
            line_num = y // line_threshold
            if line_num not in lines:
                lines[line_num] = []
            lines[line_num].append((x, pred))
        
        # Sort lines top to bottom, and chars left to right within each line
        sorted_lines = []
        for line_num in sorted(lines.keys()):
            line_chars = sorted(lines[line_num], key=lambda item: item[0])
            sorted_lines.append(line_chars)
        
        # Build text with spacing
        text_lines = []
        for line_chars in sorted_lines:
            if not line_chars:
                continue
            
            line_text = []
            prev_x = None
            
            for x, char in line_chars:
                if prev_x is not None:
                    # Calculate spacing between characters
                    gap = x - prev_x
                    # If gap is larger than typical character width, add space
                    if gap > median_height * 0.8:  # Adjust threshold as needed
                        num_spaces = max(1, int(gap / (median_height * 0.8)))
                        line_text.append(" " * num_spaces)
                
                line_text.append(char)
                prev_x = x + median_height  # Approximate next position
            
            text_lines.append("".join(line_text))
        
        return "\n".join(text_lines)

    def save_text_file(self, text: str, base_filename: str, variant_name: str) -> str:
        """Save OCR text to a file."""
        text_filename = f"asomtavruli_text_{base_filename}_{variant_name}.txt"
        text_path = os.path.join(self.output_dir, text_filename)
        
        with open(text_path, 'w', encoding='utf-8') as f:
            f.write(text)
        
        return text_path

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
        skip_labels = skip_labels or set([","])

        rgb = cv2.cvtColor(base_img, cv2.COLOR_GRAY2RGB) if base_img.ndim == 2 else base_img
        pil = Image.fromarray(rgb)
        draw = ImageDraw.Draw(pil)
        font = self._get_font(size=20)

        for (x, y, w, h), pred in zip(boxes, predictions):
            if pred in skip_labels:
                continue
            draw.rectangle([x, y, x + w, y + h], outline="red", width=1)
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
        """Segment, classify, and return (boxes, predictions)."""
        boxes = self.segment_letters(binary_or_gray_img)
        crops = [binary_or_gray_img[y : y + h, x : x + w] for (x, y, w, h) in boxes]
        preds = [self.classify_crop(c) for c in crops]
        return boxes, preds

    def run_on_image(self, image_path: str, *, name_suffix: str = "single", show: bool = False) -> str:
        """Load image (grayscale), segment + classify + draw + save."""
        gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if gray is None:
            raise FileNotFoundError(f"Could not read image: {image_path}")
        boxes, preds = self.run_on_array(gray)
        out_path = self.draw_predictions(gray, boxes, preds, name_suffix=name_suffix, show=show)
        return out_path

    def run_on_all_thresholds(self, image_path: str, *, show: bool = False, 
                              generate_text: bool = False) -> List[str]:
        """
        Apply all threshold variants and render each result.
        Optionally generate text files with OCR results.
        """
        if TM is None:
            raise ImportError("ThresholdManager could not be imported.")
        
        original = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if original is None:
            raise FileNotFoundError(f"Could not read image: {image_path}")

        base_filename = os.path.splitext(os.path.basename(image_path))[0]
        
        tm = TM()
        variants: List[np.ndarray] = tm.threshold_variants_from_image(original)

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
            
            # Generate text file if requested
            if generate_text and preds:
                text_content = self.generate_text_from_boxes(boxes, preds, variant.shape)
                if text_content:
                    text_path = self.save_text_file(text_content, base_filename, name)
                    print(f"[INFO] {name}: saved text file {os.path.basename(text_path)}")
            
            # Draw visualization
            out_path = self.draw_predictions(variant, boxes, preds, 
                                            name_suffix=f"{base_filename}_{name}", 
                                            show=show)
            saved_paths.append(out_path)
            
        return saved_paths


__all__ = ["AsomtavruliOCR", "DynamicCNN"]