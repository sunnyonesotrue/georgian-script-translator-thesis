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
# 🧠 Nuskhuri Model Definition
# ----------------------------
class NuskhuriDynamicCNN(nn.Module):
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
            nn.Linear(in_channels, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.pool(x)
        x = self.classifier(x)
        return x


# ----------------------------
# 📦 Nuskhuri OCR Wrapper
# ----------------------------
class NuskhuriOCR:
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
        self.model_path = model_path
        self.data_path = data_path
        self.font_path = font_path
        self.output_dir = output_dir or os.path.join(os.getcwd(), "nuskhuri_ocr_outputs")
        os.makedirs(self.output_dir, exist_ok=True)

        self.fixed_num_classes = fixed_num_classes
        self.f0 = f0
        self.num_levels = num_levels
        self.blocks_per_level = blocks_per_level
        self.dropout_rate = dropout_rate
        self.image_size = image_size

        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        filters = [self.f0 * (2 ** i) for i in range(self.num_levels)]
        self.model = NuskhuriDynamicCNN(
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
        
        # Translation mapping from Nuskhuri to modern Georgian
        self.nuskhuri_to_modern = {
            'ⴀ': 'ა', 'ⴁ': 'ბ', 'ⴂ': 'გ', 'ⴃ': 'დ', 'ⴄ': 'ე', 'ⴅ': 'ვ',
            'ⴆ': 'ზ', 'ⴇ': 'თ', 'ⴈ': 'ი', 'ⴉ': 'კ', 'ⴊ': 'ლ', 'ⴋ': 'მ',
            'ⴌ': 'ნ', 'ⴍ': 'ო', 'ⴎ': 'პ', 'ⴏ': 'ჟ', 'ⴐ': 'რ', 'ⴑ': 'ს',
            'ⴒ': 'ტ', 'ⴓ': 'უ', 'ⴔ': 'ფ', 'ⴕ': 'ქ', 'ⴖ': 'ღ', 'ⴗ': 'ყ',
            'ⴘ': 'შ', 'ⴙ': 'ჩ', 'ⴚ': 'ც', 'ⴛ': 'ძ', 'ⴜ': 'წ', 'ⴝ': 'ჭ',
            'ⴞ': 'ხ', 'ⴟ': 'ჯ', 'ⴠ': 'ჰ', 'ⴡ': 'ჱ', 'ⴢ': 'ჲ', 'ⴣ': 'ჳ',
            'ⴤ': 'ჴ', 'ⴥ': 'ჵ',
            # Add Asomtavruli variants for compatibility
            'Ⴀ': 'ა', 'Ⴁ': 'ბ', 'Ⴂ': 'გ', 'Ⴃ': 'დ', 'Ⴄ': 'ე', 'Ⴅ': 'ვ',
            'Ⴆ': 'ზ', 'Ⴇ': 'თ', 'Ⴈ': 'ი', 'Ⴉ': 'კ', 'Ⴊ': 'ლ', 'Ⴋ': 'მ',
            'Ⴌ': 'ნ', 'Ⴍ': 'ო', 'Ⴎ': 'პ', 'Ⴏ': 'ჟ', 'Ⴐ': 'რ', 'Ⴑ': 'ს',
            'Ⴒ': 'ტ', 'Ⴓ': 'უ', 'Ⴔ': 'ფ', 'Ⴕ': 'ქ', 'Ⴖ': 'ღ', 'Ⴗ': 'ყ',
            'Ⴘ': 'შ', 'Ⴙ': 'ჩ', 'Ⴚ': 'ც', 'Ⴛ': 'ძ', 'Ⴜ': 'წ', 'Ⴝ': 'ჭ',
            'Ⴞ': 'ხ', 'Ⴟ': 'ჯ', 'Ⴠ': 'ჰ', 'Ⴡ': 'ჱ', 'Ⴢ': 'ჲ', 'Ⴣ': 'ჳ',
            'Ⴤ': 'ჴ', 'Ⴥ': 'ჵ',
        }

    # ------------------------
    # 🔍 Segmentation
    # ------------------------
    @staticmethod
    def clean_binary_image(bin_img: np.ndarray, aggressive: bool = True) -> np.ndarray:
        """Aggressively clean binary image before segmentation."""
        kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
        cleaned = cv2.morphologyEx(bin_img, cv2.MORPH_OPEN, kernel_small, iterations=1)
        
        kernel_connect = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 1))
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel_connect, iterations=1)
        
        nb, labels, stats, centroids = cv2.connectedComponentsWithStats(cleaned, connectivity=8)
        h, w = cleaned.shape
        output = np.zeros_like(cleaned)
        
        for i in range(1, nb):
            x, y, bw, bh, area = stats[i]
            aspect_ratio = float(bh) / float(bw) if bw > 0 else 0
            
            if area < 10:
                continue
            if x <= 2 or y <= 2 or x + bw >= w - 2 or y + bh >= h - 2:
                continue
            if bw < 2 or bh < 3:
                continue
            if bw > w * 0.3 or bh > h * 0.5:
                continue
            if aspect_ratio > 15 or (aspect_ratio < 0.1 and area < 50):
                continue
            if area > 5000:
                continue
                
            output[labels == i] = 255
        
        return output

    @staticmethod
    def segment_letters(binary_or_gray_img, min_area: int = 15, max_area: int = 4000) -> List[Tuple[int, int, int, int]]:
        """IMPROVED segmentation with better noise filtering."""
        th = binary_or_gray_img
        
        white_ratio = np.mean(th == 255)
        if white_ratio > 0.5:
            th = cv2.bitwise_not(th)
        
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            th, connectivity=8
        )
        
        h, w = th.shape
        boxes = []
        
        for i in range(1, num_labels):
            x, y, bw, bh, area = stats[i]
            
            if area < min_area or area > max_area:
                continue
            if bw < 2 or bh < 3:
                continue
            if x <= 1 or y <= 1 or x + bw >= w - 1 or y + bh >= h - 1:
                continue
            
            aspect_ratio = float(bh) / float(bw) if bw > 0 else 0
            density = area / float(bw * bh) if (bw * bh) > 0 else 0
            
            if aspect_ratio > 12:
                continue
            if aspect_ratio < 0.15 and area < 100:
                continue
            if density < 0.1:
                continue
            
            boxes.append((x, y, bw, bh))
        
        if boxes:
            heights = [bh for (_, _, _, bh) in boxes]
            median_height = int(np.median(heights)) if heights else 10
            line_threshold = max(median_height // 2, 5)
        else:
            line_threshold = 10
        
        boxes = sorted(boxes, key=lambda b: (b[1] // line_threshold, b[0]))
        
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
            logits1 = self.model(self._to_tensor(crop))
            conf1 = torch.softmax(logits1, dim=1).max().item()
            pred1 = int(logits1.argmax(dim=1).item())
            
            crop_inv = cv2.bitwise_not(crop)
            logits2 = self.model(self._to_tensor(crop_inv))
            conf2 = torch.softmax(logits2, dim=1).max().item()
            pred2 = int(logits2.argmax(dim=1).item())
            
            if conf2 > conf1:
                pred = pred2
            else:
                pred = pred1
        
        return self.idx_to_class.get(pred, "Error in Classify_Crop")

    # ------------------------
    # 📝 NEW: Text Generation
    # ------------------------
    def generate_text_from_boxes(self, boxes: List[Tuple[int, int, int, int]], 
                                 predictions: List[str],
                                 image_shape: Tuple[int, int], 
                                 translate_text: bool) -> str:
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
                
                line_text.append(self.nuskhuri_to_modern[char] if translate_text and char in self.nuskhuri_to_modern else char)
                prev_x = x + median_height  # Approximate next position
            
            text_lines.append("".join(line_text))
        
        return "\n".join(text_lines)

    def save_text_file(self, text: str, base_filename: str, variant_name: str,
                       label: str = "") -> str:
        """Save OCR text to a file.
        
        label  — optional suffix appended before .txt, e.g. 'original' or
                 'translated'.  Empty string keeps the old naming convention.
        """
        suffix = f"_{label}" if label else ""
        text_filename = f"nuskhuri_text_{base_filename}_{variant_name}{suffix}.txt"
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
        translate_onto_image: bool = False,
        ) -> str:
        """Draw boxes + labels and save to output_dir.
        
        When translate_onto_image is True the drawn label is the modern
        Georgian equivalent rather than the original Nuskhuri character.
        """
        skip_labels = skip_labels or set([","])

        if base_img.ndim == 2:
            rgb = cv2.cvtColor(base_img, cv2.COLOR_GRAY2RGB)
        else:
            rgb = base_img.copy()
        
        pil = Image.fromarray(rgb)
        draw = ImageDraw.Draw(pil)
        font = self._get_font(size=20)

        for (x, y, w, h), pred in zip(boxes, predictions):
            if pred in skip_labels:
                continue
            
            draw.rectangle([x, y, x + w, y + h], outline="red", width=1)
            
            text_y = y - 18 if y - 18 >= 0 else y + h + 2
            label = (
                self.nuskhuri_to_modern.get(pred, pred)
                if translate_onto_image
                else pred
            )
            draw.text((x, text_y), label, fill="red", font=font)

        out_path = os.path.join(self.output_dir, f"nuskhuri_ocr_result_{name_suffix}.png")
        pil.save(out_path)
        
        if show:
            try:
                pil.show(title=f"Nuskhuri: {name_suffix}")
            except Exception:
                pass
        
        return out_path

    # ------------------------
    # ▶️ Pipeline Entrypoints
    # ------------------------
    def run_on_array(self, binary_or_gray_img) -> Tuple[List[Tuple[int, int, int, int]], List[str]]:
        """IMPROVED: Segment and classify with better diagnostics."""
        bin_img = binary_or_gray_img
        
        white_ratio = np.mean(bin_img == 255)
        print(f"[DEBUG] White pixel ratio: {white_ratio:.3f}")
        
        boxes = self.segment_letters(bin_img, min_area=15, max_area=4000)
        print(f"[DEBUG] Found {len(boxes)} components after filtering")
        
        if len(boxes) == 0:
            print("[WARN] No components found!")
            return [], []
        
        crops = []
        for (x, y, w, h) in boxes:
            pad = 2
            x1 = max(0, x - pad)
            y1 = max(0, y - pad)
            x2 = min(bin_img.shape[1], x + w + pad)
            y2 = min(bin_img.shape[0], y + h + pad)
            crop = bin_img[y1:y2, x1:x2]
            crops.append(crop)
        
        preds = [self.classify_crop(c) for c in crops]
        
        return boxes, preds

    def run_on_image(self, image_path: str, *, name_suffix: str = "single", show: bool = False) -> str:
        gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if gray is None:
            raise FileNotFoundError(f"Could not read image: {image_path}")
        boxes, preds = self.run_on_array(gray)
        out_path = self.draw_predictions(gray, boxes, preds, name_suffix=name_suffix, show=show)
        return out_path

    def run_on_all_thresholds(self, image_path: str, *, show: bool = False, 
                              generate_text: bool = False, translate_text: bool = False,
                              translate_onto_image: bool = False,
                              generate_original_text: bool = False) -> List[str]:
        """Apply threshold methods with classification and optional text/image output.
        
        generate_text          → save a translated (or raw) text file per variant
        translate_text         → when True, text files contain Modern Georgian
        translate_onto_image   → when True, images show Modern Georgian labels
        generate_original_text → save a text file with original script chars per variant
        """
        if TM is None:
            raise ImportError("ThresholdManager could not be imported.")

        original = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if original is None:
            raise FileNotFoundError(f"Could not read image: {image_path}")

        base_filename = os.path.splitext(os.path.basename(image_path))[0]

        tm = TM()
        variants = tm.run_all_Nuskuri_Thresholds(original)

        saved_paths: List[str] = []
        
        for name, binary in variants.items():
            try:
                if not isinstance(binary, np.ndarray):
                    print(f"[ERROR] {name}: not ndarray, skipping")
                    continue
                
                white_pct = np.mean(binary == 255)
                print(f"[INFO] {name}: {white_pct:.1%} white pixels")
                
                nb, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
                num_components = nb - 1
                print(f"[INFO] {name}: {num_components} raw components")
                
                boxes = []
                for i in range(1, nb):
                    x, y, w, h = stats[i][:4]
                    boxes.append((x, y, w, h))
                
                predictions = []
                for (x, y, w, h) in boxes:
                    pad = 2
                    x1 = max(0, x - pad)
                    y1 = max(0, y - pad)
                    x2 = min(binary.shape[1], x + w + pad)
                    y2 = min(binary.shape[0], y + h + pad)
                    crop = binary[y1:y2, x1:x2]
                    
                    try:
                        pred = self.classify_crop(crop)
                        predictions.append(pred)
                    except Exception as e:
                        print(f"[WARN] Classification failed for box at ({x},{y}): {e}")
                        predictions.append("???")
                
                print(f"[INFO] {name}: classified {len(predictions)} components")
                
                # Translated text file (existing option)
                if generate_text and predictions:
                    text_content = self.generate_text_from_boxes(boxes, predictions, binary.shape, translate_text)
                    if text_content:
                        text_path = self.save_text_file(text_content, base_filename, name, label="translated" if translate_text else "")
                        print(f"[INFO] {name}: saved text file {os.path.basename(text_path)}")

                # Original-script text file (new option)
                if generate_original_text and predictions:
                    orig_content = self.generate_text_from_boxes(boxes, predictions, binary.shape, translate_text=False)
                    if orig_content:
                        orig_path = self.save_text_file(orig_content, base_filename, name, label="original")
                        print(f"[INFO] {name}: saved original text file {os.path.basename(orig_path)}")
                
                # Draw visualization (optionally with translated labels on image)
                vis_bgr = cv2.cvtColor(original, cv2.COLOR_GRAY2BGR)
                
                for (x, y, w, h) in boxes:
                    cv2.rectangle(vis_bgr, (x, y), (x+w, y+h), (0, 0, 255), 1)
                
                vis_rgb = cv2.cvtColor(vis_bgr, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(vis_rgb)
                draw = ImageDraw.Draw(pil_img)
                
                try:
                    if self.font_path and os.path.exists(self.font_path):
                        font = ImageFont.truetype(self.font_path, 14)
                    else:
                        font = ImageFont.load_default()
                except:
                    font = ImageFont.load_default()
                
                for (x, y, w, h), pred in zip(boxes, predictions):
                    text_y = y - 5 if y >= 15 else y + h + 15
                    label = (
                        self.nuskhuri_to_modern.get(pred, pred)
                        if translate_onto_image
                        else pred
                    )
                    draw.text((x, text_y), label, fill=(0, 0, 255), font=font)
                
                vis = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
                
                out_path = os.path.join(self.output_dir, f"nuskhuri_ocr_result_{base_filename}_{name}.png")
                cv2.imwrite(out_path, vis)
                saved_paths.append(out_path)
                print(f"[INFO] {name}: saved with {len(predictions)} predictions")
                
            except Exception as e:
                print(f"[ERROR] {name}: {e}")
                import traceback
                traceback.print_exc()

        return saved_paths


__all__ = ["NuskhuriOCR", "NuskhuriDynamicCNN"]