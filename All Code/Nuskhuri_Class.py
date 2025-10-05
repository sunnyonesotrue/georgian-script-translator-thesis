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

    # ------------------------
    # 🔍 FIXED Segmentation
    # ------------------------
    @staticmethod
    def clean_binary_image(bin_img: np.ndarray, aggressive: bool = True) -> np.ndarray:
        """
        Aggressively clean binary image before segmentation.
        Removes noise while preserving letter structures.
        """
        # 1. Remove tiny isolated specks (salt noise)
        kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
        cleaned = cv2.morphologyEx(bin_img, cv2.MORPH_OPEN, kernel_small, iterations=1)
        
        # 2. Connect broken strokes slightly
        kernel_connect = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 1))
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel_connect, iterations=1)
        
        # 3. Remove components by size and shape
        nb, labels, stats, centroids = cv2.connectedComponentsWithStats(cleaned, connectivity=8)
        h, w = cleaned.shape
        
        output = np.zeros_like(cleaned)
        
        for i in range(1, nb):  # skip background (0)
            x, y, bw, bh, area = stats[i]
            
            # Filter criteria
            aspect_ratio = float(bh) / float(bw) if bw > 0 else 0
            
            # CRITICAL FILTERS:
            # Remove if too small
            if area < 10:
                continue
                
            # Remove if touching image edges (likely page border artifacts)
            if x <= 2 or y <= 2 or x + bw >= w - 2 or y + bh >= h - 2:
                continue
                
            # Remove if dimensions are unrealistic for letters
            if bw < 2 or bh < 3:  # too thin/short
                continue
            if bw > w * 0.3 or bh > h * 0.5:  # too large (likely page artifacts)
                continue
                
            # Remove extreme aspect ratios (noise often has weird shapes)
            if aspect_ratio > 15 or (aspect_ratio < 0.1 and area < 50):
                continue
                
            # Remove if area is too large for single letter
            if area > 5000:
                continue
                
            # Keep this component
            output[labels == i] = 255
        
        return output


    @staticmethod
    def segment_letters(binary_or_gray_img, min_area: int = 15, max_area: int = 4000) -> List[Tuple[int, int, int, int]]:
        """
        IMPROVED segmentation with better noise filtering.
        Returns boxes sorted by reading order (top-to-bottom, left-to-right).
        
        Args:
            binary_or_gray_img: Input image (can be binary, grayscale, or path)
            min_area: Minimum component area (pixels)
            max_area: Maximum component area (pixels)
        """
        th = binary_or_gray_img
        # # Step 1: Normalize to grayscale
        # if isinstance(binary_or_gray_img, np.ndarray):
        #     gray = binary_or_gray_img
        #     if gray.ndim == 3:
        #         gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
        # elif isinstance(binary_or_gray_img, str):
        #     gray = cv2.imread(binary_or_gray_img, cv2.IMREAD_GRAYSCALE)
        #     if gray is None:
        #         raise FileNotFoundError(f"Could not read image: {binary_or_gray_img}")
        # else:
        #     try:
        #         from PIL import Image as _PILImage
        #         if isinstance(binary_or_gray_img, _PILImage.Image):
        #             gray = np.array(binary_or_gray_img.convert("L"), dtype=np.uint8)
        #         else:
        #             raise TypeError()
        #     except Exception:
        #         raise TypeError(f"Unsupported input type: {type(binary_or_gray_img)}")
        #
        # Step 2: Ensure binary with correct polarity
        # u = np.unique(binary_or_gray_img)
        # if len(u) <= 3 and set(int(v) for v in u).issubset({0, 255}):
        #     th = gray
        # else:
        #     _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        
        # CRITICAL: Ensure white text on black background
        white_ratio = np.mean(th == 255)
        if white_ratio > 0.5:
            th = cv2.bitwise_not(th)
        
        # Step 3: CLEAN the binary image aggressively
        # This is the key improvement - remove noise before segmentation
        # cleaned = NuskhuriOCR.clean_binary_image(th, aggressive=True)
        
        # Step 4: Connected components on cleaned image
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            th, connectivity=8
        )
        
        h, w = th.shape
        boxes = []
        
        # Step 5: Extract valid bounding boxes with smart filtering
        for i in range(1, num_labels):
            x, y, bw, bh, area = stats[i]
            
            # Apply size constraints
            if area < min_area or area > max_area:
                continue
            
            # Dimension sanity checks
            if bw < 2 or bh < 3:
                continue
            
            # Skip edge-touching boxes (page borders)
            if x <= 1 or y <= 1 or x + bw >= w - 1 or y + bh >= h - 1:
                continue
            
            # Calculate features for additional filtering
            aspect_ratio = float(bh) / float(bw) if bw > 0 else 0
            density = area / float(bw * bh) if (bw * bh) > 0 else 0
            
            # Filter unrealistic shapes
            if aspect_ratio > 12:  # too tall/thin (likely noise)
                continue
            if aspect_ratio < 0.15 and area < 100:  # too wide/short and small
                continue
            if density < 0.1:  # too sparse (likely broken noise)
                continue
            
            boxes.append((x, y, bw, bh))
        
        # Step 6: Sort by reading order (line-by-line, left-to-right)
        # Use adaptive line height detection
        if boxes:
            heights = [bh for (_, _, _, bh) in boxes]
            median_height = int(np.median(heights)) if heights else 10
            line_threshold = max(median_height // 2, 5)
        else:
            line_threshold = 10
        
        # Sort: group by approximate line (y-coordinate), then by x
        boxes = sorted(boxes, key=lambda b: (b[1] // line_threshold, b[0]))
        
        return boxes


    def _to_tensor(self, crop: np.ndarray) -> torch.Tensor:
        if crop.ndim == 2:
            pil_img = Image.fromarray(crop).convert("RGB")
        else:
            pil_img = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
        
        # Apply transform
        tensor = self.transform(pil_img).unsqueeze(0).to(self.device)
        
        # DEBUG: What does the model actually see?
        # Reverse normalization to see original range
        denorm = tensor.squeeze() * 0.5 + 0.5  # Back to [0, 1]
        print(f"    [Tensor] After norm: min={tensor.min():.3f}, max={tensor.max():.3f}")
        print(f"    [Tensor] Denormalized: min={denorm.min():.3f}, max={denorm.max():.3f}, mean={denorm.mean():.3f}")
    
        return tensor

    def classify_crop(self, crop: np.ndarray) -> str:
        with torch.no_grad():
            # Try original polarity
            logits1 = self.model(self._to_tensor(crop))
            conf1 = torch.softmax(logits1, dim=1).max().item()
            pred1 = int(logits1.argmax(dim=1).item())
            
            # Try inverted polarity
            crop_inv = cv2.bitwise_not(crop)
            logits2 = self.model(self._to_tensor(crop_inv))
            conf2 = torch.softmax(logits2, dim=1).max().item()
            pred2 = int(logits2.argmax(dim=1).item())
            
            # Use whichever has higher confidence
            if conf2 > conf1:
                pred = pred2
                print(f"    [Classify] Used inverted (conf: {conf2:.3f} vs {conf1:.3f})")
            else:
                pred = pred1
                print(f"    [Classify] Used original (conf: {conf1:.3f} vs {conf2:.3f})")
        
        return self.idx_to_class.get(pred, "Error in Classify_Crop")

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
        skip_labels = skip_labels or set([","])

        # Ensure we're drawing on a color image
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
            
            # Draw rectangle
            draw.rectangle([x, y, x + w, y + h], outline="red", width=1)  # Changed to lime green, thicker
            
            # Draw label
            text_y = y - 18 if y - 18 >= 0 else y + h + 2
            draw.text((x, text_y), pred, fill="red", font=font)

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
        """
        IMPROVED: Segment and classify with better diagnostics.
        """
        # Ensure white-on-black
        bin_img = self._ensure_white_text_on_black(binary_or_gray_img)
        
        # Diagnostic
        white_ratio = np.mean(bin_img == 255)
        print(f"[DEBUG] White pixel ratio: {white_ratio:.3f} (target: <0.3 for text)")
        
        # Segment with improved method
        boxes = self.segment_letters(bin_img, min_area=15, max_area=4000)
        print(f"[DEBUG] Found {len(boxes)} components after filtering")
        
        if len(boxes) == 0:
            print("[WARN] No components found! Image may be:")
            print("  - Too noisy (try stricter threshold)")
            print("  - Wrong polarity (check white_ratio above)")
            print("  - Empty/blank")
            return [], []
        
        # Extract crops and classify
        crops = []
        for (x, y, w, h) in boxes:
            # Add small padding to crops (helps classification)
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

    def run_on_all_thresholds_only_boxes(self, image_path: str, *, show: bool = False) -> List[str]:
        """Apply threshold methods - EXACT match to test script logic."""
        if TM is None:
            raise ImportError("ThresholdManager could not be imported.")

        original = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if original is None:
            raise FileNotFoundError(f"Could not read image: {image_path}")

        tm = TM()
        variants = tm.run_all_Nuskuri_Thresholds(original)

        saved_paths: List[str] = []
        
        for name, binary in variants.items():
            try:
                # DO NOT call any conversion - use binary directly from threshold method
                # This matches: binary = method(img) in test script
                
                # Verify it's actually binary
                if not isinstance(binary, np.ndarray):
                    print(f"[ERROR] {name}: not ndarray, skipping")
                    continue
                
                white_pct = np.mean(binary == 255)
                print(f"[INFO] {name}: {white_pct:.1%} white pixels")
                
                # Segment EXACTLY like test script
                nb, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
                num_components = nb - 1
                print(f"[INFO] {name}: {num_components} raw components")
                
                # Draw on original grayscale EXACTLY like test script
                vis = cv2.cvtColor(original, cv2.COLOR_GRAY2BGR)
                
                for i in range(1, min(nb, 1000)):
                    x, y, w, h = stats[i][:4]
                    cv2.rectangle(vis, (x, y), (x+w, y+h), (0, 255, 0), 2)
                
                # Save
                out_path = os.path.join(self.output_dir, f"nuskhuri_ocr_result_{name}.png")
                cv2.imwrite(out_path, vis)
                saved_paths.append(out_path)
                print(f"[INFO] {name}: saved {num_components} boxes")
                
            except Exception as e:
                print(f"[ERROR] {name}: {e}")
                import traceback
                traceback.print_exc()

        return saved_paths
    
    def run_on_all_thresholds(self, image_path: str, *, show: bool = False) -> List[str]:
        """Apply threshold methods with classification."""
        if TM is None:
            raise ImportError("ThresholdManager could not be imported.")

        original = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if original is None:
            raise FileNotFoundError(f"Could not read image: {image_path}")

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
                
                # Get components and classify
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
                
                # === USE PIL FOR DRAWING (supports Unicode) ===
                vis_bgr = cv2.cvtColor(original, cv2.COLOR_GRAY2BGR)
                
                # Draw rectangles with OpenCV (fast)
                for (x, y, w, h) in boxes:
                    cv2.rectangle(vis_bgr, (x, y), (x+w, y+h), (0, 0, 255), 1) # Red boxes (BGR - (0, 0, 255) )
                
                # Convert to PIL for text drawing
                vis_rgb = cv2.cvtColor(vis_bgr, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(vis_rgb)
                draw = ImageDraw.Draw(pil_img)
                
                # Load Georgian font
                try:
                    if self.font_path and os.path.exists(self.font_path):
                        font = ImageFont.truetype(self.font_path, 14)
                    else:
                        font = ImageFont.load_default()
                except:
                    font = ImageFont.load_default()
                
                # Draw text labels with PIL
                for (x, y, w, h), pred in zip(boxes, predictions):
                    text_y = y - 5 if y >= 15 else y + h + 15
                    draw.text((x, text_y), pred, fill=(0, 0, 255), font=font) # Red text (BGR - (0, 0, 255))
                
                # Convert back to BGR for saving
                vis = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
                
                # Save
                out_path = os.path.join(self.output_dir, f"nuskhuri_ocr_result_{name}.png")
                cv2.imwrite(out_path, vis)
                saved_paths.append(out_path)
                print(f"[INFO] {name}: saved with {len(predictions)} predictions")
                
            except Exception as e:
                print(f"[ERROR] {name}: {e}")
                import traceback
                traceback.print_exc()

        return saved_paths

__all__ = ["NuskhuriOCR", "NuskhuriDynamicCNN"]