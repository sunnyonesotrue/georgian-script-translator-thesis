import cv2
import numpy as np
import os
from torch.utils.data import Dataset
from PIL import Image


class ThresholdManager:
    """
    UPDATED VERSION with improved Nuskhuri methods
    """
    normalMinValueDiscovered = 148
    valueThatSubtractsInAdaptives = 10
    kernel = np.ones((3, 3), np.uint8)

    def __init__(self, path: str = ""):
        self.path = path or "/Users/sunnysideup/Documents/data for ORCs/Asomtavruli data/raw/toBeProcessed/"
        self.outputPath = "/Users/sunnysideup/Documents/data for ORCs/Asomtavruli data/Thresholded images"
        print(f"ThresholdManager created with path: {self.path}")

    def ensure_valid_image(self, image):
        if image is None:
            raise ValueError("Image is None. Failed to load image. Check the path and file integrity.")
        if image.ndim == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        if image.dtype != np.uint8:
            image = (255 * (image / image.max())).astype(np.uint8)
        return image

    # ========== ASOMTAVRULI METHODS (UNCHANGED) ==========
    def threshold_variants_from_image(self, image: np.ndarray) -> list:
        image = self.ensure_valid_image(image)
        variants = [
            cv2.bitwise_not(self.mean_Threshold_meadianFiltering(image)),
            cv2.bitwise_not(self.mean_Threshold_Closing(image)),
            cv2.bitwise_not(self.gaussian_threshold_median_filtering(image)),
            cv2.bitwise_not(self.gaussian_threshold_morphological_closing(image)),
            cv2.bitwise_not(self.otsus_threshold(image)),
            cv2.bitwise_not(self.mean_Threshold_Bilateral_Filtering(image)),
            cv2.bitwise_not(self.gaussian_threshold_Bilateral_Filtering(image)),
            cv2.bitwise_not(self.mean_Threshold_NLMD(image)),
            cv2.bitwise_not(self.gaussian_Threshold_NLMD(image)),
            cv2.bitwise_not(self.AllFiltersTogether(image)),
        ]
        return variants

    def applyAndSaveAll(self, inputPath: str = None, outputPath: str = None):
        inputPath = inputPath or self.path
        outputPath = outputPath or self.outputPath
        os.makedirs(outputPath, exist_ok=True)

        for filename in os.listdir(inputPath):
            if filename.lower().endswith((".png", ".jpg", ".jpeg", ".tif")):
                path = os.path.join(inputPath, filename)
                OGImage = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
                try:
                    OGImage = self.ensure_valid_image(OGImage)
                except ValueError as e:
                    print(f"❌ Skipping {filename}: {e}")
                    continue

                cv2.imwrite(os.path.join(outputPath, f"{filename}_Mean-Medain.png"), self.mean_Threshold_meadianFiltering(OGImage))
                cv2.imwrite(os.path.join(outputPath, f"{filename}_Mean-Closing.png"), self.mean_Threshold_Closing(OGImage))
                cv2.imwrite(os.path.join(outputPath, f"{filename}_Gaussian-Median.png"), self.gaussian_threshold_median_filtering(OGImage))
                cv2.imwrite(os.path.join(outputPath, f"{filename}_Gaussian-Closing.png"), self.gaussian_threshold_morphological_closing(OGImage))
                cv2.imwrite(os.path.join(outputPath, f"{filename}_Otsu.png"), self.otsus_threshold(OGImage))
                cv2.imwrite(os.path.join(outputPath, f"{filename}_MeanThreshold_Bilateral.png"), self.mean_Threshold_Bilateral_Filtering(OGImage))
                cv2.imwrite(os.path.join(outputPath, f"{filename}_GaussianThreshold_Bilateral.png"), self.gaussian_threshold_Bilateral_Filtering(OGImage))
                cv2.imwrite(os.path.join(outputPath, f"{filename}_Mean_NLMD.png"), self.mean_Threshold_NLMD(OGImage))
                cv2.imwrite(os.path.join(outputPath, f"{filename}_Gaussian_NLMD.png"), self.gaussian_Threshold_NLMD(OGImage))
                cv2.imwrite(os.path.join(outputPath, f"{filename}_Gaussian_allFilters.png"), self.AllFiltersTogether(OGImage))

    def AllFiltersTogether(self, image):
        image = self.ensure_valid_image(image)
        image = self.mean_Threshold_meadianFiltering(image)
        image = self.mean_Threshold_Closing(image)
        image = self.mean_Threshold_Bilateral_Filtering(image)
        image = self.mean_Threshold_NLMD(image)
        image = self.gaussian_threshold_median_filtering(image)
        image = self.gaussian_threshold_morphological_closing(image)
        image = self.gaussian_threshold_Bilateral_Filtering(image)
        image = self.gaussian_Threshold_NLMD(image)
        image = self.otsus_threshold(image)
        return image

    def mean_Threshold_meadianFiltering(self, image):
        image = self.ensure_valid_image(image)
        ThresholdMean = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                              cv2.THRESH_BINARY, 199, self.valueThatSubtractsInAdaptives)
        return cv2.medianBlur(ThresholdMean, 3)

    def mean_Threshold_Closing(self, image):
        image = self.ensure_valid_image(image)
        ThresholdMean = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                              cv2.THRESH_BINARY, 199, self.valueThatSubtractsInAdaptives)
        closed = cv2.morphologyEx(ThresholdMean, cv2.MORPH_CLOSE, self.kernel)
        return cv2.morphologyEx(closed, cv2.MORPH_OPEN, self.kernel)
    
    def mean_Threshold_Bilateral_Filtering(self, image):
        image = self.ensure_valid_image(image)
        ThresholdMean = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                              cv2.THRESH_BINARY, 199, self.valueThatSubtractsInAdaptives)
        bilateral = cv2.bilateralFilter(ThresholdMean, 15, 75, 75)
        return bilateral
    
    def mean_Threshold_NLMD(self, image):
        image = self.ensure_valid_image(image)
        ThresholdMean = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                              cv2.THRESH_BINARY, 199, self.valueThatSubtractsInAdaptives)
        NLMD = cv2.fastNlMeansDenoising(ThresholdMean, None, 10, 7, 21)
        return NLMD

    def gaussian_threshold_median_filtering(self, image):
        image = self.ensure_valid_image(image)
        ThresholdGaussian = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                                  cv2.THRESH_BINARY, 199, self.valueThatSubtractsInAdaptives)
        return cv2.medianBlur(ThresholdGaussian, 3)

    def gaussian_threshold_morphological_closing(self, image):
        image = self.ensure_valid_image(image)
        ThresholdGaussian = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                                  cv2.THRESH_BINARY, 199, self.valueThatSubtractsInAdaptives)
        closed = cv2.morphologyEx(ThresholdGaussian, cv2.MORPH_CLOSE, self.kernel)
        return cv2.morphologyEx(closed, cv2.MORPH_OPEN, self.kernel)
    
    def gaussian_threshold_Bilateral_Filtering(self, image):
        image = self.ensure_valid_image(image)
        ThresholdGaussian = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                                  cv2.THRESH_BINARY, 199, self.valueThatSubtractsInAdaptives)
        bilateral = cv2.bilateralFilter(ThresholdGaussian, 15, 75, 75)
        return bilateral

    def gaussian_Threshold_NLMD(self, image):
        image = self.ensure_valid_image(image)
        ThresholdGaussian = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                                cv2.THRESH_BINARY, 199, self.valueThatSubtractsInAdaptives)
        NLMD = cv2.fastNlMeansDenoising(ThresholdGaussian, None, 10, 7, 21)
        return NLMD

    def otsus_threshold(self, image):
        image = self.ensure_valid_image(image)
        _, thresholdOtsu = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        closed = cv2.morphologyEx(thresholdOtsu, cv2.MORPH_CLOSE, self.kernel)
        closed = cv2.morphologyEx(closed, cv2.MORPH_OPEN, self.kernel)
        return closed

    @staticmethod
    def showImages(*args):
        images = [img for img in args if isinstance(img, np.ndarray)]
        combined = np.hstack(images)
        cv2.imshow("Combined Images", combined)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
        
    # ========== NUSKHURI METHODS (NEW) ==========

    def combo30_nuskhuri_conservative(self, gray):
        """
        CONSERVATIVE version - captures only the most obvious letters.
        Use this if combo29 still has too much noise.
        """
        import cv2
        import numpy as np
        
        # Strong illumination correction
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (61, 61))
        bg = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel)
        norm = cv2.divide(gray, bg, scale=255)
        
        # Strong denoising
        denoised = cv2.bilateralFilter(norm, d=7, sigmaColor=70, sigmaSpace=7)
        
        # Contrast enhancement
        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
        enhanced = clahe.apply(denoised)
        
        # Single adaptive threshold with conservative parameters
        binary = cv2.adaptiveThreshold(
            enhanced, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            31,  # medium window
            10   # higher C = more conservative
        )
        
        # Minimal morphology
        kernel_tiny = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
        cleaned = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_tiny, iterations=1)
        
        # VERY strict filtering
        nb, labels, stats, _ = cv2.connectedComponentsWithStats(cleaned, connectivity=8)
        h, w = cleaned.shape
        output = np.zeros_like(cleaned)
        
        for i in range(1, nb):
            x, y, bw, bh, area = stats[i]
            
            # Strict criteria
            if area < 20 or area > 3000:  # tighter size range
                continue
            if x <= 3 or y <= 3 or x + bw >= w - 3 or y + bh >= h - 3:
                continue
            if bw < 3 or bh < 4:  # slightly larger minimum
                continue
            
            aspect_ratio = bh / bw if bw > 0 else 0
            density = area / (bw * bh) if (bw * bh) > 0 else 0
            
            if aspect_ratio > 12 or aspect_ratio < 0.15:
                continue
            if density < 0.15:  # stricter density
                continue
            
            output[labels == i] = 255
        
        return output
    
    def combo31_ultra_conservative(self, gray):
        """
        ULTRA-CONSERVATIVE with brightness adaptation.
        """
        import cv2
        import numpy as np
        
        # Determine if image is bright (light background) or dark
        mean_intensity = gray.mean()
        is_bright_background = mean_intensity > 127
        
        print(f"  [combo31] Mean intensity: {mean_intensity:.1f}, {'bright' if is_bright_background else 'dark'} background")
        
        if is_bright_background:
            # Dark text on light background - need to invert logic
            # Use UPPER percentile (brightest pixels are background)
            threshold_value = np.percentile(gray, 85)  # darkest 15%
            _, binary = cv2.threshold(gray, threshold_value, 255, cv2.THRESH_BINARY_INV)
        else:
            # Light text on dark background
            threshold_value = np.percentile(gray, 15)  # brightest 15%
            _, binary = cv2.threshold(gray, threshold_value, 255, cv2.THRESH_BINARY)
        
        print(f"  [combo31] Threshold value: {threshold_value:.1f}, white pixels: {np.mean(binary==255):.1%}")
        
        # If we got nothing or everything, try Otsu instead
        white_ratio = np.mean(binary == 255)
        if white_ratio < 0.02 or white_ratio > 0.4:
            print(f"  [combo31] Percentile failed ({white_ratio:.1%}), falling back to Otsu")
            _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
        
        # Morphological cleaning
        kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
        cleaned = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_open, iterations=1)
        
        # Component filtering
        nb, labels, stats, _ = cv2.connectedComponentsWithStats(cleaned, connectivity=8)
        h, w = cleaned.shape
        output = np.zeros_like(cleaned)
        
        kept_count = 0
        for i in range(1, nb):
            x, y, bw, bh, area = stats[i]
            
            # Size filter
            if area < 20 or area > 3000:
                continue
            
            # Edge filter
            if x <= 2 or y <= 2 or x + bw >= w - 2 or y + bh >= h - 2:
                continue
            
            # Dimension filter
            if bw < 3 or bh < 4:
                continue
            
            # Aspect ratio
            aspect = bh / bw if bw > 0 else 0
            if aspect > 12 or aspect < 0.15:
                continue
            
            # Density
            density = area / (bw * bh) if (bw * bh) > 0 else 0
            if density < 0.15:
                continue
            
            output[labels == i] = 255
            kept_count += 1
        
        print(f"  [combo31] Kept {kept_count} components from {nb-1} raw")
        return output
    
    def ultra(self, gray):
        """
        Enhanced version with multi-stage adaptive thresholding and intelligent fallbacks.
        """
        import cv2
        import numpy as np
        
        # === STAGE 1: Image Analysis ===
        mean_intensity = gray.mean()
        std_intensity = gray.std()
        is_bright_background = mean_intensity > 127
        
        print(f"  [combo31] Mean: {mean_intensity:.1f}, Std: {std_intensity:.1f}")
        
        # === STAGE 2: Adaptive Percentile Selection ===
        # Adjust percentile based on contrast level
        if std_intensity < 25:
            # Low contrast: be more aggressive
            percentile_val = 90 if is_bright_background else 10
        elif std_intensity < 40:
            # Medium contrast: balanced
            percentile_val = 88 if is_bright_background else 12
        else:
            # High contrast: be conservative
            percentile_val = 85 if is_bright_background else 15
        
        threshold_value = np.percentile(gray, percentile_val)
        
        if is_bright_background:
            _, binary = cv2.threshold(gray, threshold_value, 255, cv2.THRESH_BINARY_INV)
        else:
            _, binary = cv2.threshold(gray, threshold_value, 255, cv2.THRESH_BINARY)
        
        white_ratio = np.mean(binary == 255)
        print(f"  [combo31] Percentile {percentile_val} -> threshold {threshold_value:.1f} -> {white_ratio:.1%} white")
        
        # === STAGE 3: Intelligent Fallback ===
        if white_ratio < 0.01:
            # Too sparse: try more relaxed threshold
            print(f"  [combo31] Too sparse, relaxing threshold...")
            percentile_val = 80 if is_bright_background else 20
            threshold_value = np.percentile(gray, percentile_val)
            if is_bright_background:
                _, binary = cv2.threshold(gray, threshold_value, 255, cv2.THRESH_BINARY_INV)
            else:
                _, binary = cv2.threshold(gray, threshold_value, 255, cv2.THRESH_BINARY)
            white_ratio = np.mean(binary == 255)
            print(f"  [combo31] Relaxed to {white_ratio:.1%} white")
        
        elif white_ratio > 0.4:
            # Too dense: try Otsu or stricter threshold
            print(f"  [combo31] Too dense, trying Otsu...")
            _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
            white_ratio = np.mean(binary == 255)
            
            if white_ratio > 0.3:
                # Otsu still too dense, use very strict percentile
                print(f"  [combo31] Otsu still dense ({white_ratio:.1%}), using strict percentile...")
                percentile_val = 93 if is_bright_background else 7
                threshold_value = np.percentile(gray, percentile_val)
                if is_bright_background:
                    _, binary = cv2.threshold(gray, threshold_value, 255, cv2.THRESH_BINARY_INV)
                else:
                    _, binary = cv2.threshold(gray, threshold_value, 255, cv2.THRESH_BINARY)
                white_ratio = np.mean(binary == 255)
                print(f"  [combo31] Strict result: {white_ratio:.1%} white")
        
        # === STAGE 4: Adaptive Morphological Cleaning ===
        # Adjust kernel size based on image size and noise level
        h, w = gray.shape
        img_diag = np.sqrt(h*h + w*w)
        kernel_size = max(2, min(4, int(img_diag / 1000)))  # scale with image size
        
        kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        cleaned = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_open, iterations=1)
        
        # Additional closing to reconnect broken strokes
        kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 1))
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel_close, iterations=1)
        
        # === STAGE 5: Intelligent Component Filtering ===
        nb, labels, stats, _ = cv2.connectedComponentsWithStats(cleaned, connectivity=8)
        
        # Calculate adaptive thresholds based on image size
        img_area = h * w
        min_area = max(15, int(img_area * 0.000015))  # ~0.0015% of image
        max_area = min(4000, int(img_area * 0.0008))   # ~0.08% of image
        
        # Collect all component stats first for adaptive filtering
        component_areas = [stats[i, cv2.CC_STAT_AREA] for i in range(1, nb)]
        if component_areas:
            median_area = np.median(component_areas)
            area_std = np.std(component_areas)
            
            # Adjust min_area if we're getting too many tiny components
            if len(component_areas) > 2000:
                min_area = max(min_area, int(median_area * 0.3))
                print(f"  [combo31] Too many components, raising min_area to {min_area}")
        
        output = np.zeros_like(cleaned)
        kept_count = 0
        
        for i in range(1, nb):
            x, y, bw, bh, area = stats[i][:5]
            
            # Size filter (adaptive)
            if area < min_area or area > max_area:
                continue
            
            # Edge filter (adaptive margin)
            margin = max(3, int(min(h, w) * 0.003))
            if x <= margin or y <= margin or x + bw >= w - margin or y + bh >= h - margin:
                continue
            
            # Dimension filter
            min_width = max(3, int(w * 0.002))
            min_height = max(4, int(h * 0.002))
            if bw < min_width or bh < min_height:
                continue
            
            # Aspect ratio (allow taller for decorative capitals)
            aspect = bh / bw if bw > 0 else 0
            if aspect > 15 or aspect < 0.1:
                continue
            
            # Density (relaxed for thin Nuskhuri strokes)
            density = area / (bw * bh) if (bw * bh) > 0 else 0
            if density < 0.12:  # lowered from 0.15
                continue
            
            # Compactness check (perimeter^2 / area)
            # Helps filter noise vs real letter shapes
            contours, _ = cv2.findContours(
                (labels == i).astype(np.uint8),
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE
            )
            if contours:
                perimeter = cv2.arcLength(contours[0], True)
                if perimeter > 0:
                    compactness = (perimeter * perimeter) / (area + 1)
                    # Letters have compactness roughly 15-150
                    # Noise tends to be very high (>200) or very low (<10)
                    if compactness < 10 or compactness > 200:
                        continue
            
            output[labels == i] = 255
            kept_count += 1
        
        print(f"  [combo31] Kept {kept_count} components from {nb-1} raw")
        
        # === STAGE 6: Final Validation ===
        # If we got unreasonable results, warn user
        if kept_count < 20:
            print(f"  [combo31] WARNING: Very few components detected!")
            print(f"  [combo31] Image may be too faded or need manual preprocessing")
        elif kept_count > 1000:
            print(f"  [combo31] WARNING: Still too many components!")
            print(f"  [combo31] Consider manual preprocessing or tighter filters")
        
        return output
    
    def ultra_enhanced(self, gray):
        """
        Multi-strategy adaptive thresholding with intelligent fallbacks.
        """
        import cv2
        import numpy as np
        
        # === STAGE 1: Image Analysis ===
        mean_intensity = gray.mean()
        std_intensity = gray.std()
        is_bright_background = mean_intensity > 127
        
        print(f"  [combo31] Mean: {mean_intensity:.1f}, Std: {std_intensity:.1f}")
        
        # === STAGE 2: Strategy Selection ===
        # Different images need different approaches
        
        if mean_intensity > 170:
            # VERY bright images (like Image 1): aggressive percentile
            percentile_val = 88 if std_intensity > 35 else 85
            strategy = "very_bright"
            
        elif mean_intensity > 127:
            # Medium-bright: balanced
            percentile_val = 80 if std_intensity > 35 else 75
            strategy = "medium_bright"
            
        else:
            # Darker images (like Image 2): need more lenient approach
            if std_intensity > 35:
                percentile_val = 70  # relaxed for high contrast dark images
            else:
                percentile_val = 65  # very relaxed for low contrast dark images
            strategy = "dark"
        
        print(f"  [combo31] Strategy: {strategy}, Percentile: {percentile_val}")
        
        threshold_value = np.percentile(gray, percentile_val)
        
        if is_bright_background:
            _, binary = cv2.threshold(gray, threshold_value, 255, cv2.THRESH_BINARY_INV)
        else:
            _, binary = cv2.threshold(gray, threshold_value, 255, cv2.THRESH_BINARY)
        
        white_ratio = np.mean(binary == 255)
        print(f"  [combo31] Initial threshold {threshold_value:.1f} -> {white_ratio:.1%} white")
        
        # === STAGE 3: Validation and Adjustment ===
        target_min = 0.03  # want at least 3% ink
        target_max = 0.25  # but not more than 25%
        
        if white_ratio < target_min:
            # Too sparse - relax threshold progressively
            print(f"  [combo31] Too sparse, relaxing...")
            
            for relaxed_percentile in [percentile_val - 10, percentile_val - 20, percentile_val - 30]:
                if relaxed_percentile < 50:
                    break
                threshold_value = np.percentile(gray, relaxed_percentile)
                if is_bright_background:
                    _, binary = cv2.threshold(gray, threshold_value, 255, cv2.THRESH_BINARY_INV)
                else:
                    _, binary = cv2.threshold(gray, threshold_value, 255, cv2.THRESH_BINARY)
                
                white_ratio = np.mean(binary == 255)
                print(f"  [combo31]   Try percentile {relaxed_percentile} -> {white_ratio:.1%}")
                
                if white_ratio >= target_min:
                    break
            
        elif white_ratio > target_max:
            # Too dense - try Otsu or stricter percentile
            print(f"  [combo31] Too dense, trying Otsu...")
            _, binary_otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
            otsu_ratio = np.mean(binary_otsu == 255)
            
            if target_min < otsu_ratio < target_max:
                binary = binary_otsu
                white_ratio = otsu_ratio
                print(f"  [combo31] Using Otsu: {white_ratio:.1%}")
            else:
                # Otsu didn't help, use stricter percentile
                strict_percentile = min(95, percentile_val + 10)
                threshold_value = np.percentile(gray, strict_percentile)
                if is_bright_background:
                    _, binary = cv2.threshold(gray, threshold_value, 255, cv2.THRESH_BINARY_INV)
                else:
                    _, binary = cv2.threshold(gray, threshold_value, 255, cv2.THRESH_BINARY)
                white_ratio = np.mean(binary == 255)
                print(f"  [combo31] Stricter percentile {strict_percentile}: {white_ratio:.1%}")
        
        # === STAGE 4: Adaptive Morphology ===
        h, w = gray.shape
        
        # Scale kernel with image size
        base_kernel_size = 2 if min(h, w) < 1500 else 3
        kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (base_kernel_size, base_kernel_size))
        cleaned = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_open, iterations=1)
        
        # Gentle stroke reconnection (horizontal bias for script)
        kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 1))
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel_close, iterations=1)
        
        # === STAGE 5: Adaptive Component Filtering ===
        nb, labels, stats, _ = cv2.connectedComponentsWithStats(cleaned, connectivity=8)
        
        img_area = h * w
        
        # Adaptive size thresholds
        min_area = max(12, int(img_area * 0.000012))
        max_area = min(5000, int(img_area * 0.001))
        
        # Analyze component distribution
        component_areas = [stats[i, cv2.CC_STAT_AREA] for i in range(1, nb)]
        
        if len(component_areas) > 2000:
            # Too many components - raise minimum
            median_area = np.median(component_areas)
            min_area = max(min_area, int(median_area * 0.25))
            print(f"  [combo31] Raising min_area to {min_area} (too many components)")
        elif len(component_areas) < 50:
            # Too few - lower minimum
            min_area = max(8, min_area // 2)
            print(f"  [combo31] Lowering min_area to {min_area} (too few components)")
        
        output = np.zeros_like(cleaned)
        kept_count = 0
        
        for i in range(1, nb):
            x, y, bw, bh, area = stats[i][:5]
            
            # Size filter
            if area < min_area or area > max_area:
                continue
            
            # Edge filter
            margin = max(2, int(min(h, w) * 0.002))
            if x <= margin or y <= margin or x + bw >= w - margin or y + bh >= h - margin:
                continue
            
            # Dimension filter
            if bw < 3 or bh < 4:
                continue
            
            # Aspect ratio (relaxed for decorative capitals)
            aspect = bh / bw if bw > 0 else 0
            if aspect > 18 or aspect < 0.08:
                continue
            
            # Density (very relaxed for thin strokes)
            density = area / (bw * bh) if (bw * bh) > 0 else 0
            if density < 0.10:
                continue
            
            # Compactness check
            contours, _ = cv2.findContours(
                (labels == i).astype(np.uint8),
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE
            )
            if contours and len(contours[0]) > 5:
                perimeter = cv2.arcLength(contours[0], True)
                if perimeter > 0:
                    compactness = (perimeter * perimeter) / (area + 1)
                    if compactness > 250:  # too fragmented
                        continue
            
            output[labels == i] = 255
            kept_count += 1
        
        print(f"  [combo31] Kept {kept_count} components from {nb-1} raw")
        
        # === STAGE 6: Quality Check ===
        if kept_count < 30:
            print(f"  [combo31] WARNING: Very few components!")
        elif kept_count > 800:
            print(f"  [combo31] WARNING: Still noisy (consider manual review)")
        else:
            print(f"  [combo31] Component count looks reasonable")
        
        return output
    
    # ========== MAIN RUNNER ==========
    
    def run_all_Nuskuri_Thresholds(self, gray_img):
        """
        UPDATED: Runs all Nuskhuri-optimized threshold methods.
        Returns dict of {name: binary_image}
        """
        funcs = [
            self.combo30_nuskhuri_conservative,
            self.combo31_ultra_conservative,
            self.ultra,
            self.ultra_enhanced
        ]
        
        out = {}
        for i, f in enumerate(funcs, 1):
            name = f.__name__.replace('_', '-')
            try:
                result = f(gray_img)
                # Ensure white-on-black polarity
                if np.mean(result == 255) > 0.5:
                    result = cv2.bitwise_not(result)
                out[name] = result
            except Exception as e:
                print(f"[WARN] {name} failed: {e}")
        
        return out


# ========== DATASET CLASS (UNCHANGED) ==========

class ThresholdedTestDataset(Dataset):
    def __init__(self, image_folder, transform, threshold_manager):
        self.paths = [
            os.path.join(image_folder, f)
            for f in os.listdir(image_folder)
            if f.lower().endswith((".png", ".jpg", ".jpeg", ".tif"))
        ]
        self.transform = transform
        self.tm = threshold_manager

    def __getitem__(self, idx):
        path = self.paths[idx]
        image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        try:
            image = self.tm.ensure_valid_image(image)
        except ValueError as e:
            raise RuntimeError(f"Failed to process {path}: {e}")
        variants = self.tm.threshold_variants_from_image(image)
        tensor_variants = [self.transform(Image.fromarray(v).convert("L")) for v in variants]
        return tensor_variants, os.path.basename(path)

    def __len__(self):
        return len(self.paths)


if __name__ == "__main__":
    obj = ThresholdManager("")
    obj.applyAndSaveAll()