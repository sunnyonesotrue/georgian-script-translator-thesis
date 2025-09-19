import cv2
import numpy as np
import os
from torch.utils.data import Dataset
from PIL import Image


class ThresholdManager:
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
                cv2.imwrite(os.path.join(outputPath, f"{filename}_Gaussian_NLMD.png"), self.AllFiltersTogether(OGImage))

    
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
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        NLMD = cv2.fastNlMeansDenoising(ThresholdGaussian, None, 10, 7, 21)
        return NLMD

    def otsus_threshold(self, image):
        image = self.ensure_valid_image(image)
        _, thresholdOtsu = cv2.threshold(image, 120, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        closed = cv2.morphologyEx(thresholdOtsu, cv2.MORPH_CLOSE, self.kernel)
        closed = cv2.morphologyEx(thresholdOtsu, cv2.MORPH_CLOSE, self.kernel)
        return closed

    @staticmethod
    def adaptive_median_filter(image, S_max):
        padded_image = np.pad(image, S_max // 2, mode='constant', constant_values=0)
        output_image = np.copy(image)
        rows, cols = image.shape

        for i in range(rows):
            for j in range(cols):
                S = 3
                while S <= S_max:
                    sub_img = padded_image[i:i + S, j:j + S]
                    Z_min = np.min(sub_img)
                    Z_max = np.max(sub_img)
                    Z_m = np.median(sub_img)
                    Z_xy = image[i, j]

                    if Z_min < Z_m < Z_max:
                        output_image[i, j] = Z_xy if Z_min < Z_xy < Z_max else Z_m
                        break
                    S += 2
                else:
                    output_image[i, j] = Z_m
        return output_image

    @staticmethod
    def showImages(*args):
        images = [img for img in args if isinstance(img, np.ndarray)]
        combined = np.hstack(images)
        cv2.imshow("Combined Images", combined)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


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


# ✅ Optional example usage for quick testing
if __name__ == "__main__":
    obj = ThresholdManager("")
    obj.applyAndSaveAll()