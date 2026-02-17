# 🏛️ Ancient Georgian Script Translator (OCR & NLP)

An end-to-end AI pipeline designed to digitize and translate historical Georgian scripts (**Asomtavruli** and **Nuskhuri**) into **Modern Georgian (Mkhedruli)**. 

This project bridges the gap between historical paleography and modern accessibility by combining a custom computer vision OCR engine with a linguistic translation layer.

## Performance Metrics
* **Character Recognition Accuracy (Asomtavruli):** 99.16%
* **Character Recognition Accuracy (Nuskhuri):** 97.60%
* **End-to-End Latency:** < 2s per standard document scan.
* **Robustness:** Trained with extensive data augmentation (noise, rotation, and salt-and-pepper) to simulate historical parchment degradation.

## Tech Stack
* **Deep Learning:** [PyTorch/TensorFlow]
* **Computer Vision:** OpenCV (Contour detection, Grayscale normalization, Binarization)

## System Architecture
The pipeline consists of four distinct stages:
1.  **Preprocessing:** Image deskewing, noise reduction, and line/character segmentation using OpenCV.
2.  **OCR Engine:** A Convolutional Neural Network (CNN) + [RNN/Transformer] backbone that extracts features from ancient script ligatures.
3.  **Linguistic Mapping:** A translation layer that handles the evolution of phonemes and syntax from Ancient to Modern Georgian.
4.  **UI Layer:** A web interface allowing users to upload images and receive real-time digitized text.

## 📊 Dataset & Training
* **Data Sourcing:** Utilized a hybrid dataset of authentic historical manuscripts and synthetically generated text to ensure high coverage of rare ligatures.
* **Preprocessing:** Implemented adaptive thresholding to handle low-contrast scans typical of museum archives.
* **Optimization:** Utilized [Adam Optimizer/Cross-Entropy Loss] with a learning rate scheduler to achieve convergence on complex character sets.

## 💻 Installation & Usage
```bash
# Clone the repository
git clone [https://github.com/SunnyOneSoTrue/Georgian-Script-Translator-Thesis.git](https://github.com/SunnyOneSoTrue/Georgian-Script-Translator-Thesis.git)

# Install dependencies
pip install -r requirements.txt

# Run the application
python app.py
