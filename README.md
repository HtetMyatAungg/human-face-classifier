# human-face-classifier
# 🧠 Human vs Non-Human Face Classifier

A real-time binary image classifier that distinguishes human faces from non-human faces (animals, cartoons, AI-generated) using **MobileNetV2 transfer learning** and live webcam inference.

![Python](https://img.shields.io/badge/Python-3.10+-blue?style=flat-square&logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?style=flat-square&logo=tensorflow)
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green?style=flat-square&logo=opencv)
![Accuracy](https://img.shields.io/badge/Accuracy-99%25-brightgreen?style=flat-square)

---

## 📋 Overview

| | |
|---|---|
| **Model** | MobileNetV2 (transfer learning, ImageNet weights) |
| **Task** | Binary classification — human vs non-human face |
| **Dataset** | 13,414 images (80/20 train/val split) |
| **Accuracy** | 99% validation accuracy |
| **Inference** | 30+ FPS live webcam |
| **Framework** | TensorFlow / Keras |

---

## 🏗️ Architecture

```
Input (224×224×3)
    ↓
MobileNetV2 base — FROZEN (pre-trained on ImageNet)
    ↓
GlobalAveragePooling2D
    ↓
Dropout(0.2)
    ↓
Dense(1, activation='sigmoid')
    ↓
Output: 0.0 (human) → 1.0 (non-human)
```

Transfer learning strategy: the MobileNetV2 base is frozen so only the classification head (~1,281 parameters) is trained, enabling 99% accuracy in just 5 epochs on 13K images.

---

## ⚡ Features

- **Transfer learning** with MobileNetV2 pre-trained on ImageNet
- **Optimised tf.data pipeline** with AUTOTUNE prefetching — parallelised batch loading reduces training time
- **Haar Cascade face detection** — classifies only detected face regions, not the entire frame
- **Batched inference** — all faces in a frame processed in a single forward pass (`model()` instead of `model.predict()`)
- **Model warm-up** — eliminates first-frame latency spike
- **Live FPS counter** displayed on screen
- **Consistent preprocessing** — identical normalisation at training and inference time

---

## 🚀 Setup

### 1. Clone the repo
```bash
git clone https://github.com/HtetMyatAungg/human-face-classifier.git
cd human-face-classifier
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Prepare your dataset

Organise your images in this structure:
```
train/
├── human_faces/
│   ├── img001.jpg
│   └── ...
└── non_human_faces/
    ├── img001.jpg
    └── ...
```

Update `INPUT_DIR` in `human_reconition.py` to point to your `train/` folder.

### 4. Run
```bash
python human_reconition.py
```

The script will:
1. Train the model (5 epochs)
2. Evaluate and print validation accuracy
3. Save the model to `human_recognition.keras`
4. Open your webcam for live inference

Press **`q`** to quit the webcam feed.

---

## ⚙️ Configuration

All settings are at the top of `human_reconition.py`:

| Variable | Default | Description |
|---|---|---|
| `INPUT_DIR` | *(your path)* | Path to training dataset |
| `IMG_SIZE` | `224` | Input image size (MobileNetV2 standard) |
| `BATCH_SIZE` | `32` | Training batch size |
| `EPOCHS` | `5` | Number of training epochs |
| `HUMAN_THRESHOLD` | `0.65` | Sigmoid threshold — raise if real faces misclassify |
| `MODEL_PATH` | `human_recognition.keras` | Where to save the trained model |

---

## 📁 Project Structure

```
human-face-classifier/
├── human_reconition.py     # Main script — training + live inference
├── requirements.txt        # Python dependencies
├── README.md               # This file
└── .gitignore              # Excludes datasets, models, cache
```

---

## 🛠️ Tech Stack

- **Python 3.10+**
- **TensorFlow / Keras** — model building, training, inference
- **MobileNetV2** — pre-trained backbone (ImageNet weights)
- **OpenCV** — webcam capture, Haar Cascade face detection, frame rendering
- **NumPy** — array manipulation and batch preparation

---

## 📊 Results

| Metric | Value |
|---|---|
| Validation Accuracy | **99%** |
| Training Epochs | 5 |
| Training Images | 10,731 |
| Validation Images | 2,683 |
| Inference Speed | **30+ FPS** (CPU) |

---

## 👤 Author

**Htet Myat Aung (Henry)**
BSc Computer Science (Artificial Intelligence) — Royal Holloway, University of London

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue?style=flat-square&logo=linkedin)](https://linkedin.com/in/htetmyataung)
[![GitHub](https://img.shields.io/badge/GitHub-Follow-black?style=flat-square&logo=github)](https://github.com/HtetMyatAungg)

---

## 📄 License

MIT License — free to use, modify, and distribute.
