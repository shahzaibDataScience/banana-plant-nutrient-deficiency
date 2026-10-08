# 🌿 Plant Leaf Nutrient Deficiency Detection

Final Year Project — classify plant leaf nutrient deficiencies from leaf images using deep learning (transfer learning with EfficientNetB0).

## 📌 Overview

Many crop diseases and yield losses begin as nutrient deficiencies that show up on leaves. This project trains a convolutional neural network to look at a leaf photo and predict which nutrient the plant is missing (or whether the leaf is healthy).

**Classes (8):** Boron, Calcium, Healthy, Iron, Magnesium, Manganese, Potassium, Sulphur

## 📊 Dataset

- **11,150** leaf images in total
- Split: **7,784** train / **2,242** validation / **1,124** test

## 🧠 Model

- **EfficientNetB0** (transfer learning)
- Two-phase training: train the classifier head on a frozen base first, then fine-tune
- Data augmentation + preprocessing for robustness

## 📈 Results

| Metric | Score |
|---|---|
| Test accuracy | **86.65%** |
| Macro F1-score | **0.87** |

## 📁 Repository structure

```
├── fyp-2025.ipynb   # Training & evaluation notebook
├── ap2.py           # Inference / demo script
├── robo_data/       # Dataset folder
└── .gitignore
```

## 🚀 How to run

1. Install dependencies:
   ```bash
   pip install tensorflow numpy pandas matplotlib scikit-learn
   ```
2. Open `fyp-2025.ipynb` in Jupyter and run the cells top to bottom.
3. Use `ap2.py` to run inference on a new leaf image.

## 🛠️ Tech stack

Python • TensorFlow/Keras • EfficientNetB0 • Jupyter • scikit-learn
