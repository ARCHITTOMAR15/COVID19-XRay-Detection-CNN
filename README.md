# COVID19-XRay-Detection-CNN

# COVID-19 Detection from Chest X-Ray Images using CNN and Transfer Learning
## Live Demo

### Streamlit Application:
[[https://your-app-name.streamlit.app](https://share.streamlit.io/user/archittomar15)](https://covid19-xray-detection-cnn-coukabrlrqeknojrg3racn.streamlit.app)

## Project Overview

This project focuses on the automated detection of COVID-19 from Chest X-Ray images using Deep Learning and Computer Vision techniques.

The goal is to classify chest X-ray scans into three categories:

- COVID-19
- Normal
- Viral Pneumonia

Multiple deep learning architectures were explored, including:

1. Basic CNN
2. VGG16 Transfer Learning
3. VGG16 with Data Augmentation
4. ResNet50 with Class Weights and Early Stopping

Among all models, VGG16 achieved the best overall performance with an accuracy of **98.48%** on the test dataset.

---

## Dataset

Dataset Source:

🔗 https://www.kaggle.com/datasets/pranavraikokte/covid19-image-dataset

### Classes

- Covid
- Normal
- Viral Pneumonia

### Image Processing

- Images resized to 224 × 224
- Pixel normalization (0–1)
- Label encoding
- Train / Validation / Test split

---

## Technologies Used

- Python
- TensorFlow
- Keras
- OpenCV
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Scikit-Learn
- Streamlit

---

## Model Architecture

### VGG16 Transfer Learning

Pre-trained VGG16 model with ImageNet weights:

- VGG16 Base Model (Frozen Layers)
- Global Average Pooling Layer
- Dense Layer (128 neurons, ReLU)
- Dropout (0.5)
- Dense Layer (64 neurons, ReLU)
- Output Layer (3 neurons, Softmax)

### Fine-Tuning

- Initial training with frozen layers
- Selected VGG16 layers unfrozen
- Fine-tuned with a lower learning rate
- 
## Results

### VGG16 Performance

| Metric | Score |
|----------|---------|
| Test Accuracy | 98.48% |
| Test Loss | 0.0888 |
| Weighted F1 Score | 0.98 |
| ROC-AUC Score | 0.9971 |

---
## Confusion Matrix

| Actual \ Predicted | Covid | Normal | Viral Pneumonia |
|-------------------|--------|---------|----------------|
| Covid | 26 | 0 | 0 |
| Normal | 0 | 19 | 1 |
| Viral Pneumonia | 0 | 0 | 20 |

---

## Classification Report

| Class | Precision | Recall | F1-Score |
|---------|-----------|---------|-----------|
| Covid | 1.00 | 1.00 | 1.00 |
| Normal | 1.00 | 0.95 | 0.97 |
| Viral Pneumonia | 0.95 | 1.00 | 0.98 |

### Overall Performance

- Accuracy: 98%
- Macro Average F1 Score: 98%
- Weighted Average F1 Score: 98%

---

## Model Comparison

| Model | Test Accuracy | Weighted F1 Score | ROC-AUC |
|---------|-------------|------------------|----------|
| Basic CNN | 66.67% | 0.66 | N/A |
| VGG16 | 98.48% | 0.98 | 0.9971 |
| VGG16 + Data Augmentation | 95.00% | 0.95 | 0.9978 |
| ResNet50 | 72.73% | 0.71 | 0.9010 |

### Best Model

VGG16 Transfer Learning

Reasons:
- Highest Accuracy
- Strong Precision and Recall
- Excellent ROC-AUC Score
- Minimal Misclassifications

---

