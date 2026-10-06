# PPE Detection using YOLO26n
# 🦺 PPE Detection using YOLO26n

A Computer Vision object detection project for detecting **Personal Protective Equipment (PPE)** using **YOLO26n** and the **Ultralytics YOLO framework**.

The model is trained to detect five PPE-related classes:

- 🪖 Helmet
- ❌ No Helmet
- 🦺 Vest
- ❌ No Vest
- 👤 Person

The complete training workflow is documented in the included Google Colab/Jupyter notebook:

**`pp_detection_yolo.ipynb`**

---

## 📌 Project Overview

Personal Protective Equipment plays an important role in workplace safety, particularly in construction and industrial environments.

Manual monitoring of PPE compliance can be time-consuming and difficult to scale. This project explores an automated computer vision approach that detects people and PPE-related objects from images and videos.

The trained YOLO26n model performs object detection by predicting:

1. The **class** of the detected object.
2. The **location** of the object using a bounding box.
3. The **confidence score** associated with the prediction.

---

## 🎯 Project Objectives

The main objectives of this project are:

- Detect people in visual data.
- Detect workers wearing helmets.
- Detect workers without helmets.
- Detect safety vests.
- Detect absence of safety vests.
- Train a lightweight YOLO object detection model.
- Evaluate model performance using object-detection metrics.
- Perform inference on images and videos.

---

## 🏷️ Detection Classes

The model detects the following five classes:

| Class ID | Class |
|---:|---|
| 0 | Helmet |
| 1 | No Helmet |
| 2 | No Vest |
| 3 | Person |
| 4 | Vest |

---

## 🧠 Model

### YOLO26n

This project uses **YOLO26n**, the nano variant of the YOLO26 object detection model.

The `n` variant is designed to provide a lightweight model suitable for relatively efficient inference while maintaining useful object detection performance.

The model was trained using the **Ultralytics** implementation.

### Model Information

| Parameter | Value |
|---|---|
| Model | YOLO26n |
| Parameters | 2,375,811 |
| GFLOPs | 5.3 |
| Task | Object Detection |
| Framework | Ultralytics |
| Deep Learning Backend | PyTorch |

---

# 🛠️ Technology Stack

- **Python**
- **YOLO26n**
- **Ultralytics**
- **PyTorch**
- **CUDA**
- **OpenCV**
- **NumPy**
- **PyYAML**
- **Kaggle**
- **Google Colab**
- **NVIDIA Tesla T4 GPU**

PyTorch is used as the deep-learning backend through the Ultralytics YOLO framework.

---

# 📂 Dataset

The project uses a **Personal Protective Equipment (PPE) dataset** containing images and YOLO-format annotations for PPE detection.

The dataset contains the five target classes:

```text
Helmet
No Helmet
No Vest
Person
Vest
```

The dataset itself is **not included in this GitHub repository**.

The dataset configuration is handled through a YOLO `data.yaml` file during training and evaluation.

---

# 🔄 Project Workflow

The overall workflow is:

```text
PPE Dataset
     │
     ▼
Dataset Preparation
     │
     ▼
YOLO Dataset Configuration
     │
     ▼
YOLO26n Model
     │
     ▼
Model Training
     │
     ▼
Validation
     │
     ▼
Performance Evaluation
     │
     ▼
best.pt
     │
     ▼
Image / Video Inference
```

---

# ⚙️ Training Configuration

The model was trained using the following configuration:

| Training Parameter | Value |
|---|---|
| Model | YOLO26n |
| Epochs | 50 |
| Image Size | 640 × 640 |
| Batch Size | 16 |
| Device | CUDA GPU |
| GPU | NVIDIA Tesla T4 |
| Patience | 10 |
| Task | Object Detection |

Training was performed in **Google Colab** using GPU acceleration.

---

# 🏋️ Model Training

The model was initialized using the YOLO26n pretrained model and trained on the PPE dataset.

The main training configuration used in the project follows this approach:

```python
from ultralytics import YOLO

model = YOLO("yolo26n.pt")

results = model.train(
    data="data.yaml",
    epochs=50,
    imgsz=640,
    batch=16,
    device=0,
    patience=10
)
```

The trained model weights were saved after training.

The primary trained checkpoint used for evaluation and inference is:

```text
best.pt
```

---

# 📊 Model Evaluation

The trained model was evaluated using the validation dataset.

### Validation Dataset

- **Images:** 406
- **Instances:** 3,095

### Overall Performance

| Metric | Score |
|---|---:|
| **Precision** | **91.25%** |
| **Recall** | **85.34%** |
| **mAP@50** | **91.60%** |
| **mAP@50-95** | **59.73%** |

---

# 📈 Class-wise Performance

The validation results for individual classes were:

| Class | Precision | Recall | mAP@50 | mAP@50-95 |
|---|---:|---:|---:|---:|
| Helmet | 91.2% | 84.2% | 91.1% | 57.8% |
| No Helmet | 90.1% | 78.8% | 87.6% | 50.8% |
| No Vest | 87.9% | 83.8% | 88.5% | 51.3% |
| Person | 92.1% | 90.2% | 94.7% | 68.9% |
| Vest | 94.9% | 89.7% | 96.0% | 69.9% |

---

# 📐 Evaluation Metrics Explained

## Precision

Precision measures how many of the detections made by the model were correct.

```text
Precision = TP / (TP + FP)
```

The model achieved:

**91.25% Precision**

A high precision indicates that a large proportion of the model's positive detections were correct.

---

## Recall

Recall measures how many of the actual objects were successfully detected.

```text
Recall = TP / (TP + FN)
```

The model achieved:

**85.34% Recall**

This means the model successfully detected a large proportion of the objects present in the validation dataset.

---

## mAP@50

mAP@50 represents mean Average Precision at an Intersection over Union (IoU) threshold of 0.50.

The model achieved:

**91.60% mAP@50**

This is the primary headline object-detection performance metric for this project.

---

## mAP@50-95

mAP@50-95 evaluates Average Precision across multiple IoU thresholds from 0.50 through 0.95.

The model achieved:

**59.73% mAP@50-95**

This is a stricter metric because it evaluates how accurately the model localizes objects at increasingly higher IoU thresholds.

---

# 🧪 Validation Results

The validation run was performed using the trained `best.pt` model.

The evaluation output reported:

```text
Precision   : 0.9125
Recall      : 0.8534
mAP50       : 0.9160
mAP50-95    : 0.5973
```

Converted to percentages:

```text
Precision   : 91.25%
Recall      : 85.34%
mAP@50      : 91.60%
mAP@50-95   : 59.73%
```

---

# 🔍 Best Performing Classes

Based on mAP@50:

| Rank | Class | mAP@50 |
|---:|---|---:|
| 1 | Vest | **96.0%** |
| 2 | Person | **94.7%** |
| 3 | Helmet | **91.1%** |
| 4 | No Vest | **88.5%** |
| 5 | No Helmet | **87.6%** |

The **Vest** class achieved the highest mAP@50 at **96.0%**.

---

# ⚠️ Areas for Improvement

The `No Helmet` class has a recall of **78.8%**, which is lower than the recall of several other classes.

This indicates that some actual no-helmet instances may be missed by the model.

Possible future improvements include:

- Adding more no-helmet training examples.
- Increasing dataset diversity.
- Adding difficult lighting conditions.
- Adding different camera angles.
- Including partially occluded workers.
- Improving annotation quality.
- Experimenting with different model sizes.
- Hyperparameter tuning.
- Training for a larger number of epochs where appropriate.

---

# 🖼️ Image Inference

The trained model can be used for image prediction with Ultralytics.

Example:

```python
from ultralytics import YOLO

model = YOLO("best.pt")

results = model.predict(
    source="image.jpg",
    imgsz=640,
    conf=0.40,
    save=True
)
```

The model generates bounding boxes around detected objects along with class labels and confidence scores.

---

# 🎥 Video Inference

The trained model can also process video files.

Example:

```python
from ultralytics import YOLO

model = YOLO("best.pt")

results = model.predict(
    source="video.mp4",
    imgsz=640,
    conf=0.40,
    save=True
)
```

The model processes the video frames and generates an annotated output containing the detected PPE classes.

---

# 🎚️ Confidence Threshold

The inference examples use:

```text
conf = 0.40
```

This means detections below a confidence score of 40% are filtered out.

The threshold can be adjusted depending on the application.

For example:

```python
conf=0.50
```

can be used when fewer but more confident detections are preferred.

---

# 📓 Notebook

The complete training and evaluation workflow is available in:

```text
pp_detection_yolo.ipynb
```

The notebook contains the project workflow used for dataset preparation, model training, model evaluation, and related experimentation.

---

# 📁 Repository Structure

```text
ppe-detection-yolo26/
│
├── pp_detection_yolo.ipynb
├── README.md
└── ...
```

The training dataset and other large experimental files are not required to be stored in the GitHub repository.

---

# 🚀 How to Use

## 1. Clone the repository

```bash
git clone https://github.com/sumit-teotia/ppe-detection-yolo26.git
```

```bash
cd ppe-detection-yolo26
```

## 2. Install dependencies

Install the required packages:

```bash
pip install ultralytics torch torchvision opencv-python numpy pandas matplotlib pyyaml
```

## 3. Obtain the trained model

Place the trained model checkpoint in your working directory:

```text
best.pt
```

## 4. Run inference

Use the Ultralytics prediction interface with an image or video source.

---

# 💻 Hardware

Training was performed using:

```text
GPU: NVIDIA Tesla T4
CUDA: Enabled
```

The model contained:

```text
2,375,811 parameters
5.3 GFLOPs
```

---

# 📌 Key Results

The final trained model achieved:

### ⭐ 91.60% mAP@50

with:

- **91.25% Precision**
- **85.34% Recall**
- **59.73% mAP@50-95**

The model demonstrated strong detection performance across the five PPE classes, with the highest mAP@50 achieved for the **Vest** class at **96.0%**.

---

# 🔮 Future Improvements

Potential future development includes:

- Real-time webcam inference.
- CCTV-based PPE monitoring.
- Automated PPE violation alerts.
- Real-time safety compliance monitoring.
- Larger and more diverse datasets.
- Model optimization for edge devices.
- ONNX/TensorRT deployment.
- Performance optimization for real-time inference.
- Improved detection of difficult/occluded PPE cases.
- Integration with a monitoring dashboard.

---

# 👨‍💻 Author

## Sumit Teotia

GitHub:  
https://github.com/sumit-teotia

LinkedIn:  
https://www.linkedin.com/in/sumit-teotia13/

---

## 📄 Project Repository

GitHub Repository:

https://github.com/sumit-teotia/ppe-detection-yolo26

---

## ⭐ Project Summary

This project demonstrates an end-to-end **Computer Vision object detection workflow** using YOLO26n for PPE detection.

The workflow covers:

```text
Dataset Preparation
        ↓
YOLO26n Training
        ↓
Validation
        ↓
Performance Evaluation
        ↓
Model Selection
        ↓
Image / Video Inference
```

The final model achieved **91.60% mAP@50** on the validation dataset.
