# 🩸 Leukemia Detection Using Deep Learning

Automated leukemia detection system using CNNs on microscopic blood cell images.

## 📋 Project Overview
- Goal: Build AI model to detect leukemia from blood cell images
- Dataset: C-NMC Leukemia Dataset (Gupta & Gupta, 2019)
- Method: Convolutional Neural Networks (CNN) with Transfer Learning
- Deliverable: Trained model + Streamlit web app with Grad-CAM explainability
- Timeline: 9 weeks (Nov 2025 - Jan 2026)

## 📊 Dataset Information
- Total Images: 10,661 training samples
- Classes: 
  - Normal cells (hem): 3,389 images (32%)
  - Leukemia cells (all): 7,272 images (68%)
- Image Size: Resized to 224x224 RGB
- Split: 70% train, 15% validation, 15% test

## 🎯 Current Progress

### Week 1 ✅
- Dataset exploration and visualization
- Learned CNN fundamentals
- Built MNIST practice classifier (98% accuracy)

### Week 2 ✅
- Preprocessing Pipeline: Image resize, normalization, data splitting
- Baseline CNN Model:
  - Architecture: 3 Conv blocks + 2 Dense layers
  - Test Accuracy: 85.0%
  - Leukemia Recall: 94% (critical metric)
  - Precision: 86% (Leukemia), 83% (Normal)
  - Status: Shows overfitting - addressed in transfer learning

### Week 3-4 🚧 (In Progress)
- Transfer learning with ResNet50
- Target: 92-95% accuracy

### Week 5-7 (Planned)
- Model optimization and hyperparameter tuning
- Grad-CAM implementation for explainability

### Week 8-9 (Planned)
- Streamlit web application development
- Deployment

## 🗂️ Repository Structure
```
leukemia-detection-deep-learning/
├── notebooks/      # Training notebooks
├── src/           # Reusable functions
├── models/        # Saved models (.h5)
├── results/       # Graphs and metrics
└── requirements.txt
```

## 🛠️ Technologies Used
- Deep Learning: TensorFlow, Keras
- Image Processing: OpenCV, PIL
- Data Science: NumPy, Pandas, Scikit-learn
- Visualization: Matplotlib, Seaborn
- Deployment: Streamlit (upcoming)

## 📚 References
Gupta, A., & Gupta, R. (2019). ALL Challenge dataset of ISBI 2019. 
[Kaggle Dataset](https://www.kaggle.com/datasets/andrewmvd/leukemia-classification)

## 👨‍💻 Author
Alisha Quaser
Amrita Vishwa Vidhyapeetham
Project Deadline: January 2, 2026

---

Note: This is an educational project for academic purposes.