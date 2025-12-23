# Trained Models

## Files:

### `baseline_cnn_model.h5`
- **Architecture:** Baseline CNN (3 Conv blocks + 2 Dense layers)
- **Training Date:** December 24, 2024
- **Test Accuracy:** 85.0%
- **Leukemia Recall:** 94%
- **Precision:** 86% (Leukemia), 83% (Normal)
- **Epochs:** 20
- **Dataset:** C-NMC Leukemia (10,661 images)
- **Input Size:** 224x224x3
- **Purpose:** Baseline model for comparison

## Model Architecture:
```
Conv2D(32) -> MaxPool -> 
Conv2D(64) -> MaxPool -> 
Conv2D(128) -> MaxPool -> 
Flatten -> Dense(128) -> Dropout(0.5) -> Dense(1, sigmoid)
```

## How to Load:
```python
from tensorflow import keras
model = keras.models.load_model('models/baseline_cnn_model.h5')
```

## Performance:
- Overall Accuracy: 85%
- Precision: 0.86 (Leukemia), 0.83 (Normal)
- Recall: 0.94 (Leukemia), 0.67 (Normal)
- F1-Score: 0.90 (Leukemia), 0.74 (Normal)

## Next Models:
- Transfer learning (ResNet50) - Coming Week 3
- Fine-tuned model with Grad-CAM - Coming Week 6