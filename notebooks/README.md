# 📓 Emotion Detection Training Notebooks

This directory contains Jupyter notebooks for training both CNN and YOLO emotion detection models with comprehensive evaluation and visualization.

## 📋 Available Notebooks

### 1. `cnn_emotion_training.ipynb`
- **Purpose**: Train a CNN model using the traditional emotion dataset
- **Dataset**: Uses `../data/train/` and `../data/val/` folders
- **Architecture**: ResNet18-based CNN with custom classifier
- **Output**: `cnn_emotion_model.pth`

### 2. `yolo_emotion_training.ipynb`
- **Purpose**: Train a YOLOv11 model using the YOLO emotion dataset
- **Dataset**: Uses `../data/Facial Emotion.v1i.yolov11/` dataset
- **Architecture**: YOLOv11 nano for object detection + classification
- **Output**: `yolo_emotion_model.pt`

## 🚀 Quick Start

### 1. Install Dependencies
```bash
# Install notebook requirements
pip install -r requirements_notebooks.txt

# Or install individually
pip install torch torchvision ultralytics jupyter matplotlib seaborn scikit-learn
```

### 2. Start Jupyter
```bash
# From the emotion-detection directory
jupyter notebook notebooks/
```

### 3. Run Training
1. Open either notebook in Jupyter
2. Run all cells sequentially
3. Monitor training progress and visualizations
4. Check saved models in `../checkpoints/`

## 📊 What You'll Get

### Comprehensive Visualizations
- **Dataset Distribution**: Bar charts showing class balance
- **Training Curves**: Loss and accuracy over epochs
- **Overfitting Analysis**: Automatic detection of overfitting/underfitting
- **Performance Metrics**: Confusion matrix, ROC curves, classification reports
- **Sample Predictions**: Visual results on test images

### Model Evaluation
- **Accuracy Metrics**: Overall and per-class accuracy
- **F1, Precision, Recall**: Weighted averages
- **Confusion Matrix**: Normalized heatmap
- **ROC Analysis**: One-vs-rest curves for each emotion
- **Performance Assessment**: Automatic status evaluation

### Saved Outputs
- **Models**: Best performing models saved automatically
- **Metrics**: JSON files with comprehensive training statistics
- **Visualizations**: High-quality PNG plots saved to `../visualizations/`

## 🎯 Model Comparison

| Aspect | CNN Model | YOLO Model |
|--------|-----------|------------|
| **Task** | Classification only | Detection + Classification |
| **Input** | Cropped face images | Full images with bounding boxes |
| **Output** | Emotion probabilities | Bounding boxes + emotion labels |
| **Speed** | Fast inference | Moderate inference |
| **Use Case** | Pre-cropped faces | Real-world images |

## 📈 Performance Analysis

Both notebooks include automatic performance analysis:

### ✅ Good Performance Indicators
- Validation accuracy > 70%
- Training-validation gap < 5%
- Stable learning curves
- Balanced confusion matrix

### ⚠️ Warning Signs
- **Overfitting**: Training accuracy >> Validation accuracy
- **Underfitting**: Both accuracies plateau at low values
- **Data Issues**: Severe class imbalance or poor quality

### 🔧 Automatic Recommendations
The notebooks provide actionable recommendations:
- **Overfitting**: Increase regularization, add dropout, get more data
- **Underfitting**: Increase model complexity, train longer
- **Class Imbalance**: Use weighted loss, data augmentation

## 🛠️ Customization Options

### CNN Model
```python
# Modify in notebook
config.batch_size = 32        # Adjust for your GPU memory
config.learning_rate = 0.001  # Learning rate
config.num_epochs = 50        # Maximum epochs
config.patience = 10          # Early stopping patience
```

### YOLO Model
```python
# Modify in notebook
config.epochs = 100           # Maximum epochs
config.batch_size = 16        # Adjust for your GPU memory
config.img_size = 640         # Input image size
config.model_size = 'yolo11n' # Model size (n/s/m/l/x)
```

## 📁 Output Structure

After running the notebooks:

```
emotion-detection/
├── checkpoints/
│   ├── cnn_emotion_model.pth     # CNN model
│   ├── yolo_emotion_model.pt     # YOLO model
│   ├── cnn_metrics.json          # CNN training metrics
│   └── yolo_metrics.json         # YOLO training metrics
└── visualizations/
    ├── dataset_distribution.png
    ├── cnn_training_analysis.png
    ├── cnn_evaluation_metrics.png
    ├── yolo_dataset_distribution.png
    ├── yolo_training_analysis.png
    └── yolo_sample_predictions.png
```

## 🧪 Testing Your Models

After training, test your models using the main script:

```bash
# Test CNN model
python main.py --model checkpoints/cnn_emotion_model.pth --image test_image.jpg

# Test YOLO model
python main.py --model checkpoints/yolo_emotion_model.pt --image test_image.jpg

# Webcam testing
python main.py --model checkpoints/cnn_emotion_model.pth --webcam
```

## 🔍 Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   ```python
   # Reduce batch size
   config.batch_size = 8  # or smaller
   ```

2. **Slow Training**
   ```python
   # Use smaller model or reduce image size
   config.img_size = 224  # for YOLO
   ```

3. **Poor Performance**
   - Check dataset quality and balance
   - Increase training epochs
   - Adjust learning rate
   - Add data augmentation

### GPU Issues
```bash
# Check GPU availability
python -c "import torch; print(torch.cuda.is_available())"

# Check GPU memory
nvidia-smi
```

## 📚 Understanding the Metrics

### For CNN Models
- **Accuracy**: Overall correct predictions percentage
- **F1-Score**: Harmonic mean of precision and recall
- **Confusion Matrix**: Shows which emotions are confused with others
- **ROC Curves**: True positive vs false positive rates

### For YOLO Models
- **mAP@50**: Mean Average Precision at IoU threshold 0.5
- **mAP@50-95**: Mean Average Precision across IoU thresholds 0.5-0.95
- **Precision**: Correct detections / Total detections
- **Recall**: Correct detections / Total ground truth objects

## 🎓 Best Practices

1. **Data Quality**: Ensure good quality, balanced datasets
2. **Monitoring**: Watch for overfitting during training
3. **Validation**: Always validate on unseen data
4. **Comparison**: Train both models to compare performance
5. **Documentation**: Save metrics and visualizations for analysis

## 🤝 Contributing

To add new features or models:
1. Follow the existing notebook structure
2. Include comprehensive visualizations
3. Add automatic performance analysis
4. Update this README with new features

---

**Happy Training! 🚀**

For questions or issues, check the main project README or create an issue in the repository.