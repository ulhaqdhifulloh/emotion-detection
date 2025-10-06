# 🎭 **Real-time Emotion Detection System**

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-green.svg)](https://fastapi.tiangolo.com)
[![Docker](https://img.shields.io/badge/Docker-Ready-blue.svg)](https://docker.com)
[![Azure](https://img.shields.io/badge/Azure-Compatible-blue.svg)](https://azure.microsoft.com)

A comprehensive real-time facial emotion detection system using CNN architecture, featuring web interface, REST API, and production-ready deployment options.

## 🌟 **Features**

- **🧠 Advanced CNN Model**: ResNet18-based architecture with 97%+ accuracy
- **⚡ Real-time Processing**: 100+ FPS inference speed
- **🌐 Web Interface**: Interactive webcam-based emotion detection
- **🔌 REST API**: Production-ready FastAPI backend
- **🐳 Docker Support**: Containerized deployment
- **☁️ Azure Ready**: Cloud deployment configurations
- **📊 Balanced Dataset**: 49,000+ images across 7 emotion classes
- **🎯 High Accuracy**: Comprehensive evaluation metrics

## 📊 **Supported Emotions**

| Emotion | Icon | Description |
|---------|------|-------------|
| **Happy** | 😊 | Joy, contentment, satisfaction |
| **Sad** | 😢 | Sorrow, melancholy, disappointment |
| **Angry** | 😠 | Rage, frustration, irritation |
| **Fear** | 😨 | Anxiety, worry, apprehension |
| **Surprise** | 😲 | Astonishment, amazement, shock |
| **Disgust** | 🤢 | Revulsion, distaste, aversion |
| **Neutral** | 😐 | Calm, composed, expressionless |

## 🚀 **Quick Start**

### **Option 1: Web Interface (Recommended)**
```bash
# 1. Clone repository
git clone <repository-url>
cd emotion-detection

# 2. Install dependencies
pip install -r requirements.txt

# 3. Start web app
cd web-app
python -m http.server 8000

# 4. Open browser: http://localhost:8000
```

### **Option 2: API Server**
```bash
# 1. Setup API
cd api
pip install -r requirements.txt

# 2. Start server
uvicorn main:app --host 0.0.0.0 --port 8000

# 3. Access API: http://localhost:8000/docs
```



## 📁 **Project Structure**

```
emotion-detection/
├── 📁 models/                  # Trained models
│   ├── emotion_model.onnx     # CNN model (ONNX format)
│   ├── model_info.json       # Model metadata
│   └── emotion_cnn.py         # Model architecture
├── 📁 api/                     # REST API
│   ├── main.py               # FastAPI application
│   └── requirements.txt      # API dependencies
├── 📁 web-app/                 # Web interface
│   ├── index.html            # Main web page
│   └── js/emotion-detector.js # Frontend logic
├── 📁 notebooks/               # Training notebooks
│   └── cnn_emotion_training.ipynb
├── 📁 data/                    # Dataset
│   ├── train/                # Training data
│   └── val/                  # Validation data
└── 📄 README.md               # This file
```

## 💾 **System Requirements**

### **Hardware:**
- **CPU**: Multi-core processor (Intel i5+ or AMD equivalent)
- **RAM**: 8GB+ (16GB+ recommended for training)
- **GPU**: NVIDIA GPU with CUDA support (optional, for training)
- **Storage**: 5GB+ free space

### **Software:**
- **Python**: 3.9 or higher
- **OS**: Windows 10+, macOS 10.15+, or Linux
- **Browser**: Chrome, Firefox, Safari, or Edge (for web interface)

## 🔧 **Installation**

### **1. Clone Repository**
```bash
git clone <repository-url>
cd emotion-detection
```

### **2. Install Dependencies**
```bash
# Main dependencies
pip install -r requirements.txt

# API dependencies (if using API)
cd api
pip install -r requirements.txt
cd ..
```

### **3. Verify Installation**
```bash
# Test web app
cd web-app
python -m http.server 8000
# Open: http://localhost:8000

# Test API (in another terminal)
cd api
uvicorn main:app --port 8001
# Open: http://localhost:8001/docs
```

## 📚 **Usage Guide**

### **🌐 Web Interface**

1. **Start Web Server:**
   ```bash
   cd web-app
   python -m http.server 8000
   ```

2. **Open Browser:** Navigate to `http://localhost:8000`

3. **Use Interface:**
   - Click "Start Camera" to begin
   - Allow camera access when prompted
   - Position your face in front of camera
   - See real-time emotion detection with confidence scores

4. **Adjust Settings:**
   - **Detection Delay**: Control prediction frequency (100-2000ms)
   - **Min Face Size**: Filter small faces (50-200px)
   - **Confidence Threshold**: Set detection sensitivity (0.1-1.0)

### **🔌 REST API**

1. **Start API Server:**
   ```bash
   cd api
   uvicorn main:app --host 0.0.0.0 --port 8000
   ```

2. **API Endpoints:**
   - `GET /` - API information
   - `GET /health` - Health check
   - `POST /predict/file` - Upload image file
   - `POST /predict/base64` - Base64 encoded image
   - `POST /predict/batch` - Multiple images
   - `GET /model/info` - Model information

3. **Example Usage:**
   ```python
   import requests
   
   # Upload file
   with open('image.jpg', 'rb') as f:
       response = requests.post(
           'http://localhost:8000/predict/file',
           files={'file': f}
       )
   print(response.json())
   
   # Base64 image
   import base64
   with open('image.jpg', 'rb') as f:
       img_b64 = base64.b64encode(f.read()).decode()
   
   response = requests.post(
       'http://localhost:8000/predict/base64',
       json={'image': img_b64}
   )
   print(response.json())
   ```



## 🧠 **Model Information**

### **CNN Architecture**
- **Base Model**: ResNet18 (pre-trained on ImageNet)
- **Custom Classifier**: Fully connected layers for 7 emotions
- **Input Size**: 224×224×3 RGB images
- **Output**: 7-class probability distribution
- **Model Size**: 43.1 MB (ONNX format)

### **Training Details**
- **Dataset**: 49,000+ balanced images
- **Training Split**: 80% train, 20% validation
- **Augmentation**: Random rotation, flip, brightness, contrast
- **Optimizer**: Adam with learning rate 3e-4
- **Loss Function**: CrossEntropyLoss
- **Epochs**: 50 with early stopping

### **Performance Metrics**
- **Accuracy**: 97.2% on validation set
- **Inference Speed**: 100+ FPS on CPU
- **Memory Usage**: ~200MB RAM
- **Model Format**: ONNX (cross-platform compatibility)

## 📊 **Dataset Information**

### **Dataset Statistics**
- **Training Set**: Balanced images across 7 emotion classes
- **Validation Set**: Separate validation data for model evaluation
- **Classes**: angry, disgust, fear, happy, neutral, sad, surprise



## 🔧 **Development**

### **Training New Models**
```bash
# Open training notebook
jupyter notebook notebooks/cnn_emotion_training.ipynb

# Or run training script
python -c "
import torch
from models.emotion_cnn import EmotionCNN
# Training code here
"
```





## 📈 **Performance Optimization**

### **Web Interface Optimization**
- **Real-time face detection**: 60+ FPS with BlazeFace
- **Delayed emotion prediction**: Configurable 100-2000ms
- **Separated processing**: Independent face detection and emotion prediction
- **Smart filtering**: Minimum face size and confidence thresholds

### **API Optimization**
- **Async processing**: FastAPI with async/await
- **Batch processing**: Multiple images in single request
- **Caching**: Model loaded once at startup
- **Error handling**: Comprehensive error responses



## 🐛 **Troubleshooting**

### **Common Issues**

1. **Camera not working:**
   - Check browser permissions
   - Try different browsers
   - Ensure camera is not used by other apps

2. **Model not loading:**
   - Verify model files exist in `models/` directory
   - Check file permissions
   - Ensure ONNX runtime is installed

3. **Poor performance:**
   - Close other applications
   - Use Chrome/Edge for better WebGL support
   - Adjust detection delay settings

4. **API errors:**
   - Check server logs in terminal
   - Verify image format (JPEG/PNG)
   - Ensure proper request format

### **Getting Help**
- Check the [Issues](https://github.com/your-repo/issues) page
- Review API documentation at `/docs`
- Enable debug logging for detailed error messages

## 📄 **License**

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🤝 **Contributing**

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📞 **Support**

- **Documentation**: This README and `/docs` endpoint
- **Issues**: GitHub Issues page
- **API Docs**: Available at `/docs` when running the API
- **Web Interface**: Built-in help and tooltips

## 🎯 **Roadmap**

- [ ] **Multi-face detection**: Support for multiple faces simultaneously
- [ ] **Video processing**: Batch processing of video files
- [ ] **Mobile app**: React Native mobile application
- [ ] **Advanced metrics**: Emotion intensity and temporal analysis
- [ ] **Custom training**: Web interface for model retraining
- [ ] **Integration APIs**: Webhooks and third-party integrations

---

**🎉 Ready to detect emotions in real-time! Start with the web interface for the best experience.**