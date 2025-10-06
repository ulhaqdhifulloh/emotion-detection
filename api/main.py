#!/usr/bin/env python3
"""
Emotion Detection API
FastAPI-based REST API for real-time emotion detection
"""

from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
import cv2
import numpy as np
import onnxruntime as ort
import json
import io
from PIL import Image
import base64
from typing import List, Dict, Optional
import logging
from datetime import datetime
import os
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Emotion Detection API",
    description="Real-time facial emotion detection using CNN model",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class EmotionDetector:
    def __init__(self, model_path: str, model_info_path: str):
        """Initialize the emotion detector"""
        self.model_path = model_path
        self.model_info_path = model_info_path
        self.session = None
        self.model_info = None
        self.emotion_classes = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
        
        # Load model and info
        self.load_model()
        
    def load_model(self):
        """Load ONNX model and model info"""
        try:
            # Load ONNX model
            self.session = ort.InferenceSession(
                self.model_path,
                providers=['CPUExecutionProvider']  # Use CPU for better compatibility
            )
            
            # Load model info
            if os.path.exists(self.model_info_path):
                with open(self.model_info_path, 'r') as f:
                    self.model_info = json.load(f)
                    self.emotion_classes = self.model_info.get('class_names', self.emotion_classes)
            
            logger.info(f"Model loaded successfully: {self.model_path}")
            logger.info(f"Emotion classes: {self.emotion_classes}")
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise RuntimeError(f"Model loading failed: {e}")
    
    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """Preprocess image for model inference"""
        try:
            # Resize to model input size
            image = cv2.resize(image, (224, 224))
            
            # Convert BGR to RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Normalize to [0, 1]
            image = image.astype(np.float32) / 255.0
            
            # Apply ImageNet normalization
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            image = (image - mean) / std
            
            # Convert to CHW format and add batch dimension
            image = np.transpose(image, (2, 0, 1))  # HWC to CHW
            image = np.expand_dims(image, axis=0)   # Add batch dimension
            
            return image
            
        except Exception as e:
            logger.error(f"Image preprocessing failed: {e}")
            raise ValueError(f"Image preprocessing failed: {e}")
    
    def predict_emotion(self, image: np.ndarray) -> Dict:
        """Predict emotion from image"""
        try:
            # Preprocess image
            input_tensor = self.preprocess_image(image)
            
            # Run inference
            input_name = self.session.get_inputs()[0].name
            outputs = self.session.run(None, {input_name: input_tensor})
            
            # Get predictions
            logits = outputs[0][0]  # Remove batch dimension
            
            # Apply softmax
            exp_logits = np.exp(logits - np.max(logits))
            probabilities = exp_logits / np.sum(exp_logits)
            
            # Create results
            results = []
            for i, (emotion, prob) in enumerate(zip(self.emotion_classes, probabilities)):
                results.append({
                    "emotion": emotion,
                    "confidence": float(prob),
                    "rank": i + 1
                })
            
            # Sort by confidence
            results.sort(key=lambda x: x["confidence"], reverse=True)
            
            # Update ranks
            for i, result in enumerate(results):
                result["rank"] = i + 1
            
            return {
                "predictions": results,
                "top_emotion": results[0]["emotion"],
                "top_confidence": results[0]["confidence"],
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Emotion prediction failed: {e}")
            raise RuntimeError(f"Emotion prediction failed: {e}")

# Initialize global detector
detector = None

@app.on_event("startup")
async def startup_event():
    """Initialize the emotion detector on startup"""
    global detector
    
    model_path = "models/emotion_model.onnx"
    model_info_path = "models/model_info.json"
    
    # Check if model files exist
    if not os.path.exists(model_path):
        logger.error(f"Model file not found: {model_path}")
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    try:
        detector = EmotionDetector(model_path, model_info_path)
        logger.info("Emotion Detection API started successfully")
    except Exception as e:
        logger.error(f"Failed to initialize detector: {e}")
        raise

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Emotion Detection API",
        "version": "1.0.0",
        "status": "running",
        "endpoints": {
            "health": "/health",
            "predict_file": "/predict/file",
            "predict_base64": "/predict/base64",
            "model_info": "/model/info",
            "docs": "/docs"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    global detector
    
    if detector is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return {
        "status": "healthy",
        "model_loaded": detector is not None,
        "timestamp": datetime.now().isoformat()
    }

@app.get("/model/info")
async def get_model_info():
    """Get model information"""
    global detector
    
    if detector is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return {
        "model_path": detector.model_path,
        "emotion_classes": detector.emotion_classes,
        "model_info": detector.model_info,
        "input_shape": [1, 3, 224, 224],
        "output_classes": len(detector.emotion_classes)
    }

@app.post("/predict/file")
async def predict_emotion_file(file: UploadFile = File(...)):
    """Predict emotion from uploaded image file"""
    global detector
    
    if detector is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    # Validate file type
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    try:
        # Read image file
        contents = await file.read()
        
        # Convert to numpy array
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            raise HTTPException(status_code=400, detail="Invalid image file")
        
        # Predict emotion
        result = detector.predict_emotion(image)
        
        # Add metadata
        result["metadata"] = {
            "filename": file.filename,
            "content_type": file.content_type,
            "image_shape": image.shape
        }
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.post("/predict/base64")
async def predict_emotion_base64(data: Dict[str, str]):
    """Predict emotion from base64 encoded image"""
    global detector
    
    if detector is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    if "image" not in data:
        raise HTTPException(status_code=400, detail="Missing 'image' field in request body")
    
    try:
        # Decode base64 image
        image_data = data["image"]
        
        # Remove data URL prefix if present
        if image_data.startswith("data:image"):
            image_data = image_data.split(",")[1]
        
        # Decode base64
        image_bytes = base64.b64decode(image_data)
        
        # Convert to numpy array
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            raise HTTPException(status_code=400, detail="Invalid image data")
        
        # Predict emotion
        result = detector.predict_emotion(image)
        
        # Add metadata
        result["metadata"] = {
            "input_type": "base64",
            "image_shape": image.shape
        }
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.post("/predict/batch")
async def predict_emotion_batch(files: List[UploadFile] = File(...)):
    """Predict emotions for multiple images"""
    global detector
    
    if detector is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    if len(files) > 10:  # Limit batch size
        raise HTTPException(status_code=400, detail="Maximum 10 files per batch")
    
    results = []
    
    for file in files:
        try:
            # Validate file type
            if not file.content_type.startswith("image/"):
                results.append({
                    "filename": file.filename,
                    "error": "File must be an image"
                })
                continue
            
            # Read and process image
            contents = await file.read()
            nparr = np.frombuffer(contents, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if image is None:
                results.append({
                    "filename": file.filename,
                    "error": "Invalid image file"
                })
                continue
            
            # Predict emotion
            result = detector.predict_emotion(image)
            result["filename"] = file.filename
            results.append(result)
            
        except Exception as e:
            results.append({
                "filename": file.filename,
                "error": str(e)
            })
    
    return {
        "batch_results": results,
        "total_processed": len(results),
        "timestamp": datetime.now().isoformat()
    }

@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler"""
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"}
    )

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        workers=1
    )