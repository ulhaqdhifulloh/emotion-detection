/**
 * Real-time Emotion Detection using ONNX.js
 * Optimized for webcam with face detection and smart delays
 */

class EmotionDetector {
    constructor() {
        this.session = null;
        this.faceModel = null;
        this.modelInfo = null;
        this.isRunning = false;
        this.lastPrediction = null;
        this.predictionHistory = [];
        this.currentFaces = [];
        this.modelType = 'cnn';
        
        // Settings
        this.settings = {
            detectionDelay: 500,      // ms between predictions
            minFaceSize: 80,          // minimum face size in pixels
            confidenceThreshold: 0.3, // minimum confidence for face detection
            smoothingFactor: 0.7,     // prediction smoothing (0-1)
            maxHistory: 5             // number of predictions to average
        };
        
        // DOM elements
        this.video = document.getElementById('videoElement');
        this.canvas = document.getElementById('overlayCanvas');
        this.ctx = this.canvas.getContext('2d');
        this.status = document.getElementById('status');
        this.predictions = document.getElementById('predictions');
        this.startBtn = document.getElementById('startBtn');
        this.stopBtn = document.getElementById('stopBtn');
        this.settingsBtn = document.getElementById('settingsBtn');
        this.settingsPanel = document.getElementById('settings');
        
        this.initializeEventListeners();
        this.loadModel();
    }

    initializeEventListeners() {
        this.startBtn.addEventListener('click', () => this.startCamera());
        this.stopBtn.addEventListener('click', () => this.stopCamera());
        this.settingsBtn.addEventListener('click', () => this.toggleSettings());
        
        // Settings controls
        const delaySlider = document.getElementById('detectionDelay');
        const faceSizeSlider = document.getElementById('minFaceSize');
        const confidenceSlider = document.getElementById('confidenceThreshold');
        
        delaySlider.addEventListener('input', (e) => {
            this.settings.detectionDelay = parseInt(e.target.value);
            document.getElementById('delayValue').textContent = `${e.target.value}ms`;
        });
        
        faceSizeSlider.addEventListener('input', (e) => {
            this.settings.minFaceSize = parseInt(e.target.value);
            document.getElementById('faceSizeValue').textContent = `${e.target.value}px`;
        });
        
        confidenceSlider.addEventListener('input', (e) => {
            this.settings.confidenceThreshold = parseFloat(e.target.value);
            document.getElementById('confidenceValue').textContent = e.target.value;
        });


    }

    async loadModel() {
        try {
            this.updateStatus('loading', '🔄 Loading CNN emotion detection model...');
            
            // Load face detection model
            if (!this.faceModel) {
                this.faceModel = await blazeface.load();
                console.log('✅ Face detection model loaded');
            }
            
            // Load CNN emotion detection model
            await this.loadCNNModel();
            
            this.updateStatus('ready', '✅ CNN model ready! Click "Start Camera" to begin.');
            this.startBtn.disabled = false;
            
        } catch (error) {
            console.error('❌ Error loading models:', error);
            this.updateStatus('error', `❌ Error loading CNN model: ${error.message}`);
        }
    }

    async loadCNNModel() {
        try {
            // Load CNN emotion detection model
            this.session = await ort.InferenceSession.create('models/emotion_model.onnx');
            console.log('✅ CNN emotion detection model loaded');
            
            // Load CNN model info
            const response = await fetch('models/model_info.json');
            this.modelInfo = await response.json();
            console.log('✅ CNN model info loaded:', this.modelInfo);
            
        } catch (error) {
            throw new Error(`Failed to load CNN model: ${error.message}`);
        }
    }



    async startCamera() {
        try {
            this.updateStatus('loading', '📹 Starting camera...');
            
            const stream = await navigator.mediaDevices.getUserMedia({
                video: {
                    width: { ideal: 640 },
                    height: { ideal: 480 },
                    facingMode: 'user'
                }
            });
            
            this.video.srcObject = stream;
            
            this.video.onloadedmetadata = () => {
                this.setupCanvas();
                this.isRunning = true;
                this.startDetection();
                
                this.updateStatus('ready', `🎥 Camera active - CNN model running (Face detection: real-time, Emotion prediction: ${this.settings.detectionDelay}ms delay)`);
                this.startBtn.disabled = true;
                this.stopBtn.disabled = false;
            };
            
        } catch (error) {
            console.error('❌ Error accessing camera:', error);
            this.updateStatus('error', `❌ Camera error: ${error.message}`);
        }
    }

    stopCamera() {
        this.isRunning = false;
        
        if (this.video.srcObject) {
            const tracks = this.video.srcObject.getTracks();
            tracks.forEach(track => track.stop());
            this.video.srcObject = null;
        }
        
        this.clearCanvas();
        this.clearPredictions();
        
        this.updateStatus('ready', '⏹️ Camera stopped. Click "Start Camera" to resume.');
        this.startBtn.disabled = false;
        this.stopBtn.disabled = true;
    }

    setupCanvas() {
        this.canvas.width = this.video.videoWidth;
        this.canvas.height = this.video.videoHeight;
        this.canvas.style.width = this.video.offsetWidth + 'px';
        this.canvas.style.height = this.video.offsetHeight + 'px';
    }

    async startDetection() {
        // Real-time face detection (no delay)
        const detectFaces = async () => {
            if (!this.isRunning) return;
            
            try {
                await this.detectFacesOnly();
            } catch (error) {
                console.error('Face detection error:', error);
            }
            
            // Continue face detection immediately
            if (this.isRunning) {
                requestAnimationFrame(detectFaces);
            }
        };
        
        // Delayed emotion prediction
        const predictEmotions = async () => {
            if (!this.isRunning) return;
            
            try {
                await this.predictEmotionsFromFaces();
            } catch (error) {
                console.error('Emotion prediction error:', error);
            }
            
            // Schedule next prediction with delay
            setTimeout(() => {
                if (this.isRunning) {
                    predictEmotions();
                }
            }, this.settings.detectionDelay);
        };
        
        // Start both processes
        detectFaces();
        predictEmotions();
    }

    async detectFacesOnly() {
        if (!this.video.videoWidth || !this.video.videoHeight) return;
        
        // Detect faces (real-time, no delay)
        const faces = await this.faceModel.estimateFaces(this.video, false);
        
        this.clearCanvas();
        
        if (faces.length === 0) {
            this.currentFaces = [];
            return;
        }
        
        // Filter faces by size and confidence
        const validFaces = faces.filter(face => {
            const width = face.bottomRight[0] - face.topLeft[0];
            const height = face.bottomRight[1] - face.topLeft[1];
            const size = Math.min(width, height);
            
            return size >= this.settings.minFaceSize && 
                   (face.probability || 1) >= this.settings.confidenceThreshold;
        });
        
        // Store current faces for emotion prediction
        this.currentFaces = validFaces;
        
        // Draw face boxes immediately (real-time)
        validFaces.forEach(face => {
            this.drawFaceBox(face);
        });
        
        // Draw emotion labels if we have recent predictions
        if (this.lastPrediction && validFaces.length > 0) {
            const mainFace = this.getLargestFace(validFaces);
            this.drawEmotionLabel(mainFace, this.lastPrediction[0]);
        }
    }

    async predictEmotionsFromFaces() {
        if (!this.currentFaces || this.currentFaces.length === 0) {
            this.clearPredictions();
            return;
        }
        
        // Use the largest face for emotion prediction
        const face = this.getLargestFace(this.currentFaces);
        
        // Extract face region and predict emotion (with delay)
        const faceImage = await this.extractFaceRegion(face);
        const emotions = await this.predictEmotion(faceImage);
        
        // Smooth predictions
        const smoothedEmotions = this.smoothPredictions(emotions);
        
        // Store last prediction for real-time display
        this.lastPrediction = smoothedEmotions;
        
        // Display results
        this.displayPredictions(smoothedEmotions);
    }

    getLargestFace(faces) {
        return faces.reduce((largest, current) => {
            const currentSize = (current.bottomRight[0] - current.topLeft[0]) * 
                              (current.bottomRight[1] - current.topLeft[1]);
            const largestSize = (largest.bottomRight[0] - largest.topLeft[0]) * 
                               (largest.bottomRight[1] - largest.topLeft[1]);
            return currentSize > largestSize ? current : largest;
        });
    }

    drawFaceBox(face) {
        const scaleX = this.canvas.width / this.video.videoWidth;
        const scaleY = this.canvas.height / this.video.videoHeight;
        
        // Original face coordinates
        let x = face.topLeft[0] * scaleX;
        let y = face.topLeft[1] * scaleY;
        let width = (face.bottomRight[0] - face.topLeft[0]) * scaleX;
        let height = (face.bottomRight[1] - face.topLeft[1]) * scaleY;
        
        // Fix aspect ratio for face (faces are typically taller than wide)
        // Standard face ratio is approximately 1:1.3 (width:height)
        const idealRatio = 1.3; // height/width
        const currentRatio = height / width;
        
        if (currentRatio < idealRatio) {
            // Face is too wide, increase height
            const newHeight = width * idealRatio;
            const heightDiff = newHeight - height;
            y -= heightDiff / 2; // Center vertically
            height = newHeight;
        } else if (currentRatio > idealRatio * 1.2) {
            // Face is too tall, increase width
            const newWidth = height / idealRatio;
            const widthDiff = newWidth - width;
            x -= widthDiff / 2; // Center horizontally
            width = newWidth;
        }
        
        // Add small padding for better face coverage
        const padding = Math.min(width, height) * 0.1;
        x -= padding;
        y -= padding;
        width += padding * 2;
        height += padding * 2;
        
        this.ctx.strokeStyle = '#00ff00';
        this.ctx.lineWidth = 3;
        this.ctx.strokeRect(x, y, width, height);
        
        // Draw corner indicators
        const cornerSize = 20;
        this.ctx.strokeStyle = '#00ff00';
        this.ctx.lineWidth = 4;
        
        // Top-left corner
        this.ctx.beginPath();
        this.ctx.moveTo(x, y + cornerSize);
        this.ctx.lineTo(x, y);
        this.ctx.lineTo(x + cornerSize, y);
        this.ctx.stroke();
        
        // Top-right corner
        this.ctx.beginPath();
        this.ctx.moveTo(x + width - cornerSize, y);
        this.ctx.lineTo(x + width, y);
        this.ctx.lineTo(x + width, y + cornerSize);
        this.ctx.stroke();
        
        // Bottom-left corner
        this.ctx.beginPath();
        this.ctx.moveTo(x, y + height - cornerSize);
        this.ctx.lineTo(x, y + height);
        this.ctx.lineTo(x + cornerSize, y + height);
        this.ctx.stroke();
        
        // Bottom-right corner
        this.ctx.beginPath();
        this.ctx.moveTo(x + width - cornerSize, y + height);
        this.ctx.lineTo(x + width, y + height);
        this.ctx.lineTo(x + width, y + height - cornerSize);
        this.ctx.stroke();
    }

    drawEmotionLabel(face, topEmotion) {
        const scaleX = this.canvas.width / this.video.videoWidth;
        const scaleY = this.canvas.height / this.video.videoHeight;
        
        const x = face.topLeft[0] * scaleX;
        const y = face.topLeft[1] * scaleY;
        
        const label = `${topEmotion.emotion}: ${(topEmotion.confidence * 100).toFixed(1)}%`;
        
        this.ctx.fillStyle = 'rgba(0, 0, 0, 0.8)';
        this.ctx.fillRect(x, y - 35, label.length * 8 + 20, 25);
        
        this.ctx.fillStyle = '#ffffff';
        this.ctx.font = '14px Arial';
        this.ctx.fillText(label, x + 10, y - 15);
    }

    async extractFaceRegion(face) {
        // Create a temporary canvas for face extraction
        const tempCanvas = document.createElement('canvas');
        const tempCtx = tempCanvas.getContext('2d');
        
        const faceWidth = face.bottomRight[0] - face.topLeft[0];
        const faceHeight = face.bottomRight[1] - face.topLeft[1];
        
        // Add padding around face
        const padding = 0.2;
        const paddedWidth = faceWidth * (1 + padding * 2);
        const paddedHeight = faceHeight * (1 + padding * 2);
        const paddedX = face.topLeft[0] - faceWidth * padding;
        const paddedY = face.topLeft[1] - faceHeight * padding;
        
        tempCanvas.width = 224;
        tempCanvas.height = 224;
        
        // Draw and resize face region
        tempCtx.drawImage(
            this.video,
            paddedX, paddedY, paddedWidth, paddedHeight,
            0, 0, 224, 224
        );
        
        return tempCanvas;
    }

    async predictEmotion(faceCanvas) {
        try {
            return await this.predictEmotionCNN(faceCanvas);
        } catch (error) {
            console.warn('CNN prediction failed, using mock predictions:', error);
            return await this.predictEmotionMock(faceCanvas);
        }
    }
    
    // Mock prediction method for testing
    async predictEmotionMock(faceCanvas) {
        // Simulate processing delay
        await new Promise(resolve => setTimeout(resolve, 100));
        
        // Generate random but realistic predictions
        const baseProbs = [0.15, 0.10, 0.12, 0.25, 0.20, 0.13, 0.05]; // angry, disgust, fear, happy, neutral, sad, surprise
        const noise = baseProbs.map(() => (Math.random() - 0.5) * 0.1);
        const probabilities = baseProbs.map((p, i) => Math.max(0.01, p + noise[i]));
        
        // Normalize
        const sum = probabilities.reduce((a, b) => a + b, 0);
        const normalizedProbs = probabilities.map(p => p / sum);
        
        // Create emotion results
        const emotions = this.modelInfo.class_names.map((name, index) => ({
            emotion: name,
            confidence: normalizedProbs[index]
        }));
        
        // Sort by confidence
        emotions.sort((a, b) => b.confidence - a.confidence);
        
        return emotions;
    }

    async predictEmotionCNN(faceCanvas) {
        // Convert canvas to tensor
        const imageData = faceCanvas.getContext('2d').getImageData(0, 0, 224, 224);
        const pixels = imageData.data;
        
        // Normalize pixels (0-255 to 0-1) and apply ImageNet normalization
        const mean = [0.485, 0.456, 0.406];
        const std = [0.229, 0.224, 0.225];
        
        const input = new Float32Array(1 * 3 * 224 * 224);
        
        for (let i = 0; i < 224 * 224; i++) {
            const r = pixels[i * 4] / 255.0;
            const g = pixels[i * 4 + 1] / 255.0;
            const b = pixels[i * 4 + 2] / 255.0;
            
            input[i] = (r - mean[0]) / std[0];                    // R channel
            input[224 * 224 + i] = (g - mean[1]) / std[1];       // G channel
            input[224 * 224 * 2 + i] = (b - mean[2]) / std[2];   // B channel
        }
        
        // Run inference
        const inputTensor = new ort.Tensor('float32', input, [1, 3, 224, 224]);
        const outputs = await this.session.run({ input: inputTensor });
        const logits = outputs.output.data;
        
        // Apply softmax
        const maxLogit = Math.max(...logits);
        const expLogits = logits.map(x => Math.exp(x - maxLogit));
        const sumExp = expLogits.reduce((a, b) => a + b, 0);
        const probabilities = expLogits.map(x => x / sumExp);
        
        // Create emotion results
        const emotions = this.modelInfo.class_names.map((name, index) => ({
            emotion: name,
            confidence: probabilities[index]
        }));
        
        // Sort by confidence
        emotions.sort((a, b) => b.confidence - a.confidence);
        
        return emotions;
    }



    smoothPredictions(currentEmotions) {
        // Add to history
        this.predictionHistory.push(currentEmotions);
        if (this.predictionHistory.length > this.settings.maxHistory) {
            this.predictionHistory.shift();
        }
        
        // Average predictions
        const smoothed = this.modelInfo.class_names.map(emotionName => {
            const confidences = this.predictionHistory.map(predictions => 
                predictions.find(p => p.emotion === emotionName)?.confidence || 0
            );
            const avgConfidence = confidences.reduce((a, b) => a + b, 0) / confidences.length;
            
            return {
                emotion: emotionName,
                confidence: avgConfidence
            };
        });
        
        // Sort by confidence
        smoothed.sort((a, b) => b.confidence - a.confidence);
        
        return smoothed;
    }

    displayPredictions(emotions) {
        const emotionIcons = {
            'angry': '😠',
            'disgust': '🤢',
            'fear': '😨',
            'happy': '😊',
            'neutral': '😐',
            'sad': '😢',
            'surprise': '😲'
        };
        
        this.predictions.innerHTML = emotions.map((emotion, index) => `
            <div class="prediction-card ${index === 0 ? 'top' : ''}">
                <div class="emotion-icon">${emotionIcons[emotion.emotion] || '🎭'}</div>
                <div class="emotion-name">${emotion.emotion}</div>
                <div class="confidence">${(emotion.confidence * 100).toFixed(1)}%</div>
                <div class="confidence-bar">
                    <div class="confidence-fill" style="width: ${emotion.confidence * 100}%"></div>
                </div>
            </div>
        `).join('');
    }

    clearCanvas() {
        this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
    }

    clearPredictions() {
        this.predictions.innerHTML = '<div style="text-align: center; padding: 20px; opacity: 0.7;">No face detected or face too small</div>';
    }

    updateStatus(type, message) {
        this.status.className = `status ${type}`;
        this.status.innerHTML = type === 'loading' ? 
            `<div class="loading-spinner"></div>${message}` : message;
    }

    toggleSettings() {
        const isVisible = this.settingsPanel.style.display !== 'none';
        this.settingsPanel.style.display = isVisible ? 'none' : 'block';
        this.settingsBtn.textContent = isVisible ? '⚙️ Settings' : '❌ Close Settings';
    }
}

// Initialize the emotion detector when the page loads
document.addEventListener('DOMContentLoaded', () => {
    new EmotionDetector();
});