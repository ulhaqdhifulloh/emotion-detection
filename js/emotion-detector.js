/**
 * Emotion Detection using FastAPI backend with click-to-capture
 * Opens camera; user captures a photo; backend returns prediction
*/

class EmotionDetector {
    constructor() {
        this.modelInfo = null;
        this.isRunning = false;
        this.lastPrediction = null;
        // Default API base URL
        this.defaultApiBaseUrl = 'https://emotion-detection-api.azurewebsites.net/';
        // this.defaultApiBaseUrl = 'http://localhost:3003'; // alternatif lokal
        this.apiBaseUrl = this.defaultApiBaseUrl;
        this.apiMode = 'default'; // 'default' | 'custom'

        // Settings
        this.settings = { tta: false, fp16: true };
        
        // DOM elements
        this.video = document.getElementById('videoElement');
        this.status = document.getElementById('status');
        this.predictions = document.getElementById('predictions');
        this.startBtn = document.getElementById('startBtn');
        this.stopBtn = document.getElementById('stopBtn');
        this.captureBtn = document.getElementById('captureBtn');
        this.settingsBtn = document.getElementById('settingsBtn');
        this.settingsPanel = document.getElementById('settings');
        this.apiBaseUrlInput = document.getElementById('apiBaseUrlInput');
        this.apiModeDefault = document.getElementById('apiModeDefault');
        this.apiModeCustom = document.getElementById('apiModeCustom');
        this.ttaCheckbox = document.getElementById('ttaCheckbox');
        this.fp16Checkbox = document.getElementById('fp16Checkbox');
        
        this.initializeEventListeners();
        this.loadModel();
    }

    // Hapus trailing slash dan spasi supaya tidak jadi //predict
    sanitizeBaseUrl(url) {
        return (url || '').trim().replace(/\/+$/, '');
    }

    initializeEventListeners() {
        this.startBtn.addEventListener('click', () => this.startCamera());
        this.stopBtn.addEventListener('click', () => this.stopCamera());
        this.captureBtn.addEventListener('click', () => this.captureAndPredict());
        this.settingsBtn.addEventListener('click', () => this.toggleSettings());
        if (this.apiModeDefault) {
            this.apiModeDefault.addEventListener('change', () => {
                if (this.apiModeDefault.checked) {
                    this.apiMode = 'default';
                    if (this.apiBaseUrlInput) {
                        this.apiBaseUrlInput.disabled = true;
                        this.apiBaseUrlInput.placeholder = this.defaultApiBaseUrl;
                    }
                    this.apiBaseUrl = this.sanitizeBaseUrl(this.defaultApiBaseUrl);
                }
            });
        }
        if (this.apiModeCustom) {
            this.apiModeCustom.addEventListener('change', () => {
                if (this.apiModeCustom.checked) {
                    this.apiMode = 'custom';
                    if (this.apiBaseUrlInput) {
                        this.apiBaseUrlInput.disabled = false;
                        const val = this.sanitizeBaseUrl(this.apiBaseUrlInput.value || '');
                        this.apiBaseUrl = val || this.sanitizeBaseUrl(this.defaultApiBaseUrl);
                    }
                }
            });
        }
        if (this.apiBaseUrlInput) {
            this.apiBaseUrlInput.addEventListener('change', () => {
                if (this.apiMode === 'custom') {
                    const val = this.sanitizeBaseUrl(this.apiBaseUrlInput.value || '');
                    this.apiBaseUrl = val || this.sanitizeBaseUrl(this.defaultApiBaseUrl);
                }
            });
        }
        if (this.ttaCheckbox) {
            this.ttaCheckbox.addEventListener('change', () => {
                this.settings.tta = !!this.ttaCheckbox.checked;
            });
        }
        if (this.fp16Checkbox) {
            this.fp16Checkbox.addEventListener('change', () => {
                this.settings.fp16 = !!this.fp16Checkbox.checked;
            });
        }
    }

    async loadModel() {
        try {
            this.updateStatus('loading', '🔄 Initializing API...');

            // Load config for API base URL and inference settings
            await this.loadConfig();

            // Initialize API (fetch health and classes)
            await this.initAPI();

            this.updateStatus('ready', '✅ Backend siap! Klik "Start Camera" lalu "Capture Photo".');
            this.startBtn.disabled = false;

            // Reflect settings into UI inputs
            // Default: use default mode and disable input
            if (this.apiBaseUrlInput) {
                this.apiBaseUrlInput.value = '';
                this.apiBaseUrlInput.disabled = true;
                this.apiBaseUrlInput.placeholder = this.sanitizeBaseUrl(this.defaultApiBaseUrl);
            }
            if (this.apiModeDefault) this.apiModeDefault.checked = true;
            if (this.apiModeCustom) this.apiModeCustom.checked = false;
            if (this.ttaCheckbox) this.ttaCheckbox.checked = !!this.settings.tta;
            if (this.fp16Checkbox) this.fp16Checkbox.checked = !!this.settings.fp16;
        } catch (error) {
            console.error('❌ Initialization error:', error);
            this.updateStatus('error', `❌ Init error: ${error.message}`);
        }
    }

    async loadConfig() {
        try {
            const resp = await fetch('models/web_config.json');
            if (resp.ok) {
                const cfg = await resp.json();
                // Optional API base URL in config
                if (cfg.api_base_url) {
                    this.apiBaseUrl = this.sanitizeBaseUrl(cfg.api_base_url);
                }
                // Apply API/inference settings if present
                const api = cfg.api_settings || cfg.inference_settings || {};
                this.settings.tta = api.tta ?? this.settings.tta;
                this.settings.fp16 = api.fp16 ?? this.settings.fp16;
            }
        } catch (e) {
            console.warn('Failed to load web_config.json, using defaults:', e);
        }
    }

    async initAPI() {
        // Probe health to get classes and device
        try {
            const base = this.sanitizeBaseUrl(this.apiBaseUrl);
            const resp = await fetch(`${base}/health`);
            if (!resp.ok) throw new Error(`Health status ${resp.status}`);
            const data = await resp.json();
            const classes = Array.isArray(data.classes) ? data.classes : ['anger','fear','joy','sad'];
            this.modelInfo = { class_names: classes };
            console.log('✅ API health:', data);
        } catch (e) {
            console.warn('API health failed, falling back to default classes:', e);
            // Fallback: read local model_info.json (if present) or default 4 classes
            try {
                const r = await fetch('models/model_info.json');
                if (r.ok) {
                    const info = await r.json();
                    this.modelInfo = { class_names: info.class_names };
                } else {
                    this.modelInfo = { class_names: ['anger','fear','joy','sad'] };
                }
            } catch (_) {
                this.modelInfo = { class_names: ['anger','fear','joy','sad'] };
            }
        }
    }

    // Using API backend only



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
                this.isRunning = true;
                this.updateStatus('ready', `🎥 Camera active - Click "Capture Photo" to predict`);
                this.startBtn.disabled = true;
                this.stopBtn.disabled = false;
                this.captureBtn.disabled = false;
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
        this.clearPredictions();
        
        this.updateStatus('ready', '⏹️ Camera stopped. Click "Start Camera" to resume.');
        this.startBtn.disabled = false;
        this.stopBtn.disabled = true;
        this.captureBtn.disabled = true;
    }
    async captureAndPredict() {
        if (!this.video || !this.video.videoWidth) return;
        // Capture current frame
        const canvas = document.createElement('canvas');
        const ctx = canvas.getContext('2d');
        canvas.width = this.video.videoWidth;
        canvas.height = this.video.videoHeight;
        ctx.drawImage(this.video, 0, 0, canvas.width, canvas.height);
        
        // Predict via API with the full frame (server will crop face)
        try {
            const emotions = await this.predictEmotionViaAPI(canvas);
            this.lastPrediction = emotions;
            this.displayPredictions(emotions);
        } catch (error) {
            console.error('Prediction error:', error);
            this.updateStatus('error', `❌ Prediction error: ${error.message}`);
        }
    }

    async predictEmotion(faceCanvas) {
        try {
            return await this.predictEmotionViaAPI(faceCanvas);
        } catch (error) {
            console.warn('API prediction failed, using mock predictions:', error);
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

    async predictEmotionViaAPI(faceCanvas) {
        const blob = await new Promise(res => faceCanvas.toBlob(res, 'image/jpeg', 0.9));
        const form = new FormData();
        form.append('file', blob, 'face.jpg');

        const base = this.sanitizeBaseUrl(this.apiBaseUrl);
        const url = `${base}/predict?tta=${this.settings.tta}&fp16=${this.settings.fp16}`;
        const resp = await fetch(url, { method: 'POST', body: form });
        if (!resp.ok) throw new Error(`API status ${resp.status}`);
        const data = await resp.json();

        const emotions = Object.entries(data.probs)
            .map(([emotion, confidence]) => ({ emotion, confidence }))
            .sort((a, b) => b.confidence - a.confidence);
        // Ensure modelInfo class_names reflect API classes for smoothing
        if (!this.modelInfo || !Array.isArray(this.modelInfo.class_names)) {
            this.modelInfo = { class_names: emotions.map(e => e.emotion) };
        }
        return emotions;
    }



    // Removed smoothing history; predictions shown per capture

    displayPredictions(emotions) {
        const emotionIcons = {
            'anger': '😠',
            'disgust': '🤢',
            'fear': '😨',
            'joy': '😊',
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

    // No overlay canvas to clear

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