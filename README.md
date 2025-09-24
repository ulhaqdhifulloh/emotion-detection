# Emotion Detection (CV) – 5-class API

## Data
- Folder: data/train & data/val dengan 7 kelas (angry, disgust, fear, happy, neutral, sad, surprise)
- Mapping ke 5 kelas internal: anger, joy, sad, fear, neutral (API menampilkan 'love' sebagai alias dari 'neutral' bila diperlukan)

## Training
python src/train.py --data_root data --out_dir checkpoints --epochs 15 --bs 64 --lr 3e-4

## Inference (Webcam)
python src/test_cam.py

## API (FastAPI)
uvicorn serving.app:app --host 0.0.0.0 --port 8080

### Endpoints
- POST /predict (form-data: file)
- POST /predict_b64 (JSON: image_base64)
- GET  /labels
- GET  /healthz

## Docker
docker build -t emotion-api -f serving/Dockerfile .
docker run -p 8080:8080 -e CKPT_PATH=/app/best.pt emotion-api