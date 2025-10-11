# Emotion Detection API (FastAPI) — Panduan

Server FastAPI untuk menyajikan model emosi PyTorch dengan endpoint sederhana dan konfigurasi minimal. Image Docker hanya berisi API (tanpa website).

## Fitur

- Endpoint: `GET /health`, `POST /predict` (gambar tunggal), `POST /predict-batch` (banyak gambar)
- Preprocessing: crop wajah (Haar Cascade), CLAHE, grayscale→RGB, resize+normalize
- Opsi TTA (flip horizontal) dan FP16 autocast di GPU
- CORS diaktifkan (dapat dikonfigurasi via environment variable)

## Kebutuhan

- Python 3.11 (untuk run lokal) atau Docker
- File checkpoint hasil training (contoh: `cnn_emotion_model_v6-2.pth`)

## Environment Variables

- `CHECKPOINT_PATH` (wajib, kecuali Anda mount ke `/models/cnn_emotion_model_v6-2.pth`)
- `IMG_SIZE` (default `224`)
- `CORS_ALLOW_ORIGINS` (default `*`)

## Jalankan Lokal (tanpa Docker)

1. Buat dan aktifkan virtual environment.
2. Install dependency:
   ```bash
   pip install -r requirements.txt
   ```
3. Set environment variable dan mulai server:
   - Windows PowerShell:
     ```powershell
     # Jalankan dari root repo (emotion-detection)
     $env:CHECKPOINT_PATH = ".\\models\\fine-tune\\cnn_emotion_model_v6-2.pth"
     uvicorn api/main:app --host 0.0.0.0 --port 8000
     ```
   - macOS/Linux:
     ```bash
     export CHECKPOINT_PATH="/absolute/path/to/models/fine-tune/cnn_emotion_model_v6-2.pth"
     uvicorn main:app --host 0.0.0.0 --port 8000
     ```

## Docker: Build dan Run (API saja)

Build image dari folder `api`:

```powershell
docker build -t emotion-api .
```

Jalankan container dengan mount checkpoint dan expose port `8000`:

Windows PowerShell (jalankan dari root repo):

```powershell
docker run --rm -p 8000:8000 ^
  -e CHECKPOINT_PATH=/models/fine-tune/cnn_emotion_model_v6-2.pth ^
  -e IMG_SIZE=224 ^
  -e CORS_ALLOW_ORIGINS=* ^
  -v "${PWD}\\models:/models:ro" ^
  emotion-api:latest
```

macOS/Linux (bash, jalankan dari root repo):

```bash
docker run --rm -p 8000:8000 \
  -e CHECKPOINT_PATH=/models/fine-tune/cnn_emotion_model_v6-2.pth \
  -e IMG_SIZE=224 \
  -e CORS_ALLOW_ORIGINS=* \
  -v "$(pwd)/models:/models:ro" \
  emotion-api:latest
```

Catatan:
- Image tidak termasuk website — buka `web-app/index.html` secara terpisah.
- Untuk Linux/macOS, sesuaikan path host pada opsi `-v`.
- Jangan commit file model/cekpoin ke repository (ukuran besar). Gunakan mount Docker (`-v ./models:/models:ro`) atau path relatif lokal untuk `CHECKPOINT_PATH`.

## Endpoint API

- `GET /health`
  - Mengembalikan informasi server, device, class, dan konfigurasi.

- `POST /predict`
  - Form-data: `file` (gambar), query: `tta` (bool), `fp16` (bool)
  - Contoh respons:
    ```json
    {
      "emotion": "joy",
      "confidence": 0.92,
      "probs": {"anger":0.01, "fear":0.02, "joy":0.92, "sad":0.05},
      "entropy": 0.35,
      "latency_ms": 23.5,
      "device": "cpu",
      "tta_used": false,
      "fp16_used": false
    }
    ```

- `POST /predict-batch`
  - Form-data: `files` (banyak gambar), query: `tta`, `fp16`

## Tes Cepat (curl)

Jalankan dari root repo (`emotion-detection/`):

```bash
curl http://localhost:8000/health

curl -X POST "http://localhost:8000/predict?tta=false&fp16=true" \
  -F "file=@./images/test/joy_WIN_20251009_00_23_41_Pro.jpg"

curl -X POST "http://localhost:8000/predict-batch?tta=true&fp16=true" \
  -F "files=@./images/test/joy_WIN_20251009_00_23_41_Pro.jpg" \
  -F "files=@./images/test/sad_WIN_20251009_00_23_53_Pro.jpg"
```

## Integrasi dengan Website (contoh)

Website saat ini mengunggah gambar ke `/predict`. Sketsa kode di frontend:

```js
async function predictEmotionViaAPI(faceCanvas, apiBaseUrl, tta, fp16) {
  const blob = await new Promise(res => faceCanvas.toBlob(res, 'image/jpeg', 0.9));
  const form = new FormData();
  form.append('file', blob, 'face.jpg');

  const url = `${apiBaseUrl}/predict?tta=${tta}&fp16=${fp16}`;
  const resp = await fetch(url, { method: 'POST', body: form });
  return resp.json();
}
```

## Troubleshooting

- "Checkpoint not found": pastikan `CHECKPOINT_PATH` mengarah ke file yang valid dan dapat diakses di container.
- Unggahan gambar besar: gunakan JPEG dan batasi ukuran di sisi klien.
- Error CORS: set `CORS_ALLOW_ORIGINS` ke origin situs Anda (mis. `http://localhost:5500`).

---

API dipelihara di `api/` dengan fokus minimal pada server.