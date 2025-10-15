# Emotion Detection API (FastAPI) — Panduan Deploy Render

Server FastAPI untuk menyajikan model emosi PyTorch dengan endpoint sederhana. Folder ini siap diunggah langsung ke Render (runtime Python, tanpa Docker). Model diunduh saat startup dan disimpan di Persistent Disk.

## Fitur

- Endpoint: `GET /health`, `POST /predict` (gambar tunggal), `POST /predict-batch` (banyak gambar)
- Preprocessing: crop wajah (Haar Cascade), CLAHE, grayscale→RGB, resize+normalize
- Opsi TTA (flip horizontal) dan FP16 autocast di GPU
- CORS diaktifkan (dapat dikonfigurasi via environment variable)

## Kebutuhan

- Python 3.11 (untuk run lokal) atau Render (Python runtime)
- Persistent Disk di Render (disarankan 1–2 GB)

## Environment Variables

- `MODEL_URL`: tautan unduh langsung (gunakan format `.../resolve/main/model.pth`)
- `CHECKPOINT_PATH`: lokasi file model, contoh `/opt/render/persistent/model.pth`
- `IMG_SIZE`: default `224`
- `CORS_ALLOW_ORIGINS`: default `*`


## Deploy ke Render (tanpa Docker)

Konfigurasi `render.yaml` pada folder ini:

```
buildCommand: pip install --upgrade pip setuptools wheel && pip install -r requirements.txt
startCommand: uvicorn main:app --host 0.0.0.0 --port $PORT
```

Aktifkan Persistent Disk dan set `mountPath` ke `/opt/render/persistent`. Tambahkan ENV sesuai bagian di atas.

Contoh nilai ENV (gunakan di dashboard Render atau `.env` lokal untuk uji):

```
MODEL_URL=https://huggingface.co/username/emotion-model/resolve/main/model.pth
CHECKPOINT_PATH=/opt/render/persistent/model.pth
IMG_SIZE=224
CORS_ALLOW_ORIGINS=*
```

Model akan diunduh saat startup oleh `main.py` bila belum ada di `CHECKPOINT_PATH`.

### Update Model via ENV (fleksibel)
- Ganti `MODEL_URL` ke tautan baru (mis. `https://huggingface.co/ulhaqdhifulloh/emotion-detection/resolve/main/cnn_emotion_model_v6-2.pth`).
- Aplikasi menyimpan metadata sumber model di Persistent Disk. Saat nilai `MODEL_URL` berubah, aplikasi mendeteksi perbedaan dan akan mengunduh ulang ke `CHECKPOINT_PATH` meski file lama ada. Jadi cukup ubah ENV di Render lalu redeploy.

## Jalankan Lokal

1. Buat dan aktifkan virtual environment.
2. Install dependency:
   ```bash
   pip install -r requirements.txt
   ```
3. Set environment variable dan mulai server:
   - Windows PowerShell:
     ```powershell
     $env:MODEL_URL = "https://huggingface.co/username/emotion-model/resolve/main/model.pth"
     $env:CHECKPOINT_PATH = "C:\\temp\\model.pth"
     uvicorn main:app --host 0.0.0.0 --port 8000
     ```
   - macOS/Linux:
     ```bash
     export MODEL_URL="https://huggingface.co/username/emotion-model/resolve/main/model.pth"
     export CHECKPOINT_PATH="/tmp/model.pth"
     uvicorn main:app --host 0.0.0.0 --port 8000
     ```

## Catatan
- Tidak ada prestart.py — unduhan model dilakukan oleh `main.py` saat startup.
- Website statis ada di `web-app/` (tidak dibundel di API).

## Uji Cepat (curl)
Jalankan dari folder ini:

```bash
curl http://localhost:8000/health

curl -X POST "http://localhost:8000/predict?tta=false&fp16=true" \
  -F "file=@./../images/test/joy_WIN_20251009_00_23_41_Pro.jpg"
```

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

API dipelihara di folder ini untuk kebutuhan deploy Render.