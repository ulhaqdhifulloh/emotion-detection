# 🎭 Sistem Deteksi Emosi (Mode Capture Foto + API)

Repositori ini menyediakan sistem deteksi emosi berbasis CNN dengan dua komponen utama: backend FastAPI (PyTorch) dan web-app frontend. Web-app sekarang menggunakan mode "Capture Photo" (bukan real-time scanning) dan mengirimkan satu foto ke API untuk prediksi.

## Fitur Utama

- Backend FastAPI dengan endpoint `GET /health`, `POST /predict`, dan `POST /predict-batch`
- Preprocessing di server: crop wajah (Haar Cascade), CLAHE, grayscale→RGB, resize+normalize
- Opsi prediksi: `TTA` (augmentasi flip horizontal) dan `FP16` (autocast di GPU)
- Web-app dengan kamera, tombol "Capture Photo", dan panel pengaturan (API Base URL, TTA, FP16)
- Dukungan Docker untuk menjalankan API secara terisolasi

## Struktur Proyek

```
emotion-detection/
├── api/                  # Server FastAPI (PyTorch)
│   ├── main.py
│   └── README.md         # Panduan API (Bahasa Indonesia)
├── web-app/              # Frontend statis (mode Capture)
│   ├── index.html
│   └── js/emotion-detector.js
├── models/               # Folder model (mount ke container)
├── data/                 # Dataset (opsional, besar)
└── README.md             # Dokumen ini
```

## Prasyarat

- Model checkpoint hasil training (contoh: `cnn_emotion_model_v6-2.pth`)
- Docker (disarankan untuk API) atau Python 3.11 jika menjalankan lokal
- Browser modern untuk menjalankan web-app

## Menjalankan API dengan Docker (Direkomendasikan)

1. Masuk ke folder `api` dan build image:
   ```powershell
   docker build -t emotion-api .
   ```
2. Jalankan container dengan mount model dan port `8000` (dari root repo):
  - Windows PowerShell:
    ```powershell
    docker run --rm -p 8000:8000 ^
      -e CHECKPOINT_PATH=/models/fine-tune/cnn_emotion_model_v6-2.pth ^
      -e IMG_SIZE=224 ^
      -e CORS_ALLOW_ORIGINS=* ^
      -v "${PWD}\\models:/models:ro" ^
      emotion-api:latest
    ```
  - macOS/Linux (bash):
    ```bash
    docker run --rm -p 8000:8000 \
      -e CHECKPOINT_PATH=/models/fine-tune/cnn_emotion_model_v6-2.pth \
      -e IMG_SIZE=224 \
      -e CORS_ALLOW_ORIGINS=* \
      -v "$(pwd)/models:/models:ro" \
      emotion-api:latest
    ```
3. Cek kesehatan API:
   ```bash
   curl http://localhost:8000/health
   ```

Catatan:
- Sesuaikan path `-v` untuk Linux/macOS.
- Pastikan `CHECKPOINT_PATH` mengarah ke file yang valid di dalam container (`/models/...`).
- Jangan commit file model/cekpoin ke repository karena ukurannya besar. Gunakan mount Docker (`-v ./models:/models:ro`) atau path relatif lokal di `CHECKPOINT_PATH`.

## Menjalankan Web-App

Web-app adalah file statis. Jalankan server statis sederhana dari folder `web-app`:

```powershell
cd web-app
python -m http.server 5500
# Buka: http://localhost:5500
```

Alur penggunaan:
- Klik `Start Camera` untuk mengaktifkan kamera.
- Klik `Capture Photo` untuk mengambil 1 frame.
- Frontend akan mengirimkan foto ke `API Base URL` → endpoint `/predict`.
- Hasil prediksi ditampilkan sebagai label emosi dan confidence.

Pengaturan di panel Settings:
- `API Base URL` (contoh: `http://localhost:8000`)
- `Use TTA` (augmentasi flip, akurasi sedikit naik, latensi bertambah)
- `Use FP16` (efisien di GPU, abaikan jika CPU)

## Contoh Panggilan API

```bash
curl -X POST "http://localhost:8000/predict?tta=false&fp16=true" \
  -F "file=@web-app/sample.jpg"
```

Respons ringkas:

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

## Troubleshooting

- Gagal akses kamera: jalankan lewat server (`python -m http.server`), jangan buka `index.html` langsung.
- CORS error: set `CORS_ALLOW_ORIGINS` dengan origin web-app (mis. `http://localhost:5500`).
- Checkpoint tidak ditemukan: pastikan variabel `CHECKPOINT_PATH` benar dan file dimount ke container.
- Prediksi lambat: matikan `TTA`, gunakan `FP16` bila ada GPU.

## Lisensi

Repositori ini ditujukan untuk keperluan akademik/penelitian. Tidak ada lisensi tambahan yang ditetapkan di sini.