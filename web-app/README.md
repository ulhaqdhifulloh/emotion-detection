# Web App (Capture Photo) — Panduan Singkat

Frontend sederhana untuk mengambil foto dari kamera dan mengirimkannya ke API FastAPI untuk prediksi emosi. Mode ini tidak melakukan pemindaian real-time.

## Menjalankan

```powershell
cd web-app
python -m http.server 5500
# Buka: http://localhost:5500
```

Disarankan menjalankan API di `http://localhost:8000` (lihat `api/README.md`).

## Alur Penggunaan

- Klik `Start Camera` untuk mengaktifkan kamera.
- Klik `Capture Photo` untuk mengambil satu gambar.
- Gambar dikirim ke endpoint API `/predict`.
- Hasil prediksi (label emosi + confidence) ditampilkan di panel hasil.

## Pengaturan

- `API Base URL`: alamat server API, contoh `http://localhost:8000`.
- `Use TTA`: aktifkan augmentasi flip horizontal untuk sedikit peningkatan akurasi.
- `Use FP16`: aktifkan mode setengah presisi di GPU (abaikan jika CPU).

Nilai-nilai tersebut dapat diubah langsung di panel Settings. Frontend akan menggunakan nilai saat melakukan request.

## Catatan Teknis

- Web-app tidak lagi bergantung pada ONNX runtime; semua inferensi dilakukan di backend.
- API menangani deteksi wajah dan preprocessing (Haar Cascade, CLAHE, resize, normalisasi).
- Jika Anda membuka `index.html` langsung dari file, kamera bisa gagal di beberapa browser. Gunakan server lokal seperti di atas.
- Untuk CORS, pastikan API mengizinkan origin web-app, misalnya set `CORS_ALLOW_ORIGINS=http://localhost:5500` saat menjalankan container.

## Struktur

```
web-app/
├── index.html
└── js/
    └── emotion-detector.js
```

Selesai.