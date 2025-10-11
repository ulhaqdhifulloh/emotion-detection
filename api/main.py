# main.py
# FastAPI inference server for EmotionCNN (ResNet18 backbone)
# - Memuat checkpoint sekali saat start
# - Endpoint:
#     GET  /health
#     POST /predict            -> single image (multipart/form-data)
#     POST /predict-batch      -> multiple images (multipart/form-data)
# - Fitur:
#     * Preprocess identik dgn skrip Anda (face-crop + CLAHE + Grayscale->RGB + Resize+Normalize)
#     * Opsional TTA (flip horizontal)
#     * Opsional FP16 autocast saat GPU
#     * Thread-safe inference lock
#     * CORS diaktifkan (bisa batasi origins kalau perlu)

import io
import os
import time
from typing import List, Optional, Dict, Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image, ImageFilter
import cv2

from fastapi import FastAPI, File, UploadFile, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# -----------------------------
# 0) KONFIG / GLOBALS
# -----------------------------
# ENV var agar mudah pindah server tanpa ubah kode
CHECKPOINT_PATH = os.getenv("CHECKPOINT_PATH", "/models/fine-tune/cnn_emotion_model_v6-2.pth")
IMG_SIZE = int(os.getenv("IMG_SIZE", "224"))

# Normalisasi pakai nilai standar ImageNet (bisa override via ENV jika perlu)
MEAN = [float(x) for x in os.getenv("MEAN", "0.485,0.456,0.406").split(",")]
STD  = [float(x) for x in os.getenv("STD",  "0.229,0.224,0.225").split(",")]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_num_threads(max(1, os.cpu_count() or 1))
torch.backends.cudnn.benchmark = True

# Inference lock agar aman saat beberapa request paralel di GPU
from threading import Lock
_infer_lock = Lock()

# -----------------------------
# 1) MODEL (identik dengan training Anda)
# -----------------------------
class EmotionCNN(nn.Module):
    def __init__(self, num_classes=4):
        super().__init__()
        # Penting: hindari download weight saat server start.
        # Di torchvision baru, gunakan weights=None (bukan pretrained=True)
        self.backbone = models.resnet18(weights=None)
        self.backbone.fc = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(self.backbone.fc.in_features, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, num_classes)
        )
    def forward(self, x):
        return self.backbone(x)

def _strip_module_prefix(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    # Bersihkan prefix "module." jika training pakai DataParallel
    if not any(k.startswith("module.") for k in state_dict.keys()):
        return state_dict
    return {k.replace("module.", "", 1): v for k, v in state_dict.items()}

def load_checkpoint(path: str):
    ckpt = torch.load(path, map_location=device)
    # dukung beberapa kemungkinan kunci:
    state_keys = ["ema_state_dict", "model_state_dict", "state_dict", "model"]
    state_dict = None
    for k in state_keys:
        if k in ckpt:
            obj = ckpt[k]
            state_dict = obj.state_dict() if hasattr(obj, "state_dict") else obj
            break

    # Jika benar-benar pure state_dict
    if state_dict is None and isinstance(ckpt, dict) and all(isinstance(v, torch.Tensor) for v in ckpt.values()):
        state_dict = ckpt

    if state_dict is None:
        raise RuntimeError("Tidak menemukan state_dict dalam checkpoint. Pastikan kuncinya 'model_state_dict' atau serupa.")

    state_dict = _strip_module_prefix(state_dict)

    class_names = ckpt.get("class_names", ["anger", "fear", "joy", "sad"])
    model = EmotionCNN(num_classes=len(class_names)).to(device)
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    return model, class_names

# -----------------------------
# 2) TRANSFORM & FACE CROP
# -----------------------------
val_tf = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])

# disiapkan kalau ingin eksperimen:
val_tf_center = transforms.Compose([
    transforms.Resize(int(IMG_SIZE * 1.14)),
    transforms.CenterCrop(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])

# Face detector (Haar Cascade) sekali load
_haar_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

def detect_and_crop_face_pil(pil_img: Image.Image) -> Image.Image:
    """Menerima PIL RGB, kembalikan PIL RGB ter-crop wajah (margin bawah diperbesar)."""
    img = np.array(pil_img)[:, :, ::-1]  # ke BGR utk OpenCV
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = _haar_cascade.detectMultiScale(gray, 1.3, 5)
    if len(faces) == 0:
        return pil_img

    (x, y, w, h) = faces[0]
    margin_x = int(0.15 * w)   # lebih kecil horizontal
    margin_top = int(0.10 * h)
    margin_bot = int(0.35 * h) # bawah diperbesar (mulut/dagu)

    x1 = max(0, x - margin_x)
    y1 = max(0, y - margin_top)
    x2 = min(img.shape[1], x + w + margin_x)
    y2 = min(img.shape[0], y + h + margin_bot)
    face_crop = img[y1:y2, x1:x2]
    return Image.fromarray(cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB))

def preprocess_image_pil(pil_img: Image.Image) -> torch.Tensor:
    """
    Preprocess identik dgn skrip Anda:
      - face crop
      - CLAHE (grayscale) lalu kembali ke RGB
      - resize + normalize
    Return: tensor [1,3,H,W]
    """
    img = detect_and_crop_face_pil(pil_img)

    # CLAHE pada grayscale lalu kembalikan ke RGB
    g = np.array(img.convert("L"))
    clahe = cv2.createCLAHE(clipLimit=1.2, tileGridSize=(8, 8))
    g = clahe.apply(g)
    img = Image.fromarray(g).convert("RGB")

    x = val_tf(img).unsqueeze(0)  # [1,3,H,W]
    return x

# -----------------------------
# 3) INFERENCE UTIL
# -----------------------------
@torch.inference_mode()
def predict_logits(model: nn.Module, x: torch.Tensor, use_fp16: bool = False) -> torch.Tensor:
    """
    x: [N,3,H,W] on CPU or GPU.
    """
    x = x.to(device, non_blocking=True)
    if use_fp16 and device.type == "cuda":
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            return model(x)
    else:
        return model(x)

def to_response(probs: torch.Tensor, class_names: List[str]) -> Dict[str, Any]:
    # probs: [C]
    conf, idx = probs.max(dim=0)
    entropy = float(-(probs * (probs + 1e-12).log()).sum().item())
    return {
        "emotion": class_names[idx.item()],
        "confidence": float(conf.item()),
        "probs": {class_names[i]: float(probs[i].item()) for i in range(len(class_names))},
        "entropy": entropy,
    }

@torch.inference_mode()
def predict_single(model: nn.Module, pil_img: Image.Image, class_names: List[str], use_tta: bool = False, use_fp16: bool = False) -> Dict[str, Any]:
    """
    Jalankan pipeline preprocess + (opsional) TTA.
    """
    if not use_tta:
        x = preprocess_image_pil(pil_img)
        logits = predict_logits(model, x, use_fp16=use_fp16)
        probs = F.softmax(logits, dim=1)[0]
        return to_response(probs, class_names)
    else:
        # TTA ringan: identik dgn skrip Anda (asli + flip horizontal)
        # NOTE: Preprocess (crop/CLAHE/normalize) tetap dilakukan per-view
        views = []
        # original
        views.append(preprocess_image_pil(pil_img))
        # flipped
        views.append(preprocess_image_pil(pil_img.transpose(Image.FLIP_LEFT_RIGHT)))

        x = torch.cat(views, dim=0)  # [2,3,H,W]
        logits = predict_logits(model, x, use_fp16=use_fp16)
        probs = F.softmax(logits, dim=1).mean(dim=0)  # rata-rata
        return to_response(probs, class_names)

# -----------------------------
# 4) FASTAPI APP
# -----------------------------
app = FastAPI(title="Emotion Detection API", version="1.0.0")

# Bebaskan CORS sesuai kebutuhan frontend Anda
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ALLOW_ORIGINS", "*").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class PredictMeta(BaseModel):
    tta: bool = False
    fp16: bool = True

class PredictResult(BaseModel):
    emotion: str
    confidence: float
    probs: Dict[str, float]
    entropy: float
    latency_ms: float
    device: str
    tta_used: bool
    fp16_used: bool

class BatchItem(BaseModel):
    filename: str
    result: Optional[PredictResult] = None
    error: Optional[str] = None

class BatchResponse(BaseModel):
    items: List[BatchItem]

# Muat model saat startup
MODEL: Optional[nn.Module] = None
CLASS_NAMES: List[str] = []

@app.on_event("startup")
def _load_model_on_startup():
    global MODEL, CLASS_NAMES
    assert os.path.isfile(CHECKPOINT_PATH), f"Checkpoint tidak ditemukan: {CHECKPOINT_PATH}"
    MODEL, CLASS_NAMES = load_checkpoint(CHECKPOINT_PATH)
    print(f"[Startup] Model siap. Device={device}, classes={CLASS_NAMES}")

@app.get("/health")
def health():
    return {
        "status": "ok",
        "device": str(device),
        "classes": CLASS_NAMES,
        "img_size": IMG_SIZE,
        "checkpoint": CHECKPOINT_PATH
    }

@app.post("/predict", response_model=PredictResult)
def predict_endpoint(
    file: UploadFile = File(...),
    tta: bool = Query(False, description="Gunakan TTA flip horizontal"),
    fp16: bool = Query(True, description="Autocast FP16 saat GPU")
):
    global MODEL, CLASS_NAMES
    start = time.perf_counter()
    try:
        contents = file.file.read()
        pil = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception as e:
        raise ValueError(f"Gagal membaca gambar: {e}")
    finally:
        file.file.close()

    with _infer_lock:
        out = predict_single(MODEL, pil, CLASS_NAMES, use_tta=tta, use_fp16=fp16)

    latency = (time.perf_counter() - start) * 1000.0
    return {
        **out,
        "latency_ms": round(latency, 2),
        "device": str(device),
        "tta_used": bool(tta),
        "fp16_used": bool(fp16 and (device.type == "cuda"))
    }

@app.post("/predict-batch", response_model=BatchResponse)
def predict_batch_endpoint(
    files: List[UploadFile] = File(..., description="Kirim beberapa file sekaligus"),
    tta: bool = Query(False),
    fp16: bool = Query(True)
):
    global MODEL, CLASS_NAMES
    items: List[BatchItem] = []

    # Untuk efisiensi, kita tetap proses per-file karena ada face-crop/CLAHE per gambar
    for f in files:
        try:
            contents = f.file.read()
            pil = Image.open(io.BytesIO(contents)).convert("RGB")
            start = time.perf_counter()
            with _infer_lock:
                out = predict_single(MODEL, pil, CLASS_NAMES, use_tta=tta, use_fp16=fp16)
            latency = (time.perf_counter() - start) * 1000.0
            items.append(BatchItem(
                filename=f.filename,
                result=PredictResult(
                    **out,
                    latency_ms=round(latency, 2),
                    device=str(device),
                    tta_used=bool(tta),
                    fp16_used=bool(fp16 and (device.type == "cuda"))
                )
            ))
        except Exception as e:
            items.append(BatchItem(filename=f.filename, error=str(e)))
        finally:
            try:
                f.file.close()
            except Exception:
                pass

    return BatchResponse(items=items)