# serving/app.py
import io, os, base64
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from PIL import Image
import torch, torch.nn as nn
import torchvision.transforms as T

from src.models import build_model
from src.labels import CLASSES

IM_SIZE = 224
tfm = T.Compose([
    T.Resize((IM_SIZE, IM_SIZE)),
    T.ToTensor(),
    T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
])

app = FastAPI(title="Emotion API", version="1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # sebaiknya batasi di prod
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = build_model(len(CLASSES)).to(device)

ckpt_path = os.getenv("CKPT_PATH", "/app/best.pt")
state = torch.load(ckpt_path, map_location=device)
model.load_state_dict(state['model'])
model.eval()

class B64Image(BaseModel):
    image_base64: str  # "data:image/jpeg;base64,/9j/4AA..."

def _infer_pil(pil):
    x = tfm(pil).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(x)[0]
        probs = torch.softmax(logits, dim=0).cpu().numpy().tolist()
        top_idx = int(torch.argmax(logits).cpu())
    return {
        "top_label": CLASSES[top_idx],
        "probs": {c: float(probs[i]) for i,c in enumerate(CLASSES)}
    }

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    img_bytes = await file.read()
    pil = Image.open(io.BytesIO(img_bytes)).convert('RGB')
    return _infer_pil(pil)

@app.post("/predict_b64")
async def predict_b64(payload: B64Image):
    b64 = payload.image_base64.split(",")[-1]
    pil = Image.open(io.BytesIO(base64.b64decode(b64))).convert('RGB')
    return _infer_pil(pil)

@app.get("/labels")
def labels():
    return {"classes": CLASSES}

@app.get("/healthz")
def health():
    return {"status":"ok"}