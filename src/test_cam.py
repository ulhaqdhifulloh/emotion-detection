import cv2, torch
from torchvision import transforms
from PIL import Image
from models import build_model
from labels import CLASSES
import numpy as np
import time

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = build_model(num_classes=len(CLASSES)).to(device)
state = torch.load("checkpoints/best.pt", map_location=device)
model.load_state_dict(state['model'])
model.eval()

tfm = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
])

# Haarcascade sederhana (cukup untuk demo)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def detect_faces(frame_bgr):
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(60,60))
    return faces

def crop_face(frame_bgr, face_rect=None):
    if face_rect is None:
        faces = detect_faces(frame_bgr)
        if len(faces) == 0: return None
        face_rect = max(faces, key=lambda r: r[2]*r[3])  # ambil wajah terbesar
    
    x, y, w, h = face_rect
    return frame_bgr[y:y+h, x:x+w], face_rect

cap = cv2.VideoCapture(0)

# Periksa apakah kamera berhasil dibuka
if not cap.isOpened():
    print("Error: Tidak dapat mengakses kamera.")
    exit()

print("Kamera berhasil dibuka. Tekan 'q' untuk keluar.")

# Variabel untuk jeda prediksi
last_prediction_time = 0
prediction_interval = 1.0  # Jeda 1 detik antar prediksi
current_emotion = "Mendeteksi..."
current_confidence = 0.0
current_face_rect = None

while True:
    ret, frame = cap.read()
    if not ret: 
        print("Error: Tidak dapat membaca frame.")
        break
    
    # Deteksi wajah
    faces = detect_faces(frame)
    
    # Jika ada wajah terdeteksi
    if len(faces) > 0:
        # Ambil wajah terbesar
        current_face_rect = max(faces, key=lambda r: r[2]*r[3])
        x, y, w, h = current_face_rect
        
        # Gambar kotak di sekitar wajah
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
        # Prediksi emosi dengan jeda
        current_time = time.time()
        if current_time - last_prediction_time >= prediction_interval:
            face_img, _ = crop_face(frame, current_face_rect)
            if face_img is not None:
                img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
                pil = Image.fromarray(img)
                x_tensor = tfm(pil).unsqueeze(0).to(device)
                with torch.no_grad():
                    probs = torch.softmax(model(x_tensor)[0], dim=0).cpu().numpy()
                current_emotion = CLASSES[int(np.argmax(probs))]
                current_confidence = float(np.max(probs))
                last_prediction_time = current_time
        
        # Tampilkan teks emosi dan akurasi di atas kotak wajah
        text = f"{current_emotion} {current_confidence:.2f}"
        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
        cv2.rectangle(frame, (x, y - text_size[1] - 10), (x + text_size[0], y), (0, 255, 0), -1)
        cv2.putText(frame, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
    
    # Tampilkan frame
    cv2.imshow("Emotion Recognition", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release(); cv2.destroyAllWindows()