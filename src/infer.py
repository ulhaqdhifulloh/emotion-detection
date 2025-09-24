# src/infer.py
import sys, torch, torchvision.transforms as T
from PIL import Image
from .labels import CLASSES
from .models import build_model

def load_model(ckpt_path, device):
    m = build_model(len(CLASSES)).to(device)
    state = torch.load(ckpt_path, map_location=device)
    m.load_state_dict(state['model'])
    m.eval()
    return m

def predict_image(model, img_path, device):
    tfm = T.Compose([
        T.Resize((224,224)),
        T.ToTensor(),
        T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
    ])
    img = Image.open(img_path).convert('RGB')
    x = tfm(img).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(x)[0]
        probs = torch.softmax(logits, dim=0).cpu().numpy().tolist()
        top_idx = int(torch.argmax(logits).cpu())
    return CLASSES[top_idx], probs

if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = load_model(sys.argv[1], device)
    label, probs = predict_image(model, sys.argv[2], device)
    print(label, probs)