# src/transforms.py
from torchvision import transforms as T
IM_SIZE = 224
def get_train_tfms():
    return T.Compose([
        T.Resize((IM_SIZE, IM_SIZE)),
        T.RandomHorizontalFlip(),
        T.ColorJitter(0.1,0.1,0.1,0.05),
        T.RandomRotation(10),
        T.ToTensor(),
        T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
    ])
def get_val_tfms():
    return T.Compose([
        T.Resize((IM_SIZE, IM_SIZE)),
        T.ToTensor(),
        T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
    ])