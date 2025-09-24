# src/datasets.py
from torchvision.datasets import ImageFolder
from torchvision import transforms as T
from torch.utils.data import Dataset
from .labels import MAP7_TO_5, CLS2IDX

IM_SIZE = 224

def get_tfms(train=True):
    if train:
        return T.Compose([
            T.Resize((IM_SIZE, IM_SIZE)),
            T.RandomHorizontalFlip(),
            T.ColorJitter(0.1,0.1,0.1,0.05),
            T.RandomRotation(10),
            T.ToTensor(),
            T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
        ])
    else:
        return T.Compose([
            T.Resize((IM_SIZE, IM_SIZE)),
            T.ToTensor(),
            T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
        ])

class MapTarget(Dataset):
    """Bungkus ImageFolder(7 kelas) -> target 5 kelas via MAP7_TO_5."""
    def __init__(self, base_ds: ImageFolder):
        self.base = base_ds
        self.idx7_to_idx5 = {
            base_ds.class_to_idx[k]: CLS2IDX[v]
            for k,v in MAP7_TO_5.items()
            if k in base_ds.class_to_idx
        }

    def __len__(self): return len(self.base)

    def __getitem__(self, i):
        x, y7 = self.base[i]
        y5 = self.idx7_to_idx5[y7]
        return x, y5

def get_datasets(data_root="data"):
    train7 = ImageFolder(root=f"{data_root}/train", transform=get_tfms(train=True))
    val7   = ImageFolder(root=f"{data_root}/val",   transform=get_tfms(train=False))  # <- perbaikan
    return MapTarget(train7), MapTarget(val7)