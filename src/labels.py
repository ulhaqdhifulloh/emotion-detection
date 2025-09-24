# src/labels.py
DATASET7 = ["angry","disgust","fear","happy","neutral","sad","surprise"]

# Target 5 kelas (output API)
CLASSES = ["anger","joy","sad","fear","love"]

# Mapping 7->5 (lihat catatan README)
MAP7_TO_5 = {
    "angry":    "anger",
    "disgust":  "anger",
    "fear":     "fear",
    "happy":    "joy",
    "neutral":  "love",     # proxy untuk MVP
    "sad":      "sad",
    "surprise": "joy"
}

CLS2IDX = {c:i for i,c in enumerate(CLASSES)}
IDX2CLS = {i:c for c,i in CLS2IDX.items()}