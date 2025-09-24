# src/train.py
import argparse, os, json
import torch, torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score
from .datasets import get_datasets
from .labels import CLASSES
from .models import build_model

def train_one_epoch(model, loader, opt, crit, device):
    model.train()
    tot, correct, loss_sum = 0, 0, 0.0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        opt.zero_grad()
        out = model(x)
        loss = crit(out, y)
        loss.backward()
        opt.step()
        loss_sum += loss.item() * y.size(0)
        pred = out.argmax(1)
        correct += (pred == y).sum().item()
        tot += y.size(0)
    return loss_sum/tot, correct/tot

@torch.no_grad()
def evaluate(model, loader, crit, device):
    model.eval()
    tot, correct, loss_sum = 0, 0, 0.0
    all_y, all_p = [], []
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        out = model(x)
        loss = crit(out, y)
        loss_sum += loss.item() * y.size(0)
        pred = out.argmax(1)
        correct += (pred == y).sum().item()
        tot += y.size(0)
        all_y += y.cpu().tolist()
        all_p += pred.cpu().tolist()
    f1 = f1_score(all_y, all_p, average='macro')
    return loss_sum/tot, correct/tot, f1

def main(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    tr, va = get_datasets(args.data_root)
    tl = DataLoader(tr, batch_size=args.bs, shuffle=True,  num_workers=4, pin_memory=True)
    vl = DataLoader(va, batch_size=args.bs, shuffle=False, num_workers=4, pin_memory=True)

    model = build_model(len(CLASSES)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    crit = nn.CrossEntropyLoss(label_smoothing=0.05)

    best_f1, best_path = -1.0, os.path.join(args.out_dir, 'best.pt')
    os.makedirs(args.out_dir, exist_ok=True)

    for epoch in range(args.epochs):
        tr_loss, tr_acc = train_one_epoch(model, tl, opt, crit, device)
        va_loss, va_acc, va_f1 = evaluate(model, vl, crit, device)
        print(f"Epoch {epoch+1}: tr_loss={tr_loss:.4f} tr_acc={tr_acc:.3f} | "
              f"va_loss={va_loss:.4f} va_acc={va_acc:.3f} va_f1={va_f1:.3f}")

        # simpan checkpoint tiap epoch (opsional)
        torch.save({'model': model.state_dict(), 'epoch': epoch+1},
                   os.path.join(args.out_dir, f'ckpt_{epoch+1}.pt'))

        # simpan best
        if va_f1 > best_f1:
            best_f1 = va_f1
            torch.save({'model': model.state_dict()}, best_path)

    with open(os.path.join(args.out_dir, 'metrics.json'), 'w') as f:
        json.dump({'best_va_f1': best_f1}, f)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument('--data_root', default='data')
    p.add_argument('--out_dir', default='checkpoints')
    p.add_argument('--epochs', type=int, default=10)
    p.add_argument('--bs',     type=int, default=64)
    p.add_argument('--lr',     type=float, default=3e-4)
    args = p.parse_args()
    main(args)