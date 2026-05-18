# src/train_attn.py
import argparse, os, csv, json, random
from pathlib import Path

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.cuda.amp import autocast, GradScaler
from sklearn.metrics import precision_recall_fscore_support

from models import (
    MobileViTv2Classifier,
    MobileViTv2_SAM, MobileViTv2_CBAM, MobileViTv2_BAM, MobileViTv2_C2PSA
)

def set_seed(s=42):
    import numpy as np
    random.seed(s); np.random.seed(s)
    torch.manual_seed(s); torch.cuda.manual_seed_all(s)

MODEL_ZOO = {
    "baseline": MobileViTv2Classifier,
    "sam":      MobileViTv2_SAM,
    "cbam":     MobileViTv2_CBAM,
    "bam":      MobileViTv2_BAM,
    "c2psa":    MobileViTv2_C2PSA,
}

def get_loaders(root, size=224, bs=32, nw=4):
    tfm_tr = transforms.Compose([
        transforms.Resize((size,size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(p=0.1),
        transforms.ColorJitter(0.1,0.1,0.1,0.05),
        transforms.ToTensor(),
    ])
    tfm_te = transforms.Compose([transforms.Resize((size,size)), transforms.ToTensor()])
    tr = datasets.ImageFolder(os.path.join(root,"train"), tfm_tr)
    va = datasets.ImageFolder(os.path.join(root,"val"),   tfm_te)
    te = datasets.ImageFolder(os.path.join(root,"test"),  tfm_te)
    return (
        DataLoader(tr, bs, True,  num_workers=nw, pin_memory=True),
        DataLoader(va, bs, False, num_workers=nw, pin_memory=True),
        DataLoader(te, bs, False, num_workers=nw, pin_memory=True),
        tr.classes
    )

def accuracy(logits, y):
    return (logits.argmax(1) == y).float().mean().item()

def prf1_batch(logits, y):
    preds = logits.argmax(1).detach().cpu().numpy()
    ytrue = y.detach().cpu().numpy()
    p, r, f1, _ = precision_recall_fscore_support(ytrue, preds, average="macro", zero_division=0)
    pw, rw, f1w, _ = precision_recall_fscore_support(ytrue, preds, average="weighted", zero_division=0)
    return {"precision_macro": float(p), "recall_macro": float(r), "f1_macro": float(f1),
            "precision_weighted": float(pw), "recall_weighted": float(rw), "f1_weighted": float(f1w)}

def train_one_epoch(model, loader, opt, scaler, device):
    model.train()
    loss_fn = nn.CrossEntropyLoss()
    tot_loss=tot_acc=0.0; steps=0
    for x,y in loader:
        x,y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        opt.zero_grad(set_to_none=True)
        with autocast(enabled=torch.cuda.is_available()):
            out = model(x); loss = loss_fn(out,y)
        scaler.scale(loss).backward()
        scaler.step(opt); scaler.update()
        tot_loss += loss.item(); tot_acc += accuracy(out,y); steps += 1
    return tot_loss/max(steps,1), tot_acc/max(steps,1)

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    loss_fn = nn.CrossEntropyLoss()
    tot_loss=tot_acc=0.0; steps=0
    agg = {"precision_macro":0, "recall_macro":0, "f1_macro":0,
           "precision_weighted":0, "recall_weighted":0, "f1_weighted":0}
    for x,y in loader:
        x,y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        out = model(x); loss = loss_fn(out,y)
        tot_loss += loss.item(); tot_acc += accuracy(out,y); steps += 1
        m = prf1_batch(out,y)
        for k in agg: agg[k] += m[k]
    for k in agg: agg[k] /= max(steps,1)
    return tot_loss/max(steps,1), tot_acc/max(steps,1), agg

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--model", default="baseline", choices=list(MODEL_ZOO.keys()))
    ap.add_argument("--img", type=int, default=224)
    ap.add_argument("--bs",  type=int, default=32)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--wd", type=float, default=1e-2)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    set_seed(args.seed)
    data_root = Path("data")/args.dataset
    assert (data_root/"train").exists(), f"dataset not prepared at {data_root}"

    tr, va, te, classes = get_loaders(str(data_root), size=args.img, bs=args.bs, nw=4)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    Model = MODEL_ZOO[args.model]
    model = Model(num_classes=len(classes)).to(device)
    opt = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    scaler = GradScaler(enabled=torch.cuda.is_available())

    run_dir = Path("runs")/args.dataset/args.model
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir/"classes.txt").write_text("\n".join(classes), encoding="utf-8")

    csv_path = run_dir/"metrics.csv"
    with open(csv_path, "w", newline="") as f:
        csv.writer(f).writerow([
            "epoch","train_loss","train_acc","val_loss","val_acc",
            "val_precision_macro","val_recall_macro","val_f1_macro",
            "val_precision_weighted","val_recall_weighted","val_f1_weighted"
        ])

    best_val = -1.0; best_path = run_dir/"best.ckpt"
    for ep in range(1, args.epochs+1):
        tr_loss, tr_acc = train_one_epoch(model, tr, opt, scaler, device)
        va_loss, va_acc, va_m = evaluate(model, va, device)
        with open(csv_path, "a", newline="") as f:
            csv.writer(f).writerow([
                ep, f"{tr_loss:.5f}", f"{tr_acc:.4f}",
                f"{va_loss:.5f}", f"{va_acc:.4f}",
                f"{va_m['precision_macro']:.4f}", f"{va_m['recall_macro']:.4f}", f"{va_m['f1_macro']:.4f}",
                f"{va_m['precision_weighted']:.4f}", f"{va_m['recall_weighted']:.4f}", f"{va_m['f1_weighted']:.4f}"
            ])
        print(f"Epoch {ep:03d}/{args.epochs} | train_acc={tr_acc:.3f} val_acc={va_acc:.3f}")
        if va_acc > best_val:
            best_val = va_acc
            torch.save({"model": model.state_dict(), "meta": vars(args),
                        "ncls": len(classes)}, best_path)

    te_loss, te_acc, te_m = evaluate(model, te, device)
    (run_dir/"test_metrics.json").write_text(json.dumps({
        "test_loss": te_loss, "test_acc": te_acc, "best_val": best_val, **te_m
    }, indent=2), encoding="utf-8")
    print(f"[DONE] best_val={best_val:.3f} | test_acc={te_acc:.3f} | saved {best_path}")

if __name__ == "__main__":
    main()
