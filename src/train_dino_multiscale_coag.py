import argparse
import json
import os
import random
import sys
from pathlib import Path
 
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
)
 
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
 
from models.dino_multiscale_coag_classifier import DinoMultiScaleClassifier
 
 
# ─── helpers ──────────────────────────────────────────────────────────────────
 
def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
 
 
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
 
 
def get_transforms(img_size):
    tfm_tr = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(p=0.1),
        transforms.ColorJitter(0.1, 0.1, 0.1, 0.05),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])
    tfm_te = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])
    return tfm_tr, tfm_te
 
 
def evaluate(model, loader, device, criterion):
    model.eval()
    all_preds, all_labels = [], []
    total_loss = 0.0
 
    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            logits = model(imgs)
            loss   = criterion(logits, labels)
            total_loss += loss.item() * imgs.size(0)
            preds = logits.argmax(dim=1)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())
 
    n = len(all_labels)
    metrics = {
        "loss":               total_loss / n,
        "acc":                accuracy_score(all_labels, all_preds),
        "precision_macro":    precision_score(all_labels, all_preds, average="macro",  zero_division=0),
        "recall_macro":       recall_score   (all_labels, all_preds, average="macro",  zero_division=0),
        "f1_macro":           f1_score       (all_labels, all_preds, average="macro",  zero_division=0),
        "precision_weighted": precision_score(all_labels, all_preds, average="weighted", zero_division=0),
        "recall_weighted":    recall_score   (all_labels, all_preds, average="weighted", zero_division=0),
        "f1_weighted":        f1_score       (all_labels, all_preds, average="weighted", zero_division=0),
        "cm":                 confusion_matrix(all_labels, all_preds).tolist(),
    }
    return metrics
 
 
# ─── main ─────────────────────────────────────────────────────────────────────
 
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset",      required=True)
    parser.add_argument("--data_root",    default="data")
    parser.add_argument("--img",          type=int,   default=224)
    parser.add_argument("--bs",           type=int,   default=4)
    parser.add_argument("--epochs",       type=int,   default=50)
    parser.add_argument("--lr",           type=float, default=1e-4)
    parser.add_argument("--wd",           type=float, default=1e-2)
    parser.add_argument("--seed",         type=int,   default=42)
    parser.add_argument("--freeze_dinov3", action="store_true")
    parser.add_argument("--grad_clip",    type=float, default=1.0)
    parser.add_argument("--out",          default="runs/coag")
    args = parser.parse_args()
 
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
 
    # ── data ──────────────────────────────────────────────────────────────────
    data_path = Path(args.data_root) / args.dataset
    tfm_tr, tfm_te = get_transforms(args.img)
 
    train_ds = ImageFolder(data_path / "train", transform=tfm_tr)
    val_ds   = ImageFolder(data_path / "val",   transform=tfm_te)
    test_ds  = ImageFolder(data_path / "test",  transform=tfm_te)
 
    train_loader = DataLoader(train_ds, batch_size=args.bs, shuffle=True,
                              num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.bs, shuffle=False,
                              num_workers=4, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=args.bs, shuffle=False,
                              num_workers=4, pin_memory=True)
 
    num_classes = len(train_ds.classes)
    print(f"Dataset: {args.dataset} | Classes: {num_classes} | "
          f"Train: {len(train_ds)} | Val: {len(val_ds)} | Test: {len(test_ds)}")
 
    # ── model ─────────────────────────────────────────────────────────────────
    model = DinoMultiScaleClassifier(
        num_classes=num_classes,
        freeze_backbone=args.freeze_dinov3,
    ).to(device)
 
    print(f"Trainable params: {count_params(model) / 1e6:.3f} M")
 
    # ── training ──────────────────────────────────────────────────────────────
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr, weight_decay=args.wd,
    )
 
    out_dir = Path(args.out) / args.dataset
    out_dir.mkdir(parents=True, exist_ok=True)
 
    best_val_f1  = 0.0
    best_epoch   = 0
    epoch_logs   = []
 
    for epoch in range(1, args.epochs + 1):
        # train
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0
 
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = model(imgs)
            loss   = criterion(logits, labels)
            loss.backward()
            if args.grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
 
            train_loss    += loss.item() * imgs.size(0)
            train_correct += (logits.argmax(1) == labels).sum().item()
            train_total   += imgs.size(0)
 
        train_loss /= train_total
        train_acc   = train_correct / train_total
 
        # validate
        val_m = evaluate(model, val_loader, device, criterion)
 
        log = {
            "epoch":      epoch,
            "train_loss": train_loss,
            "train_acc":  train_acc,
            "val_loss":              val_m["loss"],
            "val_acc":               val_m["acc"],
            "val_precision_macro":   val_m["precision_macro"],
            "val_recall_macro":      val_m["recall_macro"],
            "val_f1_macro":          val_m["f1_macro"],
            "val_precision_weighted":val_m["precision_weighted"],
            "val_recall_weighted":   val_m["recall_weighted"],
            "val_f1_weighted":       val_m["f1_weighted"],
            "val_cm":                val_m["cm"],
        }
        epoch_logs.append(log)
 
        print(f"Epoch {epoch:3d}/{args.epochs} | "
              f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
              f"val_f1={val_m['f1_macro']:.4f} val_acc={val_m['acc']:.4f}")
 
        # save best
        if val_m["f1_macro"] > best_val_f1:
            best_val_f1 = val_m["f1_macro"]
            best_epoch  = epoch
            torch.save(model.state_dict(), out_dir / "best.ckpt")
            print(f"  ✓ New best saved (val_f1_macro={best_val_f1:.4f})")
 
    # save epoch logs
    with open(out_dir / "epoch_logs.json", "w") as f:
        json.dump(epoch_logs, f, indent=2)
 
    # ── test with best checkpoint ──────────────────────────────────────────────
    model.load_state_dict(torch.load(out_dir / "best.ckpt", map_location=device))
    test_m = evaluate(model, test_loader, device, criterion)
 
    results = {
        "num_parameters":    count_params(model),
        "best_val_f1_macro": best_val_f1,
        "best_epoch":        best_epoch,
        "test_acc":          test_m["acc"],
        "test_f1_macro":     test_m["f1_macro"],
        "test_f1_weighted":  test_m["f1_weighted"],
        "test_precision_weighted": test_m["precision_weighted"],
        "test_recall_weighted":    test_m["recall_weighted"],
        "test":              test_m,
    }
 
    with open(out_dir / "test_metrics.json", "w") as f:
        json.dump(results, f, indent=2)
 
    print("\n── Final Test Results ──────────────────────")
    print(f"  Accuracy   : {test_m['acc']:.4f}")
    print(f"  Macro-F1   : {test_m['f1_macro']:.4f}")
    print(f"  Weighted-F1: {test_m['f1_weighted']:.4f}")
    print(f"  Best epoch : {best_epoch}")
    print(f"  Saved to   : {out_dir}")
 
 
if __name__ == "__main__":
    main()