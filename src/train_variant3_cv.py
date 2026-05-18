import argparse, json, csv, os, random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix

from models.fusion_variant3 import FusionVariant3


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_transforms(img_size: int):
    tfm_train = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(p=0.1),
        transforms.ColorJitter(0.1, 0.1, 0.1, 0.05),
        transforms.ToTensor(),
    ])
    tfm_eval = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
    ])
    return tfm_train, tfm_eval


def accuracy(logits, y):
    return (logits.argmax(1) == y).float().mean().item()


@torch.no_grad()
def eval_epoch(model, loader, device, ncls):
    model.eval()
    loss_fn = nn.CrossEntropyLoss()
    all_p, all_y = [], []
    tot_loss = 0.0
    steps = 0

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        logits = model(x)
        loss = loss_fn(logits, y)

        tot_loss += loss.item()
        steps += 1
        all_p.append(logits.argmax(1).detach().cpu().numpy())
        all_y.append(y.detach().cpu().numpy())

    y_true = np.concatenate(all_y)
    y_pred = np.concatenate(all_p)

    p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    p_w, r_w, f1_w, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )

    cm = confusion_matrix(y_true, y_pred, labels=np.arange(ncls))
    cm_norm = confusion_matrix(y_true, y_pred, labels=np.arange(ncls), normalize="true")

    return {
        "loss": tot_loss / max(steps, 1),
        "acc": float((y_true == y_pred).mean()),
        "precision_macro": float(p_macro),
        "recall_macro": float(r_macro),
        "f1_macro": float(f1_macro),
        "precision_weighted": float(p_w),
        "recall_weighted": float(r_w),
        "f1_weighted": float(f1_w),
        "cm": cm.tolist(),
        "cm_norm": cm_norm.tolist(),
    }


def train_one_epoch(model, loader, opt, scaler, device):
    model.train()
    loss_fn = nn.CrossEntropyLoss()
    tot_loss = 0.0
    tot_acc = 0.0
    steps = 0

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        opt.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
            logits = model(x)
            loss = loss_fn(logits, y)

        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()

        tot_loss += loss.item()
        tot_acc += accuracy(logits, y)
        steps += 1

    return tot_loss / max(steps, 1), tot_acc / max(steps, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--data_root", default="data")
    ap.add_argument("--kfold", type=int, default=5)
    ap.add_argument("--fold", type=int, required=True)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--img", type=int, default=224)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--wd", type=float, default=1e-2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--freeze_dinov3", action="store_true")
    ap.add_argument("--save_every_epoch", action="store_true")
    ap.add_argument("--out", default="runs_variant3_cv5")
    args = ap.parse_args()

    set_seed(args.seed)

    ds_root = Path(args.data_root) / args.dataset
    train_dir = ds_root / "train"
    classes_txt = ds_root / "classes.txt"
    assert train_dir.exists(), f"Missing: {train_dir}"
    assert classes_txt.exists(), f"Missing: {classes_txt}"

    classes = classes_txt.read_text().splitlines()
    ncls = len(classes)

    tfm_train, tfm_eval = build_transforms(args.img)

    # Build one dataset for kfold from TRAIN only (scientifically correct CV)
    full_ds = datasets.ImageFolder(train_dir.as_posix(), transform=tfm_train)
    y = np.array([full_ds.samples[i][1] for i in range(len(full_ds))])

    skf = StratifiedKFold(n_splits=args.kfold, shuffle=True, random_state=args.seed)
    folds = list(skf.split(np.zeros(len(y)), y))
    tr_idx, va_idx = folds[args.fold]

    # IMPORTANT: eval dataset must not use augmentation
    full_ds_eval = datasets.ImageFolder(train_dir.as_posix(), transform=tfm_eval)

    tr_set = Subset(full_ds, tr_idx.tolist())
    va_set = Subset(full_ds_eval, va_idx.tolist())

    tr_loader = DataLoader(tr_set, batch_size=args.bs, shuffle=True, num_workers=4, pin_memory=True)
    va_loader = DataLoader(va_set, batch_size=args.bs, shuffle=False, num_workers=4, pin_memory=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = FusionVariant3(num_classes=ncls, freeze_dinov3=args.freeze_dinov3).to(device)

    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=args.wd)
    scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())

    run_dir = Path(args.out) / args.dataset / f"fold_{args.fold}"
    run_dir.mkdir(parents=True, exist_ok=True)

    (run_dir / "classes.txt").write_text("\n".join(classes), encoding="utf-8")
    (run_dir / "split.json").write_text(json.dumps({
        "dataset": args.dataset,
        "fold": args.fold,
        "kfold": args.kfold,
        "n_train": len(tr_idx),
        "n_val": len(va_idx),
        "seed": args.seed,
    }, indent=2), encoding="utf-8")

    csv_path = run_dir / "metrics.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "epoch",
            "train_loss", "train_acc",
            "val_loss", "val_acc",
            "val_precision_macro","val_recall_macro","val_f1_macro",
            "val_precision_weighted","val_recall_weighted","val_f1_weighted"
        ])

    best_f1 = -1.0
    best_path = run_dir / "best.ckpt"
    last_path = run_dir / "last.ckpt"

    for ep in range(1, args.epochs + 1):
        tr_loss, tr_acc = train_one_epoch(model, tr_loader, opt, scaler, device)
        val = eval_epoch(model, va_loader, device, ncls)

        with open(csv_path, "a", newline="") as f:
            w = csv.writer(f)
            w.writerow([
                ep,
                f"{tr_loss:.6f}", f"{tr_acc:.6f}",
                f"{val['loss']:.6f}", f"{val['acc']:.6f}",
                f"{val['precision_macro']:.6f}", f"{val['recall_macro']:.6f}", f"{val['f1_macro']:.6f}",
                f"{val['precision_weighted']:.6f}", f"{val['recall_weighted']:.6f}", f"{val['f1_weighted']:.6f}",
            ])

        # Save last (useful)
        torch.save({
            "model": model.state_dict(),
            "meta": vars(args),
            "epoch": ep,
            "ncls": ncls,
            "classes": classes,
        }, last_path)

        # Save per-epoch minimal (only if asked)
        if args.save_every_epoch:
            torch.save({
                "model": model.state_dict(),
                "meta": vars(args),
                "epoch": ep,
                "ncls": ncls,
                "classes": classes,
            }, run_dir / f"epoch_{ep:03d}.ckpt")

        # Best by macro-F1 (more scientific for imbalance)
        if val["f1_macro"] > best_f1:
            best_f1 = val["f1_macro"]
            torch.save({
                "model": model.state_dict(),
                "meta": vars(args),
                "epoch": ep,
                "best_f1_macro": best_f1,
                "ncls": ncls,
                "classes": classes,
                "best_val_metrics": val,  # includes CM + normalized CM
            }, best_path)

        print(f"[{args.dataset} fold {args.fold}] ep {ep:03d}/{args.epochs} "
              f"train_acc={tr_acc:.3f} val_acc={val['acc']:.3f} val_f1m={val['f1_macro']:.3f}")

    # Write fold summary (only useful fields)
    best = torch.load(best_path, map_location="cpu")
    (run_dir / "fold_summary.json").write_text(json.dumps({
        "dataset": args.dataset,
        "fold": args.fold,
        "best_epoch": int(best["epoch"]),
        "best_f1_macro": float(best["best_f1_macro"]),
        "best_val_metrics": best["best_val_metrics"],
    }, indent=2), encoding="utf-8")

    print(f"[DONE] {args.dataset} fold {args.fold}: best_epoch={best['epoch']} best_f1_macro={best['best_f1_macro']:.4f}")


if __name__ == "__main__":
    main()
