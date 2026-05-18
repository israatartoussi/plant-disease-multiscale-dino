import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.amp import autocast, GradScaler
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from models.dino_multiscale_classifier import DinoMultiScaleClassifier

def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_transforms(img_size: int = 224):
    tfm_train = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(p=0.1),
        transforms.ColorJitter(0.1, 0.1, 0.1, 0.05),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    tfm_eval = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])
    return tfm_train, tfm_eval


def build_loaders(
    data_root: str,
    img_size: int,
    batch_size: int,
    num_workers: int = 0,
):
    root = Path(data_root)
    tfm_train, tfm_eval = build_transforms(img_size)

    ds_train = datasets.ImageFolder((root / "train").as_posix(), transform=tfm_train)
    ds_val = datasets.ImageFolder((root / "val").as_posix(), transform=tfm_eval)
    ds_test = datasets.ImageFolder((root / "test").as_posix(), transform=tfm_eval)

    pin = torch.cuda.is_available()

    tr = DataLoader(
        ds_train,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin,
    )
    va = DataLoader(
        ds_val,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin,
    )
    te = DataLoader(
        ds_test,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin,
    )
    return tr, va, te, ds_train.classes


def accuracy_from_logits(logits, y):
    return (logits.argmax(1) == y).float().mean().item()


def save_json(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2), encoding="utf-8")


def save_ckpt(path: Path, model, meta: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "meta": meta}, path)


def load_model_from_ckpt(path: Path, model, device: str):
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model"])
    return ckpt.get("meta", {})


def train_one_epoch(model, loader, opt, scaler, device, grad_clip: float | None = None):
    model.train()
    loss_fn = nn.CrossEntropyLoss()
    tot_loss, tot_acc, steps = 0.0, 0.0, 0

    use_amp = (device == "cuda")

    for step, (x, y) in enumerate(loader, 1):
        print(f"[train] step {step}/{len(loader)}", flush=True)

        x = x.to(device)
        y = y.to(device)

        opt.zero_grad(set_to_none=True)

        with autocast(device_type=device, enabled=use_amp):
            logits = model(x)
            loss = loss_fn(logits, y)

        if use_amp:
            scaler.scale(loss).backward()
            if grad_clip and grad_clip > 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(opt)
            scaler.update()
        else:
            loss.backward()
            if grad_clip and grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            opt.step()

        tot_loss += loss.item()
        tot_acc += accuracy_from_logits(logits, y)
        steps += 1

    return tot_loss / max(1, steps), tot_acc / max(1, steps)


@torch.no_grad()
def evaluate_fullset(model, loader, device, ncls: int):
    model.eval()
    loss_fn = nn.CrossEntropyLoss()

    all_y, all_p = [], []
    tot_loss, steps = 0.0, 0

    use_amp = (device == "cuda")

    for step, (x, y) in enumerate(loader, 1):
        print(f"[eval] step {step}/{len(loader)}", flush=True)

        x = x.to(device)
        y = y.to(device)

        with autocast(device_type=device, enabled=use_amp):
            logits = model(x)
            loss = loss_fn(logits, y)

        tot_loss += loss.item()
        steps += 1

        all_y.append(y.cpu().numpy())
        all_p.append(logits.argmax(1).cpu().numpy())

    y_true = np.concatenate(all_y)
    y_pred = np.concatenate(all_p)

    acc = float((y_true == y_pred).mean())

    p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    p_w, r_w, f1_w, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )

    cm = confusion_matrix(y_true, y_pred, labels=np.arange(ncls))
    cm_norm = confusion_matrix(
        y_true, y_pred, labels=np.arange(ncls), normalize="true"
    )

    return {
        "loss": tot_loss / max(1, steps),
        "acc": acc,
        "precision_macro": float(p_macro),
        "recall_macro": float(r_macro),
        "f1_macro": float(f1_macro),
        "precision_weighted": float(p_w),
        "recall_weighted": float(r_w),
        "f1_weighted": float(f1_w),
        "cm": cm.tolist(),
        "cm_norm": cm_norm.tolist(),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--data_root", default="data")
    ap.add_argument("--img", type=int, default=224)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--wd", type=float, default=1e-2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--freeze_dinov3", action="store_true")
    ap.add_argument("--grad_clip", type=float, default=0.0)
    ap.add_argument("--out", default="runs_dino_multiscale")
    args = ap.parse_args()

    set_seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Device: {device}", flush=True)

    ds_dir = Path(args.data_root) / args.dataset
    assert (ds_dir / "train").exists(), f"Dataset not found at: {ds_dir}"

    tr, va, te, classes = build_loaders(
        data_root=ds_dir.as_posix(),
        img_size=args.img,
        batch_size=args.bs,
        num_workers=0,
    )
    ncls = len(classes)

    model = DinoMultiScaleClassifier(
        num_classes=ncls,
        pretrained_weights=None,
        freeze_backbone=args.freeze_dinov3,
    ).to(device)

    opt = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=args.wd,
    )

    scaler = GradScaler("cuda", enabled=(device == "cuda"))

    # sanity check
    x0, _ = next(iter(tr))
    x0 = x0.to(device)
    with torch.no_grad():
        out0 = model(x0)
    print("sanity output shape:", out0.shape, flush=True)

    out_dir = Path(args.out) / args.dataset / "dino_multiscale"
    out_dir.mkdir(parents=True, exist_ok=True)

    best_f1 = -1.0
    history = []

    for ep in range(1, args.epochs + 1):
        print(f"\n[INFO] Epoch {ep}/{args.epochs}", flush=True)

        tr_loss, tr_acc = train_one_epoch(
            model, tr, opt, scaler, device, grad_clip=args.grad_clip
        )
        val = evaluate_fullset(model, va, device, ncls)

        row = {
            "epoch": ep,
            "train_loss": tr_loss,
            "train_acc": tr_acc,
            **{f"val_{k}": v for k, v in val.items()},
        }
        history.append(row)

        print(
            f"[E{ep}] "
            f"tr_loss={tr_loss:.4f} | tr_acc={tr_acc:.3f} | "
            f"val_acc={val['acc']:.3f} | val_f1={val['f1_macro']:.3f}",
            flush=True,
        )

        if val["f1_macro"] > best_f1:
            best_f1 = val["f1_macro"]
            save_ckpt(
                out_dir / "best.ckpt",
                model,
                {
                    "best_f1_macro": best_f1,
                    "epoch": ep,
                    "classes": classes,
                    "args": vars(args),
                },
            )

    best_meta = load_model_from_ckpt(out_dir / "best.ckpt", model, device)
    test = evaluate_fullset(model, te, device, ncls)

    # save history
    save_json(out_dir / "history.json", history)

    save_json(
        out_dir / "test_metrics.json",
        {
            "num_parameters": count_params(model),
            "best_val_f1_macro": float(best_f1),
            "best_epoch": best_meta.get("epoch"),
            "test_acc": test["acc"],
            "test_f1_macro": test["f1_macro"],
            "test_f1_weighted": test["f1_weighted"],
            "test_precision_weighted": test["precision_weighted"],
            "test_recall_weighted": test["recall_weighted"],
            "test": test,
        },
    )

    print("\nTraining Done ✅", flush=True)
    print(f"[DONE] best_val_f1_macro={best_f1:.3f}", flush=True)
    print(f"[DONE] test_acc={test['acc']:.3f} | test_f1={test['f1_macro']:.3f}", flush=True)
    print("Number of parameters:", count_params(model), flush=True)

if __name__ == "__main__":
    main()
