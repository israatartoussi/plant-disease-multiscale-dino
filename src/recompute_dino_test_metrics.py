import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from torch.amp import autocast
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.dino_multiscale_classifier import DinoMultiScaleClassifier as BaselineModel
from models.dino_multiscale_cross_gated_classifier import (
    DinoMultiScaleClassifier as CrossGatedModel,
)
from models.dino_multiscale_gated_classifier import DinoMultiScaleClassifier as GatedModel
from models.dino_multiscale_parallel_classifier import (
    DinoMultiScaleClassifier as ParallelModel,
)


MODEL_DIR_TO_CLASS = {
    "dino_multiscale": BaselineModel,
    "dino_multiscale_gated": GatedModel,
    "dino_multiscale_cross_gated": CrossGatedModel,
    "dino_multiscale_parallel": ParallelModel,
}


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--search_root",
        action="append",
        default=None,
        help="Root directory to scan. Can be passed multiple times. Default: current directory.",
    )
    ap.add_argument(
        "--dataset",
        default=None,
        help="Optional dataset filter, e.g. corn_maize_leaf_disease",
    )
    ap.add_argument(
        "--model_dir",
        default=None,
        choices=sorted(MODEL_DIR_TO_CLASS.keys()),
        help="Optional run leaf filter, e.g. dino_multiscale_parallel",
    )
    ap.add_argument(
        "--dry_run",
        action="store_true",
        help="Compute and print what would be updated without writing files.",
    )
    return ap.parse_args()


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def build_eval_transform(img_size: int = 224):
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])


@torch.no_grad()
def evaluate_fullset(model, loader, device, ncls: int):
    model.eval()
    loss_fn = nn.CrossEntropyLoss()
    use_amp = (device == "cuda")

    all_y, all_p = [], []
    tot_loss, steps = 0.0, 0

    for x, y in loader:
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


def discover_checkpoints(search_roots, dataset_filter=None, model_dir_filter=None):
    found = []
    for root in search_roots:
        root_path = Path(root)
        if not root_path.exists():
            continue
        for ckpt in root_path.rglob("best.ckpt"):
            model_dir = ckpt.parent.name
            if model_dir not in MODEL_DIR_TO_CLASS:
                continue
            dataset = ckpt.parent.parent.name
            if dataset_filter and dataset != dataset_filter:
                continue
            if model_dir_filter and model_dir != model_dir_filter:
                continue
            found.append(ckpt)
    return sorted(set(found))


def build_model(model_dir: str, ncls: int, meta_args: dict):
    model_cls = MODEL_DIR_TO_CLASS[model_dir]
    common_kwargs = {"num_classes": ncls}

    if model_dir == "dino_multiscale_parallel":
        common_kwargs["pretrained_weights"] = None
        common_kwargs["freeze_backbone"] = bool(meta_args.get("freeze_dinov3", False))
    else:
        common_kwargs["pretrained_weights"] = None
        common_kwargs["freeze_backbone"] = bool(meta_args.get("freeze_dinov3", False))

    return model_cls(**common_kwargs)


def recompute_one(ckpt_path: Path, dry_run: bool = False):
    model_dir = ckpt_path.parent.name
    dataset = ckpt_path.parent.parent.name

    ckpt = torch.load(ckpt_path, map_location="cpu")
    meta = ckpt.get("meta", {})
    meta_args = meta.get("args", meta)

    data_root = Path(meta_args.get("data_root", "data")) / dataset
    test_dir = data_root / "test"
    if not test_dir.exists():
        raise FileNotFoundError(f"Missing test split: {test_dir}")

    img_size = int(meta_args.get("img", 224))
    batch_size = int(meta_args.get("bs", 32))
    classes = meta.get("classes")
    if not classes:
        classes_file = data_root / "classes.txt"
        if classes_file.exists():
            classes = classes_file.read_text(encoding="utf-8").splitlines()

    tfm = build_eval_transform(img_size)
    ds_test = datasets.ImageFolder(test_dir.as_posix(), transform=tfm)
    if not classes:
        classes = ds_test.classes
    ncls = len(classes)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_model(model_dir, ncls, meta_args).to(device)
    model.load_state_dict(ckpt["model"])

    loader = DataLoader(
        ds_test,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    test = evaluate_fullset(model, loader, device, ncls)
    out = {
        "num_parameters": count_params(model),
        "best_val_f1_macro": meta.get("best_f1_macro"),
        "best_epoch": meta.get("epoch"),
        "test_acc": test["acc"],
        "test_f1_macro": test["f1_macro"],
        "test_f1_weighted": test["f1_weighted"],
        "test": test,
    }

    out_path = ckpt_path.parent / "test_metrics.json"
    if not dry_run:
        out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")

    return {
        "dataset": dataset,
        "model_dir": model_dir,
        "ckpt_path": ckpt_path.as_posix(),
        "out_path": out_path.as_posix(),
        "metrics": out,
    }


def main():
    args = parse_args()
    search_roots = args.search_root or ["."]
    ckpts = discover_checkpoints(
        search_roots,
        dataset_filter=args.dataset,
        model_dir_filter=args.model_dir,
    )

    if not ckpts:
        print("No matching DINO best.ckpt files found.")
        return

    for ckpt_path in ckpts:
        result = recompute_one(ckpt_path, dry_run=args.dry_run)
        m = result["metrics"]
        mode = "DRY-RUN" if args.dry_run else "UPDATED"
        print(
            f"[{mode}] {result['dataset']} / {result['model_dir']} | "
            f"acc={m['test_acc']:.3f} | f1_macro={m['test_f1_macro']:.3f} | "
            f"f1_weighted={m['test_f1_weighted']:.3f} | "
            f"params={m['num_parameters']}"
        )


if __name__ == "__main__":
    main()
