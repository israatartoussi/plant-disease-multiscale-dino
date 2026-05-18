import argparse
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt

from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix

from torchvision import transforms
from torch.utils.data import DataLoader
from PIL import Image

from src.train_variant3_cv import ImagePathDataset, list_all_images_under_split, load_classes
from sklearn.model_selection import StratifiedKFold

from models.dino_multiscale_coag_classifier import DinoMultiScaleClassifier


def load_fold_pool(ds_root: Path):
    paths_tr, y_tr = list_all_images_under_split(ds_root, "train")
    paths_va, y_va = list_all_images_under_split(ds_root, "val")
    paths_te, y_te = list_all_images_under_split(ds_root, "test")
    paths = paths_tr + paths_va + paths_te
    labels = np.array(y_tr + y_va + y_te, dtype=np.int64)
    return paths, labels


@torch.no_grad()
def predict_logits_and_features(model, loader, device):
    model.eval()
    all_y = []
    all_p = []
    all_feat = []
    for x, y in loader:
        x = x.to(device, non_blocking=True)

        logits = model(x)
        feat = logits
        pred = logits.argmax(1).cpu().numpy()

        all_y.append(y.numpy())
        all_p.append(pred)
        all_feat.append(feat.cpu().numpy())

    y = np.concatenate(all_y)
    p = np.concatenate(all_p)
    feat = np.concatenate(all_feat)
    return y, p, feat


def plot_confusion(cm_norm, classes, out_path: Path, dpi=300):
    fig = plt.figure(figsize=(6.5, 5.5), dpi=dpi)
    ax = fig.add_subplot(111)
    im = ax.imshow(cm_norm, interpolation="nearest")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax.set_title("Confusion Matrix (normalized)")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_xticks(np.arange(len(classes)))
    ax.set_yticks(np.arange(len(classes)))
    ax.set_xticklabels(classes, rotation=35, ha="right")
    ax.set_yticklabels(classes)

    # annotate
    for i in range(cm_norm.shape[0]):
        for j in range(cm_norm.shape[1]):
            ax.text(j, i, f"{cm_norm[i, j]*100:.1f}%", ha="center", va="center")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_tsne(feat, y, classes, out_path: Path, dpi=300):
    Fz = StandardScaler().fit_transform(feat)
    perplexity = max(5, min(30, len(Fz)//3 - 1)) if len(Fz) >= 10 else 5

    Z = TSNE(
        n_components=2,
        learning_rate="auto",
        init="pca",
        perplexity=perplexity,
        random_state=42
    ).fit_transform(Fz)

    fig = plt.figure(figsize=(7, 5.5), dpi=dpi)
    ax = fig.add_subplot(111)

    for c in np.unique(y):
        m = (y == c)
        ax.scatter(Z[m, 0], Z[m, 1], s=8, alpha=0.9, label=classes[int(c)])

    ax.set_title("t-SNE (features after fusion)")
    ax.set_xlabel("t-SNE-1")
    ax.set_ylabel("t-SNE-2")
    ax.legend(frameon=False, fontsize=8, markerscale=2)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--data_root", default="data")
    ap.add_argument("--kfold", type=int, default=5)
    ap.add_argument("--fold", type=int, required=True)
    ap.add_argument("--img", type=int, default=224)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    ds_root = Path(args.data_root) / args.dataset
    out_dir = Path(args.out)

    classes = load_classes(ds_root)
    ncls = len(classes)

    paths, labels = load_fold_pool(ds_root)

    skf = StratifiedKFold(n_splits=args.kfold, shuffle=True, random_state=42)
    folds = list(skf.split(np.zeros(len(labels)), labels))
    train_idx, val_idx = folds[args.fold]
    val_paths = [paths[i] for i in val_idx]
    val_labels = labels[val_idx]

    tfm = transforms.Compose([transforms.Resize((args.img, args.img)), transforms.ToTensor()])
    val_ds = ImagePathDataset(val_paths, val_labels, tfm)
    val_loader = DataLoader(val_ds, batch_size=args.bs, shuffle=False, num_workers=4, pin_memory=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    ckpt = torch.load(args.ckpt, map_location=device)
    model = DinoMultiScaleClassifier(
        num_classes=ncls,
        freeze_backbone=True,
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    y, p, feat = predict_logits_and_features(model, val_loader, device)

    cm_norm = confusion_matrix(y, p, labels=np.arange(ncls), normalize="true")

    plot_confusion(cm_norm, classes, out_dir / "confusion_matrix_norm.png", dpi=300)
    plot_tsne(feat, y, classes, out_dir / "tsne.png", dpi=300)

    # Grad-CAM remains optional because it depends on selecting a specific internal layer to hook.
    print("[OK] wrote:", out_dir / "confusion_matrix_norm.png", "and", out_dir / "tsne.png")


if __name__ == "__main__":
    main()
