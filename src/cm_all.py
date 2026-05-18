import argparse, json
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# -------------------- Models Zoo --------------------
from models import (
    MobileViTv2Classifier, MobileViTv2_CBAM, MobileViTv2_BAM,
    MobileViTv2_SAM, MobileViTv2_C2PSA
)
ZOO = {
    "baseline": MobileViTv2Classifier,
    "cbam": MobileViTv2_CBAM,
    "bam": MobileViTv2_BAM,
    "sam": MobileViTv2_SAM,
    "c2psa": MobileViTv2_C2PSA,
}

def load_model(dataset: str, model_key: str, device: str = "cpu"):
    ckpt_path = Path("runs") / dataset / model_key / "best.ckpt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Missing checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device)
    net  = ZOO[model_key](num_classes=int(ckpt["ncls"])).to(device).eval()
    net.load_state_dict(ckpt["model"])
    return net

@torch.no_grad()
def predict_split(dataset: str, split: str, model, device: str = "cpu"):
    root = Path("data") / dataset / split
    tfm  = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    ds   = datasets.ImageFolder(root.as_posix(), tfm)
    loader = DataLoader(ds, batch_size=64, shuffle=False, num_workers=4, pin_memory=True)
    y_true, y_pred = [], []
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        logits = model(x)
        pred = logits.argmax(1).cpu().numpy()
        y_pred.append(pred)
        y_true.append(y.numpy())
    y_true = np.concatenate(y_true, axis=0)
    y_pred = np.concatenate(y_pred, axis=0)
    classes = [c for c, _ in sorted(ds.class_to_idx.items(), key=lambda kv: kv[1])]
    return y_true, y_pred, classes

def normalize_rows(cm: np.ndarray) -> np.ndarray:
    with np.errstate(divide="ignore", invalid="ignore"):
        row_sums = cm.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        cm_norm = cm.astype(np.float64) / row_sums
    return cm_norm

def plot_cm(cm: np.ndarray, class_names, title: str, out_png: Path, dpi=600, cmap="Blues"):
    fig, ax = plt.subplots(figsize=(4.5, 4.2), dpi=dpi)
    im = ax.imshow(cm, interpolation="nearest", cmap=cmap, vmin=0.0, vmax=1.0)
    ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set(
        xticks=np.arange(len(class_names)),
        yticks=np.arange(len(class_names)),
        xticklabels=class_names,
        yticklabels=class_names,
        xlabel="Predicted",
        ylabel="True",
        title=title,
    )
    ax.tick_params(axis="x", labelrotation=45, labelsize=7)
    ax.tick_params(axis="y", labelsize=7)
    thresh = 0.5
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            val = cm[i, j]
            txt = f"{val*100:.1f}%"
            ax.text(j, i, txt,
                    ha="center", va="center",
                    fontsize=6,
                    color="white" if val > thresh else "black")
    ax.set_aspect("equal")
    plt.tight_layout(pad=0.3)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png.with_suffix(".png"), bbox_inches="tight", pad_inches=0.02, dpi=dpi)
    plt.savefig(out_png.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)

def plot_cm_side_by_side(cm_a, cm_b, class_names, title_a, title_b, out_png: Path, dpi=600, cmap="Blues"):
    fig, axes = plt.subplots(1, 2, figsize=(8.5, 4.0), dpi=dpi, constrained_layout=True)
    for ax, cm, ttl in zip(axes, [cm_a, cm_b], [title_a, title_b]):
        im = ax.imshow(cm, interpolation="nearest", cmap=cmap, vmin=0.0, vmax=1.0)
        ax.set(
            xticks=np.arange(len(class_names)),
            yticks=np.arange(len(class_names)),
            xticklabels=class_names,
            yticklabels=class_names,
            xlabel="Predicted",
            ylabel="True",
            title=ttl,
        )
        ax.tick_params(axis="x", labelrotation=45, labelsize=7)
        ax.tick_params(axis="y", labelsize=7)
        thresh = 0.5
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                val = cm[i, j]
                txt = f"{val*100:.1f}%"
                ax.text(j, i, txt,
                        ha="center", va="center",
                        fontsize=6,
                        color="white" if val > thresh else "black")
        ax.set_aspect("equal")
    cax = fig.add_axes([0.92, 0.15, 0.015, 0.7])
    sm = matplotlib.cm.ScalarMappable(cmap=matplotlib.cm.get_cmap(cmap), norm=plt.Normalize(0,1))
    sm.set_array([])
    fig.colorbar(sm, cax=cax)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png.with_suffix(".png"), bbox_inches="tight", pad_inches=0.02, dpi=dpi)
    plt.savefig(out_png.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)

def run_one(dataset: str, split: str, dpi: int, outdir: Path, best_map: dict, device: str):
    print(f"==> {dataset} | split={split}")
    model_base = load_model(dataset, "baseline", device)
    y_true_b, y_pred_b, classes = predict_split(dataset, split, model_base, device)
    cm_b = confusion_matrix(y_true_b, y_pred_b, labels=np.arange(len(classes)))
    cm_b_n = normalize_rows(cm_b)

    best_key = best_map[dataset]["best_model"]
    model_best = load_model(dataset, best_key, device)
    y_true_f, y_pred_f, _ = predict_split(dataset, split, model_best, device)
    cm_f = confusion_matrix(y_true_f, y_pred_f, labels=np.arange(len(classes)))
    cm_f_n = normalize_rows(cm_f)

    tables_dir = Path("reports/tables"); tables_dir.mkdir(parents=True, exist_ok=True)
    for tag, y_true, y_pred in [("baseline", y_true_b, y_pred_b), (best_key, y_true_f, y_pred_f)]:
        rep = classification_report(y_true, y_pred, target_names=classes, digits=3, zero_division=0)
        (tables_dir / f"classification_report_{dataset}_{tag}.txt").write_text(rep)

    plot_cm(cm_b_n, classes,
            title=f"{dataset.replace('_',' ')} — baseline (normalized)",
            out_png=outdir / f"confusion_{dataset}_baseline_norm",
            dpi=dpi, cmap="Blues")
    plot_cm(cm_f_n, classes,
            title=f"{dataset.replace('_',' ')} — {best_key} (normalized)",
            out_png=outdir / f"confusion_{dataset}_{best_key}_norm",
            dpi=dpi, cmap="Blues")

    plot_cm_side_by_side(cm_b_n, cm_f_n, classes,
                         title_a="baseline (normalized)",
                         title_b=f"{best_key} (normalized)",
                         out_png=outdir / f"confusion_{dataset}_baseline_vs_{best_key}_norm",
                         dpi=dpi, cmap="Blues")

    print(f"[OK] saved PNG/PDF for {dataset}.")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test", choices=["val", "test"])
    ap.add_argument("--dpi", type=int, default=600)
    ap.add_argument("--outdir", default="reports/figures")
    ap.add_argument("--best_json", default="reports/tables/best_models.json")
    ap.add_argument("--datasets", nargs="*", default=[
        "corn_maize_leaf_disease","bean_disease_uganda","guava_disease_pakistan",
        "papaya_leaf_disease","blackgram_leaf_disease","banana_leaf_disease",
        "coconut_tree_disease","rice_leaf_disease","sunflower_disease"
    ])
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    out_dir = Path(args.outdir); out_dir.mkdir(parents=True, exist_ok=True)
    best_map = json.loads(Path(args.best_json).read_text())

    for dset in args.datasets:
        run_one(dset, args.split, args.dpi, out_dir, best_map, device)

if __name__ == "__main__":
    main()
