import argparse, json
from pathlib import Path
import numpy as np
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from matplotlib import gridspec
import seaborn as sns

from models import (
    MobileViTv2Classifier, MobileViTv2_SAM,
    MobileViTv2_CBAM, MobileViTv2_BAM, MobileViTv2_C2PSA
)
ZOO = {
    "baseline": MobileViTv2Classifier,
    "sam": MobileViTv2_SAM, "cbam": MobileViTv2_CBAM,
    "bam": MobileViTv2_BAM, "c2psa": MobileViTv2_C2PSA
}

def load_ckpt(dataset, model_key, device):
    ckpt_path = Path("runs")/dataset/model_key/"best.ckpt"
    ckpt = torch.load(ckpt_path, map_location=device)
    net = ZOO[model_key](num_classes=int(ckpt["ncls"])).to(device).eval()
    net.load_state_dict(ckpt["model"])
    return net

@torch.no_grad()
def forward_feats_any(model, x):
    if hasattr(model, "backbone") and hasattr(model.backbone, "forward_features"):
        f = model.backbone.forward_features(x)
    elif hasattr(model, "forward_features"):
        f = model.forward_features(x)
    else:
        f = model(x)
    if isinstance(f, (list, tuple)): f = f[0]
    if f.ndim >= 3: f = f.mean(dim=tuple(range(2, f.ndim)))
    return f

@torch.no_grad()
def predict_and_feats(model, loader, device):
    preds=[]; gts=[]; feats=[]
    for x,y in loader:
        x = x.to(device, non_blocking=True)
        logits = model(x)
        preds.append(logits.argmax(1).cpu().numpy())
        gts.append(y.numpy())
        feats.append(forward_feats_any(model, x).cpu().numpy())
    return (np.concatenate(preds), np.concatenate(gts), np.concatenate(feats))

def plot_cm_sbs_norm(cm_base, cm_best, classes, best_name, out_pdf, dpi=600):
    fig = plt.figure(figsize=(10,4), dpi=dpi)
    gs  = gridspec.GridSpec(1,3, width_ratios=[1,1,0.045], wspace=0.28)
    ax1 = fig.add_subplot(gs[0,0])
    ax2 = fig.add_subplot(gs[0,1])
    cax = fig.add_subplot(gs[0,2])

    vmax=1.0
    sns.heatmap(cm_base, ax=ax1, cmap="Blues", vmin=0, vmax=vmax,
                annot=True, fmt=".1%", cbar=False, square=True,
                xticklabels=classes, yticklabels=classes, annot_kws={"fontsize":8})
    sns.heatmap(cm_best, ax=ax2, cmap="Blues", vmin=0, vmax=vmax,
                annot=True, fmt=".1%", cbar=True, cbar_ax=cax, square=True,
                xticklabels=classes, yticklabels=classes, annot_kws={"fontsize":8})

    ax1.set_title("baseline (normalized)", fontsize=11)
    ax2.set_title(f"{best_name} (normalized)", fontsize=11)
    for ax in (ax1, ax2):
        ax.set_xlabel("Predicted"); ax.set_ylabel("True")
        ax.tick_params(axis='x', rotation=30, labelsize=8)
        ax.tick_params(axis='y', rotation=0,  labelsize=8)

    fig.tight_layout()
    out_pdf = Path(out_pdf)
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_pdf.with_suffix(".pdf"), dpi=dpi, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)

def plot_tsne_sbs(feats_b, labs_b, feats_f, labs_f, classes, best_name, out_pdf, dpi=600):
    def emb(feats):
        Fz = StandardScaler().fit_transform(feats)
        return TSNE(n_components=2, learning_rate="auto", init="pca",
                    perplexity=max(5, min(30, len(Fz)//3 - 1)) if len(Fz)>=10 else 5,
                    random_state=42).fit_transform(Fz)

    Zb = emb(feats_b); Zf = emb(feats_f)
    fig = plt.figure(figsize=(10,4), dpi=dpi)
    gs  = gridspec.GridSpec(1,2, width_ratios=[1,1], wspace=0.25)
    ax1 = fig.add_subplot(gs[0,0]); ax2 = fig.add_subplot(gs[0,1])

    def scatter(ax, Z, y, title):
        for c in np.unique(y):
            m = (y==c)
            ax.scatter(Z[m,0], Z[m,1], s=6, alpha=0.9, label=classes[int(c)])
        ax.set_title(title, fontsize=11); ax.legend(markerscale=2.5, fontsize=7, frameon=False)
        ax.set_xlabel("t-SNE-1"); ax.set_ylabel("t-SNE-2")

    scatter(ax1, Zb, labs_b, "baseline")
    scatter(ax2, Zf, labs_f, best_name)

    fig.tight_layout()
    out_pdf = Path(out_pdf)
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_pdf.with_suffix(".pdf"), dpi=dpi, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dpi", type=int, default=600)
    args = ap.parse_args()

    dataset = "corn_maize_leaf_disease"
    best = json.loads(Path("reports/tables/best_models.json").read_text())[dataset]["best_model"]

    device  = "cuda" if torch.cuda.is_available() else "cpu"
    data_root = Path("data")/dataset
    classes = (data_root/"classes.txt").read_text().splitlines()
    tfm = transforms.Compose([transforms.Resize((224,224)), transforms.ToTensor()])
    ds_test = datasets.ImageFolder((data_root/"test").as_posix(), tfm)
    loader  = DataLoader(ds_test, batch_size=64, shuffle=False, num_workers=4)

    base = load_ckpt(dataset, "baseline", device)
    feat = load_ckpt(dataset, best, device)

    pb, yb, fb = predict_and_feats(base, loader, device)
    pf, yf, ff = predict_and_feats(feat, loader, device)
    assert (yb == yf).all(), "labels mismatch!"

    cm_b = confusion_matrix(yb, pb, labels=np.arange(len(classes)), normalize="true")
    cm_f = confusion_matrix(yf, pf, labels=np.arange(len(classes)), normalize="true")

    out_dir = Path("reports/figures")
    plot_cm_sbs_norm(cm_b, cm_f, classes, best.upper(),
                     out_dir/f"confusion_{dataset}_baseline_vs_{best}_norm",
                     dpi=args.dpi)
    plot_tsne_sbs(fb, yb, ff, yf, classes, best.upper(),
                  out_dir/f"tsne_{dataset}_baseline_vs_{best}",
                  dpi=args.dpi)
    print("[OK] wrote:",
          out_dir/f"confusion_{dataset}_baseline_vs_{best}_norm.pdf",
          "and", out_dir/f"tsne_{dataset}_baseline_vs_{best}.pdf")

if __name__ == "__main__":
    main()
