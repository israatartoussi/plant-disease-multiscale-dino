# src/viz.py
import argparse
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import cv2

from models import (
    MobileViTv2Classifier, MobileViTv2_SAM,
    MobileViTv2_CBAM, MobileViTv2_BAM, MobileViTv2_C2PSA
)

ZOO = {
    "baseline": MobileViTv2Classifier,
    "sam": MobileViTv2_SAM, "cbam": MobileViTv2_CBAM,
    "bam": MobileViTv2_BAM, "c2psa": MobileViTv2_C2PSA
}

# ----------------------------- Utils -----------------------------
def load_best(dataset: str, model_key: str, device: str = "cpu"):
    ckpt_path = Path("runs") / dataset / model_key / "best.ckpt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Missing checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device)
    net = ZOO[model_key](num_classes=int(ckpt["ncls"])).to(device).eval()
    net.load_state_dict(ckpt["model"])
    return net

@torch.no_grad()
def forward_feats_any(model: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
    """
    حاول دائماً استخراج ميزات قبل اللوجتس.
    يرجّع [N, C] (Global Average لو لزم الأمر).
    """
    if hasattr(model, "backbone") and hasattr(model.backbone, "forward_features"):
        f = model.backbone.forward_features(x)
    elif hasattr(model, "forward_features"):
        f = model.forward_features(x)
    else:
        f = model(x)

    if isinstance(f, (list, tuple)):
        f = f[0]
    if not isinstance(f, torch.Tensor):
        raise RuntimeError("Could not obtain tensor features for t-SNE.")
    if f.ndim >= 3:  # خرائط ملامح → Global Avg
        f = f.mean(dim=tuple(range(2, f.ndim)))
    return f  # [N, C]

@torch.no_grad()
def extract_feats(model: torch.nn.Module, loader: DataLoader, device="cpu"):
    feats, labs = [], []
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        f = forward_feats_any(model, x)  # [N, C]
        feats.append(f.cpu().numpy())
        labs.append(y.numpy())
    if not feats:
        raise RuntimeError("No features collected (empty split?).")
    F = np.concatenate(feats, axis=0)
    y = np.concatenate(labs, axis=0)
    return F, y

def tsne_plot(
    feats: np.ndarray,
    labels: np.ndarray,
    class_names: list,
    out_path: Path,
    xlim=None, ylim=None,
    autoscale=False, pad=0.05, dpi=600,
    perplexity=None, title=None,
):
    if feats.size == 0:
        raise RuntimeError("Empty features for t-SNE.")

    Fz = StandardScaler().fit_transform(feats)
    n = len(Fz)
    if perplexity is None:
        perplexity = max(5, min(30, n // 3 - 1)) if n >= 10 else 5
    tsne = TSNE(n_components=2, learning_rate="auto", init="pca",
                perplexity=perplexity, random_state=42)
    Z = tsne.fit_transform(Fz)

    if autoscale or xlim is None or ylim is None:
        xmin, ymin = Z.min(axis=0); xmax, ymax = Z.max(axis=0)
        rx, ry = xmax - xmin, ymax - ymin
        rx = 1.0 if rx == 0 else rx
        ry = 1.0 if ry == 0 else ry
        xlim = (xmin - pad * rx, xmax + pad * rx)
        ylim = (ymin - pad * ry, ymax + pad * ry)

    plt.figure(figsize=(4, 4), dpi=dpi)
    for c in np.unique(labels):
        idx = (labels == c)
        plt.scatter(Z[idx, 0], Z[idx, 1], s=6, alpha=0.90, label=class_names[int(c)])
    plt.xlim(*xlim); plt.ylim(*ylim)
    if title: plt.title(title, fontsize=9)
    plt.legend(loc="best", fontsize=6, frameon=False, markerscale=2.5)
    plt.tight_layout()
    out_path = out_path.with_suffix(".png")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=dpi, bbox_inches="tight", pad_inches=0.02)
    plt.close()

# --------------------------- Grad-CAM ----------------------------
def find_last_conv(model: torch.nn.Module):
    last = None
    for m in model.modules():
        if isinstance(m, torch.nn.Conv2d):
            last = m
    return last

class GradCAM:
    """Hook على آخر Conv2d؛ يحسب CAM للفئة المختارة."""
    def __init__(self, model, target_layer: torch.nn.Module):
        self.model = model
        self.tl = target_layer
        self.activ = None
        self.grad = None
        self._h1 = self.tl.register_forward_hook(self._forward_hook)
        self._h2 = self.tl.register_full_backward_hook(self._backward_hook)

    def _forward_hook(self, m, i, o):
        if isinstance(o, torch.Tensor) and o.ndim == 4:
            self.activ = o.detach()

    def _backward_hook(self, m, gin, gout):
        g = gout[0]
        if isinstance(g, torch.Tensor) and g.ndim == 4:
            self.grad = g.detach()

    def __call__(self, logits: torch.Tensor, class_idx: torch.Tensor | None = None):
        if class_idx is None:
            class_idx = logits.argmax(dim=1)
        loss = logits.gather(1, class_idx.view(-1, 1)).sum()
        self.model.zero_grad(set_to_none=True)
        loss.backward(retain_graph=True)

        if self.activ is None or self.grad is None:
            raise RuntimeError("Grad/activ not captured. Pick another Conv layer.")

        w = self.grad.mean(dim=(2, 3), keepdim=True)      # [N,C,1,1]
        cam = (w * self.activ).sum(dim=1, keepdim=True)   # [N,1,H,W]
        cam = F.relu(cam)
        # normalize per-sample
        cmin = cam.amin(dim=(2, 3), keepdim=True)
        cmax = cam.amax(dim=(2, 3), keepdim=True)
        cam = (cam - cmin) / (cmax - cmin + 1e-6)
        return cam  # [N,1,H,W]

    def close(self):
        self._h1.remove(); self._h2.remove()

def overlay_cam(img_chw: torch.Tensor, cam_1hw: torch.Tensor):
    """img: [3,H,W] (0..1), cam: [1,H,W] (0..1)"""
    img = img_chw.permute(1, 2, 0).cpu().numpy()
    heat = (cam_1hw.squeeze().cpu().numpy() * 255).astype("uint8")
    heat = cv2.applyColorMap(heat, cv2.COLORMAP_JET)[..., ::-1] / 255.0  # BGR->RGB
    out = np.clip(0.65 * img + 0.35 * heat, 0.0, 1.0)
    return out

# ----------------------------- Main ------------------------------
def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--best_model", required=True, choices=list(ZOO.keys()))
    ap.add_argument("--split", default="test", choices=["val", "test"])
    # t-SNE controls
    ap.add_argument("--tsne_xlim", type=float, nargs=2, default=None)
    ap.add_argument("--tsne_ylim", type=float, nargs=2, default=None)
    ap.add_argument("--tsne_auto", action="store_true")
    ap.add_argument("--tsne_pad", type=float, default=0.05)
    ap.add_argument("--tsne_perplexity", type=float, default=None)
    # viz
    ap.add_argument("--n_cam", type=int, default=6)
    ap.add_argument("--dpi", type=int, default=600)
    return ap.parse_args()

def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    data_root = Path("data") / args.dataset
    classes = (data_root / "classes.txt").read_text(encoding="utf-8").splitlines()

    tfm = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    ds = datasets.ImageFolder((data_root / args.split).as_posix(), tfm)
    loader = DataLoader(ds, batch_size=32, shuffle=False, num_workers=4, pin_memory=True)

    # ---- t-SNE: baseline vs best ----
    out_dir = Path("reports/figures"); out_dir.mkdir(parents=True, exist_ok=True)
    base = load_best(args.dataset, "baseline", device)
    best = load_best(args.dataset, args.best_model, device)

    Fb, yb = extract_feats(base, loader, device)
    tsne_plot(
        Fb, yb, classes,
        out_path=out_dir / f"tsne_{args.dataset}_baseline",
        xlim=None if args.tsne_auto else args.tsne_xlim,
        ylim=None if args.tsne_auto else args.tsne_ylim,
        autoscale=args.tsne_auto or (args.tsne_xlim is None or args.tsne_ylim is None),
        pad=args.tsne_pad,
        dpi=args.dpi,
        perplexity=args.tsne_perplexity,
        title=None,
    )

    Ff, yf = extract_feats(best, loader, device)
    tsne_plot(
        Ff, yf, classes,
        out_path=out_dir / f"tsne_{args.dataset}_{args.best_model}",
        xlim=None if args.tsne_auto else args.tsne_xlim,
        ylim=None if args.tsne_auto else args.tsne_ylim,
        autoscale=args.tsne_auto or (args.tsne_xlim is None or args.tsne_ylim is None),
        pad=args.tsne_pad,
        dpi=args.dpi,
        perplexity=args.tsne_perplexity,
        title=None,
    )

    # ---- Grad-CAM: baseline vs best (آخر Conv تلقائيًا) ----
    tl_b = find_last_conv(base)
    tl_f = find_last_conv(best)
    if tl_b is None or tl_f is None:
        raise RuntimeError("No Conv2d layer found for Grad-CAM. Please pick a specific layer.")

    gcb = GradCAM(base, tl_b)
    gcf = GradCAM(best, tl_f)

    N = min(args.n_cam, len(ds))
    idxs = np.linspace(0, len(ds) - 1, num=N, dtype=int)
    for i, idx in enumerate(idxs, 1):
        img, _ = ds[idx]
        x = img.unsqueeze(0).to(device, non_blocking=True)

        base_logits = base(x)
        best_logits = best(x)

        cam_b = gcb(base_logits)[0]
        cam_f = gcf(best_logits)[0]

        cam_b = F.interpolate(cam_b.unsqueeze(0), size=img.shape[-2:], mode="bilinear", align_corners=False)[0]
        cam_f = F.interpolate(cam_f.unsqueeze(0), size=img.shape[-2:], mode="bilinear", align_corners=False)[0]

        plt.imsave(out_dir / f"gradcam_{args.dataset}_baseline_{i}.png",
                   overlay_cam(img, cam_b), dpi=args.dpi)
        plt.imsave(out_dir / f"gradcam_{args.dataset}_{args.best_model}_{i}.png",
                   overlay_cam(img, cam_f), dpi=args.dpi)

    gcb.close(); gcf.close()
    print("[OK] Saved t-SNE and Grad-CAM (baseline vs best).")

if __name__ == "__main__":
    main()
