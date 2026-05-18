import argparse, json, random
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import datasets, transforms
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import cv2

# --------- Models Zoo ----------
from models import (
    MobileViTv2Classifier, MobileViTv2_SAM,
    MobileViTv2_CBAM, MobileViTv2_BAM, MobileViTv2_C2PSA
)
ZOO = {
    "baseline": MobileViTv2Classifier,
    "sam": MobileViTv2_SAM, "cbam": MobileViTv2_CBAM,
    "bam": MobileViTv2_BAM, "c2psa": MobileViTv2_C2PSA
}
TIE_ORDER = ["c2psa","sam","bam","cbam","baseline"]  # أولوية التعادل

def load_ckpt(dataset, model_key, device):
    ckpt_path = Path("runs")/dataset/model_key/"best.ckpt"
    if not ckpt_path.exists():
        raise FileNotFoundError(ckpt_path)
    ckpt = torch.load(ckpt_path, map_location=device)
    net = ZOO[model_key](num_classes=int(ckpt["ncls"])).to(device).eval()
    net.load_state_dict(ckpt["model"])
    return net

def pick_best_model(dataset):
    bm = Path("reports/tables/best_models.json")
    if bm.exists():
        j = json.loads(bm.read_text())
        if dataset in j and "best_model" in j[dataset]:
            return j[dataset]["best_model"]

    # fallback: من test_metrics.json
    scores = {}
    for m in ZOO.keys():
        p = Path("runs")/dataset/m/"test_metrics.json"
        if p.exists():
            try:
                d = json.loads(p.read_text())
                scores[m] = float(d.get("acc", d.get("accuracy", 0.0)))
            except Exception:
                pass
    if not scores:
        raise RuntimeError(f"No test metrics for {dataset}")
    best_val = max(scores.values())
    tied = [m for m,v in scores.items() if abs(v - best_val) < 1e-12]
    for pref in TIE_ORDER:
        if pref in tied:
            return pref
    return max(scores, key=scores.get)

# ------------- Grad-CAM -------------
class SimpleGradCAM:
    def __init__(self, model, layer):
        self.m, self.layer = model, layer
        self.act = None; self.grad = None
        self.h1 = layer.register_forward_hook(self._fwd)
        # متوافقة مع إصدارات بايتورتش الجديدة
        if hasattr(layer, "register_full_backward_hook"):
            self.h2 = layer.register_full_backward_hook(self._bwd)
        else:
            self.h2 = layer.register_backward_hook(self._bwd)

    def _fwd(self, m, i, o): self.act = o.detach()
    def _bwd(self, m, gi, go): self.grad = go[0].detach()

    def __call__(self, x):
        self.m.zero_grad(set_to_none=True)
        out = self.m(x)
        cls = out.argmax(1)
        onehot = torch.zeros_like(out).scatter_(1, cls.view(-1,1), 1.0)
        (out*onehot).sum().backward(retain_graph=True)
        w = self.grad.mean(dim=(2,3), keepdim=True)
        cam = torch.relu((w*self.act).sum(dim=1, keepdim=True))
        cam = (cam - cam.amin(dim=(2,3), keepdim=True)) / (cam.amax(dim=(2,3), keepdim=True)+1e-6)
        return cam

def find_last_conv(model: torch.nn.Module):
    last = None
    for m in model.modules():
        if isinstance(m, torch.nn.Conv2d):
            last = m
    return last

def overlay(img_tensor, cam, alpha=0.35):
    img = (img_tensor.permute(1,2,0).cpu().numpy()).clip(0,1)
    heat = (cam.squeeze().cpu().numpy()*255).astype("uint8")
    heat = cv2.applyColorMap(heat, cv2.COLORMAP_JET)[:, :, ::-1]/255.0
    return np.clip((1-alpha)*img + alpha*heat, 0, 1)

def make_pair_figure(ovl_base, ovl_best, title_left, title_right, dpi, out_pdf, out_png):
    plt.figure(figsize=(8,3.2), dpi=dpi)  # صف واحد عمودان
    for i,(arr,title) in enumerate([(ovl_base, title_left),(ovl_best, title_right)]):
        ax = plt.subplot(1,2,i+1)
        ax.imshow(arr); ax.set_title(title, fontsize=10); ax.axis("off")
    plt.tight_layout(pad=0.05)
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_pdf, bbox_inches="tight", pad_inches=0.02)
    plt.savefig(out_png, bbox_inches="tight", pad_inches=0.02, dpi=dpi)
    plt.close()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", help="اسم داتاسِت محدّد (أو استخدم --all)")
    ap.add_argument("--all", action="store_true", help="شغّل على كل الداتاسات المتوفرة")
    ap.add_argument("--n", type=int, default=1, help="عدد الصور لكل داتاسِت (افتراضياً 1)")
    ap.add_argument("--dpi", type=int, default=600)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--outdir", default="reports/figures")
    args = ap.parse_args()

    random.seed(args.seed); np.random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    outdir = Path(args.outdir)

    if args.all:
        ds_list = [p.name for p in Path("data").iterdir()
                   if (p/"test").exists() and (p/"classes.txt").exists()]
        ds_list.sort()
    else:
        if not args.dataset:
            raise SystemExit("provide --dataset or use --all")
        ds_list = [args.dataset]

    for d in ds_list:
        best = pick_best_model(d)
        base = load_ckpt(d, "baseline", device)
        bst  = load_ckpt(d, best, device)

        tfm = transforms.Compose([transforms.Resize((224,224)), transforms.ToTensor()])
        test_dir = Path("data")/d/"test"
        ds = datasets.ImageFolder(test_dir.as_posix(), tfm)
        if len(ds) == 0:
            print(f"[SKIP] empty test split: {d}")
            continue
        idxs = sorted(random.sample(range(len(ds)), k=min(args.n, len(ds))))

        lb = find_last_conv(base) or base
        lf = find_last_conv(bst)  or bst
        gcb = SimpleGradCAM(base, lb)
        gcf = SimpleGradCAM(bst,  lf)

        for idx in idxs:
            img,_ = ds[idx]
            x = img.unsqueeze(0).to(device)
            cam_b = F.interpolate(gcb(x), size=img.shape[-2:], mode="bilinear", align_corners=False)[0]
            cam_f = F.interpolate(gcf(x), size=img.shape[-2:], mode="bilinear", align_corners=False)[0]
            ovl_b = overlay(img, cam_b)
            ovl_f = overlay(img, cam_f)

            title_l = "Baseline (MobileViTv2)"
            title_r = f"Best (+{best.upper()})"
            base_name = f"sbs_gradcam_{d}_baseline_vs_{best}_idx{idx}"
            out_pdf = outdir/f"{base_name}.pdf"
            out_png = outdir/f"{base_name}.png"
            make_pair_figure(ovl_b, ovl_f, title_l, title_r, args.dpi, out_pdf, out_png)
            print(f"[OK] GCAM : {out_pdf}")

if __name__ == "__main__":
    main()
