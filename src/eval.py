import argparse, json
from pathlib import Path

import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import confusion_matrix, classification_report
import numpy as np

from models import (
    MobileViTv2Classifier, MobileViTv2_SAM,
    MobileViTv2_CBAM, MobileViTv2_BAM, MobileViTv2_C2PSA
)

ZOO = {
    "baseline": MobileViTv2Classifier,
    "sam": MobileViTv2_SAM, "cbam": MobileViTv2_CBAM,
    "bam": MobileViTv2_BAM, "c2psa": MobileViTv2_C2PSA
}

def load_best(dataset: str, model_key: str, device: str = "cpu"):
    ckpt_path = Path("runs") / dataset / model_key / "best.ckpt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Missing checkpoint: {ckpt_path}")

    classes = (Path("data")/dataset/"classes.txt").read_text(encoding="utf-8").splitlines()
    ncls = len(classes)

    # safer loading; إن صار تعارض مع حفظ قديم، بدّلي للسطر من دون weights_only
    obj = torch.load(ckpt_path, map_location=device, weights_only=True)
    state_dict = obj["model"] if (isinstance(obj, dict) and "model" in obj) else obj

    net = ZOO[model_key](num_classes=ncls).to(device).eval()
    net.load_state_dict(state_dict)
    return net

def plot_confusion_normalized(cm, class_names, out_png, dpi=600):
    """cm is row-normalized (each row sums to 1.0). We display % with one decimal."""
    sns.set_theme(context="paper", style="white", font_scale=1.1)
    cm_pct = (cm * 100.0).astype(float)

    plt.figure(figsize=(4,4), dpi=dpi)
    ax = sns.heatmap(
        cm_pct, annot=True, fmt=".1f", cmap="Blues", cbar=False,
        xticklabels=class_names, yticklabels=class_names,
        linewidths=0.5, linecolor="#DDDDDD", square=True
    )
    # أضف علامة % في الخانة العلوية من الشرح
    for t in ax.texts:
        t.set_text(f"{float(t.get_text()):.1f}%")

    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    plt.tight_layout()
    out_png = Path(out_png).with_suffix(".png")
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=dpi, bbox_inches="tight", pad_inches=0.02)
    plt.close()

@torch.no_grad()
def evaluate(dataset: str, model_key: str, split: str = "test", device: str = None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    data_root = Path("data")/dataset
    classes = (data_root/"classes.txt").read_text(encoding="utf-8").splitlines()

    tfm = transforms.Compose([transforms.Resize((224,224)), transforms.ToTensor()])
    ds  = datasets.ImageFolder((data_root/split).as_posix(), tfm)
    loader = DataLoader(ds, batch_size=64, shuffle=False, num_workers=4)

    net = load_best(dataset, model_key, device)

    y_true, y_pred = [], []
    for x, y in loader:
        x = x.to(device)
        logits = net(x)
        y_true.extend(y.cpu().numpy().tolist())
        y_pred.extend(logits.argmax(1).cpu().numpy().tolist())

    # row-normalized confusion (كل صف = 1.0)
    cm_norm = confusion_matrix(
        y_true, y_pred, labels=list(range(len(classes))), normalize="true"
    )

    # حفظ التقارير (العدّ المطلق + التصنيف) إذا حبيتي ترجعيلهم
    tables_dir = Path("reports/tables"); tables_dir.mkdir(parents=True, exist_ok=True)
    cm_abs = confusion_matrix(y_true, y_pred, labels=list(range(len(classes))))
    (tables_dir/f"confusion_raw_{dataset}_{model_key}_{split}.json").write_text(
        json.dumps(cm_abs.tolist(), indent=2), encoding="utf-8"
    )
    rep = classification_report(y_true, y_pred, target_names=classes, output_dict=True, zero_division=0)
    (tables_dir/f"classification_report_{dataset}_{model_key}_{split}.json").write_text(
        json.dumps(rep, indent=2), encoding="utf-8"
    )

    # صورة CM مُطبَّعة وبجودة عالية
    fig_dir = Path("reports/figures")
    plot_confusion_normalized(cm_norm, classes, fig_dir/f"confusion_{dataset}_{model_key}.png", dpi=600)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--model", required=True, choices=list(ZOO.keys()))
    ap.add_argument("--split", default="test", choices=["val","test"])
    args = ap.parse_args()
    evaluate(args.dataset, args.model, args.split)

if __name__ == "__main__":
    main()
