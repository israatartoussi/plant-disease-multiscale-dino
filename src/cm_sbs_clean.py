import json, argparse
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

FIG_DIR  = Path("reports/figures")
TABLES   = Path("reports/tables")

def load_cm(dataset: str, model: str):
    p = TABLES / f"confusion_raw_{dataset}_{model}_test.json"
    A = np.array(json.loads(p.read_text()), dtype=float)
    # row-normalize
    rs = A.sum(axis=1, keepdims=True); rs[rs==0] = 1.0
    return A / rs

def classes_for(dataset: str):
    return (Path("data")/dataset/"classes.txt").read_text().splitlines()

def best_model_for(dataset: str):
    j = json.loads((TABLES/"best_models.json").read_text())
    return j[dataset]["best_model"]

def draw(ax, C, classes, put_ylabel=False):
    im = ax.imshow(C, cmap="Blues", vmin=0, vmax=1, interpolation="nearest")
    n = len(classes)
    ax.set_xticks(range(n)); ax.set_xticklabels(classes, rotation=30, ha="right", fontsize=9)
    ax.set_yticks(range(n)); ax.set_yticklabels(classes, fontsize=10)
    if put_ylabel: ax.set_ylabel("True", fontsize=10)
    ax.set_xlabel("Predicted", fontsize=10)
    # حدود خفيفة
    for spine in ax.spines.values(): spine.set_linewidth(0.6)
    # الأرقام داخل المربعات
    for i in range(n):
        for j in range(n):
            v = C[i, j]
            ax.text(j, i, f"{v*100:.1f}%", ha="center", va="center",
                    color=("white" if v>0.6 else "black"), fontsize=8.5)
    return im

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--best", help="اختيار الموديل الأفضل يدوياً (إن لم يُعطَ يؤخذ من best_models.json)")
    ap.add_argument("--dpi", type=int, default=600)
    args = ap.parse_args()

    ds = args.dataset
    best = args.best or best_model_for(ds)
    classes = classes_for(ds)

    C_base = load_cm(ds, "baseline")
    C_best = load_cm(ds, best)

    # شكل مضغوط شبيه بالمثال: بلا colorbar ومسافة صغيرة
    fig, axes = plt.subplots(1, 2, figsize=(7.5, 3.2), dpi=args.dpi)
    plt.subplots_adjust(left=0.07, right=0.98, bottom=0.28, top=0.98, wspace=0.22)

    draw(axes[0], C_base, classes, put_ylabel=True)
    draw(axes[1], C_best, classes,  put_ylabel=True)

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    stem = f"sbs_cm_clean_{ds}_baseline_vs_{best}"
    pdf  = FIG_DIR / f"{stem}.pdf"
    png  = FIG_DIR / f"{stem}.png"
    plt.savefig(pdf, dpi=args.dpi, bbox_inches="tight", pad_inches=0.02)
    plt.savefig(png, dpi=args.dpi, bbox_inches=0, pad_inches=0.00)
    print("[OK]", pdf, "and", png)

if __name__ == "__main__":
    main()
