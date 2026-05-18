# -*- coding: utf-8 -*-
# src/corn_cm_clean.py
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DS = "corn_maize_leaf_disease"
FIG_DIR = Path("reports/figures")
TAB_DIR = Path("reports/tables")

def load_classes(ds):
    return (Path("data")/ds/"classes.txt").read_text().splitlines()

def load_conf_raw(ds, model):
    p = TAB_DIR / f"confusion_raw_{ds}_{model}_test.json"
    C = np.array(json.loads(p.read_text()))
    return C

def row_normalize(C):
    s = C.sum(axis=1, keepdims=True)
    s[s==0] = 1
    return C / s

def plot_side_by_side(C_left, C_right, classes, out_png, out_pdf, dpi=600):
    # نفس المجال اللوني للوحيدتين + colorbar واحدة على اليمين
    vmin, vmax = 0.0, 1.0
    n = len(classes)

    fig, axes = plt.subplots(1, 2, figsize=(8.8, 3.8), dpi=dpi, constrained_layout=False)
    plt.subplots_adjust(wspace=0.18, left=0.08, right=0.90, top=0.98, bottom=0.16)

    def draw(ax, C):
        im = ax.imshow(C, cmap="Blues", vmin=vmin, vmax=vmax, interpolation="nearest")
        ax.set_xticks(range(n)); ax.set_xticklabels(classes, rotation=25, ha="right")
        ax.set_yticks(range(n)); ax.set_yticklabels(classes)
        ax.set_xlabel("Predicted"); ax.set_ylabel("True")
        # أرقام النسب داخل الخلايا
        for i in range(n):
            for j in range(n):
                val = C[i, j]*100.0
                ax.text(j, i, f"{val:.1f}%", ha="center", va="center",
                        color=("white" if C[i,j] > 0.6 else "black"), fontsize=7.5)
        return im

    imL = draw(axes[0], C_left)
    imR = draw(axes[1], C_right)

    # colorbar مشتركة
    cax = fig.add_axes([0.92, 0.16, 0.02, 0.72])
    cb = fig.colorbar(imR, cax=cax)
    cb.ax.tick_params(labelsize=8)

    # حفظ بدقّة نشر
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=dpi, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(out_pdf, dpi=dpi, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)

def main():
    # أفضل موديل من best_models.json
    best = json.loads((TAB_DIR/"best_models.json").read_text())[DS]["best_model"]

    classes = load_classes(DS)
    Cb = load_conf_raw(DS, "baseline")
    Cf = load_conf_raw(DS, best)

    Cb = row_normalize(Cb)
    Cf = row_normalize(Cf)

    out_png = FIG_DIR / f"cm_clean_{DS}_baseline_vs_{best}.png"
    out_pdf = FIG_DIR / f"cm_clean_{DS}_baseline_vs_{best}.pdf"
    plot_side_by_side(Cb, Cf, classes, out_png, out_pdf, dpi=600)
    print("[OK] wrote:", out_png, "and", out_pdf)

if __name__ == "__main__":
    main()
