import json, os
import numpy as np
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

DATASET = "corn_maize_leaf_disease"
CLASSES = (Path("data")/DATASET/"classes.txt").read_text().splitlines()
N = len(CLASSES)

def load_cm(model):
    p = Path("reports/tables")/f"confusion_raw_{DATASET}_{model}_test.json"
    A = np.array(json.loads(p.read_text()), dtype=float)
    rs = A.sum(axis=1, keepdims=True); rs[rs==0]=1.0
    return A/rs

C_base = load_cm("baseline")

best = os.environ.get("BEST_MODEL", "")
if not best:
    best = json.loads(Path("reports/tables/best_models.json").read_text())[DATASET]["best_model"]
C_best = load_cm(best)

fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.8), dpi=600)
plt.subplots_adjust(left=0.10, right=0.93, bottom=0.28, top=0.98, wspace=0.35)

def draw_left(ax, C):
    im = ax.imshow(C, cmap="Blues", vmin=0, vmax=1, interpolation="nearest")
    ax.set_xticks(range(N)); ax.set_xticklabels(CLASSES, rotation=30, ha="right", fontsize=10)
    ax.tick_params(axis="x", pad=8)
    ax.set_xlabel("Predicted", fontsize=11)
    ax.set_yticks(range(N)); ax.set_yticklabels(CLASSES, fontsize=11)
    ax.set_ylabel("True", fontsize=11)
    for i in range(N):
        for j in range(N):
            v = C[i, j]*100
            ax.text(j, i, f"{v:.1f}%", ha="center", va="center",
                    color=("white" if C[i,j] > 0.6 else "black"), fontsize=9)
    return im

def draw_right(ax, C):
    im = ax.imshow(C, cmap="Blues", vmin=0, vmax=1, interpolation="nearest")
    ax.set_xticks(range(N)); ax.set_xticklabels(CLASSES, rotation=30, ha="right", fontsize=10)
    ax.tick_params(axis="x", pad=8)
    ax.set_xlabel("Predicted", fontsize=11)
    ax.set_yticks(range(N)); ax.set_yticklabels([])
    sec = ax.secondary_yaxis('right')
    sec.set_yticks(range(N)); sec.set_yticklabels(CLASSES, fontsize=9)
    sec.set_ylabel("True", fontsize=11)
    for i in range(N):
        for j in range(N):
            v = C[i, j]*100
            ax.text(j, i, f"{v:.1f}%", ha="center", va="center",
                    color=("white" if C[i,j] > 0.6 else "black"), fontsize=9)
    return im

im1 = draw_left(axes[0], C_base)
im2 = draw_right(axes[1], C_best)

div = make_axes_locatable(axes[1])
cax = div.append_axes("right", size="2.5%", pad=0.35)
cb = plt.colorbar(im2, cax=cax)
cb.ax.tick_params(labelsize=9)
cb.set_ticks([0,0.2,0.4,0.6,0.8,1.0])

out_dir = Path("reports/figures"); out_dir.mkdir(parents=True, exist_ok=True)
pdf = out_dir/"corn_cm_clean_sbs_v4.pdf"
png = out_dir/"corn_cm_clean_sbs_v4.png"
plt.savefig(pdf, dpi=600, bbox_inches="tight", pad_inches=0.02)
plt.savefig(png, dpi=600, bbox_inches="tight", pad_inches=0.02)
print("[OK] wrote:", pdf, "and", png)
