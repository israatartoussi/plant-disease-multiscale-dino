import json, numpy as np
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DATASET = "corn_maize_leaf_disease"
# ملفات الـ CM الخام (غير مطبّعة) التي أنتجها eval.py
def raw_path(model):
    return Path("reports/tables")/f"confusion_raw_{DATASET}_{model}_test.json"

# أسماء الأصناف من ملف الداتا
classes = (Path("data")/DATASET/"classes.txt").read_text().splitlines()
n = len(classes)

# حمّلي مصفوفتين (baseline & best) وطبّعي الصفوف إلى نسب 0..1
def load_cm(model):
    A = np.array(json.loads(raw_path(model).read_text()), dtype=float)
    rs = A.sum(axis=1, keepdims=True)
    rs[rs==0] = 1.0
    return A/rs

C_base = load_cm("baseline")
C_best = load_cm(Path("reports/tables/best_models.json").exists() and
                 json.loads(Path("reports/tables/best_models.json").read_text())[DATASET]["best_model"]
                 or "baseline")

# تأكيد استخدام أفضل موديل مطلوب من المتغيّر البيئة (لو موجود)
import os
best_from_env = os.environ.get("BEST_MODEL")
if best_from_env:
    C_best = load_cm(best_from_env)

# الشكل والمارجنز (مهم لظهور الملصقات كاملة)
fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.6), dpi=600, constrained_layout=False)
plt.subplots_adjust(left=0.07, right=0.93, bottom=0.30, top=0.98, wspace=0.40)

def draw(ax, C, show_y_ticks):
    im = ax.imshow(C, cmap="Blues", vmin=0, vmax=1, interpolation="nearest")
    ax.set_xticks(range(n))
    ax.set_xticklabels(classes, rotation=30, ha="right", fontsize=10)
    ax.tick_params(axis="x", pad=8)
    ax.set_yticks(range(n))
    ax.set_yticklabels(classes if show_y_ticks else [], fontsize=10)
    ax.set_xlabel("Predicted", fontsize=11)
    # ضع "True" على يمين كل رسم فقط
    ax_secondary = ax.secondary_yaxis('right')
    ax_secondary.set_yticks(range(n))
    ax_secondary.set_yticklabels([""]*n)
    ax_secondary.set_ylabel("True", fontsize=11)
    for i in range(n):
        for j in range(n):
            val = C[i, j]*100
            ax.text(j, i, f"{val:.1f}%", ha="center", va="center",
                    color=("white" if C[i, j] > 0.6 else "black"), fontsize=9)
    return im

# baseline (يسار) مع y-ticks مرئية
draw(axes[0], C_base, show_y_ticks=True)
# best (يمين) بدون y-ticks يسار لأن عندنا "True" على اليمين فقط
draw(axes[1], C_best, show_y_ticks=False)

# شريط الألوان على يمين الشكل كله
from mpl_toolkits.axes_grid1 import make_axes_locatable
divider = make_axes_locatable(axes[1])
cax = divider.append_axes("right", size="2.5%", pad=0.10)
cb = plt.colorbar(axes[1].images[0], cax=cax)
cb.ax.tick_params(labelsize=9)
cb.set_ticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])

out_dir = Path("reports/figures")
out_dir.mkdir(parents=True, exist_ok=True)
(fig_out_pdf, fig_out_png) = (out_dir/"corn_cm_clean_sbs.pdf", out_dir/"corn_cm_clean_sbs.png")
plt.savefig(fig_out_pdf, dpi=600, bbox_inches="tight", pad_inches=0.02)
plt.savefig(fig_out_png, dpi=600, bbox_inches="tight", pad_inches=0.02)
print("[OK] wrote", fig_out_pdf, "and", fig_out_png)
