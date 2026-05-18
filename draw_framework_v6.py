import os
import matplotlib

import matplotlib
try:
    matplotlib.use("module://mplcairo.base")  # clean vector backend
except Exception:
    matplotlib.use("Agg")  # fallback (headless)

matplotlib.use("Agg")   # safe, headless
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyBboxPatch

os.makedirs("figures", exist_ok=True)

# ---------- style ----------
BLOCK_W, BLOCK_H, GAP = 2.8, 1.05, 0.85
COL = {
    "dataset":  "#f2f2f2",
    "prep":     "#f2f2f2",
    "backbone": "#cfe2ff",
    "attn":     "#d9f2e1",
    "head":     "#fff3cd",
    "train":    "#f2f2f2",
    "metrics":  "#f2f2f2",
    "edge":     "#333333",
}

def add_box(ax, x, y, w, h, txt, fc, rounded=False):
    if rounded:
        r = FancyBboxPatch((x,y), w,h, boxstyle="round,pad=0.02,rounding_size=10",
                           facecolor=fc, edgecolor=COL["edge"], linewidth=1.6, zorder=3)
    else:
        r = Rectangle((x,y), w,h, facecolor=fc, edgecolor=COL["edge"], linewidth=1.6, zorder=3)
    ax.add_patch(r)
    ax.text(x+w/2, y+h/2, txt, ha="center", va="center", fontsize=10)
    return (x, y, w, h)

def arrow_between(ax, left_box, right_box):
    lx, ly, lw, lh = left_box
    rx, ry, rw, rh = right_box
    x1, y1 = lx+lw, ly+lh/2
    x2, y2 = rx,     ry+rh/2
    ax.annotate("", xy=(x2-0.1, y2), xytext=(x1+0.1, y1),
                arrowprops=dict(arrowstyle="->", color=COL["edge"], lw=1.6))

def draw_all():
    fig, ax = plt.subplots(figsize=(16, 5), dpi=300, facecolor="white")
    ax.set_facecolor("white")
    ax.axis("off")

    # Top row
    x, y = 0.7, 2.6
    boxes = []
    for label, col in [
        ("Datasets\n(split & CV)", COL["dataset"]),
        ("Preprocess\n(224×224×3 • Augment)", COL["prep"]),
        ("Backbone\nMobileViT v2 (base)", COL["backbone"]),
    ]:
        boxes.append(add_box(ax, x, y, BLOCK_W, BLOCK_H, label, col, rounded=True))
        x += BLOCK_W + GAP

    # Attention (wider)
    W_ATT = 5.2
    boxes.append(add_box(ax, x, y, W_ATT, BLOCK_H, "Attention\nCBAM • BAM • SAM • C2PSA", COL["attn"], rounded=True))
    x += W_ATT + GAP

    # Head (wider)
    W_HEAD = 6.2
    head_txt = "Head\nGAP → BN → Dense(1024) → Dropout(0.5)\n→ Dense(#) + Softmax"
    boxes.append(add_box(ax, x, y, W_HEAD, BLOCK_H, head_txt, COL["head"], rounded=True))
    x += W_HEAD + GAP

    # Training / Metrics
    for label, col in [
        ("Training\nOptimizer • LR • Epochs\nEarly stop", COL["train"]),
        ("Metrics\nAcc • Prec • Rec • F1\nGrad-CAM • t-SNE", COL["metrics"]),
    ]:
        boxes.append(add_box(ax, x, y, BLOCK_W, BLOCK_H, label, col, rounded=True))
        x += BLOCK_W + GAP

    # arrows between boxes (straight)
    for i in range(len(boxes)-1):
        arrow_between(ax, boxes[i], boxes[i+1])

    # Title
    ax.text((0.7 + x)/2 - 0.3, y + BLOCK_H + 0.95,
            "General Experimental Framework — Common vs. Variable Components",
            ha="center", va="center", fontsize=13, fontweight="bold")

    # Bottom legend blocks (rounded)
    Lx, Ly, W_leg, H_leg = 1.2, 0.55, 6.8, 1.28
    left = FancyBboxPatch((Lx, Ly), W_leg, H_leg, boxstyle="round,pad=0.02,rounding_size=10",
                          facecolor="#cfe2ff", edgecolor=COL["edge"], lw=1.4)
    ax.add_patch(left)
    ax.text(Lx+W_leg/2, Ly+H_leg/2,
            "Common across experiments:\n• MobileViT v2 backbone\n• Data preprocessing & training setup\n• Classification head",
            ha="center", va="center", fontsize=9.6)

    Rx = Lx + W_leg + 0.9
    right = FancyBboxPatch((Rx, Ly), W_leg, H_leg, boxstyle="round,pad=0.02,rounding_size=10",
                           facecolor="#d9f2e1", edgecolor=COL["edge"], lw=1.4)
    ax.add_patch(right)
    ax.text(Rx+W_leg/2, Ly+H_leg/2,
            "Variable per experiment:\n• Attention module ∈ {CBAM, BAM, SAM, C2PSA}",
            ha="center", va="center", fontsize=9.6)

    # Limits
    ax.set_xlim(0, x + 0.6)
    ax.set_ylim(0.25, y + BLOCK_H + 1.7)

    # Rasterize everything in the PDF to avoid PDF viewer artifacts
    ax.set_rasterization_zorder(2.5)

    # Save: PNG, PDF (rasterized), and SVG
    plt.tight_layout()
    plt.savefig("figures/framework_general_v6.png", dpi=400, bbox_inches="tight", facecolor="white")
    plt.savefig("figures/framework_general_v6.pdf", dpi=400, bbox_inches="tight", facecolor="white")
    plt.savefig("figures/framework_general_v6.svg", bbox_inches="tight", facecolor="white")
    print("Saved: figures/framework_general_v6.(png|pdf|svg)")

if __name__ == "__main__":
    draw_all()
