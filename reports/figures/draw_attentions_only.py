# draw_attentions_only.py
# Generates one pastel diagram per ATTENTION variant (Baseline, CBAM, BAM, SAM, C2PSA)
# + one General Framework figure. Suits the exact setup used in your experiments.
# Run: python draw_attentions_only.py

import os
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch

# ========= Editable config (set once for all figures) =========
INPUT_SIZE = "224×224×3"          # للعرض فقط
BACKBONE = "MobileViT v2 (base)"  # العمود الفقري
USE_FLATTEN = False               # إذا رأسك فعلاً يحتاج Flatten خليه True
DENSE_UNITS = 1024
DROPOUT_P = 0.5
N_CLASSES = 3                     # عدّليها إذا لزم

ATTN_VARIANTS = [
    ("baseline", None),
    ("cbam", "CBAM"),
    ("bam", "BAM"),
    ("sam", "SAM"),
    ("c2psa", "C2PSA"),
]

# ========= Colors (pastel) =========
COLORS = {
    "input":     "#f8f1ff",
    "backbone":  "#cfe8ff",
    "attention": "#cfead9",
    "gap":       "#d8c9ff",
    "bn":        "#ffd2a1",
    "flatten":   "#a9def9",
    "dense":     "#cdeac0",
    "dropout":   "#ffcad4",
    "softmax":   "#e2f0cb",
    "output":    "#fff2b2",
    "fixed":     "#e7f0ff",
    "changed":   "#e7ffe7",
    "process":   "#f0f0f0",
}

os.makedirs("figures", exist_ok=True)

# ========= Drawing helpers =========
def add_box(ax, xy, w, h, text, fc, ec="#333333", fontsize=11, lw=1.2, ha="center"):
    rect = Rectangle(xy, w, h, linewidth=lw, edgecolor=ec, facecolor=fc, zorder=2)
    ax.add_patch(rect)
    ax.text(xy[0] + w/2, xy[1] + h/2, text, ha=ha, va="center", fontsize=fontsize)
    return rect

def add_arrow(ax, xy1, xy2, lw=1.6):
    arrow = FancyArrowPatch(posA=xy1, posB=xy2, arrowstyle='-|>', mutation_scale=12,
                            linewidth=lw, color="#333333")
    ax.add_patch(arrow)
    return arrow

def init_ax(w=6, h=8):
    fig, ax = plt.subplots(figsize=(w, h), dpi=160)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 24)
    ax.axis('off')
    return fig, ax

# ========= Architecture figure (one per attention) =========
def draw_architecture(name, attention_label=None):
    fig, ax = init_ax()

    x = 2.5
    w = 5.0
    h = 1.7
    y = 21.5
    gap = 0.9

    # Input
    add_box(ax, (x, y), w, h, f"Input\n{INPUT_SIZE}", COLORS["input"])
    add_arrow(ax, (x + w/2, y), (x + w/2, y - 0.6))

    # Backbone
    y -= (h + gap)
    add_box(ax, (x, y), w, h, BACKBONE, COLORS["backbone"])
    add_arrow(ax, (x + w/2, y), (x + w/2, y - 0.6))

    # Attention (if any) — AFTER backbone, BEFORE GAP
    if attention_label:
        y -= (h + gap)
        add_box(ax, (x, y), w, h, attention_label, COLORS["attention"])
        add_arrow(ax, (x + w/2, y), (x + w/2, y - 0.6))

    # Head
    y -= (h + gap)
    head_blocks = [
        ("Global Average Pooling", "gap"),
        ("Batch Norm", "bn"),
    ]
    if USE_FLATTEN:
        head_blocks.append(("Flatten", "flatten"))
    head_blocks += [
        (f"Dense ({DENSE_UNITS})", "dense"),
        (f"Dropout ({DROPOUT_P})", "dropout"),
        (f"Dense ({N_CLASSES})\nSoftmax", "softmax"),
    ]

    for label, key in head_blocks:
        add_box(ax, (x, y), w, h, label, COLORS[key])
        add_arrow(ax, (x + w/2, y), (x + w/2, y - 0.6))
        y -= (h + gap)

    # Output
    add_box(ax, (x, y), w, h, "Output", COLORS["output"])

    title = f"MobileViTv2 — {'Baseline' if not attention_label else '+' + attention_label}"
    ax.set_title(title, fontsize=13, pad=12)

    out_path = f"figures/arch_{name}.png"
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")

# ========= General framework (what’s fixed vs changed) =========
def draw_framework():
    fig, ax = plt.subplots(figsize=(12, 6.8), dpi=160)
    ax.set_xlim(0, 32)
    ax.set_ylim(0, 16)
    ax.axis('off')

    x = 1
    y = 10.5
    w = 5.3
    h = 2.1
    gap = 0.8

    head_line = (
        f"GAP → BN{' → Flatten' if USE_FLATTEN else ''}\n"
        f"Dense({DENSE_UNITS}) → Dropout({DROPOUT_P})\n"
        f"Dense({N_CLASSES}) → Softmax"
    )

    boxes = [
        ("Datasets\n(split & CV)", "process"),
        (f"Preprocess\n(Resize {INPUT_SIZE}/Augment)", "process"),
        (BACKBONE, "backbone"),
        ("Attention\n(None/CBAM/BAM/SAM/C2PSA)", "attention"),
        (f"Head\n{head_line}", "softmax"),
        ("Training\n(Optimizer, LR, Epochs,\nEarly stop)", "process"),
        ("Metrics\nAcc / Prec / Rec / F1\n(Grad-CAM, t-SNE)", "process"),
    ]

    prev_x = None
    for label, key in boxes:
        add_box(ax, (x, y), w, h, label, COLORS.get(key, "#f5f5f5"))
        if prev_x is not None:
            add_arrow(ax, (prev_x + w, y + h/2), (x, y + h/2))
        prev_x = x
        x += (w + gap)

    # Legend: fixed vs changed
    add_box(ax, (1, 3.6), 12.5, 3.6,
            "Fixed across experiments:\n• Backbone: MobileViTv2 (base)\n• Classification Head (as specified)\n• Data pipeline & Training setup",
            COLORS["fixed"], lw=1.4, fontsize=11, ha="left")
    add_box(ax, (14.2, 3.6), 15.2, 3.6,
            "Changed per experiment:\n• Attention Module ∈ {None, CBAM, BAM, SAM, C2PSA}",
            COLORS["changed"], lw=1.4, fontsize=11, ha="left")

    ax.set_title("General Experimental Framework — Fixed vs Changed", fontsize=14, pad=14)

    out_path = "figures/framework_general.png"
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")

# ========= Run =========
if __name__ == "__main__":
    for name, attn in ATTN_VARIANTS:
        draw_architecture(name, attn)
    draw_framework()
    print("All figures saved in ./figures")
