# -*- coding: utf-8 -*-
import argparse, re, os, json
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

FIG_DIR = Path("reports/figures")
RUNS_DIR = Path("runs")
OUT_DPI = 600
DEFAULT_PREFER = ["c2psa", "sam", "bam", "cbam", "baseline"]  # تفضيل التعادل

def ensure_rgb(im): return im.convert("RGB") if im.mode != "RGB" else im

def stitch_two(left_path, right_path, out_pdf, left_label="Baseline", right_label="Best",
               pad_px=24, lab_h=58):
    L = ensure_rgb(Image.open(left_path))
    R = ensure_rgb(Image.open(right_path))
    h = max(L.height, R.height)
    if L.height != h:
        L = L.resize((int(L.width * (h/L.height)), h), Image.LANCZOS)
    if R.height != h:
        R = R.resize((int(R.width * (h/R.height)), h), Image.LANCZOS)
    W = L.width + R.width + pad_px*3
    H = h + lab_h + pad_px*2
    canvas = Image.new("RGB", (W, H), (255,255,255))
    draw = ImageDraw.Draw(canvas)
    try: font = ImageFont.load_default()
    except: font = None
    draw.text((pad_px, pad_px), left_label, fill=(0,0,0), font=font)
    draw.text((pad_px*2 + L.width, pad_px), right_label, fill=(0,0,0), font=font)
    y0 = pad_px + lab_h - 8
    canvas.paste(L, (pad_px, y0))
    canvas.paste(R, (pad_px*2 + L.width, y0))
    out_pdf = Path(out_pdf); out_pdf.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_pdf, "PDF", resolution=OUT_DPI)

def read_test_acc(ds, model):
    p = RUNS_DIR/ds/model/"test_metrics.json"
    if not p.exists(): return None
    try:
        d = json.loads(p.read_text())
        acc = d.get("test_acc") or d.get("accuracy") or d.get("acc")
        return float(acc) if acc is not None else None
    except: return None

def choose_best_model_by_metrics(ds, prefer_order=None, tol=1e-6):
    prefer_order = prefer_order or DEFAULT_PREFER
    models = ["baseline","cbam","bam","sam","c2psa"]
    vals = {m: read_test_acc(ds, m) for m in models}
    finite = {m:v for m,v in vals.items() if v is not None}
    if not finite: return "baseline"
    mx = max(finite.values())
    ties = [m for m,v in finite.items() if abs(v-mx) <= tol]
    for m in prefer_order:
        if m in ties: return m
    return ties[0]

def first_exist(paths):
    for p in paths:
        if Path(p).exists(): return p
    return None

def confusion_pair_paths(dataset, best, normalized=True):
    tag = "_norm" if normalized else ""
    L = first_exist([FIG_DIR/f"confusion_{dataset}_baseline{tag}.png", FIG_DIR/f"confusion_{dataset}_baseline{tag}.pdf"])
    R = first_exist([FIG_DIR/f"confusion_{dataset}_{best}{tag}.png",   FIG_DIR/f"confusion_{dataset}_{best}{tag}.pdf"])
    out = FIG_DIR/f"sbs_confusion_{dataset}_baseline_vs_{best}{tag}.pdf"
    return L, R, out

def tsne_pair_paths(dataset, best):
    L = first_exist([FIG_DIR/f"tsne_{dataset}_baseline.png", FIG_DIR/f"tsne_{dataset}_baseline.pdf"])
    R = first_exist([FIG_DIR/f"tsne_{dataset}_{best}.png",   FIG_DIR/f"tsne_{dataset}_{best}.pdf"])
    out = FIG_DIR/f"sbs_tsne_{dataset}_baseline_vs_{best}.pdf"
    return L, R, out

def first_gradcam_pair(dataset, best):
    i = 0
    while i < 9999:
        L = FIG_DIR/f"gradcam_{dataset}_baseline_{i}.png"
        R = FIG_DIR/f"gradcam_{dataset}_{best}_{i}.png"
        if L.exists() and R.exists(): return str(L), str(R)
        i += 1
    return None, None

def run_one(dataset, best, normalized=True):
    # CM
    L,R,outp = confusion_pair_paths(dataset, best, normalized=normalized)
    if not (L and R) and normalized:
        L,R,outp = confusion_pair_paths(dataset, best, normalized=False)
    if L and R:
        stitch_two(L,R,outp,"Baseline", f"{best.upper()} (best)")
        print("[OK] CM   :", outp)
    else:
        print("[SKIP] CM :", dataset, "(missing CM images)")
    # t-SNE
    L,R,outp = tsne_pair_paths(dataset, best)
    if L and R:
        stitch_two(L,R,outp,"Baseline", f"{best.upper()} (best)")
        print("[OK] tSNE :", outp)
    else:
        print("[SKIP] tSNE:", dataset, "(missing t-SNE images)")
    # Grad-CAM (pair واحد فقط)
    L,R = first_gradcam_pair(dataset, best)
    if L and R:
        outp = FIG_DIR/f"sbs_gradcam_{dataset}_baseline_vs_{best}.pdf"
        stitch_two(L,R,outp,"Baseline", f"{best.upper()} (best)")
        print("[OK] GCAM :", outp)
    else:
        print("[SKIP] GCAM:", dataset, "(no matching pair)")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", help="داتا-سِت واحدة؛ إذا تُركت فارغة يشغّل كل الموجود.")
    ap.add_argument("--best", help="أفضليّة يدوية؛ وإلا تُحسب تلقائيًا مع تفضيل C2PSA ثم SAM.")
    ap.add_argument("--prefer", help="ترتيب تفضيل التعادل، مثال: c2psa,sam,bam,cbam,baseline")
    ap.add_argument("--nonnorm", action="store_true", help="استعمل CM غير مُطبّعة فقط.")
    args = ap.parse_args()

    prefer_order = [s.strip() for s in args.prefer.split(",")] if args.prefer else DEFAULT_PREFER
    if args.dataset:
        datasets = [args.dataset]
    else:
        datasets = sorted({p.parent.parent.name for p in RUNS_DIR.glob("*/baseline/test_metrics.json")})
    for ds in datasets:
        best = args.best or choose_best_model_by_metrics(ds, prefer_order=prefer_order)
        run_one(ds, best, normalized=(not args.nonnorm))

if __name__ == "__main__":
    main()
