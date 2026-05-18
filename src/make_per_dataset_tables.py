# src/make_per_dataset_tables.py
import json
from pathlib import Path

DATASETS = [
    "corn_maize_leaf_disease","bean_disease_uganda","guava_disease_pakistan",
    "papaya_leaf_disease","blackgram_leaf_disease","banana_leaf_disease",
    "coconut_tree_disease","rice_leaf_disease","sunflower_disease",
]
PRETTY = {
    "corn_maize_leaf_disease":"Corn/Maize",
    "bean_disease_uganda":"Bean (Uganda)",
    "guava_disease_pakistan":"Guava (Pakistan)",
    "papaya_leaf_disease":"Papaya",
    "blackgram_leaf_disease":"Blackgram",
    "banana_leaf_disease":"Banana",
    "coconut_tree_disease":"Coconut",
    "rice_leaf_disease":"Rice",
    "sunflower_disease":"Sunflower",
}

# ترتيب العرض
MODELS = ["baseline","cbam","bam","sam","c2psa"]
ROW_LABEL = {
    "baseline": "MobileViTv2 (baseline)",
    "cbam": "+ CBAM",
    "bam": "+ BAM",
    "sam": "+ SAM",
    "c2psa": "+ C2PSA",
}

def read_metrics(ds: str, model: str):
    """رجّع dict فيه: acc(%), prec, rec, f1  أو None إذا الملف ناقص."""
    p = Path("runs")/ds/model/"test_metrics.json"
    if not p.exists():
        return None
    d = json.loads(p.read_text())
    # مفاتيح محتملة كما استعملنا سابقًا
    acc  = d.get("test_acc") or d.get("accuracy") or d.get("acc")
    prec = d.get("precision_weighted") or d.get("precision") or d.get("prec")
    rec  = d.get("recall_weighted")    or d.get("recall")    or d.get("rec")
    f1   = d.get("f1_weighted")        or d.get("f1")        or d.get("f1_score")
    if acc is None:  # لازم على الأقل الـ accuracy
        return None
    return {
        "acc": float(acc)*100.0,  # ٪
        "prec": None if prec is None else float(prec),
        "rec":  None if rec  is None else float(rec),
        "f1":   None if f1   is None else float(f1),
    }

def col_best(values, tol=1e-12):
    """حدّد الأفضل (قد يكون أكثر من واحد)؛ values قائمة float أو None."""
    finite = [v for v in values if v is not None]
    if not finite:
        return set()
    m = max(finite)
    return {i for i,v in enumerate(values) if v is not None and abs(v-m) <= tol}

def fmtn(v, kind):
    if v is None:
        return "--"
    if kind == "acc":
        return f"{v:.2f}"
    # precision/recall/f1
    return f"{v:.3f}"

def make_table_for_dataset(ds: str):
    # اجمع المتريكس لكل موديل
    rows = []
    for m in MODELS:
        mt = read_metrics(ds, m)
        rows.append(mt)

    # أعمدة للمقارنة
    accs  = [r and r["acc"]  for r in rows]
    precs = [r and r["prec"] for r in rows]
    recs  = [r and r["rec"]  for r in rows]
    f1s   = [r and r["f1"]   for r in rows]

    b_acc  = col_best(accs)
    b_prec = col_best(precs)
    b_rec  = col_best(recs)
    b_f1   = col_best(f1s)

    def maybe_bold(text, cond):  # cond = True/False
        return f"\\textbf{{{text}}}" if cond else text

    lines = []
    lines += [
        "\\begin{table}[t]",
        "\\centering",
        f"\\caption{{Performance on {PRETTY.get(ds, ds.replace('_',' ').title())} (test split). "
        "Best per column in bold.}}",
        f"\\label{{tab:{ds}_per_model}}",
        "\\begin{tabular}{lcccc}",
        "\\toprule",
        "\\textbf{Model} & \\textbf{Accuracy (\\%)} & \\textbf{Precision} & \\textbf{Recall} & \\textbf{F1-Score}\\\\",
        "\\midrule"
    ]

    for i,m in enumerate(MODELS):
        r = rows[i]
        acc  = fmtn(None if r is None else r["acc"],  "acc")
        prec = fmtn(None if r is None else r["prec"], "other")
        rec  = fmtn(None if r is None else r["rec"],  "other")
        f1   = fmtn(None if r is None else r["f1"],   "other")

        acc  = maybe_bold(acc,  i in b_acc)
        prec = maybe_bold(prec, i in b_prec)
        rec  = maybe_bold(rec,  i in b_rec)
        f1   = maybe_bold(f1,   i in b_f1)

        lines.append(f"{ROW_LABEL[m]} & {acc} & {prec} & {rec} & {f1}\\\\")

    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]

    out_dir = Path("reports/tables")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_tex = out_dir / f"{ds}_metrics_table.tex"
    out_tex.write_text("\n".join(lines), encoding="utf-8")
    return out_tex

def main():
    outs = []
    for ds in DATASETS:
        outs.append(str(make_table_for_dataset(ds)))
    summary = Path("reports/tables/_include_all_metrics_tables.tex")
    summary.write_text("\n".join(f"\\input{{{p}}}" for p in outs), encoding="utf-8")
    print("[OK] wrote:")
    for p in outs:
        print("  -", p)
    print("You can include all at once via:")
    print("  \\input{reports/tables/_include_all_metrics_tables.tex}")

if __name__ == "__main__":
    main()
