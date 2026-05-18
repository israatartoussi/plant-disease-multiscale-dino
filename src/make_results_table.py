import csv, json
from pathlib import Path

# ترتيب الداتاست كما اعتمدناه
DATASETS = [
  "corn_maize_leaf_disease","bean_disease_uganda","guava_disease_pakistan",
  "papaya_leaf_disease","blackgram_leaf_disease","banana_leaf_disease",
  "coconut_tree_disease","rice_leaf_disease","sunflower_disease",
]
MODELS = ["baseline","cbam","bam","sam","c2psa"]

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

def read_last_acc(csv_path: Path):
    """ارجاع test_acc من آخر سطر بـ metrics.csv."""
    try:
        with csv_path.open("r", newline="") as f:
            rows = list(csv.DictReader(f))
        if not rows: 
            return None
        last = rows[-1]
        # عمود اسمه test_acc
        if "test_acc" in last and last["test_acc"] != "":
            return float(last["test_acc"])
        # أحياناً يكون acc أو val_acc فقط
        for key in ("acc","accuracy","val_acc"):
            if key in last and last[key] != "":
                return float(last[key])
    except Exception:
        return None
    return None

def read_json_acc(json_path: Path):
    """Backup: جرّب test_metrics.json"""
    try:
        d = json.loads(json_path.read_text())
        if "test_acc" in d:
            return float(d["test_acc"])
    except Exception:
        pass
    return None

def get_test_acc(run_dir: Path):
    """أولوية: metrics.csv ثم test_metrics.json"""
    csv_path  = run_dir / "metrics.csv"
    json_path = run_dir / "test_metrics.json"
    acc = None
    if csv_path.exists():
        acc = read_last_acc(csv_path)
    if acc is None and json_path.exists():
        acc = read_json_acc(json_path)
    return acc  # كنسبة (0..1)

def latex_esc(s:str) -> str:
    return s.replace("_","\\_")

def bold_best(values):
    """إرجاع strings مع \\textbf{} للأكبر (بتجاهل None)."""
    maxv = None
    for v in values:
        if v is not None:
            maxv = v if maxv is None else max(maxv, v)
    out = []
    for v in values:
        if v is None:
            out.append("--")
        else:
            txt = f"{v*100:.2f}"
            if v == maxv:
                out.append(f"\\textbf{{{txt}}}")
            else:
                out.append(txt)
    return out, maxv

def main():
    runs = Path("runs")
    out_csv = Path("reports/tables/results_summary.csv")
    out_tex = Path("reports/tables/results_summary.tex")

    # اجمع
    table = []
    for ds in DATASETS:
        row = {"dataset": ds}
        for m in MODELS:
            acc = get_test_acc(runs / ds / m)
            row[m] = acc
        table.append(row)

    # CSV
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["dataset"] + MODELS + ["best_model","best_acc(%)"])
        for r in table:
            vals = [r[m] for m in MODELS]
            _, mx = bold_best(vals)
            if mx is None:
                best_m, best_pct = "--", "--"
            else:
                best_m = MODELS[vals.index(mx)]
                best_pct = f"{mx*100:.2f}"
            w.writerow([r["dataset"]] + [("" if v is None else f"{v*100:.2f}") for v in vals] + [best_m, best_pct])

    # LaTeX
    lines = []
    lines.append("\\begin{table}[t]")
    lines.append("\\centering")
    lines.append("\\caption{Test accuracy (\\%) across attention variants. Best per dataset in bold.}")
    lines.append("\\label{tab:results_summary}")
    lines.append("\\begin{tabular}{lccccc c c}")
    lines.append("\\toprule")
    lines.append("\\textbf{Dataset} & \\textbf{Baseline} & \\textbf{CBAM} & \\textbf{BAM} & \\textbf{SAM} & \\textbf{C2PSA} & \\textbf{Best} & \\textbf{Acc (\\%)}\\\\")
    lines.append("\\midrule")
    for r in table:
        ds_name = latex_esc(PRETTY.get(r["dataset"], r["dataset"]))
        vals = [r[m] for m in MODELS]
        pretty_vals, mx = bold_best(vals)
        if mx is None:
            best_m, best_pct = "--", "--"
        else:
            best_m = MODELS[vals.index(mx)].upper()
            best_pct = f"{mx*100:.2f}"
        line = f"{ds_name} & " + " & ".join(pretty_vals) + f" & {best_m} & {best_pct}\\\\"
        lines.append(line)
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")

    out_tex.write_text("\n".join(lines), encoding="utf-8")
    print(f"[OK] wrote {out_csv} and {out_tex}")

if __name__ == "__main__":
    main()
