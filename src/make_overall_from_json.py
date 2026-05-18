import json, re
from pathlib import Path

root = Path("reports/tables")
pattern = re.compile(r"classification_report_(.+)_(baseline|cbam|bam|sam|c2psa)_test\.json$")

# ترتيب أسماء الداتا والموديلات
ORDER_DS = ["corn_maize_leaf_disease","bean_disease_uganda","guava_disease_pakistan",
            "papaya_leaf_disease","blackgram_leaf_disease","banana_leaf_disease",
            "coconut_tree_disease","rice_leaf_disease","sunflower_disease"]
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
MODELS = ["baseline","cbam","bam","sam","c2psa"]
PRETTY_M = {"baseline":"MobileViTv2 (baseline)","+cbam":"+ CBAM","+bam":"+ BAM","+sam":"+ SAM","+c2psa":"+ C2PSA"}

# اقرأ كل JSON (test)
data = {}
for p in root.glob("classification_report_*_test.json"):
    m = pattern.match(p.name)
    if not m: 
        continue
    ds, model = m.group(1), m.group(2)
    j = json.loads(p.read_text())
    # نتوقع مفاتيح: accuracy, precision_weighted, recall_weighted, f1_weighted
    data.setdefault(ds, {})[model] = {
        "acc": float(j.get("accuracy", 0.0)),
        "prec": float(j.get("precision_weighted", 0.0)),
        "rec": float(j.get("recall_weighted", 0.0)),
        "f1": float(j.get("f1_weighted", 0.0)),
    }

# دالة تنسيق
def pct(x): return f"{x*100:.2f}"

# اكتب LaTeX لكل داتا (نفس ستايلك بالمثال) + ملف شامل
out_dir = root / "per_dataset"
out_dir.mkdir(parents=True, exist_ok=True)

overall_lines = []
overall_lines.append("\\begin{table}[t]")
overall_lines.append("\\centering")
overall_lines.append("\\caption{Per-dataset results on the test split. Best per metric in bold.}")
overall_lines.append("\\label{tab:per_dataset_full}")
overall_lines.append("\\begin{tabular}{l l cccc}")
overall_lines.append("\\toprule")
overall_lines.append("\\textbf{Dataset} & \\textbf{Model} & \\textbf{Accuracy (\\%)} & \\textbf{Precision} & \\textbf{Recall} & \\textbf{F1-Score}\\\\")
overall_lines.append("\\midrule")

for ds in ORDER_DS:
    if ds not in data: 
        continue
    rows = data[ds]
    # حددي الأفضل لكل مقياس
    best_acc = max(MODELS, key=lambda m: rows.get(m,{}).get("acc",-1))
    best_pr  = max(MODELS, key=lambda m: rows.get(m,{}).get("prec",-1))
    best_rc  = max(MODELS, key=lambda m: rows.get(m,{}).get("rec",-1))
    best_f1  = max(MODELS, key=lambda m: rows.get(m,{}).get("f1",-1))

    # اكتب جدول لكل داتا
    per_lines = []
    per_lines.append("\\begin{table}[t]")
    per_lines.append("\\centering")
    per_lines.append(f"\\caption{{Results on \\texttt{{{ds}}} (test split).}}")
    per_lines.append("\\begin{tabular}{lcccc}")
    per_lines.append("\\toprule")
    per_lines.append("\\textbf{Model} & \\textbf{Accuracy (\\%)} & \\textbf{Precision} & \\textbf{Recall} & \\textbf{F1-Score}\\\\")
    per_lines.append("\\midrule")

    first_row = True
    for m in ["baseline","cbam","bam","sam","c2psa"]:
        if m not in rows: 
            continue
        r = rows[m]
        A = pct(r["acc"]); P = f"{r['prec']:.3f}"; R = f"{r['rec']:.3f}"; F = f"{r['f1']:.3f}"
        # بولد للأفضل في كل عمود
        A = f"\\textbf{{{A}}}" if m==best_acc else A
        P = f"\\textbf{{{P}}}" if m==best_pr  else P
        R = f"\\textbf{{{R}}}" if m==best_rc  else R
        F = f"\\textbf{{{F}}}" if m==best_f1  else F
        label = "MobileViTv2 (baseline)" if m=="baseline" else f"+ {m.upper()}"
        per_lines.append(f"{label} & {A} & {P} & {R} & {F}\\\\")
        # للسطر الأول بالجدول الشامل: بنكتب اسم الداتا
        ds_label = PRETTY[ds] if first_row else ""
        overall_lines.append(f"{ds_label} & {label} & {A} & {P} & {R} & {F}\\\\")
        first_row = False

    per_lines.append("\\bottomrule")
    per_lines.append("\\end{tabular}")
    per_lines.append("\\end{table}")

    (out_dir / f"{ds}_results.tex").write_text("\n".join(per_lines), encoding="utf-8")

overall_lines.append("\\bottomrule")
overall_lines.append("\\end{tabular}")
overall_lines.append("\\end{table}")

(root / "include_all_results.tex").write_text("\n".join(overall_lines), encoding="utf-8")
print("[OK] wrote per-dataset .tex files and reports/tables/include_all_results.tex from test JSON.")
