import json, math, csv
from pathlib import Path

DATASETS = [
    "corn_maize_leaf_disease","bean_disease_uganda","guava_disease_pakistan",
    "papaya_leaf_disease","blackgram_leaf_disease","banana_leaf_disease",
    "coconut_tree_disease","rice_leaf_disease","sunflower_disease",
]
ATTN_MODELS = ["cbam","bam","sam","c2psa"]            # baseline مستبعد حسب طلبك
TIE_PRIORITY = {"c2psa": 4, "sam": 3, "bam": 2, "cbam": 1}  # تفضيل عند التعادل

# خرائط مفاتيح مرنة (ننزّلها lower ونجرّب بالترتيب)
KEYS_ACC = ["accuracy", "acc", "test_accuracy", "overall_acc"]
KEYS_P   = ["precision_weighted","precision","weighted_precision","precision_macro","p"]
KEYS_R   = ["recall_weighted","recall","weighted_recall","recall_macro","r"]
KEYS_F1  = ["f1_weighted","f1","f1_score","f1_macro"]

def to_float(x, default=0.0):
    try:
        return float(x)
    except Exception:
        return default

def find_any_key(d: dict, candidates):
    """فتّش بشكل مسطّح بالـ dict (lowercase) وارجع أول قيمة موجودة."""
    # فلّطن الدكت (مستوي واحد يكفي لأن تقاريرنا غالبًا سطحية)
    flat = {}
    def walk(obj, prefix=""):
        if isinstance(obj, dict):
            for k, v in obj.items():
                walk(v, f"{prefix}{k.lower()}.")
        else:
            flat[prefix[:-1]] = obj
    walk(d)
    # جرّب كل تسمية وكل “ذيل” ممكن
    for name in candidates:
        name_l = name.lower()
        # مطابقة مباشرة
        for k, v in flat.items():
            if k.split(".")[-1] == name_l:
                return to_float(v, None)
    return None

def read_metrics(dataset: str, model: str):
    p = Path("reports/tables")/f"classification_report_{dataset}_{model}_test.json"
    if not p.exists():
        return None
    try:
        j = json.loads(p.read_text())

        acc = find_any_key(j, KEYS_ACC)
        pr  = find_any_key(j, KEYS_P)
        rc  = find_any_key(j, KEYS_R)
        f1  = find_any_key(j, KEYS_F1)

        if acc is None and pr is None and rc is None and f1 is None:
            return None
        return {
            "acc": acc or 0.0,
            "prec": pr  or 0.0,
            "rec": rc   or 0.0,
            "f1":  f1   or 0.0
        }
    except Exception:
        return None

def pct(x):  # نسبة مئوية بمنزلتين
    return f"{round((x or 0.0)*100.0, 2):.2f}"

def pick_best(models_metrics):
    """اختيار أفضل attention حسب أعلى Accuracy، ومع كسر تعادل حسب TIE_PRIORITY."""
    best_m, best_acc, best_pri = None, -1.0, -1
    for m, met in models_metrics.items():
        if not met:
            continue
        acc = met["acc"] or 0.0
        pri = TIE_PRIORITY.get(m, 0)
        if (acc > best_acc) or (math.isclose(acc, best_acc, rel_tol=1e-12, abs_tol=1e-12) and pri > best_pri):
            best_m, best_acc, best_pri = m, acc, pri
    return best_m

def main():
    rows = []
    lines = []
    lines += [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Best attention module per dataset (test split). Values from JSON reports; best in \textbf{bold}.}",
        r"\label{tab:best_attention_summary}",
        r"\begin{tabular}{l l c c c c}",
        r"\toprule",
        r"\textbf{Dataset} & \textbf{Best} & \textbf{Acc (\%)} & \textbf{Precision} & \textbf{Recall} & \textbf{F1-Score}\\",
        r"\midrule",
    ]

    for d in DATASETS:
        mm = {m: read_metrics(d, m) for m in ATTN_MODELS}
        best = pick_best(mm)
        if best is None:
            pretty = d.replace("_"," ").title()
            lines.append(fr"{pretty} & -- & -- & -- & -- & --\\")
            rows.append([d,"","","","",""])
            continue

        met = mm[best]
        acc_s = pct(met["acc"])
        p_s   = f"{(met['prec'] or 0.0):.3f}"
        r_s   = f"{(met['rec']  or 0.0):.3f}"
        f1_s  = f"{(met['f1']   or 0.0):.3f}"

        pretty = d.replace("_"," ").title()
        lines.append(fr"{pretty} & \textbf{{{best.upper()}}} & \textbf{{{acc_s}}} & \textbf{{{p_s}}} & \textbf{{{r_s}}} & \textbf{{{f1_s}}}\\")
        rows.append([d, best, acc_s, p_s, r_s, f1_s])

    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]

    out_csv = Path("reports/tables/best_attention_summary.csv")
    out_tex = Path("reports/tables/best_attention_summary.tex")
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["dataset","best_model","acc_percent","precision","recall","f1"]); w.writerows(rows)
    out_tex.write_text("\n".join(lines), encoding="utf-8")
    print(f"[OK] wrote {out_csv} and {out_tex}")
if __name__ == "__main__":
    main()
