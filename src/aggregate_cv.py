import argparse, json
from pathlib import Path
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs_dir", required=True, help="e.g. runs_variant3_cv5/dataset_name")
    ap.add_argument("--kfold", type=int, default=5)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    runs_dir = Path(args.runs_dir)
    folds = []
    for f in range(args.kfold):
        p = runs_dir / f"fold_{f}" / "fold_summary.json"
        if not p.exists():
            raise FileNotFoundError(f"Missing fold summary: {p}")
        folds.append(json.loads(p.read_text()))

    metrics = {
        "acc": [],
        "f1_macro": [],
        "precision_macro": [],
        "recall_macro": [],
        "f1_weighted": [],
        "precision_weighted": [],
        "recall_weighted": [],
    }

    for fs in folds:
        m = fs["best_val_metrics"]
        metrics["acc"].append(m["acc"])
        metrics["f1_macro"].append(m["f1_macro"])
        metrics["precision_macro"].append(m["precision_macro"])
        metrics["recall_macro"].append(m["recall_macro"])
        metrics["f1_weighted"].append(m["f1_weighted"])
        metrics["precision_weighted"].append(m["precision_weighted"])
        metrics["recall_weighted"].append(m["recall_weighted"])

    summary = {
        "kfold": args.kfold,
        "runs_dir": str(runs_dir),
        "folds": [{"fold": f["fold"], "best_epoch": f["best_epoch"], "best_f1_macro": f["best_f1_macro"]} for f in folds],
        "metrics": {}
    }

    for k, arr in metrics.items():
        arr = np.array(arr, dtype=float)
        summary["metrics"][k] = {
            "mean": float(arr.mean()),
            "std": float(arr.std(ddof=1)) if len(arr) > 1 else 0.0,
            "per_fold": [float(x) for x in arr],
        }

    outp = Path(args.out)
    outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("[OK] wrote", outp)

if __name__ == "__main__":
    main()
