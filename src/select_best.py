# src/select_best.py
import json, glob
from pathlib import Path

def main():
    root = Path("runs")
    results = {}
    for ds_dir in sorted(root.iterdir()):
        if not ds_dir.is_dir(): continue
        best_model=None; best_acc=-1.0
        for model_dir in ds_dir.iterdir():
            tj = model_dir/"test_metrics.json"
            if tj.exists():
                data = json.loads(tj.read_text())
                if data.get("test_acc", -1) > best_acc:
                    best_acc = data["test_acc"]; best_model = model_dir.name
        if best_model:
            results[ds_dir.name] = {"best_model": best_model, "test_acc": best_acc}
    out = Path("reports/tables"); out.mkdir(parents=True, exist_ok=True)
    (out/"best_models.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print("[Best models]")
    for ds, info in results.items():
        print(f"{ds}: {info['best_model']} (test_acc={info['test_acc']:.3f})")
        print(f"  eval: python src/eval.py --dataset {ds} --model baseline --split test")
        print(f"  eval: python src/eval.py --dataset {ds} --model {info['best_model']} --split test")
        print(f"  viz : python src/viz.py  --dataset {ds} --best_model {info['best_model']} --split test --n_cam 6")
    print(f"\nSaved reports/tables/best_models.json")

if __name__ == "__main__":
    main()
