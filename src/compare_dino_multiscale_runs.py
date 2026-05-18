import argparse
import csv
import json
from pathlib import Path


MODEL_DIR_TO_NAME = {
    "dino_multiscale": "baseline_cbam",
    "dino_multiscale_gated": "gated_cbam",
    "dino_multiscale_cross_gated": "cross_gated_cbam",
    "dino_multiscale_parallel": "parallel",
}

MODEL_ORDER = [
    "baseline_cbam",
    "gated_cbam",
    "cross_gated_cbam",
    "parallel",
]


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--search_root",
        action="append",
        default=None,
        help="Root directory to scan. Can be passed multiple times. Default: current directory.",
    )
    ap.add_argument(
        "--csv",
        default=None,
        help="Optional CSV output path.",
    )
    ap.add_argument(
        "--status_csv",
        default=None,
        help="Optional status CSV path summarizing model coverage per dataset.",
    )
    ap.add_argument(
        "--expected_dataset",
        action="append",
        default=None,
        help="Expected dataset name. Can be passed multiple times.",
    )
    ap.add_argument(
        "--required_model",
        action="append",
        default=None,
        help="Required model label. Can be passed multiple times.",
    )
    ap.add_argument(
        "--drop_incomplete_datasets",
        action="store_true",
        help="Only keep datasets that contain all required models.",
    )
    return ap.parse_args()


def extract_metric(data: dict, top_level_key: str, nested_key: str):
    if top_level_key in data and data[top_level_key] is not None:
        return data[top_level_key]
    test = data.get("test", {})
    if nested_key in test and test[nested_key] is not None:
        return test[nested_key]
    return None


def extract_record(path: Path):
    model_dir = path.parent.name
    dataset = path.parent.parent.name
    model_name = MODEL_DIR_TO_NAME.get(model_dir)
    if model_name is None:
        return None

    data = json.loads(path.read_text(encoding="utf-8"))
    num_parameters = data.get("num_parameters")
    test_acc = extract_metric(data, "test_acc", "acc")
    test_f1_macro = extract_metric(data, "test_f1_macro", "f1_macro")
    test_f1_weighted = extract_metric(data, "test_f1_weighted", "f1_weighted")

    if test_acc is None or test_f1_macro is None:
        return None

    return {
        "dataset": dataset,
        "model": model_name,
        "num_parameters": None if num_parameters is None else int(num_parameters),
        "test_acc": float(test_acc),
        "test_f1_macro": float(test_f1_macro),
        "test_f1_weighted": None if test_f1_weighted is None else float(test_f1_weighted),
        "source_path": path.as_posix(),
    }


def find_metric_files(search_roots):
    files = []
    for root in search_roots:
        root_path = Path(root)
        if not root_path.exists():
            continue
        files.extend(root_path.rglob("test_metrics.json"))
    return sorted(set(files))


def collect_best_records(metric_files):
    best = {}
    for path in metric_files:
        rec = extract_record(path)
        if rec is None:
            continue

        key = (rec["dataset"], rec["model"])
        prev = best.get(key)
        if prev is None or rec["test_f1_macro"] > prev["test_f1_macro"]:
            best[key] = rec
    return best


def fmt_metric(value, digits=3):
    if value is None:
        return "--"
    return f"{value:.{digits}f}"


def fmt_params(value):
    if value is None:
        return "--"
    return f"{value / 1_000_000:.2f}M"


def print_table_for_dataset(dataset: str, rows: list[dict]):
    print(f"\n=== {dataset} ===")
    header = (
        f"{'model':<18}  {'params':>8}  {'acc':>7}  "
        f"{'f1_macro':>9}  {'f1_weighted':>11}  source"
    )
    print(header)
    print("-" * len(header))

    best_row = None
    for row in rows:
        if best_row is None or row["test_f1_macro"] > best_row["test_f1_macro"]:
            best_row = row

    for row in rows:
        best_mark = "*" if best_row is not None and row["model"] == best_row["model"] else " "
        print(
            f"{best_mark}{row['model']:<17}  "
            f"{fmt_params(row['num_parameters']):>8}  "
            f"{fmt_metric(row['test_acc']):>7}  "
            f"{fmt_metric(row['test_f1_macro']):>9}  "
            f"{fmt_metric(row['test_f1_weighted']):>11}  "
            f"{row['source_path']}"
        )

    if best_row is not None:
        print(
            f"Best by F1_macro: {best_row['model']} "
            f"(acc={fmt_metric(best_row['test_acc'])}, "
            f"f1_macro={fmt_metric(best_row['test_f1_macro'])})"
        )


def write_csv(path: Path, grouped_rows: dict[str, list[dict]]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "dataset",
            "model",
            "num_parameters",
            "test_acc",
            "test_f1_macro",
            "test_f1_weighted",
            "is_best_f1_macro",
            "source_path",
        ])

        for dataset, rows in grouped_rows.items():
            best_f1 = max((r["test_f1_macro"] for r in rows), default=None)
            for row in rows:
                writer.writerow([
                    dataset,
                    row["model"],
                    row["num_parameters"],
                    row["test_acc"],
                    row["test_f1_macro"],
                    row["test_f1_weighted"],
                    row["test_f1_macro"] == best_f1 if best_f1 is not None else False,
                    row["source_path"],
                ])


def build_status_rows(
    grouped_rows: dict[str, list[dict]],
    required_models: list[str],
    expected_datasets: list[str] | None,
):
    datasets = set(grouped_rows)
    if expected_datasets:
        datasets.update(expected_datasets)

    status_rows = []
    for dataset in sorted(datasets):
        rows = grouped_rows.get(dataset, [])
        present_models = {row["model"] for row in rows}
        missing_models = [model for model in required_models if model not in present_models]
        status_rows.append({
            "dataset": dataset,
            "num_present": len(present_models),
            "num_required": len(required_models),
            "status": "complete" if not missing_models else "incomplete",
            "present_models": ",".join(
                model for model in required_models if model in present_models
            ),
            "missing_models": ",".join(missing_models),
        })
    return status_rows


def write_status_csv(path: Path, status_rows: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "dataset",
            "status",
            "num_present",
            "num_required",
            "present_models",
            "missing_models",
        ])
        for row in status_rows:
            writer.writerow([
                row["dataset"],
                row["status"],
                row["num_present"],
                row["num_required"],
                row["present_models"],
                row["missing_models"],
            ])


def main():
    args = parse_args()
    search_roots = args.search_root or ["."]
    required_models = args.required_model or MODEL_ORDER
    expected_datasets = args.expected_dataset or []

    metric_files = find_metric_files(search_roots)
    best_records = collect_best_records(metric_files)

    grouped = {}
    for rec in best_records.values():
        grouped.setdefault(rec["dataset"], []).append(rec)

    for rows in grouped.values():
        rows.sort(
            key=lambda r: MODEL_ORDER.index(r["model"]) if r["model"] in MODEL_ORDER else 999
        )

    status_rows = build_status_rows(grouped, required_models, expected_datasets)

    if args.drop_incomplete_datasets:
        complete_datasets = {
            row["dataset"] for row in status_rows if row["status"] == "complete"
        }
        grouped = {
            dataset: rows for dataset, rows in grouped.items() if dataset in complete_datasets
        }

    for dataset in sorted(grouped):
        print_table_for_dataset(dataset, grouped[dataset])

    if args.csv:
        write_csv(Path(args.csv), grouped)
        print(f"\n[OK] wrote CSV: {args.csv}")

    if args.status_csv:
        write_status_csv(Path(args.status_csv), status_rows)
        print(f"[OK] wrote status CSV: {args.status_csv}")


if __name__ == "__main__":
    main()
