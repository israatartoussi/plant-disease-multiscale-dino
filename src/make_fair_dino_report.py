import csv
import json
from pathlib import Path


DATASETS = [
    "bean_disease_uganda",
    "blackgram_leaf_disease",
    "corn_maize_leaf_disease",
]

PRETTY_DATASET = {
    "bean_disease_uganda": "Bean Disease (Uganda)",
    "blackgram_leaf_disease": "Blackgram Leaf Disease",
    "corn_maize_leaf_disease": "Corn/Maize Leaf Disease",
}

MODEL_ORDER = [
    "baseline_cbam",
    "gated_cbam",
    "cross_gated_cbam",
    "parallel",
]

PRETTY_MODEL = {
    "baseline_cbam": "Baseline",
    "gated_cbam": "Gated",
    "cross_gated_cbam": "Cross-Gated",
    "parallel": "Parallel",
}

RUNS = {
    "bean_disease_uganda": {
        "baseline_cbam": "runs_fair_bean_baseline/bean_disease_uganda/dino_multiscale/test_metrics.json",
        "gated_cbam": "runs_fair_bean_gated/bean_disease_uganda/dino_multiscale_gated/test_metrics.json",
        "cross_gated_cbam": "runs_fair_bean_cross_gated/bean_disease_uganda/dino_multiscale_cross_gated/test_metrics.json",
        "parallel": "runs_fair_bean_parallel/bean_disease_uganda/dino_multiscale_parallel/test_metrics.json",
    },
    "blackgram_leaf_disease": {
        "baseline_cbam": "runs_blackgram_baseline/blackgram_leaf_disease/dino_multiscale/test_metrics.json",
        "gated_cbam": "runs_blackgram_gated/blackgram_leaf_disease/dino_multiscale_gated/test_metrics.json",
        "cross_gated_cbam": "runs_blackgram_cross_gated/blackgram_leaf_disease/dino_multiscale_cross_gated/test_metrics.json",
        "parallel": "runs_blackgram_parallel/blackgram_leaf_disease/dino_multiscale_parallel/test_metrics.json",
    },
    "corn_maize_leaf_disease": {
        "baseline_cbam": "runs_fair_corn_baseline/corn_maize_leaf_disease/dino_multiscale/test_metrics.json",
        "gated_cbam": "runs_dino_multiscale_gated/corn_maize_leaf_disease/dino_multiscale_gated/test_metrics.json",
        "cross_gated_cbam": "runs_fair_corn_cross_gated/corn_maize_leaf_disease/dino_multiscale_cross_gated/test_metrics.json",
        "parallel": "runs_fair_corn_parallel/corn_maize_leaf_disease/dino_multiscale_parallel/test_metrics.json",
    },
}


def load_record(dataset: str, model: str, metrics_path: str) -> dict:
    path = Path(metrics_path)
    data = json.loads(path.read_text(encoding="utf-8"))
    return {
        "dataset": dataset,
        "model": model,
        "num_parameters": int(data["num_parameters"]),
        "accuracy": float(data["test_acc"]),
        "f1_macro": float(data["test_f1_macro"]),
        "f1_weighted": float(data["test_f1_weighted"]),
        "best_epoch": int(data["best_epoch"]),
        "best_val_f1_macro": float(data["best_val_f1_macro"]),
        "source_path": metrics_path,
    }


def collect_rows() -> dict[str, list[dict]]:
    grouped = {}
    for dataset in DATASETS:
        rows = []
        for model in MODEL_ORDER:
            rows.append(load_record(dataset, model, RUNS[dataset][model]))
        best_f1 = max(row["f1_macro"] for row in rows)
        for row in rows:
            row["is_best_f1_macro"] = row["f1_macro"] == best_f1
        grouped[dataset] = rows
    return grouped


def fmt_metric(value: float) -> str:
    return f"{value:.3f}"


def fmt_params(value: int) -> str:
    return f"{value / 1_000_000:.2f}M"


def write_csv_outputs(grouped: dict[str, list[dict]]) -> None:
    out_dir = Path("reports/tables")
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_path = out_dir / "dino_multiscale_comparison_fair.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
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
        for dataset in DATASETS:
            for row in grouped[dataset]:
                writer.writerow([
                    row["dataset"],
                    row["model"],
                    row["num_parameters"],
                    row["accuracy"],
                    row["f1_macro"],
                    row["f1_weighted"],
                    row["is_best_f1_macro"],
                    row["source_path"],
                ])

    status_path = out_dir / "dino_multiscale_comparison_fair_status.csv"
    with status_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "dataset",
            "status",
            "num_present",
            "num_required",
            "present_models",
            "missing_models",
        ])
        for dataset in DATASETS:
            writer.writerow([
                dataset,
                "complete",
                len(MODEL_ORDER),
                len(MODEL_ORDER),
                ",".join(MODEL_ORDER),
                "",
            ])


def dataset_slug(dataset: str) -> str:
    return dataset.replace("_leaf_", "_").replace("_disease", "")


def write_tex_tables(grouped: dict[str, list[dict]]) -> None:
    out_dir = Path("reports/tables")
    include_paths = []

    for dataset in DATASETS:
        rows = grouped[dataset]
        out_path = out_dir / f"dino_fair_{dataset}.tex"
        include_paths.append(out_path.as_posix())

        lines = [
            "\\begin{table}[t]",
            "\\centering",
            f"\\caption{{Fair comparison on {PRETTY_DATASET[dataset]} using the best checkpoint selected by validation macro-F1. The best model per dataset by test macro-F1 is shown in bold.}}",
            f"\\label{{tab:dino_fair_{dataset}}}",
            "\\begin{tabular}{lcccc}",
            "\\toprule",
            "\\textbf{Model} & \\textbf{Params} & \\textbf{Accuracy} & \\textbf{F1-macro} & \\textbf{F1-weighted}\\\\",
            "\\midrule",
        ]

        for row in rows:
            model_text = PRETTY_MODEL[row["model"]]
            f1_text = fmt_metric(row["f1_macro"])
            if row["is_best_f1_macro"]:
                model_text = f"\\textbf{{{model_text}}}"
                f1_text = f"\\textbf{{{f1_text}}}"
            lines.append(
                f"{model_text} & {fmt_params(row['num_parameters'])} & "
                f"{fmt_metric(row['accuracy'])} & {f1_text} & "
                f"{fmt_metric(row['f1_weighted'])}\\\\"
            )

        lines.extend([
            "\\bottomrule",
            "\\end{tabular}",
            "\\end{table}",
        ])
        out_path.write_text("\n".join(lines), encoding="utf-8")

    include_path = out_dir / "include_dino_fair_tables.tex"
    include_path.write_text(
        "\n".join(f"\\input{{{path}}}" for path in include_paths),
        encoding="utf-8",
    )


def best_row(rows: list[dict]) -> dict:
    return max(rows, key=lambda row: row["f1_macro"])


def write_markdown_report(grouped: dict[str, list[dict]]) -> None:
    out_path = Path("reports/final_fair_results.md")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    bean_best = best_row(grouped["bean_disease_uganda"])
    blackgram_best = best_row(grouped["blackgram_leaf_disease"])
    corn_best = best_row(grouped["corn_maize_leaf_disease"])

    def table_lines(dataset: str) -> list[str]:
        rows = grouped[dataset]
        lines = [
            f"### {PRETTY_DATASET[dataset]}",
            "",
            "| Model | Num parameters | Accuracy | F1_macro | F1_weighted |",
            "|---|---:|---:|---:|---:|",
        ]
        for row in rows:
            model_name = PRETTY_MODEL[row["model"]]
            if row["is_best_f1_macro"]:
                model_name = f"**{model_name}**"
            lines.append(
                f"| {model_name} | {fmt_params(row['num_parameters'])} | "
                f"{fmt_metric(row['accuracy'])} | {fmt_metric(row['f1_macro'])} | "
                f"{fmt_metric(row['f1_weighted'])} |"
            )
        lines.append("")
        return lines

    lines = [
        "# Final Fair-Run Results",
        "",
        "## Training Details",
        "",
        "All reported DINO multiscale models were evaluated under the same experimental protocol. For each dataset, the four variants (Baseline, Gated, Cross-Gated, and Parallel) used the same image size (224), batch size (4), number of epochs (50), learning rate (1e-4), weight decay (1e-2), random seed (42), and train/evaluation splits. The training pipeline also used the same augmentations for all variants: resize to 224x224, random horizontal flip, random vertical flip with probability 0.1, color jitter (0.1, 0.1, 0.1, 0.05), tensor conversion, and ImageNet normalization. Model selection was performed with the validation macro-F1 score, and final test metrics were computed from the corresponding `best.ckpt` checkpoint.",
        "",
        "For corn_maize_leaf_disease, the gated result was taken from `runs_dino_multiscale_gated/...`, whose checkpoint metadata matches the fair protocol above (224 image size, batch size 4, 50 epochs, learning rate 1e-4, weight decay 1e-2, seed 42, and best-checkpoint evaluation).",
        "",
        "## Final Comparison Tables",
        "",
    ]

    for dataset in DATASETS:
        lines.extend(table_lines(dataset))

    lines.extend([
        "## Results",
        "",
        f"Under the controlled fair-training setting, the best model differed across datasets. On Bean Disease (Uganda), the Gated variant achieved the strongest test performance with an accuracy of {fmt_metric(bean_best['accuracy'])} and a macro-F1 of {fmt_metric(bean_best['f1_macro'])}, outperforming Baseline, Cross-Gated, and Parallel. On Blackgram Leaf Disease, the Parallel variant was clearly superior, reaching {fmt_metric(blackgram_best['accuracy'])} accuracy and {fmt_metric(blackgram_best['f1_macro'])} macro-F1; this margin was substantial relative to the other variants and indicates that the more expressive fusion strategy was beneficial for this harder five-class dataset. In contrast, on Corn/Maize Leaf Disease, the Baseline model remained best, with {fmt_metric(corn_best['accuracy'])} accuracy and {fmt_metric(corn_best['f1_macro'])} macro-F1, while the added fusion mechanisms produced similar but slightly lower macro-F1 values.",
        "",
        "A broader pattern also emerges from the comparison. The gains from architectural complexity were dataset-dependent rather than uniform. Gated fusion was most effective on the three-class Bean dataset, Parallel fusion was most effective on the more heterogeneous Blackgram dataset, and the simpler Baseline model was already sufficient for Corn/Maize. These results suggest that adding multiscale interaction capacity can improve performance, but the benefit depends on the difficulty of the dataset and on whether the added fusion mechanism matches the structure of the visual decision boundaries.",
        "",
        "## Discussion",
        "",
        "The differences between Parallel, Gated, and Cross-Gated can be explained by how each variant controls information flow across scales. The Parallel model has the largest capacity, because it preserves multiple branches and lets the classifier combine richer multiscale descriptors. This can be advantageous when disease symptoms appear at different spatial extents or when inter-class boundaries are visually complex, which is consistent with its strong performance on Blackgram. However, the same extra capacity can be unnecessary on easier datasets, where it may add optimization difficulty without producing a better representation, as observed on Corn/Maize and Bean.",
        "",
        "The Gated model introduces an adaptive weighting mechanism that can suppress less useful scale features while preserving the most informative ones. This design appears to offer a good balance between flexibility and control, which likely explains why it performed best on Bean Disease (Uganda). In that dataset, the model may benefit from selectively emphasizing disease cues without carrying the full parameter burden of the Parallel variant. The result suggests that adaptive selection is helpful when the task is not extremely complex, but still requires finer control than a simple baseline can provide.",
        "",
        "The Cross-Gated model is more selective still, because the gating of one stream depends on information from another stream. In principle, that interaction can improve feature coordination, but it can also make learning more sensitive to optimization and to the quality of the cross-scale signals. If one branch becomes noisy or less reliable early in training, cross-gating may suppress useful evidence instead of reinforcing it. This likely explains why Cross-Gated remained competitive but did not become the top model on any of the three finalized fair datasets.",
        "",
        "Taken together, the final results indicate that there is no single universally best fusion strategy. Parallel fusion seems strongest when high representational capacity is needed, Gated fusion seems strongest when selective emphasis matters more than maximum capacity, and Cross-Gated may require better tuning or more data to consistently realize its theoretical advantage. For the present study, this dataset-specific behavior is itself an important finding, because it shows that the value of added fusion complexity should be judged relative to dataset difficulty rather than assumed to generalize uniformly across plant disease benchmarks.",
        "",
    ])

    out_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    grouped = collect_rows()
    write_csv_outputs(grouped)
    write_tex_tables(grouped)
    write_markdown_report(grouped)
    print("Wrote final fair comparison outputs to reports/.")


if __name__ == "__main__":
    main()
