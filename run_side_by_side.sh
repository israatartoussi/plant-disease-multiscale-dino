#!/usr/bin/env bash
set -euo pipefail

DATASETS=(
  corn_maize_leaf_disease bean_disease_uganda guava_disease_pakistan
  papaya_leaf_disease blackgram_leaf_disease banana_leaf_disease
  coconut_tree_disease rice_leaf_disease sunflower_disease
)

for d in "${DATASETS[@]}"; do
  BEST=$(python - <<PY
import json; from pathlib import Path
j=json.loads(Path("reports/tables/best_models.json").read_text())
print(j["$d"]["best_model"])
PY
)
  echo "==> SIDE-BY-SIDE $d (best=$BEST)"

  # (اختياري) أعيدي توليد الـ viz لضمان وجود الملفات
  PYTHONPATH=. python -u src/viz.py \
    --dataset "$d" --best_model "$BEST" --split test \
    --tsne_auto --dpi 600 --n_cam 6

  # دمج الصور
  PYTHONPATH=. python -u src/make_side_by_side.py \
    --dataset "$d" --best_model "$BEST" --n_cam 6 --dpi 600 --outdir reports/figures
done
