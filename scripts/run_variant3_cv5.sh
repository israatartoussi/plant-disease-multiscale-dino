#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

source ./env.sh

DATASETS=(
  banana_leaf_disease
  bean_disease_uganda
  blackgram_leaf_disease
  coconut_tree_disease
  corn_maize_leaf_disease
  guava_disease_pakistan
  papaya_leaf_disease
  sunflower_disease
)

K=5
EPOCHS=100
BS=32
IMG=224
LR=3e-4
WD=1e-2
SEED=42
OUT="runs_variant3_cv5"

# speed settings
export TOKENIZERS_PARALLELISM=false

for ds in "${DATASETS[@]}"; do
  for fold in $(seq 0 $((K-1))); do
    echo "========================================"
    echo "Dataset: $ds | Fold: $fold/$((K-1)) | Epochs: $EPOCHS"
    echo "OUT: $OUT"
    echo "========================================"

    python -m src.train_variant3_cv \
      --dataset "$ds" \
      --data_root "data" \
      --kfold "$K" \
      --fold "$fold" \
      --epochs "$EPOCHS" \
      --bs "$BS" \
      --img "$IMG" \
      --lr "$LR" \
      --wd "$WD" \
      --seed "$SEED" \
      --freeze_dinov3 \
      --out "$OUT"
  done

  # after all folds
  python -m src.aggregate_cv \
    --runs_dir "$OUT/$ds" \
    --kfold "$K" \
    --out "$OUT/$ds/cv_summary.json"
done

echo "[DONE] CV5 finished. Check: $OUT/"
