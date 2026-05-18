#!/usr/bin/env bash
set -u
set +e

# MUST run as a file (not pasted in terminal)
PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT" || exit 1

echo "[INFO] PROJECT_ROOT=$PROJECT_ROOT"

source "$PROJECT_ROOT/env.sh"

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

mkdir -p "$OUT"

echo "=== Variant 3 | CV=$K | Epochs=$EPOCHS ==="
echo "Output dir: $OUT"

for ds in "${DATASETS[@]}"; do
  echo "########################################"
  echo "DATASET: $ds"
  echo "########################################"

  for fold in $(seq 0 $((K-1))); do
    echo ""
    echo "---- Fold $fold / $((K-1)) ----"

    python -u src/train_variant3_cv.py \
      --dataset "$ds" \
      --data_root data \
      --kfold "$K" \
      --fold "$fold" \
      --epochs "$EPOCHS" \
      --bs "$BS" \
      --img "$IMG" \
      --lr "$LR" \
      --wd "$WD" \
      --seed "$SEED" \
      --freeze_dinov3 \
      --save_every_epoch \
      --out "$OUT"

    RET=$?
    if [ $RET -ne 0 ]; then
      echo "[WARN] Fold $fold failed for dataset $ds (exit code=$RET)"
      echo "       Continuing..."
    fi
  done

  echo ""
  echo ">>> Aggregating CV results for $ds"
  python -u src/aggregate_cv.py \
    --runs_dir "$OUT/$ds" \
    --kfold "$K" \
    --out "$OUT/$ds/cv_summary.json"
done

echo "========================================"
echo "[DONE] All datasets finished."
echo "Check results in: $OUT/"
echo "========================================"
