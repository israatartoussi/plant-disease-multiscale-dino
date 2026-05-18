#!/usr/bin/env bash
set -euo pipefail

EPOCHS=100
BS=32

# نشغّل بايثون مع PYTHONPATH مضبوط
py(){ PYTHONPATH=. python -u "$@"; }

DATASETS=(
  corn_maize_leaf_disease
  bean_disease_uganda
  guava_disease_pakistan
  papaya_leaf_disease
  blackgram_leaf_disease
  banana_leaf_disease
  coconut_tree_disease
  rice_leaf_disease
  sunflower_disease
)
MODELS=(baseline cbam bam sam c2psa)

mkdir -p logs
ts(){ date "+%Y-%m-%d %H:%M:%S"; }
TSNE_XLIM="-800 800"; TSNE_YLIM="-800 800"; NCAM=6

for d in "${DATASETS[@]}"; do
  echo "[$(ts)] ===== DATASET: $d ====="
  for m in "${MODELS[@]}"; do
    # تخطّي لو التجربة خلصت قبل
    if [ -f "runs/$d/$m/best.ckpt" ]; then
      echo "[$(ts)] >>> SKIP $d / $m (already has best.ckpt)"
      continue
    fi
    echo "[$(ts)] >>> TRAIN $d / $m (epochs=$EPOCHS, bs=$BS)"
    py src/train_attn.py --dataset "$d" --model "$m" --epochs "$EPOCHS" --bs "$BS" \
      |& tee "logs/${d}__${m}.log"
  done

  echo "[$(ts)] >>> SELECT BEST for $d"
  py src/select_best.py |& tee "logs/${d}__select_best.log"

  BEST=$(py - <<PY
import json; d="$d"
p="reports/tables/best_models.json"
print(json.load(open(p))[d]["best_model"])
PY
)
  echo "[$(ts)] >>> BEST for $d = $BEST"

  echo "[$(ts)] >>> CONFUSION (baseline + $BEST) for $d"
  py src/eval.py --dataset "$d" --model baseline --split test |& tee -a "logs/${d}__eval.log"
  py src/eval.py --dataset "$d" --model "$BEST"  --split test |& tee -a "logs/${d}__eval.log"

  echo "[$(ts)] >>> VIZ (t-SNE & Grad-CAM) baseline vs $BEST for $d"
  py src/viz.py --dataset "$d" --best_model "$BEST" --split test \
    --n_cam "$NCAM" --tsne_xlim $TSNE_XLIM --tsne_ylim $TSNE_YLIM \
    |& tee -a "logs/${d}__viz.log"

  echo "[$(ts)] ===== DONE DATASET: $d ====="
done

echo "[$(ts)] ALL DONE. Figures under reports/figures, tables under reports/tables."
