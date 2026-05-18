# Plant Disease Classification with DINOv3 and Multiscale Attention

This repository contains a set of plant disease classification experiments built around a DINOv3-based multiscale feature extractor and several attention-driven fusion strategies. The project focuses on comparing multiple architectural variants across plant leaf disease datasets using Macro-F1 as the main selection metric.

## Overview

The codebase includes five model solutions:

1. `Baseline CBAM`
   A DINO multiscale classifier with the base CBAM-style fusion pipeline.
2. `Gated CBAM`
   Adds learnable gating to adaptively weight multiscale features before fusion.
3. `Cross-Gated CBAM`
   Uses cross-stream gating so one scale can modulate another during fusion.
4. `Parallel`
   Preserves parallel multiscale branches and combines richer descriptors at the classifier head.
5. `CoAG (Solution 5)`
   The fifth fusion solution implemented through the cross-validation pipeline in `src/train_variant3_cv.py`.

## Datasets Evaluated

The README summarizes the five datasets for which results for all five solutions are available in the repository:

- `bean_disease_uganda`
- `blackgram_leaf_disease`
- `corn_maize_leaf_disease`
- `coconut_tree_disease`
- `sunflower_disease`

Additional dataset configurations also exist in [configs/datasets.yaml](/home/itartoussi/PlantDiseaseClassification/configs/datasets.yaml:1).

## Repository Layout

- [src](/home/itartoussi/PlantDiseaseClassification/src): training, evaluation, reporting, and CV scripts
- [models](/home/itartoussi/PlantDiseaseClassification/models): DINO multiscale classifier definitions and fusion modules
- [configs](/home/itartoussi/PlantDiseaseClassification/configs): dataset configuration files
- [reports](/home/itartoussi/PlantDiseaseClassification/reports): generated figures, tables, and reports

## Installation

Python 3.10+ is recommended.

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
```

Install PyTorch first using the command appropriate for your CUDA or CPU environment from the official PyTorch site, then install the common Python dependencies used by this project:

```bash
pip install numpy scikit-learn pyyaml pandas matplotlib seaborn pillow tqdm
pip install torchvision
```

This project also expects an external DINOv3 codebase under `third_party/dinov3_repo`, which is intentionally not included in version control. After placing that dependency locally, expose it with:

```bash
source env.sh
```

## Expected Data Layout

Datasets are not committed to the repository. The training scripts expect a structure like:

```text
data/
  <dataset_name>/
    train/
    val/
    test/
```

For the CoAG cross-validation script, the training split is expected under `data/<dataset_name>/train/`, with `classes.txt` alongside the dataset.

## Training Commands

Examples below use `bean_disease_uganda`; replace the dataset name as needed.

### 1. Baseline CBAM

```bash
source env.sh
python src/train_dino_multiscale-baseline-cbam.py \
  --dataset bean_disease_uganda \
  --data_root data \
  --img 224 \
  --bs 4 \
  --epochs 50 \
  --lr 1e-4 \
  --wd 1e-2 \
  --out runs_dino_multiscale
```

### 2. Gated CBAM

```bash
source env.sh
python src/train_dino_multiscale_gated.py \
  --dataset bean_disease_uganda \
  --data_root data \
  --img 224 \
  --bs 4 \
  --epochs 50 \
  --lr 1e-4 \
  --wd 1e-2 \
  --out runs_dino_multiscale_gated
```

### 3. Cross-Gated CBAM

```bash
source env.sh
python src/train_dino_multiscale_cross_gated.py \
  --dataset bean_disease_uganda \
  --data_root data \
  --img 224 \
  --bs 4 \
  --epochs 50 \
  --lr 1e-4 \
  --wd 1e-2 \
  --out runs_dino_multiscale_cross_gated
```

### 4. Parallel

```bash
source env.sh
python src/train_dino_multiscale_parallel.py \
  --dataset bean_disease_uganda \
  --data_root data \
  --img 224 \
  --bs 4 \
  --epochs 50 \
  --lr 1e-4 \
  --wd 1e-2 \
  --out runs_dino_multiscale_parallel
```

### 5. CoAG (Solution 5)

Single-fold example:

```bash
source env.sh
python src/train_variant3_cv.py \
  --dataset bean_disease_uganda \
  --data_root data \
  --kfold 5 \
  --fold 0 \
  --img 224 \
  --bs 32 \
  --epochs 100 \
  --lr 3e-4 \
  --wd 1e-2 \
  --out runs_variant3_cv5
```

Run all folds:

```bash
for fold in 0 1 2 3 4; do
  python src/train_variant3_cv.py --dataset bean_disease_uganda --fold "$fold"
done
```

## Results

The table below reports Macro-F1 for the five solutions across the five datasets summarized in this repository.

| Dataset | Baseline CBAM | Gated CBAM | Cross-Gated CBAM | Parallel | CoAG |
|---|---:|---:|---:|---:|---:|
| Bean Disease (Uganda) | 0.794 | 0.873 | 0.805 | 0.747 | 0.167 |
| Blackgram Leaf Disease | 0.073 | 0.397 | 0.446 | 0.732 | 0.439 |
| Corn/Maize Leaf Disease | 0.904 | 0.889 | 0.886 | 0.895 | 0.275 |
| Coconut Tree Disease | 0.978 | 0.994 | 0.974 | 0.994 | 0.033 |
| Sunflower Disease | 0.823 | 0.862 | 0.826 | 0.852 | 0.631 |

## Notes on Result Sources

- `Baseline CBAM`, `Gated CBAM`, `Cross-Gated CBAM`, and `Parallel` values above come from saved `test_metrics.json` files for each variant.
- `CoAG` values come from the mean validation Macro-F1 reported in the corresponding 5-fold `cv_summary.json` files under `runs_variant3_cv5/`.
- Because CoAG is summarized from cross-validation while the other four values are held-out test metrics, the comparison is useful for repository documentation but should not be interpreted as a perfectly matched benchmarking protocol.

## Important Notes

- DINOv3 backbone weights are not included.
- Datasets are not included.
- External repositories under `third_party/` are not included.
- Training outputs, checkpoints, and experiment artifacts are excluded from version control.

## License / Usage

If you plan to release this repository publicly, add a project license file before publishing.
