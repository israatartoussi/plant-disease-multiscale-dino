# Plant Disease Classification with DINOv3 Multiscale Solutions

![Python](https://img.shields.io/badge/python-3.10-blue)

This repository contains a focused research codebase for plant disease classification using DINOv3 multiscale representations and five lightweight fusion/classification variants. The project studies how different multiscale interaction strategies affect Macro-F1 across several plant disease benchmarks.

## Models

- `Baseline CBAM`
  A DINOv3 multiscale baseline with CBAM-based fusion.
- `Gated CBAM`
  Adds learnable gated refinement before CBAM fusion.
- `Cross-Gated CBAM`
  Uses cross-scale gating to modulate one scale with context from others.
- `Parallel`
  Combines gated and cross-gated pathways in a higher-capacity parallel design.
- `CoAG (Ours)`
  CoAG applies mutual Co-Attention Gating between consecutive scale pairs (`S1↔S2`, `S2↔S3`, `S3↔S4`), inspired by the CoAG block in MambaCAFU (Bui et al., 2025).

## Results

Macro-F1 across the five evaluated datasets is summarized below.

| Model | Active Params | Bean | Blackgram | Corn | Sunflower | Coconut |
|---|---:|---:|---:|---:|---:|---:|
| Baseline CBAM | 1.08 M | 0.813 | 0.073 | 0.879 | 0.823 | 0.978 |
| Gated CBAM | 11.70 M | 0.899 | 0.717 | 0.889 | 0.862 | 0.994 |
| Cross-Gated CBAM | 2.27 M | 0.847 | 0.446 | 0.898 | 0.826 | 0.974 |
| Parallel | 12.89 M | 0.751 | 0.732 | 0.894 | 0.852 | 0.994 |
| CoAG (Ours) | 3.08 M | 0.804 | 0.673 | — | 0.916 | 0.989 |

`†` Blackgram results for `Gated CBAM` and `CoAG` use `lr=1e-5`. Corn `CoAG` result is pending.

## Repository Layout

- `models/__init__.py`
- `models/dino_multiscale_classifier.py`
- `models/dino_multiscale_gated_classifier.py`
- `models/dino_multiscale_cross_gated_classifier.py`
- `models/dino_multiscale_parallel_classifier.py`
- `models/dino_multiscale_coag_classifier.py`
- `src/train_dino_multiscale-baseline-cbam.py`
- `src/train_dino_multiscale_gated.py`
- `src/train_dino_multiscale_cross_gated.py`
- `src/train_dino_multiscale_parallel.py`
- `src/train_dino_multiscale_coag.py`
- `src/data.py`

## Requirements

Python `3.10` is recommended.

```bash
pip install torch torchvision numpy scikit-learn pillow
```

## External Dependency

The training scripts rely on an external DINOv3 implementation that is not included in this repository. In particular, the model files expect `dinounet.dinov3.models.vision_transformer.vit_small` to be available in the Python path.

## Expected Dataset Layout

```text
data/
  <dataset_name>/
    train/
    val/
    test/
```

## Example Training Commands

Baseline:

```bash
python src/train_dino_multiscale-baseline-cbam.py --dataset bean_disease_uganda --data_root data
```

Gated:

```bash
python src/train_dino_multiscale_gated.py --dataset bean_disease_uganda --data_root data
```

Cross-Gated:

```bash
python src/train_dino_multiscale_cross_gated.py --dataset bean_disease_uganda --data_root data
```

Parallel:

```bash
python src/train_dino_multiscale_parallel.py --dataset bean_disease_uganda --data_root data
```

CoAG:

```bash
python src/train_dino_multiscale_coag.py --dataset bean_disease_uganda --data_root data
```

## Notes

- Datasets are not included.
- DINOv3 backbone weights are not included.
- Training outputs, checkpoints, and large artifacts are excluded by `.gitignore`.
