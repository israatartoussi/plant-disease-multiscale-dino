# Plant Disease Classification with DINOv3 Multiscale Solutions

This repository contains a minimal version of the project focused only on the five DINOv3 multiscale classification solutions used for plant disease recognition.

## Included Models

- `Baseline CBAM`
- `Gated CBAM`
- `Cross-Gated CBAM`
- `Parallel`
- `CoAG`

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

Python 3.10+ is recommended.

Install the core dependencies you need for training:

```bash
pip install torch torchvision numpy scikit-learn pillow
```

## External Dependency

The training code depends on an external DINOv3 implementation that is not included in this repository. The model files expect `dinounet.dinov3.models.vision_transformer.vit_small` to be available in your Python path.

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
