# Final Fair-Run Results

## Training Details

All reported DINO multiscale models were evaluated under the same experimental protocol. For each dataset, the four variants (Baseline, Gated, Cross-Gated, and Parallel) used the same image size (224), batch size (4), number of epochs (50), learning rate (1e-4), weight decay (1e-2), random seed (42), and train/evaluation splits. The training pipeline also used the same augmentations for all variants: resize to 224x224, random horizontal flip, random vertical flip with probability 0.1, color jitter (0.1, 0.1, 0.1, 0.05), tensor conversion, and ImageNet normalization. Model selection was performed with the validation macro-F1 score, and final test metrics were computed from the corresponding `best.ckpt` checkpoint.

For corn_maize_leaf_disease, the gated result was taken from `runs_dino_multiscale_gated/...`, whose checkpoint metadata matches the fair protocol above (224 image size, batch size 4, 50 epochs, learning rate 1e-4, weight decay 1e-2, seed 42, and best-checkpoint evaluation).

## Final Comparison Tables

### Bean Disease (Uganda)

| Model | Num parameters | Accuracy | F1_macro | F1_weighted |
|---|---:|---:|---:|---:|
| Baseline | 22.67M | 0.792 | 0.794 | 0.793 |
| **Gated** | 33.30M | 0.873 | 0.873 | 0.873 |
| Cross-Gated | 23.86M | 0.807 | 0.805 | 0.804 |
| Parallel | 34.48M | 0.746 | 0.747 | 0.747 |

### Blackgram Leaf Disease

| Model | Num parameters | Accuracy | F1_macro | F1_weighted |
|---|---:|---:|---:|---:|
| Baseline | 22.68M | 0.224 | 0.073 | 0.082 |
| Gated | 33.30M | 0.404 | 0.397 | 0.399 |
| Cross-Gated | 23.86M | 0.462 | 0.446 | 0.455 |
| **Parallel** | 34.48M | 0.737 | 0.732 | 0.736 |

### Corn/Maize Leaf Disease

| Model | Num parameters | Accuracy | F1_macro | F1_weighted |
|---|---:|---:|---:|---:|
| **Baseline** | 22.67M | 0.921 | 0.904 | 0.920 |
| Gated | 33.30M | 0.915 | 0.889 | 0.915 |
| Cross-Gated | 23.86M | 0.910 | 0.886 | 0.910 |
| Parallel | 34.48M | 0.918 | 0.895 | 0.919 |

## Results

Under the controlled fair-training setting, the best model differed across datasets. On Bean Disease (Uganda), the Gated variant achieved the strongest test performance with an accuracy of 0.873 and a macro-F1 of 0.873, outperforming Baseline, Cross-Gated, and Parallel. On Blackgram Leaf Disease, the Parallel variant was clearly superior, reaching 0.737 accuracy and 0.732 macro-F1; this margin was substantial relative to the other variants and indicates that the more expressive fusion strategy was beneficial for this harder five-class dataset. In contrast, on Corn/Maize Leaf Disease, the Baseline model remained best, with 0.921 accuracy and 0.904 macro-F1, while the added fusion mechanisms produced similar but slightly lower macro-F1 values.

A broader pattern also emerges from the comparison. The gains from architectural complexity were dataset-dependent rather than uniform. Gated fusion was most effective on the three-class Bean dataset, Parallel fusion was most effective on the more heterogeneous Blackgram dataset, and the simpler Baseline model was already sufficient for Corn/Maize. These results suggest that adding multiscale interaction capacity can improve performance, but the benefit depends on the difficulty of the dataset and on whether the added fusion mechanism matches the structure of the visual decision boundaries.

## Discussion

The differences between Parallel, Gated, and Cross-Gated can be explained by how each variant controls information flow across scales. The Parallel model has the largest capacity, because it preserves multiple branches and lets the classifier combine richer multiscale descriptors. This can be advantageous when disease symptoms appear at different spatial extents or when inter-class boundaries are visually complex, which is consistent with its strong performance on Blackgram. However, the same extra capacity can be unnecessary on easier datasets, where it may add optimization difficulty without producing a better representation, as observed on Corn/Maize and Bean.

The Gated model introduces an adaptive weighting mechanism that can suppress less useful scale features while preserving the most informative ones. This design appears to offer a good balance between flexibility and control, which likely explains why it performed best on Bean Disease (Uganda). In that dataset, the model may benefit from selectively emphasizing disease cues without carrying the full parameter burden of the Parallel variant. The result suggests that adaptive selection is helpful when the task is not extremely complex, but still requires finer control than a simple baseline can provide.

The Cross-Gated model is more selective still, because the gating of one stream depends on information from another stream. In principle, that interaction can improve feature coordination, but it can also make learning more sensitive to optimization and to the quality of the cross-scale signals. If one branch becomes noisy or less reliable early in training, cross-gating may suppress useful evidence instead of reinforcing it. This likely explains why Cross-Gated remained competitive but did not become the top model on any of the three finalized fair datasets.

Taken together, the final results indicate that there is no single universally best fusion strategy. Parallel fusion seems strongest when high representational capacity is needed, Gated fusion seems strongest when selective emphasis matters more than maximum capacity, and Cross-Gated may require better tuning or more data to consistently realize its theoretical advantage. For the present study, this dataset-specific behavior is itself an important finding, because it shows that the value of added fusion complexity should be judged relative to dataset difficulty rather than assumed to generalize uniformly across plant disease benchmarks.
