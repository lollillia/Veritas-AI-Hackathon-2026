# Model performance: `models/fusion_v3_image`

**Status: experiment** — part of the image-only vs image + weather comparison (`docs/weather-input-comparison.md`). Saved as `model.pt` by `train/fusion.py`; the app cannot load it.

## Model
- MobileNetV2 + no weather input (image only), 5 epochs, batch 16, AdamW 5e-5, seed 0
- Labels: merged (`common_rust`, `healthy`, `leaf_blight`, `leaf_spot`)
- Data: `data/splits_v3.csv`
- Command: `python -m train.fusion --manifest data/splits_v3.csv --out models/fusion_v3_image --epochs 5`

## Training
| Epoch | Dev accuracy |
|---|---|
| 1 | 90.7% |
| 2 | 90.5% |
| 3 | 91.6% |
| 4 | 92.0% |
| 5 | 91.9% |

## Results (test)

| Subset | Images | Accuracy | Macro F1 |
|---|---|---|---|
| all | 1210 | **91.8%** | 92.2% |
| photos with weather | 433 | 95.4% | 95.8% |
| photos without weather | 777 | 89.8% | 90.0% |

### Test (1210 images)

- **Accuracy:** 91.8%
- **Macro F1:** 92.2%

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| common_rust | 95.1% | 91.2% | 93.1% | 339 |
| healthy | 99.6% | 99.6% | 99.6% | 224 |
| leaf_blight | 91.5% | 89.8% | 90.6% | 383 |
| leaf_spot | 82.5% | 89.0% | 85.6% | 264 |

Confusion matrix (rows = true, columns = predicted)

| true \ predicted | common_rust | healthy | leaf_blight | leaf_spot |
|---|---|---|---|---|
| **common_rust** | 309 | 1 | 11 | 18 |
| **healthy** | 1 | 223 | 0 | 0 |
| **leaf_blight** | 7 | 0 | 344 | 32 |
| **leaf_spot** | 8 | 0 | 21 | 235 |

| Source | Images | Correct | Accuracy |
|---|---|---|---|
| cds | 157 | 147 | 93.6% |
| kaggle | 417 | 388 | 93.0% |
| maize_multiformat | 72 | 48 | 66.7% |
| ndisan | 274 | 264 | 96.4% |
| rlnst | 290 | 264 | 91.0% |

## Limitations
- One run per model; the weather comparison found no reliable evidence that weather input helps (see the comparison doc).
- Capture days are shared between train and test, so weather can act as a day fingerprint.
