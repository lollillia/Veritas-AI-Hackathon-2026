# Model performance: `models/fusion_v3_weather`

**Status: experiment** — part of the image-only vs image + weather comparison (`docs/weather-input-comparison.md`). Saved as `model.pt` by `train/fusion.py`; the app cannot load it.

## Model
- MobileNetV2 + weather branch (24 weather summaries + has-weather flag), 5 epochs, batch 16, AdamW 5e-5, seed 0
- Labels: merged (`common_rust`, `healthy`, `leaf_blight`, `leaf_spot`)
- Data: `data/splits_v3.csv`
- Command: `python -m train.fusion --manifest data/splits_v3.csv --out models/fusion_v3_weather --epochs 5 --with-weather`

## Training
| Epoch | Dev accuracy |
|---|---|
| 1 | 91.0% |
| 2 | 91.0% |
| 3 | 91.8% |
| 4 | 93.1% |
| 5 | 93.1% |

## Results (test)

| Subset | Images | Accuracy | Macro F1 |
|---|---|---|---|
| all | 1210 | **93.7%** | 94.1% |
| photos with weather | 433 | 96.1% | 96.3% |
| photos without weather | 777 | 92.4% | 92.7% |

### Test (1210 images)

- **Accuracy:** 93.7%
- **Macro F1:** 94.1%

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| common_rust | 97.5% | 91.4% | 94.4% | 339 |
| healthy | 99.6% | 99.6% | 99.6% | 224 |
| leaf_blight | 94.3% | 91.1% | 92.7% | 383 |
| leaf_spot | 84.6% | 95.5% | 89.7% | 264 |

Confusion matrix (rows = true, columns = predicted)

| true \ predicted | common_rust | healthy | leaf_blight | leaf_spot |
|---|---|---|---|---|
| **common_rust** | 310 | 1 | 12 | 16 |
| **healthy** | 1 | 223 | 0 | 0 |
| **leaf_blight** | 4 | 0 | 349 | 30 |
| **leaf_spot** | 3 | 0 | 9 | 252 |

| Source | Images | Correct | Accuracy |
|---|---|---|---|
| cds | 157 | 149 | 94.9% |
| kaggle | 417 | 392 | 94.0% |
| maize_multiformat | 72 | 63 | 87.5% |
| ndisan | 274 | 265 | 96.7% |
| rlnst | 290 | 265 | 91.4% |

## Limitations
- One run per model; the weather comparison found no reliable evidence that weather input helps (see the comparison doc).
- Capture days are shared between train and test, so weather can act as a day fingerprint.
