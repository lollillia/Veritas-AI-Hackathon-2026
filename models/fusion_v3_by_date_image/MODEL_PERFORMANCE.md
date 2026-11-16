# Model performance: `models/fusion_v3_by_date_image`

**Status: experiment** — part of the image-only vs image + weather comparison (`docs/weather-input-comparison.md`). Saved as `model.pt` by `train/fusion.py`; the app cannot load it.

## Model
- MobileNetV2 + no weather input (image only), 5 epochs, batch 16, AdamW 5e-5, seed 0
- Labels: merged (`common_rust`, `healthy`, `leaf_blight`, `leaf_spot`)
- Data: `data/splits_v3_by_date.csv` — every (city, capture date) kept in one set
- Command: `python -m train.fusion --manifest data/splits_v3_by_date.csv --out models/fusion_v3_by_date_image --epochs 5`

## Training
| Epoch | Dev accuracy |
|---|---|
| 1 | 90.0% |
| 2 | 91.1% |
| 3 | 91.3% |
| 4 | 92.0% |
| 5 | 91.4% |

## Results (test)

| Subset | Images | Accuracy | Macro F1 |
|---|---|---|---|
| all | 1149 | **90.9%** | 91.8% |
| photos with weather | 372 | 88.2% | 63.8% |
| photos without weather | 777 | 92.3% | 92.4% |

### Test (1149 images)

- **Accuracy:** 90.9%
- **Macro F1:** 91.8%

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| common_rust | 95.5% | 92.2% | 93.8% | 387 |
| healthy | 99.3% | 99.3% | 99.3% | 145 |
| leaf_blight | 81.0% | 91.9% | 86.1% | 236 |
| leaf_spot | 90.3% | 85.8% | 88.0% | 381 |

Confusion matrix (rows = true, columns = predicted)

| true \ predicted | common_rust | healthy | leaf_blight | leaf_spot |
|---|---|---|---|---|
| **common_rust** | 357 | 1 | 10 | 19 |
| **healthy** | 0 | 144 | 1 | 0 |
| **leaf_blight** | 3 | 0 | 217 | 16 |
| **leaf_spot** | 14 | 0 | 40 | 327 |

| Source | Images | Correct | Accuracy |
|---|---|---|---|
| cds | 224 | 188 | 83.9% |
| kaggle | 417 | 391 | 93.8% |
| maize_multiformat | 72 | 58 | 80.6% |
| ndisan | 146 | 139 | 95.2% |
| rlnst | 290 | 269 | 92.8% |

## Limitations
- One run per model; the weather comparison found no reliable evidence that weather input helps (see the comparison doc).
- Photos with weather in this test set are only two capture days, each a single class.
