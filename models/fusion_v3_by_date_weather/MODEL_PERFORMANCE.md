# Model performance: `models/fusion_v3_by_date_weather`

**Status: experiment** — part of the image-only vs image + weather comparison (`docs/weather-input-comparison.md`). Saved as `model.pt` by `train/fusion.py`; the app cannot load it.

## Model
- MobileNetV2 + weather branch (24 weather summaries + has-weather flag), 5 epochs, batch 16, AdamW 5e-5, seed 0
- Labels: merged (`common_rust`, `healthy`, `leaf_blight`, `leaf_spot`)
- Data: `data/splits_v3_by_date.csv` — every (city, capture date) kept in one set
- Command: `python -m train.fusion --manifest data/splits_v3_by_date.csv --out models/fusion_v3_by_date_weather --epochs 5 --with-weather`

## Training
| Epoch | Dev accuracy |
|---|---|
| 1 | 89.8% |
| 2 | 90.4% |
| 3 | 92.0% |
| 4 | 91.9% |
| 5 | 91.7% |

## Results (test)

| Subset | Images | Accuracy | Macro F1 |
|---|---|---|---|
| all | 1149 | **91.8%** | 92.6% |
| photos with weather | 372 | 91.9% | 66.2% |
| photos without weather | 777 | 91.8% | 91.9% |

### Test (1149 images)

- **Accuracy:** 91.8%
- **Macro F1:** 92.6%

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| common_rust | 97.8% | 91.7% | 94.7% | 387 |
| healthy | 99.3% | 100.0% | 99.7% | 145 |
| leaf_blight | 82.4% | 91.1% | 86.5% | 236 |
| leaf_spot | 89.7% | 89.2% | 89.5% | 381 |

Confusion matrix (rows = true, columns = predicted)

| true \ predicted | common_rust | healthy | leaf_blight | leaf_spot |
|---|---|---|---|---|
| **common_rust** | 355 | 1 | 12 | 19 |
| **healthy** | 0 | 145 | 0 | 0 |
| **leaf_blight** | 1 | 0 | 215 | 20 |
| **leaf_spot** | 7 | 0 | 34 | 340 |

| Source | Images | Correct | Accuracy |
|---|---|---|---|
| cds | 224 | 201 | 89.7% |
| kaggle | 417 | 388 | 93.0% |
| maize_multiformat | 72 | 58 | 80.6% |
| ndisan | 146 | 140 | 95.9% |
| rlnst | 290 | 268 | 92.4% |

## Limitations
- One run per model; the weather comparison found no reliable evidence that weather input helps (see the comparison doc).
- Photos with weather in this test set are only two capture days, each a single class.
