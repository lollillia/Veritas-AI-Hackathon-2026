# Model performance: `models/corn_fungal_v6`

**Status: baseline for v6.** It is the first model with all 11 labels (the app's current label set) and scores about the same as `corn_fungal_v5_bgaug` on v5's test photos while also recognising the four new diseases. Not yet used by the app (`config.yaml` unchanged).

## Model
- MobileNetV2 (`google/mobilenet_v2_1.0_224`), 5 epochs, batch 16, BatchNorm momentum fix, seed 0, no background augmentation
- Labels (11): `brown_spot`, `common_rust`, `curvularia_leaf_spot`, `eyespot`, `gray_leaf_spot`, `healthy`, `northern_leaf_blight`, `northern_leaf_spot`, `phaeosphaeria_leaf_spot`, `southern_leaf_blight`, `southern_rust`
- Data: `data/splits_v6.csv` — v5 plus `china_2`, `xtu502` and 7 `up_field` southern rust photos; 13,894 train / 1,748 dev / 1,719 test
- Command: `python -m train.train --manifest data/splits_v6.csv --out models/corn_fungal_v6 --epochs 5` (via `models/run_v6.sh`)

## Training
| Epoch | Dev accuracy |
|---|---|
| 1 | 88.3% |
| 2 | 90.5% |
| 3 | 90.9% |
| 4 | 91.3% |
| 5 | 91.3% |

## Results

| Evaluation | Images | Accuracy | Macro F1 |
|---|---|---|---|
| dev | 1,748 | 91.3% | |
| **v6 test** | 1,719 | **91.6%** | 83.7% |
| v5 test photos (same set the v5 models were scored on) | 1,632 | 91.9% | 85.2% |

### Compared with the v5 models on v5's test photos

| | `corn_fungal_v5` | `corn_fungal_v5_bgaug` | **`corn_fungal_v6`** | `corn_fungal_v6_bgaug` |
|---|---|---|---|---|
| All (1,632) | 90.6% | 92.1% | **91.9%** | 91.3% |
| Field-photo sources (651) | 87.6% | 89.6% | 89.4% | 88.8% |
| Lab-style sources (981) | 92.6% | 93.8% | 93.6% | 93.0% |
| `up_field` (124) | 73.4% | 75.8% | 75.0% | 75.0% |
| `maize_multiformat` (72) | 56.9% | 69.4% | 69.4% | 59.7% |

The v6 models choose among 11 labels instead of 7, so a v5 photo can now also be mistaken for one of the new diseases. Even so, v6 is 1.3 points above the v5 baseline.

### v6 test (1,719 images)

- **Accuracy:** 91.6%
- **Macro F1:** 83.7%

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| brown_spot | 83.3% | 90.9% | 87.0% | 11 |
| common_rust | 93.4% | 94.6% | 94.0% | 404 |
| curvularia_leaf_spot | 77.8% | 70.0% | 73.7% | 10 |
| eyespot | 100.0% | 60.0% | 75.0% | 40 |
| gray_leaf_spot | 85.8% | 90.6% | 88.1% | 352 |
| healthy | 94.5% | 97.3% | 95.9% | 369 |
| northern_leaf_blight | 93.4% | 90.3% | 91.8% | 424 |
| northern_leaf_spot | 96.2% | 96.2% | 96.2% | 52 |
| phaeosphaeria_leaf_spot | 63.6% | 43.8% | 51.9% | 16 |
| southern_leaf_blight | 83.3% | 75.0% | 78.9% | 20 |
| southern_rust | 86.4% | 90.5% | 88.4% | 21 |

Confusion matrix (rows = true, columns = predicted)

| true \ predicted | brown_spot | common_rust | curvularia_leaf_spot | eyespot | gray_leaf_spot | healthy | northern_leaf_blight | northern_leaf_spot | phaeosphaeria_leaf_spot | southern_leaf_blight | southern_rust |
|---|---|---|---|---|---|---|---|---|---|---|---|
| **brown_spot** | 10 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 |
| **common_rust** | 1 | 382 | 1 | 0 | 5 | 4 | 8 | 0 | 2 | 0 | 1 |
| **curvularia_leaf_spot** | 0 | 1 | 7 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 2 |
| **eyespot** | 0 | 13 | 0 | 24 | 1 | 1 | 0 | 1 | 0 | 0 | 0 |
| **gray_leaf_spot** | 1 | 3 | 0 | 0 | 319 | 10 | 17 | 0 | 0 | 2 | 0 |
| **healthy** | 0 | 2 | 0 | 0 | 5 | 359 | 1 | 0 | 2 | 0 | 0 |
| **northern_leaf_blight** | 0 | 6 | 0 | 0 | 35 | 0 | 383 | 0 | 0 | 0 | 0 |
| **northern_leaf_spot** | 0 | 0 | 1 | 0 | 0 | 1 | 0 | 50 | 0 | 0 | 0 |
| **phaeosphaeria_leaf_spot** | 0 | 2 | 0 | 0 | 5 | 2 | 0 | 0 | 7 | 0 | 0 |
| **southern_leaf_blight** | 0 | 0 | 0 | 0 | 2 | 3 | 0 | 0 | 0 | 15 | 0 |
| **southern_rust** | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 1 | 19 |

| Source | Images | Correct | Accuracy |
|---|---|---|---|
| cds | 157 | 150 | 95.5% |
| china_2 | 82 | 73 | 89.0% |
| kaggle | 416 | 391 | 94.0% |
| maize_multiformat | 72 | 50 | 69.4% |
| mendeley_bd | 213 | 207 | 97.2% |
| ndisan | 274 | 264 | 96.4% |
| neau | 85 | 82 | 96.5% |
| rlnst | 290 | 263 | 90.7% |
| up_field | 125 | 93 | 74.4% |
| xtu502 | 5 | 2 | 40.0% |


### New classes

| Class | Train photos | Test photos | F1 |
|---|---|---|---|
| southern_rust | 172 | 21 | 88.4% |
| southern_leaf_blight | 160 | 20 | 78.9% |
| brown_spot | 92 | 11 | 87.0% |
| curvularia_leaf_spot | 84 | 10 | 73.7% |

Southern and common rust are rarely confused (1 common rust photo predicted southern rust, none the other way). Most new-class test photos come from `china_2`, so these scores partly measure "recognises `china_2`", and with 10–21 test photos each they could be off by 10–20 points.

## Limitations
- Single training run (seed 0). Differences of 1–2 points between models are within run-to-run noise, and `maize_multiformat` (72 photos) has swung by 10+ points between runs.
- The four new classes and Phaeosphaeria (16 test photos, F1 51.9%) are small and mostly from one dataset each.
- Eyespot recall is low (60%): 13 of 40 eyespot photos are predicted as common rust.
- Test sets come from the same sources as training; unseen farms are harder (see `train/issues/PLANTVILLAGE_BACKGROUND_ISSUE.md`).
