# Model performance: `models/corn_v2`

**Status: superseded.** Merged labels (`common_rust`, `healthy`, `leaf_blight`, `leaf_spot`), not accepted by the app.

## Model
- MobileNetV2, 5 epochs, batch 16, BatchNorm momentum fix
- Data: `data/splits_v2.csv` — v1 plus `maize_multiformat`; 7,439 train / 932 dev / 936 test
- Command: `python -m train.train --manifest data/splits_v2.csv --out models/corn_v2 --epochs 5`

## Training
| Epoch | Dev accuracy |
|---|---|
| 1 | 90.0% |
| 2 | 90.8% |
| 3 | 91.2% |
| 4 | 91.5% |
| 5 | 91.3% |

## Results

| Evaluation | Images | Accuracy | Macro F1 |
|---|---|---|---|
| v2 test | 936 | 92.5% | 93.4% |
| `maize_multiformat` test photos only | 72 | 73.6% | |

Compared with `models/corn` on the same v2 test: 92.5% vs 91.5% overall, and 73.6% vs 55.6% on the 72 `maize_multiformat` test photos — training on that dataset's photos helped most where the older model failed; healthy precision rose from 86.7% to 99.3%.

### v2 test (936 images)

- **Accuracy:** 92.5%
- **Macro F1:** 93.4%

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| common_rust | 97.8% | 91.7% | 94.6% | 241 |
| healthy | 99.3% | 100.0% | 99.7% | 145 |
| leaf_blight | 88.7% | 93.0% | 90.8% | 286 |
| leaf_spot | 88.6% | 88.6% | 88.6% | 264 |

Confusion matrix (rows = true, columns = predicted)

| true \ predicted | common_rust | healthy | leaf_blight | leaf_spot |
|---|---|---|---|---|
| **common_rust** | 221 | 0 | 6 | 14 |
| **healthy** | 0 | 145 | 0 | 0 |
| **leaf_blight** | 3 | 1 | 266 | 16 |
| **leaf_spot** | 2 | 0 | 28 | 234 |

| Source | Images | Correct | Accuracy |
|---|---|---|---|
| cds | 157 | 149 | 94.9% |
| kaggle | 417 | 396 | 95.0% |
| maize_multiformat | 72 | 53 | 73.6% |
| rlnst | 290 | 268 | 92.4% |

## Limitations
- Only 72 `maize_multiformat` test photos: that figure could easily be off by about 10 points.
- Single training run (seed 0); differences of a point or two between models are within run-to-run noise.
- Test sets come from the same sources as training. Field photos from unseen farms are harder (see `models/PLANTVILLAGE_BACKGROUND_ISSUE.md`).
- Near-duplicate grouping cannot rule out every copy of the same leaf across splits.
