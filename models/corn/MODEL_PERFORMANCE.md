# Model performance: `models/corn`

**Status: superseded.** Uses the old merged labels (`common_rust`, `healthy`, `leaf_blight`, `leaf_spot`), which the app no longer accepts.

## Model
- MobileNetV2 (`google/mobilenet_v2_1.0_224`), 5 epochs, batch 16, BatchNorm momentum fix
- Data: `data/splits.csv` (v1) — sources `kaggle`, `rlnst`, `cds`; 6,914 train / 865 dev / 864 test
- Command: `python -m train.train --manifest data/splits.csv --out models/corn --epochs 5`

## Training
| Epoch | Dev accuracy |
|---|---|
| 1 | 91.9% |
| 2 | 93.1% |
| 3 | 93.4% |
| 4 | 94.8% |
| 5 | 94.3% |

## Results

| Evaluation | Images | Accuracy | Macro F1 |
|---|---|---|---|
| v1 test | 864 | 94.4% | 95.1% |
| v2 test (v1 test + 72 unseen `maize_multiformat`) | 936 | 91.5% | 91.7% |
| **Unseen source**: all `maize_multiformat` photos | 664 | **37.8%** | 33.0% |

### v1 test (864 images)

- **Accuracy:** 94.4%
- **Macro F1:** 95.1%

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| common_rust | 98.2% | 97.4% | 97.8% | 227 |
| healthy | 99.3% | 100.0% | 99.6% | 135 |
| leaf_blight | 93.9% | 89.2% | 91.5% | 260 |
| leaf_spot | 89.1% | 94.2% | 91.6% | 242 |

Confusion matrix (rows = true, columns = predicted)

| true \ predicted | common_rust | healthy | leaf_blight | leaf_spot |
|---|---|---|---|---|
| **common_rust** | 221 | 0 | 3 | 3 |
| **healthy** | 0 | 135 | 0 | 0 |
| **leaf_blight** | 2 | 1 | 232 | 25 |
| **leaf_spot** | 2 | 0 | 12 | 228 |

| Source | Images | Correct | Accuracy |
|---|---|---|---|
| cds | 157 | 153 | 97.5% |
| kaggle | 417 | 393 | 94.2% |
| rlnst | 290 | 270 | 93.1% |

### Unseen source: maize_multiformat (all images)

- **Accuracy:** 37.8%
- **Macro F1:** 33.0%

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| common_rust | 86.5% | 23.7% | 37.2% | 135 |
| healthy | 28.9% | 99.1% | 44.7% | 113 |
| leaf_blight | 51.0% | 49.5% | 50.2% | 216 |
| leaf_spot | 0.0% | 0.0% | 0.0% | 200 |

Confusion matrix (rows = true, columns = predicted)

| true \ predicted | common_rust | healthy | leaf_blight | leaf_spot |
|---|---|---|---|---|
| **common_rust** | 32 | 64 | 15 | 24 |
| **healthy** | 0 | 112 | 1 | 0 |
| **leaf_blight** | 0 | 104 | 107 | 5 |
| **leaf_spot** | 5 | 108 | 87 | 0 |

| Source | Images | Correct | Accuracy |
|---|---|---|---|
| maize_multiformat | 664 | 251 | 37.8% |

## Limitations
- On a dataset it never saw it labelled about half the diseased leaves "healthy" and never predicted leaf spot.
- Single training run (seed 0); differences of a point or two between models are within run-to-run noise.
- Test sets come from the same sources as training. Field photos from unseen farms are harder (see `models/PLANTVILLAGE_BACKGROUND_ISSUE.md`).
- Near-duplicate grouping cannot rule out every copy of the same leaf across splits.
