# Model performance: `models/corn_fungal`

**Status: superseded by `corn_fungal_v5`** (no Phaeosphaeria label). Its labels are a subset of the app's, so the app could load it.

## Model
- MobileNetV2 (`google/mobilenet_v2_1.0_224`), 5 epochs, batch 16, BatchNorm momentum fix
- Labels: `common_rust`, `eyespot`, `gray_leaf_spot`, `healthy`, `northern_leaf_blight`, `northern_leaf_spot`
- Data: `data/splits_v4.csv` — v3 plus `mendeley_bd` and `neau`; 12,181 train / 1,528 dev / 1,508 test
- Command: `python -m train.train --manifest data/splits_v4.csv --out models/corn_fungal --epochs 5`

## Training
| Epoch | Dev accuracy |
|---|---|
| 1 | 91.2% |
| 2 | 92.4% |
| 3 | 92.9% |
| 4 | 93.8% |
| 5 | 93.7% |

## Results

| Split | Images | Accuracy | Macro F1 |
|---|---|---|---|
| dev | 1528 | 93.7% | 94.3% |
| **test** | 1508 | **93.2%** | 94.3% |

### Test (1508 images)

- **Accuracy:** 93.2%
- **Macro F1:** 94.3%

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| common_rust | 96.4% | 93.3% | 94.8% | 372 |
| eyespot | 100.0% | 94.7% | 97.3% | 38 |
| gray_leaf_spot | 82.8% | 91.7% | 87.0% | 289 |
| healthy | 99.1% | 97.1% | 98.1% | 346 |
| northern_leaf_blight | 92.7% | 89.8% | 91.2% | 411 |
| northern_leaf_spot | 94.5% | 100.0% | 97.2% | 52 |

Confusion matrix (rows = true, columns = predicted)

| true \ predicted | common_rust | eyespot | gray_leaf_spot | healthy | northern_leaf_blight | northern_leaf_spot |
|---|---|---|---|---|---|---|
| **common_rust** | 347 | 0 | 13 | 2 | 9 | 1 |
| **eyespot** | 0 | 36 | 1 | 0 | 0 | 1 |
| **gray_leaf_spot** | 3 | 0 | 265 | 1 | 20 | 0 |
| **healthy** | 1 | 0 | 9 | 336 | 0 | 0 |
| **northern_leaf_blight** | 9 | 0 | 32 | 0 | 369 | 1 |
| **northern_leaf_spot** | 0 | 0 | 0 | 0 | 0 | 52 |

| Source | Images | Correct | Accuracy |
|---|---|---|---|
| cds | 157 | 153 | 97.5% |
| kaggle | 417 | 387 | 92.8% |
| maize_multiformat | 72 | 53 | 73.6% |
| mendeley_bd | 213 | 203 | 95.3% |
| ndisan | 274 | 262 | 95.6% |
| neau | 85 | 83 | 97.6% |
| rlnst | 290 | 264 | 91.0% |

### Recall by class and source (test)

| Class | Source | Recall |
|---|---|---|
| common_rust | kaggle | 125/130 (96%) |
| common_rust | maize_multiformat | 5/14 (36%) |
| common_rust | mendeley_bd | 33/33 (100%) |
| common_rust | ndisan | 90/98 (92%) |
| common_rust | rlnst | 94/97 (97%) |
| eyespot | maize_multiformat | 12/12 (100%) |
| eyespot | neau | 24/26 (92%) |
| gray_leaf_spot | cds | 52/55 (95%) |
| gray_leaf_spot | kaggle | 53/57 (93%) |
| gray_leaf_spot | maize_multiformat | 0/10 (0%) |
| gray_leaf_spot | mendeley_bd | 86/89 (97%) |
| gray_leaf_spot | rlnst | 74/78 (95%) |
| healthy | kaggle | 113/115 (98%) |
| healthy | maize_multiformat | 10/10 (100%) |
| healthy | mendeley_bd | 84/91 (92%) |
| healthy | ndisan | 78/79 (99%) |
| healthy | neau | 31/31 (100%) |
| healthy | rlnst | 20/20 (100%) |
| northern_leaf_blight | cds | 49/50 (98%) |
| northern_leaf_blight | kaggle | 96/115 (83%) |
| northern_leaf_blight | maize_multiformat | 26/26 (100%) |
| northern_leaf_blight | ndisan | 94/97 (97%) |
| northern_leaf_blight | neau | 28/28 (100%) |
| northern_leaf_blight | rlnst | 76/95 (80%) |
| northern_leaf_spot | cds | 52/52 (100%) |

## Limitations
- `northern_leaf_spot` comes only from CD&S; small classes have few test photos, so their figures are rough.
- Single training run (seed 0); differences of a point or two between models are within run-to-run noise.
- Test sets come from the same sources as training. Field photos from unseen farms are harder (see `models/PLANTVILLAGE_BACKGROUND_ISSUE.md`).
- Near-duplicate grouping cannot rule out every copy of the same leaf across splits.
