# Model performance: `models/corn_fungal_v5`

**Status: current baseline** for the background-augmentation comparison (`corn_fungal_v5_bgaug`). Matches the app's labels.

## Model
- MobileNetV2 (`google/mobilenet_v2_1.0_224`), 5 epochs, batch 16, BatchNorm momentum fix
- Labels: `common_rust`, `eyespot`, `gray_leaf_spot`, `healthy`, `northern_leaf_blight`, `northern_leaf_spot`, `phaeosphaeria_leaf_spot` (the app's label set)
- Data: `data/splits_v5.csv` — v4 plus single-label `up_field` (University of Pretoria) field photos; 13,174 train / 1,654 dev / 1,632 test
- Command: `python -m train.train --manifest data/splits_v5.csv --out models/corn_fungal_v5 --epochs 5`

## Training
| Epoch | Dev accuracy |
|---|---|
| 1 | 90.1% |
| 2 | 91.7% |
| 3 | 92.1% |
| 4 | 92.0% |
| 5 | 92.3% |

## Results

| Split | Images | Accuracy | Macro F1 |
|---|---|---|---|
| dev | 1654 | 92.3% | 89.7% |
| **test** | 1632 | **90.6%** | 83.5% |

### Test (1632 images)

- **Accuracy:** 90.6%
- **Macro F1:** 83.5%

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| common_rust | 94.7% | 88.5% | 91.5% | 382 |
| eyespot | 100.0% | 57.9% | 73.3% | 38 |
| gray_leaf_spot | 83.5% | 89.2% | 86.2% | 351 |
| healthy | 93.0% | 96.5% | 94.7% | 369 |
| northern_leaf_blight | 90.5% | 92.2% | 91.4% | 425 |
| northern_leaf_spot | 96.2% | 98.1% | 97.1% | 52 |
| phaeosphaeria_leaf_spot | 66.7% | 40.0% | 50.0% | 15 |

Confusion matrix (rows = true, columns = predicted)

| true \ predicted | common_rust | eyespot | gray_leaf_spot | healthy | northern_leaf_blight | northern_leaf_spot | phaeosphaeria_leaf_spot |
|---|---|---|---|---|---|---|---|
| **common_rust** | 338 | 0 | 18 | 9 | 14 | 0 | 3 |
| **eyespot** | 12 | 22 | 1 | 2 | 0 | 1 | 0 |
| **gray_leaf_spot** | 2 | 0 | 313 | 10 | 25 | 1 | 0 |
| **healthy** | 0 | 0 | 12 | 356 | 1 | 0 | 0 |
| **northern_leaf_blight** | 4 | 0 | 28 | 1 | 392 | 0 | 0 |
| **northern_leaf_spot** | 0 | 0 | 0 | 1 | 0 | 51 | 0 |
| **phaeosphaeria_leaf_spot** | 1 | 0 | 3 | 4 | 1 | 0 | 6 |

| Source | Images | Correct | Accuracy |
|---|---|---|---|
| cds | 157 | 151 | 96.2% |
| kaggle | 417 | 386 | 92.6% |
| maize_multiformat | 72 | 41 | 56.9% |
| mendeley_bd | 213 | 206 | 96.7% |
| ndisan | 274 | 258 | 94.2% |
| neau | 85 | 81 | 95.3% |
| rlnst | 290 | 264 | 91.0% |
| up_field | 124 | 91 | 73.4% |

### Recall by class and source (test)

| Class | Source | Recall |
|---|---|---|
| common_rust | kaggle | 122/130 (94%) |
| common_rust | maize_multiformat | 5/14 (36%) |
| common_rust | mendeley_bd | 32/33 (97%) |
| common_rust | ndisan | 83/98 (85%) |
| common_rust | rlnst | 90/97 (93%) |
| common_rust | up_field | 6/10 (60%) |
| eyespot | maize_multiformat | 0/12 (0%) |
| eyespot | neau | 22/26 (85%) |
| gray_leaf_spot | cds | 52/55 (95%) |
| gray_leaf_spot | kaggle | 51/57 (89%) |
| gray_leaf_spot | maize_multiformat | 0/10 (0%) |
| gray_leaf_spot | mendeley_bd | 88/89 (99%) |
| gray_leaf_spot | rlnst | 71/78 (91%) |
| gray_leaf_spot | up_field | 51/62 (82%) |
| healthy | kaggle | 113/115 (98%) |
| healthy | maize_multiformat | 10/10 (100%) |
| healthy | mendeley_bd | 86/91 (95%) |
| healthy | ndisan | 79/79 (100%) |
| healthy | neau | 31/31 (100%) |
| healthy | rlnst | 20/20 (100%) |
| healthy | up_field | 17/23 (74%) |
| northern_leaf_blight | cds | 48/50 (96%) |
| northern_leaf_blight | kaggle | 100/115 (87%) |
| northern_leaf_blight | maize_multiformat | 26/26 (100%) |
| northern_leaf_blight | ndisan | 96/97 (99%) |
| northern_leaf_blight | neau | 28/28 (100%) |
| northern_leaf_blight | rlnst | 83/95 (87%) |
| northern_leaf_blight | up_field | 11/14 (79%) |
| northern_leaf_spot | cds | 51/52 (98%) |
| phaeosphaeria_leaf_spot | up_field | 6/15 (40%) |
## Background-swap test

Lab-style test photos (`kaggle`, `rlnst`, `ndisan`) with a usable leaf mask, scored with their original background and with it replaced. A drop means the model relies on the background.

| Background | all | common_rust | kaggle | rlnst | ndisan | healthy |
|---|---|---|---|---|---|---|
| original | 93.5% (323) | 91.2% (181) | 97.1% (103) | 91.5% (59) | 91.9% (161) | 100.0% (57) |
| flat | 91.6% (323) | 91.7% (181) | 94.2% (103) | 88.1% (59) | 91.3% (161) | 100.0% (57) |
| field | 83.0% (323) | 72.9% (181) | 78.6% (103) | 79.7% (59) | 87.0% (161) | 100.0% (57) |

On realistic field-photo backgrounds the same leaves lose 10.5 points overall and 18 for common rust, consistent with PlantVillage's class-specific backgrounds.

## Limitations
- `northern_leaf_spot` comes only from CD&S and `phaeosphaeria_leaf_spot` only from `up_field`; small classes have few test photos, so their figures are rough.
- Single training run (seed 0); differences of a point or two between models are within run-to-run noise.
- Test sets come from the same sources as training. Field photos from unseen farms are harder (see `models/PLANTVILLAGE_BACKGROUND_ISSUE.md`).
- Near-duplicate grouping cannot rule out every copy of the same leaf across splits.
