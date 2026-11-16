# Model performance: `models/corn_fungal_v5_bgaug`

**Status: candidate to replace `corn_fungal_v5` as the app's model** — same data and settings plus background augmentation; better on field photos in this run. Matches the app's labels.

## Model
- MobileNetV2 (`google/mobilenet_v2_1.0_224`), 5 epochs, batch 16, BatchNorm momentum fix — identical to the baseline `corn_fungal_v5`
- Labels: `common_rust`, `eyespot`, `gray_leaf_spot`, `healthy`, `northern_leaf_blight`, `northern_leaf_spot`, `phaeosphaeria_leaf_spot`
- Data: `data/splits_v5.csv` — 13,174 train / 1,654 dev / 1,632 test (same split as the baseline)
- **Background augmentation** (the only difference): lab-style training photos (`kaggle`, `rlnst`, `ndisan`) with a usable leaf mask — 2,964 of them — get their background replaced on 70% of loads per epoch with a random flat colour, gradient, blotchy noise or a patch of one of 4,555 training-set field photos (`up_field`, `cds`, `mendeley_bd`, `neau`). Masks: `data_prep/background.py`, cached by `data_prep/leaf_masks.py` (3,625 masks over all splits). Dev and test images are never altered. Why: see `models/PLANTVILLAGE_BACKGROUND_ISSUE.md`.
- Command: `python -m data_prep.leaf_masks --manifest data/splits_v5.csv` then `python -m train.train --manifest data/splits_v5.csv --out models/corn_fungal_v5_bgaug --epochs 5 --bg-augment`

## Side-by-side with the baseline (same test set)

| | Images | Baseline `corn_fungal_v5` | Augmented `corn_fungal_v5_bgaug` | Change (points) |
|---|---|---|---|---|
| **Test accuracy, all** | 1,632 | 90.6% | 92.1% | +1.5 |
| Macro F1 | 1,632 | 83.5% | 84.5% | +1.1 |
| **Field-photo sources** | 651 | 87.6% | 89.6% | +2.0 |
| Lab-style sources | 981 | 92.6% | 93.8% | +1.2 |
| Dev accuracy (final epoch) | 1,654 | 92.3% | 91.0% | -1.3 |

### By source

| Source | Images | Baseline `corn_fungal_v5` | Augmented `corn_fungal_v5_bgaug` | Change (points) |
|---|---|---|---|---|
| up_field (field) | 124 | 73.4% | 75.8% | +2.4 |
| cds (field) | 157 | 96.2% | 97.5% | +1.3 |
| mendeley_bd (field) | 213 | 96.7% | 96.7% | +0.0 |
| neau (field) | 85 | 95.3% | 94.1% | -1.2 |
| maize_multiformat (field) | 72 | 56.9% | 69.4% | +12.5 |
| kaggle (lab-style) | 417 | 92.6% | 93.8% | +1.2 |
| rlnst (lab-style) | 290 | 91.0% | 93.8% | +2.8 |
| ndisan (lab-style) | 274 | 94.2% | 93.8% | -0.4 |

### By class

| Class | Test images | Precision (base → aug) | Recall (base → aug) | F1 baseline | F1 augmented | F1 change |
|---|---|---|---|---|---|---|
| common_rust | 382 | 94.7% → 95.1% | 88.5% → 91.4% | 91.5% | 93.2% | +1.7 |
| eyespot | 38 | 100.0% → 100.0% | 57.9% → 55.3% | 73.3% | 71.2% | -2.1 |
| gray_leaf_spot | 351 | 83.5% → 87.8% | 89.2% → 90.3% | 86.2% | 89.0% | +2.8 |
| healthy | 369 | 93.0% → 93.0% | 96.5% → 97.6% | 94.7% | 95.2% | +0.6 |
| northern_leaf_blight | 425 | 90.5% → 91.5% | 92.2% → 93.9% | 91.4% | 92.7% | +1.3 |
| northern_leaf_spot | 52 | 96.2% → 98.1% | 98.1% → 98.1% | 97.1% | 98.1% | +0.9 |
| phaeosphaeria_leaf_spot | 15 | 66.7% → 75.0% | 40.0% → 40.0% | 50.0% | 52.2% | +2.2 |

### Training curve

| Epoch | Baseline dev | Augmented dev |
|---|---|---|
| 1 | 90.1% | 88.3% |
| 2 | 91.7% | 91.0% |
| 3 | 92.1% | 91.0% |
| 4 | 92.0% | 90.7% |
| 5 | 92.3% | 91.0% |

The augmented model starts lower on dev (mostly lab-style photos, where backgrounds are a usable shortcut) and ends about one point below the baseline there, yet scores higher on the test set — especially on field photos.

## What this shows
- **Better on field photos:** +2.0 points on field-photo sources overall; `maize_multiformat` +12.5 (69.4% vs 56.9%), `up_field` +2.4, `cds` +1.3; `mendeley_bd` unchanged, `neau` −1.2 (one image).
- **No cost on lab-style photos:** +1.2 points there as well.
- **Most classes improve slightly** (gray leaf spot F1 +2.8, common rust +1.7, northern leaf blight +1.3); eyespot slips (−2.1) and Phaeosphaeria stays weak (52%, 15 test images).

## Augmented model in detail (test)

### Test (1632 images)

- **Accuracy:** 92.1%
- **Macro F1:** 84.5%

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| common_rust | 95.1% | 91.4% | 93.2% | 382 |
| eyespot | 100.0% | 55.3% | 71.2% | 38 |
| gray_leaf_spot | 87.8% | 90.3% | 89.0% | 351 |
| healthy | 93.0% | 97.6% | 95.2% | 369 |
| northern_leaf_blight | 91.5% | 93.9% | 92.7% | 425 |
| northern_leaf_spot | 98.1% | 98.1% | 98.1% | 52 |
| phaeosphaeria_leaf_spot | 75.0% | 40.0% | 52.2% | 15 |

Confusion matrix (rows = true, columns = predicted)

| true \ predicted | common_rust | eyespot | gray_leaf_spot | healthy | northern_leaf_blight | northern_leaf_spot | phaeosphaeria_leaf_spot |
|---|---|---|---|---|---|---|---|
| **common_rust** | 349 | 0 | 9 | 9 | 13 | 0 | 2 |
| **eyespot** | 12 | 21 | 2 | 2 | 0 | 1 | 0 |
| **gray_leaf_spot** | 1 | 0 | 317 | 11 | 22 | 0 | 0 |
| **healthy** | 0 | 0 | 8 | 360 | 1 | 0 | 0 |
| **northern_leaf_blight** | 4 | 0 | 22 | 0 | 399 | 0 | 0 |
| **northern_leaf_spot** | 0 | 0 | 0 | 1 | 0 | 51 | 0 |
| **phaeosphaeria_leaf_spot** | 1 | 0 | 3 | 4 | 1 | 0 | 6 |

| Source | Images | Correct | Accuracy |
|---|---|---|---|
| cds | 157 | 153 | 97.5% |
| kaggle | 417 | 391 | 93.8% |
| maize_multiformat | 72 | 50 | 69.4% |
| mendeley_bd | 213 | 206 | 96.7% |
| ndisan | 274 | 257 | 93.8% |
| neau | 85 | 80 | 94.1% |
| rlnst | 290 | 272 | 93.8% |
| up_field | 124 | 94 | 75.8% |

### Recall by class and source (test)

| Class | Source | Baseline recall | Augmented recall |
|---|---|---|---|
| common_rust | kaggle | 122/130 | 124/130 |
| common_rust | maize_multiformat | 5/14 | 14/14 |
| common_rust | mendeley_bd | 32/33 | 32/33 |
| common_rust | ndisan | 83/98 | 82/98 |
| common_rust | rlnst | 90/97 | 91/97 |
| common_rust | up_field | 6/10 | 6/10 |
| eyespot | maize_multiformat | 0/12 | 0/12 |
| eyespot | neau | 22/26 | 21/26 |
| gray_leaf_spot | cds | 52/55 | 53/55 |
| gray_leaf_spot | kaggle | 51/57 | 51/57 |
| gray_leaf_spot | maize_multiformat | 0/10 | 0/10 |
| gray_leaf_spot | mendeley_bd | 88/89 | 87/89 |
| gray_leaf_spot | rlnst | 71/78 | 75/78 |
| gray_leaf_spot | up_field | 51/62 | 51/62 |
| healthy | kaggle | 113/115 | 114/115 |
| healthy | maize_multiformat | 10/10 | 10/10 |
| healthy | mendeley_bd | 86/91 | 87/91 |
| healthy | ndisan | 79/79 | 79/79 |
| healthy | neau | 31/31 | 31/31 |
| healthy | rlnst | 20/20 | 20/20 |
| healthy | up_field | 17/23 | 19/23 |
| northern_leaf_blight | cds | 48/50 | 49/50 |
| northern_leaf_blight | kaggle | 100/115 | 102/115 |
| northern_leaf_blight | maize_multiformat | 26/26 | 26/26 |
| northern_leaf_blight | ndisan | 96/97 | 96/97 |
| northern_leaf_blight | neau | 28/28 | 28/28 |
| northern_leaf_blight | rlnst | 83/95 | 86/95 |
| northern_leaf_blight | up_field | 11/14 | 12/14 |
| northern_leaf_spot | cds | 51/52 | 51/52 |
| phaeosphaeria_leaf_spot | up_field | 6/15 | 6/15 |

## Not done yet
- **Background-swap test (2026-11-15):** on 323 lab-style test photos, field backgrounds cost 3.1 points for this model vs 10.5 for the same model without augmentation (common rust −3.9 vs −18.2). The field backgrounds come from the training photos this model saw as augmentation patches, so the test favours it somewhat. Details: `train/issues/PLANTVILLAGE_BACKGROUND_ISSUE.md`.

## Limitations
- One training run per model (seed 0); several sources have fewer than 100 test photos, so differences of a point or two are within run-to-run noise. The direction is consistent across most sources, and the `maize_multiformat` gain (+9 of 72 photos) is the clearest single change.
- Only 2,964 of the 8,094 lab-style training photos (kaggle 3,350, rlnst 2,308, ndisan 2,436) have a usable mask; the rest (leaves filling the frame, cluttered borders) are trained unchanged. A few masks keep a second background tone or a shadow attached to the leaf.
- `northern_leaf_spot` comes only from CD&S and `phaeosphaeria_leaf_spot` only from `up_field`; small classes have few test photos.
- Test sets come from the same sources as training; accuracy on a completely new farm or camera is untested.
