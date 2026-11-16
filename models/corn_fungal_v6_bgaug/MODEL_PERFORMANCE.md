# Model performance: `models/corn_fungal_v6_bgaug`

**Status: experiment.** It is the same as `corn_fungal_v6` plus background augmentation. In this run augmentation did **not** repeat its v5 gain: it is level with the baseline overall (91.3% vs 91.6% test accuracy, macro F1 84.9% vs 83.7%). Not used by the app.

## Model
- MobileNetV2 (`google/mobilenet_v2_1.0_224`), 5 epochs, batch 16, BatchNorm momentum fix, seed 0, 11 labels; otherwise identical to `corn_fungal_v6`
- Data: `data/splits_v6.csv` — 13,894 train / 1,748 dev / 1,719 test
- **Background augmentation:** 2,964 lab-style training photos (`kaggle`, `rlnst`, `ndisan`) with a usable leaf mask get their background replaced on 70% of loads, using a flat colour, gradient, noise or a patch from 5,228 field photos (now including `china_2`)
- Command: `python -m data_prep.leaf_masks --manifest data/splits_v6.csv` then `python -m train.train --manifest data/splits_v6.csv --out models/corn_fungal_v6_bgaug --epochs 5 --bg-augment`

## Side-by-side with the baseline

| | Images | `corn_fungal_v6` | `corn_fungal_v6_bgaug` | Change (points) |
|---|---|---|---|---|
| **v6 test accuracy** | 1,719 | 91.6% | 91.3% | −0.3 |
| v6 test macro F1 | 1,719 | 83.7% | 84.9% | +1.2 |
| v5 test photos | 1,632 | 91.9% | 91.3% | −0.6 |
| Field-photo sources (v5 test) | 651 | 89.4% | 88.8% | −0.6 |
| Lab-style sources (v5 test) | 981 | 93.6% | 93.0% | −0.6 |
| Dev accuracy (final epoch) | 1,748 | 91.3% | 91.8% | +0.5 |

### By source (v6 test)

| Source | Images | Baseline | Augmented | Change |
|---|---|---|---|---|
| up_field (field) | 125 | 74.4% | 74.4% | 0.0 |
| cds (field) | 157 | 95.5% | 96.2% | +0.7 |
| mendeley_bd (field) | 213 | 97.2% | 98.1% | +0.9 |
| neau (field) | 85 | 96.5% | 96.5% | 0.0 |
| maize_multiformat (field) | 72 | 69.4% | 59.7% | −9.7 |
| china_2 (field) | 82 | 89.0% | 93.9% | +4.9 |
| kaggle (lab-style) | 416 | 94.0% | 93.5% | −0.5 |
| rlnst (lab-style) | 290 | 90.7% | 90.3% | −0.4 |
| ndisan (lab-style) | 274 | 96.4% | 95.3% | −1.1 |
| xtu502 (web) | 5 | 40.0% | 40.0% | 0.0 |

### By class (F1, v6 test)

| Class | Test photos | Baseline | Augmented | Change |
|---|---|---|---|---|
| brown_spot | 11 | 87.0% | 90.9% | +3.9 |
| common_rust | 404 | 94.0% | 91.8% | −2.2 |
| curvularia_leaf_spot | 10 | 73.7% | 77.8% | +4.1 |
| eyespot | 40 | 75.0% | 71.4% | −3.6 |
| gray_leaf_spot | 352 | 88.1% | 88.5% | +0.4 |
| healthy | 369 | 95.9% | 96.7% | +0.8 |
| northern_leaf_blight | 424 | 91.8% | 91.6% | −0.2 |
| northern_leaf_spot | 52 | 96.2% | 96.2% | 0.0 |
| phaeosphaeria_leaf_spot | 16 | 51.9% | 53.8% | +1.9 |
| southern_leaf_blight | 20 | 78.9% | 90.5% | +11.6 |
| southern_rust | 21 | 88.4% | 84.4% | −4.0 |

The augmented model mixes up common rust and eyespot more: 14 common rust photos are predicted as eyespot (baseline 0), while eyespot recall rose from 60% to 75%.

### Training curve

| Epoch | Baseline dev | Augmented dev |
|---|---|---|
| 1 | 88.3% | 87.9% |
| 2 | 90.5% | 90.6% |
| 3 | 90.9% | 91.0% |
| 4 | 91.3% | 91.6% |
| 5 | 91.3% | 91.8% |

## What this shows
- **The v5 field-photo gain did not repeat.** On v5, augmentation added +2.0 points on field photos; here it is −0.6. The largest swing is again `maize_multiformat` (72 photos), which went +12.5 on v5 and −9.7 here. That points to run-to-run noise on small test sources, not a reliable effect either way.
- **Overall the two methods are level.** On ordinary test accuracy they can only be told apart with several seeds per method. The background-swap test, which directly measures reliance on backgrounds, does separate them: field backgrounds cost this model 2.8 points vs 8.4 for the baseline.
- `corn_fungal_v6` (baseline) stays the v6 reference model.

### v6 test (1,719 images)

- **Accuracy:** 91.3%
- **Macro F1:** 84.9%

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| brown_spot | 90.9% | 90.9% | 90.9% | 11 |
| common_rust | 93.6% | 90.1% | 91.8% | 404 |
| curvularia_leaf_spot | 87.5% | 70.0% | 77.8% | 10 |
| eyespot | 68.2% | 75.0% | 71.4% | 40 |
| gray_leaf_spot | 87.7% | 89.2% | 88.5% | 352 |
| healthy | 95.5% | 97.8% | 96.7% | 369 |
| northern_leaf_blight | 91.9% | 91.3% | 91.6% | 424 |
| northern_leaf_spot | 94.4% | 98.1% | 96.2% | 52 |
| phaeosphaeria_leaf_spot | 70.0% | 43.8% | 53.8% | 16 |
| southern_leaf_blight | 86.4% | 95.0% | 90.5% | 20 |
| southern_rust | 79.2% | 90.5% | 84.4% | 21 |

Confusion matrix (rows = true, columns = predicted)

| true \ predicted | brown_spot | common_rust | curvularia_leaf_spot | eyespot | gray_leaf_spot | healthy | northern_leaf_blight | northern_leaf_spot | phaeosphaeria_leaf_spot | southern_leaf_blight | southern_rust |
|---|---|---|---|---|---|---|---|---|---|---|---|
| **brown_spot** | 10 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 |
| **common_rust** | 0 | 364 | 0 | 14 | 6 | 4 | 11 | 0 | 2 | 0 | 3 |
| **curvularia_leaf_spot** | 0 | 1 | 7 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 2 |
| **eyespot** | 0 | 7 | 1 | 30 | 1 | 0 | 0 | 1 | 0 | 0 | 0 |
| **gray_leaf_spot** | 0 | 7 | 0 | 0 | 314 | 8 | 21 | 0 | 1 | 1 | 0 |
| **healthy** | 0 | 2 | 0 | 0 | 4 | 361 | 1 | 0 | 0 | 1 | 0 |
| **northern_leaf_blight** | 1 | 6 | 0 | 0 | 29 | 0 | 387 | 1 | 0 | 0 | 0 |
| **northern_leaf_spot** | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 51 | 0 | 0 | 0 |
| **phaeosphaeria_leaf_spot** | 0 | 2 | 0 | 0 | 3 | 4 | 0 | 0 | 7 | 0 | 0 |
| **southern_leaf_blight** | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 19 | 0 |
| **southern_rust** | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 1 | 19 |

| Source | Images | Correct | Accuracy |
|---|---|---|---|
| cds | 157 | 151 | 96.2% |
| china_2 | 82 | 77 | 93.9% |
| kaggle | 416 | 389 | 93.5% |
| maize_multiformat | 72 | 43 | 59.7% |
| mendeley_bd | 213 | 209 | 98.1% |
| ndisan | 274 | 261 | 95.3% |
| neau | 85 | 82 | 96.5% |
| rlnst | 290 | 262 | 90.3% |
| up_field | 125 | 93 | 74.4% |
| xtu502 | 5 | 2 | 40.0% |


## Limitations
- Single training run (seed 0) per model; see above.
- **Background-swap test (2026-11-15):** on 323 lab-style test photos, field backgrounds cost 2.8 points for this model vs 8.4 for the same model without augmentation (common rust −4.4 vs −13.3). The field backgrounds come from the training photos this model saw as augmentation patches, so the test favours it somewhat. Details: `train/issues/PLANTVILLAGE_BACKGROUND_ISSUE.md`.
- The four new classes and Phaeosphaeria have 10–21 test photos each, mostly from one dataset.
- Test sets come from the same sources as training; unseen farms are harder (see `train/issues/PLANTVILLAGE_BACKGROUND_ISSUE.md`).
