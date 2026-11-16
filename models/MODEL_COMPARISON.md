# Model comparison: `corn`, `corn_fungal_v6`, `corn_fungal_v6_bgaug`

All three are MobileNetV2 (`google/mobilenet_v2_1.0_224`), 5 epochs, batch 16, BatchNorm momentum fix, seed 0,
one training run each. Details per model are in `models/<name>/MODEL_PERFORMANCE.md`.

| | `corn` | `corn_fungal_v6` | `corn_fungal_v6_bgaug` |
|---|---|---|---|
| Role | first model (superseded) | v6 baseline | **the website's model** |
| Training data | `data/splits.csv` (v1): Kaggle, RL-NST, CD&S; 6,914 train photos | `data/splits_v6.csv`: 10 datasets; 13,894 train photos | same as `corn_fungal_v6` |
| Labels | 4 merged: common rust, healthy, leaf blight, leaf spot | 11: healthy + 10 fungal diseases | 11 |
| Background augmentation | no | no | yes (lab-style photos get random backgrounds on 70% of loads) |

## Headline numbers

| Evaluation | `corn` | `corn_fungal_v6` | `corn_fungal_v6_bgaug` |
|---|---|---|---|
| Own test set: accuracy | 94.4% (864 photos, v1) | 91.6% (1,719 photos, v6) | 91.3% (1,719 photos, v6) |
| Own test set: macro F1 | 95.1% | 83.7% | 84.9% |
| v5 test photos (1,632, same set for both v6 models) | — | 91.9% | 91.3% |
| Field-photo sources in v5 test (651) | — | 89.4% | 88.8% |
| Lab-style sources in v5 test (981) | — | 93.6% | 93.0% |
| A dataset never seen in training (`maize_multiformat`, all 664 photos) | **37.8%** | — (trained on it) | — (trained on it) |
| Final dev accuracy | 94.3% | 91.3% | 91.8% |

The test sets differ: `corn` was scored on 4 merged labels and the v1 test photos, the v6 models on 11 labels and
the harder v6 test photos (with field photos from South Africa, China and more). So `corn`'s 94.4% is not better
than the v6 models' ~91.5%. Its 37.8% on an unseen dataset shows how far it was from working on new photos.

## Reliance on backgrounds (background-swap test)

The same 323 lab-style test photos, scored with their original background and with a field-photo background
from photos no model trained on (`python -m train.bg_swap --pool unseen`). A big drop means the model
leans on the background, not the leaf.

| | `corn` | `corn_fungal_v6` | `corn_fungal_v6_bgaug` |
|---|---|---|---|
| Original background | not run | 95.7% | 94.1% |
| Field background | not run | 85.4% | 91.3% |
| **Drop** | — | **−10.2** | **−2.8** |

## By source (v6 test set)

| Source | Photos | `corn_fungal_v6` | `corn_fungal_v6_bgaug` |
|---|---|---|---|
| cds (field, USA) | 157 | 95.5% | 96.2% |
| mendeley_bd (field, Bangladesh) | 213 | 97.2% | 98.1% |
| neau (field, China) | 85 | 96.5% | 96.5% |
| china_2 (field, China) | 82 | 89.0% | 93.9% |
| up_field (field, South Africa) | 125 | 74.4% | 74.4% |
| maize_multiformat (field) | 72 | 69.4% | 59.7% |
| kaggle (lab-style) | 416 | 94.0% | 93.5% |
| rlnst (lab-style) | 290 | 90.7% | 90.3% |
| ndisan (lab-style) | 274 | 96.4% | 95.3% |
| xtu502 (web) | 5 | 40.0% | 40.0% |

`corn` per source on its v1 test set: cds 97.5% (157), kaggle 94.2% (417), rlnst 93.1% (290).

## By class (F1)

| Class | `corn` (v1 test) | `corn_fungal_v6` (v6 test) | `corn_fungal_v6_bgaug` (v6 test) | v6 test photos |
|---|---|---|---|---|
| Healthy | 99.6% | 95.9% | 96.7% | 369 |
| Common rust | 97.8% | 94.0% | 91.8% | 404 |
| Southern rust | — | 88.4% | 84.4% | 21 |
| Northern leaf blight | 91.5% (as "leaf blight") | 91.8% | 91.6% | 424 |
| Southern leaf blight | — | 78.9% | 90.5% | 20 |
| Gray leaf spot | 91.6% (as "leaf spot") | 88.1% | 88.5% | 352 |
| Northern leaf spot | (in "leaf spot") | 96.2% | 96.2% | 52 |
| Eyespot | — | 75.0% | 71.4% | 40 |
| Phaeosphaeria leaf spot | — | 51.9% | 53.8% | 16 |
| Brown spot | — | 87.0% | 90.9% | 11 |
| Curvularia leaf spot | — | 73.7% | 77.8% | 10 |

Classes with 10–52 test photos have rough figures (a few photos move them by 5–10 points).

## Place and date weighting (website model only)

On 676 v6 test photos with a real place and capture date, `corn_fungal_v6_bgaug` scored 95.4% on the photo
alone and 95.1% with place, season and weather weighting (4 answers changed: 1 to right, 3 to wrong). See
`models/corn_fungal_v6_bgaug/weighting_check.md`.

## Summary

- `corn` looked strong on its own test set (94.4%) but collapsed to 37.8% on a dataset it never saw.
- `corn_fungal_v6` and `corn_fungal_v6_bgaug` are level on ordinary test accuracy (91.6% vs 91.3%, within
  run-to-run noise), and both recognise all 11 labels.
- `corn_fungal_v6_bgaug` relies far less on backgrounds (−2.8 vs −10.2 points), which is why the website uses it.
- Hard field photos remain the weak spot for both v6 models (about 74% on South African photos, 60–70% on the
  multi-format set).

_Generated 2026-11-15 from each model's evaluation files._
