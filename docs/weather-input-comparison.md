# Image only vs image + weather — comparison

Date: 2026-11-14. Models in `models/fusion_*` (not in git); raw results in each `results.json`.

## Setup

- **Model:** MobileNetV2 (`google/mobilenet_v2_1.0_224`), 5 epochs, batch 16, AdamW 5e-5, seed 0, BatchNorm momentum fix. One run per cell.
- **Weather input:** the 24 summaries from `data/weather.csv` (mean / min / max temperature, precipitation, rainy days, humidity for the 7 days, 1, 3 and 6 months before capture), standardised on the training set, plus a has-weather flag. The capture date is never an input. Photos without weather get zeros and flag 0. The image-only model is the same code with the weather branch switched off.
- **Coverage:** only 4,590 of 12,301 images (37%) have weather: CD&S, `ndisan` and 25 Kaggle photos.
- **Two splits:**
  - `splits_v3.csv` — dates are shared between train and test (431 of 433 test photos with weather share their day with training photos).
  - `splits_v3_by_date.csv` — every (city, capture date) is entirely in one set, so test days are unseen. Its test photos with weather are only two days: CD&S 20 Aug 2020 (224 photos, all leaf spot) and `ndisan` 30 Jul 2023 (146, all rust), plus 2 Kaggle photos.
- **Why the second split:** each (city, date) has its own weather vector, and the date alone predicts the label well (baseline row below), so on v3 weather can act as a shortcut.

## Results (test set)

| | v3: image only | v3: image + weather | by date: image only | by date: image + weather |
|---|---|---|---|---|
| Test photos | 1,210 | 1,210 | 1,149 | 1,149 |
| **Accuracy, all** | 91.8% | **93.7%** | 90.9% | **91.8%** |
| Macro F1, all | 92.2% | 94.1% | 91.8% | 92.6% |
| Accuracy, photos with weather | 95.4% (433) | 96.1% (433) | 88.2% (372) | 91.9% (372) |
| Accuracy, photos without weather | 89.8% (777) | 92.4% (777) | 92.3% (777) | 91.8% (777) |
| F1 common_rust | 93.1% | 94.4% | 93.8% | 94.7% |
| F1 healthy | 99.6% | 99.6% | 99.3% | 99.7% |
| F1 leaf_blight | 90.6% | 92.7% | 86.1% | 86.5% |
| F1 leaf_spot | 85.6% | 89.7% | 88.0% | 89.5% |
| Accuracy cds | 93.6% | 94.9% | 83.9% | 89.7% |
| Accuracy ndisan | 96.4% | 96.7% | 95.2% | 95.9% |
| Accuracy kaggle | 93.0% | 94.0% | 93.8% | 93.0% |
| Accuracy rlnst | 91.0% | 91.4% | 92.8% | 92.4% |
| Accuracy maize_multiformat (no weather) | 66.7% | 87.5% | 80.6% | 80.6% |
| **Date-only guess** (no image; majority label of the same (city, date) in train), photos with weather | 81.1% | | 0.5% (no test day seen in train) | |

## What this shows

- **No reliable evidence that weather helps.** Image + weather is ahead on both splits (+1.9 and +0.9 points), but:
  - On v3 the biggest gains are on photos **without** weather (+2.6 points; maize_multiformat 66.7% → 87.5% on 72 photos), which the weather input cannot cause. That points to ordinary run-to-run variation (one run each), or the has-weather flag acting as a hint about which dataset a photo comes from.
  - On the date-held-out split the gain on photos with weather (+3.7 points) comes from two test days that each contain a single class. CD&S has more leaf spot than blight, so Indiana-summer weather may simply push predictions toward leaf spot — a dataset hint, not a disease–weather link. One-class test days cannot tell the difference.
- **Same-day leakage is real even without weather:** the image-only model scores 95.4% on photos with weather when their days are shared with training, but 88.2% when the days are unseen. Photos from the same day look alike (same field, light, camera).
- **The date-only baseline** (81% on v3) shows how much a model could gain from memorising days; the v3 comparison cannot separate that from real signal.

## What would settle it

- Repeat each model with several seeds (for example 3 × 4 runs, about 2 hours) to measure run-to-run variation.
- Hold out more days, ideally leave-one-day-out across CD&S and `ndisan`, so test days include several classes.
- Until then, treat weather as useful for analysis (for example, whether errors cluster after wet periods), not as a proven model input.

## Reproduce

```bash
python -m data_prep.date_split                    # data/splits_v3_by_date.csv
python -m train.fusion --manifest data/splits_v3.csv --out models/fusion_v3_image --epochs 5
python -m train.fusion --manifest data/splits_v3.csv --out models/fusion_v3_weather --epochs 5 --with-weather
python -m train.fusion --manifest data/splits_v3_by_date.csv --out models/fusion_v3_by_date_image --epochs 5
python -m train.fusion --manifest data/splits_v3_by_date.csv --out models/fusion_v3_by_date_weather --epochs 5 --with-weather
```
