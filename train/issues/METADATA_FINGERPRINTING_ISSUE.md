# Metadata fingerprinting: why photo metadata and weather are not used as model inputs

**Status (updated 2026-11-15):** place and date are now used at prediction time through a sourced knowledge file (`app/knowledge/diseases.yaml`), not learned from our data; see README "Place, date and weather". On 676 test photos: 95.4% photo-only vs 95.1% weighted (4 answers changed: 1 to right, 3 to wrong). Training still never uses metadata. Original status: Metadata (capture date, GPS, city) and the weather derived from it are kept for analysis, not as training inputs. The deployed models (`corn_fungal_v6`, the `gated` detector) use the image only.

## The problem in one sentence

In our datasets, a photo's capture day, place and weather identify **which dataset and which photo session** it came from far better than they describe the disease. A model given them learns "this day / this place means this label", not "this weather favours this disease".

## How it happens

Most datasets were collected as a few photo sessions at one place, often one disease per session:

- **Each dataset has its own place.** 21 of the 22 cities in the metadata appear in only one dataset (e.g. West Lafayette = CD&S, Sampang = `ndisan`, Harbin = `neau`, Gurudaspur = `mendeley_bd`). The city alone names the dataset.
- **Each day is mostly one class.** v5 has 6,755 photos with a city and a capture date, spread over only 57 distinct (city, day) combinations. 37% of those photos come from days where every photo has the same label. Guessing each photo's label as the most common label of its day is right 77% of the time.
- **Weather is a per-day fingerprint.** Weather is looked up per (city, day), so every photo from the same session gets an identical 24-number weather vector. To a model, that vector is a session ID.
- **Which photos have metadata at all is a dataset hint.** Only 41% of v5 photos have a city and date. Almost all of them come from 4 field datasets (CD&S, `ndisan`, `neau`, `mendeley_bd`); PlantVillage-style Kaggle and RL-NST photos have none. A "has weather" flag therefore says "field dataset vs lab dataset".

## What we measured

From the image-only vs image + weather comparison (`docs/weather-input-comparison.md`, 14 Nov 2026, MobileNetV2, one run each):

| Observation | Number | What it shows |
|---|---|---|
| Label guessed from the capture day alone, no image (v3 split, test photos with weather) | **81.1%** | The date is a strong shortcut whenever test days also appear in training |
| Same guess when every test day is unseen (date-held-out split) | 0.5% | The shortcut is pure memorisation: it vanishes on new days |
| Image-only model on photos whose day is in training vs unseen | 95.4% vs **88.2%** | Same-day photos look alike (same field, light, camera), so even pixels leak the session |
| Image + weather vs image only, all test photos (v3) | 93.7% vs 91.8% | Looks like a gain... |
| ...on photos **without** weather | 92.4% vs 89.8% | ...but the biggest gain is where weather cannot help: noise, or the has-weather flag acting as a dataset hint |
| Date-held-out test days with weather | 2 days, each one class (CD&S 20 Aug 2020 all leaf spot; `ndisan` 30 Jul 2023 all rust) | One-class days cannot separate "weather predicts disease" from "weather predicts dataset" |

Conclusion: **no reliable evidence that weather helps**, and clear evidence that date and weather can act as shortcuts.

## Bad metadata makes it worse

- **`china_2`:** all 499 of its 2023 photos carry one EXIF date, 2023-12-06, across 8 provinces, including common rust in Gansu in December. That is almost certainly a processing date, not a capture date. Its weather rows describe the wrong day, but they would still work perfectly as a fingerprint for that batch.
- **`20237613`:** dates come from EXIF, but location is published only per region. A site for one camera's photos (Baynesfield) is inferred, not recorded.
- **`23669328` (Iowa State):** most leaves were scanned January–April and likely grown indoors, so outdoor weather would be wrong for them.
- **GPS** is present on only some photos of some datasets. City comes from GPS or a documented dataset site; when neither exists, the photo has no weather at all, which is itself informative.

## Why it matters for the app

A judge or farmer uploading a photo gives us a new day, often a new place, and usually no reliable metadata at all. A model that leaned on day or place fingerprints would score well on our test set and then lose that advantage on every real upload. This is the same kind of failure as the PlantVillage background shortcut (`PLANTVILLAGE_BACKGROUND_ISSUE.md`): the test set rewards it, the real world does not.

## What we do now

- **Image-only models.** `train/train.py` uses pixels only. The weather branch exists only in the experiment script `train/fusion.py`.
- **Date-held-out splits for any metadata experiment.** `data/splits_v3_by_date.csv` keeps every (city, day) in one set (`python -m data_prep.date_split`).
- **The capture date is never an input**, even in the fusion experiment. Weather summaries are used instead, and they still fingerprint the day (above).
- **Questionable metadata is flagged.** Examples are the `china_2` 2023-12-06 rows in `data/weather.csv` and `data/raw/IDADP/README.md`, and the Iowa State dataset left out of the weather data. Datasets outside the manifest are excluded from weather by default (`python -m data_prep.weather`; `--include-unused` to add them).

## What would be needed to use metadata safely

1. **Leave-one-day-out (or leave-one-site-out) evaluation**, with test days that each contain several classes, so a model cannot win by recognising the session.
2. **Several seeds per model** to separate 1–3 point differences from run-to-run noise.
3. **Field data where many diseases share the same days and places** (multi-disease plots or several fields per day). Without that, weather and disease are confounded with dataset.
4. **Verified capture dates and locations**, not EXIF dates of unknown origin.
5. **Metadata the app can actually get at prediction time**, e.g. the user's location and today's date. Then the model would be trained only on inputs available for a real upload.

Until then, metadata and weather are useful for **analysis** (for example: do errors cluster after wet weeks? which datasets overlap in time?), not as model inputs.

## Numbers in this note

- v5 counts: `data/splits_v5.csv` joined with `data/image_metadata.json`, computed 15 Nov 2026.
- Model comparison: `docs/weather-input-comparison.md` and `models/fusion_v3_*/MODEL_PERFORMANCE.md` (local).
