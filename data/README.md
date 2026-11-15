# Data

Everything the corn leaf model is trained and evaluated on. Image files live in `data/raw/` and are
**not** in git; the CSV files listed below are committed and record exactly which images are used and
how. More detail (label mapping, known issues, licenses) is in [`docs/data-sources.md`](../docs/data-sources.md),
and each dataset folder in `raw/` has its own `README.md` with per-class counts.

## Raw datasets (`data/raw/`)

| Folder | Source | Images | Classes used (label) | Where / when | License | Status |
|---|---|---|---|---|---|---|
| `data` | [Kaggle smaranjitghose/corn-or-maize-leaf-disease-dataset](https://www.kaggle.com/datasets/smaranjitghose/corn-or-maize-leaf-disease-dataset) | 4,188 | Blight (leaf_blight), Common_Rust, Gray_Leaf_Spot (leaf_spot), Healthy | mostly PlantVillage-style crops; unknown | not checked | used since v1 (4,186) |
| `rlnst` | `RL-NST(Data).zip` from the user; publisher unknown | 2,886 | Common_rust, Healthy, Leaf_Blight, Leaf_spot | unknown | unknown | used since v1; `Streak_virus` excluded |
| `cds` | [OSF s6ru5](https://osf.io/s6ru5) CD&S `Dataset_Original` ([paper](https://arxiv.org/abs/2110.12084)) | 1,571 | gls, nls (leaf_spot), nlb (leaf_blight) | near West Lafayette, Indiana, USA; Jul–Aug 2020 | none stated | used since v1 |
| `An Open-Source Multi-Format Maize Leaf Image Datas` | folder from the user; publisher not identified | 5,770 (994 originals) | `RAW_Image/`: Corn_Leaf_Blight, Common_Rust, Gray_Leaf_Spot + Eyespot (leaf_spot), Healthy_Leaf | unknown | unknown | `RAW_Image` used since v2 (664); copies, streak and bacterial streak not used |
| `kaggle_corn_leaf_disease` | [Kaggle ndisan/corn-leaf-disease](https://www.kaggle.com/datasets/ndisan/corn-leaf-disease) | 4,000 | Daun Sehat (healthy), Hawar Daun (leaf_blight), Karat Daun (common_rust) | Polagan Village, Sampang, East Java, Indonesia; Apr–Jul 2023 | MIT | used in v3 (2,994); `Bercak Daun` not used (copies) |
| `yusuf_corn` | [Kaggle yusufmurtaza01/corn-leaf-diseases](https://www.kaggle.com/datasets/yusufmurtaza01/corn-leaf-diseases) | 4,027 | — | — | not checked | not used: 3,992 images duplicate `data` |
| `Final corn dataset` | [Mendeley vy629dngm8](https://data.mendeley.com/datasets/vy629dngm8/1) "Seasonal Corn Leaf Disease Dataset" | 2,943 | would be: Healthy, Gray_leaf_spot, Common_rust | Gurudaspur, Natore, Bangladesh | CC BY 4.0 | downloaded, not yet checked or used |
| `Northeast Agricultural University` | folder from the user (details in its `README.txt`) | 856 | would be: healthy, northern leaf blight | Harbin, China; Jul–Aug 2021 | unknown | downloaded, not yet checked or used |

## Generated files (`data/`)

| File | Rows | What it is |
|---|---|---|
| `splits.csv` | 8,643 | v1 train/dev/test manifest (`kaggle`, `rlnst`, `cds`): one row per image with source, label, near-duplicate group and split. `models/corn` was trained on it |
| `splits_v2.csv` | 9,307 | v1 plus `maize_multiformat`, append-only (no v1 image moved). `models/corn_v2` was trained on it |
| `splits_v3.csv` | 12,301 | v2 plus `ndisan`, append-only (no v2 image moved). Newest manifest |
| `splits_v3_by_date.csv` | 12,301 | v3 with every (city, capture date) kept in one set, for testing weather input fairly |
| `splits_dropped.csv`, `splits_v2_dropped.csv`, `splits_v3_dropped.csv` | 2 / 28 / 34 | images left out of each manifest and why (exact duplicate, unreadable) |
| `image_metadata.csv` | 12,301 | per image in v3: capture date, GPS (rounded to ~110 m), camera, documented site and city |
| `weather.csv` | 40 | weather summaries (7 days, 1, 3 and 6 months before capture) per distinct (city, capture date); join on `city` + `capture_date` |

## Other folders

| Folder | What it is |
|---|---|
| `raw/weather_cache/`, `raw/geocode_cache/` | cached Open-Meteo and OpenStreetMap responses, so the weather and city lookups can be regenerated without network calls |
| `combined/` | legacy symlink folder (7,858 links) from the first training run, before the manifests existed; no longer used and safe to delete |

## Rebuilding

Never regenerate an existing manifest (the builder refuses without `--force`): each one records what a
model was trained on. Add data on top of the newest one instead:

```bash
python -m data_prep.build_splits --add NAME=PATH --base data/splits_v3.csv --out data/splits_v4.csv
python -m data_prep.image_metadata --manifest data/splits_v4.csv
python -m data_prep.weather
```
