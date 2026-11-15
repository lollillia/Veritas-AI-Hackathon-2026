# Data

Everything the corn leaf model is trained and evaluated on. The study covers **fungal** leaf diseases:
the labels are `healthy`, `common_rust`, `northern_leaf_blight`, `gray_leaf_spot`, `northern_leaf_spot`
and `eyespot`. Image files live in `data/raw/` and are **not** in git; the CSV files listed below are
committed and record exactly which images are used and how. More detail (label mapping, known issues,
licenses) is in [`docs/data-sources.md`](../docs/data-sources.md), and each dataset folder in `raw/` has its
own `README.md` with per-class counts.

## Raw datasets (`data/raw/`)

| Folder | Source | Images | Folders used (label) | Where / when | License | Status |
|---|---|---|---|---|---|---|
| `data` | [Kaggle smaranjitghose/corn-or-maize-leaf-disease-dataset](https://www.kaggle.com/datasets/smaranjitghose/corn-or-maize-leaf-disease-dataset) | 4,188 | Blight (northern_leaf_blight), Common_Rust, Gray_Leaf_Spot, Healthy | mostly PlantVillage-style crops; unknown | not checked | used since v1 (4,186) |
| `rlnst` | `RL-NST(Data).zip` from the user; publisher unknown | 2,886 | Common_rust, Healthy, Leaf_Blight (northern_leaf_blight), Leaf_spot (gray_leaf_spot) | unknown | unknown | used since v1; `Streak_virus` excluded |
| `cds` | [OSF s6ru5](https://osf.io/s6ru5) CD&S `Dataset_Original` ([paper](https://arxiv.org/abs/2110.12084)) | 1,571 | gls (gray_leaf_spot), nlb (northern_leaf_blight), nls (northern_leaf_spot) | near West Lafayette, Indiana, USA; Jul–Aug 2020 | none stated | used since v1 |
| `An Open-Source Multi-Format Maize Leaf Image Datas` | folder from the user; publisher not identified | 5,770 (994 originals) | `RAW_Image/`: Corn_Leaf_Blight, Common_Rust, Gray_Leaf_Spot, Eyespot, Healthy_Leaf | unknown | unknown | `RAW_Image` used since v2 (664); copies, streak and bacterial streak not used |
| `kaggle_corn_leaf_disease` | [Kaggle ndisan/corn-leaf-disease](https://www.kaggle.com/datasets/ndisan/corn-leaf-disease) | 4,000 | Daun Sehat (healthy), Hawar Daun (northern_leaf_blight), Karat Daun (common_rust) | Polagan Village, Sampang, East Java, Indonesia; Apr–Jul 2023 | MIT | used since v3 (2,994); `Bercak Daun` not used (copies) |
| `Final corn dataset` | [Mendeley vy629dngm8](https://data.mendeley.com/datasets/vy629dngm8/1) "Seasonal Corn Leaf Disease Dataset" | 2,943 (2,060 unique) | Healthy, Gray_leaf_spot, Common_rust | Gurudaspur, Natore, Bangladesh; Dec 2022 – Feb 2024 | CC BY 4.0 | used in v4 (2,060); 604 internal duplicate files dropped; bacterial streak and chlorotic mottle virus not used |
| `Northeast Agricultural University` | folder from the user (details in its `README.txt`) | 856 | healthy maize leaves, maize northern leaf blight, maize northern anthracnose (eyespot) | Harbin, China; Jul–Aug 2021 | unknown | used in v4 (856) |
| `yusuf_corn` | [Kaggle yusufmurtaza01/corn-leaf-diseases](https://www.kaggle.com/datasets/yusufmurtaza01/corn-leaf-diseases) | 4,027 | — | — | not checked | not used: 3,992 images duplicate `data` |
| `china_2` | folder from the user with `desc.json` (a Chinese description per image). Probably the CAAS image–text corn disease dataset ([*Journal of Agricultural Big Data* 2025](https://opaj.napstic.cn/periodicalArticle/0120251000190036)); unconfirmed | 1,655 (965 new) | `data/` by code: NLB, CR, SLB, SCR (southern rust), GLS, SUM (dwarf mosaic), CBS (brown spot), CLS (Curvularia), NLS | new images: Shandong, Gansu, Henan, Guizhou, Yunnan, Anhui (China), 2014–2023 | unknown | not used; GLS, NLB and NLS (690) are byte-identical CD&S copies |
| `23669328` | [Iowa State figshare 23669328](https://iastate.figshare.com/articles/dataset/Images_and_annotations_of_em_Puccinia_sorghi_em_infected_maize_leaves/23669328) (source `isu_rust`) | 1,309 | scanned leaf strips (TIFF) with pustule outlines; 1,033 with pustules (common_rust) | Ames, Iowa, USA; 2018–2021 | CC BY-NC 4.0 | not used: scanner style would be a shortcut for common rust |
| `xtu502` | [GitHub xtu502/maize-disease-identification](https://github.com/xtu502/maize-disease-identification), test images of Mobile-DANet ([Chen et al. 2021](https://doi.org/10.1111/ppa.13322)) | 533 | `36_Pub_Maize` (4 classes × 100); `33_Col_Maize` (133: gray leaf spot, eyespot, Phaeosphaeria leaf spot, southern rust, smut, ear rot, crazy top, Goss's wilt) | web/field photos, mostly no EXIF | none stated | not used; all 400 `36_Pub_Maize` and 12 `33_Col_Maize` images are copies of `data` |

## Diseases in the raw images

10 diseases plus healthy leaves; **6 are fungal**, 2 viral and 1 bacterial. The study and the app use
the fungal ones, with northern anthracnose merged into eyespot (user decision). Images are the unique
photos in `splits_v4.csv`; the last three rows are not used.

| Disease | Cause | Type | Images in v4 | Datasets | Label |
|---|---|---|---|---|---|
| Northern leaf blight | *Exserohilum turcicum* | fungus | 4,081 | `data`, `rlnst`, `cds`, multi-format, `kaggle_corn_leaf_disease`, Northeast Agricultural University | `northern_leaf_blight` |
| Common rust | *Puccinia sorghi* | fungus | 3,524 | `data`, `rlnst`, multi-format, `kaggle_corn_leaf_disease`, `Final corn dataset` | `common_rust` |
| Gray leaf spot | *Cercospora zeae-maydis* | fungus | 2,864 | `data`, `rlnst`, `cds`, multi-format, `Final corn dataset` | `gray_leaf_spot` |
| Northern leaf spot | *Bipolaris zeicola* | fungus | 551 | `cds` only | `northern_leaf_spot` |
| Eyespot / northern anthracnose | *Kabatiella zeae* | fungus | 372 (109 + 263) | multi-format, Northeast Agricultural University | `eyespot` |
| *Healthy* | — | — | 3,825 | all except `cds` | `healthy` |
| Maize streak | Maize streak virus (leafhopper-borne) | virus | — (210 on disk, +1,890 in the RL-NST zip) | multi-format, `rlnst` zip | not used |
| Maize chlorotic mottle | Maize chlorotic mottle virus | virus | — (89 on disk) | `Final corn dataset` | not used |
| Bacterial leaf streak | *Xanthomonas vasicola* pv. *vasculorum* | bacterium | — (290 on disk) | multi-format, `Final corn dataset` | not used |

Notes:
- "Blight" folders (`data/Blight`, `Corn_Leaf_Blight`, `Hawar Daun`) are assumed to be northern leaf
  blight and `Karat Daun` common rust; the dataset pages do not say.
- "Northern anthracnose" (玉米北方炭疽病) usually refers to *Kabatiella zeae*, the cause of eyespot, so it is
  merged into `eyespot`. The dataset's README does not name the pathogen.
- `northern_leaf_spot` comes from a single dataset (CD&S) and `eyespot` from two small ones, so a model
  may partly learn those datasets' look instead of the disease.

**Fungal diseases only in unused datasets** (`china_2`, `xtu502`). Their `image_metadata.csv` already uses these descriptive labels, ready if the classes are added:

| Disease | Cause | New images | Datasets | Label |
|---|---|---|---|---|
| Southern rust | *Puccinia polysora* | 212 (195 + 17) | `china_2`, `xtu502` | `southern_rust` |
| Southern leaf blight | *Bipolaris maydis* | 200 | `china_2` | `southern_leaf_blight` |
| Brown spot | *Physoderma maydis* | 115 | `china_2` | `brown_spot` |
| Curvularia leaf spot | *Curvularia lunata* | 105 | `china_2` | `curvularia_leaf_spot` |

`xtu502` also adds 17 eyespot and 11 Phaeosphaeria leaf spot photos from other sources. `china_2` adds 220 common rust photos from China.

**Label problem in `data` (Kaggle):** `xtu502` holds resized copies of a few Kaggle photos under different labels. 3 photos in Kaggle's `Blight` folder are gray leaf spot (one Iowa State photo clearly so), 2 are Goss's wilt, and 4 of Kaggle's `Common_Rust` photos are labelled southern rust in `xtu502`. These photos are in the v5 training splits.

## Generated files (`data/`)

| File | Rows | What it is |
|---|---|---|
| `splits.csv` | 8,643 | v1 train/dev/test manifest (`kaggle`, `rlnst`, `cds`), merged labels. `models/corn` was trained on it |
| `splits_v2.csv` | 9,307 | v1 plus `maize_multiformat`, append-only. `models/corn_v2` was trained on it |
| `splits_v3.csv` | 12,301 | v2 plus `ndisan`, append-only, merged labels (`leaf_blight`, `leaf_spot`) |
| `splits_v3_by_date.csv` | 12,301 | v3 with every (city, capture date) kept in one set; used for the weather-input comparison |
| **`splits_v4.csv`** | **15,217** | v3 plus `mendeley_bd` and `neau`, append-only (no v3 image moved), **fungal labels**. Newest manifest; `models/corn_fungal` is trained on it |
| `splits_dropped.csv`, `splits_v2_dropped.csv`, `splits_v3_dropped.csv`, `splits_v4_dropped.csv` | 2 / 28 / 34 / 638 | images left out of each manifest and why (exact duplicate, unreadable) |
| `image_metadata.json` | 11 datasets | index linking each dataset's `raw/<folder>/image_metadata.csv` (committed): per image — label, capture date, GPS (rounded to ~110 m), camera, documented site and city. 8 datasets / 16,460 images are in v5; 3 more (`23669328`, `china_2`, `xtu502`; 3,497 images) are marked `"in_manifest": false`. `data_prep.image_metadata.read_metadata()` returns the v5 rows; pass `include_unused=True` for all |
| `weather.csv` | 71 | weather summaries (7 days, 1, 3 and 6 months before capture) per distinct (city, capture date); join on `city` + `capture_date`. 57 rows are for v5 photos; 14 are `china_2` rows, built with `python -m data_prep.weather --include-unused`. **9 of those `china_2` rows (dated 2023-12-06) are unreliable:** all 499 of `china_2`'s 2023 photos carry that one EXIF date across 8 provinces, including rust in December, so it is probably a processing date. `23669328` is left out: most of its leaves were scanned Jan–Apr in Iowa, so they were likely grown indoors. `xtu502` has no dates |

## Other folders

| Folder | What it is |
|---|---|
| `raw/weather_cache/`, `raw/geocode_cache/` | cached Open-Meteo and OpenStreetMap responses, so the weather and city lookups can be regenerated without network calls |

## Rebuilding

Never regenerate an existing manifest (the builder refuses without `--force`): each one records what a
model was trained on. Add data on top of the newest one instead:

```bash
python -m data_prep.build_splits --add NAME=PATH --base data/splits_v4.csv --out data/splits_v5.csv
python -m data_prep.image_metadata --manifest data/splits_v5.csv   # per-folder CSVs + image_metadata.json
python -m data_prep.weather
```

The `--add` list must repeat every source beyond the three defaults (`maize_multiformat`, `ndisan`,
`neau`, `mendeley_bd`); their folder paths are in `docs/data-sources.md`.
