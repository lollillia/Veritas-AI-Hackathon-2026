# Data sources

Where every training, dev and test image comes from. The per-image record is `data/splits.csv`
(source, original folder, label, content hash, near-duplicate group, split). Exact duplicates and
unreadable files that were left out are in `data/splits_dropped.csv`.

Two manifests exist:

| File | Images | What it is |
|---|---|---|
| `data/splits.csv` (v1) | 8,643 | `kaggle`, `rlnst`, `cds`, split 6,914 / 865 / 864. The model in `models/corn` was trained on its train split |
| `data/splits_v4.csv` (v4) | 15,217 | v3 plus `mendeley_bd` (2,060 unique) and `neau` (856), append-only with `--base data/splits_v3.csv`: no v3 image moved. Labels are the individual fungal diseases (`northern_leaf_blight`, `gray_leaf_spot`, `northern_leaf_spot`, `eyespot`, `common_rust`, `healthy`); v1–v3 store the earlier merged labels |
| `data/splits_v3.csv` (v3) | 12,301 | v2 plus `ndisan` (2,994 phone photos), built append-only with `--base data/splits_v2.csv`: all 9,307 v2 images keep their split; the new images are split 2,436 / 284 / 274 with each leaf's repeated shots kept together |
| `data/splits_v2.csv` (v2) | 9,307 | v1 plus `maize_multiformat`, built **append-only** with `--base data/splits.csv`: all 8,643 v1 images keep their v1 split; the 664 new images are split 525 / 67 / 72 (groups of burst shots stay together, so not exactly 80/10/10) |

The images are not in the repository (`data/raw/` is gitignored). **Do not regenerate
`data/splits.csv` or `data/splits_v2.csv`**: they are the record of what each model trained on.
The builder's grouping rule changed after v1 was made, so a fresh build of the same files would move
about 560 images between splits. The builder refuses to overwrite an existing file unless given `--force`.

## Photo metadata

`data/image_metadata.csv` lists, for every image in the newest manifest (`data/splits_v3.csv`):
`capture_date` (EXIF capture time reduced to its date), `gps_lat` / `gps_lon` (rounded to 3 decimals,
about 110 m), `camera`, `site` (the dataset-level collection site where the source documents one) and
`city`. The city is reverse-geocoded from the photo's GPS with OpenStreetMap Nominatim at city level,
or, when the photo has no GPS, from the documented site's coordinates. Rural points can resolve to a
county instead of a town (CD&S: 1,412 photos "West Lafayette", 159 "Tippecanoe County" for the same
field). Blank means unknown. Regenerate with `python -m data_prep.image_metadata` (lookups cached in
`data/raw/geocode_cache/`).

| Source | Images | With capture date | With GPS | Notes |
|---|---|---|---|---|
| `cds` | 1,571 | all | all | one site in west-central Indiana (about 40.5° N, 87.0° W), 15 Jul to 21 Aug 2020, iPhone 11 Pro |
| `kaggle` | 4,186 | 157 | 25 | the rest stripped (3 more carry empty GPS placeholders); the few located ones are mostly Nebraska, two in Kenya |
| `rlnst` | 2,886 | none | none | |
| `maize_multiformat` | 664 | none | none | stripped even in `RAW_Image`; the 6 unreadable HEIC originals were not checked |
| `ndisan` | 2,994 | all | none (empty placeholders) | vivo V2029, 11 days from 11 Apr to 30 Jul 2023; `site` from the dataset page; the camera clock may be off (EXIF times 05–08 and 15–17 vs the page's 12:00–14:00 WIB) |

## Weather before capture

`data/weather.csv` summarises the weather leading up to each distinct (`city`, `capture_date`) in
`data/image_metadata.csv`: 40 rows for 19 cities, covering all 4,590 images that have a city and a date
(CD&S, the 25 located Kaggle photos, and the `ndisan` photos via their documented site). Join on `city` and
`capture_date`. Regenerate with `python -m data_prep.weather` (one request per city; cached in
`data/raw/weather_cache/`).

For each window — `7d` (7 days), `1m` (30 days), `3m` (90 days), `6m` (182 days), all ending the day
before the capture date — the file gives mean daily temperature, lowest minimum, highest maximum, total
precipitation, rainy days (1 mm or more), mean relative humidity and the number of days with data.

- **Source:** Open-Meteo historical weather archive (https://open-meteo.com), ERA5 reanalysis: modelled
  daily values on a grid of roughly 10–25 km, queried at the city's centre point, not a station at the
  field. Reanalysis data tends to count more light-rain days than rain gauges do.
- **Dates:** windows depend only on the capture date, so the `ndisan` camera-clock doubt (hours) does not
  matter unless the date itself is wrong.
- **Examples:** Sampang before 11 Apr 2023 (end of the wet season): 1,434 mm over 6 months, 7 rainy days in
  the last week; before 30 Jul 2023 (dry season): 0.4 mm in the last week. West Lafayette before
  19 Aug 2020: 1.7 mm in the last week, 92 mm in the last month.

## Sources used

| `source` | What it is | Where it came from | Images used | License |
|---|---|---|---|---|
| `kaggle` | Corn or Maize Leaf Disease Dataset: mostly 256×256 PlantVillage-style leaf cutouts, with about 10% field photos (estimated from a 200-image sample). Classes: Blight, Common_Rust, Gray_Leaf_Spot, Healthy | Kaggle `smaranjitghose/corn-or-maize-leaf-disease-dataset` (https://www.kaggle.com/datasets/smaranjitghose/corn-or-maize-leaf-disease-dataset), downloaded with `kaggle datasets download ... --unzip` into `data/raw/data` | 4,186 | not checked |
| `rlnst` | RL-NST: 224×224 leaf crops. Folders Common_rust, Healthy, Leaf_Blight, Leaf_spot (Streak_virus excluded) | `RL-NST(Data).zip` supplied by the user; extracted without the `Streak_virus` folder into `data/raw/rlnst/RL-NST`. Original publisher unknown | 2,886 | unknown |
| `cds` | CD&S (Corn Disease and Severity): full-resolution handheld field photos. Folders gls, nlb, nls in the authors' `train` and `test` splits, which are pooled here | OSF project `s6ru5`, folder `Dataset_Original` (https://osf.io/s6ru5), downloaded through the OSF API into `data/raw/cds`. Paper: Ahmad et al. 2021, https://arxiv.org/abs/2110.12084 | 1,571 | **none stated** on the OSF project; check before redistributing or publishing results |

| `maize_multiformat` | "An Open-Source Multi-Format Maize Leaf Image Dataset": full-resolution phone photos (mostly 3024×4032), 7 classes, many burst shots of the same leaf | Folder supplied by the user at `data/raw/An Open-Source Multi-Format Maize Leaf Image Datas/`; publisher not identified. **Only `RAW_Image/` is used** (see below) | 664 (v2 only) | unknown |

| `ndisan` | Kaggle "Corn Leaf Disease" (`ndisan/corn-leaf-disease`): 3120×4160 phone photos of single leaves on white paper, 5 shots per leaf. **Only `Daun Sehat`, `Hawar Daun`, `Karat Daun` are used** | https://www.kaggle.com/datasets/ndisan/corn-leaf-disease, downloaded by the user into `data/raw/kaggle_corn_leaf_disease`. Collected in Polagan Village, Sampang Regency, Madura, East Java, Indonesia (per the page) | 2,994 (v3 only) | MIT (per the page) |

| `mendeley_bd` | Mendeley Data "Seasonal Corn Leaf Disease Dataset" (vy629dngm8 v1): field photos from several phones, Dec 2022 – Feb 2024 (EXIF dates on 1,203, GPS on 591). **Only `Healthy`, `Gray_leaf_spot`, `Common_rust` used** | https://data.mendeley.com/datasets/vy629dngm8/1 (confirmed by the user), folder `data/raw/Final corn dataset`. Gurudaspur, Natore, Bangladesh | 2,060 (v4 only; 604 duplicate files inside the dataset dropped) | CC BY 4.0 |
| `neau` | Northeast Agricultural University maize leaves: field photos resized to 256×256 (EXIF dates 11 Jul – 11 Aug 2021, Huawei phone) | folder supplied by the user (`data/raw/Northeast Agricultural University`, details in its `README.txt`). Eastern experimental field, Harbin, China | 856 (v4 only) | unknown |

Totals: 8,643 images in v1, 9,307 in v2, 12,301 in v3, 15,217 in v4.

### What was left out of `maize_multiformat`

The folder holds 5,789 files, but they are mostly copies of the same ~1,000 photos:

| Sub-folder | Files | Why not used |
|---|---|---|
| `Converted raw Images/` | 994 JPEG + 6 HEIC | resized copies of `RAW_Image` (same file names) |
| `Augmented_Images/` | 2,982 | 224×224 augmented copies (`org_`, `aug_N_`); rotations and crops would leak across splits because near-duplicate grouping cannot catch them |
| `Annotation/` | 793 JPEG + 6 HEIC + label CSVs | a subset of the originals with box annotations |
| `RAW_Image/Maize_Streak_Disease/` | 210 | excluded on purpose (streak virus) |
| `RAW_Image/Bacterial_Leaf_Streak/` | 100 | no label for it |
| `RAW_Image/Healthy_Leaf/*.HEIC` | 6 | HEIC cannot be read without an extra library; listed as unreadable in `splits_v2_dropped.csv` |
| exact duplicate files inside `RAW_Image/` | 20 | consecutive copies (e.g. `IMG_608`–`IMG_610` of `IMG_607`); listed in `splits_v2_dropped.csv` |

## Label mapping

Folder names are mapped to labels in `train/labels.py`. Since v4 the labels are the individual fungal
diseases; in the table below, rows written for v1–v3 show the label they map to today (blights to
`northern_leaf_blight`, gray leaf spots to `gray_leaf_spot`, `nls` to `northern_leaf_spot`).

| Source folder | Label | Basis |
|---|---|---|
| Kaggle `Healthy`, RL-NST `Healthy` | `healthy` | folder name |
| Kaggle `Common_Rust`, RL-NST `Common_rust` | `common_rust` | folder name |
| Kaggle `Blight`, RL-NST `Leaf_Blight`, CD&S `nlb` | `leaf_blight` | assumed to be northern corn leaf blight; RL-NST confirmed by the user and cross-checked with a model and Claude |
| Kaggle `Gray_Leaf_Spot`, RL-NST `Leaf_spot`, CD&S `gls` and `nls` | `leaf_spot` | gray and northern leaf spot merged on purpose; RL-NST `Leaf_spot` as gray leaf spot confirmed by the user, cross-checked (weakly) |
| `maize_multiformat` `Healthy_Leaf`, `Common_Rust`, `Corn_Leaf_Blight`, `Gray_Leaf_Spot` | `healthy`, `common_rust`, `leaf_blight`, `leaf_spot` | folder name; "Corn Leaf Blight" assumed to be a leaf blight |
| `ndisan` `Daun Sehat`, `Hawar Daun`, `Karat Daun` | `healthy`, `leaf_blight`, `common_rust` | Indonesian folder names; the dataset page lists healthy, leaf blight, leaf spot and leaf rust ("karat" assumed to be common rust) |
| `ndisan` `Bercak Daun` | not used | copies and 180° rotations of Kaggle `smaranjitghose` images already in `kaggle` |
| `maize_multiformat` `Eyespot`, `neau` `maize northern anthracnose` | `eyespot` | user decisions: eyespot is its own label; northern anthracnose (usually *Kabatiella zeae*) is merged into it |
| `mendeley_bd` `Healthy`, `Gray_leaf_spot`, `Common_rust`; `neau` `healthy maize leaves`, `maize northern leaf blight` | `healthy`, `gray_leaf_spot`, `common_rust`, `healthy`, `northern_leaf_blight` | folder names |
| `mendeley_bd` `Bacterial Leaf Streak`, `Maize Chlorotic Mottle Virus` | not used | not fungal |
| RL-NST `Streak_virus`, `maize_multiformat` `Maize_Streak_Disease`, `Bacterial_Leaf_Streak` | not used | excluded on purpose, and unmapped so they can never be trained on |

## Notes on `ndisan` (`data/raw/kaggle_corn_leaf_disease`)

- **Source:** https://www.kaggle.com/datasets/ndisan/corn-leaf-disease (confirmed by the user; folder names, file names, 4,000 images and 5.49 GB total all match). License on the page: MIT. The authors ask to be credited in academic use.
- **Where:** per the page, "a corn field with an area of approximately 10 hectares owned by the Department of Agriculture in Polagan Village, Sampang District" (Sampang Regency, Madura, East Java, Indonesia). The photos carry no usable GPS (placeholder tags only).
- **How:** 16 MP phone camera (EXIF: vivo V2029, 3120×4160), white paper behind each leaf, **5 photos of each leaf**. Labels by the Sampang Regency agriculture department, validated by the regional plant-pest observation (POPT) coordinator.
- **When:** EXIF capture dates on 11 days between 11 April and 30 July 2023. The page says shooting was at 12:00–14:00 WIB, but EXIF times are 05:00–08:00 and 15:00–17:00, so the camera clock is probably not reliable.
- **Folders (Indonesian):** `Daun Sehat` healthy, `Hawar Daun` leaf blight, `Karat Daun` leaf rust (the 3,000 phone photos, 1,000 each), and `Bercak Daun` leaf spot.
- **Do not use `Bercak Daun`:** the page says the data was combined with the Kaggle `smaranjitghose` corn dataset; 574 of these files are byte-identical to images already in our `kaggle` source and most of the rest are those images rotated 180° (`augmented_…`), which near-duplicate grouping cannot catch.
- **Repeated shots:** the 5 shots of each leaf are grouped by their filename timestamps (`IMG_20230728_171448`): shots within 20 seconds of the previous one in the same folder form one group, so they stay in one split. Files ending in ` (1)` are not copies; every file has its own timestamp (the suffix comes from merging folders). 6 exact duplicate files were dropped.

## Looked at and not used

| Candidate | Why it was not used |
|---|---|
| Hugging Face `Prachi1234/corn-leaf-disease` | byte-identical to the Kaggle dataset (all 4,186 unique images) |
| Hugging Face `Project-AgML/corn_maize_leaf_disease` | byte-identical to the Kaggle dataset, labels included |
| Kaggle `yusufmurtaza01/corn-leaf-diseases` (YOLO detection format) | 3,992 of its 4,027 images are byte-identical to the Kaggle set |
| Kaggle `taufiqnoviant/daun-jagung` | a different set (3 classes, Kediri Regency, East Java; CC BY-NC-ND 4.0); not downloaded |
| Mendeley `vy629dngm8` v1 ("Seasonal Corn Leaf Disease Dataset", Natore, Bangladesh; CC BY 4.0) | link checked; matches no local folder |
| OpenDataLab `OpenDataLab/CD_and_S` | the page could not be read; the OSF source above was used instead |
| Kaggle notebook `fannysalsabila/plant-disease-classification-using-resnet` | code only; its listed data source is a grape dataset |

## Known data issues

- **Two images carry conflicting labels.** The Kaggle dataset has two exact duplicate files that sit in both the `Blight` and the `Gray_Leaf_Spot` folders (`Corn_Blight (77).jpg` / `Corn_Gray_Spot (19).jpg`, and `Corn_Blight (1104).jpg` / `Corn_Gray_Spot (564).jpg`). The builder keeps the copy that sorts first (the Blight one) and records the other in `splits_dropped.csv`. One of the two labels is wrong.
- **Grouping rules changed over time.** v1 linked any images whose 64-bit difference hashes were within 4 bits; later builds link only images with the same label whose pixels also correlate at 0.95 or more (plain leaves with similar framing produced huge false groups, which dropped 1,256 new photos in a first v3 attempt), and add groups of repeated shots by filename timestamp. Because of this, a fresh rebuild of v2 gives the same splits but different `group_id` values, and future fresh builds no longer group repeated shots of the same leaf in datasets without timestamps (`maize_multiformat`, `rlnst`). The committed manifests are kept as they are and new data is always added with `--base`.
- **Near-duplicates are only linked within one label.** Smooth photos of different classes can have close hashes; without this rule a 10-photo healthy burst was chained to leaf-spot images. v1 was built before this rule and contains three small cross-label groups; it was left unchanged because the model in `models/corn` was trained on it.
- **Near-duplicate grouping is narrow.** Images are grouped when their 64-bit difference hashes are within 4 bits, which catches near-identical files (43 images fall in such groups, the largest of size 3). It does not catch rotated, flipped or cropped copies of the same photo, nor photos of the same leaf or plant taken separately. Leakage between splits through such copies cannot be ruled out.
- **CD&S photos come from one field campaign**, so photos of the same plants can sit in different splits.
- **Few field photos for healthy leaves and common rust.** All healthy photos are leaf crops (Kaggle, RL-NST).

## Rebuilding

1. Download the three sources into the locations in the table above.
2. To add a source without moving any existing image, build a new file on top of the newest manifest (about a minute): `python -m data_prep.build_splits --add "maize_multiformat=data/raw/An Open-Source Multi-Format Maize Leaf Image Datas/maize (corn) leaf images/RAW_Image" --base data/splits.csv --out data/splits_v2.csv`. (A rebuild with the current builder reproduces v2's splits but not its `group_id` values; see Known data issues.) A completely fresh split (new file name, seed 0) is `python -m data_prep.build_splits --out data/splits_new.csv`; models trained on another manifest cannot be scored on it.
3. `python -m train.train --manifest data/splits.csv --out models/corn --epochs 5`, then `python -m train.evaluate --manifest data/splits.csv --model models/corn --split dev` (or `test`).
