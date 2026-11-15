# Data sources

Where every training, dev and test image comes from. The per-image record is `data/splits.csv`
(source, original folder, label, content hash, near-duplicate group, split). Exact duplicates and
unreadable files that were left out are in `data/splits_dropped.csv`.

Two manifests exist:

| File | Images | What it is |
|---|---|---|
| `data/splits.csv` (v1) | 8,643 | `kaggle`, `rlnst`, `cds`, split 6,914 / 865 / 864. The model in `models/corn` was trained on its train split |
| `data/splits_v2.csv` (v2) | 9,307 | v1 plus `maize_multiformat`, built **append-only** with `--base data/splits.csv`: all 8,643 v1 images keep their v1 split; the 664 new images are split 525 / 67 / 72 (groups of burst shots stay together, so not exactly 80/10/10) |

The images are not in the repository (`data/raw/` is gitignored). **Do not regenerate
`data/splits.csv` or `data/splits_v2.csv`**: they are the record of what each model trained on.
The builder's grouping rule changed after v1 was made, so a fresh build of the same files would move
about 560 images between splits. The builder refuses to overwrite an existing file unless given `--force`.

## Photo metadata

`data/image_metadata.csv` lists, for every image in `data/splits_v2.csv`, the capture date and time, GPS
location (rounded to 3 decimals, about 110 m) and camera taken from the image's embedded EXIF data.
Blank means the image carries none. Regenerate with `python -m data_prep.image_metadata`.

| Source | Images | With capture date | With GPS | Notes |
|---|---|---|---|---|
| `cds` | 1,571 | all | all | one site in west-central Indiana (about 40.5° N, 87.0° W), 15 Jul to 21 Aug 2020, iPhone 11 Pro |
| `kaggle` | 4,186 | 157 | 25 | the rest stripped (3 more carry empty GPS placeholders); the few located ones are mostly Nebraska, two in Kenya |
| `rlnst` | 2,886 | none | none | |
| `maize_multiformat` | 664 | none | none | stripped even in `RAW_Image`; the 6 unreadable HEIC originals were not checked |

## Sources used

| `source` | What it is | Where it came from | Images used | License |
|---|---|---|---|---|
| `kaggle` | Corn or Maize Leaf Disease Dataset: mostly 256×256 PlantVillage-style leaf cutouts, with about 10% field photos (estimated from a 200-image sample). Classes: Blight, Common_Rust, Gray_Leaf_Spot, Healthy | Kaggle `smaranjitghose/corn-or-maize-leaf-disease-dataset` (https://www.kaggle.com/datasets/smaranjitghose/corn-or-maize-leaf-disease-dataset), downloaded with `kaggle datasets download ... --unzip` into `data/raw/data` | 4,186 | not checked |
| `rlnst` | RL-NST: 224×224 leaf crops. Folders Common_rust, Healthy, Leaf_Blight, Leaf_spot (Streak_virus excluded) | `RL-NST(Data).zip` supplied by the user; extracted without the `Streak_virus` folder into `data/raw/rlnst/RL-NST`. Original publisher unknown | 2,886 | unknown |
| `cds` | CD&S (Corn Disease and Severity): full-resolution handheld field photos. Folders gls, nlb, nls in the authors' `train` and `test` splits, which are pooled here | OSF project `s6ru5`, folder `Dataset_Original` (https://osf.io/s6ru5), downloaded through the OSF API into `data/raw/cds`. Paper: Ahmad et al. 2021, https://arxiv.org/abs/2110.12084 | 1,571 | **none stated** on the OSF project; check before redistributing or publishing results |

| `maize_multiformat` | "An Open-Source Multi-Format Maize Leaf Image Dataset": full-resolution phone photos (mostly 3024×4032), 7 classes, many burst shots of the same leaf | Folder supplied by the user at `data/raw/An Open-Source Multi-Format Maize Leaf Image Datas/`; publisher not identified. **Only `RAW_Image/` is used** (see below) | 664 (v2 only) | unknown |

Totals: 8,643 images in v1, 9,307 in v2.

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

Folder names are mapped to labels in `train/labels.py`.

| Source folder | Label | Basis |
|---|---|---|
| Kaggle `Healthy`, RL-NST `Healthy` | `healthy` | folder name |
| Kaggle `Common_Rust`, RL-NST `Common_rust` | `common_rust` | folder name |
| Kaggle `Blight`, RL-NST `Leaf_Blight`, CD&S `nlb` | `leaf_blight` | assumed to be northern corn leaf blight; RL-NST confirmed by the user and cross-checked with a model and Claude |
| Kaggle `Gray_Leaf_Spot`, RL-NST `Leaf_spot`, CD&S `gls` and `nls` | `leaf_spot` | gray and northern leaf spot merged on purpose; RL-NST `Leaf_spot` as gray leaf spot confirmed by the user, cross-checked (weakly) |
| `maize_multiformat` `Healthy_Leaf`, `Common_Rust`, `Corn_Leaf_Blight`, `Gray_Leaf_Spot` | `healthy`, `common_rust`, `leaf_blight`, `leaf_spot` | folder name; "Corn Leaf Blight" assumed to be a leaf blight |
| `maize_multiformat` `Eyespot` | `leaf_spot` | user decision: eyespot (Kabatiella) counts as a leaf spot |
| RL-NST `Streak_virus`, `maize_multiformat` `Maize_Streak_Disease`, `Bacterial_Leaf_Streak` | not used | excluded on purpose, and unmapped so they can never be trained on |

## Looked at and not used

| Candidate | Why it was not used |
|---|---|
| Hugging Face `Prachi1234/corn-leaf-disease` | byte-identical to the Kaggle dataset (all 4,186 unique images) |
| Hugging Face `Project-AgML/corn_maize_leaf_disease` | byte-identical to the Kaggle dataset, labels included |
| Kaggle `yusufmurtaza01/corn-leaf-diseases` (YOLO detection format) | 3,992 of its 4,027 images are byte-identical to the Kaggle set |
| OpenDataLab `OpenDataLab/CD_and_S` | the page could not be read; the OSF source above was used instead |
| Kaggle notebook `fannysalsabila/plant-disease-classification-using-resnet` | code only; its listed data source is a grape dataset |

## Known data issues

- **Two images carry conflicting labels.** The Kaggle dataset has two exact duplicate files that sit in both the `Blight` and the `Gray_Leaf_Spot` folders (`Corn_Blight (77).jpg` / `Corn_Gray_Spot (19).jpg`, and `Corn_Blight (1104).jpg` / `Corn_Gray_Spot (564).jpg`). The builder keeps the copy that sorts first (the Blight one) and records the other in `splits_dropped.csv`. One of the two labels is wrong.
- **Near-duplicates are only linked within one label.** Smooth photos of different classes can have close hashes; without this rule a 10-photo healthy burst was chained to leaf-spot images. v1 was built before this rule and contains three small cross-label groups; it was left unchanged because the model in `models/corn` was trained on it.
- **Near-duplicate grouping is narrow.** Images are grouped when their 64-bit difference hashes are within 4 bits, which catches near-identical files (43 images fall in such groups, the largest of size 3). It does not catch rotated, flipped or cropped copies of the same photo, nor photos of the same leaf or plant taken separately. Leakage between splits through such copies cannot be ruled out.
- **CD&S photos come from one field campaign**, so photos of the same plants can sit in different splits.
- **Few field photos for healthy leaves and common rust.** All healthy photos are leaf crops (Kaggle, RL-NST).

## Rebuilding

1. Download the three sources into the locations in the table above.
2. To add a source without moving any existing image, build a new file on top of the newest manifest (about a minute): `python -m data_prep.build_splits --add "maize_multiformat=data/raw/An Open-Source Multi-Format Maize Leaf Image Datas/maize (corn) leaf images/RAW_Image" --base data/splits.csv --out data/splits_v2.csv`. v2 rebuilt this way reproduces byte-for-byte. A completely fresh split (new file name, seed 0) is `python -m data_prep.build_splits --out data/splits_new.csv`; models trained on another manifest cannot be scored on it.
3. `python -m train.train --manifest data/splits.csv --out models/corn --epochs 5`, then `python -m train.evaluate --manifest data/splits.csv --model models/corn --split dev` (or `test`).
