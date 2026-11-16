# Data used by each model

Which images each model in `models/` was trained and tested on, and where to download them. Every image's
path, label and split (train / dev / test) is recorded in the committed manifest named for each model, so the
exact data can be rebuilt. Image files are not in git (`data/raw/` is about 30 GB).

**One download for everything:** the whole `data/raw` folder (without the unused Iowa State scans) is stored as
a private Hugging Face dataset; `scripts/train_from_hf.sh YOUR_USERNAME/veritas-corn-raw` downloads it. The
original sources are listed below for attribution and re-download.

## Models at a glance

| Model | Manifest | Labels | Inputs | Images (train / dev / test) | Datasets | Download links |
|---|---|---|---|---|---|---|
| `corn` | `data/splits.csv` | 4 | image only | 6,914 / 865 / 864 | `cds`, `kaggle`, `rlnst` | cds: https://osf.io/s6ru5<br>kaggle: https://www.kaggle.com/datasets/smaranjitghose/corn-or-maize-leaf-disease-dataset<br>rlnst: https://zenodo.org/records/19355313 |
| `corn_v2` | `data/splits_v2.csv` | 4 | image only | 7,439 / 932 / 936 | `cds`, `kaggle`, `maize_multiformat`, `rlnst` | cds: https://osf.io/s6ru5<br>kaggle: https://www.kaggle.com/datasets/smaranjitghose/corn-or-maize-leaf-disease-dataset<br>maize_multiformat: https://www.scidb.cn/en/detail?dataSetId=19536c73f6d74946a212719a94f53ab3<br>rlnst: https://zenodo.org/records/19355313 |
| `fusion_v3_image` | `data/splits_v3.csv` | 4 | image only | 9,875 / 1,216 / 1,210 | `cds`, `kaggle`, `maize_multiformat`, `ndisan`, `rlnst` | cds: https://osf.io/s6ru5<br>kaggle: https://www.kaggle.com/datasets/smaranjitghose/corn-or-maize-leaf-disease-dataset<br>maize_multiformat: https://www.scidb.cn/en/detail?dataSetId=19536c73f6d74946a212719a94f53ab3<br>ndisan: https://www.kaggle.com/datasets/ndisan/corn-leaf-disease<br>rlnst: https://zenodo.org/records/19355313 |
| `fusion_v3_weather` | `data/splits_v3.csv` | 4 | image + `data/weather.csv` | 9,875 / 1,216 / 1,210 | `cds`, `kaggle`, `maize_multiformat`, `ndisan`, `rlnst` | cds: https://osf.io/s6ru5<br>kaggle: https://www.kaggle.com/datasets/smaranjitghose/corn-or-maize-leaf-disease-dataset<br>maize_multiformat: https://www.scidb.cn/en/detail?dataSetId=19536c73f6d74946a212719a94f53ab3<br>ndisan: https://www.kaggle.com/datasets/ndisan/corn-leaf-disease<br>rlnst: https://zenodo.org/records/19355313 |
| `fusion_v3_by_date_image` | `data/splits_v3_by_date.csv` | 4 | image only | 9,847 / 1,305 / 1,149 | `cds`, `kaggle`, `maize_multiformat`, `ndisan`, `rlnst` | cds: https://osf.io/s6ru5<br>kaggle: https://www.kaggle.com/datasets/smaranjitghose/corn-or-maize-leaf-disease-dataset<br>maize_multiformat: https://www.scidb.cn/en/detail?dataSetId=19536c73f6d74946a212719a94f53ab3<br>ndisan: https://www.kaggle.com/datasets/ndisan/corn-leaf-disease<br>rlnst: https://zenodo.org/records/19355313 |
| `fusion_v3_by_date_weather` | `data/splits_v3_by_date.csv` | 4 | image + `data/weather.csv` | 9,847 / 1,305 / 1,149 | `cds`, `kaggle`, `maize_multiformat`, `ndisan`, `rlnst` | cds: https://osf.io/s6ru5<br>kaggle: https://www.kaggle.com/datasets/smaranjitghose/corn-or-maize-leaf-disease-dataset<br>maize_multiformat: https://www.scidb.cn/en/detail?dataSetId=19536c73f6d74946a212719a94f53ab3<br>ndisan: https://www.kaggle.com/datasets/ndisan/corn-leaf-disease<br>rlnst: https://zenodo.org/records/19355313 |
| `corn_fungal` | `data/splits_v4.csv` | 6 | image only | 12,181 / 1,528 / 1,508 | `cds`, `kaggle`, `maize_multiformat`, `mendeley_bd`, `ndisan`, `neau`, `rlnst` | cds: https://osf.io/s6ru5<br>kaggle: https://www.kaggle.com/datasets/smaranjitghose/corn-or-maize-leaf-disease-dataset<br>maize_multiformat: https://www.scidb.cn/en/detail?dataSetId=19536c73f6d74946a212719a94f53ab3<br>mendeley_bd: https://data.mendeley.com/datasets/vy629dngm8/1<br>ndisan: https://www.kaggle.com/datasets/ndisan/corn-leaf-disease<br>neau: https://figshare.com/articles/dataset/data_zip/21626348<br>rlnst: https://zenodo.org/records/19355313 |
| `corn_fungal_v5` | `data/splits_v5.csv` | 7 | image only | 13,174 / 1,654 / 1,632 | `cds`, `kaggle`, `maize_multiformat`, `mendeley_bd`, `ndisan`, `neau`, `rlnst`, `up_field` | cds: https://osf.io/s6ru5<br>kaggle: https://www.kaggle.com/datasets/smaranjitghose/corn-or-maize-leaf-disease-dataset<br>maize_multiformat: https://www.scidb.cn/en/detail?dataSetId=19536c73f6d74946a212719a94f53ab3<br>mendeley_bd: https://data.mendeley.com/datasets/vy629dngm8/1<br>ndisan: https://www.kaggle.com/datasets/ndisan/corn-leaf-disease<br>neau: https://figshare.com/articles/dataset/data_zip/21626348<br>rlnst: https://zenodo.org/records/19355313<br>up_field: https://researchdata.up.ac.za/articles/dataset/Diseases_of_maize_in_the_field/20237613 |
| `corn_fungal_v5_bgaug` | `data/splits_v5.csv` | 7 | image only | 13,174 / 1,654 / 1,632 | `cds`, `kaggle`, `maize_multiformat`, `mendeley_bd`, `ndisan`, `neau`, `rlnst`, `up_field` | cds: https://osf.io/s6ru5<br>kaggle: https://www.kaggle.com/datasets/smaranjitghose/corn-or-maize-leaf-disease-dataset<br>maize_multiformat: https://www.scidb.cn/en/detail?dataSetId=19536c73f6d74946a212719a94f53ab3<br>mendeley_bd: https://data.mendeley.com/datasets/vy629dngm8/1<br>ndisan: https://www.kaggle.com/datasets/ndisan/corn-leaf-disease<br>neau: https://figshare.com/articles/dataset/data_zip/21626348<br>rlnst: https://zenodo.org/records/19355313<br>up_field: https://researchdata.up.ac.za/articles/dataset/Diseases_of_maize_in_the_field/20237613 |
| `corn_fungal_v6` | `data/splits_v6.csv` | 11 | image only | 13,894 / 1,748 / 1,719 | `cds`, `china_2`, `kaggle`, `maize_multiformat`, `mendeley_bd`, `ndisan`, `neau`, `rlnst`, `up_field`, `xtu502` | cds: https://osf.io/s6ru5<br>china_2: https://www.scidb.cn/en/detail?dataSetId=b17f6aac10ac4d7ba247862454e2ef45<br>kaggle: https://www.kaggle.com/datasets/smaranjitghose/corn-or-maize-leaf-disease-dataset<br>maize_multiformat: https://www.scidb.cn/en/detail?dataSetId=19536c73f6d74946a212719a94f53ab3<br>mendeley_bd: https://data.mendeley.com/datasets/vy629dngm8/1<br>ndisan: https://www.kaggle.com/datasets/ndisan/corn-leaf-disease<br>neau: https://figshare.com/articles/dataset/data_zip/21626348<br>rlnst: https://zenodo.org/records/19355313<br>up_field: https://researchdata.up.ac.za/articles/dataset/Diseases_of_maize_in_the_field/20237613<br>xtu502: https://github.com/xtu502/maize-disease-identification |
| `corn_fungal_v6_bgaug` | `data/splits_v6.csv` | 11 | image only | 13,894 / 1,748 / 1,719 | `cds`, `china_2`, `kaggle`, `maize_multiformat`, `mendeley_bd`, `ndisan`, `neau`, `rlnst`, `up_field`, `xtu502` | cds: https://osf.io/s6ru5<br>china_2: https://www.scidb.cn/en/detail?dataSetId=b17f6aac10ac4d7ba247862454e2ef45<br>kaggle: https://www.kaggle.com/datasets/smaranjitghose/corn-or-maize-leaf-disease-dataset<br>maize_multiformat: https://www.scidb.cn/en/detail?dataSetId=19536c73f6d74946a212719a94f53ab3<br>mendeley_bd: https://data.mendeley.com/datasets/vy629dngm8/1<br>ndisan: https://www.kaggle.com/datasets/ndisan/corn-leaf-disease<br>neau: https://figshare.com/articles/dataset/data_zip/21626348<br>rlnst: https://zenodo.org/records/19355313<br>up_field: https://researchdata.up.ac.za/articles/dataset/Diseases_of_maize_in_the_field/20237613<br>xtu502: https://github.com/xtu502/maize-disease-identification |
| `corn_fungal_v6_aug2` | `data/splits_v6.csv` | 11 | image only | 13,894 / 1,748 / 1,719 | `cds`, `china_2`, `kaggle`, `maize_multiformat`, `mendeley_bd`, `ndisan`, `neau`, `rlnst`, `up_field`, `xtu502` | cds: https://osf.io/s6ru5<br>china_2: https://www.scidb.cn/en/detail?dataSetId=b17f6aac10ac4d7ba247862454e2ef45<br>kaggle: https://www.kaggle.com/datasets/smaranjitghose/corn-or-maize-leaf-disease-dataset<br>maize_multiformat: https://www.scidb.cn/en/detail?dataSetId=19536c73f6d74946a212719a94f53ab3<br>mendeley_bd: https://data.mendeley.com/datasets/vy629dngm8/1<br>ndisan: https://www.kaggle.com/datasets/ndisan/corn-leaf-disease<br>neau: https://figshare.com/articles/dataset/data_zip/21626348<br>rlnst: https://zenodo.org/records/19355313<br>up_field: https://researchdata.up.ac.za/articles/dataset/Diseases_of_maize_in_the_field/20237613<br>xtu502: https://github.com/xtu502/maize-disease-identification |

## Data per manifest

Images per dataset in each manifest (train / dev / test). Models sharing a manifest used exactly the same images.

### `data/splits.csv`: `corn`

Labels: `common_rust`, `healthy`, `leaf_blight`, `leaf_spot`.

| Dataset (`source`) | Train | Dev | Test | Total |
|---|---|---|---|---|
| `cds` | 1,256 | 158 | 157 | 1,571 |
| `kaggle` | 3,350 | 419 | 417 | 4,186 |
| `rlnst` | 2,308 | 288 | 290 | 2,886 |
| **Total** | **6,914** | **865** | **864** | **8,643** |

Left out (duplicates, unreadable, excluded): 2 images, listed with reasons in `data/splits_dropped.csv`.

### `data/splits_v2.csv`: `corn_v2`

Labels: `common_rust`, `healthy`, `leaf_blight`, `leaf_spot`.

| Dataset (`source`) | Train | Dev | Test | Total |
|---|---|---|---|---|
| `cds` | 1,256 | 158 | 157 | 1,571 |
| `kaggle` | 3,350 | 419 | 417 | 4,186 |
| `maize_multiformat` | 525 | 67 | 72 | 664 |
| `rlnst` | 2,308 | 288 | 290 | 2,886 |
| **Total** | **7,439** | **932** | **936** | **9,307** |

Left out (duplicates, unreadable, excluded): 28 images, listed with reasons in `data/splits_v2_dropped.csv`.

### `data/splits_v3.csv`: `fusion_v3_image`, `fusion_v3_weather`

Labels: `common_rust`, `healthy`, `leaf_blight`, `leaf_spot`.

| Dataset (`source`) | Train | Dev | Test | Total |
|---|---|---|---|---|
| `cds` | 1,256 | 158 | 157 | 1,571 |
| `kaggle` | 3,350 | 419 | 417 | 4,186 |
| `maize_multiformat` | 525 | 67 | 72 | 664 |
| `ndisan` | 2,436 | 284 | 274 | 2,994 |
| `rlnst` | 2,308 | 288 | 290 | 2,886 |
| **Total** | **9,875** | **1,216** | **1,210** | **12,301** |

Left out (duplicates, unreadable, excluded): 34 images, listed with reasons in `data/splits_v3_dropped.csv`.

### `data/splits_v3_by_date.csv`: `fusion_v3_by_date_image`, `fusion_v3_by_date_weather`

Labels: `common_rust`, `healthy`, `leaf_blight`, `leaf_spot`.

| Dataset (`source`) | Train | Dev | Test | Total |
|---|---|---|---|---|
| `cds` | 1,230 | 117 | 224 | 1,571 |
| `kaggle` | 3,351 | 418 | 417 | 4,186 |
| `maize_multiformat` | 525 | 67 | 72 | 664 |
| `ndisan` | 2,433 | 415 | 146 | 2,994 |
| `rlnst` | 2,308 | 288 | 290 | 2,886 |
| **Total** | **9,847** | **1,305** | **1,149** | **12,301** |

Same images as `data/splits_v3.csv`, re-split so every (city, capture date) is in one set.

### `data/splits_v4.csv`: `corn_fungal`

Labels: `common_rust`, `eyespot`, `gray_leaf_spot`, `healthy`, `northern_leaf_blight`, `northern_leaf_spot`.

| Dataset (`source`) | Train | Dev | Test | Total |
|---|---|---|---|---|
| `cds` | 1,256 | 158 | 157 | 1,571 |
| `kaggle` | 3,350 | 419 | 417 | 4,186 |
| `maize_multiformat` | 525 | 67 | 72 | 664 |
| `mendeley_bd` | 1,621 | 226 | 213 | 2,060 |
| `ndisan` | 2,436 | 284 | 274 | 2,994 |
| `neau` | 685 | 86 | 85 | 856 |
| `rlnst` | 2,308 | 288 | 290 | 2,886 |
| **Total** | **12,181** | **1,528** | **1,508** | **15,217** |

Left out (duplicates, unreadable, excluded): 638 images, listed with reasons in `data/splits_v4_dropped.csv`.

### `data/splits_v5.csv`: `corn_fungal_v5`, `corn_fungal_v5_bgaug`

Labels: `common_rust`, `eyespot`, `gray_leaf_spot`, `healthy`, `northern_leaf_blight`, `northern_leaf_spot`, `phaeosphaeria_leaf_spot`.

| Dataset (`source`) | Train | Dev | Test | Total |
|---|---|---|---|---|
| `cds` | 1,256 | 158 | 157 | 1,571 |
| `kaggle` | 3,350 | 419 | 417 | 4,186 |
| `maize_multiformat` | 525 | 67 | 72 | 664 |
| `mendeley_bd` | 1,621 | 226 | 213 | 2,060 |
| `ndisan` | 2,436 | 284 | 274 | 2,994 |
| `neau` | 685 | 86 | 85 | 856 |
| `rlnst` | 2,308 | 288 | 290 | 2,886 |
| `up_field` | 993 | 126 | 124 | 1,243 |
| **Total** | **13,174** | **1,654** | **1,632** | **16,460** |

Left out (duplicates, unreadable, excluded): 653 images, listed with reasons in `data/splits_v5_dropped.csv`.

### `data/splits_v6.csv`: `corn_fungal_v6`, `corn_fungal_v6_bgaug`, `corn_fungal_v6_aug2`

Labels: `brown_spot`, `common_rust`, `curvularia_leaf_spot`, `eyespot`, `gray_leaf_spot`, `healthy`, `northern_leaf_blight`, `northern_leaf_spot`, `phaeosphaeria_leaf_spot`, `southern_leaf_blight`, `southern_rust`.

| Dataset (`source`) | Train | Dev | Test | Total |
|---|---|---|---|---|
| `cds` | 1,256 | 158 | 157 | 1,571 |
| `china_2` | 668 | 85 | 82 | 835 |
| `kaggle` | 3,346 | 419 | 416 | 4,181 |
| `maize_multiformat` | 525 | 67 | 72 | 664 |
| `mendeley_bd` | 1,621 | 226 | 213 | 2,060 |
| `ndisan` | 2,436 | 284 | 274 | 2,994 |
| `neau` | 685 | 86 | 85 | 856 |
| `rlnst` | 2,308 | 288 | 290 | 2,886 |
| `up_field` | 998 | 127 | 125 | 1,250 |
| `xtu502` | 51 | 8 | 5 | 64 |
| **Total** | **13,894** | **1,748** | **1,719** | **17,361** |

Left out (duplicates, unreadable, excluded): 1,752 images, listed with reasons in `data/splits_v6_dropped.csv`.

## Weather data (fusion_v3_* models only)

`fusion_v3_weather` and `fusion_v3_by_date_weather` also take weather summaries from `data/weather.csv`
(committed), built from the [Open-Meteo historical weather archive](https://open-meteo.com) (ERA5 reanalysis)
for each photo's city and capture date (`data/image_metadata.json`). No other model uses weather or metadata
as an input.

## Datasets and download links

| `source` | Dataset | Folder in this project | Download | License |
|---|---|---|---|---|
| `kaggle` | Corn or Maize Leaf Disease Dataset (Kaggle, smaranjitghose) | `data/raw/data` | [kaggle.com/datasets/smaranjitghose/corn-or-maize-leaf-disease-dataset](https://www.kaggle.com/datasets/smaranjitghose/corn-or-maize-leaf-disease-dataset) | not checked (Kaggle terms) |
| `rlnst` | RL-NST leaf crops | `data/raw/rlnst/RL-NST` | [zenodo.org/records/19355313](https://zenodo.org/records/19355313) ("RL-NST: Augmented Corn Leaf Disease Dataset", `RL-NST(Data).zip`, same archive as ours; `Streak_virus` not used) | CC BY 4.0 |
| `cds` | CD&S: Corn Disease and Severity (Indiana, USA) | `data/raw/cds` | [osf.io/s6ru5](https://osf.io/s6ru5) (folder `Dataset_Original`; paper [arXiv 2110.12084](https://arxiv.org/abs/2110.12084)) | none stated |
| `maize_multiformat` | An Open-Source Multi-Format Maize Leaf Image Dataset | `data/raw/An Open-Source Multi-Format Maize Leaf Image Datas` | [scidb.cn … 19536c73f6d74946a212719a94f53ab3](https://www.scidb.cn/en/detail?dataSetId=19536c73f6d74946a212719a94f53ab3) (Science Data Bank) | unknown |
| `ndisan` | Corn Leaf Disease (Kaggle, ndisan; Sampang, Indonesia) | `data/raw/kaggle_corn_leaf_disease` | [kaggle.com/datasets/ndisan/corn-leaf-disease](https://www.kaggle.com/datasets/ndisan/corn-leaf-disease) | MIT |
| `mendeley_bd` | Seasonal Corn Leaf Disease Dataset (Mendeley; Gurudaspur, Bangladesh) | `data/raw/Final corn dataset` | [data.mendeley.com/datasets/vy629dngm8/1](https://data.mendeley.com/datasets/vy629dngm8/1) | CC BY 4.0 |
| `neau` | Northeast Agricultural University maize leaves (Harbin, China) | `data/raw/Northeast Agricultural University` | [figshare.com/articles/dataset/data_zip/21626348](https://figshare.com/articles/dataset/data_zip/21626348) (`data.zip`; same 315 / 263 / 278 images per class as ours) | CC BY 4.0 |
| `up_field` | Diseases of maize in the field (University of Pretoria, South Africa) | `data/raw/20237613` | [researchdata.up.ac.za … /20237613](https://researchdata.up.ac.za/articles/dataset/Diseases_of_maize_in_the_field/20237613) (DOI 10.25403/UPresearchdata.20237613.v1) | CC BY 4.0 |
| `china_2` | IDADP – Corn Disease Image Description Dataset (Science Data Bank) | `data/raw/IDADP` | https://www.scidb.cn/en/detail?dataSetId=b17f6aac10ac4d7ba247862454e2ef45 | unknown |
| `xtu502` | Mobile-DANet test images (Chen et al. 2021) | `data/raw/xtu502` | [github.com/xtu502/maize-disease-identification](https://github.com/xtu502/maize-disease-identification) (`Proj_data.rar`; paper [doi 10.1111/ppa.13322](https://doi.org/10.1111/ppa.13322)) | none stated |

Notes:
- Every dataset used by a model now has a public download link. Use the Hugging Face copy for one download, and
  check licences before sharing any images.
- Not used by any model: `data/raw/23669328` (Iowa State common-rust scans), `data/raw/yusuf_corn` (duplicates
  of `kaggle`), `data/raw/southern_corn_rust_dataset` (partial download of
  https://www.scidb.cn/en/detail?dataSetId=557575359759908864). See `data/README.md`.
- Per-dataset details (classes, places, dates, duplicates) are in each `data/raw/<folder>/README.md` and
  `data/README.md`.

_Counts generated 2026-11-15 from the committed manifests._
