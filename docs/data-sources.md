# Data sources

Where every training, dev and test image comes from. The per-image record is `data/splits.csv`
(source, original folder, label, content hash, near-duplicate group, split). Exact duplicates and
unreadable files that were left out are in `data/splits_dropped.csv`.

The images are not in the repository (`data/raw/` is gitignored). Rebuild the manifest with
`python -m data_prep.build_splits` after placing the downloads as described below.

## Sources used

| `source` | What it is | Where it came from | Images used | License |
|---|---|---|---|---|
| `kaggle` | Corn or Maize Leaf Disease Dataset: mostly 256×256 PlantVillage-style leaf cutouts, with about 10% field photos (estimated from a 200-image sample). Classes: Blight, Common_Rust, Gray_Leaf_Spot, Healthy | Kaggle `smaranjitghose/corn-or-maize-leaf-disease-dataset` (https://www.kaggle.com/datasets/smaranjitghose/corn-or-maize-leaf-disease-dataset), downloaded with `kaggle datasets download ... --unzip` into `data/raw/data` | 4,186 | not checked |
| `rlnst` | RL-NST: 224×224 leaf crops. Folders Common_rust, Healthy, Leaf_Blight, Leaf_spot (Streak_virus excluded) | `RL-NST(Data).zip` supplied by the user; extracted without the `Streak_virus` folder into `data/raw/rlnst/RL-NST`. Original publisher unknown | 2,886 | unknown |
| `cds` | CD&S (Corn Disease and Severity): full-resolution handheld field photos. Folders gls, nlb, nls in the authors' `train` and `test` splits, which are pooled here | OSF project `s6ru5`, folder `Dataset_Original` (https://osf.io/s6ru5), downloaded through the OSF API into `data/raw/cds`. Paper: Ahmad et al. 2021, https://arxiv.org/abs/2110.12084 | 1,571 | **none stated** on the OSF project; check before redistributing or publishing results |

Total: 8,643 images in the manifest.

## Label mapping

Folder names are mapped to labels in `train/labels.py`.

| Source folder | Label | Basis |
|---|---|---|
| Kaggle `Healthy`, RL-NST `Healthy` | `healthy` | folder name |
| Kaggle `Common_Rust`, RL-NST `Common_rust` | `common_rust` | folder name |
| Kaggle `Blight`, RL-NST `Leaf_Blight`, CD&S `nlb` | `leaf_blight` | assumed to be northern corn leaf blight; RL-NST confirmed by the user and cross-checked with a model and Claude |
| Kaggle `Gray_Leaf_Spot`, RL-NST `Leaf_spot`, CD&S `gls` and `nls` | `leaf_spot` | gray and northern leaf spot merged on purpose; RL-NST `Leaf_spot` as gray leaf spot confirmed by the user, cross-checked (weakly) |
| RL-NST `Streak_virus` | not used | excluded on purpose, and unmapped so it can never be trained on |

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
- **Near-duplicate grouping is narrow.** Images are grouped when their 64-bit difference hashes are within 4 bits, which catches near-identical files (43 images fall in such groups, the largest of size 3). It does not catch rotated, flipped or cropped copies of the same photo, nor photos of the same leaf or plant taken separately. Leakage between splits through such copies cannot be ruled out.
- **CD&S photos come from one field campaign**, so photos of the same plants can sit in different splits.
- **Few field photos for healthy leaves and common rust.** All healthy photos are leaf crops (Kaggle, RL-NST).

## Rebuilding

1. Download the three sources into the locations in the table above.
2. `python -m data_prep.build_splits` (about a minute) writes `data/splits.csv` and `data/splits_dropped.csv`. The split uses seed 0 and is reproducible from the same files.
3. `python -m train.train --manifest data/splits.csv --out models/corn --epochs 5`, then `python -m train.evaluate --manifest data/splits.csv --model models/corn --split dev` (or `test`).
