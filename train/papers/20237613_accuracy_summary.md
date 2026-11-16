# Accuracy summary — Craze et al. 2022 (`20237613.pdf`)

**Paper:** Craze, H.A.; Pillay, N.; Joubert, F.; Berger, D.K. "Deep Learning Diagnostics of Gray Leaf Spot in Maize under Mixed Disease Field Conditions." *Plants* 2022, 11, 1942. https://doi.org/10.3390/plants11151942 (CC BY 4.0). Same file as `data/raw/20237613/20237613.pdf` — the paper behind the University of Pretoria field dataset (`up_field`).

## Task and test sets

- **Binary classification:** does the image show gray leaf spot (GLS) or not (`notGLS`).
- **IF ("in-field") test set:** 4,664 images (after 8-fold augmentation by mirroring and 90° rotation), 46% GLS / 54% notGLS — always guessing notGLS would score about 54%.
- **PV (PlantVillage) test set:** 963 images (143 GLS, 820 notGLS).
- **IFNB:** the IF images with backgrounds removed by a Mask R-CNN leaf model ("leafRCNN").
- Accuracy = (TP + TN) / all images; precision and recall below are for the GLS class. All values are from the paper's tables.

## Results

| Model | Trained on | Tested on | Accuracy | GLS precision | GLS recall | GLS F1 | Table |
|---|---|---|---|---|---|---|---|
| **GLS_net** (modified VGG16) | field photos (IF) | **field photos (IF)** | **73.4%** | 86.3% | 50.0% | 63.3% | 5 |
| GLS_net | field photos (IF) | PlantVillage (PV) | 78.6% | 37.5% | 65.7% | 47.7% | 6 |
| GLS_net_pv | PlantVillage | PlantVillage | **94.1%** | 89.8% | 67.8% | 77.3% | 7 |
| GLS_net_pv | PlantVillage | **field photos (IF)** | **55.1%** | 61.2% | **5.2%** | 9.7% | 8 |
| GLS_net_noBackground (leafRCNN + GLS_net) | field, background removed (IFNB) | field, background removed (IFNB) | 72.6% | 85.0% | 48.8% | 62.0% | 10 |
| GLS_net_noBackground | field, background removed (IFNB) | PlantVillage (PV) | 76.1% | 35.6% | 75.5% | 48.4% | 11 |

Also reported: the selected GLS_net run reached 75.3% (the paper says "on the IF training set"); the leafRCNN leaf-segmentation model is evaluated separately (Table 9).

## Findings

- **PlantVillage training does not transfer to the field.** The PlantVillage-trained model drops from 94.1% (PlantVillage) to 55.1% (field) — barely above always guessing notGLS — and finds only 5.2% of GLS cases in field photos.
- **Field training generalises better:** 73.4% on field photos and 78.6% on PlantVillage, without ever seeing PlantVillage images.
- **Background removal did not help** on field photos (72.6% vs 73.4%; the difference was reported as not significant).
- The authors attribute the PlantVillage gap to class-specific backgrounds and lab conditions (gray leaf spot and blight on grey backgrounds, common rust on black, healthy with no background).

## Relation to this project

- Our models are **not directly comparable**: this paper is GLS yes/no on all photos, including multi-disease ones; ours pick one of seven classes and use only single-label `up_field` photos, with different splits.
- For reference, on the `up_field` test photos (124 single-label images, `splits_v5.csv`): `corn_fungal_v5` 73.4%, `corn_fungal_v5_bgaug` 75.8% — in the same range as GLS_net's 73.4% on these hard field photos.
- The paper's main finding (PlantVillage models collapse on field data) is why this project added background augmentation; see `models/PLANTVILLAGE_BACKGROUND_ISSUE.md`.
