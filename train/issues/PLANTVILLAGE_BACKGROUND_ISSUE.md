# The PlantVillage background problem

Date: 2026-11-14, updated 2026-11-15. Status: **mitigated, not solved.** Background augmentation clearly reduces the models' reliance on backgrounds (background-swap drop about 10 → 3 points), but field-photo accuracy is still the weak spot and its gain did not repeat on v6. See [Results](#results-2026-11-15).

## What PlantVillage is

PlantVillage is a public dataset of about 54,000 leaf photos (14 crops, 38 disease/healthy classes)
built at Penn State (Hughes & Salathé, 2015). Each photo shows one detached leaf on a plain background
under controlled light. A 2016 study reported 99.35% accuracy on it, and it became the standard benchmark
for plant-disease classifiers. Its maize part has four classes: gray leaf spot (*Cercospora*), common
rust, northern leaf blight and healthy.

## The problem

The background and setup differ systematically **by class**, so a model can classify by background
instead of by disease:

| PlantVillage maize class | Background |
|---|---|
| Gray leaf spot | grey |
| Northern leaf blight | grey |
| Common rust | black |
| Healthy | none (cut out) |

Evidence (Craze, Pillay, Joubert & Berger 2022, *Plants* 11, 1942 — the paper behind
`../../data/raw/20237613`): a gray-leaf-spot model trained on PlantVillage scored **94.1% on PlantVillage but
55.1% on real field photos**. A model trained on mixed-disease field photos scored 73.4% on field photos
and 78% on PlantVillage, i.e. it generalised far better. Removing backgrounds with Mask R-CNN did not help
on field photos (72.6% vs 73.4%).

## How it affects this project

- **Affected sources:** `kaggle` (`../../data/raw/data`, Kaggle `smaranjitghose`) is mostly PlantVillage maize
  images (256×256 cut-outs; its 1,162 healthy images match PlantVillage's count) plus about 10% PlantDoc web
  photos. `rlnst` (224×224 crops on grey backgrounds, origin unknown) looks similar. `ndisan` photos are on
  white paper (one background for all classes, so less of a class cue, but still a lab-style setting).
  Together these are about 10,000 of the 16,460 images in `splits_v5.csv`.
- **Symptoms seen so far:**
  - Models score 93–94% on held-out test photos, which are mostly from the same lab-style sources, but
    much less on field photos: the first 4-class model scored **37.8%** on a dataset it never saw
    (`maize_multiformat`), and `maize_multiformat` stays the weakest source (about 74%).
  - Images from the same capture day are recognised far better when that day is also in training (95.4%
    vs 88.2%), i.e. models pick up setting cues, not only disease.
- **What counts as success:** accuracy on **field photos** (`up_field`, `cds`, `mendeley_bd`, `neau`,
  `maize_multiformat`), not overall accuracy.

## Options considered

| Option | Verdict |
|---|---|
| Replace backgrounds during training (augmentation) | **chosen first**: cut the leaf out of plain-background photos and paste it on random flat colours, gradients, noise or patches of training-set field photos, on 70% of lab-style training images per epoch; dev/test and the app are unchanged |
| Cut out every leaf and make all backgrounds black, in training and in the app | not first: needs a segmentation model in the app for every upload; tan/brown lesions can be cut out as "background"; clean lab edges vs ragged field edges become a new cue; the Craze paper found no gain |
| Tighter crops so the leaf fills the frame | cheap complement, not done yet |
| Weight field photos higher, or fine-tune on field photos only | possible later |
| Hold out whole datasets when testing | the most honest generalisation measure; possible later |

## Implementation (branch `feat/data-split`)

- `../../data_prep/background.py` — `leaf_mask` (background colour estimated from the image border; background =
  background-coloured region connected to the border, so lesions inside the leaf stay leaf; images without a
  plain border get no mask) and `replace_background` (soft edges).
- `../../data_prep/leaf_masks.py` — computes and caches masks for `kaggle`, `rlnst`, `ndisan` in
  `data/raw/mask_cache/`.
- `python -m train.train --manifest data/splits_v5.csv --bg-augment` — replacement during training only.

## How it is judged

1. Train the augmented model and compare with the same model without augmentation (same data) on field-photo test accuracy per source and class.
2. **Background-swap test:** re-score lab-style test photos with their backgrounds replaced. A model that relies on backgrounds loses accuracy; the augmented one should not.
3. Watch for a new cue: if mask edges differ between lab and field photos, the model could learn those instead.

## Results (2026-11-15)

### 1. Field-photo test accuracy: mixed

| Data | Without augmentation | With augmentation | Change |
|---|---|---|---|
| v5 (7 labels), field-photo sources, 651 test photos | 87.6% | 89.6% | +2.0 |
| v6 (11 labels), same 651 photos | 89.4% | 88.8% | −0.6 |
| `maize_multiformat` (72 photos), v5 / v6 | 56.9% / 69.4% | 69.4% / 59.7% | +12.5 / −9.7 |
| `up_field` (124 photos), v5 / v6 | 73.4% / 75.0% | 75.8% / 75.0% | +2.4 / 0.0 |

One training run per model. The swings on small sources (72 photos moving ±10 points) show that run-to-run noise is as large as the effect, so this check is inconclusive.

### 2. Background-swap test: augmentation works

The 323 lab-style test photos (`kaggle`, `rlnst`, `ndisan`) of `splits_v5.csv` that have a leaf mask, scored three times: original background, a flat colour, and a patch of a real field photo. The same photos, masks and random backgrounds (seed 123) are used for every model. Script: `bg_swap.py` (scratchpad); results in `models/<model>/bg_swap_rerun.json`.

| Model | Original | Flat | Field | Drop on field backgrounds |
|---|---|---|---|---|
| `corn_fungal_v5` | 93.5% | 91.6% | 83.0% | **−10.5** |
| `corn_fungal_v5_bgaug` | 93.5% | 92.6% | 90.4% | **−3.1** |
| `corn_fungal_v6` | 95.7% | 92.6% | 87.3% | **−8.4** |
| `corn_fungal_v6_bgaug` | 94.1% | 92.9% | 91.3% | **−2.8** |

Where the drop comes from, field-background change in points:

| | Photos | v5 | v5 + aug | v6 | v6 + aug |
|---|---|---|---|---|---|
| common rust | 181 | −18.2 | −3.9 | −13.3 | −4.4 |
| `kaggle` (PlantVillage) | 103 | −18.4 | −2.9 | −7.8 | −1.9 |
| `rlnst` | 59 | −11.9 | −3.4 | −8.5 | +1.7 |
| `ndisan` | 161 | −5.0 | −3.1 | −8.7 | −5.0 |
| northern leaf blight | 72 | −1.4 | −2.8 | −4.2 | 0.0 |
| healthy | 57 | 0.0 | 0.0 | 0.0 | −1.8 |
| gray leaf spot | 13 | 0.0 | −7.7 | 0.0 | 0.0 |

- **The shortcut is real and mostly about common rust on PlantVillage backgrounds.** Without augmentation, putting a field background behind the same rust leaves costs 13–18 points.
- **Augmentation removes most of it,** in both v5 and v6: the overall drop shrinks from 10.5 → 3.1 and 8.4 → 2.8 points. Common rust's drop shrinks from 18.2 → 3.9 and 13.3 → 4.4. Unlike check 1, this result repeats across both pairs of models.
- **Caveat: the test favours the augmented models somewhat.** Its field backgrounds are patches of the same training-set field photos the augmentation pasted in during training, so the augmented models have seen those exact backgrounds. A fairer version would use field backgrounds the model never saw (e.g. dev/test field photos, or new ones).
- Accuracy on the original photos is unchanged or slightly lower with augmentation (93.5 → 93.5, 95.7 → 94.1).

### 3. Mask-edge cue: not checked yet

## Where this leaves us

- The background shortcut has been shown, and augmentation reduces it a lot in both v5 and v6.
- The website's model (`corn_fungal_v6`, behind the Claude corn-leaf check) is the **non-augmented** one; it still loses 8.4 points (13 on common rust) when lab leaves appear on field backgrounds. **`corn_fungal_v6_bgaug` is the more background-robust choice** with near-identical test accuracy (91.3% vs 91.6%).
- Field-photo accuracy remains the real gap (about 75% on `up_field`, 60–70% on `maize_multiformat`); background augmentation alone does not close it.

## Next steps

1. Repeat the swap test with backgrounds from field photos no model trained on.
2. Train 2–3 seeds of `corn_fungal_v6` and `corn_fungal_v6_bgaug` to settle the field-photo comparison.
3. Check the mask-edge cue (check 3).
4. Consider switching the website to `corn_fungal_v6_bgaug`.
5. More field photos for training, and tighter crops (see options above).
