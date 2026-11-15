# Train / dev / test split and data provenance — Design

Date: 2026-11-14
Status: Draft, awaiting approval

## Intent and why

The corn leaf model was trained and scored on splits that were made ad hoc: random 10% validation inside `train.py`, plus CD&S's own `test` folder used as an extra test. Several problems this session came from that:

- The same dataset appeared under three names (Kaggle, two Hugging Face mirrors); only a hash comparison showed it.
- The old model had trained on about half of the new dev set, which inflated its scores.
- The test set had no healthy or common-rust photos, so it said nothing about those classes.
- The data-prep script lived only in a scratch folder, so the combined training folder could not be rebuilt from the repo.
- Where each image came from was recorded only in symlink name prefixes and in prose.

The user wants the data split into training, dev and test sets, and wants the data source of every image tracked. This design does both with one artifact: a committed manifest.

## Goals

- One pooled dataset (all sources, de-duplicated) split roughly **80% train / 10% dev / 10% test**, so every source and every class appears in all three sets.
- **Dev** is for watching training; **test** is touched only for final scores. Neither is used to pick settings.
- A committed **`data/splits.csv` manifest**: one row per image with its source, original folder, label, content hash, near-duplicate group and split. Anyone with the same downloads can rebuild the exact split.
- A short **`docs/data-sources.md`** listing each source: name, URL, how it was obtained, image counts, license status and known caveats.
- Scripts in the repo (not scratch) that build the manifest, train from it, and evaluate from it, with per-class and per-source precision and recall.
- Retrain the model on the new train split and re-score on dev and test.

## Non-goals

- No change to the app, the label set or the detector interface.
- No holding out a whole source (that was option C, not chosen).
- No per-plant or per-leaf grouping: the datasets carry no such metadata.
- No new data is downloaded; no image files are committed (only the manifest).
- Not a data-versioning system (no DVC and the like).

## Anti-goals (failure even if it technically works)

- The same photo, or a near-identical one, in two different splits.
- A test set that is missing a class or a source.
- A manifest that cannot be rebuilt into the same split by someone else, or that depends on files only on this machine.
- Silently discarding images: every dropped duplicate is recorded.
- Quoting scores from a model that trained on its own dev or test images.

## Constraints

- Raw data stays under the gitignored `data/raw/`; paths in the manifest are relative to the repo root.
- Label mapping stays in `train/labels.py` (Streak_virus remains unmapped).
- Labels: `healthy`, `common_rust`, `leaf_blight`, `leaf_spot`.
- CD&S has no stated license (OSF project); the sources doc must say so.
- The user rejected adding multi-folder, hash-deduplicating input to `train.py`'s `--data` option earlier; this design does not do that. Deduplication happens once, in the manifest builder, and `train.py` only gains an optional `--manifest` mode.

## Approaches considered

| Approach | Trade-off |
|---|---|
| **A. Pool all sources, stratified random 80/10/10, group near-duplicates (chosen)** | Simple; every source and class in every split. CD&S's own train/test split is discarded, and same-campaign leakage cannot be fully ruled out. |
| B. Keep CD&S's test split and add test/dev from other sources | Respects the authors' split, but test sizes differ by source. |
| C. Hold out a whole source as test | Honest about new sources; removes a source from training. |

Chosen by the user: A.

## Design

### Pool

All unique images from: Kaggle corn dataset (4,186), RL-NST without `Streak_virus` (2,886), CD&S `Dataset_Original` train and test together (1,571). That is 8,643 images. Exact duplicates (same MD5) are dropped, keeping the first and recording the dropped ones.

### Near-duplicate groups

Compute a 64-bit difference hash for every image. Images within Hamming distance 4 are linked into one group (connected components). A group is never split across sets. Singletons are the common case.

### Split

Within each (source, class) stratum, shuffle the groups with a fixed seed and assign them to train, dev and test to reach 80/10/10 of that stratum's images. A group that spans strata goes with its first member.

### Manifest `data/splits.csv`

Columns: `path` (repo-relative), `md5`, `source` (`kaggle`, `rlnst`, `cds`), `original_folder`, `label`, `group_id`, `split`. Also `data/splits_dropped.csv` listing exact duplicates removed and what they duplicate.

### Code

- `data_prep/build_splits.py`: scans `data/raw/`, builds both CSVs, prints per-split, per-source and per-class counts and the number of images in multi-image groups.
- `train/train.py`: new optional `--manifest data/splits.csv`; trains on `split == train`, evaluates each epoch on `split == dev`. Existing `--data` mode is unchanged.
- `train/evaluate.py`: `--manifest`, `--model`, `--split dev|test`; writes per-class precision, recall and F1, a confusion matrix, and a per-source accuracy table.
- Tests for the grouping, split assignment (no group in two splits, ratios, class and source coverage, reproducibility with the seed), and manifest round trip.

### Sources doc

`docs/data-sources.md` with one table: source, URL, access method, images used, license, caveats (mirrors that were byte-identical; Streak_virus excluded; label-mapping assumptions).

### After building

Retrain 5 epochs with the existing settings on the new train split, evaluate on dev and test, and regenerate `models/corn/MODEL_PERFORMANCE.md` from the manifest-based evaluation. Move the previous model's summary note to say its splits overlapped.

## Left to the builder

- Exact near-duplicate threshold if 4 produces unreasonably large groups (report cluster sizes; the default is 4).
- CSV details (quoting, column order) and the evaluation script's output format.
- Epoch count stays 5 unless training fails to converge.

## My calls (not discussed with the user)

- 80/10/10 ratios and the fixed seed (0).
- Near-duplicate grouping by difference hash, threshold 4.
- Manifest committed to git, images not; paths relative to the repo root.
- `docs/data-sources.md` as the human-readable source record.
- Adding `--manifest` to `train.py` and a separate `train/evaluate.py`.
- Retraining after the split, and treating all earlier scores as superseded.
