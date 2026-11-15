# Data Split and Provenance Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Pool all corn-leaf datasets, split them 80/10/10 into train/dev/test with near-duplicates kept together, record every image's source in a committed manifest, train and evaluate from that manifest, and retrain the model.

**Architecture:** `data_prep/` builds the manifest (scan sources, drop exact duplicates, group near-duplicates by perceptual hash, stratified group split). `train/train.py` gains an optional `--manifest` mode; a new `train/evaluate.py` scores a model on the dev or test split with per-class and per-source metrics. Pure logic is unit tested with tiny synthetic data; the real run happens in the last tasks.

**Tech Stack:** Python 3.14, Pillow, numpy, pytest; existing torch/transformers for training and evaluation.

**Spec:** `docs/superpowers/specs/2026-11-14-data-split-and-provenance-design.md`

## Global Constraints

- Labels are exactly `healthy`, `common_rust`, `leaf_blight`, `leaf_spot`; label mapping stays in `train/labels.py` (`Streak_virus` stays unmapped).
- Split ratios 80/10/10, fixed seed 0; stratified by (`source`, `label`); a near-duplicate group is never split across sets.
- Near-duplicate rule: 64-bit difference hash, Hamming distance <= 4 links two images (connected components).
- Sources and their `source` names: `kaggle` (`data/raw/data`), `rlnst` (`data/raw/rlnst/RL-NST`), `cds` (`data/raw/cds`, both its `train` and `test` folders pooled).
- Manifest `data/splits.csv` columns, in order: `path,md5,source,original_folder,label,group_id,split`. Paths are relative to the repo root. Dropped images go to `data/splits_dropped.csv` with columns `path,md5,reason,duplicate_of`.
- Do NOT add multi-folder or hash-deduplicating input to `train.py`'s `--data` option; `--manifest` is a separate optional mode and the `--data` behaviour is unchanged.
- Images and models are never committed; the two CSVs and docs are. `data/splits*.csv` must be un-ignored in `.gitignore` while `data/raw/` and `data/combined/` stay ignored.
- Scores are only quoted from a model that did not train on the split it is scored on.

## Review Focus

Each line has a test in the task named in brackets.

1. The same image saved under two names, and a slightly resized or re-encoded copy, end up in one group and therefore one split. [Task 1, 2]
2. An exact duplicate across two sources is kept once and appears in `splits_dropped.csv`; an unreadable or non-image file is skipped and recorded, not a crash. [Task 3]
3. A tiny stratum (1 to 9 images) never crashes the split and never produces a group in two splits. [Task 2]
4. The manifest uses repo-relative paths, and `train --manifest` fails with a clear message naming the first missing file. [Task 3, 4]
5. Evaluating a split that lacks a class (zero support) reports "n/a" instead of dividing by zero. [Task 5]

---

### Task 1: Near-duplicate grouping

**Files:**
- Create: `data_prep/__init__.py`, `data_prep/grouping.py`, `tests/test_grouping.py`

**Interfaces:**
- Produces:
  - `dhash(path: Path | str) -> int`: 64-bit difference hash (grayscale, resize to 9x8, compare horizontal neighbours); raises `ValueError("not a valid image")` if undecodable.
  - `group_ids(hashes: list[int], threshold: int = 4) -> list[int]`: connected components over pairs with Hamming distance <= threshold; returns one id per input, ids numbered from 0 in order of first appearance. Must handle ~10,000 hashes in a few seconds (numpy, chunked).

- [ ] **Step 1: Write failing tests** in `tests/test_grouping.py`:
  - `test_identical_files_get_same_hash` (same pixels saved as two different file names).
  - `test_resized_copy_is_within_threshold` (a 64x64 gradient image and a 48x48 resize of it: Hamming distance <= 4).
  - `test_different_images_are_far_apart` (a left-to-right gradient vs a right-to-left gradient: distance > 4).
  - `test_group_ids_links_chains` (ints a, b, c where a~b and b~c but a and c differ by 6 bits: all three share one group id; a far-away d gets another).
  - `test_group_ids_are_numbered_by_first_appearance` (`[x, y, x']` gives `[0, 1, 0]`).
  - `test_group_ids_empty_and_single` (`[]` gives `[]`; one hash gives `[0]`).
  - `test_dhash_rejects_garbage` (non-image file raises `ValueError`).
- [ ] **Step 2: Run `.venv/bin/pytest tests/test_grouping.py -q`; expect FAIL** (module not found).
- [ ] **Step 3: Implement** `dhash` and `group_ids` with the signatures above (union-find over chunked numpy XOR + popcount).
- [ ] **Step 4: Run the tests; expect PASS**, then run `.venv/bin/pytest -q` for the full suite.
- [ ] **Step 5: Commit** `feat: near-duplicate grouping by perceptual hash`.

---

### Task 2: Stratified group split

**Files:**
- Create: `data_prep/splitting.py`, `tests/test_splitting.py`

**Interfaces:**
- Consumes: rows as `dict` with keys `source`, `label`, `group_id` (int).
- Produces: `assign_splits(rows: list[dict], ratios=(0.8, 0.1, 0.1), seed: int = 0) -> list[str]`: one value of `"train"`, `"dev"` or `"test"` per input row, in input order. Rows sharing `group_id` always get the same split. Within each (`source`, `label`) stratum, groups are shuffled with the seed and assigned to hit the ratios of that stratum's image count (largest-remainder rounding at the group level is fine). A group spanning strata is assigned with its first member's stratum.

- [ ] **Step 1: Write failing tests** in `tests/test_splitting.py`, using synthetic rows (several sources and labels, mostly singleton groups, a few multi-image groups):
  - `test_no_group_spans_two_splits`.
  - `test_overall_ratios_close_to_80_10_10` (1,000 singleton rows: counts 800/100/100 within +-2).
  - `test_every_stratum_with_enough_images_appears_in_all_splits` (strata with >= 20 images have all three splits).
  - `test_same_seed_reproduces_and_different_seed_differs`.
  - `test_tiny_strata_do_not_crash` (strata of 1, 2 and 9 images; output length equals input length; one-image strata get `"train"`).
  - `test_big_group_is_kept_whole_even_if_it_exceeds_the_dev_share` (a 30-image group in a 100-image stratum stays in one split).
  - `test_empty_input_returns_empty_list`.
- [ ] **Step 2: Run `.venv/bin/pytest tests/test_splitting.py -q`; expect FAIL.**
- [ ] **Step 3: Implement** `assign_splits` as specified.
- [ ] **Step 4: Run the tests; expect PASS**, then the full suite.
- [ ] **Step 5: Commit** `feat: stratified group split`.

---

### Task 3: Manifest builder

**Files:**
- Create: `data_prep/manifest.py`, `data_prep/build_splits.py`, `tests/test_manifest.py`, `tests/test_build_splits.py`
- Modify: `.gitignore` (add `!data/splits.csv` and `!data/splits_dropped.csv`, keeping `data/` ignored by listing `data/*` then the two exceptions)

**Interfaces:**
- Consumes: `folder_to_label` (`train/labels.py`), `dhash`/`group_ids` (Task 1), `assign_splits` (Task 2).
- Produces, in `data_prep/manifest.py`:
  - `COLUMNS = ["path", "md5", "source", "original_folder", "label", "group_id", "split"]`
  - `write_manifest(rows: list[dict], path: Path) -> None` and `read_manifest(path: Path) -> list[dict]` (round trip; `group_id` returned as int).
  - `samples_for_split(rows: list[dict], split: str, root: Path) -> list[tuple[Path, str]]`: `(root / row["path"], row["label"])` for that split, sorted by path; raises `FileNotFoundError("manifest file missing: <path>")` naming the first missing file.
- Produces, in `data_prep/build_splits.py`:
  - `SOURCES: dict[str, str]` = `{"kaggle": "data/raw/data", "rlnst": "data/raw/rlnst/RL-NST", "cds": "data/raw/cds"}`
  - `scan_source(name: str, root: Path, repo_root: Path) -> list[dict]`: every image file (`.jpg/.jpeg/.png`, case-insensitive) under any folder whose name `folder_to_label` recognises, at any depth; each dict has `path` (repo-relative), `source`, `original_folder`, `label`.
  - `build(repo_root: Path, sources: dict[str, str] = SOURCES, out_dir: Path | None = None) -> dict`: scans, drops exact duplicates (first by sorted path wins) and unreadable files into `splits_dropped.csv` (`reason` is `duplicate` or `unreadable`; `duplicate_of` is the kept path or empty), computes hash groups, assigns splits, writes both CSVs, returns the counts by split, source and label.
  - CLI: `python -m data_prep.build_splits` prints those counts and the number of images in multi-image groups.

- [ ] **Step 1: Write failing tests.**
  - `tests/test_manifest.py`: `test_round_trip_keeps_values_and_types`; `test_samples_for_split_filters_and_sorts`; `test_samples_for_split_names_first_missing_file` (message contains the missing relative path).
  - `tests/test_build_splits.py`, using a `tmp_path` repo with three small fake sources of distinct synthetic images (class folders `Healthy`, `Blight`, plus a `Streak_virus` folder and a `Tomato___healthy` folder):
    - `test_unmapped_folders_are_ignored` (Streak_virus and tomato images appear in neither CSV).
    - `test_exact_duplicate_across_sources_kept_once_and_listed` (same bytes in `kaggle` and `rlnst`: one manifest row, one dropped row with `reason == "duplicate"` and the kept path in `duplicate_of`).
    - `test_unreadable_file_is_recorded_not_fatal` (a `.jpg` containing text appears in the dropped CSV with `reason == "unreadable"`).
    - `test_paths_are_repo_relative_and_files_exist` (every manifest path is relative and `(repo_root / path).is_file()`).
    - `test_near_duplicate_pair_shares_group_and_split` (an image and a slightly resized copy have equal `group_id` and `split`).
    - `test_uppercase_extension_is_included` (`x.JPG` is scanned).
    - `test_same_inputs_give_identical_manifest` (build twice; files byte-identical).
- [ ] **Step 2: Run `.venv/bin/pytest tests/test_manifest.py tests/test_build_splits.py -q`; expect FAIL.**
- [ ] **Step 3: Implement** the modules to the interfaces above; update `.gitignore`.
- [ ] **Step 4: Run the tests; expect PASS**, then the full suite.
- [ ] **Step 5: Commit** `feat: manifest builder for the pooled split`.

---

### Task 4: `train.py --manifest` mode

**Files:**
- Modify: `train/train.py`
- Create: `tests/test_train_args.py`

**Interfaces:**
- Consumes: `read_manifest`, `samples_for_split` (Task 3).
- Produces, in `train/train.py`: `parse_args(argv: list[str] | None = None) -> argparse.Namespace`, where `--data` (one path, as today) and `--manifest` (one path) are mutually exclusive and one is required; `--out`, `--epochs`, `--base-model`, `--val-fraction` unchanged. In `main()`, with `--manifest`, training samples are the manifest's `train` rows and per-epoch evaluation uses its `dev` rows (the manifest path's parent's parent is the repo root, or `--repo-root` default `.`); the existing `--data` code path is untouched.

- [ ] **Step 1: Write failing tests** in `tests/test_train_args.py`: `test_data_and_manifest_are_mutually_exclusive` (both given: `SystemExit`); `test_one_of_data_or_manifest_is_required`; `test_data_mode_defaults_unchanged` (`--data d` gives `epochs == 3`, `val_fraction == 0.1`, `manifest is None`); `test_manifest_mode_parses` (`--manifest m.csv` gives `data is None`).
- [ ] **Step 2: Run `.venv/bin/pytest tests/test_train_args.py -q`; expect FAIL.**
- [ ] **Step 3: Implement** `parse_args`, move argument parsing into it, and add the manifest branch to `main()` (reuse the existing Dataset class and Trainer setup; keep the BatchNorm fix call).
- [ ] **Step 4: Run the tests and the full suite; expect PASS**, and `.venv/bin/python -m py_compile train/train.py`.
- [ ] **Step 5: Commit** `feat: train from the split manifest`.

---

### Task 5: Evaluation script

**Files:**
- Create: `train/evaluate.py`, `tests/test_evaluate.py`

**Interfaces:**
- Consumes: `read_manifest`, `samples_for_split` (Task 3); `Detector`-free: loads the model directly with transformers.
- Produces:
  - `compute_metrics(true: list[str], pred: list[str], labels: list[str]) -> dict` with keys `accuracy` (float), `macro_f1` (float over classes with support > 0), `per_class` (`{label: {"precision": float | None, "recall": float | None, "f1": float | None, "support": int}}`, `None` when the denominator is zero), `confusion` (`{true_label: {pred_label: count}}`).
  - `metrics_by_source(rows_true_pred_source: list[tuple[str, str, str]]) -> dict[str, dict]`: accuracy and counts per source.
  - `format_markdown(metrics: dict, by_source: dict, title: str) -> str`: the tables, with `n/a` for `None`.
  - CLI: `python -m train.evaluate --manifest data/splits.csv --model models/corn --split dev|test [--json out.json]` prints the markdown and optionally writes JSON. Uses the model's own `id2label`, batch size 32, MPS if available else CPU.

- [ ] **Step 1: Write failing tests** in `tests/test_evaluate.py`: `test_perfect_predictions` (accuracy 1.0, all precision/recall 1.0); `test_known_confusion_values` (a hand-computed 6-image example: exact precision, recall, F1, confusion); `test_zero_support_class_reports_none_not_error` (a label absent from `true` and `pred` has `None` metrics and does not affect `macro_f1`); `test_class_never_predicted_has_none_precision` (recall 0.0, precision `None`); `test_metrics_by_source_groups_correctly`; `test_format_markdown_shows_na_for_none`.
- [ ] **Step 2: Run `.venv/bin/pytest tests/test_evaluate.py -q`; expect FAIL.**
- [ ] **Step 3: Implement** the functions and the CLI.
- [ ] **Step 4: Run the tests and the full suite; expect PASS.**
- [ ] **Step 5: Commit** `feat: manifest-based evaluation script`.

---

### Task 6: Build the real manifest and the sources doc

**Files:**
- Create: `data/splits.csv`, `data/splits_dropped.csv` (generated), `docs/data-sources.md`

- [ ] **Step 1: Run `.venv/bin/python -m data_prep.build_splits`.** Expected: about 8,643 kept images (Kaggle 4,186; RL-NST 2,886; CD&S 1,571), counts per split near 80/10/10 with all four labels and all three sources in every split. Read the printed counts and the dropped-file counts (exact duplicates expected: 2 from Kaggle). If any group is huge (more than 20 images), report it and decide the threshold under the spec's "left to the builder" clause, ledgering a Ruling.
- [ ] **Step 2: Sanity-check the manifest** with a short script (not committed): every path exists; no `md5` appears twice; no `group_id` appears in two splits; every (source, label) stratum with >= 20 images appears in all three splits.
- [ ] **Step 3: Write `docs/data-sources.md`:** one table of the three sources (name, URL, how obtained, image counts used, license status, caveats): Kaggle `smaranjitghose/corn-or-maize-leaf-disease-dataset` (also mirrored by HF `Prachi1234/corn-leaf-disease` and `Project-AgML/corn_maize_leaf_disease`, byte-identical, so used once); RL-NST from the user's `RL-NST(Data).zip` with `Streak_virus` excluded and the label mapping `Leaf_Blight` -> `leaf_blight`, `Leaf_spot` -> `leaf_spot` confirmed by the user; CD&S OSF project `s6ru5` `Dataset_Original`, no license stated. Include the manifest and dropped-file counts and how to rebuild.
- [ ] **Step 4: Commit** the manifest CSVs and the doc: `data: pooled split manifest and sources doc`.

---

### Task 7: Retrain, evaluate, update docs

**Files:**
- Modify: `README.md` (training section), `models/corn/MODEL_PERFORMANCE.md` (regenerated, gitignored), `models/corn_4class/MODEL_PERFORMANCE.md` (note)

- [ ] **Step 1: Back up the current model:** move `models/corn` to `models/corn_pre_split` (it trained on images that are now in dev and test).
- [ ] **Step 2: Train:** `.venv/bin/python -m train.train --manifest data/splits.csv --out models/corn --epochs 5`, in the background with the log in `models/train.log`. Expected: it prints the counts of train and dev rows, resets momentum on 52 BatchNorm layers, and per-epoch dev accuracy that is stable (no 40-60% swings); `models/corn/config.json` exists at the end. If dev accuracy swings, treat it as a bug (systematic-debugging) before continuing.
- [ ] **Step 3: Evaluate:** run `train.evaluate` for `--split dev` and `--split test`, saving JSON beside the model. Expected: the dev accuracy equals the trainer's final dev accuracy.
- [ ] **Step 4: Regenerate `models/corn/MODEL_PERFORMANCE.md`** from those results: the split table by source and class, per-class precision, recall and F1 for dev and test, per-source accuracy, the confusion matrix, and honest limitations (same-campaign leakage in CD&S cannot be ruled out beyond near-duplicate grouping; test and dev were not used to tune; single run; no "not a corn leaf" class; no license for CD&S). State that all earlier scores are superseded.
- [ ] **Step 5: Add a note** at the top of `models/corn_4class/MODEL_PERFORMANCE.md` and of `models/corn_pre_split` (if it has a summary) that their splits overlap the current dev and test sets.
- [ ] **Step 6: Update the README training section** to describe `--manifest`, `data_prep.build_splits`, `train.evaluate` and `docs/data-sources.md`.
- [ ] **Step 7: Run the full suite** (`.venv/bin/pytest -q`; expect all pass) and **commit** `docs: training and evaluation via the split manifest`.
