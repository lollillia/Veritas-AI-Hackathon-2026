"""Datasets labelled by a one-hot CSV instead of class folders.

Format (University of Pretoria "Diseases of maize in the field"): columns `imgID_id, filePath`, then one
0/1 column per class. Only images with exactly one class whose column name maps to a label (via
train.labels.folder_to_label) are used; multi-label images and unmapped classes are skipped and counted.
"""
import csv
from pathlib import Path

from train.labels import folder_to_label


def scan_onehot_csv(name: str, csv_path: Path, image_dir: Path, repo_root: Path) -> tuple[list[dict], dict]:
    """(rows like build_splits.scan_source returns, counts of skipped images by reason)."""
    rows, skipped = [], {"multi_label": 0, "no_label": 0, "missing_file": 0}
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        class_cols = reader.fieldnames[2:]
        for rec in reader:
            on = [c for c in class_cols if rec[c].strip() == "1"]
            if len(on) > 1:
                skipped["multi_label"] += 1
                continue
            label = folder_to_label(on[0]) if on else None
            if label is None:
                skipped["no_label"] += 1
                continue
            path = Path(image_dir) / Path(rec[reader.fieldnames[1]]).name
            if not path.is_file():
                skipped["missing_file"] += 1
                continue
            rows.append({"path": path.relative_to(repo_root).as_posix(), "source": name,
                         "original_folder": on[0], "label": label})
    rows.sort(key=lambda r: r["path"])
    return rows, skipped
