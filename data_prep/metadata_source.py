"""Datasets labelled by this project's own per-folder image_metadata.csv (data/raw/<folder>/image_metadata.csv).

Used for datasets whose folders do not name the class (e.g. china_2, one flat folder) or whose labels were
curated by hand (e.g. xtu502). Rows with a blank label, or a label that train.labels.folder_to_label does not
map to itself, are skipped and counted.
"""
import csv
from pathlib import Path

from train.labels import folder_to_label


def scan_metadata_csv(name: str, csv_path: Path, repo_root: Path) -> tuple[list[dict], dict]:
    """(rows like build_splits.scan_source returns, counts of skipped rows by reason)."""
    rows, skipped = [], {"no_label": 0, "missing_file": 0}
    with open(csv_path, newline="") as f:
        for rec in csv.DictReader(f):
            label = rec["label"].strip()
            if not label or folder_to_label(label) != label:
                skipped["no_label"] += 1
                continue
            path = Path(repo_root) / rec["path"]
            if not path.is_file():
                skipped["missing_file"] += 1
                continue
            rows.append({"path": rec["path"], "source": name, "original_folder": path.parent.name, "label": label})
    rows.sort(key=lambda r: r["path"])
    return rows, skipped
