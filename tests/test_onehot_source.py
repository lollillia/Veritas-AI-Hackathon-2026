import csv
from pathlib import Path

from PIL import Image

from data_prep.onehot_source import scan_onehot_csv

COLS = ["imgID_id", "filePath", "GLS", "NCLB", "PLS", "CR", "SR", "NoFoliarSymptoms", "Other", "UnidentifiedDisease"]


def _repo(tmp_path, rows):
    img_dir = tmp_path / "data" / "raw" / "up" / "leaf_images"
    img_dir.mkdir(parents=True)
    with open(tmp_path / "data" / "raw" / "up" / "Database.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(COLS)
        for i, (name, flags) in enumerate(rows):
            if not name.startswith("missing"):
                Image.new("RGB", (8, 8), (i, i, i)).save(img_dir / name, format="JPEG")
            w.writerow([i, name] + [1 if c in flags else 0 for c in COLS[2:]])
    return tmp_path / "data" / "raw" / "up" / "Database.csv", img_dir


def test_only_single_label_images_with_a_label_are_returned(tmp_path):
    csv_path, img_dir = _repo(tmp_path, [
        ("a.JPG", {"GLS"}), ("b.JPG", {"PLS"}), ("c.JPG", {"NoFoliarSymptoms"}),
        ("d.JPG", {"GLS", "NCLB"}),            # multi-label: skipped
        ("e.JPG", {"SR"}),                      # southern rust
        ("f.JPG", {"Other"}), ("g.JPG", {"UnidentifiedDisease"}),   # no label: skipped
        ("h.JPG", set()),                       # no flags at all: skipped
    ])
    rows, skipped = scan_onehot_csv("up", csv_path, img_dir, tmp_path)
    assert [(Path(r["path"]).name, r["label"], r["original_folder"]) for r in rows] == [
        ("a.JPG", "gray_leaf_spot", "GLS"), ("b.JPG", "phaeosphaeria_leaf_spot", "PLS"), ("c.JPG", "healthy", "NoFoliarSymptoms"),
        ("e.JPG", "southern_rust", "SR")]
    assert all(r["source"] == "up" and r["path"].startswith("data/raw/up/leaf_images/") for r in rows)
    assert skipped == {"multi_label": 1, "no_label": 3, "missing_file": 0}


def test_missing_files_are_counted_not_fatal(tmp_path):
    csv_path, img_dir = _repo(tmp_path, [("a.JPG", {"CR"}), ("missing.JPG", {"CR"})])
    rows, skipped = scan_onehot_csv("up", csv_path, img_dir, tmp_path)
    assert [r["label"] for r in rows] == ["common_rust"] and skipped["missing_file"] == 1
