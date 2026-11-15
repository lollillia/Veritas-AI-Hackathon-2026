from pathlib import Path

import pytest

from data_prep.manifest import COLUMNS, read_manifest, samples_for_split, write_manifest


def _row(path, split, label="healthy", group_id=0):
    return {"path": path, "md5": "m" + path, "source": "kaggle", "original_folder": "Healthy",
            "label": label, "group_id": group_id, "split": split}


def test_round_trip_keeps_values_and_types(tmp_path):
    rows = [_row("data/raw/data/Healthy/a.jpg", "train", group_id=3), _row("data/raw/data/Blight/b.jpg", "dev", "leaf_blight", 4)]
    write_manifest(rows, tmp_path / "m.csv")
    back = read_manifest(tmp_path / "m.csv")
    assert back == rows
    assert isinstance(back[0]["group_id"], int)
    assert (tmp_path / "m.csv").read_text().splitlines()[0] == ",".join(COLUMNS)


def test_samples_for_split_filters_and_sorts(tmp_path):
    for name in ("b.jpg", "a.jpg", "c.jpg"):
        (tmp_path / name).write_bytes(b"x")
    rows = [_row("b.jpg", "train"), _row("a.jpg", "train"), _row("c.jpg", "dev", "leaf_spot")]
    out = samples_for_split(rows, "train", tmp_path)
    assert out == [(tmp_path / "a.jpg", "healthy"), (tmp_path / "b.jpg", "healthy")]


def test_samples_for_split_names_first_missing_file(tmp_path):
    (tmp_path / "a.jpg").write_bytes(b"x")
    rows = [_row("a.jpg", "train"), _row("gone/z.jpg", "train"), _row("gone/y.jpg", "train")]
    with pytest.raises(FileNotFoundError, match="manifest file missing: gone/y.jpg"):
        samples_for_split(rows, "train", tmp_path)
