import csv
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from data_prep.build_splits import SOURCES, build, scan_source
from data_prep.manifest import read_manifest


def _noise(seed, size=32):
    arr = np.random.default_rng(seed).integers(0, 255, (size, size, 3), dtype=np.uint8)
    return Image.fromarray(arr)


def _structured(size=64):
    img = Image.new("L", (size, size), 90)
    d = ImageDraw.Draw(img)
    for i, shade in enumerate([200, 40, 160, 70, 220]):
        r = size // 2 - i * size // 12
        d.ellipse([size // 2 - r, size // 2 - r // 2, size // 2 + r, size // 2 + r], fill=shade)
    d.rectangle([2, 2, size // 3, size // 5], fill=250)
    return img.convert("RGB")


def _save(img, path, **kw):
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path, **kw)


def make_repo(root: Path) -> Path:
    raw = root / "data" / "raw"
    _save(_noise(1), raw / "data" / "Healthy" / "a.png")
    _save(_noise(2), raw / "data" / "Blight" / "b.png")
    _save(_noise(10), raw / "data" / "Tomato___healthy" / "tomato.png")          # other crop
    _save(_noise(3), raw / "rlnst" / "RL-NST" / "Healthy" / "c.png")
    _save(_noise(11), raw / "rlnst" / "RL-NST" / "Streak_virus" / "s.png")       # never trained on
    _save(_noise(4), raw / "cds" / "train" / "gls" / "d.png")
    _save(_noise(5), raw / "cds" / "test" / "nlb" / "e.png")
    return root


def _manifest(root):
    return read_manifest(root / "data" / "splits.csv")


def _dropped(root):
    with open(root / "data" / "splits_dropped.csv", newline="") as f:
        return list(csv.DictReader(f))


def test_unmapped_folders_are_ignored(tmp_path):
    make_repo(tmp_path)
    build(tmp_path)
    paths = " ".join(r["path"] for r in _manifest(tmp_path))
    assert "Streak_virus" not in paths and "Tomato" not in paths
    assert "Streak_virus" not in " ".join(r["path"] for r in _dropped(tmp_path))


def test_labels_and_sources_come_from_folders(tmp_path):
    make_repo(tmp_path)
    build(tmp_path)
    by = {Path(r["path"]).name: r for r in _manifest(tmp_path)}
    assert (by["a.png"]["source"], by["a.png"]["label"], by["a.png"]["original_folder"]) == ("kaggle", "healthy", "Healthy")
    assert (by["b.png"]["label"], by["d.png"]["label"], by["e.png"]["label"]) == ("leaf_blight", "leaf_spot", "leaf_blight")
    assert by["d.png"]["source"] == by["e.png"]["source"] == "cds"


def test_exact_duplicate_across_sources_kept_once_and_listed(tmp_path):
    make_repo(tmp_path)
    dup = tmp_path / "data" / "raw" / "rlnst" / "RL-NST" / "Healthy" / "a_copy.png"
    dup.write_bytes((tmp_path / "data" / "raw" / "data" / "Healthy" / "a.png").read_bytes())
    build(tmp_path)
    names = [Path(r["path"]).name for r in _manifest(tmp_path)]
    assert (names.count("a.png") + names.count("a_copy.png")) == 1
    rows = _dropped(tmp_path)
    assert len(rows) == 1 and rows[0]["reason"] == "duplicate"
    kept = next(r["path"] for r in _manifest(tmp_path) if Path(r["path"]).name in ("a.png", "a_copy.png"))
    assert rows[0]["duplicate_of"] == kept


def test_unreadable_file_is_recorded_not_fatal(tmp_path):
    make_repo(tmp_path)
    (tmp_path / "data" / "raw" / "data" / "Healthy" / "bad.jpg").write_text("not an image")
    counts = build(tmp_path)
    rows = _dropped(tmp_path)
    assert [r["reason"] for r in rows] == ["unreadable"]
    assert rows[0]["path"].endswith("bad.jpg") and rows[0]["duplicate_of"] == ""
    assert counts["dropped"]["unreadable"] == 1


def test_paths_are_repo_relative_and_files_exist(tmp_path):
    make_repo(tmp_path)
    build(tmp_path)
    for r in _manifest(tmp_path):
        assert not Path(r["path"]).is_absolute()
        assert (tmp_path / r["path"]).is_file()


def test_near_duplicate_pair_shares_group_and_split(tmp_path):
    make_repo(tmp_path)
    big = _structured(64)
    _save(big, tmp_path / "data" / "raw" / "data" / "Healthy" / "pair_a.png")
    _save(big.resize((48, 48), Image.LANCZOS), tmp_path / "data" / "raw" / "rlnst" / "RL-NST" / "Healthy" / "pair_b.jpg", quality=90)
    build(tmp_path)
    by = {Path(r["path"]).name: r for r in _manifest(tmp_path)}
    assert by["pair_a.png"]["group_id"] == by["pair_b.jpg"]["group_id"]
    assert by["pair_a.png"]["split"] == by["pair_b.jpg"]["split"]


def test_uppercase_extension_is_included(tmp_path):
    make_repo(tmp_path)
    _save(_noise(20), tmp_path / "data" / "raw" / "cds" / "test" / "nls" / "nls_197.JPG", format="JPEG")
    build(tmp_path)
    assert "nls_197.JPG" in [Path(r["path"]).name for r in _manifest(tmp_path)]


def test_same_inputs_give_identical_manifest(tmp_path):
    make_repo(tmp_path)
    build(tmp_path)
    first = (tmp_path / "data" / "splits.csv").read_bytes(), (tmp_path / "data" / "splits_dropped.csv").read_bytes()
    build(tmp_path)
    assert first == ((tmp_path / "data" / "splits.csv").read_bytes(), (tmp_path / "data" / "splits_dropped.csv").read_bytes())


def test_scan_source_returns_relative_paths_only_for_mapped_folders(tmp_path):
    make_repo(tmp_path)
    rows = scan_source("kaggle", tmp_path / SOURCES["kaggle"], tmp_path)
    assert sorted(r["original_folder"] for r in rows) == ["Blight", "Healthy"]
    assert all(r["path"].startswith("data/raw/data/") for r in rows)
