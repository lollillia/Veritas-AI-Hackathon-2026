import csv

import pytest
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
    assert (by["b.png"]["label"], by["d.png"]["label"], by["e.png"]["label"]) == ("northern_leaf_blight", "gray_leaf_spot", "northern_leaf_blight")
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


def _bulk(root: Path, source_dir: str, folder: str, n: int, seed0: int):
    for i in range(n):
        _save(_noise(seed0 + i), root / "data" / "raw" / source_dir / folder / f"img_{i}.png")


def test_extra_source_and_custom_output_names_leave_the_original_manifest_alone(tmp_path):
    make_repo(tmp_path)
    build(tmp_path)
    original = (tmp_path / "data" / "splits.csv").read_bytes()
    _bulk(tmp_path, "extra", "Healthy_Leaf", 3, 500)
    sources = {**SOURCES, "extra": "data/raw/extra"}
    build(tmp_path, sources, manifest_name="splits_v2.csv", dropped_name="splits_v2_dropped.csv")
    assert (tmp_path / "data" / "splits.csv").read_bytes() == original
    v2 = read_manifest(tmp_path / "data" / "splits_v2.csv")
    assert (tmp_path / "data" / "splits_v2_dropped.csv").is_file()
    assert {r["source"] for r in v2} >= {"extra", "kaggle"}
    assert [r["label"] for r in v2 if r["source"] == "extra"] == ["healthy"] * 3


def test_existing_images_keep_their_split_when_the_new_source_has_no_near_duplicates_of_them(tmp_path):
    make_repo(tmp_path)
    _bulk(tmp_path, "data", "Healthy", 60, 1000)
    _bulk(tmp_path, "rlnst/RL-NST", "Leaf_Blight", 40, 2000)
    build(tmp_path)
    before = {r["md5"]: r["split"] for r in read_manifest(tmp_path / "data" / "splits.csv")}
    _bulk(tmp_path, "extra", "Healthy_Leaf", 50, 3000)
    sources = {**SOURCES, "extra": "data/raw/extra"}
    build(tmp_path, sources, manifest_name="splits_v2.csv", dropped_name="splits_v2_dropped.csv")
    after = {r["md5"]: r["split"] for r in read_manifest(tmp_path / "data" / "splits_v2.csv")}
    assert len(after) > len(before)
    assert all(after[m] == s for m, s in before.items())


def test_heic_file_is_recorded_as_unreadable(tmp_path):
    make_repo(tmp_path)
    (tmp_path / "data" / "raw" / "data" / "Healthy" / "phone.heic").write_bytes(b"\x00\x00\x00 ftypheic")
    build(tmp_path)
    rows = _dropped(tmp_path)
    assert [(Path(r["path"]).name, r["reason"]) for r in rows] == [("phone.heic", "unreadable")]


def test_cli_parse_args_extra_sources_and_output():
    from data_prep.build_splits import parse_args

    args = parse_args(["--add", "extra=data/raw/x y", "--add", "more=data/raw/z", "--out", "data/splits_v2.csv"])
    assert args.add == {"extra": "data/raw/x y", "more": "data/raw/z"}
    assert str(args.out) == "data/splits_v2.csv"
    default = parse_args([])
    assert default.add == {} and str(default.out) == "data/splits.csv"


def test_near_duplicates_with_different_labels_are_not_grouped(tmp_path):
    make_repo(tmp_path)
    big = _structured(64)
    _save(big, tmp_path / "data" / "raw" / "data" / "Healthy" / "smooth_a.png")
    _save(big.resize((48, 48), Image.LANCZOS), tmp_path / "data" / "raw" / "data" / "Blight" / "smooth_b.jpg", quality=90)
    build(tmp_path)
    by = {Path(r["path"]).name: r for r in _manifest(tmp_path)}
    assert by["smooth_a.png"]["group_id"] != by["smooth_b.jpg"]["group_id"]


def _v1_then_extra(tmp_path, n_extra=0):
    make_repo(tmp_path)
    _bulk(tmp_path, "data", "Healthy", 60, 1000)
    _bulk(tmp_path, "rlnst/RL-NST", "Leaf_Blight", 40, 2000)
    build(tmp_path)
    if n_extra:
        _bulk(tmp_path, "extra", "Healthy_Leaf", n_extra, 3000)
    return {**SOURCES, "extra": "data/raw/extra"}


def _build_v2(tmp_path, sources):
    return build(tmp_path, sources, manifest_name="splits_v2.csv", dropped_name="splits_v2_dropped.csv",
                 base_manifest=tmp_path / "data" / "splits.csv")


def test_base_manifest_keeps_every_existing_split_even_when_groups_change(tmp_path):
    sources = _v1_then_extra(tmp_path, n_extra=20)
    # a new near-duplicate of a base image, under a different label, used to reshuffle strata
    big = _structured(64)
    _save(big, tmp_path / "data" / "raw" / "data" / "Blight" / "smooth.png")
    build(tmp_path)                                         # refresh v1 with the smooth image
    _save(big.resize((48, 48), Image.LANCZOS), tmp_path / "data" / "raw" / "extra" / "Healthy_Leaf" / "smooth_copy.jpg", quality=90)
    before = {r["md5"]: r["split"] for r in read_manifest(tmp_path / "data" / "splits.csv")}
    counts = _build_v2(tmp_path, sources)
    after = {r["md5"]: r["split"] for r in read_manifest(tmp_path / "data" / "splits_v2.csv")}
    assert all(after[m] == s for m, s in before.items())
    assert counts["base_kept"] == len(before) and counts["base_missing"] == 0


def test_new_image_linked_to_a_base_image_inherits_its_split(tmp_path):
    sources = _v1_then_extra(tmp_path)
    big = _structured(64)
    _save(big, tmp_path / "data" / "raw" / "data" / "Healthy" / "pair_a.png")
    build(tmp_path)
    _save(big.resize((48, 48), Image.LANCZOS), tmp_path / "data" / "raw" / "extra" / "Healthy_Leaf" / "pair_b.jpg", quality=90)
    counts = _build_v2(tmp_path, sources)
    by = {Path(r["path"]).name: r for r in read_manifest(tmp_path / "data" / "splits_v2.csv")}
    assert by["pair_b.jpg"]["split"] == by["pair_a.png"]["split"]
    assert counts["new_linked_to_base"] == 1


def test_exact_duplicate_of_a_base_image_keeps_the_base_copy(tmp_path):
    make_repo(tmp_path)
    build(tmp_path)
    base_file = tmp_path / "data" / "raw" / "data" / "Healthy" / "a.png"
    early = tmp_path / "data" / "raw" / "AAA" / "Healthy_Leaf" / "a_again.png"    # sorts before data/raw/data
    early.parent.mkdir(parents=True)
    early.write_bytes(base_file.read_bytes())
    sources = {**SOURCES, "aaa": "data/raw/AAA"}
    _build_v2(tmp_path, sources)
    names = [Path(r["path"]).name for r in read_manifest(tmp_path / "data" / "splits_v2.csv")]
    assert "a.png" in names and "a_again.png" not in names
    with open(tmp_path / "data" / "splits_v2_dropped.csv", newline="") as f:
        dropped = list(csv.DictReader(f))
    assert [(Path(d["path"]).name, Path(d["duplicate_of"]).name) for d in dropped] == [("a_again.png", "a.png")]


def test_new_images_are_split_about_80_10_10(tmp_path):
    sources = _v1_then_extra(tmp_path, n_extra=100)
    _build_v2(tmp_path, sources)
    from collections import Counter
    c = Counter(r["split"] for r in read_manifest(tmp_path / "data" / "splits_v2.csv") if r["source"] == "extra")
    assert abs(c["train"] - 80) <= 2 and abs(c["dev"] - 10) <= 2 and abs(c["test"] - 10) <= 2


def test_base_rows_missing_from_disk_are_counted(tmp_path):
    sources = _v1_then_extra(tmp_path, n_extra=3)
    (tmp_path / "data" / "raw" / "data" / "Healthy" / "img_0.png").unlink()
    counts = _build_v2(tmp_path, sources)
    assert counts["base_missing"] == 1


def test_cli_parse_args_base():
    from data_prep.build_splits import parse_args

    args = parse_args(["--base", "data/splits.csv", "--out", "data/splits_v2.csv"])
    assert str(args.base) == "data/splits.csv"
    assert parse_args([]).base is None


def test_cli_refuses_to_overwrite_an_existing_manifest(tmp_path, monkeypatch):
    from data_prep.build_splits import main

    make_repo(tmp_path)
    build(tmp_path)
    original = (tmp_path / "data" / "splits.csv").read_bytes()
    monkeypatch.chdir(tmp_path)
    with pytest.raises(SystemExit, match="--force"):
        main([])
    assert (tmp_path / "data" / "splits.csv").read_bytes() == original
    main(["--force"])                                     # explicit overwrite still possible
    main(["--out", "data/splits_new.csv"])                # a new file is fine
    assert (tmp_path / "data" / "splits_new.csv").is_file()


def test_new_image_bridging_base_images_in_different_splits_is_left_out(tmp_path, monkeypatch):
    import hashlib

    import data_prep.build_splits as bs
    from data_prep.manifest import write_manifest

    make_repo(tmp_path)
    folder = tmp_path / "data" / "raw" / "data" / "Healthy"
    base_img = _structured(64)
    _save(base_img, folder / "a.png")                                     # three near-identical pictures
    _save(base_img.resize((60, 60), Image.LANCZOS), folder / "b.png")
    _save(base_img.resize((62, 62), Image.LANCZOS), folder / "bridge.png")
    md5 = lambda p: hashlib.md5(p.read_bytes()).hexdigest()
    base_rows = [{"path": f"data/raw/data/Healthy/{n}", "md5": md5(folder / n), "source": "kaggle",
                  "original_folder": "Healthy", "label": "healthy", "group_id": i, "split": s}
                 for i, (n, s) in enumerate([("a.png", "train"), ("b.png", "test")])]
    write_manifest(base_rows, tmp_path / "data" / "base.csv")
    real = bs.dhash
    fake = {"a.png": 0, "b.png": 0xFF, "bridge.png": 0x0F}                # a~bridge, bridge~b, a-b 8 bits
    monkeypatch.setattr(bs, "dhash", lambda p: fake.get(Path(p).name, real(p) | (1 << 63)))
    counts = build(tmp_path, manifest_name="v2.csv", dropped_name="v2_dropped.csv",
                   base_manifest=tmp_path / "data" / "base.csv")
    names = [Path(r["path"]).name for r in read_manifest(tmp_path / "data" / "v2.csv")]
    assert "bridge.png" not in names and "a.png" in names and "b.png" in names
    with open(tmp_path / "data" / "v2_dropped.csv", newline="") as f:
        dropped = {Path(d["path"]).name: d["reason"] for d in csv.DictReader(f)}
    assert dropped["bridge.png"] == "bridges_base_splits"
    assert counts["new_dropped_bridging"] == 1

def test_filename_time_parses_phone_names_only():
    from datetime import datetime

    from data_prep.build_splits import filename_time

    assert filename_time("IMG_20230728_171448.jpg") == datetime(2023, 7, 28, 17, 14, 48)
    assert filename_time("IMG_20230729_071305 (1).jpg") == datetime(2023, 7, 29, 7, 13, 5)
    assert filename_time("IMG_756.jpeg") is None and filename_time("Corn_Blight (77).jpg") is None


def test_shots_seconds_apart_share_a_group_and_split_even_if_they_look_different(tmp_path):
    make_repo(tmp_path)
    folder = tmp_path / "data" / "raw" / "data" / "Healthy"
    _save(_noise(7001), folder / "IMG_20230728_171448.jpg", format="JPEG")
    _save(_noise(7002), folder / "IMG_20230728_171500.jpg", format="JPEG")        # 12 s later
    _save(_noise(7003), folder / "IMG_20230728_171600.jpg", format="JPEG")        # 60 s later
    build(tmp_path)
    by = {Path(r["path"]).name: r for r in _manifest(tmp_path)}
    a, b, c = by["IMG_20230728_171448.jpg"], by["IMG_20230728_171500.jpg"], by["IMG_20230728_171600.jpg"]
    assert a["group_id"] == b["group_id"] and a["split"] == b["split"]
    assert c["group_id"] != a["group_id"]


def test_look_alikes_need_pixel_confirmation_within_and_across_sources(tmp_path, monkeypatch):
    import data_prep.build_splits as bs

    make_repo(tmp_path)
    a = tmp_path / "data" / "raw" / "data" / "Healthy" / "lookalike_a.png"
    b = tmp_path / "data" / "raw" / "rlnst" / "RL-NST" / "Healthy" / "lookalike_b.png"
    c = tmp_path / "data" / "raw" / "data" / "Healthy" / "lookalike_c.png"
    _save(_noise(8001), a); _save(_noise(8002), b); _save(_noise(8003), c)      # different pictures
    real = bs.dhash
    def stub(path):
        return 12345 if Path(path).name.startswith("lookalike_") else real(path) | (1 << 63)
    monkeypatch.setattr(bs, "dhash", stub)                                       # all three "look alike"
    build(tmp_path)
    by = {Path(r["path"]).name: r for r in _manifest(tmp_path)}
    assert by["lookalike_a.png"]["group_id"] != by["lookalike_c.png"]["group_id"]   # same source: not confirmed either
    assert by["lookalike_b.png"]["group_id"] != by["lookalike_a.png"]["group_id"]   # other source: not confirmed


def test_cross_source_resized_copy_is_still_grouped(tmp_path):
    make_repo(tmp_path)
    big = _structured(64)
    _save(big, tmp_path / "data" / "raw" / "data" / "Healthy" / "copy_a.png")
    _save(big.resize((48, 48), Image.LANCZOS), tmp_path / "data" / "raw" / "rlnst" / "RL-NST" / "Healthy" / "copy_b.jpg", quality=90)
    build(tmp_path)
    by = {Path(r["path"]).name: r for r in _manifest(tmp_path)}
    assert by["copy_a.png"]["group_id"] == by["copy_b.jpg"]["group_id"]


def test_build_includes_one_hot_csv_sources(tmp_path):
    make_repo(tmp_path)
    up = tmp_path / "data" / "raw" / "up"
    (up / "leaf_images").mkdir(parents=True)
    _save(_noise(9101), up / "leaf_images" / "x.JPG", format="JPEG")
    _save(_noise(9102), up / "leaf_images" / "y.JPG", format="JPEG")
    (up / "Database.csv").write_text("imgID_id,filePath,GLS,NCLB,PLS,CR,SR,NoFoliarSymptoms,Other,UnidentifiedDisease\n"
                                     "1,x.JPG,0,0,1,0,0,0,0,0\n2,y.JPG,1,1,0,0,0,0,0,0\n")
    counts = build(tmp_path, onehot_sources={"up": ("data/raw/up/Database.csv", "data/raw/up/leaf_images")})
    by = {Path(r["path"]).name: r for r in _manifest(tmp_path)}
    assert by["x.JPG"]["label"] == "phaeosphaeria_leaf_spot" and by["x.JPG"]["source"] == "up"
    assert "y.JPG" not in by
    assert counts["onehot_skipped"]["up"]["multi_label"] == 1


def test_cli_parse_args_onehot():
    from data_prep.build_splits import parse_args

    args = parse_args(["--add-onehot", "up=data/raw/up/Database.csv,data/raw/up/leaf_images"])
    assert args.add_onehot == {"up": ("data/raw/up/Database.csv", "data/raw/up/leaf_images")}
    assert parse_args([]).add_onehot == {}
