import pytest

from data_prep.manifest import write_manifest
from train.train import parse_args, samples_from_manifest


def test_data_and_manifest_are_mutually_exclusive():
    with pytest.raises(SystemExit):
        parse_args(["--data", "d", "--manifest", "m.csv"])


def test_one_of_data_or_manifest_is_required():
    with pytest.raises(SystemExit):
        parse_args([])


def test_data_mode_defaults_unchanged():
    args = parse_args(["--data", "d"])
    assert str(args.data) == "d"
    assert args.epochs == 3 and args.val_fraction == 0.1 and args.manifest is None
    assert str(args.out) == "models/corn"


def test_manifest_mode_parses():
    args = parse_args(["--manifest", "m.csv", "--epochs", "5"])
    assert str(args.manifest) == "m.csv" and args.data is None and args.epochs == 5


def _row(path, split, label="healthy"):
    return {"path": path, "md5": path, "source": "kaggle", "original_folder": "Healthy",
            "label": label, "group_id": 0, "split": split}


def test_samples_from_manifest_returns_train_and_dev(tmp_path):
    for n in ("a.jpg", "b.jpg", "c.jpg", "d.jpg"):
        (tmp_path / n).write_bytes(b"x")
    write_manifest([_row("a.jpg", "train"), _row("b.jpg", "train", "leaf_spot"), _row("c.jpg", "dev"), _row("d.jpg", "test")],
                   tmp_path / "m.csv")
    train, dev = samples_from_manifest(tmp_path / "m.csv", tmp_path)
    assert [p.name for p, _ in train] == ["a.jpg", "b.jpg"]
    assert [p.name for p, _ in dev] == ["c.jpg"]       # test rows are never returned


def test_missing_file_exits_with_a_message_naming_it(tmp_path):
    (tmp_path / "a.jpg").write_bytes(b"x")
    write_manifest([_row("a.jpg", "train"), _row("gone/z.jpg", "dev")], tmp_path / "m.csv")
    with pytest.raises(SystemExit, match="gone/z.jpg"):
        samples_from_manifest(tmp_path / "m.csv", tmp_path)


def test_empty_train_or_dev_exits(tmp_path):
    (tmp_path / "a.jpg").write_bytes(b"x")
    write_manifest([_row("a.jpg", "train")], tmp_path / "m.csv")      # no dev rows
    with pytest.raises(SystemExit, match="dev"):
        samples_from_manifest(tmp_path / "m.csv", tmp_path)


def test_background_augmentation_flags():
    args = parse_args(["--manifest", "m.csv", "--bg-augment"])
    assert args.bg_augment and args.bg_prob == 0.7 and str(args.mask_cache) == "data/raw/mask_cache"
    assert not parse_args(["--manifest", "m.csv"]).bg_augment


def test_background_augmentation_needs_a_manifest():
    import pytest as _pytest
    with _pytest.raises(SystemExit):
        parse_args(["--data", "d", "--bg-augment"])


def test_std_augment_and_exclude_source_parse():
    args = parse_args(["--manifest", "m.csv", "--std-augment", "--exclude-source", "up_field", "--exclude-source", "cds"])
    assert args.std_augment is True and args.exclude_source == ["up_field", "cds"]
    plain = parse_args(["--manifest", "m.csv"])
    assert plain.std_augment is False and plain.exclude_source == []


def test_exclude_source_needs_a_manifest():
    with pytest.raises(SystemExit):
        parse_args(["--data", "d", "--exclude-source", "cds"])


def test_samples_from_manifest_can_leave_out_whole_sources(tmp_path):
    for n in ("a.jpg", "b.jpg", "c.jpg", "d.jpg"):
        (tmp_path / n).write_bytes(b"x")
    rows = [_row("a.jpg", "train"), _row("b.jpg", "train"), _row("c.jpg", "dev"), _row("d.jpg", "dev")]
    rows[1]["source"] = rows[3]["source"] = "up_field"
    write_manifest(rows, tmp_path / "m.csv")
    train, dev = samples_from_manifest(tmp_path / "m.csv", tmp_path, exclude_sources={"up_field"})
    assert [p.name for p, _ in train] == ["a.jpg"] and [p.name for p, _ in dev] == ["c.jpg"]


def test_background_pool_leaves_out_excluded_sources():
    from train.train import background_pool
    rows = [{"path": "a.jpg", "source": "cds", "split": "train"},
            {"path": "b.jpg", "source": "up_field", "split": "train"},
            {"path": "c.jpg", "source": "cds", "split": "dev"},
            {"path": "d.jpg", "source": "kaggle", "split": "train"}]
    from pathlib import Path
    pool = background_pool(rows, Path("."), exclude_sources={"up_field"})
    assert [p.name for p in pool] == ["a.jpg"]          # field, train split, not excluded
