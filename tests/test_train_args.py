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
