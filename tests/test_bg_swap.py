from pathlib import Path

from train.bg_swap import swap_pool, swap_test_rows


def _r(path, source, split, md5=None):
    return {"path": path, "source": source, "split": split, "md5": md5 or path, "label": "healthy"}


ROWS = [_r("a", "cds", "train"), _r("b", "cds", "dev"), _r("c", "up_field", "test"),
        _r("d", "kaggle", "test"), _r("e", "kaggle", "test"), _r("f", "rlnst", "train")]


def test_training_pool_is_train_split_field_photos():
    assert [p.name for p in swap_pool(ROWS, Path("."), "train")] == ["a"]


def test_unseen_pool_is_dev_and_test_field_photos_only():
    assert sorted(p.name for p in swap_pool(ROWS, Path("."), "unseen")) == ["b", "c"]


def test_test_rows_are_lab_style_test_photos_with_a_mask(tmp_path):
    (tmp_path / "d.png").write_bytes(b"x")                        # only d has a cached mask
    assert [r["path"] for r in swap_test_rows(ROWS, tmp_path)] == ["d"]
