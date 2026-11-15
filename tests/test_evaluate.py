import pytest

from train.evaluate import compute_metrics, format_markdown, metrics_by_source

LABELS = ["a", "b"]


def test_perfect_predictions():
    m = compute_metrics(["a", "b", "a"], ["a", "b", "a"], LABELS)
    assert m["accuracy"] == 1.0 and m["macro_f1"] == 1.0
    assert all(m["per_class"][l]["precision"] == 1.0 and m["per_class"][l]["recall"] == 1.0 for l in LABELS)


def test_known_confusion_values():
    true = ["a", "a", "a", "b", "b", "b"]
    pred = ["a", "a", "b", "b", "b", "a"]
    m = compute_metrics(true, pred, LABELS)
    assert m["accuracy"] == pytest.approx(4 / 6)
    for l in LABELS:
        pc = m["per_class"][l]
        assert pc["precision"] == pytest.approx(2 / 3) and pc["recall"] == pytest.approx(2 / 3)
        assert pc["f1"] == pytest.approx(2 / 3) and pc["support"] == 3
    assert m["confusion"] == {"a": {"a": 2, "b": 1}, "b": {"a": 1, "b": 2}}
    assert m["macro_f1"] == pytest.approx(2 / 3)


def test_zero_support_class_reports_none_not_error():
    m = compute_metrics(["a", "a", "b"], ["a", "a", "b"], ["a", "b", "c"])
    c = m["per_class"]["c"]
    assert c == {"precision": None, "recall": None, "f1": None, "support": 0}
    assert m["macro_f1"] == 1.0                      # class c does not drag the average


def test_class_never_predicted_has_none_precision_and_zero_recall():
    m = compute_metrics(["a", "b", "b"], ["a", "a", "a"], LABELS)
    b = m["per_class"]["b"]
    assert b["precision"] is None and b["recall"] == 0.0 and b["f1"] == 0.0 and b["support"] == 2


def test_predicted_but_absent_class_has_zero_precision_and_none_recall():
    m = compute_metrics(["a", "a"], ["a", "b"], LABELS)
    b = m["per_class"]["b"]
    assert b["precision"] == 0.0 and b["recall"] is None and b["f1"] is None


def test_unknown_label_is_an_error():
    with pytest.raises(ValueError, match="zzz"):
        compute_metrics(["a"], ["zzz"], LABELS)


def test_empty_input_does_not_divide_by_zero():
    m = compute_metrics([], [], LABELS)
    assert m["accuracy"] is None and m["macro_f1"] == 0.0


def test_metrics_by_source_groups_correctly():
    rows = [("a", "a", "kaggle"), ("a", "b", "kaggle"), ("b", "b", "cds")]
    out = metrics_by_source(rows)
    assert out["kaggle"] == {"n": 2, "correct": 1, "accuracy": 0.5}
    assert out["cds"] == {"n": 1, "correct": 1, "accuracy": 1.0}


def test_format_markdown_shows_na_for_none():
    m = compute_metrics(["a", "b", "b"], ["a", "a", "a"], LABELS)
    text = format_markdown(m, metrics_by_source([("a", "a", "kaggle")]), "Dev split")
    assert "Dev split" in text and "n/a" in text and "kaggle" in text
    assert "None" not in text


def _r(path, split, source):
    return {"path": path, "split": split, "source": source, "label": "healthy"}


def test_select_rows_by_split_and_source():
    from train.evaluate import select_rows

    rows = [_r("b", "test", "x"), _r("a", "train", "x"), _r("c", "dev", "y"), _r("d", "test", "y")]
    assert [r["path"] for r in select_rows(rows, "test")] == ["b", "d"]
    assert [r["path"] for r in select_rows(rows, "all", "x")] == ["a", "b"]
    assert [r["path"] for r in select_rows(rows, "all")] == ["a", "b", "c", "d"]
    assert select_rows(rows, "dev", "x") == []
