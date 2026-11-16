from train.weighting_check import compare


def _r(true, photo, weighted, source="cds"):
    return {"true": true, "photo_top": photo, "weighted_top": weighted, "source": source}


def test_compare_counts_accuracy_and_changes():
    rows = [_r("a", "a", "a"), _r("a", "b", "a"), _r("b", "b", "a"), _r("c", "a", "b")]
    c = compare(rows)
    assert c["n"] == 4 and c["photo_acc"] == 0.5 and c["weighted_acc"] == 0.5
    assert c["changed_to_right"] == 1 and c["changed_to_wrong"] == 1 and c["changed_wrong_to_wrong"] == 1


def test_compare_per_source():
    c = compare([_r("a", "a", "a", "cds"), _r("a", "b", "a", "ndisan"), _r("a", "b", "b", "ndisan")])
    assert c["by_source"]["cds"] == {"n": 1, "photo_acc": 1.0, "weighted_acc": 1.0}
    assert c["by_source"]["ndisan"] == {"n": 2, "photo_acc": 0.0, "weighted_acc": 0.5}
