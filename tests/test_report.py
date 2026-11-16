from train.report import build_report, recommend


def _eval(acc, f1, sources, classes):
    return {"split": "test", "n": sum(n for n, _ in sources.values()),
            "metrics": {"accuracy": acc, "macro_f1": f1,
                        "per_class": {c: {"precision": f, "recall": f, "f1": f, "support": 10} for c, f in classes.items()},
                        "confusion": {}},
            "by_source": {s: {"n": n, "correct": round(n * a), "accuracy": a} for s, (n, a) in sources.items()}}


NEW = _eval(0.93, 0.86, {"cds": (100, 0.97), "kaggle": (100, 0.94), "up_field": (50, 0.80)},
            {"common_rust": 0.95, "healthy": 0.97})
BASE = _eval(0.913, 0.849, {"cds": (100, 0.96), "kaggle": (100, 0.93), "up_field": (50, 0.744)},
             {"common_rust": 0.918, "healthy": 0.967})
SWAP_NEW = {"original": {"all": [300, 323]}, "field": {"all": [292, 323]}}
SWAP_BASE = {"original": {"all": [304, 323]}, "field": {"all": [295, 323]}}


def test_report_compares_with_the_baseline_and_says_how_to_enable():
    text = build_report("corn_new", NEW, BASE, SWAP_NEW, SWAP_BASE, None, "corn_fungal_v6_bgaug")
    assert "| Test accuracy | 93.0% | 91.3% | +1.7 |" in text
    assert "| up_field | 50 | 80.0% | 74.4% | +5.6 |" in text
    assert "| common_rust | 95.0% | 91.8% | +3.2 |" in text
    assert "weights_path: models/corn_new" in text and "python -m uvicorn app.main:build_app --factory" in text


def test_report_without_baseline_still_lists_the_results():
    text = build_report("corn_new", NEW, None, None, None, None, None)
    assert "93.0%" in text and "Baseline" not in text and "weights_path: models/corn_new" in text


def test_recommend_switching_only_if_not_worse():
    assert recommend(NEW, BASE, SWAP_NEW, SWAP_BASE)[0] is True
    worse = _eval(0.90, 0.80, {"cds": (100, 0.9)}, {"healthy": 0.9})
    assert recommend(worse, BASE, None, None)[0] is False
    leaky = {"original": {"all": [310, 323]}, "field": {"all": [270, 323]}}     # -12 points on field backgrounds
    assert recommend(NEW, BASE, leaky, SWAP_BASE)[0] is False
