from collections import Counter, defaultdict

from data_prep.splitting import assign_splits


def _rows(counts, big_groups=()):
    """counts: {(source, label): n}. Singletons get unique group ids; big_groups adds
    (source, label, size) groups of identical group_id."""
    rows, gid = [], 0
    for (source, label), n in counts.items():
        for _ in range(n):
            rows.append({"source": source, "label": label, "group_id": gid})
            gid += 1
    for source, label, size in big_groups:
        for _ in range(size):
            rows.append({"source": source, "label": label, "group_id": gid})
        gid += 1
    return rows


MIX = {("kaggle", "healthy"): 300, ("kaggle", "leaf_spot"): 200, ("rlnst", "healthy"): 100,
       ("rlnst", "leaf_blight"): 250, ("cds", "leaf_spot"): 150}


def test_no_group_spans_two_splits():
    rows = _rows(MIX, big_groups=[("kaggle", "healthy", 12), ("cds", "leaf_spot", 7), ("rlnst", "healthy", 3)])
    splits = assign_splits(rows)
    seen = defaultdict(set)
    for row, s in zip(rows, splits):
        seen[row["group_id"]].add(s)
    assert all(len(v) == 1 for v in seen.values())


def test_overall_ratios_close_to_80_10_10():
    rows = _rows({("kaggle", "healthy"): 1000})
    counts = Counter(assign_splits(rows))
    assert abs(counts["train"] - 800) <= 2 and abs(counts["dev"] - 100) <= 2 and abs(counts["test"] - 100) <= 2


def test_every_stratum_with_enough_images_appears_in_all_splits():
    rows = _rows(MIX)
    splits = assign_splits(rows)
    per = defaultdict(set)
    for row, s in zip(rows, splits):
        per[(row["source"], row["label"])].add(s)
    assert all(v == {"train", "dev", "test"} for v in per.values())


def test_same_seed_reproduces_and_different_seed_differs():
    rows = _rows(MIX)
    assert assign_splits(rows, seed=0) == assign_splits(rows, seed=0)
    assert assign_splits(rows, seed=0) != assign_splits(rows, seed=1)


def test_tiny_strata_do_not_crash():
    rows = _rows({("a", "healthy"): 1, ("b", "healthy"): 2, ("c", "healthy"): 9})
    splits = assign_splits(rows)
    assert len(splits) == len(rows)
    assert splits[0] == "train"          # the single-image stratum
    assert set(splits) <= {"train", "dev", "test"}


def test_big_group_is_kept_whole_even_if_it_exceeds_the_dev_share():
    rows = _rows({("a", "healthy"): 70}, big_groups=[("a", "healthy", 30)])
    splits = assign_splits(rows)
    big = {s for r, s in zip(rows, splits) if r["group_id"] == 70}
    assert len(big) == 1


def test_empty_input_returns_empty_list():
    assert assign_splits([]) == []


def test_output_is_in_input_order_and_one_per_row():
    rows = _rows(MIX)
    assert len(assign_splits(rows)) == len(rows)
