from collections import Counter, defaultdict

from data_prep.date_split import split_by_date


def _row(md5, split, gid, source="ndisan", label="healthy"):
    return {"md5": md5, "split": split, "group_id": gid, "source": source, "label": label}


def test_each_city_date_stays_in_one_split_and_rows_without_dates_keep_theirs():
    rows, meta = [], {}
    gid = 0
    for d in range(40):                                         # 40 capture days, 10 photos each
        for k in range(10):
            md5 = f"d{d}_{k}"
            rows.append(_row(md5, ["train", "dev", "test"][k % 3], gid)); gid += 1
            meta[md5] = ("Sampang", f"2023-07-{d:02d}")
    for k in range(30):                                         # no metadata
        rows.append(_row(f"n{k}", "test" if k < 3 else "train", gid, source="rlnst")); gid += 1
    out = split_by_date(rows, meta)
    by_day = defaultdict(set)
    for r, s in zip(rows, out):
        if r["md5"] in meta:
            by_day[meta[r["md5"]]].add(s)
        else:
            assert s == r["split"]
    assert all(len(v) == 1 for v in by_day.values())
    c = Counter(s for r, s in zip(rows, out) if r["md5"] in meta)
    assert c["train"] >= 280 and c["dev"] >= 20 and c["test"] >= 20


def test_existing_groups_are_not_broken_across_days():
    rows = [_row("a", "train", 0), _row("b", "test", 0), _row("c", "dev", 1)]   # a and b were one group
    meta = {"a": ("X", "2020-01-01"), "c": ("X", "2020-01-02")}                 # b has no metadata
    out = split_by_date(rows, meta)
    assert out[0] == out[1]


def test_deterministic():
    rows = [_row(f"m{i}", "train", i) for i in range(50)]
    meta = {f"m{i}": ("X", f"2020-01-{i % 25:02d}") for i in range(50)}
    assert split_by_date(rows, meta) == split_by_date(rows, meta)
