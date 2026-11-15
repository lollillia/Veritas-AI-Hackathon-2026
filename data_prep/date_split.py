"""Date-held-out variant of a split: every (city, capture_date) sits entirely in one set.

    python -m data_prep.date_split [--manifest data/splits_v3.csv] [--out data/splits_v3_by_date.csv]

Weather summaries identify their date exactly, so a model given weather can memorise "this day ->
this class" when the same day appears in train and test. Here images with a city and date are
regrouped so whole days go to train, dev or test (stratified by source); existing near-duplicate and
burst groups stay together. Images without a city and date keep their original split.
"""
import argparse
import csv
from collections import Counter
from pathlib import Path

from data_prep.grouping import merge_partitions
from data_prep.manifest import read_manifest, write_manifest
from data_prep.splitting import assign_splits


def split_by_date(rows: list[dict], meta_by_md5: dict, seed: int = 0) -> list[str]:
    """New split per row; meta_by_md5 maps md5 -> (city, capture_date) for images that have both."""
    day_ids: dict = {}
    days = [day_ids.setdefault(meta_by_md5.get(r["md5"], ("", i)), len(day_ids)) for i, r in enumerate(rows)]
    combined = merge_partitions([r["group_id"] for r in rows], days)
    dated_groups = {g for g, r in zip(combined, rows) if r["md5"] in meta_by_md5}
    chosen = [i for i, g in enumerate(combined) if g in dated_groups]
    new = assign_splits([{"source": rows[i]["source"], "label": "*", "group_id": combined[i]} for i in chosen], seed=seed)
    out = [r["split"] for r in rows]
    for i, split in zip(chosen, new):
        out[i] = split
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default=Path("data/splits_v3.csv"), type=Path)
    parser.add_argument("--metadata", default=Path("data/image_metadata.csv"), type=Path)
    parser.add_argument("--out", default=Path("data/splits_v3_by_date.csv"), type=Path)
    args = parser.parse_args()
    rows = read_manifest(args.manifest)
    with open(args.metadata, newline="") as f:
        meta = {m["md5"]: (m["city"], m["capture_date"]) for m in csv.DictReader(f) if m["city"] and m["capture_date"]}
    new = split_by_date(rows, meta)
    moved = sum(r["split"] != s for r, s in zip(rows, new))
    for r, s in zip(rows, new):
        r["split"] = s
    write_manifest(rows, args.out)
    print(f"wrote {args.out}: {dict(Counter(new))}; {moved} images changed split")


if __name__ == "__main__":
    main()
