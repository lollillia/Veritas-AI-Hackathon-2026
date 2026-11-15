"""Build the pooled train/dev/test manifest.

    python -m data_prep.build_splits
    python -m data_prep.build_splits --add NAME=PATH [--add ...] --base data/splits.csv --out data/splits_v2.csv

Scans the raw datasets, drops exact duplicates and unreadable files (recording them), groups
near-duplicate images, assigns splits, and writes data/splits.csv and data/splits_dropped.csv.
"""
import argparse
import hashlib
import re
import sys
from datetime import datetime
from collections import Counter
from pathlib import Path

import numpy as np

from data_prep.grouping import _popcount, burst_group_ids, dhash, group_ids, merge_partitions, thumb_vector
from data_prep.manifest import read_manifest, write_dropped, write_manifest
from data_prep.metadata_source import scan_metadata_csv
from data_prep.onehot_source import scan_onehot_csv
from data_prep.splitting import assign_splits
from train.labels import folder_to_label

SOURCES: dict[str, str] = {
    "kaggle": "data/raw/data",
    "rlnst": "data/raw/rlnst/RL-NST",
    "cds": "data/raw/cds",
}
# Shots whose phone-style file names (IMG_YYYYMMDD_HHMMSS) are this close are treated as the same
# leaf: the ndisan dataset photographs each leaf 5 times in a row.
BURST_WINDOW_S = 20
# Look-alike (hash) matches must be confirmed by pixel correlation: plain leaves with similar framing can
# have close hashes and chain into huge false groups (real copies, even resized, score about 0.99).
# Repeated shots of one leaf are grouped by filename timestamps instead (BURST_WINDOW_S).
LOOK_ALIKE_MIN_CORRELATION = 0.95
_PHONE_NAME = re.compile(r"IMG_(\d{8})_(\d{6})")
# .heic is scanned only so unreadable phone photos show up in the dropped list.
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".heic"}


def filename_time(name: str) -> datetime | None:
    """Capture time encoded in a phone-style file name, or None."""
    m = _PHONE_NAME.search(name)
    if not m:
        return None
    try:
        return datetime.strptime(m.group(1) + m.group(2), "%Y%m%d%H%M%S")
    except ValueError:
        return None


def scan_source(name: str, root: Path, repo_root: Path) -> list[dict]:
    """Every image directly inside any folder (at any depth) whose name maps to a label."""
    rows = []
    for folder in sorted([root, *root.rglob("*")], key=str):
        if not folder.is_dir():
            continue
        label = folder_to_label(folder.name)
        if label is None:
            continue
        for f in sorted(folder.iterdir()):
            if f.is_file() and f.suffix.lower() in IMAGE_SUFFIXES:
                rows.append({
                    "path": f.relative_to(repo_root).as_posix(),
                    "source": name,
                    "original_folder": folder.name,
                    "label": label,
                })
    return rows


def build(repo_root: Path, sources: dict[str, str] = SOURCES, out_dir: Path | None = None,
          manifest_name: str = "splits.csv", dropped_name: str = "splits_dropped.csv",
          base_manifest: Path | None = None, onehot_sources: dict | None = None,
          metadata_sources: dict | None = None, exclude: set | None = None) -> dict:
    """Scan, de-duplicate, group and split; write the manifest and the dropped list.

    With base_manifest (an earlier splits.csv), every image already in it keeps its split
    (append-only): exact duplicates prefer the base copy, a new image grouped with a base
    image takes that image's split, and only the remaining new images are split 80/10/10. A new image
    that is a (pixel-confirmed) copy of an image with a different label is left out.

    metadata_sources maps a source name to a dataset's image_metadata.csv (labels from its label column);
    exclude is a set of repo-relative paths to leave out (recorded as "excluded"), e.g. mislabelled images.
    """
    repo_root = Path(repo_root)
    out_dir = Path(out_dir) if out_dir else repo_root / "data"
    base_rows = read_manifest(base_manifest) if base_manifest else []
    base_split = {r["md5"]: r["split"] for r in base_rows}
    base_paths = {r["path"] for r in base_rows}

    candidates = []
    for name in sorted(sources):
        candidates += scan_source(name, repo_root / sources[name], repo_root)
    onehot_skipped = {}
    for name, (csv_path, image_dir) in sorted((onehot_sources or {}).items()):
        rows, onehot_skipped[name] = scan_onehot_csv(name, repo_root / csv_path, repo_root / image_dir, repo_root)
        candidates += rows
    metadata_skipped = {}
    for name, csv_path in sorted((metadata_sources or {}).items()):
        rows, metadata_skipped[name] = scan_metadata_csv(name, repo_root / csv_path, repo_root)
        candidates += rows
    candidates.sort(key=lambda r: (r["path"] not in base_paths, r["path"]))   # base copies win ties

    kept, dropped = [], []
    seen: dict[str, str] = {}                      # md5 -> path of the kept file
    exclude = set(exclude or ())
    for row in candidates:
        full = repo_root / row["path"]
        if row["path"] in exclude:
            dropped.append({"path": row["path"], "md5": "", "reason": "excluded", "duplicate_of": ""})
            continue
        md5 = hashlib.md5(full.read_bytes()).hexdigest()
        try:
            h = dhash(full)
        except ValueError:
            dropped.append({"path": row["path"], "md5": md5, "reason": "unreadable", "duplicate_of": ""})
            continue
        if md5 in seen:
            dropped.append({"path": row["path"], "md5": md5, "reason": "duplicate", "duplicate_of": seen[md5]})
            continue
        seen[md5] = row["path"]
        kept.append({**row, "md5": md5, "_hash": h})
    kept.sort(key=lambda r: r["path"])
    dropped.sort(key=lambda d: d["path"])

    thumbs: dict[int, object] = {}

    def confirmed(i: int, j: int) -> bool:
        for k in (i, j):
            if k not in thumbs:
                thumbs[k] = thumb_vector(repo_root / kept[k]["path"])
        return float(thumbs[i] @ thumbs[j]) >= LOOK_ALIKE_MIN_CORRELATION

    hashes = [r.pop("_hash") for r in kept]
    look_alike = group_ids(hashes, keys=[r["label"] for r in kept], allow=confirmed)
    bursts = burst_group_ids([filename_time(Path(r["path"]).name) for r in kept],
                             [(r["source"], r["original_folder"], r["label"]) for r in kept], BURST_WINDOW_S)
    for row, gid in zip(kept, merge_partitions(look_alike, bursts)):
        row["group_id"] = gid

    extra: dict = {}
    if base_rows:
        # A new image that copies an image filed under another label: one of the two labels is wrong, and
        # keeping both could put the same picture in train and test with different answers.
        arr = np.array(hashes, dtype=np.uint64)
        other_label: dict[int, int] = {}
        for i, row in enumerate(kept):
            if row["md5"] in base_split:
                continue
            for j in np.nonzero(_popcount(arr ^ arr[i]) <= 4)[0].tolist():
                if j != i and kept[j]["label"] != row["label"] and confirmed(i, j):
                    other_label[i] = j
                    break
        for i, j in other_label.items():
            dropped.append({"path": kept[i]["path"], "md5": kept[i]["md5"], "reason": "copy_with_other_label",
                            "duplicate_of": kept[j]["path"]})
        kept = [r for i, r in enumerate(kept) if i not in other_label]
        group_split: dict[int, str] = {}
        base_splits_of: dict[int, set] = {}
        for row in kept:
            if row["md5"] in base_split:
                group_split.setdefault(row["group_id"], base_split[row["md5"]])
                base_splits_of.setdefault(row["group_id"], set()).add(base_split[row["md5"]])
        conflicting = {g for g, splits in base_splits_of.items() if len(splits) > 1}
        # A new image grouped with base images from different splits would be a near-duplicate
        # of an image in another split whatever split it got, so it is left out and recorded.
        bridging = [r for r in kept if r["md5"] not in base_split and r["group_id"] in conflicting]
        for r in bridging:
            dropped.append({"path": r["path"], "md5": r["md5"], "reason": "bridges_base_splits", "duplicate_of": ""})
        kept = [r for r in kept if not (r["md5"] not in base_split and r["group_id"] in conflicting)]
        dropped.sort(key=lambda d: d["path"])
        fresh = [r for r in kept if r["md5"] not in base_split and r["group_id"] not in group_split]
        for row, split in zip(fresh, assign_splits(fresh)):
            row["split"] = split
        linked = 0
        for row in kept:
            if row["md5"] in base_split:
                row["split"] = base_split[row["md5"]]
            elif row["group_id"] in group_split:
                row["split"] = group_split[row["group_id"]]
                linked += 1
        kept_md5 = {r["md5"] for r in kept}
        extra = {
            "base_kept": sum(r["md5"] in base_split for r in kept),
            "base_missing": sum(m not in kept_md5 for m in base_split),
            "new_images": sum(r["md5"] not in base_split for r in kept),
            "new_linked_to_base": linked,
            "groups_with_conflicting_base_splits": len(conflicting),
            "new_dropped_bridging": len(bridging),
            "new_dropped_other_label_copy": len(other_label),
        }
    else:
        for row, split in zip(kept, assign_splits(kept)):
            row["split"] = split

    write_manifest(kept, out_dir / manifest_name)
    write_dropped(dropped, out_dir / dropped_name)

    sizes = Counter(r["group_id"] for r in kept)
    return {
        "total": len(kept),
        "by_split": dict(Counter(r["split"] for r in kept)),
        "by_source": dict(Counter(r["source"] for r in kept)),
        "by_label": dict(Counter(r["label"] for r in kept)),
        "by_split_source": {f"{s}/{src}": n for (s, src), n in sorted(Counter((r["split"], r["source"]) for r in kept).items())},
        "by_split_label": {f"{s}/{lab}": n for (s, lab), n in sorted(Counter((r["split"], r["label"]) for r in kept).items())},
        "dropped": dict(Counter(d["reason"] for d in dropped)),
        "multi_image_group_images": sum(n for n in sizes.values() if n > 1),
        "largest_group": max(sizes.values(), default=0),
        **({"onehot_skipped": onehot_skipped} if onehot_skipped else {}),
        **({"metadata_skipped": metadata_skipped} if metadata_skipped else {}),
        **extra,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--add", action="append", default=[], metavar="NAME=PATH",
                        help="extra source folder, relative to the repo root (repeatable)")
    parser.add_argument("--out", default=Path("data/splits.csv"), type=Path,
                        help="manifest to write; the dropped list goes beside it as <name>_dropped.csv")
    parser.add_argument("--add-onehot", action="append", default=[], metavar="NAME=CSV,IMAGE_DIR",
                        help="extra source labelled by a one-hot CSV (repeatable)")
    parser.add_argument("--add-metadata", action="append", default=[], metavar="NAME=CSV",
                        help="extra source labelled by its image_metadata.csv label column (repeatable)")
    parser.add_argument("--exclude", action="append", default=[], metavar="PATH",
                        help="repo-relative image path to leave out, e.g. a mislabelled image (repeatable)")
    parser.add_argument("--force", action="store_true",
                        help="overwrite --out if it already exists")
    parser.add_argument("--base", type=Path, default=None,
                        help="earlier manifest whose images keep their split (append-only)")
    args = parser.parse_args(argv)
    extra = {}
    for item in args.add:
        name, sep, path = item.partition("=")
        if not sep or not name or not path:
            parser.error(f"--add expects NAME=PATH, got '{item}'")
        extra[name] = path
    args.add = extra
    onehot = {}
    for item in args.add_onehot:
        name, sep, rest = item.partition("=")
        csv_path, comma, image_dir = rest.partition(",")
        if not (sep and name and comma and csv_path and image_dir):
            parser.error(f"--add-onehot expects NAME=CSV,IMAGE_DIR, got '{item}'")
        onehot[name] = (csv_path, image_dir)
    args.add_onehot = onehot
    metadata = {}
    for item in args.add_metadata:
        name, sep, csv_path = item.partition("=")
        if not (sep and name and csv_path):
            parser.error(f"--add-metadata expects NAME=CSV, got '{item}'")
        metadata[name] = csv_path
    args.add_metadata = metadata
    return args


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    root = Path.cwd()
    if (root / args.out).exists() and not args.force:
        raise SystemExit(
            f"{args.out} already exists and may be the manifest a model was trained on; rebuilding it "
            f"can move images between splits. To add data use --base {args.out} --out <new file>; "
            f"to overwrite anyway pass --force.")
    counts = build(root, {**SOURCES, **args.add}, out_dir=root / args.out.parent,
                   manifest_name=args.out.name, dropped_name=f"{args.out.stem}_dropped.csv",
                   base_manifest=args.base, onehot_sources=args.add_onehot,
                   metadata_sources=args.add_metadata, exclude=set(args.exclude))
    for key, value in counts.items():
        print(f"{key}: {value}")
    print(f"wrote {args.out} and {args.out.parent / (args.out.stem + '_dropped.csv')}", file=sys.stderr)


if __name__ == "__main__":
    main()
