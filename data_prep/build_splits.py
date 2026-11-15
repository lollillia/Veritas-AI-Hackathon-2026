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

from data_prep.grouping import burst_group_ids, dhash, group_ids, merge_partitions, thumb_vector
from data_prep.manifest import read_manifest, write_dropped, write_manifest
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
          base_manifest: Path | None = None) -> dict:
    """Scan, de-duplicate, group and split; write the manifest and the dropped list.

    With base_manifest (an earlier splits.csv), every image already in it keeps its split
    (append-only): exact duplicates prefer the base copy, a new image grouped with a base
    image takes that image's split, and only the remaining new images are split 80/10/10.
    """
    repo_root = Path(repo_root)
    out_dir = Path(out_dir) if out_dir else repo_root / "data"
    base_rows = read_manifest(base_manifest) if base_manifest else []
    base_split = {r["md5"]: r["split"] for r in base_rows}
    base_paths = {r["path"] for r in base_rows}

    candidates = []
    for name in sorted(sources):
        candidates += scan_source(name, repo_root / sources[name], repo_root)
    candidates.sort(key=lambda r: (r["path"] not in base_paths, r["path"]))   # base copies win ties

    kept, dropped = [], []
    seen: dict[str, str] = {}                      # md5 -> path of the kept file
    for row in candidates:
        full = repo_root / row["path"]
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

    look_alike = group_ids([r.pop("_hash") for r in kept], keys=[r["label"] for r in kept], allow=confirmed)
    bursts = burst_group_ids([filename_time(Path(r["path"]).name) for r in kept],
                             [(r["source"], r["original_folder"], r["label"]) for r in kept], BURST_WINDOW_S)
    for row, gid in zip(kept, merge_partitions(look_alike, bursts)):
        row["group_id"] = gid

    extra: dict = {}
    if base_rows:
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
        **extra,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--add", action="append", default=[], metavar="NAME=PATH",
                        help="extra source folder, relative to the repo root (repeatable)")
    parser.add_argument("--out", default=Path("data/splits.csv"), type=Path,
                        help="manifest to write; the dropped list goes beside it as <name>_dropped.csv")
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
                   base_manifest=args.base)
    for key, value in counts.items():
        print(f"{key}: {value}")
    print(f"wrote {args.out} and {args.out.parent / (args.out.stem + '_dropped.csv')}", file=sys.stderr)


if __name__ == "__main__":
    main()
