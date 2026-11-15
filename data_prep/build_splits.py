"""Build the pooled train/dev/test manifest.

    python -m data_prep.build_splits
    python -m data_prep.build_splits --add NAME=PATH [--add ...] --out data/splits_v2.csv

Scans the raw datasets, drops exact duplicates and unreadable files (recording them), groups
near-duplicate images, assigns splits, and writes data/splits.csv and data/splits_dropped.csv.
"""
import argparse
import hashlib
import sys
from collections import Counter
from pathlib import Path

from data_prep.grouping import dhash, group_ids
from data_prep.manifest import write_dropped, write_manifest
from data_prep.splitting import assign_splits
from train.labels import folder_to_label

SOURCES: dict[str, str] = {
    "kaggle": "data/raw/data",
    "rlnst": "data/raw/rlnst/RL-NST",
    "cds": "data/raw/cds",
}
# .heic is scanned only so unreadable phone photos show up in the dropped list.
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".heic"}


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
          manifest_name: str = "splits.csv", dropped_name: str = "splits_dropped.csv") -> dict:
    repo_root = Path(repo_root)
    out_dir = Path(out_dir) if out_dir else repo_root / "data"

    candidates = []
    for name in sorted(sources):
        candidates += scan_source(name, repo_root / sources[name], repo_root)
    candidates.sort(key=lambda r: r["path"])

    kept, hashes, dropped = [], [], []
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
        kept.append({**row, "md5": md5})
        hashes.append(h)

    for row, gid in zip(kept, group_ids(hashes, keys=[r["label"] for r in kept])):
        row["group_id"] = gid
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
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--add", action="append", default=[], metavar="NAME=PATH",
                        help="extra source folder, relative to the repo root (repeatable)")
    parser.add_argument("--out", default=Path("data/splits.csv"), type=Path,
                        help="manifest to write; the dropped list goes beside it as <name>_dropped.csv")
    args = parser.parse_args(argv)
    extra = {}
    for item in args.add:
        name, sep, path = item.partition("=")
        if not sep or not name or not path:
            parser.error(f"--add expects NAME=PATH, got '{item}'")
        extra[name] = path
    args.add = extra
    return args


def main() -> None:
    args = parse_args()
    root = Path.cwd()
    counts = build(root, {**SOURCES, **args.add}, out_dir=root / args.out.parent,
                   manifest_name=args.out.name, dropped_name=f"{args.out.stem}_dropped.csv")
    for key, value in counts.items():
        print(f"{key}: {value}")
    print(f"wrote {args.out} and {args.out.parent / (args.out.stem + '_dropped.csv')}", file=sys.stderr)


if __name__ == "__main__":
    main()
