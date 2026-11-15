"""Compute and cache leaf masks for lab-style sources (used by train.train --bg-augment).

    python -m data_prep.leaf_masks [--manifest data/splits_v5.csv] [--cache data/raw/mask_cache]

Writes <md5>.png for every image whose background is plain enough to cut out; images without a
usable mask are skipped (they are trained on unchanged).
"""
import argparse
from collections import Counter
from pathlib import Path

from PIL import Image

from data_prep.background import LAB_SOURCES, leaf_mask
from data_prep.manifest import read_manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default=Path("data/splits_v5.csv"), type=Path)
    parser.add_argument("--cache", default=Path("data/raw/mask_cache"), type=Path)
    args = parser.parse_args()
    args.cache.mkdir(parents=True, exist_ok=True)
    stats = Counter()
    for r in read_manifest(args.manifest):
        if r["source"] not in LAB_SOURCES:
            continue
        out = args.cache / f"{r['md5']}.png"
        if out.is_file():
            stats[(r["source"], "masked")] += 1
            continue
        with Image.open(r["path"]) as im:
            im.draft("RGB", (320, 320))
            m = leaf_mask(im)
        if m is None:
            stats[(r["source"], "no mask")] += 1
            continue
        Image.fromarray((m * 255).astype("uint8")).save(out)
        stats[(r["source"], "masked")] += 1
    for (source, kind), n in sorted(stats.items()):
        print(f"{source:8s} {kind:8s} {n}")


if __name__ == "__main__":
    main()
