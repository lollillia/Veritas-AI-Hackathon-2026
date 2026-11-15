"""Background-swap test: does a model rely on photo backgrounds?

    python -m train.bg_swap --model models/corn_fungal_v6_bgaug [--manifest data/splits_v5.csv] \
        [--pool train|unseen] [--json out.json]

Lab-style test photos (data_prep.background.LAB_SOURCES) that have a cached leaf mask are scored three times:
original background, a flat colour, and a patch of a real field photo. A model that relies on backgrounds loses
accuracy on the swapped versions. Backgrounds are seeded (123), so every model sees the same images.

--pool train (the original test) takes field patches from training-split field photos. Background-augmented
models were trained on those same patches, which favours them. --pool unseen takes them from dev and test
field photos instead, which no model trained on.
"""
import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

from data_prep.background import FIELD_SOURCES, LAB_SOURCES


def swap_pool(rows: list[dict], root: Path, which: str) -> list[Path]:
    """Field photos to cut background patches from: training split ("train") or dev + test ("unseen")."""
    splits = {"train": {"train"}, "unseen": {"dev", "test"}}[which]
    return [Path(root) / r["path"] for r in rows if r["source"] in FIELD_SOURCES and r["split"] in splits]


def swap_test_rows(rows: list[dict], mask_cache: Path) -> list[dict]:
    """Lab-style test photos with a cached leaf mask, sorted by path."""
    return sorted((r for r in rows if r["split"] == "test" and r["source"] in LAB_SOURCES
                   and (Path(mask_cache) / f"{r['md5']}.png").is_file()), key=lambda r: r["path"])


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--manifest", default=Path("data/splits_v5.csv"), type=Path)
    parser.add_argument("--pool", choices=["train", "unseen"], default="train")
    parser.add_argument("--mask-cache", default=Path("data/raw/mask_cache"), type=Path)
    parser.add_argument("--repo-root", default=Path("."), type=Path)
    parser.add_argument("--json", type=Path)
    args = parser.parse_args(argv)

    import torch
    from PIL import Image
    from transformers import AutoImageProcessor, AutoModelForImageClassification

    from data_prep.background import load_mask, replace_background
    from data_prep.manifest import read_manifest

    rows = read_manifest(args.manifest)
    test = swap_test_rows(rows, args.mask_cache)
    pool = swap_pool(rows, args.repo_root, args.pool)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    proc = AutoImageProcessor.from_pretrained(args.model)
    model = AutoModelForImageClassification.from_pretrained(args.model).to(device).eval()
    results = {}
    for variant in ("original", "flat", "field"):
        rng, correct = random.Random(123), defaultdict(lambda: [0, 0])
        for i in range(0, len(test), 32):
            batch, images = test[i:i + 32], []
            for r in batch:
                with Image.open(args.repo_root / r["path"]) as im:
                    im = im.convert("RGB")
                if variant != "original":
                    im = replace_background(im, load_mask(args.mask_cache, r["md5"]), rng, kind=variant, pool=pool)
                images.append(im)
            with torch.no_grad():
                logits = model(proc(images, return_tensors="pt")["pixel_values"].to(device)).logits
            for r, p in zip(batch, logits.argmax(-1).tolist()):
                for key in ("all", r["source"], r["label"]):
                    correct[key][0] += model.config.id2label[p] == r["label"]
                    correct[key][1] += 1
        results[variant] = {k: v for k, v in correct.items()}
        a, n = correct["all"]
        print(f"{variant:8s} {a}/{n} = {a / n:.1%}")
    print(f"field-background drop: {(results['field']['all'][0] - results['original']['all'][0]) / len(test):+.1%} "
          f"({len(test)} photos, pool={args.pool}, {len(pool)} field photos)")
    if args.json:
        args.json.write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
