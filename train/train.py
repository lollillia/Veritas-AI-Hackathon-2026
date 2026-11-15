"""Fine-tune a small image classifier on corn leaf photos.

Usage:
    python -m train.train --data <imagefolder dir> --out models/corn --epochs 3
    python -m train.train --manifest data/splits.csv --out models/corn --epochs 5

<imagefolder dir> has one sub-folder per class (e.g. PlantVillage layout). Only folders
recognised by train.labels.folder_to_label are used; all others are ignored.
Requires: pip install -r requirements-ml.txt
"""
import argparse
import random
from pathlib import Path

from train.labels import folder_to_label

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}


def collect_samples(data_dir: Path) -> list[tuple[Path, str]]:
    samples = []
    for folder in sorted(p for p in data_dir.iterdir() if p.is_dir()):
        label = folder_to_label(folder.name)
        if label is None:
            continue
        samples += [(f, label) for f in folder.iterdir() if f.suffix.lower() in IMAGE_SUFFIXES]
    return samples


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--data", type=Path, help="image-folder dataset; 10%% is held out at random")
    source.add_argument("--manifest", type=Path, help="data/splits.csv from data_prep.build_splits")
    parser.add_argument("--repo-root", default=".", type=Path, help="manifest paths are relative to this")
    parser.add_argument("--out", default="models/corn", type=Path)
    parser.add_argument("--epochs", default=3, type=int)
    parser.add_argument("--base-model", default="google/mobilenet_v2_1.0_224")
    parser.add_argument("--val-fraction", default=0.1, type=float)
    parser.add_argument("--bg-augment", action="store_true",
                        help="replace backgrounds of lab-style training photos (needs --manifest and cached masks)")
    parser.add_argument("--bg-prob", default=0.7, type=float, help="chance per training image and epoch")
    parser.add_argument("--mask-cache", default=Path("data/raw/mask_cache"), type=Path)
    args = parser.parse_args(argv)
    if args.bg_augment and not args.manifest:
        parser.error("--bg-augment needs --manifest")
    return args


def samples_from_manifest(manifest: Path, root: Path) -> tuple[list[tuple[Path, str]], list[tuple[Path, str]]]:
    """(train samples, dev samples) from the manifest. The test split is never returned."""
    from data_prep.manifest import read_manifest, samples_for_split

    rows = read_manifest(manifest)
    try:
        train = samples_for_split(rows, "train", root)
        dev = samples_for_split(rows, "dev", root)
    except FileNotFoundError as exc:
        raise SystemExit(str(exc))
    for name, part in (("train", train), ("dev", dev)):
        if not part:
            raise SystemExit(f"No {name} rows in {manifest}")
    return train, dev


def main() -> None:
    args = parse_args()

    import torch
    from PIL import Image
    from transformers import AutoImageProcessor, AutoModelForImageClassification, Trainer, TrainingArguments

    if args.manifest:
        train_samples, val_samples = samples_from_manifest(args.manifest, args.repo_root)
        samples = train_samples + val_samples
        print(f"manifest: {len(train_samples)} train, {len(val_samples)} dev images")
    else:
        samples = collect_samples(args.data)
        if not samples:
            raise SystemExit(f"No corn class folders with images found under {args.data}")
    labels = sorted({label for _, label in samples})
    label2id = {label: i for i, label in enumerate(labels)}
    id2label = {i: label for label, i in label2id.items()}
    print(f"{len(samples)} images, classes: {labels}")

    if not args.manifest:
        random.Random(0).shuffle(samples)
        n_val = max(1, int(len(samples) * args.val_fraction))
        val_samples, train_samples = samples[:n_val], samples[n_val:]

    processor = AutoImageProcessor.from_pretrained(args.base_model)

    augment_info, field_pool = {}, []
    if args.bg_augment:
        from data_prep.background import FIELD_SOURCES, LAB_SOURCES, load_mask, replace_background
        from data_prep.manifest import read_manifest

        train_rows = [r for r in read_manifest(args.manifest) if r["split"] == "train"]
        field_pool = [args.repo_root / r["path"] for r in train_rows if r["source"] in FIELD_SOURCES]
        for r in train_rows:
            if r["source"] in LAB_SOURCES and (args.mask_cache / f"{r['md5']}.png").is_file():
                augment_info[str(args.repo_root / r["path"])] = r["md5"]
        aug_rng = random.Random(0)
        print(f"background augmentation: {len(augment_info)} lab-style training images with masks, "
              f"{len(field_pool)} field photos as background patches, p={args.bg_prob}")

    class Ds(torch.utils.data.Dataset):
        def __init__(self, items, augment=False):
            self.items, self.augment = items, augment

        def __len__(self):
            return len(self.items)

        def __getitem__(self, i):
            path, label = self.items[i]
            image = Image.open(path).convert("RGB")
            md5 = augment_info.get(str(path)) if self.augment else None
            if md5 and aug_rng.random() < args.bg_prob:
                mask = load_mask(args.mask_cache, md5)
                if mask is not None:
                    image = replace_background(image, mask, aug_rng, pool=field_pool)
            pixel_values = processor(image, return_tensors="pt")["pixel_values"][0]
            return {"pixel_values": pixel_values, "labels": label2id[label]}

    model = AutoModelForImageClassification.from_pretrained(
        args.base_model,
        num_labels=len(labels),
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True,
    )

    from train.batchnorm import use_standard_batchnorm_momentum

    print(f"reset momentum on {use_standard_batchnorm_momentum(model)} BatchNorm layers")

    def accuracy(eval_pred):
        logits, ref = eval_pred
        return {"accuracy": float((logits.argmax(-1) == ref).mean())}

    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir=str(args.out / "_checkpoints"),
            num_train_epochs=args.epochs,
            per_device_train_batch_size=16,
            eval_strategy="epoch",
            save_strategy="no",
            remove_unused_columns=False,
            report_to=[],
        ),
        train_dataset=Ds(train_samples, augment=args.bg_augment),
        eval_dataset=Ds(val_samples),
        compute_metrics=accuracy,
    )
    trainer.train()
    print(trainer.evaluate())

    args.out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(args.out)
    processor.save_pretrained(args.out)
    print(f"Saved model to {args.out}")


if __name__ == "__main__":
    main()
