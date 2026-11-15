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
    return parser.parse_args(argv)


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

    class Ds(torch.utils.data.Dataset):
        def __init__(self, items):
            self.items = items

        def __len__(self):
            return len(self.items)

        def __getitem__(self, i):
            path, label = self.items[i]
            image = Image.open(path).convert("RGB")
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
        train_dataset=Ds(train_samples),
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
