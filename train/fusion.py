"""Corn leaf classifier with optional weather input, for comparing image-only vs image + weather.

    python -m train.fusion --manifest data/splits_v3.csv --out models/fusion_v3_image --epochs 5
    python -m train.fusion --manifest data/splits_v3.csv --out models/fusion_v3_weather --epochs 5 --with-weather

Both variants use the same MobileNetV2 backbone, data, schedule and seed; --with-weather adds the
24 weather summaries (data_prep.weather_features) through a small layer joined to the image features
before the classifier. The date itself is never an input. Writes results.json (dev and test metrics,
split by images with and without weather) and model.pt in --out.
Requires: pip install -r requirements-ml.txt
"""
import argparse
import csv
import json
import random
from pathlib import Path

import torch
from torch import nn

from data_prep.weather_features import FEATURES, Standardizer, raw_weather_by_md5

N_TAB = len(FEATURES) + 1          # 24 standardised values + has-weather flag


class FusionClassifier(nn.Module):
    def __init__(self, backbone: nn.Module, feat_dim: int, num_labels: int, n_tab: int = N_TAB, use_tab: bool = False):
        super().__init__()
        self.backbone, self.use_tab = backbone, use_tab
        self.tab = nn.Sequential(nn.Linear(n_tab, 32), nn.ReLU()) if use_tab else None
        self.head = nn.Sequential(nn.Dropout(0.2), nn.Linear(feat_dim + (32 if use_tab else 0), num_labels))

    def forward(self, pixel_values, tab=None, labels=None):
        feats = self.backbone(pixel_values).pooler_output
        if self.use_tab:
            feats = torch.cat([feats, self.tab(tab)], dim=1)
        logits = self.head(feats)
        out = {"logits": logits}
        if labels is not None:
            out["loss"] = nn.functional.cross_entropy(logits, labels)
        return out


def main() -> None:
    from PIL import Image
    from transformers import AutoImageProcessor, AutoModel, get_linear_schedule_with_warmup

    from data_prep.manifest import read_manifest
    from train.batchnorm import use_standard_batchnorm_momentum
    from train.evaluate import compute_metrics, metrics_by_source

    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--metadata", default=Path("data/image_metadata.json"), type=Path)
    parser.add_argument("--weather", default=Path("data/weather.csv"), type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--epochs", default=5, type=int)
    parser.add_argument("--with-weather", action="store_true")
    parser.add_argument("--base-model", default="google/mobilenet_v2_1.0_224")
    parser.add_argument("--seed", default=0, type=int)
    args = parser.parse_args()

    random.seed(args.seed); torch.manual_seed(args.seed)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    rows = read_manifest(args.manifest)
    from data_prep.image_metadata import read_metadata

    meta = read_metadata(args.metadata)
    with open(args.weather, newline="") as f:
        raw = raw_weather_by_md5(meta, list(csv.DictReader(f)))
    labels = sorted({r["label"] for r in rows})
    label2id = {l: i for i, l in enumerate(labels)}
    split = {s: [r for r in rows if r["split"] == s] for s in ("train", "dev", "test")}
    scaler = Standardizer.fit([raw[r["md5"]] for r in split["train"] if r["md5"] in raw])
    print(f"train {len(split['train'])}, dev {len(split['dev'])}, test {len(split['test'])}; "
          f"with weather: {sum(r['md5'] in raw for r in rows)}; weather input: {args.with_weather}", flush=True)

    processor = AutoImageProcessor.from_pretrained(args.base_model)

    class Ds(torch.utils.data.Dataset):
        def __init__(self, items):
            self.items = items

        def __len__(self):
            return len(self.items)

        def __getitem__(self, i):
            r = self.items[i]
            with Image.open(r["path"]) as img:
                img.draft("RGB", (512, 512))                       # fast decode of large JPEGs
                px = processor(img.convert("RGB"), return_tensors="pt")["pixel_values"][0]
            tab = torch.tensor(scaler.transform(raw.get(r["md5"])), dtype=torch.float32)
            return px, tab, label2id[r["label"]]

    def loader(items, shuffle):
        return torch.utils.data.DataLoader(Ds(items), batch_size=16, shuffle=shuffle,
                                           generator=torch.Generator().manual_seed(args.seed))

    backbone = AutoModel.from_pretrained(args.base_model)
    model = FusionClassifier(backbone, backbone.config.hidden_size if hasattr(backbone.config, "hidden_size") else 1280,
                             len(labels), use_tab=args.with_weather)
    print(f"reset momentum on {use_standard_batchnorm_momentum(model)} BatchNorm layers", flush=True)
    model.to(device)
    train_dl = loader(split["train"], True)
    opt = torch.optim.AdamW(model.parameters(), lr=5e-5)
    sched = get_linear_schedule_with_warmup(opt, 0, len(train_dl) * args.epochs)

    @torch.no_grad()
    def predict(items):
        model.eval()
        preds = []
        for px, tab, _ in loader(items, False):
            preds += model(pixel_values=px.to(device), tab=tab.to(device))["logits"].argmax(-1).tolist()
        return [labels[p] for p in preds]

    for epoch in range(1, args.epochs + 1):
        model.train()
        total = 0.0
        for step, (px, tab, y) in enumerate(train_dl, 1):
            loss = model(pixel_values=px.to(device), tab=tab.to(device), labels=y.to(device))["loss"]
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
            total += loss.item()
            if step % 100 == 0:
                print(f"epoch {epoch} step {step}/{len(train_dl)} loss {total / step:.4f}", flush=True)
        dev_pred = predict(split["dev"])
        acc = sum(p == r["label"] for p, r in zip(dev_pred, split["dev"])) / len(split["dev"])
        print(f"epoch {epoch}: train loss {total / len(train_dl):.4f}, dev accuracy {acc:.4f}", flush=True)

    results = {"manifest": str(args.manifest), "with_weather": args.with_weather, "epochs": args.epochs}
    for name in ("dev", "test"):
        items = split[name]
        pred = predict(items)
        true = [r["label"] for r in items]
        has = [r["md5"] in raw for r in items]
        part = lambda keep: compute_metrics([t for t, k in zip(true, keep) if k], [p for p, k in zip(pred, keep) if k], labels)
        results[name] = {
            "n": len(items), "n_with_weather": sum(has),
            "all": compute_metrics(true, pred, labels),
            "with_weather": part(has), "without_weather": part([not h for h in has]),
            "by_source": metrics_by_source([(t, p, r["source"]) for t, p, r in zip(true, pred, items)]),
        }
        print(f"{name}: accuracy {results[name]['all']['accuracy']:.4f} | with weather "
              f"{results[name]['with_weather']['accuracy']} | without {results[name]['without_weather']['accuracy']}", flush=True)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "results.json").write_text(json.dumps(results, indent=2))
    torch.save({"state_dict": model.state_dict(), "labels": labels, "with_weather": args.with_weather,
                "scaler": scaler.to_dict(), "base_model": args.base_model}, args.out / "model.pt")
    print(f"saved {args.out}", flush=True)


if __name__ == "__main__":
    main()
