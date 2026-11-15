"""Score a trained model on the dev or test split of the manifest.

    python -m train.evaluate --manifest data/splits.csv --model models/corn --split test [--json out.json]
    python -m train.evaluate --manifest data/splits_v2.csv --split all --source maize_multiformat

--split all is only meaningful for a source the model never trained on.

Prints per-class precision, recall and F1, a confusion matrix and per-source accuracy.
Requires: pip install -r requirements-ml.txt
"""
import argparse
import json
from pathlib import Path


def compute_metrics(true: list[str], pred: list[str], labels: list[str]) -> dict:
    """Accuracy, per-class precision/recall/F1 and a confusion matrix.

    A metric is None when its denominator is zero (precision: class never predicted; recall
    and F1: class has no true examples). Macro F1 averages only classes that have support.
    """
    known = set(labels)
    for value in [*true, *pred]:
        if value not in known:
            raise ValueError(f"label '{value}' is not in {labels}")
    confusion = {t: {p: 0 for p in labels} for t in labels}
    for t, p in zip(true, pred):
        confusion[t][p] += 1

    per_class = {}
    for label in labels:
        tp = confusion[label][label]
        support = sum(confusion[label].values())
        predicted = sum(confusion[t][label] for t in labels)
        precision = tp / predicted if predicted else None
        recall = tp / support if support else None
        if support == 0:
            f1 = None
        elif precision is None or precision + recall == 0:
            f1 = 0.0
        else:
            f1 = 2 * precision * recall / (precision + recall)
        per_class[label] = {"precision": precision, "recall": recall, "f1": f1, "support": support}

    f1s = [c["f1"] for c in per_class.values() if c["f1"] is not None]
    n = len(true)
    return {
        "accuracy": sum(t == p for t, p in zip(true, pred)) / n if n else None,
        "macro_f1": sum(f1s) / len(f1s) if f1s else 0.0,
        "per_class": per_class,
        "confusion": confusion,
    }


def select_rows(rows: list[dict], split: str, source: str | None = None) -> list[dict]:
    """Manifest rows for one split ("train", "dev", "test" or "all"), optionally one source, by path."""
    chosen = [r for r in rows if (split == "all" or r["split"] == split) and (source is None or r["source"] == source)]
    return sorted(chosen, key=lambda r: r["path"])


def metrics_by_source(rows: list[tuple[str, str, str]]) -> dict[str, dict]:
    """rows are (true, pred, source); returns {source: {"n", "correct", "accuracy"}}."""
    out: dict[str, dict] = {}
    for true, pred, source in rows:
        d = out.setdefault(source, {"n": 0, "correct": 0})
        d["n"] += 1
        d["correct"] += true == pred
    return {s: {**d, "accuracy": d["correct"] / d["n"]} for s, d in out.items()}


def _pct(x: float | None) -> str:
    return "n/a" if x is None else f"{x:.1%}"


def format_markdown(metrics: dict, by_source: dict, title: str) -> str:
    labels = list(metrics["per_class"])
    lines = [f"## {title}", "",
             f"- **Accuracy:** {_pct(metrics['accuracy'])}",
             f"- **Macro F1:** {_pct(metrics['macro_f1'])}", "",
             "| Class | Precision | Recall | F1 | Support |", "|---|---|---|---|---|"]
    for label, c in metrics["per_class"].items():
        lines.append(f"| {label} | {_pct(c['precision'])} | {_pct(c['recall'])} | {_pct(c['f1'])} | {c['support']} |")
    lines += ["", "Confusion matrix (rows = true, columns = predicted)", "",
              "| true \\ predicted | " + " | ".join(labels) + " |", "|---|" + "---|" * len(labels)]
    for t in labels:
        lines.append(f"| **{t}** | " + " | ".join(str(metrics["confusion"][t][p]) for p in labels) + " |")
    lines += ["", "| Source | Images | Correct | Accuracy |", "|---|---|---|---|"]
    for source, d in sorted(by_source.items()):
        lines.append(f"| {source} | {d['n']} | {d['correct']} | {_pct(d['accuracy'])} |")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--model", default="models/corn", type=Path)
    parser.add_argument("--split", required=True, choices=["dev", "test", "all"])
    parser.add_argument("--source", default=None, help="only images from this source")
    parser.add_argument("--repo-root", default=".", type=Path)
    parser.add_argument("--json", type=Path, help="also write the metrics here")
    args = parser.parse_args()

    import torch
    from PIL import Image
    from transformers import AutoImageProcessor, AutoModelForImageClassification

    from data_prep.manifest import read_manifest

    chosen = select_rows(read_manifest(args.manifest), args.split, args.source)
    if not chosen:
        raise SystemExit(f"No rows for split '{args.split}' and source '{args.source}' in {args.manifest}")
    for r in chosen:
        if not (args.repo_root / r["path"]).is_file():
            raise SystemExit(f"manifest file missing: {r['path']}")
    samples = [(args.repo_root / r["path"], r["label"]) for r in chosen]
    sources = [r["source"] for r in chosen]

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    processor = AutoImageProcessor.from_pretrained(args.model)
    model = AutoModelForImageClassification.from_pretrained(args.model).to(device).eval()
    labels = sorted(model.config.id2label.values())

    pred: list[str] = []
    with torch.no_grad():
        for i in range(0, len(samples), 32):
            batch = [Image.open(p).convert("RGB") for p, _ in samples[i:i + 32]]
            x = processor(batch, return_tensors="pt")["pixel_values"].to(device)
            pred += [model.config.id2label[j] for j in model(x).logits.argmax(-1).tolist()]

    true = [label for _, label in samples]
    metrics = compute_metrics(true, pred, labels)
    by_source = metrics_by_source(list(zip(true, pred, sources)))
    print(format_markdown(metrics, by_source, f"{args.split} split{' / ' + args.source if args.source else ''} ({len(samples)} images, model {args.model})"))
    if args.json:
        args.json.write_text(json.dumps({"split": args.split, "n": len(samples), "metrics": metrics, "by_source": by_source}, indent=2))


if __name__ == "__main__":
    main()
