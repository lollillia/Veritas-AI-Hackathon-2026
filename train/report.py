"""Write a performance report for a newly trained model, compared with the website's current model.

    python -m train.report --model models/corn_retrained --baseline models/corn_fungal_v6_bgaug

Reads <model>/eval_v6.json (train.evaluate --json), and if present <model>/bg_swap_unseen.json (train.bg_swap
--json) and <model>/weighting_check.md (train.weighting_check). Writes <model>/REPORT.md with a recommendation
and the steps to enable the model in the website.
"""
import argparse
import json
from pathlib import Path

FIELD_SOURCES = {"up_field", "cds", "mendeley_bd", "neau", "maize_multiformat", "china_2"}
MAX_EXTRA_BACKGROUND_DROP = 2.0      # points; a new model may lean on backgrounds at most this much more


def _pct(x: float | None) -> str:
    return "—" if x is None else f"{x * 100:.1f}%"


def _diff(a: float | None, b: float | None) -> str:
    return "—" if a is None or b is None else f"{(a - b) * 100:+.1f}"


def _field_acc(ev: dict) -> float | None:
    rows = [v for s, v in ev["by_source"].items() if s in FIELD_SOURCES]
    n = sum(v["n"] for v in rows)
    return sum(v["correct"] for v in rows) / n if n else None


def _swap_drop(swap: dict | None) -> float | None:
    if not swap:
        return None
    (o, n), (f, _) = swap["original"]["all"], swap["field"]["all"]
    return (f - o) / n


def recommend(new: dict, base: dict | None, swap_new: dict | None, swap_base: dict | None) -> tuple[bool, str]:
    """(switch?, reason). Switch only if test accuracy is not lower and background reliance is not clearly worse."""
    if base is None:
        return True, "No baseline to compare with."
    a_new, a_base = new["metrics"]["accuracy"], base["metrics"]["accuracy"]
    if a_new < a_base:
        return False, f"Test accuracy is lower than the current model ({_pct(a_new)} vs {_pct(a_base)})."
    d_new, d_base = _swap_drop(swap_new), _swap_drop(swap_base)
    if d_new is not None and d_base is not None and (d_base - d_new) * 100 > MAX_EXTRA_BACKGROUND_DROP:
        return False, (f"It relies more on backgrounds than the current model (field-background drop "
                       f"{d_new * 100:+.1f} vs {d_base * 100:+.1f} points).")
    return True, f"Test accuracy is at least as good ({_pct(a_new)} vs {_pct(a_base)}) and background reliance is not worse."


def build_report(name: str, new: dict, base: dict | None, swap_new: dict | None, swap_base: dict | None,
                 weighting_md: str | None, base_name: str | None) -> str:
    has_base = base is not None
    head = ["| | New model |" + (" Baseline (website) | Change |" if has_base else ""),
            "|---|---|" + ("---|---|" if has_base else "")]

    lines = [f"# Training report: `models/{name}`", ""]
    if has_base:
        lines += [f"Compared with the website's current model, `models/{base_name}`, on the same test photos "
                  f"(`data/splits_v6.csv`).", ""]
    summary = [("Test accuracy", new["metrics"]["accuracy"], base["metrics"]["accuracy"] if has_base else None),
               ("Macro F1", new["metrics"]["macro_f1"], base["metrics"]["macro_f1"] if has_base else None),
               ("Field-photo sources", _field_acc(new), _field_acc(base) if has_base else None)]
    lines += head
    for label, a, b in summary:
        lines.append(f"| {label} | {_pct(a)} |" + (f" {_pct(b)} | {_diff(a, b)} |" if has_base else ""))
    d_new, d_base = _swap_drop(swap_new), _swap_drop(swap_base)
    if d_new is not None:
        lines.append(f"| Drop on field backgrounds (lower is better) | {d_new * 100:+.1f} |"
                     + (f" {'—' if d_base is None else f'{d_base * 100:+.1f}'} | |" if has_base else ""))
    lines += ["", f"Test photos: {new['n']}.", ""]

    lines += ["## By source", "", "| Source | Photos | New model |" + (" Baseline | Change |" if has_base else ""),
              "|---|---|---|" + ("---|---|" if has_base else "")]
    for s, v in sorted(new["by_source"].items()):
        b = base["by_source"].get(s, {}).get("accuracy") if has_base else None
        lines.append(f"| {s} | {v['n']} | {_pct(v['accuracy'])} |" + (f" {_pct(b)} | {_diff(v['accuracy'], b)} |" if has_base else ""))

    lines += ["", "## By class (F1)", "", "| Class | New model |" + (" Baseline | Change |" if has_base else ""),
              "|---|---|" + ("---|---|" if has_base else "")]
    for c, v in sorted(new["metrics"]["per_class"].items()):
        b = base["metrics"]["per_class"].get(c, {}).get("f1") if has_base else None
        lines.append(f"| {c} | {_pct(v['f1'])} |" + (f" {_pct(b)} | {_diff(v['f1'], b)} |" if has_base else ""))

    if weighting_md:
        lines += ["", "## Place and date weighting check", "", weighting_md.split("\n", 1)[-1].strip()]

    switch, reason = recommend(new, base, swap_new, swap_base)
    lines += ["", "## Recommendation", "", f"**{'Switch the website to this model.' if switch else 'Keep the current model.'}** {reason}",
              "", "## Enable it in the website", "",
              "1. In `config.yaml`, set the detector and point it at this model:", "", "```yaml",
              "detector: finetuned", "finetuned:", f"  weights_path: models/{name}", "```", "",
              f"   (If you use `detector: gated`, change `gated.weights_path` to `models/{name}` instead.)", "",
              "2. Restart the website from the project folder, with the virtual environment active:", "",
              "```bash", "python -m uvicorn app.main:build_app --factory", "```", "",
              "3. Open http://127.0.0.1:8000 and check a few photos. To go back, set `weights_path` to the previous "
              "model folder and restart.", ""]
    return "\n".join(lines)


def _load(path: Path):
    return json.loads(path.read_text()) if path.is_file() else None


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--baseline", default=Path("models/corn_fungal_v6_bgaug"), type=Path)
    parser.add_argument("--out", type=Path, help="default: <model>/REPORT.md")
    args = parser.parse_args(argv)
    new = _load(args.model / "eval_v6.json")
    if new is None:
        raise SystemExit(f"{args.model / 'eval_v6.json'} not found; run train.evaluate --json first")
    base = _load(args.baseline / "eval_v6.json") if args.baseline else None
    wc = args.model / "weighting_check.md"
    text = build_report(args.model.name, new, base, _load(args.model / "bg_swap_unseen.json"),
                        _load(args.baseline / "bg_swap_unseen.json") if base else None,
                        wc.read_text() if wc.is_file() else None, args.baseline.name if base else None)
    out = args.out or args.model / "REPORT.md"
    out.write_text(text)
    print(text)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
