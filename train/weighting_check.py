"""Does place + date weighting help? Photo-only vs weighted accuracy on test photos with a real city and date.

    python -m train.weighting_check --model models/corn_fungal_v6_bgaug --manifest data/splits_v6.csv \
        --out models/corn_fungal_v6_bgaug/weighting_check.md

The factors come from the sourced knowledge file, not from our data, so this check cannot be won by
memorising photo sessions. china_2 photos dated 2023-12-06 are skipped (a processing date, not a capture date).
"""
import argparse
from collections import defaultdict
from datetime import date
from pathlib import Path

UNRELIABLE_DATES = {("china_2", "2023-12-06")}


def compare(rows: list[dict]) -> dict:
    def acc(rs, key):
        return round(sum(r[key] == r["true"] for r in rs) / len(rs), 4) if rs else 0.0
    by = defaultdict(list)
    for r in rows:
        by[r["source"]].append(r)
    changed = [r for r in rows if r["photo_top"] != r["weighted_top"]]
    return {
        "n": len(rows), "photo_acc": acc(rows, "photo_top"), "weighted_acc": acc(rows, "weighted_top"),
        "changed": len(changed),
        "changed_to_right": sum(r["weighted_top"] == r["true"] for r in changed),
        "changed_to_wrong": sum(r["photo_top"] == r["true"] for r in changed),
        "changed_wrong_to_wrong": sum(r["photo_top"] != r["true"] and r["weighted_top"] != r["true"] for r in changed),
        "by_source": {s: {"n": len(rs), "photo_acc": acc(rs, "photo_top"), "weighted_acc": acc(rs, "weighted_top")}
                      for s, rs in sorted(by.items())},
    }


def main(argv=None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=Path("models/corn_fungal_v6_bgaug"), type=Path)
    parser.add_argument("--manifest", default=Path("data/splits_v6.csv"), type=Path)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args(argv)

    from transformers import pipeline

    from app.context.geocode import PlaceQuery, geocode
    from app.context.weather import recent_weather
    from app.imaging import load_rgb_image
    from app.knowledge.loader import load_knowledge
    from app.weighting import weigh
    from data_prep.image_metadata import read_metadata
    from data_prep.manifest import read_manifest

    meta = {m["md5"]: m for m in read_metadata(include_unused=True) if m["city"] and m["capture_date"]}
    test = [r for r in read_manifest(args.manifest) if r["split"] == "test" and r["md5"] in meta
            and (r["source"], meta[r["md5"]]["capture_date"]) not in UNRELIABLE_DATES]
    print(f"{len(test)} test photos with a city and capture date")
    knowledge, pipe = load_knowledge(), pipeline("image-classification", model=str(args.model))
    places, weathers, rows = {}, {}, []
    for i, r in enumerate(test):
        m = meta[r["md5"]]
        city, d = m["city"], date.fromisoformat(m["capture_date"][:10])
        if city not in places:
            places[city] = geocode(PlaceQuery(city=city))
        place = places[city]
        if place and (city, d) not in weathers:
            weathers[(city, d)] = recent_weather(place.lat, place.lon, d)
        scores = {a["label"]: float(a["score"]) for a in pipe(load_rgb_image(Path(r["path"]).read_bytes()), top_k=None)}
        w = weigh(scores, knowledge, place, d, weathers.get((city, d)))
        rows.append({"true": r["label"], "photo_top": max(scores, key=scores.get),
                     "weighted_top": max(w.final, key=w.final.get), "source": r["source"]})
        if i % 100 == 0:
            print(f"{i}/{len(test)}", flush=True)
    c = compare(rows)
    lines = [f"# Place + date weighting check: `{args.model}`", "",
             f"Test photos of `{args.manifest}` with a real city and capture date (`data/image_metadata.json`): "
             f"**{c['n']}**. Places found: {sum(p is not None for p in places.values())} of {len(places)}; "
             f"weather found for {sum(w is not None for w in weathers.values())} of {len(weathers)} city-days.", "",
             "| | Photo only | Weighted |", "|---|---|---|",
             f"| **Accuracy** | {c['photo_acc']:.1%} | {c['weighted_acc']:.1%} |", "",
             f"Answers changed by weighting: {c['changed']} (wrong → right {c['changed_to_right']}, "
             f"right → wrong {c['changed_to_wrong']}, wrong → other wrong {c['changed_wrong_to_wrong']}).", "",
             "| Source | Photos | Photo only | Weighted |", "|---|---|---|---|"]
    lines += [f"| {s} | {v['n']} | {v['photo_acc']:.1%} | {v['weighted_acc']:.1%} |" for s, v in c["by_source"].items()]
    text = "\n".join(lines) + "\n"
    print(text)
    if args.out:
        args.out.write_text(text)


if __name__ == "__main__":
    main()
