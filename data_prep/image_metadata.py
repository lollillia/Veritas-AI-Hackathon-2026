"""Capture date, GPS location and camera for every image in a manifest, where the photo has them.

    python -m data_prep.image_metadata [--manifest data/splits_v5.csv] [--index data/image_metadata.json]

Writes each dataset folder's rows to data/raw/<folder>/image_metadata.csv and a JSON index
(data/image_metadata.json) that links them; read everything with read_metadata().

Values come from the images' embedded EXIF data (capture time reduced to its date); "site" is the
dataset-level location where the source documents one; "city" is looked up from the photo's GPS, or
else from the site's coordinates. Blank cells mean unknown (most datasets strip EXIF). GPS is rounded to 3 decimals (about 110 m) so exact farm locations are
not published with the repository.
"""
import argparse
import csv
import math
from collections import Counter
from datetime import datetime
from pathlib import Path

from PIL import Image

from data_prep.locations import SITE_COORDS, SOURCE_SITES, reverse_city

COLUMNS = ["path", "md5", "source", "label", "capture_date", "gps_lat", "gps_lon", "camera", "site", "city"]
GPS_DECIMALS = 3
_EXIF_IFD, _GPS_IFD = 0x8769, 0x8825
_DATETIME_ORIGINAL, _DATETIME = 0x9003, 0x0132
_MAKE, _MODEL = 0x010F, 0x0110


def _iso(value) -> str:
    try:
        return datetime.strptime(str(value).strip().rstrip("\x00"), "%Y:%m:%d %H:%M:%S").isoformat()
    except (TypeError, ValueError):
        return ""


def _degrees(dms, ref) -> float:
    d, m, s = (float(x) for x in dms)
    value = d + m / 60 + s / 3600
    return -value if str(ref).upper() in ("S", "W") else value


def read_photo_metadata(path: Path | str) -> dict:
    """{"capture_datetime", "gps_lat", "gps_lon", "camera"}; empty strings where unavailable."""
    out = {"capture_datetime": "", "gps_lat": "", "gps_lon": "", "camera": ""}
    try:
        with Image.open(path) as img:
            exif = img.getexif()
            sub = exif.get_ifd(_EXIF_IFD)
            gps = exif.get_ifd(_GPS_IFD)
    except Exception:
        return out
    out["capture_datetime"] = _iso(sub.get(_DATETIME_ORIGINAL)) or _iso(exif.get(_DATETIME))
    try:
        if all(k in gps for k in (1, 2, 3, 4)):
            lat, lon = _degrees(gps[2], gps[1]), _degrees(gps[4], gps[3])
            # Some phones write placeholder GPS (null refs, 0/0 values); treat it as no location.
            if math.isfinite(lat) and math.isfinite(lon) and str(gps[1]).strip("\x00") and str(gps[3]).strip("\x00"):
                out["gps_lat"] = round(lat, GPS_DECIMALS)
                out["gps_lon"] = round(lon, GPS_DECIMALS)
    except (TypeError, ValueError, ZeroDivisionError):
        out["gps_lat"] = out["gps_lon"] = ""
    camera = " ".join(str(exif.get(t, "")).strip().rstrip("\x00") for t in (_MAKE, _MODEL)).strip()
    out["camera"] = camera
    return out


def locate_city(gps_lat, gps_lon, site: str, resolve_city) -> dict:
    """{"city", "city_lat", "city_lon"} from the photo's GPS, else from the documented site; blanks if neither."""
    if gps_lat != "" and gps_lon != "":
        return resolve_city(float(gps_lat), float(gps_lon))
    if site in SITE_COORDS:
        return resolve_city(*SITE_COORDS[site])
    return {"city": "", "city_lat": "", "city_lon": ""}


def write_metadata_csv(rows: list[dict], repo_root: Path, out_path: Path, resolve_city=None) -> dict:
    """One output row per manifest row; returns how many rows have each kind of value.

    resolve_city(lat, lon) -> {"city", ...} turns the photo's GPS, or else its documented site's
    coordinates, into a city; without it the city column stays blank. Capture times are reduced to
    their date.
    """
    counts = {"images": 0, "capture_date": 0, "gps": 0, "camera": 0, "city": 0}
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            meta = read_photo_metadata(Path(repo_root) / row["path"])
            site = SOURCE_SITES.get(row["source"], "")
            city = locate_city(meta["gps_lat"], meta["gps_lon"], site, resolve_city)["city"] if resolve_city else ""
            writer.writerow({"path": row["path"], "md5": row["md5"], "source": row["source"], "label": row.get("label", ""),
                             "capture_date": meta["capture_datetime"][:10], "gps_lat": meta["gps_lat"],
                             "gps_lon": meta["gps_lon"], "camera": meta["camera"], "site": site, "city": city})
            counts["images"] += 1
            counts["capture_date"] += bool(meta["capture_datetime"])
            counts["gps"] += meta["gps_lat"] != ""
            counts["camera"] += bool(meta["camera"])
            counts["city"] += bool(city)
    return counts


def write_per_folder(rows: list[dict], raw_root: Path) -> dict[str, int]:
    """Also write each dataset folder's rows to data/raw/<folder>/image_metadata.csv (same columns)."""
    by_folder: dict[str, list[dict]] = {}
    for row in rows:
        parts = Path(row["path"]).parts
        if len(parts) > 3 and parts[:2] == ("data", "raw"):
            by_folder.setdefault(parts[2], []).append(row)
    by_folder = {k: v for k, v in by_folder.items() if (Path(raw_root) / k).is_dir()}   # skip missing folders
    for folder, folder_rows in by_folder.items():
        with open(Path(raw_root) / folder / "image_metadata.csv", "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=COLUMNS, lineterminator="\n")
            writer.writeheader()
            writer.writerows(folder_rows)
    return {k: len(v) for k, v in by_folder.items()}


def unused_folders(raw_root: Path, per_folder: dict[str, int]) -> dict[str, int]:
    """{folder: rows} for dataset folders that have an image_metadata.csv but are not in the manifest."""
    out = {}
    for csv_path in sorted(Path(raw_root).glob("*/image_metadata.csv")):
        folder = csv_path.parent.name
        if folder not in per_folder:
            with open(csv_path, newline="") as f:
                out[folder] = sum(1 for _ in csv.DictReader(f))
    return out


def write_index(repo_root: Path, per_folder: dict[str, int], index_path: Path, manifest: str,
                unused: dict[str, int] | None = None) -> dict:
    """JSON index linking each dataset folder's image_metadata.csv (paths relative to the index).

    Folders in `unused` (metadata written, but not in the manifest) are listed with "in_manifest": false.
    """
    import json

    index_path = Path(index_path)
    unused = unused or {}
    datasets = []
    for folder in sorted({**per_folder, **unused}):
        csv_path = Path(repo_root) / "data" / "raw" / folder / "image_metadata.csv"
        with open(csv_path, newline="") as f:
            sources = Counter(r["source"] for r in csv.DictReader(f))
        entry = {"folder": folder, "source": sources.most_common(1)[0][0] if sources else "",
                 "images": per_folder.get(folder, unused.get(folder)),
                 "metadata": csv_path.resolve().relative_to(index_path.resolve().parent).as_posix()}
        if folder not in per_folder:
            entry["in_manifest"] = False
        datasets.append(entry)
    index = {
        "description": "Per-image metadata (label, capture date, GPS rounded to ~110 m, camera, documented site, "
                       "city), one CSV per dataset folder. Read all rows with data_prep.image_metadata.read_metadata. "
                       "total_images counts the manifest's images; datasets marked \"in_manifest\": false are not "
                       "in the manifest (unused_images) and are skipped unless read_metadata(include_unused=True).",
        "manifest": manifest, "columns": COLUMNS, "total_images": sum(per_folder.values()), "datasets": datasets,
    }
    if unused:
        index["unused_images"] = sum(unused.values())
    index_path.write_text(json.dumps(index, indent=2) + "\n")
    return index


def read_metadata(index_path: Path = Path("data/image_metadata.json"), include_unused: bool = False) -> list[dict]:
    """Metadata rows of the manifest's datasets (plus "in_manifest": false ones if include_unused), from the per-folder files."""
    import json

    index_path = Path(index_path)
    rows = []
    for d in json.loads(index_path.read_text())["datasets"]:
        if not include_unused and d.get("in_manifest", True) is False:
            continue
        p = index_path.parent / d["metadata"]
        if not p.is_file():
            raise FileNotFoundError(f"metadata file missing: {d['metadata']}")
        with open(p, newline="") as f:
            rows += list(csv.DictReader(f))
    return rows


def main() -> None:
    import tempfile

    from data_prep.manifest import read_manifest

    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default=Path("data/splits_v5.csv"), type=Path)
    parser.add_argument("--index", default=Path("data/image_metadata.json"), type=Path)
    args = parser.parse_args()
    cache = Path("data/raw/geocode_cache")
    with tempfile.TemporaryDirectory() as tmp:
        all_rows = Path(tmp) / "all.csv"
        counts = write_metadata_csv(read_manifest(args.manifest), Path.cwd(), all_rows,
                                    resolve_city=lambda lat, lon: reverse_city(lat, lon, cache))
        with open(all_rows, newline="") as f:
            per_folder = write_per_folder(list(csv.DictReader(f)), Path("data/raw"))
    unused = unused_folders(Path("data/raw"), per_folder)
    write_index(Path.cwd(), per_folder, args.index, args.manifest.as_posix(), unused=unused)
    print(f"metadata: {counts}")
    print(f"wrote image_metadata.csv into {len(per_folder)} dataset folders and the index {args.index}")


if __name__ == "__main__":
    main()
