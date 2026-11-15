"""Capture date, GPS location and camera for every image in a manifest, where the photo has them.

    python -m data_prep.image_metadata [--manifest data/splits_v2.csv] [--out data/image_metadata.csv]

Values come from the images' embedded EXIF data. Blank cells mean the image carries no such data
(most datasets strip it). GPS is rounded to 3 decimals (about 110 m) so exact farm locations are
not published with the repository.
"""
import argparse
import csv
import math
from datetime import datetime
from pathlib import Path

from PIL import Image

COLUMNS = ["path", "md5", "source", "capture_datetime", "gps_lat", "gps_lon", "camera"]
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


def write_metadata_csv(rows: list[dict], repo_root: Path, out_path: Path) -> dict:
    """One output row per manifest row; returns how many rows have each kind of value."""
    counts = {"images": 0, "capture_datetime": 0, "gps": 0, "camera": 0}
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            meta = read_photo_metadata(Path(repo_root) / row["path"])
            writer.writerow({"path": row["path"], "md5": row["md5"], "source": row["source"], **meta})
            counts["images"] += 1
            counts["capture_datetime"] += bool(meta["capture_datetime"])
            counts["gps"] += meta["gps_lat"] != ""
            counts["camera"] += bool(meta["camera"])
    return counts


def main() -> None:
    from data_prep.manifest import read_manifest

    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default=Path("data/splits_v2.csv"), type=Path)
    parser.add_argument("--out", default=Path("data/image_metadata.csv"), type=Path)
    args = parser.parse_args()
    counts = write_metadata_csv(read_manifest(args.manifest), Path.cwd(), args.out)
    print(f"wrote {args.out}: {counts}")


if __name__ == "__main__":
    main()
