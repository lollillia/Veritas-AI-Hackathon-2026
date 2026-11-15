"""Historical weather at each distinct (GPS location, capture time) in image_metadata.csv.

    python -m data_prep.weather [--metadata data/image_metadata.csv] [--out data/weather.csv]

Source: Open-Meteo historical weather archive (ERA5 reanalysis, hourly, roughly 10-25 km grid),
https://open-meteo.com. One request per location and day; the hour nearest the capture time is
used. Capture times come from EXIF and carry no time zone, so they are treated as local time at
the site ("timezone=auto"). Raw responses are cached in data/raw/weather_cache/ (not committed).
"""
import argparse
import csv
import json
import urllib.parse
import urllib.request
from datetime import datetime, timedelta
from pathlib import Path

API = "https://archive-api.open-meteo.com/v1/archive"
HOURLY = ["temperature_2m", "relative_humidity_2m", "dew_point_2m", "precipitation",
          "cloud_cover", "wind_speed_10m", "weather_code"]
_OUT_NAMES = {
    "temperature_2m": "temperature_2m_c", "relative_humidity_2m": "relative_humidity_2m_pct",
    "dew_point_2m": "dew_point_2m_c", "precipitation": "precipitation_mm",
    "cloud_cover": "cloud_cover_pct", "wind_speed_10m": "wind_speed_10m_kmh", "weather_code": "weather_code",
}
COLUMNS = (["gps_lat", "gps_lon", "capture_datetime", "weather_hour_local", "timezone"]
           + [_OUT_NAMES[v] for v in HOURLY] + ["grid_lat", "grid_lon", "source", "note"])
SOURCE = "Open-Meteo historical archive (ERA5 reanalysis)"


def weather_keys(rows: list[dict]) -> list[tuple[float, float, str]]:
    """Distinct (lat, lon, capture_datetime) from metadata rows that have all three, sorted."""
    keys = {(float(r["gps_lat"]), float(r["gps_lon"]), r["capture_datetime"])
            for r in rows if r.get("gps_lat") and r.get("gps_lon") and r.get("capture_datetime")}
    return sorted(keys)


def nearest_hour(capture_datetime: str) -> str:
    t = datetime.fromisoformat(capture_datetime)
    t = (t + timedelta(minutes=30)).replace(minute=0, second=0, microsecond=0)
    return t.strftime("%Y-%m-%dT%H:00")


def fetch_day(lat: float, lon: float, date: str, cache_dir: Path) -> dict:
    """Hourly weather for one location and day (local time), cached on disk."""
    cache = Path(cache_dir) / f"{lat:.3f}_{lon:.3f}_{date}.json"
    if cache.is_file():
        return json.loads(cache.read_text())
    query = urllib.parse.urlencode({"latitude": f"{lat:.3f}", "longitude": f"{lon:.3f}", "start_date": date,
                                    "end_date": date, "hourly": ",".join(HOURLY), "timezone": "auto"})
    req = urllib.request.Request(f"{API}?{query}", headers={"User-Agent": "dataset-weather-lookup"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        data = json.load(resp)
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(data))
    return data


def weather_rows(keys: list[tuple[float, float, str]], fetch) -> list[dict]:
    """One output row per key; fetch(lat, lon, date) is called once per location and day."""
    days: dict = {}
    out = []
    for lat, lon, captured in keys:
        hour = nearest_hour(captured)
        day_key = (lat, lon, hour[:10])
        if day_key not in days:
            try:
                days[day_key] = (fetch(*day_key), "")
            except Exception as exc:                       # network or API error: keep the row, blank weather
                days[day_key] = (None, f"fetch failed: {exc}")
        data, error = days[day_key]
        row = {c: "" for c in COLUMNS}
        row.update({"gps_lat": lat, "gps_lon": lon, "capture_datetime": captured, "weather_hour_local": hour,
                    "source": SOURCE})
        if data is None:
            row["note"] = error
        else:
            row.update({"timezone": data.get("timezone", ""), "grid_lat": data.get("latitude", ""),
                        "grid_lon": data.get("longitude", "")})
            hourly = data.get("hourly", {})
            times = hourly.get("time", [])
            if hour in times:
                i = times.index(hour)
                for var in HOURLY:
                    values = hourly.get(var, [])
                    row[_OUT_NAMES[var]] = values[i] if i < len(values) and values[i] is not None else ""
            else:
                row["note"] = "hour not in response"
        out.append(row)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", default=Path("data/image_metadata.csv"), type=Path)
    parser.add_argument("--out", default=Path("data/weather.csv"), type=Path)
    parser.add_argument("--cache", default=Path("data/raw/weather_cache"), type=Path)
    args = parser.parse_args()
    with open(args.metadata, newline="") as f:
        keys = weather_keys(list(csv.DictReader(f)))
    rows = weather_rows(keys, lambda lat, lon, date: fetch_day(lat, lon, date, args.cache))
    with open(args.out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    filled = sum(r["temperature_2m_c"] != "" for r in rows)
    print(f"wrote {args.out}: {len(rows)} rows, {filled} with weather, {len(rows) - filled} blank")


if __name__ == "__main__":
    main()
