"""Weather leading up to each photo: summaries for the last 7 days, 1 month, 3 months and 6 months.

    python -m data_prep.weather [--metadata data/image_metadata.csv] [--out data/weather.csv]

One row per distinct (city, capture date) in image_metadata.csv. The city comes from the photo's
GPS or, failing that, its documented dataset site (see data_prep.locations); weather is queried at
the city's centre point. Each window covers the N days ending the day before the capture date, so
the capture day itself is not included. Source: Open-Meteo historical weather archive (ERA5
reanalysis, daily values on a roughly 10-25 km grid), https://open-meteo.com; one request per city
covering all its windows, cached in data/raw/weather_cache/.

Join back to image_metadata.csv on city and capture_date.
"""
import argparse
import csv
import json
import urllib.parse
import urllib.request
from datetime import date, timedelta
from pathlib import Path

API = "https://archive-api.open-meteo.com/v1/archive"
DAILY = ["temperature_2m_mean", "temperature_2m_max", "temperature_2m_min", "precipitation_sum",
         "relative_humidity_2m_mean"]
WINDOWS = {"7d": 7, "1m": 30, "3m": 90, "6m": 182}
RAINY_DAY_MM = 1.0
STATS = ["temp_mean_c", "temp_min_c", "temp_max_c", "precip_mm", "rainy_days", "rh_mean_pct", "days"]
COLUMNS = (["city", "city_lat", "city_lon", "capture_date", "timezone"]
           + [f"{w}_{s}" for w in WINDOWS for s in STATS] + ["source", "note"])
SOURCE = "Open-Meteo historical archive (ERA5 reanalysis)"


def weather_keys(rows: list[dict], locate) -> list[dict]:
    """Distinct (city, capture date) from metadata rows, sorted.

    locate(row) -> {"city", "city_lat", "city_lon"} (or None) gives the point to query for a row's city.
    """
    keys = {}
    for r in rows:
        if not (r.get("city") and r.get("capture_date")):
            continue
        k = (r["city"], r["capture_date"])
        if k not in keys:
            loc = locate(r)
            if loc and loc.get("city_lat") not in (None, ""):
                keys[k] = {"city": k[0], "capture_date": k[1],
                           "city_lat": float(loc["city_lat"]), "city_lon": float(loc["city_lon"])}
    return [keys[k] for k in sorted(keys)]


def _r(x: float) -> float:
    return round(x, 1)


def summarise(daily: dict, capture_date: str, n_days: int) -> dict:
    """Stats over the n_days ending the day before capture_date; "days" counts days with data."""
    end = date.fromisoformat(capture_date) - timedelta(days=1)
    start = end - timedelta(days=n_days - 1)
    idx = [i for i, t in enumerate(daily.get("time", [])) if start <= date.fromisoformat(t) <= end]
    def vals(name):
        series = daily.get(name, [])
        return [series[i] for i in idx if i < len(series) and series[i] is not None]
    mean, tmax, tmin = vals("temperature_2m_mean"), vals("temperature_2m_max"), vals("temperature_2m_min")
    rain, rh = vals("precipitation_sum"), vals("relative_humidity_2m_mean")
    return {
        "temp_mean_c": _r(sum(mean) / len(mean)) if mean else "",
        "temp_min_c": _r(min(tmin)) if tmin else "",
        "temp_max_c": _r(max(tmax)) if tmax else "",
        "precip_mm": _r(sum(rain)) if rain else "",
        "rainy_days": sum(x >= RAINY_DAY_MM for x in rain) if rain else "",
        "rh_mean_pct": _r(sum(rh) / len(rh)) if rh else "",
        "days": len(mean),
    }


def fetch_range(lat: float, lon: float, start: str, end: str, cache_dir: Path) -> dict:
    """Daily weather for one location and date range (local time), cached on disk."""
    cache = Path(cache_dir) / f"daily_{lat:.3f}_{lon:.3f}_{start}_{end}.json"
    if cache.is_file():
        return json.loads(cache.read_text())
    query = urllib.parse.urlencode({"latitude": f"{lat:.3f}", "longitude": f"{lon:.3f}", "start_date": start,
                                    "end_date": end, "daily": ",".join(DAILY), "timezone": "auto"})
    req = urllib.request.Request(f"{API}?{query}", headers={"User-Agent": "dataset-weather-lookup"})
    with urllib.request.urlopen(req, timeout=120) as resp:
        data = json.load(resp)
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(data))
    return data


def weather_rows(keys: list[dict], fetch) -> list[dict]:
    """fetch(lat, lon, start, end) is called once per city, covering every window of every date."""
    longest = max(WINDOWS.values())
    by_city: dict = {}
    for k in keys:
        by_city.setdefault((k["city"], k["city_lat"], k["city_lon"]), []).append(k["capture_date"])
    fetched = {}
    for (city, lat, lon), dates in by_city.items():
        start = (date.fromisoformat(min(dates)) - timedelta(days=longest)).isoformat()
        end = (date.fromisoformat(max(dates)) - timedelta(days=1)).isoformat()
        try:
            fetched[city] = (fetch(lat, lon, start, end), "")
        except Exception as exc:                           # network or API error: keep rows, blank weather
            fetched[city] = (None, f"fetch failed: {exc}")
    out = []
    for k in keys:
        data, error = fetched[k["city"]]
        row = {c: "" for c in COLUMNS}
        row.update({"city": k["city"], "city_lat": k["city_lat"], "city_lon": k["city_lon"],
                    "capture_date": k["capture_date"], "source": SOURCE, "note": error})
        if data is not None:
            row["timezone"] = data.get("timezone", "")
            short = []
            for w, n in WINDOWS.items():
                stats = summarise(data.get("daily", {}), k["capture_date"], n)
                row.update({f"{w}_{s}": v for s, v in stats.items()})
                if stats["days"] < n:
                    short.append(f"{w} has {stats['days']}/{n} days")
            row["note"] = "; ".join(short)
        out.append(row)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", default=Path("data/image_metadata.csv"), type=Path)
    parser.add_argument("--out", default=Path("data/weather.csv"), type=Path)
    parser.add_argument("--cache", default=Path("data/raw/weather_cache"), type=Path)
    args = parser.parse_args()
    from data_prep.image_metadata import locate_city
    from data_prep.locations import reverse_city

    geocode_cache = Path("data/raw/geocode_cache")
    resolve = lambda lat, lon: reverse_city(lat, lon, geocode_cache)        # cached by image_metadata
    with open(args.metadata, newline="") as f:
        keys = weather_keys(list(csv.DictReader(f)),
                            lambda r: locate_city(r["gps_lat"], r["gps_lon"], r["site"], resolve))
    rows = weather_rows(keys, lambda lat, lon, s, e: fetch_range(lat, lon, s, e, args.cache))
    with open(args.out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    complete = sum(r["note"] == "" for r in rows)
    print(f"wrote {args.out}: {len(rows)} city-date rows, {len({r['city'] for r in rows})} cities, "
          f"{complete} complete, {len(rows) - complete} with notes")


if __name__ == "__main__":
    main()
