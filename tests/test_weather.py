from datetime import date, timedelta

import pytest

from data_prep.weather import COLUMNS, WINDOWS, summarise, weather_keys, weather_rows


def _m(city, t, lat=1.0, lon=2.0):
    return {"city": city, "capture_date": t[:10], "_lat": lat, "_lon": lon}


def locate(row):
    return {"city": row["city"], "city_lat": row["_lat"], "city_lon": row["_lon"]} if row["city"] else None


def test_keys_are_distinct_city_and_capture_date():
    rows = [_m("Sampang, East Java, Indonesia", "2023-07-28T17:14:48"), _m("Sampang, East Java, Indonesia", "2023-07-28T06:00:00"),
            _m("Sampang, East Java, Indonesia", "2023-07-29T06:00:00"), _m("", "2023-07-29T06:00:00"),
            _m("West Lafayette, Indiana, United States", "")]
    keys = weather_keys(rows, locate)
    assert [(k["city"], k["capture_date"]) for k in keys] == [
        ("Sampang, East Java, Indonesia", "2023-07-28"), ("Sampang, East Java, Indonesia", "2023-07-29")]
    assert keys[0]["city_lat"] == 1.0


def _daily(start: date, n: int):
    days = [start + timedelta(days=i) for i in range(n)]
    return {"time": [d.isoformat() for d in days],
            "temperature_2m_mean": [float(i) for i in range(n)],
            "temperature_2m_max": [float(i) + 5 for i in range(n)],
            "temperature_2m_min": [float(i) - 5 for i in range(n)],
            "precipitation_sum": [2.0 if i % 2 else 0.0 for i in range(n)],
            "relative_humidity_2m_mean": [60] * n}


def test_summarise_uses_the_days_before_capture_only():
    daily = _daily(date(2023, 7, 1), 31)                     # 1..31 July, day index = value
    s = summarise(daily, "2023-07-29", 7)                    # 22..28 July -> indices 21..27
    assert s == {"temp_mean_c": 24.0, "temp_min_c": 16.0, "temp_max_c": 32.0,
                 "precip_mm": 8.0, "rainy_days": 4, "rh_mean_pct": 60.0, "days": 7}


def test_summarise_reports_missing_days():
    s = summarise(_daily(date(2023, 7, 25), 4), "2023-07-29", 7)
    assert s["days"] == 4


def test_weather_rows_fetch_once_per_city_covering_all_windows():
    calls = []
    def fetch(lat, lon, start, end):
        calls.append((lat, lon, start, end))
        first = date.fromisoformat(start)
        n = (date.fromisoformat(end) - first).days + 1
        return {"timezone": "Asia/Jakarta", "daily": _daily(first, n)}
    keys = weather_keys([_m("Sampang, East Java, Indonesia", "2023-07-28T17:14:48"),
                         _m("Sampang, East Java, Indonesia", "2023-04-11T06:00:00")], locate)
    rows = weather_rows(keys, fetch)
    longest = max(WINDOWS.values())
    assert calls == [(1.0, 2.0, (date(2023, 4, 11) - timedelta(days=longest)).isoformat(), "2023-07-27")]
    assert list(rows[0]) == COLUMNS
    for w, n in WINDOWS.items():
        assert rows[0][f"{w}_days"] == n and rows[0][f"{w}_rainy_days"] != ""
    assert rows[0]["timezone"] == "Asia/Jakarta" and rows[0]["note"] == ""


def test_failed_fetch_keeps_the_row_with_a_note():
    def fetch(*a):
        raise OSError("offline")
    rows = weather_rows(weather_keys([_m("X", "2020-01-10T00:00:00")], locate), fetch)
    assert rows[0]["7d_temp_mean_c"] == "" and rows[0]["note"].startswith("fetch failed")


def test_window_names():
    assert WINDOWS == {"7d": 7, "1m": 30, "3m": 90, "6m": 182}


def test_output_has_no_city_coordinates():
    assert "city_lat" not in COLUMNS and "city_lon" not in COLUMNS
    assert COLUMNS[:3] == ["city", "capture_date", "timezone"]


def _index_with_unused(tmp_path):
    import csv

    from data_prep.image_metadata import COLUMNS as META_COLUMNS, write_index
    for folder, gps in (("cds", ("40.1", "-86.9")), ("china_2", ("37.6", "116.8"))):
        d = tmp_path / "data" / "raw" / folder
        d.mkdir(parents=True)
        with open(d / "image_metadata.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=META_COLUMNS); w.writeheader()
            w.writerow({c: "" for c in META_COLUMNS} | {"path": f"data/raw/{folder}/a.jpg", "md5": folder, "source": folder,
                        "capture_date": "2022-08-01", "gps_lat": gps[0], "gps_lon": gps[1],
                        "city": f"city {float(gps[0])}"})
    index = tmp_path / "data" / "image_metadata.json"
    write_index(tmp_path, {"cds": 1}, index, "m.csv", unused={"china_2": 1})
    return index


def _run_main(monkeypatch, tmp_path, *extra):
    import csv

    import data_prep.locations as locations
    import data_prep.weather as weather
    monkeypatch.setattr(locations, "reverse_city",
                        lambda lat, lon, cache: {"city": f"city {lat}", "city_lat": lat, "city_lon": lon})
    def fake_fetch(lat, lon, start, end, cache):
        first = date.fromisoformat(start)
        return {"timezone": "UTC", "daily": _daily(first, (date.fromisoformat(end) - first).days + 1)}
    monkeypatch.setattr(weather, "fetch_range", fake_fetch)
    out = tmp_path / "weather.csv"
    weather.main(["--metadata", str(_index_with_unused(tmp_path)), "--out", str(out), *extra])
    with open(out, newline="") as f:
        return sorted(r["city"] for r in csv.DictReader(f))


def test_main_uses_only_manifest_datasets_by_default(monkeypatch, tmp_path):
    assert _run_main(monkeypatch, tmp_path) == ["city 40.1"]


def test_main_include_unused_adds_unused_datasets(monkeypatch, tmp_path):
    assert _run_main(monkeypatch, tmp_path, "--include-unused") == ["city 37.6", "city 40.1"]
