import pytest

from data_prep.weather import COLUMNS, nearest_hour, weather_keys, weather_rows


def _m(lat, lon, t):
    return {"gps_lat": lat, "gps_lon": lon, "capture_datetime": t}


def test_weather_keys_are_distinct_and_need_all_three_values():
    rows = [_m("40.473", "-86.993", "2020-08-19T09:36:39"), _m("40.473", "-86.993", "2020-08-19T09:36:39"),
            _m("40.473", "-86.993", "2020-08-19T09:54:48"), _m("", "", "2020-08-19T09:00:00"),
            _m("41.5", "-96.5", "")]
    assert weather_keys(rows) == [(40.473, -86.993, "2020-08-19T09:36:39"), (40.473, -86.993, "2020-08-19T09:54:48")]


@pytest.mark.parametrize("t,expected", [
    ("2020-08-19T09:29:59", "2020-08-19T09:00"),
    ("2020-08-19T09:30:00", "2020-08-19T10:00"),
    ("2020-08-19T23:45:00", "2020-08-20T00:00"),      # rolls into the next day
])
def test_nearest_hour(t, expected):
    assert nearest_hour(t) == expected


def _day(date, temps):
    times = [f"{date}T{h:02d}:00" for h in range(24)]
    return {"latitude": 40.48, "longitude": -87.0, "timezone": "America/Indiana/Indianapolis",
            "hourly": {"time": times, "temperature_2m": temps, "relative_humidity_2m": [80] * 24,
                       "dew_point_2m": [18.5] * 24, "precipitation": [0.0] * 24, "cloud_cover": [40] * 24,
                       "wind_speed_10m": [7.2] * 24, "weather_code": [2] * 24}}


def test_weather_rows_pick_the_capture_hour_and_fetch_each_location_day_once():
    calls = []
    def fetch(lat, lon, date):
        calls.append((lat, lon, date))
        return _day(date, [float(h) for h in range(24)])
    keys = [(40.473, -86.993, "2020-08-19T09:36:39"), (40.473, -86.993, "2020-08-19T09:10:00"),
            (40.473, -86.993, "2020-08-19T23:50:00")]
    rows = weather_rows(keys, fetch)
    assert [r["weather_hour_local"] for r in rows] == ["2020-08-19T10:00", "2020-08-19T09:00", "2020-08-20T00:00"]
    assert [r["temperature_2m_c"] for r in rows] == [10.0, 9.0, 0.0]
    assert rows[0]["relative_humidity_2m_pct"] == 80 and rows[0]["timezone"] == "America/Indiana/Indianapolis"
    assert rows[0]["grid_lat"] == 40.48 and rows[0]["capture_datetime"] == "2020-08-19T09:36:39"
    assert sorted(calls) == [(40.473, -86.993, "2020-08-19"), (40.473, -86.993, "2020-08-20")]
    assert list(rows[0]) == COLUMNS


def test_missing_hour_or_failed_fetch_gives_blank_weather():
    def fetch(lat, lon, date):
        if lat == 1.0:
            raise OSError("network down")
        return {"latitude": 0, "longitude": 0, "timezone": "UTC", "hourly": {"time": [], "temperature_2m": []}}
    rows = weather_rows([(1.0, 2.0, "2015-07-01T12:00:00"), (3.0, 4.0, "2015-07-01T12:00:00")], fetch)
    assert all(r["temperature_2m_c"] == "" for r in rows)
    assert rows[0]["note"].startswith("fetch failed") and rows[1]["note"] == "hour not in response"
