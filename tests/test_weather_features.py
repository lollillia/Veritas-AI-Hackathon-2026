import pytest

from data_prep.weather_features import FEATURES, Standardizer, raw_weather_by_md5


def _w(city, d, base=1.0):
    row = {"city": city, "capture_date": d}
    row.update({f: base + i for i, f in enumerate(FEATURES)})
    return {k: str(v) for k, v in row.items()}


def test_features_are_six_stats_for_four_windows_and_no_date():
    assert len(FEATURES) == 24
    assert FEATURES[:6] == ["7d_temp_mean_c", "7d_temp_min_c", "7d_temp_max_c", "7d_precip_mm", "7d_rainy_days", "7d_rh_mean_pct"]
    assert not any("date" in f or "lat" in f or "lon" in f for f in FEATURES)


def test_raw_weather_joins_on_city_and_capture_date():
    meta = [{"md5": "a", "city": "Sampang", "capture_date": "2023-07-28"},
            {"md5": "b", "city": "Sampang", "capture_date": "2023-07-29"},       # no weather row
            {"md5": "c", "city": "", "capture_date": ""}]
    out = raw_weather_by_md5(meta, [_w("Sampang", "2023-07-28")])
    assert list(out) == ["a"] and out["a"] == [1.0 + i for i in range(24)]


def test_rows_with_a_blank_value_are_treated_as_missing():
    w = _w("X", "2020-01-01"); w["6m_rh_mean_pct"] = ""
    assert raw_weather_by_md5([{"md5": "a", "city": "X", "capture_date": "2020-01-01"}], [w]) == {}


def test_standardizer_centres_scales_and_flags():
    s = Standardizer.fit([[0.0] * 23 + [10.0], [2.0] * 23 + [10.0]])     # last feature has no variance
    assert s.transform([1.0] * 23 + [10.0]) == pytest.approx([0.0] * 24 + [1.0])
    assert s.transform([2.0] * 23 + [10.0])[0] == pytest.approx(1.0)
    assert s.transform(None) == [0.0] * 24 + [0.0]                      # missing: zeros, flag 0
    assert len(s.transform(None)) == len(FEATURES) + 1


def test_standardizer_round_trips_through_a_dict():
    s = Standardizer.fit([[float(i)] * 24 for i in range(5)])
    t = Standardizer.from_dict(s.to_dict())
    assert t.transform([3.0] * 24) == s.transform([3.0] * 24)
