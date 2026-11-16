from datetime import date

import pytest

from app.context.geocode import Place
from app.knowledge.loader import DiseaseInfo, WeatherRule
from app.weighting import region_level, season_state, spectrum, weather_state, weigh

S = "https://example.org"


def _info(label, regions=None, months=None, weather=None):
    return DiseaseInfo(label, label, "p", "about", [S], regions or {}, months, S if months else None, weather)


RUST = _info("common_rust", {"US": ("common", S), "US-FL": ("rare", S)}, [6, 7, 8], WeatherRule(16, 25, 80, None, S))
SR = _info("southern_rust", {"US": ("occasional", S)}, [7, 8, 9], WeatherRule(25, 31, 80, None, S))
HEALTHY = _info("healthy")
K = {"common_rust": RUST, "southern_rust": SR, "healthy": HEALTHY}
IOWA = Place("Ames, Iowa, United States", 42.0, -93.6, "US", "US-IA")
FLORIDA = Place("Gainesville, Florida, United States", 29.6, -82.3, "US", "US-FL")


def _wx(t=20.0, rh=90.0, rainy=10):
    stats = {"temp_mean_c": t, "rh_mean_pct": rh, "rainy_days": rainy, "days": 14}
    return {"14d": stats, "30d": {**stats, "days": 30}}


def test_no_place_no_date_returns_the_photo_scores_unchanged():
    scores = {"common_rust": 0.6, "southern_rust": 0.3, "healthy": 0.1}
    w = weigh(scores, K, None, None, None)
    assert w.final == pytest.approx(scores) and w.why == []
    assert all(f["combined"] == 1.0 for f in w.factors.values())


def test_region_uses_subdivision_before_country():
    assert region_level(RUST, FLORIDA) == "rare" and region_level(RUST, IOWA) == "common"


def test_unknown_region_is_neutral():
    kenya = Place("Kenya", -1.0, 38.0, "KE", "")
    assert region_level(RUST, kenya) == "unknown" and region_level(RUST, None) is None


def test_season_in_and_off():
    assert season_state(RUST, 7, 42.0) == "in" and season_state(RUST, 1, 42.0) == "off"
    assert season_state(RUST, None, 42.0) is None and season_state(RUST, 7, None) is None


def test_southern_hemisphere_shifts_months():
    assert season_state(RUST, 1, -34.6) == "in"      # Argentina January = northern July
    assert season_state(RUST, 7, -34.6) == "off"


def test_tropics_has_no_season_factor():
    assert season_state(RUST, 7, -15.0) is None and season_state(RUST, 7, 10.0) is None


def test_weather_favourable_neutral_unfavourable():
    assert weather_state(RUST, _wx(20, 90)) == "favourable"
    assert weather_state(RUST, _wx(27, 90)) == "neutral"          # warm-ish, inside +5
    assert weather_state(RUST, _wx(33, 90)) == "unfavourable"     # far too hot
    assert weather_state(RUST, _wx(20, 50, rainy=0)) == "unfavourable"   # very dry
    assert weather_state(RUST, None) is None and weather_state(HEALTHY, _wx()) is None


def test_combined_factor_is_clamped():
    w = weigh({"common_rust": 0.5, "healthy": 0.5}, K, IOWA, date(2025, 7, 1), _wx(20, 90))
    assert w.factors["common_rust"]["combined"] == 1.5             # 1.3 x 1.2 x 1.3 = 2.03 -> 1.5
    cold_dry_off = _info("southern_rust", {"US": ("not_reported", S)}, [7], WeatherRule(25, 31, 80, None, S))
    w2 = weigh({"southern_rust": 0.5, "healthy": 0.5}, {"southern_rust": cold_dry_off, "healthy": HEALTHY},
               IOWA, date(2025, 1, 1), _wx(5, 40, rainy=0))
    assert w2.factors["southern_rust"]["combined"] == 0.3          # 0.4 x 0.6 x 0.7 -> 0.3


def test_healthy_factor_is_always_one():
    w = weigh({"common_rust": 0.5, "healthy": 0.5}, K, IOWA, date(2025, 7, 1), _wx())
    assert w.factors["healthy"]["combined"] == 1.0


def test_final_sums_to_one():
    w = weigh({"common_rust": 0.5, "southern_rust": 0.3, "healthy": 0.2}, K, IOWA, date(2025, 7, 1), _wx())
    assert sum(w.final.values()) == pytest.approx(1.0)


def test_clear_photo_is_not_overturned():
    w = weigh({"southern_rust": 0.9, "common_rust": 0.1}, K, IOWA, date(2025, 7, 1), _wx(20, 90))
    assert max(w.final, key=w.final.get) == "southern_rust"


def test_why_names_the_top_labels_factors():
    w = weigh({"common_rust": 0.6, "healthy": 0.4}, K, IOWA, date(2025, 7, 1), _wx(20, 90))
    text = " · ".join(w.why)
    assert "common in Ames (+)" in text and "in season in July (+)" in text and "(+)" in text


def test_spectrum_top3_plus_other_sums_to_100():
    s = spectrum({"a": 0.5, "b": 0.2, "c": 0.15, "d": 0.1, "e": 0.05})
    assert [x["label"] for x in s] == ["a", "b", "c", "other"] and sum(x["pct"] for x in s) == 100


def test_spectrum_without_other_when_three_labels_cover_all():
    s = spectrum({"a": 0.333, "b": 0.333, "c": 0.334})
    assert [x["label"] for x in s] == ["c", "a", "b"] and sum(x["pct"] for x in s) == 100
