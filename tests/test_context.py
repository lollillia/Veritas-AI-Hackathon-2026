from datetime import date, timedelta

import pytest

from app.context.geocode import Place, PlaceQuery, geocode
from app.context.weather import recent_weather

AMES = [{"lat": "42.0267", "lon": "-93.6170", "display_name": "Ames, Story County, Iowa, United States",
         "address": {"city": "Ames", "state": "Iowa", "ISO3166-2-lvl4": "US-IA", "country_code": "us"}}]


class Fetch:
    def __init__(self, reply=None, exc=None):
        self.reply, self.exc, self.urls = reply, exc, []

    def __call__(self, url):
        self.urls.append(url)
        if self.exc:
            raise self.exc
        return self.reply


def test_place_query_text_skips_blanks():
    assert PlaceQuery(city="Ames", region="", country="USA").text() == "Ames, USA"
    assert PlaceQuery().is_empty() and not PlaceQuery(country="Kenya").is_empty()


def test_geocode_parses_country_and_subdivision(tmp_path):
    p = geocode(PlaceQuery("Ames", "Iowa", "USA"), tmp_path, fetch=Fetch(AMES))
    assert p == Place("Ames, Story County, Iowa, United States", 42.0267, -93.617, "US", "US-IA")


def test_geocode_country_only_has_no_subdivision(tmp_path):
    reply = [{"lat": "-1.0", "lon": "38.0", "display_name": "Kenya", "address": {"country_code": "ke"}}]
    p = geocode(PlaceQuery(country="Kenya"), tmp_path, fetch=Fetch(reply))
    assert p.country_code == "KE" and p.subdivision_code == ""


def test_geocode_returns_none_on_no_match_and_on_error(tmp_path):
    assert geocode(PlaceQuery("Nowhere"), tmp_path, fetch=Fetch([])) is None
    assert geocode(PlaceQuery("Ames"), tmp_path / "x", fetch=Fetch(exc=OSError("down"))) is None
    assert geocode(PlaceQuery(), tmp_path, fetch=Fetch(AMES)) is None


def test_geocode_uses_the_cache_second_time(tmp_path):
    f = Fetch(AMES)
    geocode(PlaceQuery("Ames", "Iowa", "USA"), tmp_path, fetch=f)
    geocode(PlaceQuery("ames", "iowa", "usa"), tmp_path, fetch=f)
    assert len(f.urls) == 1


def test_geocode_sends_unicode_city_encoded(tmp_path):
    f = Fetch([])
    geocode(PlaceQuery("São Paulo", country="Brazil"), tmp_path, fetch=f)
    assert "S%C3%A3o+Paulo" in f.urls[0] and "nominatim.openstreetmap.org/search" in f.urls[0]


def _daily(start: date, n: int, temp=20.0, rh=90.0, rain=2.0):
    days = [(start + timedelta(days=i)).isoformat() for i in range(n)]
    return {"timezone": "UTC", "daily": {"time": days, "temperature_2m_mean": [temp] * n,
            "temperature_2m_max": [temp + 5] * n, "temperature_2m_min": [temp - 5] * n,
            "precipitation_sum": [rain] * n, "relative_humidity_2m_mean": [rh] * n}}


def test_recent_weather_old_date_uses_archive_with_rounded_coordinates(tmp_path):
    photo = date(2025, 7, 18)
    f = Fetch(_daily(photo - timedelta(days=30), 30))
    recent_weather(42.02671, -93.61701, photo, tmp_path, fetch=f, today=date(2026, 11, 15))
    assert "archive-api.open-meteo.com" in f.urls[0] and "latitude=42.03" in f.urls[0] and "longitude=-93.62" in f.urls[0]
    assert "start_date=2025-06-18" in f.urls[0] and "end_date=2025-07-17" in f.urls[0]


def test_recent_weather_yesterday_uses_forecast_past_days(tmp_path):
    today = date(2026, 11, 15)
    f = Fetch(_daily(today - timedelta(days=31), 32))
    recent_weather(42.0, -93.6, today - timedelta(days=1), tmp_path, fetch=f, today=today)
    assert "api.open-meteo.com/v1/forecast" in f.urls[0] and "past_days=31" in f.urls[0]


def test_recent_weather_summarises_14_and_30_days(tmp_path):
    photo = date(2025, 7, 18)
    w = recent_weather(42.0, -93.6, photo, tmp_path, fetch=Fetch(_daily(photo - timedelta(days=30), 30)),
                       today=date(2026, 11, 15))
    assert w["14d"]["days"] == 14 and w["30d"]["days"] == 30
    assert w["14d"]["temp_mean_c"] == 20.0 and w["30d"]["rainy_days"] == 30 and w["14d"]["rh_mean_pct"] == 90.0


def test_recent_weather_returns_none_on_error(tmp_path):
    assert recent_weather(1.0, 2.0, date(2025, 7, 18), tmp_path, fetch=Fetch(exc=OSError("down")),
                          today=date(2026, 11, 15)) is None


def test_geocode_asks_for_english_place_names(tmp_path):
    f = Fetch([])
    geocode(PlaceQuery("Chengdu", country="China"), tmp_path, fetch=f)
    assert "accept-language=en" in f.urls[0]
