from datetime import date

from app.assess import assess
from app.context.geocode import Place, PlaceQuery
from app.detectors.base import Detector, Result
from app.knowledge.loader import DiseaseInfo, WeatherRule

S = "https://example.org"
ABOUT = "Common rust about text."
K = {"common_rust": DiseaseInfo("common_rust", "Common rust", "Puccinia sorghi", ABOUT, [S],
                                {"US": ("common", S)}, [6, 7, 8], S, WeatherRule(16, 25, 80, None, S)),
     "gray_leaf_spot": DiseaseInfo("gray_leaf_spot", "Gray leaf spot", "Cercospora", "GLS about.", [S],
                                   {"US": ("rare", S)}, None, None, None),
     "healthy": DiseaseInfo("healthy", "Healthy leaf", "none", "Healthy about.", [S], {}, None, None, None)}
IOWA = Place("Ames, Story County, Iowa, United States", 42.03, -93.62, "US", "US-IA")
WX = {"14d": {"temp_mean_c": 20.0, "rh_mean_pct": 90.0, "rainy_days": 8, "days": 14},
      "30d": {"temp_mean_c": 20.0, "rh_mean_pct": 90.0, "rainy_days": 15, "days": 30}}


class Det(Detector):
    def __init__(self, result):
        self.result = result

    def predict(self, image_bytes):
        return self.result


SCORED = Result("gray_leaf_spot", 0.5, "x", scores={"gray_leaf_spot": 0.5, "common_rust": 0.4, "healthy": 0.1})


class Calls:
    def __init__(self, place=IOWA, weather=WX):
        self.place, self.weather, self.geo, self.wx = place, weather, [], []

    def geocode(self, q):
        self.geo.append(q)
        return self.place

    def recent_weather(self, lat, lon, d):
        self.wx.append((lat, lon, d))
        return self.weather


def _run(result=SCORED, query=PlaceQuery("Ames", "Iowa", "USA"), d=date(2025, 7, 18), calls=None):
    calls = calls or Calls()
    body = assess(Det(result), K, b"img", query, d, 0.5, geocode=calls.geocode, recent_weather=calls.recent_weather)
    return body, calls


def test_photo_only_when_no_place_or_date():
    body, calls = _run(query=PlaceQuery(), d=None)
    assert body["spectrum"] == body["photo_spectrum"] and body["label"] == "gray_leaf_spot"
    assert any("Add the place and date" in n for n in body["notices"]) and calls.geo == [] and calls.wx == []


def test_place_and_date_change_the_spectrum_and_explain_why():
    body, _ = _run()
    assert body["label"] == "common_rust" and body["spectrum"] != body["photo_spectrum"]
    assert body["why"] and body["place"]["display_name"].startswith("Ames") and body["weather"] == WX
    assert body["factors"]["common_rust"]["combined"] == 1.5


def test_country_only_place_is_used_and_shown():
    kenya = Place("Kenya", -1.0, 38.0, "KE", "")
    body, calls = _run(query=PlaceQuery(country="Kenya"), calls=Calls(place=kenya))
    assert calls.geo == [PlaceQuery(country="Kenya")] and body["place"]["display_name"] == "Kenya"


def test_ambiguous_city_returns_the_understood_place_name():
    paris = Place("Paris, Lamar County, Texas, United States", 33.66, -95.55, "US", "US-TX")
    body, _ = _run(query=PlaceQuery("Paris"), calls=Calls(place=paris))
    assert body["place"]["display_name"] == "Paris, Lamar County, Texas, United States"


def test_place_not_found_notice():
    body, calls = _run(calls=Calls(place=None))
    assert any("couldn't find that place" in n for n in body["notices"]) and calls.wx == []
    assert body["spectrum"] == body["photo_spectrum"]


def test_weather_unavailable_notice():
    body, _ = _run(calls=Calls(weather=None))
    assert any("Weather wasn't available" in n for n in body["notices"]) and body["weather"] is None


def test_not_corn_leaf_makes_no_lookups():
    body, calls = _run(result=Result("not_corn_leaf", 1.0, "A cat."))
    assert body["label"] == "not_corn_leaf" and body["spectrum"] == [] and body["about"] is None
    assert calls.geo == [] and calls.wx == [] and body["note"] == "A cat."


def test_detector_without_scores_is_not_weighted():
    body, calls = _run(result=Result("common_rust", 0.8, "Claude says rust."))
    assert body["label"] == "common_rust" and body["confidence"] == 0.8 and calls.geo == []
    assert any("one answer only" in n for n in body["notices"])
    assert body["spectrum"] == [{"label": "common_rust", "pct": 80}, {"label": "other", "pct": 20}]


def test_about_comes_from_the_knowledge_file():
    body, _ = _run()
    assert body["about"] == {"title": "Common rust", "text": ABOUT, "sources": [S]}


def test_not_sure_uses_final_confidence():
    body, _ = _run(query=PlaceQuery(), d=None)
    assert body["confidence"] == 0.5 and body["not_sure"] is False
    low = Result("gray_leaf_spot", 0.4, "x", scores={"gray_leaf_spot": 0.4, "common_rust": 0.35, "healthy": 0.25})
    body2, _ = _run(result=low, query=PlaceQuery(), d=None)
    assert body2["not_sure"] is True


def test_about_shows_only_web_links():
    k = dict(K)
    k["common_rust"] = DiseaseInfo("common_rust", "Common rust", "Puccinia sorghi", ABOUT,
                                   [S, "notes/rust.md", "http://example.org/b"], {"US": ("common", S)},
                                   [6, 7, 8], S, WeatherRule(16, 25, 80, None, S))
    body = assess(Det(SCORED), k, b"img", PlaceQuery("Ames", "Iowa", "USA"), date(2025, 7, 18), 0.5,
                  geocode=Calls().geocode, recent_weather=Calls().recent_weather)
    assert body["about"]["sources"] == [S, "http://example.org/b"]
