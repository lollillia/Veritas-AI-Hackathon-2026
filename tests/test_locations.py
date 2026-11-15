from data_prep.locations import SITE_COORDS, SOURCE_SITES, city_name, reverse_city


def test_city_name_prefers_city_then_town_village_county():
    assert city_name({"city": "West Lafayette", "county": "Tippecanoe County", "state": "Indiana", "country": "United States"}) \
        == "West Lafayette, Indiana, United States"
    assert city_name({"town": "Simat", "state": "Uasin Gishu County", "country": "Kenya"}) == "Simat, Uasin Gishu County, Kenya"
    assert city_name({"county": "Sampang", "state": "East Java", "country": "Indonesia"}) == "Sampang, East Java, Indonesia"
    assert city_name({}) == ""


def test_reverse_city_uses_the_cache_and_returns_the_place_centre(tmp_path):
    calls = []
    def fetch(lat, lon):
        calls.append((lat, lon))
        return {"address": {"city": "Sampang", "state": "East Java", "country": "Indonesia"}, "lat": "-7.1955709", "lon": "113.2511017"}
    first = reverse_city(-7.205, 113.247, tmp_path, fetch=fetch)
    again = reverse_city(-7.205, 113.247, tmp_path, fetch=fetch)
    assert first == again == {"city": "Sampang, East Java, Indonesia", "city_lat": -7.196, "city_lon": 113.251}
    assert calls == [(-7.205, 113.247)]


def test_failed_lookup_gives_blank_city(tmp_path):
    def fetch(lat, lon):
        raise OSError("offline")
    assert reverse_city(1.0, 2.0, tmp_path, fetch=fetch) == {"city": "", "city_lat": "", "city_lon": ""}


def test_documented_site_has_coordinates():
    assert SITE_COORDS[SOURCE_SITES["ndisan"]] == (-7.205, 113.247)
