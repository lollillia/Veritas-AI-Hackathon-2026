"""Where photos were taken: documented dataset sites and GPS -> city lookups.

Reverse geocoding uses OpenStreetMap Nominatim at city level (zoom 10), at most one request per
second, cached in data/raw/geocode_cache/. Only the (already rounded) coordinates are sent.
"""
import json
import time
import urllib.parse
import urllib.request
from pathlib import Path

# Dataset-level collection site, where the source documents one.
SOURCE_SITES = {
    "cds": "near West Lafayette, Indiana, USA (from photo GPS)",
    "ndisan": "Polagan Village, Sampang Regency, Madura, East Java, Indonesia (dataset page)",
}
# Coordinates for documented sites, used when a photo has no GPS of its own.
SITE_COORDS: dict[str, tuple[float, float]] = {
    # OpenStreetMap Nominatim: village "Polagan, Sampang, Jawa Timur 69211, Indonesia".
    SOURCE_SITES["ndisan"]: (-7.205, 113.247),
}
NOMINATIM = "https://nominatim.openstreetmap.org/reverse"
_last_call = [0.0]


def city_name(address: dict) -> str:
    place = next((address[k] for k in ("city", "town", "village", "municipality", "county") if address.get(k)), "")
    parts = [p for p in (place, address.get("state", ""), address.get("country", "")) if p]
    return ", ".join(parts) if place else ""


def _nominatim(lat: float, lon: float) -> dict:
    wait = 1.1 - (time.time() - _last_call[0])
    if wait > 0:
        time.sleep(wait)
    query = urllib.parse.urlencode({"lat": lat, "lon": lon, "format": "json", "zoom": 10, "accept-language": "en"})
    req = urllib.request.Request(f"{NOMINATIM}?{query}", headers={"User-Agent": "dataset-weather-lookup"})
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return json.load(resp)
    finally:
        _last_call[0] = time.time()


def reverse_city(lat: float, lon: float, cache_dir: Path, fetch=None) -> dict:
    """{"city", "city_lat", "city_lon"} for a coordinate; blanks if the lookup fails."""
    blank = {"city": "", "city_lat": "", "city_lon": ""}
    cache = Path(cache_dir) / f"{lat:.3f}_{lon:.3f}.json"
    if cache.is_file():
        data = json.loads(cache.read_text())
    else:
        try:
            data = (fetch or _nominatim)(lat, lon)
        except Exception:
            return blank
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps(data))
    name = city_name(data.get("address", {}))
    if not name:
        return blank
    return {"city": name, "city_lat": round(float(data["lat"]), 3), "city_lon": round(float(data["lon"]), 3)}
