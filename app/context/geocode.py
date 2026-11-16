"""Place typed by the user -> map point, via OpenStreetMap Nominatim (cached, at most one request per second)."""
import hashlib
import json
import logging
import threading
import time
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path

log = logging.getLogger("veritas")
SEARCH = "https://nominatim.openstreetmap.org/search"
USER_AGENT = "corn-leaf-demo"
TIMEOUT_S = 10
_lock, _last = threading.Lock(), [0.0]


@dataclass(frozen=True)
class PlaceQuery:
    city: str = ""
    region: str = ""
    country: str = ""
    zip: str = ""

    def text(self) -> str:
        return ", ".join(p.strip() for p in (self.city, self.region, self.zip, self.country) if p and p.strip())

    def is_empty(self) -> bool:
        return not self.text()


@dataclass(frozen=True)
class Place:
    display_name: str
    lat: float
    lon: float
    country_code: str          # ISO 3166-1 alpha-2, upper case, e.g. "US"
    subdivision_code: str      # ISO 3166-2, e.g. "US-IA", or ""


def _http_json(url: str):
    with _lock:                                       # Nominatim usage policy: at most 1 request per second
        wait = 1.0 - (time.monotonic() - _last[0])
        if wait > 0:
            time.sleep(wait)
        _last[0] = time.monotonic()
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=TIMEOUT_S) as resp:
        return json.load(resp)


def geocode(query: PlaceQuery, cache_dir: Path = Path("data/raw/app_cache/geocode"), fetch=None) -> Place | None:
    text = query.text()
    if not text:
        return None
    cache = Path(cache_dir) / f"{hashlib.sha1(text.lower().encode()).hexdigest()}.json"
    try:
        if cache.is_file():
            hits = json.loads(cache.read_text())
        else:
            url = f"{SEARCH}?{urllib.parse.urlencode({'q': text, 'format': 'jsonv2', 'addressdetails': 1, 'limit': 1})}"
            hits = (fetch or _http_json)(url)
            cache.parent.mkdir(parents=True, exist_ok=True)
            cache.write_text(json.dumps(hits))
    except Exception:
        log.warning("place lookup failed for %r", text, exc_info=True)
        return None
    if not hits:
        return None
    h, addr = hits[0], hits[0].get("address", {})
    return Place(h.get("display_name", text), float(h["lat"]), float(h["lon"]),
                 str(addr.get("country_code", "")).upper(), str(addr.get("ISO3166-2-lvl4", "")).upper())
