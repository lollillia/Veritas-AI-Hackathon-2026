"""Weather for the 14 and 30 days before the photo date, from Open-Meteo (cached)."""
import json
import logging
import urllib.parse
import urllib.request
from datetime import date, timedelta
from pathlib import Path

from data_prep.weather import API as ARCHIVE, DAILY, summarise

log = logging.getLogger("veritas")
FORECAST = "https://api.open-meteo.com/v1/forecast"
TIMEOUT_S = 10
ARCHIVE_LAG_DAYS = 6          # the archive lags a few days behind today


def _http_json(url: str):
    req = urllib.request.Request(url, headers={"User-Agent": "corn-leaf-demo"})
    with urllib.request.urlopen(req, timeout=TIMEOUT_S) as resp:
        return json.load(resp)


def recent_weather(lat: float, lon: float, photo_date: date,
                   cache_dir: Path = Path("data/raw/app_cache/weather"), fetch=None,
                   today: date | None = None) -> dict | None:
    today = today or date.today()
    lat, lon = round(lat, 2), round(lon, 2)
    base = {"latitude": f"{lat:.2f}", "longitude": f"{lon:.2f}", "daily": ",".join(DAILY), "timezone": "auto"}
    if photo_date <= today - timedelta(days=ARCHIVE_LAG_DAYS):
        params = {**base, "start_date": (photo_date - timedelta(days=30)).isoformat(),
                  "end_date": (photo_date - timedelta(days=1)).isoformat()}
        url, key = f"{ARCHIVE}?{urllib.parse.urlencode(params)}", f"a_{lat}_{lon}_{photo_date}"
    else:
        params = {**base, "past_days": 31, "forecast_days": 1}
        url, key = f"{FORECAST}?{urllib.parse.urlencode(params)}", f"f_{lat}_{lon}_{today}"
    cache = Path(cache_dir) / f"{key}.json"
    try:
        if cache.is_file():
            data = json.loads(cache.read_text())
        else:
            data = (fetch or _http_json)(url)
            cache.parent.mkdir(parents=True, exist_ok=True)
            cache.write_text(json.dumps(data))
        daily = data.get("daily", {})
        out = {"14d": summarise(daily, photo_date.isoformat(), 14), "30d": summarise(daily, photo_date.isoformat(), 30)}
    except Exception:
        log.warning("weather lookup failed", exc_info=True)
        return None
    if not out["14d"]["days"]:
        return None
    return out
