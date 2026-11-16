"""Weigh the photo model's per-disease scores with sourced region, season and weather factors.

final(label) ∝ photo score × clamp(region × season × weather, 0.3, 1.5). Factors that cannot be judged
(no place, no date, no rule, no weather) count as 1.0, and `healthy` is always 1.0, so the photo dominates.
"""
import calendar
from dataclasses import dataclass
from datetime import date

from app.context.geocode import Place
from app.knowledge.loader import DiseaseInfo

REGION_FACTORS = {"common": 1.3, "occasional": 1.0, "rare": 0.6, "not_reported": 0.4, "unknown": 1.0}
SEASON_FACTORS = {"in": 1.2, "off": 0.6}
WEATHER_FACTORS = {"favourable": 1.3, "neutral": 1.0, "unfavourable": 0.7}
CLAMP = (0.3, 1.5)
TROPICS_LAT = 23.5


@dataclass(frozen=True)
class Weighting:
    final: dict[str, float]
    factors: dict[str, dict]
    why: list[str]


def region_level(info: DiseaseInfo, place: Place | None) -> str | None:
    if place is None:
        return None
    for code in (place.subdivision_code, place.country_code):
        if code and code in info.regions:
            return info.regions[code][0]
    return "unknown"


def season_state(info: DiseaseInfo, month: int | None, lat: float | None) -> str | None:
    if month is None or lat is None or not info.season_months or abs(lat) < TROPICS_LAT:
        return None
    if lat < 0:
        month = (month + 5) % 12 + 1          # southern hemisphere: shift by 6 months
    return "in" if month in info.season_months else "off"


def weather_state(info: DiseaseInfo, weather: dict | None) -> str | None:
    rule = info.weather
    if rule is None or not weather:
        return None
    w14, w30 = weather.get("14d") or {}, weather.get("30d") or {}
    t = w14.get("temp_mean_c")
    if t in ("", None):
        return None
    rh, rainy = w14.get("rh_mean_pct"), w30.get("rainy_days")
    rh = None if rh in ("", None) else float(rh)
    rainy = None if rainy in ("", None) else int(rainy)
    moist = (rule.rh_min_pct is not None and rh is not None and rh >= rule.rh_min_pct) or \
            (rule.rainy_days_min is not None and rainy is not None and rainy >= rule.rainy_days_min)
    dry = rh is not None and rainy is not None and rh < (rule.rh_min_pct or 80) - 15 and rainy < 2
    if rule.temp_min_c <= t <= rule.temp_max_c and moist:
        return "favourable"
    if t < rule.temp_min_c - 5 or t > rule.temp_max_c + 5 or dry:
        return "unfavourable"
    return "neutral"


def _why(info: DiseaseInfo, place, month, levels: dict, weather) -> list[str]:
    out, where = [], (place.display_name.split(",")[0] if place else "")
    region, season, wx = levels["region"], levels["season"], levels["weather"]
    if region == "common":
        out.append(f"common in {where} (+)")
    elif region in ("rare", "not_reported"):
        out.append(f"{'rare' if region == 'rare' else 'not reported'} in {where} (−)")
    if season:
        out.append(f"{'in' if season == 'in' else 'off'} season in {calendar.month_name[month]} "
                   f"({'+' if season == 'in' else '−'})")
    if wx in ("favourable", "unfavourable"):
        t = weather["14d"]["temp_mean_c"]
        out.append(f"{t:.0f}°C average, last 2 weeks {'suit it (+)' if wx == 'favourable' else 'do not suit it (−)'}")
    return out


def weigh(scores: dict[str, float], knowledge: dict[str, DiseaseInfo], place: Place | None,
          photo_date: date | None, weather: dict | None) -> Weighting:
    month = photo_date.month if photo_date else None
    lat = place.lat if place else None
    factors, levels_by, raw = {}, {}, {}
    for label, p in scores.items():
        info = knowledge.get(label)
        if label == "healthy" or info is None:
            levels = {"region": None, "season": None, "weather": None}
        else:
            levels = {"region": region_level(info, place),
                      "season": season_state(info, month, lat) if place else None,
                      "weather": weather_state(info, weather) if place and photo_date else None}
        f = {"region": REGION_FACTORS.get(levels["region"], 1.0),
             "season": SEASON_FACTORS.get(levels["season"], 1.0),
             "weather": WEATHER_FACTORS.get(levels["weather"], 1.0)}
        f["combined"] = round(min(max(f["region"] * f["season"] * f["weather"], CLAMP[0]), CLAMP[1]), 4)
        if label == "healthy":
            f["combined"] = 1.0
        factors[label], levels_by[label] = f, levels
        raw[label] = p * f["combined"]
    total = sum(raw.values()) or 1.0
    final = {k: v / total for k, v in raw.items()}
    top = max(final, key=final.get) if final else None
    why = _why(knowledge[top], place, month, levels_by[top], weather) if top in knowledge and top != "healthy" else []
    return Weighting(final, factors, why)


def spectrum(probs: dict[str, float], top: int = 3) -> list[dict]:
    """Top labels + "other" as integer percentages summing to 100 (largest remainder)."""
    ranked = sorted(probs.items(), key=lambda kv: -kv[1])
    parts = [(k, v) for k, v in ranked[:top]]
    rest = sum(v for _, v in ranked[top:])
    if rest > 1e-9:
        parts.append(("other", rest))
    total = sum(v for _, v in parts) or 1.0
    exact = [(k, v / total * 100) for k, v in parts]
    pcts = {k: int(x) for k, x in exact}
    short = 100 - sum(pcts.values())
    for k, x in sorted(exact, key=lambda kx: -(kx[1] - int(kx[1])))[:short]:
        pcts[k] += 1
    return [{"label": k, "pct": pcts[k]} for k, _ in parts if pcts[k] > 0 or k != "other"]
