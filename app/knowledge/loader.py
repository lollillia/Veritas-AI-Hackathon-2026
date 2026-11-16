"""The disease knowledge file (app/knowledge/diseases.yaml): fixed "About" text and the sourced region,
season and weather facts used to weigh the photo model's answer.

Every factor must trace to a source listed in SOURCES.md; load_knowledge refuses a file that breaks a rule
and lists every problem at once.
"""
import re
from dataclasses import dataclass
from pathlib import Path

import yaml

from app.detectors.base import LABELS

LEVELS = ("common", "occasional", "rare", "not_reported", "unknown")
MIN_ABOUT_WORDS = 40
_URL = re.compile(r"https?://\S+")


@dataclass(frozen=True)
class WeatherRule:
    temp_min_c: float            # favourable 14-day mean temperature range
    temp_max_c: float
    rh_min_pct: float | None     # favourable if the 14-day mean relative humidity is at least this...
    rainy_days_min: int | None   # ...or the last 30 days had at least this many rainy days
    source: str


@dataclass(frozen=True)
class DiseaseInfo:
    label: str
    name: str
    pathogen: str
    about: str
    sources: list[str]
    regions: dict[str, tuple[str, str]]     # "US" or "US-IA" -> (level, source URL or "" for unknown)
    season_months: list[int] | None         # months when it is typically found, northern temperate zone
    season_source: str | None
    weather: WeatherRule | None


class KnowledgeError(Exception):
    pass


def _default_labels() -> tuple[str, ...]:
    return tuple(label for label in LABELS if label != "not_corn_leaf")


def load_knowledge(path: Path = Path("app/knowledge/diseases.yaml"),
                   sources_md: Path = Path("app/knowledge/SOURCES.md"),
                   labels: tuple[str, ...] | None = None) -> dict[str, DiseaseInfo]:
    labels = labels or _default_labels()
    raw = yaml.safe_load(Path(path).read_text()) or {}
    known = {u.rstrip(").,") for u in _URL.findall(Path(sources_md).read_text())} if Path(sources_md).is_file() else set()
    problems: list[str] = []

    def check_source(where: str, src) -> str:
        if not src:
            problems.append(f"{where}: needs a source")
            return ""
        if str(src) not in known:
            problems.append(f"{where}: source {src} is not in SOURCES.md")
        return str(src)

    missing = [label for label in labels if label not in raw]
    extra = [key for key in raw if key not in labels]
    if missing:
        problems.append(f"missing: {', '.join(missing)}")
    if extra:
        problems.append(f"unknown labels: {', '.join(extra)}")

    out: dict[str, DiseaseInfo] = {}
    for label in labels:
        entry = raw.get(label)
        if not isinstance(entry, dict):
            continue
        for key in ("name", "pathogen", "about"):
            if not entry.get(key):
                problems.append(f"{label}: needs {key}")
        about = str(entry.get("about", ""))
        if len(about.split()) < MIN_ABOUT_WORDS:
            problems.append(f"{label}: about has {len(about.split())} words, needs at least {MIN_ABOUT_WORDS}")
        sources = [str(s) for s in entry.get("sources", [])]
        for s in sources:
            check_source(f"{label} sources", s)
        if label == "healthy" and any(entry.get(k) for k in ("regions", "season", "weather")):
            problems.append("healthy: must not have regions, season or weather")

        regions = {}
        for code, spec in (entry.get("regions") or {}).items():
            spec = spec or {}
            level = spec.get("level")
            if level not in LEVELS:
                problems.append(f"{label} region {code}: level {level} is not one of {', '.join(LEVELS)}")
                continue
            src = check_source(f"{label} region {code}", spec.get("source")) if level != "unknown" else str(spec.get("source", "") or "")
            regions[str(code).upper()] = (level, src)

        months, season_src = None, None
        if entry.get("season"):
            months = [int(m) for m in entry["season"].get("months", [])]
            bad = [m for m in months if not 1 <= m <= 12]
            if bad or not months:
                problems.append(f"{label} season: months {bad or months} must be 1-12")
            season_src = check_source(f"{label} season", entry["season"].get("source"))

        weather = None
        if entry.get("weather"):
            w = entry["weather"]
            lo, hi = (w.get("temp_c") or [None, None])[:2]
            if lo is None or hi is None or float(lo) >= float(hi):
                problems.append(f"{label} weather: temp_c must be [low, high] with low < high, got {w.get('temp_c')}")
            if w.get("rh_min_pct") is None and w.get("rainy_days_min") is None:
                problems.append(f"{label} weather: needs a moisture condition (rh_min_pct or rainy_days_min)")
            src = check_source(f"{label} weather", w.get("source"))
            if lo is not None and hi is not None:
                weather = WeatherRule(float(lo), float(hi),
                                      None if w.get("rh_min_pct") is None else float(w["rh_min_pct"]),
                                      None if w.get("rainy_days_min") is None else int(w["rainy_days_min"]), src)

        out[label] = DiseaseInfo(label, str(entry.get("name", "")), str(entry.get("pathogen", "")), about, sources,
                                 regions, months, season_src, weather)
    if problems:
        raise KnowledgeError("knowledge file problems:\n- " + "\n- ".join(problems))
    return out
