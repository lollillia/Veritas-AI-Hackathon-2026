# Verity Corn: Photo + Place + Time Weighting Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let the user give the place and date of a corn-leaf photo, fetch the recent weather, and weigh the photo model's per-disease percentages with sourced region, season and weather factors. Show the result as a certainty spectrum with fixed "About" text, on a redesigned "Verity corn" page.

**Architecture:**
- **Detector output:** the detector now returns per-label scores.
- **Pure weighting module:** turns scores + a reviewed YAML knowledge file + place/date/weather into final percentages.
- **Lookup modules:** place → map point (Nominatim) and weather (Open-Meteo), both cached and injectable.
- **Orchestration:** an `assess` function builds the API response.
- **Page and check:** `static/index.html` is rewritten to the sketch, and an offline check script compares photo-only vs weighted accuracy on test photos with real metadata.

**Tech Stack:** Python 3.14, FastAPI, PyYAML, Pillow, urllib (no new dependencies), Hugging Face transformers pipeline, plain HTML/CSS/JS, pytest.

**Spec:** `docs/superpowers/specs/2026-11-15-verity-corn-place-time-weighting-design.md`

## Global Constraints

- **Region factors:** common 1.3, occasional 1.0, rare 0.6, not_reported 0.4, unknown 1.0.
- **Season factors:** in season 1.2, off season 0.6. Tropics (|lat| < 23.5) → 1.0. Southern hemisphere (lat < −23.5) → months shifted by 6.
- **Weather factors:** favourable 1.3, neutral 1.0, unfavourable 0.7.
- **Combined factor:** `healthy` is always 1.0. The product region × season × weather is clamped to [0.3, 1.5].
- **Missing inputs:** count as 1.0. No place → no region, season or weather factor. No date → no season or weather factor.
- **Weather windows:** 14 and 30 days ending the day before the photo date.
- **Outside services:** every outside lookup has a timeout of 10 s or less. Coordinates sent to Open-Meteo are rounded to 2 decimals. Nominatim gets at most 1 request/s with the generic User-Agent `corn-leaf-demo`. Never send the user's email or any personal data beyond the typed place.
- **Caches:** `data/raw/app_cache/geocode/` and `data/raw/app_cache/weather/` (gitignored by `data/raw/*`). Nothing else is stored.
- **Dates:** a photo date in the future, invalid, or before 1990-01-01 is rejected with HTTP 400 and a message.
- **Knowledge file:** every factor level other than `unknown`, every season and every weather rule has a source URL that appears in `app/knowledge/SOURCES.md`.
- **Spectrum:** top 3 labels + `other`; integer percentages summing to exactly 100.
- **Page:**
  - Title "Verity corn", tagline "detecting fungal diseases in corn".
  - Logo `static/corn-logo.png` (the user's drawing, already committed), on a light backdrop in dark mode.
  - Wordmark lines straight. Fonts: Fredoka (display), Nunito Sans (body).
  - Phone width and dark mode supported.
  - Footnote: "Place names are sent to OpenStreetMap and rounded coordinates to Open-Meteo to look up the location and weather. Nothing is stored."
- **Commits:** end every commit message with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`; author via `-c user.name="${USER}" -c user.email="noreply@example.com"`.

## Review Focus

1. **Place with only a country** (no city): the region factor uses the country level, weather comes from the country's map point, and the page shows the place as understood. Test in Task 4 and Task 6.
2. **Ambiguous city without a country** ("Paris"): Nominatim's first match is used and its full display name is returned, so the user can see what was understood. Test in Task 6.
3. **Photo date is today or yesterday:** weather comes from the forecast API's `past_days`, not the archive. A date of tomorrow is rejected. Test in Task 4 and Task 7.
4. **Southern-hemisphere and tropical places** (Brazil, Kenya): the season shift and the tropics rule apply. Test in Task 5.
5. **Detector without scores** (`llm_vision`), or a "not a corn leaf" result: there's no weighting and no crash, and the page shows the single answer with a notice. Test in Task 6.

---

### Task 1: Detectors return per-label scores

**Files:**
- Modify: `app/detectors/base.py` (`Result`)
- Modify: `app/detectors/hf_pretrained.py` (`predict`)
- Test: `tests/test_hf_pretrained.py`, `tests/test_registry.py`, `tests/test_gated.py`

**Interfaces:**
- Produces: `Result(label, confidence, note, scores: dict[str, float] | None = None)`. `scores` keys must be in `LABELS`, values 0–1. Excluded from equality (`field(default=None, compare=False)`).
- Produces: `HfPretrainedDetector.predict` returns `scores` for every model label (mapped through `label_map`), with label = argmax.
- `GatedDetector` passes the inner `Result` through unchanged; its `not_corn_leaf` result has `scores=None`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_registry.py
def test_result_scores_must_use_known_labels():
    with pytest.raises(ValueError):
        Result("healthy", 0.9, "x", scores={"made_up": 0.9})

def test_result_scores_default_to_none_and_do_not_affect_equality():
    assert Result("healthy", 0.9, "x").scores is None
    assert Result("healthy", 0.9, "x", scores={"healthy": 0.9}) == Result("healthy", 0.9, "x")

# tests/test_hf_pretrained.py  (FakePipeline must honour top_k=None by returning all labels)
def test_predict_returns_a_score_for_every_label():
    r = detector_with({"common_rust": 0.7, "healthy": 0.2, "eyespot": 0.1}).predict(_img())
    assert r.label == "common_rust" and r.scores == pytest.approx({"common_rust": 0.7, "healthy": 0.2, "eyespot": 0.1})

# tests/test_gated.py
def test_corn_leaf_result_keeps_the_inner_scores(): ...   # FakeInner returns scores; gated result has the same scores
```

- [ ] **Step 2: Run them and check they fail:** `pytest tests/test_registry.py tests/test_hf_pretrained.py tests/test_gated.py -q`. Expected: FAIL (`scores` unknown).
- [ ] **Step 3: Implement.** Add the field and its validation in `Result.__post_init__`. In `predict`, call `self.pipeline(image, top_k=None)` and build `scores` from all entries.
- [ ] **Step 4: Run the full suite:** `pytest -q`. Expected: all pass.
- [ ] **Step 5: Commit** `feat: detectors return per-label scores`.

### Task 2: Knowledge file loader and validation

**Files:**
- Create: `app/knowledge/__init__.py`, `app/knowledge/loader.py`
- Test: `tests/test_knowledge.py` (uses a small YAML written to `tmp_path`)

**Interfaces:**
- Produces:

```python
@dataclass(frozen=True)
class WeatherRule:
    temp_min_c: float; temp_max_c: float          # favourable 14-day mean temperature range
    rh_min_pct: float | None                      # favourable if 14-day mean RH >= this
    rainy_days_min: int | None                    # ...or 30-day rainy days >= this
    source: str

@dataclass(frozen=True)
class DiseaseInfo:
    label: str; name: str; pathogen: str; about: str; sources: list[str]
    regions: dict[str, tuple[str, str]]           # "US" or "US-IA" -> (level, source URL)
    season_months: list[int] | None; season_source: str | None   # northern temperate months 1-12
    weather: WeatherRule | None

class KnowledgeError(Exception): ...
def load_knowledge(path: Path = Path("app/knowledge/diseases.yaml"),
                   sources_md: Path = Path("app/knowledge/SOURCES.md")) -> dict[str, DiseaseInfo]
```

- **YAML shape per label key:** `name`, `pathogen`, `about`, `sources: [url…]`, `regions: {CODE: {level, source}}`, `season: {months: [..], source}` (optional), `weather: {temp_c: [lo, hi], rh_min_pct, rainy_days_min, source}` (optional; at least one of the two moisture keys).
- **Validation:** `load_knowledge` raises `KnowledgeError` listing every problem:
  - a label in `app.detectors.base.LABELS` missing (except `not_corn_leaf`), or an extra key
  - a level not in {common, occasional, rare, not_reported, unknown}
  - a month outside 1–12
  - `temp_c` lo ≥ hi
  - a non-unknown level, season or weather without a source
  - a source URL not present in `SOURCES.md`
  - `about` shorter than 40 words
  - `healthy` with `regions`, `season` or `weather`

- [ ] **Step 1: Write failing tests:**
  - `test_valid_file_loads` (2 labels via a monkeypatched label list, fields parsed into dataclasses)
  - `test_missing_label_is_reported`
  - `test_bad_level_month_and_temp_range_are_all_reported` (one `KnowledgeError` naming all three)
  - `test_unsourced_entry_is_reported`
  - `test_source_not_in_sources_md_is_reported`
  - `test_short_about_is_reported`
  - `test_healthy_must_not_have_factors`
- [ ] **Step 2: Run them:** `pytest tests/test_knowledge.py -q`. Expected: FAIL (module missing).
- [ ] **Step 3: Implement** `load_knowledge` with `yaml.safe_load`.
- [ ] **Step 4: Run** `pytest -q`. Expected: all pass.
- [ ] **Step 5: Commit** `feat: knowledge file loader with validation`.

### Task 3: Write the knowledge file from the user's sources

**Files:**
- Create: `app/knowledge/SOURCES.md`: the user's full source list from the conversation of 2026-11-15, grouped as they gave it. Each URL is marked `used`, `paywalled`, `unreachable` or `off-topic` after reading.
- Create: `app/knowledge/diseases.yaml`: 10 diseases + `healthy`, in the Task 2 shape.
- Test: `tests/test_knowledge.py::test_real_knowledge_file_is_valid`

**Interfaces:**
- Consumes: `load_knowledge` (Task 2).
- Produces: the file loaded by Tasks 5–8.

**Content rules:**
- Read each source with WebFetch. Use only what a source says.
- `about`: 80–150 words, plain language — what it is, the pathogen, what it looks like, where and when it occurs, why it matters, look-alikes. Tar spot, Goss's wilt, Diplodia leaf streak and bacterial stripe may be named here as look-alikes only.
- `regions`: cover at least US (with state overrides where a source distinguishes, e.g. southern rust: southern states common, northern Corn Belt occasional/late), CA, MX, BR, AR, ZA, KE, NG, CN, IN, ID, BD, plus any country a source names. Anything without a source stays out (= unknown).
- `season.months`: northern temperate months when the disease is typically found.
- `weather`: the favourable temperature range and moisture condition the sources give (e.g. common rust 16–23 °C with high humidity; southern rust 27 °C+; gray leaf spot warm with long leaf wetness).
- Where sources disagree, use the more conservative level and note it in `SOURCES.md`.

- [ ] **Step 1: Write the failing test** `test_real_knowledge_file_is_valid`: `load_knowledge()` succeeds and returns all 11 labels except `not_corn_leaf`.
- [ ] **Step 2: Run it:** `pytest tests/test_knowledge.py::test_real_knowledge_file_is_valid -q`. Expected: FAIL (file missing).
- [ ] **Step 3: Read the sources and write `SOURCES.md` and `diseases.yaml`.**
- [ ] **Step 4: Run** `pytest -q`. Expected: all pass.
- [ ] **Step 5: Commit** `feat: disease knowledge file from the source list`. The final message of the run must ask the user to review `app/knowledge/diseases.yaml` and `SOURCES.md` before the demo (spec: "The user reviews the file before it goes live").

### Task 4: Place and weather lookups

**Files:**
- Create: `app/context/__init__.py`, `app/context/geocode.py`, `app/context/weather.py`
- Test: `tests/test_context.py` (fake `fetch` callables; no network)

**Interfaces:**
- Produces:

```python
@dataclass(frozen=True)
class PlaceQuery:
    city: str = ""; region: str = ""; country: str = ""; zip: str = ""
    def is_empty(self) -> bool
    def text(self) -> str                        # non-empty parts joined by ", "

@dataclass(frozen=True)
class Place:
    display_name: str; lat: float; lon: float
    country_code: str                            # upper-case ISO 3166-1 alpha-2, e.g. "US"
    subdivision_code: str                        # ISO 3166-2, e.g. "US-IA", or ""

def geocode(query: PlaceQuery, cache_dir: Path = Path("data/raw/app_cache/geocode"), fetch=None) -> Place | None
def recent_weather(lat: float, lon: float, photo_date: date,
                   cache_dir: Path = Path("data/raw/app_cache/weather"), fetch=None,
                   today: date | None = None) -> dict | None
    # {"14d": stats, "30d": stats} using data_prep.weather.summarise; None if the service fails or no data
```

- **geocode:**
  - Calls Nominatim `/search` with `q=query.text()`, `format=jsonv2`, `addressdetails=1`, `limit=1`, User-Agent `corn-leaf-demo`, timeout 10 s, at most 1 request/s (reuse the throttle pattern in `data_prep/locations.py`).
  - Cache key: sha1 of the lower-cased query text.
  - `subdivision_code` comes from `address["ISO3166-2-lvl4"]`.
  - Returns None for an empty query, no match or a network error.
- **recent_weather:**
  - Coordinates rounded to 2 decimals.
  - If `photo_date <= today − 6 days`, use the archive API (`data_prep.weather.API`) for start = photo_date − 30 d, end = photo_date − 1 d. Otherwise use `https://api.open-meteo.com/v1/forecast` with `past_days=31&forecast_days=1`.
  - Same daily variables as `data_prep.weather.DAILY`; timeout 10 s.

- [ ] **Step 1: Write failing tests:**
  - `test_place_query_text_skips_blanks`
  - `test_geocode_parses_country_and_subdivision`
  - `test_geocode_country_only_has_no_subdivision`
  - `test_geocode_returns_none_on_no_match_and_on_error`
  - `test_geocode_uses_the_cache_second_time`
  - `test_geocode_sends_unicode_city_encoded` ("São Paulo")
  - `test_recent_weather_old_date_uses_archive_with_rounded_coordinates`
  - `test_recent_weather_yesterday_uses_forecast_past_days`
  - `test_recent_weather_summarises_14_and_30_days`
  - `test_recent_weather_returns_none_on_error`
- [ ] **Step 2: Run them:** `pytest tests/test_context.py -q`. Expected: FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run** `pytest -q`. Expected: all pass.
- [ ] **Step 5: Commit** `feat: place and recent-weather lookups`.

### Task 5: Weighting

**Files:**
- Create: `app/weighting.py`
- Test: `tests/test_weighting.py`

**Interfaces:**
- Consumes: `DiseaseInfo`, `WeatherRule` (Task 2); `Place` (Task 4).
- Produces:

```python
REGION_FACTORS = {"common": 1.3, "occasional": 1.0, "rare": 0.6, "not_reported": 0.4, "unknown": 1.0}
SEASON_FACTORS = {"in": 1.2, "off": 0.6}
WEATHER_FACTORS = {"favourable": 1.3, "neutral": 1.0, "unfavourable": 0.7}
CLAMP = (0.3, 1.5)

def region_level(info: DiseaseInfo, place: Place | None) -> str | None        # subdivision, then country, else "unknown"; None without place
def season_state(info: DiseaseInfo, month: int | None, lat: float | None) -> str | None   # "in"/"off"/None
def weather_state(info: DiseaseInfo, weather: dict | None) -> str | None     # "favourable"/"neutral"/"unfavourable"/None

@dataclass(frozen=True)
class Weighting:
    final: dict[str, float]                     # sums to 1
    factors: dict[str, dict]                    # label -> {"region","season","weather","combined"} (numbers; 1.0 when not applied)
    why: list[str]                              # human phrases for the top label's non-1.0 factors

def weigh(scores: dict[str, float], knowledge: dict[str, DiseaseInfo], place: Place | None,
          photo_date: date | None, weather: dict | None) -> Weighting
def spectrum(probs: dict[str, float], top: int = 3) -> list[dict]           # [{"label", "pct"}], "other" last if > 0
```

- **Weather rule:**
  - Uses the 14-day mean temperature `t` and moisture, where moisture is met if `rh_mean_pct >= rh_min_pct` (14 d) or `rainy_days >= rainy_days_min` (30 d).
  - favourable = `temp_min_c <= t <= temp_max_c` and moisture met.
  - unfavourable = `t < temp_min_c − 5` or `t > temp_max_c + 5`, or (14-day RH < `rh_min_pct − 15` and 30-day rainy days < 2).
  - Otherwise neutral. None if no rule or no weather.
- **Season rule:** month taken as given; for lat < −23.5, month is shifted by 6 (wrapping) before the check; |lat| < 23.5 → None.
- **Why phrases:** e.g. "common in Iowa (+)", "rare in Kenya (−)", "in season in July (+)", "off season in January (−)", "warm, humid last 2 weeks (+)", "dry last month (−)". Place name from `place.display_name`'s first part.
- **Spectrum:** largest-remainder rounding to integers summing to 100.

- [ ] **Step 1: Write failing tests (values from Global Constraints):**
  - `test_no_place_no_date_returns_the_photo_scores_unchanged`
  - `test_region_uses_subdivision_before_country`
  - `test_unknown_region_is_neutral`
  - `test_season_in_and_off`
  - `test_southern_hemisphere_shifts_months` (Brazil lat −15 → tropics → None; Argentina lat −34, July checked as January)
  - `test_tropics_has_no_season_factor`
  - `test_weather_favourable_neutral_unfavourable` (one test, three cases)
  - `test_combined_factor_is_clamped` (1.3×1.2×1.3 → 1.5; 0.4×0.6×0.7 → 0.3)
  - `test_healthy_factor_is_always_one`
  - `test_final_sums_to_one`
  - `test_clear_photo_is_not_overturned` (photo 0.9 rare-disease vs 0.1 common-disease → top stays)
  - `test_why_names_the_top_labels_factors`
  - `test_spectrum_top3_plus_other_sums_to_100`
  - `test_spectrum_without_other_when_three_labels_cover_all`
- [ ] **Step 2: Run them:** `pytest tests/test_weighting.py -q`. Expected: FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run** `pytest -q`. Expected: all pass.
- [ ] **Step 5: Commit** `feat: region, season and weather weighting`.

### Task 6: Assess: one photo + place + date → response body

**Files:**
- Create: `app/assess.py`
- Test: `tests/test_assess.py` (fake detector, fake `geocode` and `recent_weather` callables, small knowledge dict)

**Interfaces:**
- Consumes: `Detector`/`Result` (Task 1), `load_knowledge` output (Task 2), `PlaceQuery`, `Place`, `geocode`, `recent_weather` signatures (Task 4), `weigh`, `spectrum` (Task 5).
- Produces:

```python
def assess(detector, knowledge, image_bytes: bytes, place_query: PlaceQuery, photo_date: date | None,
           min_confidence: float, geocode=geocode, recent_weather=recent_weather) -> dict
```

- **Response keys:**
  - `label`, `confidence`, `note`, `not_sure` — `label` is the final top label, `confidence` its final score, `note` = `NOTES[label]` for weighted results, otherwise the detector's note.
  - `spectrum`, `photo_spectrum`, `factors`, `why`
  - `place` (`{"display_name","lat","lon"}` or null), `weather` (`{"14d":…,"30d":…}` or null)
  - `about` (`{"title","text","sources"}` from knowledge, or null for `not_corn_leaf`)
  - `notices` (list of strings)
- **Notices:**
  - "We couldn't find that place, so the answer uses the photo alone."
  - "Weather wasn't available, so it wasn't used."
  - "Add the place and date to weigh in local conditions." (when both are missing)
  - "This model gives one answer only, so place and date weren't used." (scores None and label ≠ not_corn_leaf)
- `not_corn_leaf` → no lookups are made, `spectrum` is `[]`, `about` is null.
- Geocoding and weather run only when needed (place present; weather needs place + date).

- [ ] **Step 1: Write failing tests:**
  - `test_photo_only_when_no_place_or_date` (notice present, spectrum = photo spectrum)
  - `test_place_and_date_change_the_spectrum_and_explain_why`
  - `test_country_only_place_is_used_and_shown`
  - `test_ambiguous_city_returns_the_understood_place_name`
  - `test_place_not_found_notice`
  - `test_weather_unavailable_notice`
  - `test_not_corn_leaf_makes_no_lookups`
  - `test_detector_without_scores_is_not_weighted`
  - `test_about_comes_from_the_knowledge_file`
  - `test_not_sure_uses_final_confidence`
- [ ] **Step 2: Run them:** `pytest tests/test_assess.py -q`. Expected: FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run** `pytest -q`. Expected: all pass.
- [ ] **Step 5: Commit** `feat: assess combines photo, place, date and weather`.

### Task 7: API accepts place and date

**Files:**
- Modify: `app/api.py` (`create_app`, `/predict`, new `GET /corn-logo.png`)
- Modify: `app/main.py` (load knowledge at startup; fail fast with the `KnowledgeError` message)
- Test: `tests/test_api.py`, `tests/test_wiring.py`

**Interfaces:**
- Consumes: `assess` (Task 6), `PlaceQuery` (Task 4), `load_knowledge` (Task 2).
- Produces: `create_app(detector, min_confidence=0.5, knowledge=None, geocode=geocode, recent_weather=recent_weather)`. With `knowledge=None` it loads the real file.
- **`/predict`:** form fields `city, region, country, zip, day, month, year` (all optional strings).
  - A date is used only when all three of day, month and year are given.
  - Some but not all → 400 "Give the day, month and year, or leave all three empty."
  - Invalid date → 400 "That date doesn't exist."
  - Future date → 400 "The photo date can't be in the future."
  - Before 1990 → 400 "Please give a date from 1990 onwards."
  - Existing upload checks (400/413/415/502) are unchanged.

- [ ] **Step 1: Write failing tests:**
  - `test_predict_with_place_and_date_returns_spectrum_and_about` (fakes injected)
  - `test_predict_without_place_or_date_still_works`
  - `test_partial_date_is_rejected`
  - `test_invalid_date_is_rejected`
  - `test_future_date_is_rejected`
  - `test_today_is_accepted`
  - `test_date_before_1990_is_rejected`
  - `test_logo_is_served`
- [ ] **Step 2: Run them:** `pytest tests/test_api.py -q`. Expected: FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run** `pytest -q`. Expected: all pass.
- [ ] **Step 5: Commit** `feat: /predict takes place and date`.

### Task 8: The Verity corn page

**Files:**
- Rewrite: `static/index.html`
- Test: `tests/test_api.py` (page checks)

**Interfaces:**
- Consumes: the `/predict` response (Task 6/7) and `/corn-logo.png`.
- **Element ids:** `photo`, `city`, `region`, `country`, `zip`, `day`, `month`, `year`, `check`, `result`, `spectrum`, `photo-spectrum`, `why`, `notices`, `about`.

**Layout and behaviour (from the sketch and spec):**
- Header: logo `img` + "Verity" (green) over "corn" (black), straight, + tagline.
- Upload box on the left; Place and Time field groups on the right; they stack on phones.
- Result: disease name between ✶ marks.
- Spectrum: one bar split by `pct`, labelled, with a colour per label; the "Photo alone: …" line below.
- "Why it moved" line; notices; About title, text and source links.
- Keep today's client checks (type and size), the "not sure" state and the "not a corn leaf" state.
- Footnote text from Global Constraints. Display names for all 11 labels.
- Use theme tokens for light and dark; the logo sits on a light backdrop in dark mode.

- [ ] **Step 1: Write failing tests:**
  - `test_page_has_place_and_time_fields` (all ids present)
  - `test_page_shows_the_brand_and_footnote` ("Verity", "detecting fungal diseases in corn", the footnote text, `corn-logo.png`)
  - Keep `test_index_page_has_a_display_name_for_every_label`.
- [ ] **Step 2: Run them:** `pytest tests/test_api.py -q`. Expected: FAIL.
- [ ] **Step 3: Implement the page.**
- [ ] **Step 4: Run** `pytest -q`. Expected: all pass. Then run the app (`uvicorn app.main:build_app --factory`) and post one corn photo with "Ames, Iowa, United States" / 18 7 2025, and one with no place, via the page. Expected: a spectrum, a why line and about text; then the photo-only notice.
- [ ] **Step 5: Commit** `feat: Verity corn page with place, time and spectrum`.

### Task 9: Accuracy check and docs

**Files:**
- Create: `train/weighting_check.py`
- Test: `tests/test_weighting_check.py`
- Modify: `README.md` (inputs, weighting, knowledge file, privacy), `train/issues/METADATA_FINGERPRINTING_ISSUE.md` (status: place/time now used via sourced knowledge, with the check's result)

**Interfaces:**
- Consumes: `weigh` (Task 5), `geocode`, `recent_weather` (Task 4), `load_knowledge` (Task 2), the HF pipeline with `top_k=None`.
- Produces:
  - `compare(rows: list[dict]) -> dict`, where each row is `{"true", "photo_top", "weighted_top", "source"}`. Returns accuracy photo-only vs weighted, overall and per source, plus counts of changed answers: changed→right, changed→wrong, changed wrong→wrong.
  - CLI: `python -m train.weighting_check --model models/corn_fungal_v6_bgaug --manifest data/splits_v6.csv --out models/corn_fungal_v6_bgaug/weighting_check.md`, run on test photos whose `data/image_metadata.json` row has a city and capture date. The place query is the metadata city string.

- [ ] **Step 1: Write failing tests:** `test_compare_counts_accuracy_and_changes`, `test_compare_per_source`.
- [ ] **Step 2: Run them:** `pytest tests/test_weighting_check.py -q`. Expected: FAIL.
- [ ] **Step 3: Implement**, then run the CLI. Expected: a report file; weighted overall accuracy ≥ photo-only. If lower, record it in the report and the issue note, and do not tune factors on the test set.
- [ ] **Step 4: Update README and the issue note with the result; run** `pytest -q`. Expected: all pass.
- [ ] **Step 5: Commit** `feat: weighting accuracy check; docs`.
