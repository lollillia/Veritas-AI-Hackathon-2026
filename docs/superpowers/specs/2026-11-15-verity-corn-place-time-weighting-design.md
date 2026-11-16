# Verity Corn: photo + place + time weighting — design

Date: 2026-11-15. Status: draft for approval.

## Intent

The photo model alone struggles on field photos, and feeding photo metadata (date, place, weather) into
training made models memorise our datasets' photo sessions instead of learning about disease
(`train/issues/METADATA_FINGERPRINTING_ISSUE.md`). This design uses place and time **without learning them
from our data**: the user says where and when the photo was taken, and expert sources say where, when and in
what weather each disease occurs. That knowledge nudges the photo model's answer.

It is the hackathon demo for a judge panel, renamed **Verity Corn** ("detecting fungal diseases in corn").
Judges should be able to see the answer, how sure we are across diseases, and why place and weather moved
the result.

## Goals

- The user uploads a leaf photo and enters the **place** (city, state/region, country; ZIP optional) and
  **date** (day, month, year) it was taken.
- The website fetches the **weather** for the weeks before that date.
- The final answer combines the photo model's per-disease percentages with **region, season and weather
  factors** from a reviewed knowledge file. This is approach B, rule-based (user decision).
- The page shows the final disease, a **spectrum** of % certainty per disease (e.g. 60% common rust, 30%
  northern leaf blight, 10% other), a short "why" line, and a fixed **About the disease** text per disease
  (user decision).
- A new **playful, artsy** title and logo recreated from the user's sketch (user decision).

## Non-goals

- Training any model on metadata or weather.
- Reading the source websites live on each upload. They are condensed once into the knowledge file.
- New labels (tar spot, Goss's wilt, Diplodia leaf streak, bacterial stripe). They may be named in "About"
  texts as look-alikes only.
- Supporting place and time weighting for `llm_vision` or `hf_pretrained` (see "Detectors").

## Anti-goals (failure even if it "works")

- Place, season or weather **overriding a clear photo**. The factors must stay mild.
- **False precision**: percentages that look exact but come from guessed factor values with no source.
- Sending more user data to outside services than needed, or keeping it.
- A result that silently ignores the place or date the user typed (for example when geocoding fails)
  without saying so.

## Approaches considered

| | Approach | Verdict |
|---|---|---|
| A | Claude gets the photo, model percentages, place, date, weather and knowledge file and returns the final answer | not chosen: needs internet for every answer, final percentages hard to test or justify |
| **B** | **Rule-based weighting**: knowledge file → region, season and weather factors × the photo model's percentages; fixed "About" text | **chosen (user)**: transparent, testable, works offline except the weather and place lookup |
| C | Train a model with metadata and weather inputs | rejected: the fingerprinting problem |

## How it works

1. **Corn-leaf check** (unchanged): the `gated` detector asks Claude "is this a corn leaf?". If not, the
   result is "not a corn leaf" with no spectrum. If Claude is unreachable, continue without the check.
2. **Photo percentages**: the fine-tuned model (`models/corn_fungal_v6_bgaug`) returns a probability for
   each of its 11 labels, not just the top one.
3. **Place → coordinates**: forward-geocode "city, state/region, country[, ZIP]" with OpenStreetMap Nominatim
   (at most 1 request per second, cached on disk, generic User-Agent). Keep the display name, latitude,
   longitude and country code.
4. **Weather**: Open-Meteo daily values (mean, min and max temperature, precipitation, relative humidity)
   for the **14 and 30 days before** the photo date. Use the historical archive for dates more than about
   5 days ago, and the forecast API's `past_days` for recent dates. Coordinates are rounded to 2 decimals
   before sending. Cached.
5. **Factors per disease** (from the knowledge file):
   - **region**: how common the disease is in that country/region. Levels: common ×1.3, occasional ×1.0,
     rare ×0.6, not reported ×0.4, unknown ×1.0.
   - **season**: in season ×1.2, off season ×0.6. Months are given for the northern temperate zone and
     shifted 6 months for the southern hemisphere; in the tropics (|latitude| < 23.5°) the season factor
     is 1.0 and weather decides.
   - **weather**: favourable ×1.3, neutral ×1.0, unfavourable ×0.7, judged from the 14/30-day summary
     against the disease's documented conditions (temperature range, humidity or rain, rainy days).
   - `healthy` always has factor 1.0.
   - The product of the three is clamped to **[0.3, 1.5]**, so the photo still dominates.
6. **Combine**: final score = photo probability × combined factor, renormalised to 100%. Missing inputs
   mean factor 1.0 for that part (no place → no region, season or weather factor, since the hemisphere is
   unknown; no date → no season or weather factor). With neither, the result is the photo alone.
7. **Result**: the final disease (highest score), the spectrum (top 3 + "other", summing to 100%), the
   photo-only percentages for comparison, a "why" line naming the factors that changed the top disease
   (e.g. "cool, wet last 3 weeks (+) · common in Iowa in July (+)"), the place as understood ("Ames, Iowa,
   United States") and the weather summary used. `not_sure` stays as today, judged on the final score
   against `min_confidence`.

## The knowledge file

`app/knowledge/diseases.yaml`, one entry per label (10 diseases + healthy):

- `name`, `pathogen`, `type` (fungus)
- `about`: fixed general text, about 80–150 words: what it is, what it looks like, where and when it occurs,
  why it matters, common look-alikes. Plain language.
- `regions`: country → level (common / occasional / rare / not reported), each with a source; for countries
  where sources differ by state or province (e.g. southern rust in the US), state/province entries override
  the country level
- `season`: in-season months (northern temperate), with a source
- `weather`: favourable conditions (temperature range, humidity or rain, leaf-wetness proxy), with a source
- `sources`: the URLs used, from the user's list

It is built once by reading the user's sources (list in the conversation of 15 Nov 2026; saved alongside
the file as `app/knowledge/SOURCES.md`). Paywalled or unreachable sources are marked as such and not used.
Every factor value must trace to a source. Where sources disagree or say nothing, the level is
"unknown" (×1.0). The user reviews the file before it goes live.

## The page

Layout follows the user's sketch: header (logo + "Verity corn" + tagline); upload box left; place fields
(city, state/region, country, ZIP) and time fields (day, month, year) right; then Result (disease name),
spectrum bar, "why" line, About the disease.

- **Logo and title**: recreated in a playful, artsy style: a cartoon corn cob with two leaves, one leaf
  carrying three green check marks; "Verity" in green, "corn" in heavy black, a hand-lettered feel, as
  inline SVG plus a Google Fonts display font.
- **Spectrum**: one horizontal bar split into coloured segments with labels and percentages; the top
  disease is emphasised. Below it, smaller text: "Photo alone: …" for comparison.
- Place and date are **optional** but encouraged (a hint says they improve the answer). Dates in the
  future or invalid dates are rejected with a message. Defaults: none (no guessing today's date).
- A footnote says place names go to OpenStreetMap and rounded coordinates to Open-Meteo, and nothing is
  stored.
- Works on a phone and in dark mode, like the current page.

## API

`POST /predict` (multipart): `file` (required), and optional `city`, `region`, `country`, `zip`, `day`,
`month`, `year`. Response adds to today's fields: `spectrum` (list of `{label, pct}`), `photo_spectrum`,
`factors` (per label: region, season, weather, combined), `why`, `place` (display name, lat, lon or
null), `weather` (summary or null), `about` (`{title, text, sources}`), and `notices` (e.g. "couldn't find
that place; used the photo alone"). Existing fields (`label`, `confidence`, `note`, `not_sure`) remain.

## Detectors

Weighting needs per-label probabilities. `Result` gains an optional `scores` (label → probability);
`hf_pretrained` / `finetuned` fill it, `gated` passes it through, and `llm_vision` leaves it empty. With no
scores, the page shows the single answer at its confidence, no weighting, and a notice.

## Failure handling

- Place not found → no region or weather factor, notice shown.
- Weather service down → no weather factor, notice shown.
- Corn-leaf check down → as today, the photo model answers alone.
- All lookups have short timeouts (≤ 10 s each) so a slow service cannot hang the demo.

## How we'll know it works

- Unit tests: factor maths (clamping, renormalising, missing inputs, hemisphere shift, tropics), knowledge
  file validation (every label present, levels and months valid, every factor has a source), API input
  validation, and each fallback.
- **Accuracy check**: on v6 test photos that have a real city and capture date in `data/image_metadata.json`,
  compare photo-only vs weighted accuracy. Expected: weighted is not worse overall. Report per source and
  per class, and how often weighting changed the top answer (and whether for better or worse). Because
  the factors come from outside sources, not our data, this check is not subject to fingerprinting.
- A live run through the page with a few photos and places.

## Decided by the user

Approach B; place fields (city, state/region, country); date fields (day, month, year); fetch recent
weather; show a spectrum of certainty; fixed general "About" text per disease; recreate the title and
logo in a playful artsy style; the sources list.

## My calls (adjustable)

ZIP optional; region levels by country with state/province overrides; 14- and 30-day weather windows; factor values (×0.3–1.5 range and the levels above); top 3 +
"other" in the spectrum; showing photo-only percentages; place and date optional; knowledge file location
and format; tropics rule.

## Left to the builder

Exact fonts and colours; SVG details of the logo; wording of the "why" line; how weather thresholds are
encoded per disease (as long as each is sourced); cache locations.
