"""One photo + optional place and date -> the response the page shows."""
from datetime import date

from app.context.geocode import PlaceQuery, geocode as _geocode
from app.context.weather import recent_weather as _recent_weather
from app.detectors.hf_pretrained import NOTES
from app.weighting import spectrum, weigh

NO_PLACE_FOUND = "We couldn't find that place, so the answer uses the photo alone."
NO_WEATHER = "Weather wasn't available, so it wasn't used."
ADD_CONTEXT = "Add the place and date to weigh in local conditions."
SINGLE_ANSWER = "This model gives one answer only, so place and date weren't used."


def _about(knowledge, label):
    info = knowledge.get(label)
    return {"title": info.name, "text": info.about, "sources": list(info.sources)} if info else None


def assess(detector, knowledge, image_bytes: bytes, place_query: PlaceQuery, photo_date: date | None,
           min_confidence: float, geocode=_geocode, recent_weather=_recent_weather) -> dict:
    result = detector.predict(image_bytes)
    body = {"label": result.label, "confidence": result.confidence, "note": result.note, "not_sure": False,
            "spectrum": [], "photo_spectrum": [], "factors": {}, "why": [], "place": None, "weather": None,
            "about": None, "notices": []}
    if result.label == "not_corn_leaf":
        return body

    if not result.scores:
        one = {result.label: result.confidence, "other": max(0.0, 1.0 - result.confidence)}
        body["spectrum"] = body["photo_spectrum"] = spectrum(one, top=1)
        body["about"] = _about(knowledge, result.label)
        body["not_sure"] = result.confidence < min_confidence
        body["notices"].append(SINGLE_ANSWER)
        return body

    place = weather = None
    if place_query.is_empty() and photo_date is None:
        body["notices"].append(ADD_CONTEXT)
    if not place_query.is_empty():
        place = geocode(place_query)
        if place is None:
            body["notices"].append(NO_PLACE_FOUND)
        elif photo_date is not None:
            weather = recent_weather(place.lat, place.lon, photo_date)
            if weather is None:
                body["notices"].append(NO_WEATHER)

    w = weigh(result.scores, knowledge, place, photo_date, weather)
    label = max(w.final, key=w.final.get)
    body.update({
        "label": label,
        "confidence": round(w.final[label], 4),
        "note": NOTES.get(label, result.note),
        "not_sure": w.final[label] < min_confidence,
        "spectrum": spectrum(w.final),
        "photo_spectrum": spectrum(result.scores),
        "factors": w.factors,
        "why": w.why,
        "place": {"display_name": place.display_name, "lat": place.lat, "lon": place.lon} if place else None,
        "weather": weather,
        "about": _about(knowledge, label),
    })
    return body
