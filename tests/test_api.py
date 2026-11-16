import io

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app.api import MAX_UPLOAD_BYTES, create_app
from app.detectors.base import Detector, DetectorError, Result


class FakeDetector(Detector):
    def __init__(self, result=None, exc=None):
        self.result = result or Result("common_rust", 0.9, "Orange pustules.")
        self.exc = exc
        self.seen = None

    def predict(self, image_bytes):
        self.seen = image_bytes
        if self.exc:
            raise self.exc
        return self.result


def _img(fmt="JPEG"):
    buf = io.BytesIO()
    Image.new("RGB", (16, 16), (0, 128, 0)).save(buf, format=fmt)
    return buf.getvalue()


def _post(client, data, name="leaf.jpg", ctype="image/jpeg"):
    return client.post("/predict", files={"file": (name, data, ctype)})


def client_for(detector=None, min_confidence=0.5):
    return TestClient(create_app(detector or FakeDetector(), min_confidence))


def test_predict_returns_result_json():
    r = _post(client_for(), _img())
    assert r.status_code == 200
    body = r.json()
    assert {k: body[k] for k in ("label", "confidence", "note", "not_sure")} == {
        "label": "common_rust",
        "confidence": 0.9,
        "note": "Orange pustules.",
        "not_sure": False,
    }


def test_low_confidence_sets_not_sure():
    det = FakeDetector(Result("common_rust", 0.3, "x"))
    assert _post(client_for(det), _img()).json()["not_sure"] is True


def test_not_corn_leaf_is_never_not_sure():
    det = FakeDetector(Result("not_corn_leaf", 0.1, "x"))
    assert _post(client_for(det), _img()).json()["not_sure"] is False


def test_original_bytes_reach_detector():
    det = FakeDetector()
    data = _img()
    _post(client_for(det), data)
    assert det.seen == data


def test_corrupt_file_named_jpg_returns_415():
    r = _post(client_for(), b"not an image")
    assert r.status_code == 415
    assert r.json()["error"]


def test_text_file_returns_415():
    r = _post(client_for(), b"hello world", name="notes.txt", ctype="text/plain")
    assert r.status_code == 415


def test_gif_returns_415():
    r = _post(client_for(), _img("GIF"), name="a.gif", ctype="image/gif")
    assert r.status_code == 415


@pytest.mark.parametrize("fmt", ["PNG", "WEBP"])
def test_png_and_webp_accepted(fmt):
    assert _post(client_for(), _img(fmt), name=f"a.{fmt.lower()}").status_code == 200


def test_oversized_upload_returns_413():
    r = _post(client_for(), b"\x00" * (MAX_UPLOAD_BYTES + 1))
    assert r.status_code == 413
    assert r.json()["error"]


def test_empty_file_returns_400():
    assert _post(client_for(), b"").status_code == 400


def test_missing_file_field_returns_400():
    r = client_for().post("/predict", data={"x": "y"})
    assert r.status_code == 400
    assert r.json()["error"]


def test_detector_error_returns_502_with_message():
    r = _post(client_for(FakeDetector(exc=DetectorError("service down"))), _img())
    assert r.status_code == 502
    assert r.json()["error"] == "service down"


def test_unexpected_exception_returns_502_generic():
    r = _post(client_for(FakeDetector(exc=RuntimeError("secret internals"))), _img())
    assert r.status_code == 502
    assert "secret internals" not in r.json()["error"]


def test_index_page_served():
    r = client_for().get("/")
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/html")


def test_mpo_phone_jpeg_is_accepted():
    # Many phone cameras save multi-picture JPEGs, which Pillow reports as format "MPO".
    buf = io.BytesIO()
    im = Image.new("RGB", (16, 16), (0, 128, 0))
    im.save(buf, format="MPO", save_all=True, append_images=[Image.new("RGB", (16, 16))])
    assert Image.open(io.BytesIO(buf.getvalue())).format == "MPO"
    assert _post(client_for(), buf.getvalue()).status_code == 200


def test_detector_runs_off_the_event_loop():
    import asyncio

    class ThreadCheck(FakeDetector):
        def predict(self, image_bytes):
            try:
                asyncio.get_running_loop()
            except RuntimeError:
                return super().predict(image_bytes)  # no loop in this thread: good
            raise AssertionError("detector ran on the event loop thread")

    assert _post(client_for(ThreadCheck()), _img()).status_code == 200


def test_index_page_has_a_display_name_for_every_label():
    from app.detectors.base import LABELS

    import re

    html = client_for().get("/").text
    for label in LABELS:
        # a whole key, not the tail of a longer one (leaf_blight vs northern_leaf_blight)
        assert re.search(rf"(?<![\w]){label}:", html), f"UI has no display name for {label}"


# --- place and date ---------------------------------------------------------------------------------
from datetime import date, timedelta

from app.context.geocode import Place

SCORED = Result("gray_leaf_spot", 0.5, "x", scores={"gray_leaf_spot": 0.5, "common_rust": 0.4, "healthy": 0.1})
IOWA = Place("Ames, Story County, Iowa, United States", 42.03, -93.62, "US", "US-IA")
WX = {"14d": {"temp_mean_c": 20.0, "rh_mean_pct": 90.0, "rainy_days": 8, "days": 14},
      "30d": {"temp_mean_c": 20.0, "rh_mean_pct": 90.0, "rainy_days": 15, "days": 30}}


def ctx_client(result=SCORED):
    app = create_app(FakeDetector(result), 0.5, geocode=lambda q: IOWA, recent_weather=lambda lat, lon, d: WX)
    return TestClient(app)


def _post_ctx(client, **fields):
    return client.post("/predict", files={"file": ("leaf.jpg", _img(), "image/jpeg")}, data=fields)


def test_predict_with_place_and_date_returns_spectrum_and_about():
    body = _post_ctx(ctx_client(), city="Ames", region="Iowa", country="USA", day="18", month="7", year="2025").json()
    assert body["spectrum"] and body["about"]["title"] == "Gray leaf spot"      # real knowledge file
    assert body["factors"]["common_rust"]["combined"] == 1.5 and body["factors"]["healthy"]["combined"] == 1.0
    assert body["place"]["display_name"].startswith("Ames") and body["why"]


def test_predict_without_place_or_date_still_works():
    body = _post_ctx(ctx_client()).json()
    assert body["label"] == "gray_leaf_spot" and body["spectrum"] == body["photo_spectrum"]


def test_partial_date_is_rejected():
    r = _post_ctx(ctx_client(), day="18", month="7")
    assert r.status_code == 400 and "day, month and year" in r.json()["error"]


def test_invalid_date_is_rejected():
    r = _post_ctx(ctx_client(), day="31", month="2", year="2025")
    assert r.status_code == 400 and "doesn't exist" in r.json()["error"]


def test_future_date_is_rejected():
    d = date.today() + timedelta(days=1)
    r = _post_ctx(ctx_client(), day=str(d.day), month=str(d.month), year=str(d.year))
    assert r.status_code == 400 and "future" in r.json()["error"]


def test_today_is_accepted():
    d = date.today()
    assert _post_ctx(ctx_client(), day=str(d.day), month=str(d.month), year=str(d.year)).status_code == 200


def test_date_before_1990_is_rejected():
    r = _post_ctx(ctx_client(), day="1", month="1", year="1989")
    assert r.status_code == 400 and "1990" in r.json()["error"]


def test_logo_is_served():
    r = client_for().get("/corn-logo.png")
    assert r.status_code == 200 and r.headers["content-type"] == "image/png"
