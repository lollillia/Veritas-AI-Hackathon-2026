import logging
from datetime import date
from pathlib import Path

from fastapi import FastAPI, File, Form, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse, JSONResponse

from app.assess import assess
from app.context.geocode import PlaceQuery, geocode as _geocode
from app.context.weather import recent_weather as _recent_weather
from app.detectors.base import Detector, DetectorError
from app.imaging import load_rgb_image
from app.knowledge.loader import load_knowledge

MAX_UPLOAD_BYTES = 10 * 1024 * 1024
# MPO = multi-picture JPEG, which many phone cameras produce.
ALLOWED_FORMATS = {"JPEG", "MPO", "PNG", "WEBP"}
STATIC = Path(__file__).resolve().parent.parent / "static"
INDEX_HTML = STATIC / "index.html"
LOGO = STATIC / "corn-logo.png"
EARLIEST = date(1990, 1, 1)

log = logging.getLogger("veritas")


def _error(status: int, message: str) -> JSONResponse:
    return JSONResponse(status_code=status, content={"error": message})


def parse_photo_date(day: str, month: str, year: str) -> tuple[date | None, str | None]:
    """(date or None, error message or None). All three empty means no date."""
    parts = [x.strip() for x in (day, month, year)]
    if not any(parts):
        return None, None
    if not all(parts):
        return None, "Give the day, month and year, or leave all three empty."
    try:
        d = date(int(parts[2]), int(parts[1]), int(parts[0]))
    except ValueError:
        return None, "That date doesn't exist."
    if d > date.today():
        return None, "The photo date can't be in the future."
    if d < EARLIEST:
        return None, "Please give a date from 1990 onwards."
    return d, None


def create_app(detector: Detector, min_confidence: float = 0.5, knowledge=None,
               geocode=_geocode, recent_weather=_recent_weather) -> FastAPI:
    app = FastAPI(title="Verity Corn")
    knowledge = knowledge if knowledge is not None else load_knowledge()

    @app.get("/")
    def index():
        return FileResponse(INDEX_HTML, media_type="text/html")

    @app.get("/corn-logo.png")
    def logo():
        return FileResponse(LOGO, media_type="image/png")

    @app.post("/predict")
    async def predict(file: UploadFile | None = File(None), city: str = Form(""), region: str = Form(""),
                      country: str = Form(""), zip: str = Form(""), day: str = Form(""), month: str = Form(""),
                      year: str = Form("")):
        photo_date, date_error = parse_photo_date(day, month, year)
        if date_error:
            return _error(400, date_error)
        if file is None:
            return _error(400, "No file was uploaded.")
        data = await file.read(MAX_UPLOAD_BYTES + 1)
        if not data:
            return _error(400, "The uploaded file is empty.")
        if len(data) > MAX_UPLOAD_BYTES:
            return _error(413, "The image is too large (maximum 10 MB).")
        try:
            image = await run_in_threadpool(load_rgb_image, data)
        except ValueError:
            return _error(415, "That file is not a valid JPEG, PNG or WebP image.")
        if image.format not in ALLOWED_FORMATS:
            return _error(415, "Only JPEG, PNG and WebP images are supported.")
        try:
            query = PlaceQuery(city=city, region=region, country=country, zip=zip)
            return await run_in_threadpool(assess, detector, knowledge, data, query, photo_date, min_confidence,
                                           geocode, recent_weather)
        except DetectorError as exc:
            return _error(502, str(exc))
        except Exception:
            log.exception("detector failed")
            return _error(502, "Something went wrong while analysing the image.")

    return app
