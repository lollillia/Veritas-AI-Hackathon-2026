import logging
from pathlib import Path

from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse, JSONResponse

from app.detectors.base import Detector, DetectorError
from app.imaging import load_rgb_image

MAX_UPLOAD_BYTES = 10 * 1024 * 1024
ALLOWED_FORMATS = {"JPEG", "PNG", "WEBP"}
INDEX_HTML = Path(__file__).resolve().parent.parent / "static" / "index.html"

log = logging.getLogger("veritas")


def _error(status: int, message: str) -> JSONResponse:
    return JSONResponse(status_code=status, content={"error": message})


def create_app(detector: Detector, min_confidence: float = 0.5) -> FastAPI:
    app = FastAPI(title="Veritas")

    @app.get("/")
    def index():
        return FileResponse(INDEX_HTML, media_type="text/html")

    @app.post("/predict")
    async def predict(file: UploadFile | None = File(None)):
        if file is None:
            return _error(400, "No file was uploaded.")
        data = await file.read(MAX_UPLOAD_BYTES + 1)
        if not data:
            return _error(400, "The uploaded file is empty.")
        if len(data) > MAX_UPLOAD_BYTES:
            return _error(413, "The image is too large (maximum 10 MB).")
        try:
            image = load_rgb_image(data)
        except ValueError:
            return _error(415, "That file is not a valid JPEG, PNG or WebP image.")
        if image.format not in ALLOWED_FORMATS:
            return _error(415, "Only JPEG, PNG and WebP images are supported.")
        try:
            result = detector.predict(data)
        except DetectorError as exc:
            return _error(502, str(exc))
        except Exception:
            log.exception("detector failed")
            return _error(502, "Something went wrong while analysing the image.")
        return {
            "label": result.label,
            "confidence": result.confidence,
            "note": result.note,
            "not_sure": result.label != "not_corn_leaf" and result.confidence < min_confidence,
        }

    return app
