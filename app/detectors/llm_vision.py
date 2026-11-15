import base64
import json
import logging
import os
import re

import anthropic

from app.detectors.base import LABELS, Detector, DetectorError, Result
from app.imaging import load_rgb_image, to_jpeg_bytes

log = logging.getLogger("veritas")

UNREADABLE = "The model returned an unreadable answer; please try again."
UNAVAILABLE = "The vision service is unavailable; please try again."
BAD_KEY = "The vision API key was rejected; check ANTHROPIC_API_KEY."
BAD_MODEL = "The configured vision model was not found; check llm_vision.model in config.yaml."

PROMPT = f"""You are an expert plant pathologist. Look at the attached photo.
The photo may NOT be a corn (maize) leaf; it could be anything.

Classify it with exactly one of these labels:
{", ".join(LABELS)}

These are fungal diseases: "common_rust" (Puccinia sorghi), "northern_leaf_blight" (Exserohilum
turcicum), "gray_leaf_spot" (Cercospora zeae-maydis), "northern_leaf_spot" (Bipolaris zeicola) and
"eyespot" (Kabatiella zeae, also called northern anthracnose) and "phaeosphaeria_leaf_spot"
(Phaeosphaeria maydis).
Use "not_corn_leaf" whenever the image is not a corn/maize leaf.
Reply with JSON only, no other text, in this form:
{{"label": "<one label>", "confidence": <number from 0 to 1>, "note": "<one short sentence describing what you see>"}}"""

_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE)


def _parse_reply(text: str) -> Result:
    try:
        data = json.loads(_FENCE.sub("", text.strip()))
        conf = data["confidence"]
        if isinstance(conf, bool) or not isinstance(conf, (int, float)):
            raise ValueError("confidence must be a number")
        return Result(label=data["label"], confidence=float(conf), note=str(data["note"]))
    except (ValueError, KeyError, TypeError, AttributeError) as exc:  # JSONDecodeError is a ValueError
        raise DetectorError(UNREADABLE) from exc


class LlmVisionDetector(Detector):
    def __init__(self, model: str = "claude-sonnet-5-5", api_key_env: str = "ANTHROPIC_API_KEY", client=None):
        self.model = model
        self.client = client or anthropic.Anthropic(
            api_key=os.environ[api_key_env], timeout=30.0, max_retries=1
        )

    def predict(self, image_bytes: bytes) -> Result:
        try:
            jpeg = to_jpeg_bytes(load_rgb_image(image_bytes))
        except ValueError as exc:
            raise DetectorError("That file is not a valid image.") from exc
        content = [
            {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/jpeg",
                    "data": base64.b64encode(jpeg).decode(),
                },
            },
            {"type": "text", "text": PROMPT},
        ]
        try:
            reply = self.client.messages.create(
                model=self.model,
                max_tokens=4096,  # thinking tokens count against this; a small value truncates the answer
                messages=[{"role": "user", "content": content}],
            )
        except anthropic.APIError as exc:
            log.exception("vision API call failed")
            if isinstance(exc, (anthropic.AuthenticationError, anthropic.PermissionDeniedError)):
                raise DetectorError(BAD_KEY) from exc
            if isinstance(exc, anthropic.NotFoundError):
                raise DetectorError(BAD_MODEL) from exc
            raise DetectorError(UNAVAILABLE) from exc
        # The reply may start with non-text blocks (e.g. "thinking"); use the first text block.
        text = next((b.text for b in reply.content if getattr(b, "type", None) == "text"), None)
        if text is None:
            log.error("vision reply had no text block (stop_reason=%s)", getattr(reply, "stop_reason", None))
            raise DetectorError(UNREADABLE)
        return _parse_reply(text)
