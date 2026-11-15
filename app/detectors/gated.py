"""Claude checks "is this a corn leaf?", then our fine-tuned model names the disease.

The fine-tuned model was trained on corn leaves only, so it cannot answer "not a corn leaf". The gate adds
that answer. If the gate cannot answer (no network, API error, unreadable reply), the photo goes to the
fine-tuned model anyway, so the website keeps working offline.
"""
import json
import logging
import os
import re

import anthropic

from app.detectors.base import Detector, DetectorError, Result
from app.detectors.llm_vision import ask_vision
from app.imaging import load_rgb_image

log = logging.getLogger("veritas")

GATE_PROMPT = """Look at the attached photo. Does it show a leaf of a corn (maize) plant, healthy or diseased,
as the main subject? Leaves of other plants (beans, wheat, sorghum, grasses...) do not count.
Reply with JSON only, no other text, in this form:
{"corn_leaf": true or false, "note": "<one short sentence describing what the photo shows>"}"""

_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE)


def _parse_gate(text: str) -> tuple[bool, str]:
    data = json.loads(_FENCE.sub("", text.strip()))
    flag = data["corn_leaf"]
    if not isinstance(flag, bool):
        raise ValueError("corn_leaf must be true or false")
    return flag, str(data.get("note", ""))


class GatedDetector(Detector):
    def __init__(self, model: str = "claude-sonnet-5-5", api_key_env: str = "ANTHROPIC_API_KEY",
                 weights_path: str | None = None, client=None, inner: Detector | None = None):
        self.model = model
        self.client = client or anthropic.Anthropic(api_key=os.environ[api_key_env], timeout=30.0, max_retries=1)
        if inner is None:
            from app.detectors.finetuned import FinetunedDetector   # imports torch; only when really needed
            inner = FinetunedDetector(weights_path=weights_path)
        self.inner = inner

    def predict(self, image_bytes: bytes) -> Result:
        try:
            load_rgb_image(image_bytes)
        except ValueError as exc:
            raise DetectorError("That file is not a valid image.") from exc
        try:
            is_corn, note = _parse_gate(ask_vision(self.client, self.model, image_bytes, GATE_PROMPT))
        except (DetectorError, ValueError, KeyError, TypeError, AttributeError):
            log.warning("corn-leaf check unavailable; using the fine-tuned model alone", exc_info=True)
            return self.inner.predict(image_bytes)
        if not is_corn:
            return Result("not_corn_leaf", 1.0, note or "This image does not appear to show a corn leaf.")
        return self.inner.predict(image_bytes)
