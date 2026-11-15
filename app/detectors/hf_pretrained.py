import logging

from app.detectors.base import LABELS, Detector, DetectorError, Result
from app.imaging import load_rgb_image

log = logging.getLogger("veritas")

NOTES: dict[str, str] = {
    "healthy": "No signs of disease were detected on this leaf.",
    "gray_leaf_spot": "Long, narrow, tan to gray rectangular lesions run along the leaf veins.",
    "common_rust": "Small, round to elongated reddish-brown pustules appear on both leaf surfaces.",
    "northern_leaf_blight": "Long, cigar-shaped gray-green to tan lesions spread across the leaf.",
    "northern_leaf_spot": "Small, oval, tan lesions with darker borders are scattered across the leaf.",
    "not_corn_leaf": "This image does not appear to show a corn leaf.",
}


class HfPretrainedDetector(Detector):
    """Image-classification model from Hugging Face (or a local dir in HF format).

    label_map maps the model's own labels to our labels; None means identity.
    The HF_TOKEN env var is picked up by the transformers library if needed.
    """

    def __init__(self, model_id: str, label_map: dict[str, str] | None = None, pipeline=None):
        if pipeline is None:
            from transformers import pipeline as hf_pipeline  # lazy: heavy import

            pipeline = hf_pipeline("image-classification", model=model_id)
        self.pipeline = pipeline
        self.label_map = label_map or {}

        model_labels = list(pipeline.model.config.id2label.values())
        bad = [m for m in model_labels if self.label_map.get(m, m) not in LABELS]
        if bad:
            raise ValueError(
                f"model labels not mapped to one of {list(LABELS)}: {bad}. "
                "Add them to label_map in config.yaml."
            )

    def predict(self, image_bytes: bytes) -> Result:
        try:
            image = load_rgb_image(image_bytes)
        except ValueError as exc:
            raise DetectorError("That file is not a valid image.") from exc
        try:
            top = self.pipeline(image, top_k=1)[0]
            label = self.label_map.get(top["label"], top["label"])
            return Result(label, min(max(float(top["score"]), 0.0), 1.0), NOTES[label])
        except Exception as exc:
            log.exception("hf pipeline failed")
            raise DetectorError("The model failed to analyse the image.") from exc
