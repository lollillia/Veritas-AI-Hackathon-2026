from abc import ABC, abstractmethod
from dataclasses import dataclass, field

LABELS: tuple[str, ...] = (
    "healthy",
    "common_rust",  # Puccinia sorghi
    "northern_leaf_blight",  # Exserohilum turcicum
    "gray_leaf_spot",  # Cercospora zeae-maydis
    "northern_leaf_spot",  # Bipolaris zeicola
    "eyespot",  # Kabatiella zeae (includes "northern anthracnose")
    "phaeosphaeria_leaf_spot",  # Phaeosphaeria maydis
    "southern_rust",  # Puccinia polysora
    "southern_leaf_blight",  # Bipolaris maydis
    "brown_spot",  # Physoderma maydis
    "curvularia_leaf_spot",  # Curvularia lunata
    "not_corn_leaf",
)


class DetectorError(Exception):
    """A backend failure whose message is safe to show to the user."""


@dataclass(frozen=True)
class Result:
    label: str
    confidence: float
    note: str
    # Probability per label, when the detector has them (used for place/time weighting). Not part of equality.
    scores: dict[str, float] | None = field(default=None, compare=False)

    def __post_init__(self):
        if self.label not in LABELS:
            raise ValueError(f"unknown label '{self.label}'")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"confidence {self.confidence} outside 0-1")
        if self.scores is not None:
            bad = [k for k, v in self.scores.items() if k not in LABELS or not 0.0 <= v <= 1.0]
            if bad:
                raise ValueError(f"invalid scores for {bad}")


class Detector(ABC):
    @abstractmethod
    def predict(self, image_bytes: bytes) -> Result: ...
