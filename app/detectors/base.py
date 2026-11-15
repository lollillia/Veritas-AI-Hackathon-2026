from abc import ABC, abstractmethod
from dataclasses import dataclass

LABELS: tuple[str, ...] = (
    "healthy",
    "common_rust",  # Puccinia sorghi
    "northern_leaf_blight",  # Exserohilum turcicum
    "gray_leaf_spot",  # Cercospora zeae-maydis
    "northern_leaf_spot",  # Bipolaris zeicola
    "eyespot",  # Kabatiella zeae (includes "northern anthracnose")
    "not_corn_leaf",
)


class DetectorError(Exception):
    """A backend failure whose message is safe to show to the user."""


@dataclass(frozen=True)
class Result:
    label: str
    confidence: float
    note: str

    def __post_init__(self):
        if self.label not in LABELS:
            raise ValueError(f"unknown label '{self.label}'")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"confidence {self.confidence} outside 0-1")


class Detector(ABC):
    @abstractmethod
    def predict(self, image_bytes: bytes) -> Result: ...
