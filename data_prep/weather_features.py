"""Weather summaries as model input: 24 numbers per image (6 stats x 4 windows), never the date.

Joined to images through image_metadata.csv (city + capture_date). Images without weather get zeros
and a has-weather flag of 0 after standardisation; the scaler is fitted on training images only.
"""
from data_prep.weather import WINDOWS

_STATS = ["temp_mean_c", "temp_min_c", "temp_max_c", "precip_mm", "rainy_days", "rh_mean_pct"]
FEATURES = [f"{w}_{s}" for w in WINDOWS for s in _STATS]


def raw_weather_by_md5(metadata_rows: list[dict], weather_rows: list[dict]) -> dict[str, list[float]]:
    """{md5: 24 raw values} for images whose (city, capture_date) has a complete weather row."""
    by_key = {}
    for w in weather_rows:
        try:
            by_key[(w["city"], w["capture_date"])] = [float(w[f]) for f in FEATURES]
        except (KeyError, ValueError):
            continue                                       # blank or missing value: treat as no weather
    out = {}
    for m in metadata_rows:
        key = (m.get("city", ""), m.get("capture_date", ""))
        if key[0] and key[1] and key in by_key:
            out[m["md5"]] = by_key[key]
    return out


class Standardizer:
    def __init__(self, means: list[float], stds: list[float]):
        self.means, self.stds = means, stds

    @classmethod
    def fit(cls, vectors: list[list[float]]) -> "Standardizer":
        n = len(vectors)
        means = [sum(v[i] for v in vectors) / n for i in range(len(FEATURES))]
        stds = []
        for i, m in enumerate(means):
            var = sum((v[i] - m) ** 2 for v in vectors) / n
            stds.append(var ** 0.5 if var > 0 else 1.0)
        return cls(means, stds)

    def transform(self, vector: list[float] | None) -> list[float]:
        """Standardised values plus a has-weather flag; zeros and flag 0 when missing."""
        if vector is None:
            return [0.0] * len(FEATURES) + [0.0]
        return [(x - m) / s for x, m, s in zip(vector, self.means, self.stds)] + [1.0]

    def to_dict(self) -> dict:
        return {"features": FEATURES, "means": self.means, "stds": self.stds}

    @classmethod
    def from_dict(cls, d: dict) -> "Standardizer":
        return cls(d["means"], d["stds"])
